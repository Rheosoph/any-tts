use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module};
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelAssetDir, ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::models::vibevoice::config::VibeVoicePreprocessorConfig;
use crate::models::vibevoice::diffusion::{DpmSolverMultistepScheduler, VibeVoiceDiffusionHead};
use crate::models::vibevoice::generation::{
    generation_seed, random_normal_tensor, stack_latents, SimpleRng,
};
use crate::models::vibevoice::processor::VibeVoiceTokenizerSpec;
use crate::models::vibevoice::runtime::SpeechConnector;
use crate::models::vibevoice::speech_tokenizer::{
    TokenizerStreamState, VibeVoiceAcousticTokenizer,
};
use crate::tokenizer::TextTokenizer;
use crate::traits::{ModelInfo, SynthesisRequest, TtsModel};

use super::config::VibeVoiceRealtimeConfig;
use super::preset::VoicePreset;
use super::processor::VibeVoiceRealtimeProcessor;
use super::runtime::{BinaryClassifier, RealtimeDecoderState, RealtimeLanguageModel};

const DEFAULT_CFG_SCALE: f32 = 1.5;
const TTS_TEXT_WINDOW_SIZE: usize = 5;
const TTS_SPEECH_WINDOW_SIZE: usize = 6;

pub struct VibeVoiceRealtimeModel {
    config: VibeVoiceRealtimeConfig,
    preprocessor_config: VibeVoicePreprocessorConfig,
    device: Device,
    dtype: DType,
    files: ModelFiles,
    processor: VibeVoiceRealtimeProcessor,
    language_model: Mutex<RealtimeLanguageModel>,
    tts_language_model: Mutex<RealtimeLanguageModel>,
    tts_input_types: Embedding,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    acoustic_connector: SpeechConnector,
    prediction_head: VibeVoiceDiffusionHead,
    noise_scheduler: Mutex<DpmSolverMultistepScheduler>,
    tts_eos_classifier: BinaryClassifier,
    speech_scaling_factor: f32,
    speech_bias_factor: f32,
    voice_names: Vec<String>,
    voice_cache: Mutex<HashMap<String, VoicePreset>>,
}

impl TtsModel for VibeVoiceRealtimeModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let device = config.device.resolve()?;
        let dtype = resolve_runtime_dtype(&device, config.dtype.to_candle());
        let files = config.resolve_files()?;
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = VibeVoiceRealtimeConfig::from_bytes(config_bytes.as_ref())?;
        let preprocessor_config = if let Some(asset) = &files.preprocessor_config {
            VibeVoicePreprocessorConfig::from_bytes(asset.read_bytes()?.as_ref())?
        } else {
            VibeVoicePreprocessorConfig::default()
        };
        let tokenizer = TextTokenizer::from_asset(
            files
                .tokenizer
                .as_ref()
                .expect("validated by resolve_files"),
        )?;
        let tokenizer_spec = VibeVoiceTokenizerSpec::from_tokenizer(&tokenizer)?;
        let processor =
            VibeVoiceRealtimeProcessor::new(tokenizer, tokenizer_spec, preprocessor_config.clone());
        let vb = ModelFiles::load_safetensors_vb(&files.weights, dtype, &device)?;
        let model_vb = vb.pp("model");

        let lm_layer_count = model_config
            .decoder_config
            .num_hidden_layers
            .saturating_sub(model_config.tts_backbone_num_hidden_layers);
        let language_model = RealtimeLanguageModel::load(
            &model_config,
            model_vb.pp("language_model"),
            &device,
            dtype,
            lm_layer_count,
            false,
        )?;
        let tts_language_model = RealtimeLanguageModel::load(
            &model_config,
            model_vb.pp("tts_language_model"),
            &device,
            dtype,
            model_config.tts_backbone_num_hidden_layers,
            true,
        )?;
        let tts_input_types = candle_nn::embedding(
            2,
            model_config.decoder_config.hidden_size,
            model_vb.pp("tts_input_types"),
        )?;
        let acoustic_tokenizer = VibeVoiceAcousticTokenizer::load_decoder_only(
            &model_config.acoustic_tokenizer_config,
            model_vb.pp("acoustic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::load(
            model_config.acoustic_vae_dim,
            model_config.decoder_config.hidden_size,
            model_vb.pp("acoustic_connector"),
        )?;
        let prediction_head = VibeVoiceDiffusionHead::load(
            &model_config.diffusion_head_config,
            model_vb.pp("prediction_head"),
        )?;
        let tts_eos_classifier = BinaryClassifier::load(
            model_config.decoder_config.hidden_size,
            vb.pp("tts_eos_classifier"),
        )?;
        let voice_names = discover_voice_names(files.voices_dir.as_ref())?;

        info!(
            "Loaded VibeVoice Realtime on {:?}: base_layers={}, tts_layers={}, hidden_size={}, voices={}",
            device,
            lm_layer_count,
            model_config.tts_backbone_num_hidden_layers,
            model_config.decoder_config.hidden_size,
            voice_names.len(),
        );

        Ok(Self {
            config: model_config.clone(),
            preprocessor_config,
            device,
            dtype,
            files,
            processor,
            language_model: Mutex::new(language_model),
            tts_language_model: Mutex::new(tts_language_model),
            tts_input_types,
            acoustic_tokenizer,
            acoustic_connector,
            prediction_head,
            noise_scheduler: Mutex::new(DpmSolverMultistepScheduler::new(
                &model_config.diffusion_head_config,
            )),
            tts_eos_classifier,
            speech_scaling_factor: load_scalar(&model_vb, "speech_scaling_factor")?,
            speech_bias_factor: load_scalar(&model_vb, "speech_bias_factor")?,
            voice_names,
            voice_cache: Mutex::new(HashMap::new()),
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        let mut samples = Vec::new();
        self.synthesize_stream(request, &mut |chunk| {
            samples.extend(chunk.samples);
            Ok(())
        })?;
        Ok(AudioSamples::new(samples, self.sample_rate()))
    }

    fn supports_streaming(&self) -> bool {
        self.acoustic_tokenizer.config().causal
    }

    fn synthesize_stream(
        &self,
        request: &SynthesisRequest,
        on_chunk: &mut dyn FnMut(AudioSamples) -> Result<(), TtsError>,
    ) -> Result<(), TtsError> {
        self.validate_request(request)?;
        let voice_name = resolve_voice_name(request.voice.as_deref(), &self.voice_names)?;
        let preset = self.load_voice_preset(&voice_name)?;
        let text_ids = self.processor.prepare_text(request)?;
        let mut rng = SimpleRng::new(generation_seed());
        let max_steps = request.max_tokens.unwrap_or_else(|| {
            self.config
                .decoder_config
                .max_position_embeddings
                .saturating_sub(preset.tts_lm.state.next_position())
        });
        let cfg_scale = request.cfg_scale.unwrap_or(DEFAULT_CFG_SCALE as f64) as f32;
        let mut decoder_state = TokenizerStreamState::default();
        let mut buffered_latents = Vec::new();
        let streaming = self.supports_streaming();
        let speech_count = self.generate_latents(
            &preset,
            &text_ids,
            cfg_scale,
            max_steps,
            &mut rng,
            &mut |latent| {
                if streaming {
                    let segment = latent.unsqueeze(0)?.unsqueeze(1)?;
                    let audio = self
                        .acoustic_tokenizer
                        .decode_stream(&self.unscale_latents(&segment)?, &mut decoder_state)?;
                    let samples = audio
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    on_chunk(AudioSamples::new(samples, self.sample_rate()))
                } else {
                    buffered_latents.push(latent.clone());
                    Ok(())
                }
            },
        )?;
        if speech_count == 0 {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime did not produce any speech tokens".to_string(),
            ));
        }

        if !streaming {
            let segment = stack_latents(&buffered_latents)?;
            let audio = self
                .acoustic_tokenizer
                .decode(&self.unscale_latents(&segment)?.transpose(1, 2)?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            on_chunk(AudioSamples::new(audio, self.sample_rate()))?;
        }
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.preprocessor_config.audio_processor.sampling_rate
    }

    fn supported_languages(&self) -> Vec<String> {
        vec!["auto".to_string(), "multilingual".to_string()]
    }

    fn supported_voices(&self) -> Vec<String> {
        self.voice_names.clone()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "VibeVoice-Realtime-0.5B".to_string(),
            variant: self.config.model_type.clone(),
            parameters: 500_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl VibeVoiceRealtimeModel {
    pub fn files(&self) -> &ModelFiles {
        &self.files
    }

    fn validate_request(&self, request: &SynthesisRequest) -> Result<(), TtsError> {
        if request.speed.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime does not support explicit speaking speed".to_string(),
            ));
        }
        if request.reference_audio.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime uses cached prompt presets, not reference audio".to_string(),
            ));
        }
        if request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime does not support pre-extracted voice embeddings".to_string(),
            ));
        }
        if self.voice_names.is_empty() {
            return Err(TtsError::FileMissing(
                "voices/ — VibeVoice Realtime cached prompt presets (*.pt)".to_string(),
            ));
        }
        Ok(())
    }

    fn load_voice_preset(&self, voice_name: &str) -> Result<VoicePreset, TtsError> {
        if let Some(cached) = self
            .voice_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("VibeVoice Realtime voice cache poisoned".to_string())
            })?
            .get(voice_name)
            .cloned()
        {
            return Ok(cached);
        }

        let file_name = format!("{voice_name}.pt");
        let asset = self
            .files
            .voices_dir
            .as_ref()
            .ok_or_else(|| {
                TtsError::FileMissing(
                    "voices/ — VibeVoice Realtime cached prompt presets (*.pt)".to_string(),
                )
            })?
            .load_file(&file_name)?;
        let preset = VoicePreset::load(&asset, &self.device, self.dtype)?;

        self.voice_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("VibeVoice Realtime voice cache poisoned".to_string())
            })?
            .insert(voice_name.to_string(), preset.clone());
        Ok(preset)
    }

    fn generate_latents(
        &self,
        preset: &VoicePreset,
        text_ids: &[u32],
        cfg_scale: f32,
        max_steps: usize,
        rng: &mut SimpleRng,
        on_latent: &mut dyn FnMut(&Tensor) -> Result<(), TtsError>,
    ) -> Result<usize, TtsError> {
        let mut text_index = 0usize;
        let mut lm_state = clone_branch_state(&preset.lm.state);
        let mut tts_state = clone_branch_state(&preset.tts_lm.state);
        let mut neg_tts_state = clone_branch_state(&preset.neg_tts_lm.state);
        let mut speech_count = 0;
        let mut budget = GenerationBudget::new(
            max_steps,
            self.config.decoder_config.max_position_embeddings,
        );

        while budget.can_advance(&[tts_state.next_position()]) {
            let remaining_text = text_ids.len().saturating_sub(text_index);
            let text_window = remaining_text.min(TTS_TEXT_WINDOW_SIZE);
            if text_window > 0 {
                for token_id in &text_ids[text_index..text_index + text_window] {
                    if !budget.can_advance(&[lm_state.next_position(), tts_state.next_position()]) {
                        return Ok(speech_count);
                    }
                    lm_state = self.advance_lm_state(&lm_state, *token_id)?;
                    tts_state = self.advance_tts_state(
                        &tts_state,
                        *token_id,
                        &lm_state.last_hidden().clone(),
                        true,
                    )?;
                }
                text_index += text_window;
            }

            for _ in 0..TTS_SPEECH_WINDOW_SIZE {
                if !budget.can_advance(&[tts_state.next_position(), neg_tts_state.next_position()])
                {
                    return Ok(speech_count);
                }
                let speech_latent = self.sample_speech_token(
                    tts_state.last_hidden(),
                    neg_tts_state.last_hidden(),
                    cfg_scale,
                    rng,
                )?;
                on_latent(&speech_latent)?;
                speech_count += 1;
                let acoustic_embed = self.acoustic_feedback_embedding(&speech_latent)?;
                tts_state = self.advance_tts_state(&tts_state, 1, &acoustic_embed, false)?;
                neg_tts_state =
                    self.advance_tts_state(&neg_tts_state, 1, &acoustic_embed, false)?;
                budget.consume();

                let eos_logit = self
                    .tts_eos_classifier
                    .forward(tts_state.last_hidden())?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                if eos_logit > 0.0 {
                    return Ok(speech_count);
                }
            }
        }

        Ok(speech_count)
    }

    fn advance_lm_state(
        &self,
        state: &RealtimeDecoderState,
        token_id: u32,
    ) -> Result<RealtimeDecoderState, TtsError> {
        let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
        let mut language_model = self.language_model.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice Realtime language model mutex poisoned".to_string())
        })?;
        language_model.load_cache_state(state.layer_caches())?;
        let embedding = language_model.embed(&token_tensor)?;
        language_model.decode_step(&embedding, state.next_position())
    }

    fn advance_tts_state(
        &self,
        state: &RealtimeDecoderState,
        token_id: u32,
        hidden_override: &Tensor,
        is_text: bool,
    ) -> Result<RealtimeDecoderState, TtsError> {
        let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
        let mut tts_language_model = self.tts_language_model.lock().map_err(|_| {
            TtsError::RuntimeError(
                "VibeVoice Realtime TTS language model mutex poisoned".to_string(),
            )
        })?;
        tts_language_model.load_cache_state(state.layer_caches())?;
        let base_embedding = tts_language_model.embed(&token_tensor)?;
        let type_token = if is_text { 1u32 } else { 0u32 };
        let type_embedding = self
            .tts_input_types
            .forward(&Tensor::new(&[type_token], &self.device)?.unsqueeze(0)?)?;
        let hidden = normalize_override_embedding(hidden_override)?;
        let input_embedding = hidden.broadcast_add(&type_embedding)?;
        let _ = base_embedding;
        tts_language_model.decode_step(&input_embedding, state.next_position())
    }

    fn sample_speech_token(
        &self,
        positive_condition: &Tensor,
        negative_condition: &Tensor,
        cfg_scale: f32,
        rng: &mut SimpleRng,
    ) -> Result<Tensor, TtsError> {
        let mut scheduler = self.noise_scheduler.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice Realtime scheduler mutex poisoned".to_string())
        })?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let condition = Tensor::cat(&[positive_condition, negative_condition], 0)?;
        let half = random_normal_tensor(
            (1, self.config.acoustic_vae_dim),
            self.dtype,
            &self.device,
            rng,
        )?;
        let mut speech = Tensor::cat(&[&half, &half], 0)?;
        let cfg_scale_tensor = Tensor::new(cfg_scale, &self.device)?;

        for timestep in scheduler.timesteps().to_vec() {
            let timestep_tensor =
                Tensor::from_vec(vec![timestep as f32, timestep as f32], (2,), &self.device)?
                    .to_dtype(self.dtype)?;
            let eps = self
                .prediction_head
                .forward(&speech, &timestep_tensor, &condition)?;
            let cond_eps = eps.narrow(0, 0, 1)?;
            let uncond_eps = eps.narrow(0, 1, 1)?;
            let guided = uncond_eps.broadcast_add(
                &cond_eps
                    .broadcast_sub(&uncond_eps)?
                    .broadcast_mul(&cfg_scale_tensor)?,
            )?;
            let expanded = Tensor::cat(&[&guided, &guided], 0)?;
            speech = scheduler.step(&expanded, &speech)?;
        }

        speech.narrow(0, 0, 1)?.squeeze(0).map_err(Into::into)
    }

    fn acoustic_feedback_embedding(&self, speech_latent: &Tensor) -> Result<Tensor, TtsError> {
        self.acoustic_connector
            .forward(&speech_latent.unsqueeze(0)?.unsqueeze(1)?)?
            .squeeze(0)?
            .squeeze(0)
            .map_err(Into::into)
    }

    fn unscale_latents(&self, latents: &Tensor) -> Result<Tensor, TtsError> {
        latents
            .broadcast_div(&Tensor::new(self.speech_scaling_factor, &self.device)?)?
            .broadcast_sub(&Tensor::new(self.speech_bias_factor, &self.device)?)
            .map_err(Into::into)
    }
}

fn resolve_runtime_dtype(device: &Device, requested: DType) -> DType {
    if matches!(device, Device::Cpu) && requested == DType::BF16 {
        return DType::F32;
    }
    if matches!(device, Device::Metal(_)) {
        return DType::F32;
    }
    requested
}

fn discover_voice_names(voices_dir: Option<&ModelAssetDir>) -> Result<Vec<String>, TtsError> {
    let Some(voices_dir) = voices_dir else {
        return Ok(Vec::new());
    };
    let mut names = voices_dir
        .file_names()?
        .into_iter()
        .filter_map(|name| name.strip_suffix(".pt").map(str::to_string))
        .collect::<Vec<_>>();
    names.sort();
    Ok(names)
}

fn resolve_voice_name(requested: Option<&str>, available: &[String]) -> Result<String, TtsError> {
    if available.is_empty() {
        return Err(TtsError::FileMissing(
            "voices/ — VibeVoice Realtime cached prompt presets (*.pt)".to_string(),
        ));
    }

    if let Some(requested) = requested {
        if let Some(exact) = available.iter().find(|voice| voice == &requested) {
            return Ok(exact.clone());
        }

        let lowered = requested.to_ascii_lowercase();
        if let Some(case_insensitive) = available
            .iter()
            .find(|voice| voice.to_ascii_lowercase() == lowered)
        {
            return Ok(case_insensitive.clone());
        }

        if let Some(partial) = available
            .iter()
            .find(|voice| voice.to_ascii_lowercase().contains(&lowered))
        {
            return Ok(partial.clone());
        }

        return Err(TtsError::UnknownVoice(requested.to_string()));
    }

    if let Some(default_voice) = available
        .iter()
        .find(|voice| voice.as_str() == "en-Carter_man")
    {
        return Ok(default_voice.clone());
    }
    Ok(available[0].clone())
}

fn normalize_override_embedding(hidden: &Tensor) -> Result<Tensor, TtsError> {
    match hidden.rank() {
        1 => hidden.unsqueeze(0)?.unsqueeze(0).map_err(Into::into),
        2 => hidden.unsqueeze(1).map_err(Into::into),
        3 => Ok(hidden.clone()),
        _ => Err(TtsError::ModelError(
            "Unexpected VibeVoice Realtime hidden override rank".to_string(),
        )),
    }
}

fn clone_branch_state(state: &RealtimeDecoderState) -> RealtimeDecoderState {
    state.clone()
}

struct GenerationBudget {
    remaining: usize,
    max_positions: usize,
}

impl GenerationBudget {
    fn new(max_audio_tokens: usize, max_positions: usize) -> Self {
        Self {
            remaining: max_audio_tokens,
            max_positions,
        }
    }

    fn can_advance(&self, positions: &[usize]) -> bool {
        self.remaining > 0
            && positions
                .iter()
                .all(|position| *position < self.max_positions)
    }

    fn consume(&mut self) {
        self.remaining = self.remaining.saturating_sub(1);
    }
}

fn load_scalar(model_vb: &candle_nn::VarBuilder, name: &str) -> Result<f32, TtsError> {
    let tensor = model_vb.get(1, name).or_else(|_| model_vb.get((), name))?;
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    values.first().copied().ok_or_else(|| {
        TtsError::ModelError("Expected scalar tensor in VibeVoice Realtime weights".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::kv_cache::KvCacheState;
    use crate::models::vibevoice::config::{
        VibeVoiceDecoderConfig, VibeVoiceDiffusionHeadConfig, VibeVoiceTokenizerConfig,
    };
    use crate::models::vibevoice_realtime::preset::VoicePresetBranch;

    fn tiny_model() -> VibeVoiceRealtimeModel {
        tiny_model_with_context(12)
    }

    fn tiny_model_with_context(max_position_embeddings: usize) -> VibeVoiceRealtimeModel {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = candle_nn::VarBuilder::zeros(dtype, &device);
        let decoder_config = VibeVoiceDecoderConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 12,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            max_position_embeddings,
            ..Default::default()
        };
        let acoustic_tokenizer_config = VibeVoiceTokenizerConfig {
            vae_dim: 2,
            encoder_n_filters: 2,
            decoder_n_filters: 2,
            encoder_ratios: vec![2],
            encoder_depths: "1-1".to_string(),
            causal: true,
            ..Default::default()
        };
        let diffusion_head_config = VibeVoiceDiffusionHeadConfig {
            hidden_size: 8,
            head_layers: 1,
            latent_size: 2,
            speech_vae_dim: Some(2),
            ddpm_num_steps: 20,
            ddpm_num_inference_steps: 2,
            ..Default::default()
        };
        let config = VibeVoiceRealtimeConfig {
            model_type: "test".to_string(),
            acoustic_vae_dim: 2,
            decoder_config,
            acoustic_tokenizer_config,
            diffusion_head_config,
            tts_backbone_num_hidden_layers: 1,
        };
        let tokenizer = TextTokenizer::from_bytes(
            br#"{
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}
        }"#,
        )
        .unwrap();
        let spec = VibeVoiceTokenizerSpec {
            speech_start_id: 2,
            speech_end_id: 3,
            speech_diffusion_id: 4,
            eos_id: 5,
            pad_id: 6,
            bos_id: None,
        };
        let preset_branch = VoicePresetBranch {
            state: RealtimeDecoderState::new(
                7,
                Tensor::zeros((1, 8), dtype, &device).unwrap(),
                vec![Some(
                    KvCacheState::from_tensors(
                        Tensor::zeros((1, 2, 7, 4), dtype, &device).unwrap(),
                        Tensor::zeros((1, 2, 7, 4), dtype, &device).unwrap(),
                    )
                    .unwrap(),
                )],
            ),
        };
        let preset = VoicePreset {
            lm: preset_branch.clone(),
            tts_lm: preset_branch.clone(),
            _neg_lm: preset_branch.clone(),
            neg_tts_lm: preset_branch,
        };
        VibeVoiceRealtimeModel {
            language_model: Mutex::new(
                RealtimeLanguageModel::load(&config, vb.clone(), &device, dtype, 1, false).unwrap(),
            ),
            tts_language_model: Mutex::new(
                RealtimeLanguageModel::load(&config, vb.clone(), &device, dtype, 1, true).unwrap(),
            ),
            tts_input_types: candle_nn::embedding(2, 8, vb.clone()).unwrap(),
            acoustic_tokenizer: VibeVoiceAcousticTokenizer::load_decoder_only(
                &config.acoustic_tokenizer_config,
                vb.clone(),
            )
            .unwrap(),
            acoustic_connector: SpeechConnector::load(2, 8, vb.clone()).unwrap(),
            prediction_head: VibeVoiceDiffusionHead::load(
                &config.diffusion_head_config,
                vb.clone(),
            )
            .unwrap(),
            noise_scheduler: Mutex::new(DpmSolverMultistepScheduler::new(
                &config.diffusion_head_config,
            )),
            tts_eos_classifier: BinaryClassifier::load(8, vb).unwrap(),
            processor: VibeVoiceRealtimeProcessor::new(
                tokenizer,
                spec,
                VibeVoicePreprocessorConfig::default(),
            ),
            speech_scaling_factor: 1.0,
            speech_bias_factor: 0.0,
            voice_names: vec!["test".to_string()],
            voice_cache: Mutex::new(HashMap::from([("test".to_string(), preset)])),
            files: ModelFiles::default(),
            preprocessor_config: VibeVoicePreprocessorConfig::default(),
            config,
            device,
            dtype,
        }
    }

    #[test]
    fn realtime_synthesis_streams_with_budget_smaller_than_prompt_and_matches_buffering() {
        let model = tiny_model();
        assert!(model.supports_streaming());
        let request = SynthesisRequest::new("hello").with_max_tokens(5);
        let mut chunks = Vec::new();
        model
            .synthesize_stream(&request, &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .unwrap();
        // One injected text position and four audio positions fit after the
        // seven-position preset, including the final valid RoPE position.
        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|chunk| chunk.samples.len() == 2));
        let streamed: Vec<_> = chunks.into_iter().flat_map(|chunk| chunk.samples).collect();
        assert_eq!(streamed, model.synthesize(&request).unwrap().samples);
        assert_eq!(
            streamed,
            model
                .synthesize(&SynthesisRequest::new("hello"))
                .unwrap()
                .samples
        );
    }

    #[test]
    fn realtime_one_audio_token_budget_does_not_count_injected_text() {
        let model = tiny_model_with_context(64);
        let text = std::iter::repeat_n("hello", 20)
            .collect::<Vec<_>>()
            .join(" ");
        let request = SynthesisRequest::new(text).with_max_tokens(1);
        assert_eq!(model.processor.prepare_text(&request).unwrap().len(), 20);
        let mut chunks = 0;
        model
            .synthesize_stream(&request, &mut |chunk| {
                chunks += 1;
                assert_eq!(chunk.samples.len(), 2);
                Ok(())
            })
            .unwrap();
        assert_eq!(chunks, 1);
    }

    #[test]
    fn realtime_text_boundary_cannot_trigger_an_extra_speech_step() {
        let model = tiny_model();
        let preset = model.load_voice_preset("test").unwrap();
        let mut callbacks = 0;
        let result = model
            .generate_latents(
                &preset,
                &[1, 2, 3, 4, 5],
                1.5,
                5,
                &mut SimpleRng::new(42),
                &mut |_| {
                    callbacks += 1;
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(result, 0);
        assert_eq!(callbacks, 0);
    }

    #[test]
    fn realtime_callback_error_cancels_generation_and_next_request_is_independent() {
        let model = tiny_model();
        let request = SynthesisRequest::new("hello").with_max_tokens(5);
        let mut callbacks = 0;
        let error = model
            .synthesize_stream(&request, &mut |_| {
                callbacks += 1;
                Err(TtsError::RuntimeError("cancel test".to_string()))
            })
            .unwrap_err();
        assert!(error.to_string().contains("cancel test"));
        assert_eq!(callbacks, 1);
        assert_eq!(model.synthesize(&request).unwrap().samples.len(), 8);
    }

    #[test]
    fn realtime_budget_counts_audio_tokens_and_checks_each_branch_boundary() {
        let mut budget = GenerationBudget::new(2, 1024);
        assert!(budget.can_advance(&[500, 900]));
        budget.consume();
        assert!(budget.can_advance(&[501, 901]));
        assert!(!budget.can_advance(&[501, 1024]));
        budget.consume();
        assert!(!budget.can_advance(&[502, 902]));
        assert!(!GenerationBudget::new(0, 1024).can_advance(&[0]));
    }

    #[test]
    fn realtime_default_capacity_is_not_charged_for_the_prompt_twice() {
        let prompt_len = 7;
        let max_positions = 12;
        let mut budget = GenerationBudget::new(max_positions - prompt_len, max_positions);
        for position in prompt_len..max_positions {
            assert!(budget.can_advance(&[position]));
            budget.consume();
        }
        assert!(!budget.can_advance(&[max_positions]));
    }
}
