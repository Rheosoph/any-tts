use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::traits::{ModelInfo, SynthesisRequest, TtsModel};

use super::config::{VibeVoiceConfig, VibeVoicePreprocessorConfig};
use super::diffusion::{DpmSolverMultistepScheduler, VibeVoiceDiffusionHead};
use super::generation::{
    feedback_mode, finish_segment, generation_seed, load_diffusion_noise_fixture,
    progress_interval, prompt_positions, random_normal_tensor, sample_encoder_output, sample_token,
    scale_acoustic_features, semantic_feedback_window, stack_latents, valid_generated_tokens,
    DecoderCacheState, DiffusionNoiseCursor, DiffusionNoiseFixture, GenerationArtifacts,
    GenerationParams, SimpleRng, TokenSequenceState,
};
use super::loader::{
    build_processor, load_components, load_preprocessor_config, resolve_runtime_dtype,
};
use super::processor::{PreparedVibeVoiceInput, VibeVoiceProcessor};
use super::runtime::{SpeechConnector, VibeVoiceLanguageModel};
use super::speech_tokenizer::{
    TokenizerStreamState, VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer,
};

const DEFAULT_CFG_SCALE: f32 = 3.0;

#[derive(Default)]
struct SemanticFeedbackState {
    acoustic: TokenizerStreamState,
    semantic: TokenizerStreamState,
}

pub struct VibeVoiceModel {
    config: VibeVoiceConfig,
    preprocessor_config: VibeVoicePreprocessorConfig,
    device: Device,
    dtype: DType,
    files: ModelFiles,
    processor: VibeVoiceProcessor,
    language_model: Mutex<VibeVoiceLanguageModel>,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    semantic_tokenizer: VibeVoiceSemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    prediction_head: VibeVoiceDiffusionHead,
    noise_scheduler: Mutex<DpmSolverMultistepScheduler>,
    speech_scaling_factor: f32,
    speech_bias_factor: f32,
}

impl TtsModel for VibeVoiceModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let device = config.device.resolve()?;
        let dtype = resolve_runtime_dtype(&device, config.dtype.to_candle());
        let files = config.resolve_files()?;
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = VibeVoiceConfig::from_bytes(config_bytes.as_ref())?;
        let preprocessor_config = load_preprocessor_config(&files)?;
        let processor = build_processor(&files, &preprocessor_config)?;
        let components = load_components(
            &files,
            &model_config,
            &device,
            dtype,
            processor.tokenizer_spec(),
        )?;

        info!(
            "Loaded VibeVoice on {:?}: layers={}, hidden_size={}, acoustic_dim={}, semantic_dim={}",
            device,
            model_config.decoder_config.num_hidden_layers,
            model_config.decoder_config.hidden_size,
            model_config.acoustic_vae_dim,
            model_config.semantic_vae_dim,
        );

        Ok(Self {
            config: model_config.clone(),
            preprocessor_config,
            device,
            dtype,
            files,
            processor,
            language_model: Mutex::new(components.language_model),
            acoustic_tokenizer: components.acoustic_tokenizer,
            semantic_tokenizer: components.semantic_tokenizer,
            acoustic_connector: components.acoustic_connector,
            semantic_connector: components.semantic_connector,
            prediction_head: components.prediction_head,
            noise_scheduler: Mutex::new(DpmSolverMultistepScheduler::new(
                &model_config.diffusion_head_config,
            )),
            speech_scaling_factor: components.speech_scaling_factor,
            speech_bias_factor: components.speech_bias_factor,
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        self.validate_request(request)?;

        let prepared = self.processor.prepare_request(request, &self.device)?;
        let mut rng = SimpleRng::new(generation_seed());
        let prompt_overrides = self.prepare_prompt_overrides(&prepared, &mut rng)?;
        let params = self.generation_params(request, prepared.input_ids.len());
        let diffusion_noise_fixture = self.validated_diffusion_noise_fixture()?;
        let artifacts = self.generate_segments(
            &prepared,
            &prompt_overrides,
            &params,
            diffusion_noise_fixture.as_ref(),
            &mut rng,
        )?;

        self.maybe_log_generated_trace(&artifacts.trace);
        self.audio_samples_from_segments(&artifacts.segments, &artifacts.trace)
    }

    fn sample_rate(&self) -> u32 {
        self.preprocessor_config.audio_processor.sampling_rate
    }

    fn supported_languages(&self) -> Vec<String> {
        vec!["auto".to_string(), "multilingual".to_string()]
    }

    fn supported_voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "VibeVoice-1.5B".to_string(),
            variant: self.config.model_type.clone(),
            parameters: 1_500_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl VibeVoiceModel {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn files(&self) -> &ModelFiles {
        &self.files
    }

    fn validate_request(&self, request: &SynthesisRequest) -> Result<(), TtsError> {
        if request.speed.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice does not support explicit speaking speed".to_string(),
            ));
        }
        if request.voice.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice does not use named voice presets; provide reference audio instead."
                    .to_string(),
            ));
        }
        if request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice does not support pre-extracted voice embeddings yet.".to_string(),
            ));
        }
        Ok(())
    }

    fn generation_params(&self, request: &SynthesisRequest, prompt_len: usize) -> GenerationParams {
        GenerationParams {
            max_new_tokens: request.max_tokens.unwrap_or_else(|| {
                self.config
                    .decoder_config
                    .max_position_embeddings
                    .saturating_sub(prompt_len)
                    .min(2_048)
            }),
            temperature: request.temperature.unwrap_or(0.0) as f32,
            cfg_scale: request.cfg_scale.unwrap_or(DEFAULT_CFG_SCALE as f64) as f32,
        }
    }

    fn validated_diffusion_noise_fixture(&self) -> Result<Option<DiffusionNoiseFixture>, TtsError> {
        let fixture = load_diffusion_noise_fixture()?;
        if let Some(fixture) = &fixture {
            let expected = self.config.diffusion_head_config.latent_size;
            if fixture.latent_size != expected {
                return Err(TtsError::ModelError(format!(
                    "VibeVoice diffusion noise fixture latent size {} does not match the model latent size {}",
                    fixture.latent_size,
                    expected,
                )));
            }
        }
        Ok(fixture)
    }

    fn prepare_prompt_overrides(
        &self,
        prepared: &PreparedVibeVoiceInput,
        rng: &mut SimpleRng,
    ) -> Result<HashMap<usize, Tensor>, TtsError> {
        let mut overrides = HashMap::new();
        let Some(speech_inputs) = &prepared.speech_inputs else {
            return Ok(overrides);
        };

        let speech_tensors = speech_inputs.speech_tensors.unsqueeze(1)?;
        let encoder_output = self.acoustic_tokenizer.encode(&speech_tensors)?;
        let acoustic_latents = sample_encoder_output(
            &encoder_output,
            &self.config.acoustic_tokenizer_config,
            &self.device,
            rng,
        )?;
        let acoustic_features = scale_acoustic_features(
            &acoustic_latents,
            self.speech_bias_factor,
            self.speech_scaling_factor,
            &self.device,
        )?;
        let acoustic_connected = self.acoustic_connector.forward(&acoustic_features)?;
        let prompt_positions = prompt_positions(&prepared.speech_input_mask);
        let available = acoustic_connected.dim(1)?;

        for (row, position) in prompt_positions.into_iter().enumerate().take(available) {
            let embed = acoustic_connected
                .narrow(1, row, 1)?
                .squeeze(0)?
                .squeeze(0)?;
            overrides.insert(position, embed);
        }

        Ok(overrides)
    }

    fn generate_segments(
        &self,
        prepared: &PreparedVibeVoiceInput,
        prompt_overrides: &HashMap<usize, Tensor>,
        params: &GenerationParams,
        diffusion_noise_fixture: Option<&DiffusionNoiseFixture>,
        rng: &mut SimpleRng,
    ) -> Result<GenerationArtifacts, TtsError> {
        let spec = self.processor.tokenizer_spec().clone();
        let valid_token_ids = valid_generated_tokens(
            spec.speech_start_id,
            spec.speech_end_id,
            spec.speech_diffusion_id,
            spec.eos_id,
            spec.bos_id,
        );
        let mut positive_state = self.build_prefill_state(&prepared.input_ids, prompt_overrides)?;
        let mut negative_state = self.single_token_decode_state(spec.speech_start_id)?;
        let mut current_segment = Vec::new();
        let mut finished_segments = Vec::new();
        let mut generated_trace = Vec::new();
        let mut diffusion_noise_cursor = diffusion_noise_fixture.map(DiffusionNoiseFixture::cursor);
        let mut semantic_feedback = SemanticFeedbackState::default();

        for _step in 0..params.max_new_tokens {
            if positive_state.next_position() >= self.config.decoder_config.max_position_embeddings
            {
                break;
            }

            let next_token = sample_token(
                positive_state.logits()?,
                &valid_token_ids,
                params.temperature,
                rng,
            )?;
            generated_trace.push(next_token);
            self.maybe_report_progress(
                generated_trace.len(),
                current_segment.len(),
                finished_segments.len(),
                positive_state.next_position(),
                params.max_new_tokens,
            );

            if next_token == spec.eos_id {
                break;
            }

            if next_token == spec.speech_diffusion_id {
                let speech_latent = self.sample_speech_token(
                    positive_state.last_hidden(),
                    negative_state.last_hidden(),
                    params.cfg_scale,
                    diffusion_noise_cursor.as_mut(),
                    rng,
                )?;
                current_segment.push(speech_latent);
                let diffusion_embed =
                    self.build_diffusion_override(&current_segment, &mut semantic_feedback)?;
                positive_state =
                    self.advance_decode_state(&positive_state, diffusion_embed.clone(), true)?;
                negative_state =
                    self.advance_decode_state(&negative_state, diffusion_embed, false)?;
                continue;
            }

            let token_embedding = self.embed_token(next_token)?;
            positive_state = self.advance_decode_state(&positive_state, token_embedding, true)?;

            if next_token == spec.speech_start_id {
                finish_segment(&mut current_segment, &mut finished_segments);
                semantic_feedback = SemanticFeedbackState::default();
                negative_state = self.single_token_decode_state(spec.speech_start_id)?;
                continue;
            }

            if next_token == spec.speech_end_id {
                finish_segment(&mut current_segment, &mut finished_segments);
                semantic_feedback = SemanticFeedbackState::default();
            }
        }

        finish_segment(&mut current_segment, &mut finished_segments);
        Ok(GenerationArtifacts {
            segments: finished_segments,
            trace: generated_trace,
        })
    }

    fn build_prefill_state(
        &self,
        token_ids: &[u32],
        embedding_overrides: &HashMap<usize, Tensor>,
    ) -> Result<DecoderCacheState, TtsError> {
        let sequence = self.build_sequence_state(token_ids, embedding_overrides)?;
        self.prefill_decode_state(&sequence.input_embeddings()?, true)
    }

    fn build_sequence_state(
        &self,
        token_ids: &[u32],
        embedding_overrides: &HashMap<usize, Tensor>,
    ) -> Result<TokenSequenceState, TtsError> {
        let token_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let language_model = self.language_model.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice language model mutex poisoned".to_string())
        })?;
        let base = language_model.embed(&token_tensor)?;
        drop(language_model);

        TokenSequenceState::from_base_embeddings(token_ids, &base, embedding_overrides)
    }

    fn prefill_decode_state(
        &self,
        input_embeddings: &Tensor,
        project_logits: bool,
    ) -> Result<DecoderCacheState, TtsError> {
        let mut language_model = self.language_model.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice language model mutex poisoned".to_string())
        })?;
        language_model.prefill(input_embeddings, project_logits)
    }

    fn single_token_decode_state(&self, token_id: u32) -> Result<DecoderCacheState, TtsError> {
        self.prefill_decode_state(&self.embed_token(token_id)?.unsqueeze(0)?, false)
    }

    fn advance_decode_state(
        &self,
        state: &DecoderCacheState,
        embedding: Tensor,
        project_logits: bool,
    ) -> Result<DecoderCacheState, TtsError> {
        let mut language_model = self.language_model.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice language model mutex poisoned".to_string())
        })?;
        language_model.load_cache_state(state.layer_caches())?;
        language_model.decode_step(&embedding, state.next_position(), project_logits)
    }

    fn embed_token(&self, token_id: u32) -> Result<Tensor, TtsError> {
        let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
        let language_model = self.language_model.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice language model mutex poisoned".to_string())
        })?;
        language_model
            .embed(&token_tensor)?
            .squeeze(0)
            .map_err(Into::into)
    }

    fn sample_speech_token(
        &self,
        positive_condition: &Tensor,
        negative_condition: &Tensor,
        cfg_scale: f32,
        diffusion_noise_cursor: Option<&mut DiffusionNoiseCursor<'_>>,
        rng: &mut SimpleRng,
    ) -> Result<Tensor, TtsError> {
        let mut scheduler = self.noise_scheduler.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice scheduler mutex poisoned".to_string())
        })?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let condition = Tensor::cat(&[positive_condition, negative_condition], 0)?;
        let mut speech = self.initial_diffusion_speech(diffusion_noise_cursor, rng)?;
        let cfg_scale_tensor = Tensor::new(cfg_scale, &self.device)?;

        for timestep in scheduler.timesteps().to_vec() {
            speech = self.guided_diffusion_step(
                &condition,
                &cfg_scale_tensor,
                &mut scheduler,
                &speech,
                timestep,
            )?;
        }

        speech.narrow(0, 0, 1)?.squeeze(0).map_err(Into::into)
    }

    fn build_diffusion_override(
        &self,
        segment_latents: &[Tensor],
        state: &mut SemanticFeedbackState,
    ) -> Result<Tensor, TtsError> {
        let feedback_mode = self.effective_feedback_mode();
        if feedback_mode == "token" {
            return self.diffusion_token_embedding();
        }

        let latest_latent = segment_latents.last().ok_or_else(|| {
            TtsError::ModelError(
                "VibeVoice cannot build feedback embeddings without at least one speech latent"
                    .to_string(),
            )
        })?;
        let acoustic_embed = self.acoustic_feedback_embedding(latest_latent)?;
        if feedback_mode == "acoustic" {
            return Ok(acoustic_embed);
        }

        let acoustic_config = self.acoustic_tokenizer.config();
        let semantic_config = self.semantic_tokenizer.config();
        let acoustic_hop = acoustic_config.decoder_ratios().iter().product::<usize>();
        let semantic_hop = semantic_config.encoder_ratios.iter().product::<usize>();
        if acoustic_config.causal
            && semantic_config.causal
            && semantic_hop != 0
            && acoustic_hop % semantic_hop == 0
            && semantic_feedback_window() == usize::MAX
        {
            let latent = latest_latent.unsqueeze(0)?.unsqueeze(1)?;
            let audio = self.acoustic_tokenizer.decode_stream(
                &self.unscale_generated_latents(&latent)?,
                &mut state.acoustic,
            )?;
            let semantic = self
                .semantic_tokenizer
                .encode_stream(&audio, &mut state.semantic)?
                .mean;
            return self.combine_semantic_feedback(&semantic, &acoustic_embed);
        }

        let segment = self.semantic_feedback_segment(segment_latents)?;
        self.semantic_feedback_embedding(&segment, &acoustic_embed)
    }

    fn audio_samples_from_segments(
        &self,
        segments: &[Vec<Tensor>],
        generated_trace: &[u32],
    ) -> Result<AudioSamples, TtsError> {
        let samples = self.decode_segments(segments)?;
        if samples.is_empty() {
            return Err(TtsError::ModelError(format!(
                "VibeVoice did not produce any speech tokens for this prompt. Generated tokens: {:?}",
                generated_trace,
            )));
        }

        Ok(AudioSamples::new(
            samples,
            self.preprocessor_config.audio_processor.sampling_rate,
        ))
    }

    fn maybe_log_generated_trace(&self, generated_trace: &[u32]) {
        if std::env::var_os("VIBEVOICE_DEBUG_TRACE").is_some() {
            eprintln!("VibeVoice generated tokens: {:?}", generated_trace);
        }
    }

    fn effective_feedback_mode(&self) -> &'static str {
        match feedback_mode() {
            Some("token") => "token",
            Some("acoustic") => "acoustic",
            Some("semantic") => "semantic",
            _ => "semantic",
        }
    }

    fn maybe_report_progress(
        &self,
        generated_tokens: usize,
        current_segment_latents: usize,
        finished_segments: usize,
        context_position: usize,
        max_new_tokens: usize,
    ) {
        let interval = progress_interval();
        if interval == 0 || generated_tokens == 0 || generated_tokens % interval != 0 {
            return;
        }

        eprintln!(
            "VibeVoice progress: generated {generated_tokens}/{max_new_tokens} token(s), current segment {current_segment_latents} latent(s), finished {finished_segments} segment(s), context position {context_position}/{}",
            self.config.decoder_config.max_position_embeddings,
        );
    }

    fn initial_diffusion_speech(
        &self,
        diffusion_noise_cursor: Option<&mut DiffusionNoiseCursor<'_>>,
        rng: &mut SimpleRng,
    ) -> Result<Tensor, TtsError> {
        let latent_size = self.config.diffusion_head_config.latent_size;
        let half = if let Some(cursor) = diffusion_noise_cursor {
            cursor.next_tensor(&self.device, self.dtype)?
        } else {
            random_normal_tensor((1, latent_size), self.dtype, &self.device, rng)?
        };
        Tensor::cat(&[&half, &half], 0).map_err(Into::into)
    }

    fn guided_diffusion_step(
        &self,
        condition: &Tensor,
        cfg_scale_tensor: &Tensor,
        scheduler: &mut DpmSolverMultistepScheduler,
        speech: &Tensor,
        timestep: usize,
    ) -> Result<Tensor, TtsError> {
        let half = speech.narrow(0, 0, 1)?;
        let combined = Tensor::cat(&[&half, &half], 0)?;
        let timestep_tensor =
            Tensor::from_vec(vec![timestep as f32, timestep as f32], (2,), &self.device)?
                .to_dtype(self.dtype)?;
        let eps = self
            .prediction_head
            .forward(&combined, &timestep_tensor, condition)?;
        let cond_eps = eps.narrow(0, 0, 1)?;
        let uncond_eps = eps.narrow(0, 1, 1)?;
        let guided = uncond_eps.broadcast_add(
            &cond_eps
                .broadcast_sub(&uncond_eps)?
                .broadcast_mul(cfg_scale_tensor)?,
        )?;
        let expanded = Tensor::cat(&[&guided, &guided], 0)?;
        scheduler.step(&expanded, speech).map_err(Into::into)
    }

    fn acoustic_feedback_embedding(&self, speech_latent: &Tensor) -> Result<Tensor, TtsError> {
        self.acoustic_connector
            .forward(&speech_latent.unsqueeze(0)?.unsqueeze(1)?)?
            .squeeze(0)?
            .squeeze(0)
            .map_err(Into::into)
    }

    fn semantic_feedback_segment(&self, segment_latents: &[Tensor]) -> Result<Tensor, TtsError> {
        let window = semantic_feedback_window().min(segment_latents.len());
        stack_latents(&segment_latents[segment_latents.len() - window..])
    }

    fn semantic_feedback_embedding(
        &self,
        segment: &Tensor,
        acoustic_embed: &Tensor,
    ) -> Result<Tensor, TtsError> {
        let decoded_audio = self
            .acoustic_tokenizer
            .decode(&self.unscale_generated_latents(segment)?.transpose(1, 2)?)?;
        let semantic = self.semantic_tokenizer.encode(&decoded_audio)?.mean;
        self.combine_semantic_feedback(&semantic, acoustic_embed)
    }

    fn combine_semantic_feedback(
        &self,
        semantic: &Tensor,
        acoustic_embed: &Tensor,
    ) -> Result<Tensor, TtsError> {
        let semantic_last = semantic.narrow(1, semantic.dim(1)? - 1, 1)?.squeeze(1)?;
        let semantic_embed = self
            .semantic_connector
            .forward(&semantic_last.unsqueeze(1)?)?
            .squeeze(0)?
            .squeeze(0)?;
        acoustic_embed
            .broadcast_add(&semantic_embed)
            .map_err(Into::into)
    }

    fn decode_segments(&self, segments: &[Vec<Tensor>]) -> Result<Vec<f32>, TtsError> {
        let mut all_samples = Vec::new();
        for segment in segments.iter().filter(|segment| !segment.is_empty()) {
            let latents = stack_latents(segment)?;
            let audio = self
                .acoustic_tokenizer
                .decode(&self.unscale_generated_latents(&latents)?.transpose(1, 2)?)?;
            let mut samples = audio
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            all_samples.append(&mut samples);
        }
        Ok(all_samples)
    }

    fn unscale_generated_latents(&self, latents: &Tensor) -> Result<Tensor, TtsError> {
        latents
            .broadcast_div(&Tensor::new(self.speech_scaling_factor, &self.device)?)?
            .broadcast_sub(&Tensor::new(self.speech_bias_factor, &self.device)?)
            .map_err(Into::into)
    }

    fn diffusion_token_embedding(&self) -> Result<Tensor, TtsError> {
        let token_id = self.processor.tokenizer_spec().speech_diffusion_id;
        self.embed_token(token_id)?.squeeze(0).map_err(Into::into)
    }
}
