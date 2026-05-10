//! Qwen3-TTS full inference model.
//!
//! Pipeline:
//!   Text + Speaker + Instruct → Talker LM (codec group 0)
//!                                   ↓
//!                        Code Predictor (codec groups 1..N)
//!                                   ↓
//!                    Speech Tokenizer Decoder (codes → 24kHz waveform)

use candle_core::{DType, Device, Tensor};
use std::sync::Mutex;
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::tokenizer::TextTokenizer;
use crate::traits::{ModelInfo, SynthesisRequest, TtsModel};

use super::code_predictor::CodePredictor;
use super::config::Qwen3TtsConfig;
use super::speech_tokenizer::{SpeechTokenizerConfig, SpeechTokenizerDecoder};
use super::talker::{TalkerGenerationConfig, TalkerLm};

/// Qwen3-TTS-12Hz TTS model.
///
/// 1.7B parameter multi-codebook language model for 10 languages.
/// Supports named speakers and instruction-based style control.
pub struct Qwen3TtsModel {
    config: Qwen3TtsConfig,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    files: ModelFiles,
    /// Wrapped in Mutex for interior mutability (KV-cache updates during generation).
    talker: Mutex<TalkerLm>,
    /// Wrapped in Mutex for interior mutability (KV-cache updates during prediction).
    code_predictor: Option<Mutex<CodePredictor>>,
    speech_tokenizer: Option<SpeechTokenizerDecoder>,
}

impl TtsModel for Qwen3TtsModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let device = config.device.resolve()?;
        let mut dtype = config.dtype.to_candle();

        // CPU does not support BF16 matmul; fall back to F32 automatically.
        if matches!(device, Device::Cpu) && dtype == DType::BF16 {
            info!("BF16 is not supported on CPU; falling back to F32");
            dtype = DType::F32;
        }
        info!("Loading Qwen3-TTS model on {:?}", device);

        // ── Resolve files (explicit → directory → HF download) ────────
        let files = config.resolve_files()?;

        // ── Parse model config ────────────────────────────────────────
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = Qwen3TtsConfig::from_bytes(config_bytes.as_ref())?;

        info!(
            "Qwen3-TTS config loaded: type={}, talker layers={}, hidden_size={}, code_groups={}",
            model_config.tts_model_type,
            model_config.talker_config.num_hidden_layers,
            model_config.talker_config.hidden_size,
            model_config.talker_config.num_code_groups,
        );

        // ── Load main model weights ───────────────────────────────────
        let vb = ModelFiles::load_safetensors_vb(&files.weights, dtype, &device)?;

        // ── Build Talker LM ──────────────────────────────────────────
        let talker = TalkerLm::load(&model_config.talker_config, vb.pp("talker"), &device, dtype)
            .map_err(|e| {
            TtsError::WeightLoadError(format!("Failed to build Talker LM: {}", e))
        })?;
        info!(
            "Talker LM loaded ({} layers)",
            model_config.talker_config.num_hidden_layers
        );

        // ── Build Code Predictor ──────────────────────────────────────
        let code_predictor =
            if let Some(ref cp_config) = model_config.talker_config.code_predictor_config {
                let cp = CodePredictor::load(
                    cp_config,
                    model_config.talker_config.hidden_size,
                    vb.pp("talker").pp("code_predictor"),
                    &device,
                    dtype,
                )
                .map_err(|e| {
                    TtsError::WeightLoadError(format!("Failed to build Code Predictor: {}", e))
                })?;
                info!(
                    "Code Predictor loaded ({} layers)",
                    cp_config.num_hidden_layers
                );
                Some(cp)
            } else {
                info!("No code_predictor_config found; single-group mode");
                None
            };

        // ── Load speech tokenizer weights ─────────────────────────────
        let speech_tokenizer = if !files.speech_tokenizer_weights.is_empty() {
            let st_vb =
                ModelFiles::load_safetensors_vb(&files.speech_tokenizer_weights, dtype, &device)?;
            let st_config = SpeechTokenizerConfig {
                num_groups: model_config.talker_config.num_code_groups,
                ..SpeechTokenizerConfig::default()
            };
            let st = SpeechTokenizerDecoder::load(&st_config, st_vb, &device).map_err(|e| {
                TtsError::WeightLoadError(format!(
                    "Failed to build Speech Tokenizer Decoder: {}",
                    e
                ))
            })?;
            info!(
                "Speech Tokenizer Decoder loaded (upsample {}x)",
                st.upsample_factor()
            );
            Some(st)
        } else {
            info!("Speech tokenizer weights not available; decode step will fail");
            None
        };

        Ok(Self {
            config: model_config,
            device,
            dtype,
            files,
            talker: Mutex::new(talker),
            code_predictor: code_predictor.map(Mutex::new),
            speech_tokenizer,
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        info!("Qwen3-TTS synthesize: \"{}\"", request.text);

        self.validate_request(request)?;

        let token_ids = self.tokenize_text(&request.text)?;
        let instruct_token_ids = self.tokenize_instruct(request.instruct.as_deref())?;
        let (speaker_id, language_id) = self.resolve_speaker_language(request);
        let (full_input, trailing_text_hidden) = self.build_input_embeddings(
            &token_ids,
            instruct_token_ids.as_deref(),
            speaker_id,
            language_id,
        )?;

        info!("Input sequence shape: {:?}", full_input.shape());

        let (codec_tokens_g0, hidden_states) =
            self.generate_codec_tokens(&full_input, &trailing_text_hidden, request)?;
        let all_codes = self.assemble_all_codes(&codec_tokens_g0, &hidden_states)?;
        let samples = self.decode_to_audio(&all_codes)?;

        info!(
            "Synthesized {} samples ({:.2}s at 24000Hz)",
            samples.len(),
            samples.len() as f32 / 24000.0
        );

        Ok(AudioSamples::new(samples, 24000))
    }

    fn sample_rate(&self) -> u32 {
        24000
    }

    fn supported_languages(&self) -> Vec<String> {
        let mut langs = self.config.languages();
        langs.sort();
        if !langs.contains(&"auto".to_string()) {
            langs.insert(0, "auto".to_string());
        }
        langs
    }

    fn supported_voices(&self) -> Vec<String> {
        self.config.speakers()
    }

    fn model_info(&self) -> ModelInfo {
        let name = if self.config.is_voice_design() {
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign".to_string()
        } else {
            "Qwen3-TTS-12Hz-1.7B-CustomVoice".to_string()
        };
        ModelInfo {
            name,
            variant: format!(
                "{} ({})",
                self.config.tts_model_type, self.config.tokenizer_type
            ),
            parameters: 1_700_000_000,
            sample_rate: 24000,
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl Qwen3TtsModel {
    /// Get a reference to the compute device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model config.
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    /// Get the resolved file paths.
    pub fn files(&self) -> &ModelFiles {
        &self.files
    }

    /// Normalize a user-supplied language tag to the key used in `codec_language_id`.
    ///
    /// Accepts ISO 639-1 codes ("en"), full names ("english"), or "auto".
    /// Returns the canonical key that exists in the model's `codec_language_id` map,
    /// or the input unchanged if no mapping is found (validation will catch it later).
    fn normalize_language(&self, lang: &str) -> String {
        // "auto" is always valid
        if lang == "auto" {
            return "auto".to_string();
        }

        // If the key already exists (case-insensitive match), return it.
        let lower = lang.to_lowercase();
        for key in self.config.talker_config.codec_language_id.keys() {
            if key.to_lowercase() == lower {
                return key.clone();
            }
        }

        // Map ISO 639-1 codes to full names.
        let full_name = match lower.as_str() {
            "en" => "english",
            "zh" | "cmn" => "chinese",
            "de" => "german",
            "fr" => "french",
            "es" => "spanish",
            "ja" | "jp" => "japanese",
            "ko" | "kr" => "korean",
            "pt" => "portuguese",
            "ru" => "russian",
            "it" => "italian",
            _ => return lang.to_string(),
        };

        // Find the canonical-cased key that matches.
        for key in self.config.talker_config.codec_language_id.keys() {
            if key.to_lowercase() == full_name {
                return key.clone();
            }
        }

        full_name.to_string()
    }

    /// Validate the synthesis request (voice cloning, voice name, language).
    fn validate_request(&self, request: &SynthesisRequest) -> Result<(), TtsError> {
        if request.reference_audio.is_some() || request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "Qwen3-TTS: voice cloning from reference audio is not yet implemented. \
                 Use a named speaker via .with_voice() instead."
                    .to_string(),
            ));
        }

        if self.config.is_voice_design() {
            if request.voice.is_some() {
                return Err(TtsError::ModelError(
                    "Qwen3-TTS VoiceDesign uses `instruct` descriptions instead of named speakers."
                        .to_string(),
                ));
            }
        } else {
            if let Some(ref voice) = request.voice {
                if !self.config.talker_config.spk_id.contains_key(voice) {
                    return Err(TtsError::UnknownVoice(voice.clone()));
                }
            }
        }

        if let Some(ref lang) = request.language {
            let normalized = self.normalize_language(lang);
            if normalized != "auto"
                && !self
                    .config
                    .talker_config
                    .codec_language_id
                    .contains_key(&normalized)
            {
                return Err(TtsError::UnsupportedLanguage(lang.clone()));
            }
        }

        Ok(())
    }

    /// Tokenize the input text using the chat template format.
    ///
    /// Returns the token IDs for the full assistant template:
    /// `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
    /// The caller should use the first 3 tokens as the role prefix and
    /// strip the last 5 tokens (closing template).
    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        let tokenizer_path = self
            .files
            .tokenizer
            .as_ref()
            .expect("validated by resolve_files");
        let tokenizer = TextTokenizer::from_asset(tokenizer_path)?;

        // Build the chat template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        let template_text = format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            text
        );
        let token_ids = tokenizer.encode(&template_text)?;

        if token_ids.len() < 8 {
            return Err(TtsError::ModelError(
                "Qwen3-TTS: text too short after tokenization".to_string(),
            ));
        }

        info!("Tokenized {} tokens (chat template)", token_ids.len());
        Ok(token_ids)
    }

    /// Tokenize an optional voice/style instruction using the upstream user-role template.
    fn tokenize_instruct(&self, instruct: Option<&str>) -> Result<Option<Vec<u32>>, TtsError> {
        let Some(instruct) = instruct.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(None);
        };

        let tokenizer_path = self
            .files
            .tokenizer
            .as_ref()
            .expect("validated by resolve_files");
        let tokenizer = TextTokenizer::from_asset(tokenizer_path)?;

        let template_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let token_ids = tokenizer.encode(&template_text)?;
        info!("Tokenized {} instruction tokens", token_ids.len());
        Ok(Some(token_ids))
    }

    /// Resolve speaker and language IDs from the request.
    fn resolve_speaker_language(&self, request: &SynthesisRequest) -> (Option<u32>, Option<u32>) {
        let speaker_id = request
            .voice
            .as_deref()
            .and_then(|v| self.config.talker_config.spk_id.get(v).copied());

        let lang_str = self.normalize_language(request.language.as_deref().unwrap_or("auto"));
        let language_id = self
            .config
            .talker_config
            .codec_language_id
            .get(&lang_str)
            .copied();

        info!(
            "Speaker ID: {:?}, Language ID: {:?}",
            speaker_id, language_id
        );
        (speaker_id, language_id)
    }

    /// Build the input embedding tensor from text tokens, speaker, and language.
    ///
    /// Returns `(input_embeds, trailing_text_hidden)`:
    /// - `input_embeds`: The combined text+codec embedding for the prefix.
    /// - `trailing_text_hidden`: Text embeddings to add at each generation step.
    ///   During generation step i, add `trailing_text_hidden[i]` to the codec embedding.
    ///   If i >= len, add tts_pad_embed instead.
    fn build_input_embeddings(
        &self,
        token_ids: &[u32],
        instruct_token_ids: Option<&[u32]>,
        speaker_id: Option<u32>,
        language_id: Option<u32>,
    ) -> Result<(Tensor, Tensor), TtsError> {
        let talker = self.talker.lock().unwrap();
        let tc = &self.config.talker_config;

        // Extract role prefix (first 3 tokens) and text content (skip first 3, last 5)
        let role_tokens = &token_ids[..3]; // <|im_start|> assistant \n
        let text_content = if token_ids.len() > 8 {
            &token_ids[3..token_ids.len() - 5]
        } else {
            &token_ids[3..token_ids.len().saturating_sub(5).max(3)]
        };

        // --- Build text channel (all positions) ---
        let mut text_ids: Vec<u32> = Vec::new();
        let mut codec_ids: Vec<u32> = Vec::new();
        let mut codec_mask: Vec<f32> = Vec::new(); // 0.0 = no codec, 1.0 = add codec

        // Positions 0-2: role prefix (text only, no codec)
        for &tok in role_tokens {
            text_ids.push(tok);
            codec_ids.push(0);
            codec_mask.push(0.0);
        }

        // Positions 3+: think/language/speaker tokens
        // Build codec prefix based on language presence
        let codec_pad = tc.codec_pad_id.unwrap_or(2148);
        let codec_bos = tc.codec_bos_id.unwrap_or(2149);
        let codec_think = tc.codec_think_id.unwrap_or(2154);
        let codec_nothink = tc.codec_nothink_id.unwrap_or(2155);
        let codec_think_bos = tc.codec_think_bos_id.unwrap_or(2156);
        let codec_think_eos = tc.codec_think_eos_id.unwrap_or(2157);

        let tts_pad = self.config.tts_pad_token_id;
        let tts_bos = self.config.tts_bos_token_id;
        let tts_eos = self.config.tts_eos_token_id;

        if let Some(lang_id) = language_id {
            // With language: think, think_bos, language, think_eos
            let think_codec = [codec_think, codec_think_bos, lang_id, codec_think_eos];
            for &c in &think_codec {
                text_ids.push(tts_pad);
                codec_ids.push(c);
                codec_mask.push(1.0);
            }
        } else {
            // Without language: nothink, think_bos, think_eos
            let think_codec = [codec_nothink, codec_think_bos, codec_think_eos];
            for &c in &think_codec {
                text_ids.push(tts_pad);
                codec_ids.push(c);
                codec_mask.push(1.0);
            }
        }

        // Speaker position
        if let Some(spk_id) = speaker_id {
            text_ids.push(tts_pad);
            codec_ids.push(spk_id);
            codec_mask.push(1.0);
        }

        // Codec pad position — reference uses tts_bos here (not tts_pad).
        // The reference builds: [tts_pad]*N + tts_bos, combined with codec[:, :-1].
        // So the last position before text has tts_bos on the text side with codec_pad on the codec side.
        text_ids.push(tts_bos);
        codec_ids.push(codec_pad);
        codec_mask.push(1.0);

        // Text content positions (each text token + codec_pad)
        for &tok in text_content {
            text_ids.push(tok);
            codec_ids.push(codec_pad);
            codec_mask.push(1.0);
        }

        // tts_eos position
        text_ids.push(tts_eos);
        codec_ids.push(codec_pad);
        codec_mask.push(1.0);

        // Generation start: tts_pad + codec_bos
        text_ids.push(tts_pad);
        codec_ids.push(codec_bos);
        codec_mask.push(1.0);

        let seq_len = text_ids.len();
        info!(
            "Input sequence: {} positions (role:3, think:{}, text:{}, special:3)",
            seq_len,
            if language_id.is_some() { 4 } else { 3 } + if speaker_id.is_some() { 1 } else { 0 },
            text_content.len(),
        );

        // --- Compute embeddings ---
        let text_ids_tensor = Tensor::new(text_ids.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?;

        let codec_ids_tensor = Tensor::new(codec_ids.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?;

        // text_projection(text_embedding(text_ids))
        let text_embeds = talker
            .embed_text(&text_ids_tensor)
            .map_err(TtsError::ComputeError)?;

        // codec_embedding(codec_ids)
        let codec_embeds = talker
            .embed_codec(&codec_ids_tensor)
            .map_err(TtsError::ComputeError)?;

        // Apply codec mask: text_embeds + codec_embeds * mask
        let mask_tensor = Tensor::new(codec_mask.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .to_dtype(text_embeds.dtype())
            .map_err(TtsError::ComputeError)?
            .reshape((1, seq_len, 1))
            .map_err(TtsError::ComputeError)?;

        let masked_codec = codec_embeds
            .broadcast_mul(&mask_tensor)
            .map_err(TtsError::ComputeError)?;

        let combined = text_embeds
            .add(&masked_codec)
            .map_err(TtsError::ComputeError)?;

        let combined = if let Some(instruct_token_ids) =
            instruct_token_ids.filter(|tokens| !tokens.is_empty())
        {
            let instruct_ids_tensor = Tensor::new(instruct_token_ids, &self.device)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(0)
                .map_err(TtsError::ComputeError)?;
            let instruct_embeds = talker
                .embed_text(&instruct_ids_tensor)
                .map_err(TtsError::ComputeError)?;
            Tensor::cat(&[&instruct_embeds, &combined], 1).map_err(TtsError::ComputeError)?
        } else {
            combined
        };

        // Compute trailing_text_hidden: text embedding added at each generation step.
        //
        // In non-streaming mode (default for CustomVoice), ALL text content is
        // already embedded in the prefill. The trailing text for generation is
        // just tts_pad_embed at every step (matching the reference implementation).
        let trailing_ids: Vec<u32> = vec![tts_pad];

        let trailing_ids_tensor = Tensor::new(trailing_ids.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?;

        let trailing_text_hidden = talker
            .embed_text(&trailing_ids_tensor)
            .map_err(TtsError::ComputeError)?;

        info!(
            "trailing_text_hidden len={} (non-streaming: tts_pad only)",
            trailing_ids.len()
        );

        Ok((combined, trailing_text_hidden))
    }

    /// Run the Talker LM to produce group-0 codec tokens and per-step group tokens.
    ///
    /// The generation loop interleaves with the Code Predictor: at each step,
    /// after the talker samples a group-0 token, the code predictor predicts
    /// groups 1-15, and the summed embedding of all 16 groups is fed back as
    /// the next step's input (matching the reference implementation).
    ///
    /// Returns `(g0_tokens, per_step_group_tokens)` where `per_step_group_tokens[i]`
    /// contains the predicted tokens for groups 1..N-1 at generation step i.
    fn generate_codec_tokens(
        &self,
        full_input: &Tensor,
        trailing_text_hidden: &Tensor,
        request: &SynthesisRequest,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>), TtsError> {
        let max_tokens = request.max_tokens.unwrap_or(2048);
        // Default to 0.9 temperature with top-k 50 (matching reference)
        let temperature = request.temperature.unwrap_or(0.9);
        let top_k = 50;

        // Build the code predictor callback if available.
        // The callback is called at each generation step with:
        //   (past_hidden, g0_token, g0_embed, device) -> (summed_embed, group_tokens)
        let has_cp = self.code_predictor.is_some();

        let (codec_tokens_g0, group_tokens) = if has_cp {
            let cp_mutex = self.code_predictor.as_ref().unwrap();
            let mut predict_fn = |past_hidden: &Tensor,
                                  g0_token: u32,
                                  g0_embed: &Tensor,
                                  dev: &Device|
             -> candle_core::Result<(Tensor, Vec<u32>)> {
                let mut cp = cp_mutex.lock().unwrap();
                cp.predict_step_and_sum(past_hidden, g0_token, g0_embed, dev)
            };
            let generation_config = TalkerGenerationConfig::new(max_tokens, temperature, top_k)
                .with_predict_and_sum_fn(&mut predict_fn);
            self.talker
                .lock()
                .unwrap()
                .generate(full_input, trailing_text_hidden, generation_config)
                .map_err(TtsError::ComputeError)?
        } else {
            self.talker
                .lock()
                .unwrap()
                .generate(
                    full_input,
                    trailing_text_hidden,
                    TalkerGenerationConfig::new(max_tokens, temperature, top_k),
                )
                .map_err(TtsError::ComputeError)?
        };

        if codec_tokens_g0.is_empty() {
            return Err(TtsError::ModelError(
                "Qwen3-TTS: Talker generated no codec tokens".to_string(),
            ));
        }

        info!(
            "Talker generated {} codec tokens (group 0)",
            codec_tokens_g0.len()
        );
        Ok((codec_tokens_g0, group_tokens))
    }

    /// Assemble all codec groups into a single tensor from the generation results.
    ///
    /// Group 0 comes from the Talker's generated tokens.
    /// Groups 1..N-1 come from the Code Predictor's per-step predictions
    /// (collected during the generation loop).
    ///
    /// Returns tensor of shape (1, num_groups, seq_len).
    fn assemble_all_codes(
        &self,
        codec_tokens_g0: &[u32],
        per_step_group_tokens: &[Vec<u32>],
    ) -> Result<Tensor, TtsError> {
        let num_groups = self.config.talker_config.num_code_groups;

        if per_step_group_tokens.is_empty() || per_step_group_tokens[0].is_empty() {
            // No code predictor — repeat group 0 for all groups
            info!("No Code Predictor; using single group mode");
            return Tensor::new(codec_tokens_g0, &self.device)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(0)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(1)
                .map_err(TtsError::ComputeError)?
                .repeat(&[1, num_groups, 1])
                .map_err(TtsError::ComputeError);
        }

        let num_extra_groups = num_groups - 1;

        // Group 0: (1, 1, seq_len)
        let g0 = Tensor::new(codec_tokens_g0, &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(1)
            .map_err(TtsError::ComputeError)?;

        let mut group_tensors: Vec<Tensor> = vec![g0];

        // Groups 1..N-1: extract from per_step_group_tokens
        for g in 0..num_extra_groups {
            let tokens: Vec<u32> = per_step_group_tokens
                .iter()
                .map(|step_tokens| {
                    if g < step_tokens.len() {
                        step_tokens[g]
                    } else {
                        0 // fallback padding
                    }
                })
                .collect();

            let gt = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(0)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(1)
                .map_err(TtsError::ComputeError)?;

            group_tensors.push(gt);
        }

        let refs: Vec<&Tensor> = group_tensors.iter().collect();
        let codes = Tensor::cat(&refs, 1).map_err(TtsError::ComputeError)?;

        info!("Assembled all codes: {:?}", codes.shape());

        Ok(codes)
    }

    /// Decode codec tokens to audio samples via the speech tokenizer.
    fn decode_to_audio(&self, all_codes: &Tensor) -> Result<Vec<f32>, TtsError> {
        let speech_tokenizer = self.speech_tokenizer.as_ref().ok_or_else(|| {
            TtsError::ModelError(
                "Qwen3-TTS: speech tokenizer not loaded; cannot decode to audio. \
                 Ensure speech_tokenizer_weights are provided or downloadable."
                    .to_string(),
            )
        })?;

        let waveform = speech_tokenizer
            .decode(all_codes)
            .map_err(TtsError::ComputeError)?;

        // The decoder may produce BF16 on Metal/GPU — always cast to F32 for output.
        let waveform = waveform
            .to_dtype(candle_core::DType::F32)
            .map_err(TtsError::ComputeError)?;
        let samples: Vec<f32> = waveform.to_vec1().map_err(TtsError::ComputeError)?;

        Ok(samples)
    }
}

impl std::fmt::Debug for Qwen3TtsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsModel")
            .field("config_type", &self.config.tts_model_type)
            .field("device", &self.device)
            .field(
                "talker_layers",
                &self.config.talker_config.num_hidden_layers,
            )
            .field("has_code_predictor", &self.code_predictor.is_some())
            .field("has_speech_tokenizer", &self.speech_tokenizer.is_some())
            .finish()
    }
}
