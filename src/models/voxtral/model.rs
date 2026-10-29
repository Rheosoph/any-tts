use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{DType as ConfigDType, ModelAsset, ModelAssetDir, ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::layers::conv::apply_weight_norm;
use crate::layers::kv_cache::KvCache;
use crate::tensor_utils::{
    apply_rotary_emb, cpu_flash_attention, precompute_rope_freqs, silu, RmsNorm,
};
use crate::traits::{ModelInfo, SynthesisRequest, TtsModel};

use super::config::{
    AcousticTransformerArgs, AudioTokenizerArgs, MultimodalAudioModelArgs, VoxtralConfig,
};
use super::tokenizer::VoxtralTokenizer;

const VOXTRAL_LANGUAGES: [&str; 9] = ["en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"];
const DEFAULT_MAX_TOKENS: usize = 2500;
const AUDIO_SPECIAL_TOKEN_COUNT: u32 = 2;
const EMPTY_AUDIO_TOKEN_ID: u32 = 0;
const END_AUDIO_TOKEN_ID: u32 = 1;
const CODEC_CHUNK_FRAMES: usize = 375;
const CFG_ALPHA: f32 = 1.2;
const ACOUSTIC_DECODE_STEPS: usize = 8;

pub struct VoxtralModel {
    config: TtsConfig,
    model_config: VoxtralConfig,
    tokenizer: VoxtralTokenizer,
    compute_dtype: DType,
    compute_device: Device,
    lm: Mutex<MistralLm>,
    acoustic_transformer: FlowMatchingAudioTransformer,
    audio_codebook_embeddings: AudioCodebookEmbeddings,
    audio_decoder: VoxtralAudioDecoder,
    preset_voices: HashMap<String, Tensor>,
    supported_voices: Vec<String>,
}

impl TtsModel for VoxtralModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let compute_device = config.device.resolve()?;
        let compute_dtype = select_compute_dtype(config.dtype, &compute_device);

        let files = config.resolve_files()?;
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = VoxtralConfig::from_bytes(config_bytes.as_ref())?;
        let tokenizer_bytes = files
            .tokenizer
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let tokenizer = VoxtralTokenizer::from_bytes(tokenizer_bytes.as_ref(), &model_config)?;

        let main_vb = load_weight_var_builder(&files.weights, compute_dtype, &compute_device)?;
        let lm = MistralLm::load(
            &model_config,
            main_vb.clone(),
            &compute_device,
            compute_dtype,
        )?;
        let acoustic_transformer = FlowMatchingAudioTransformer::load(
            &model_config.multimodal.audio_model_args,
            main_vb.pp("acoustic_transformer"),
        )?;
        let audio_codebook_embeddings = AudioCodebookEmbeddings::load(
            &model_config.multimodal.audio_model_args,
            model_config.dim,
            main_vb
                .pp("mm_audio_embeddings")
                .pp("audio_codebook_embeddings"),
        )?;

        let audio_device = if matches!(compute_device, Device::Metal(_)) {
            Device::Cpu
        } else {
            compute_device.clone()
        };
        let audio_dtype = if matches!(audio_device, Device::Cpu) {
            DType::F32
        } else {
            compute_dtype
        };
        let audio_vb =
            if same_device_kind(&audio_device, &compute_device) && audio_dtype == compute_dtype {
                main_vb.clone()
            } else {
                load_weight_var_builder(&files.weights, audio_dtype, &audio_device)?
            };
        let audio_decoder = VoxtralAudioDecoder::load(
            &model_config.multimodal.audio_tokenizer_args,
            audio_vb.pp("audio_tokenizer"),
            &audio_device,
            audio_dtype,
        )?;

        let supported_voices = model_config.multimodal.audio_tokenizer_args.voice_names();
        let preset_voices = load_preset_voices(
            files
                .voices_dir
                .as_ref()
                .expect("validated by resolve_files"),
            &supported_voices,
            model_config.dim,
            &compute_device,
            compute_dtype,
        )?;

        info!(
            "Loading native Voxtral on {:?} ({:?} weights, {:?} audio decoder)",
            compute_device, compute_dtype, audio_device
        );

        Ok(Self {
            config,
            model_config,
            tokenizer,
            compute_dtype,
            compute_device,
            lm: Mutex::new(lm),
            acoustic_transformer,
            audio_codebook_embeddings,
            audio_decoder,
            preset_voices,
            supported_voices,
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        if request.speed.is_some() {
            return Err(TtsError::ModelError(
                "Voxtral does not support explicit speaking speed".into(),
            ));
        }
        let debug_frames = std::env::var_os("VOXTRAL_DEBUG_FRAMES").is_some();

        if request.reference_audio.is_some() {
            return Err(TtsError::ModelError(
                "The open Voxtral checkpoint does not ship encoder weights, so reference-audio voice cloning is unavailable.".to_string(),
            ));
        }

        if let Some(language) = &request.language {
            if !self
                .supported_languages()
                .iter()
                .any(|value| value.eq_ignore_ascii_case(language))
            {
                return Err(TtsError::UnsupportedLanguage(language.clone()));
            }
        }

        let voice_embedding = self.resolve_voice_embedding(request)?;
        let voice_token_count = voice_embedding.dim(0)?;
        let prompt_ids = self
            .tokenizer
            .build_speech_prompt(&request.text, voice_token_count);

        let mut lm = self.lm.lock().map_err(|_| {
            TtsError::RuntimeError("Failed to lock Voxtral language model state".to_string())
        })?;
        lm.reset();

        let prompt_embeddings = self.build_prompt_embeddings(&lm, &prompt_ids, &voice_embedding)?;
        let prompt_len = prompt_embeddings.dim(1)?;
        let prompt_mask = make_causal_mask(prompt_len, self.compute_dtype, &self.compute_device)?;
        let hidden = lm.forward_embeddings(&prompt_embeddings, 0, Some(&prompt_mask))?;
        let mut last_hidden = hidden
            .narrow(1, prompt_len - 1, 1)?
            .squeeze(1)?
            .contiguous()?;

        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let mut generated_frames = Vec::with_capacity(max_tokens.saturating_add(1));

        for step in 0..max_tokens {
            let frame = self.acoustic_transformer.generate_frame(&last_hidden)?;
            let finished = frame[0] == END_AUDIO_TOKEN_ID;
            generated_frames.push(frame.clone());
            if finished {
                break;
            }

            let next_embedding = self
                .audio_codebook_embeddings
                .embed_frame(&frame, &self.compute_device)?;
            let next_embedding = next_embedding.unsqueeze(0)?.unsqueeze(0)?;
            let step_hidden = lm.forward_embeddings(&next_embedding, prompt_len + step, None)?;
            last_hidden = step_hidden.squeeze(1)?.contiguous()?;
        }
        drop(lm);

        if generated_frames.is_empty()
            || generated_frames.last().map(|frame| frame[0]) != Some(END_AUDIO_TOKEN_ID)
        {
            generated_frames.push(make_terminal_frame(
                self.model_config
                    .multimodal
                    .audio_model_args
                    .n_acoustic_codebook,
            ));
        }

        if debug_frames {
            log_generated_frame_summary(prompt_len, &generated_frames);
        }

        let mut samples = self.audio_decoder.decode_frames(&generated_frames)?;
        for sample in &mut samples {
            *sample = sample.clamp(-1.0, 1.0);
        }

        Ok(AudioSamples::new(
            samples,
            self.model_config
                .multimodal
                .audio_model_args
                .audio_encoding_args
                .sampling_rate,
        ))
    }

    fn sample_rate(&self) -> u32 {
        self.model_config
            .multimodal
            .audio_model_args
            .audio_encoding_args
            .sampling_rate
    }

    fn supported_languages(&self) -> Vec<String> {
        VOXTRAL_LANGUAGES
            .iter()
            .map(|language| (*language).to_string())
            .collect()
    }

    fn supported_voices(&self) -> Vec<String> {
        self.supported_voices.clone()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Voxtral-4B-TTS-2603".to_string(),
            variant: self.config.effective_model_ref().to_string(),
            parameters: 4_000_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl VoxtralModel {
    fn resolve_voice_embedding(&self, request: &SynthesisRequest) -> Result<Tensor, TtsError> {
        if let Some(embedding) = &request.voice_embedding {
            if embedding.model_type() != "voxtral" {
                return Err(TtsError::ModelError(format!(
                    "Voice embedding model type '{}' is not compatible with Voxtral",
                    embedding.model_type()
                )));
            }
            let tensor = embedding.to_tensor(&self.compute_device)?;
            return normalize_voice_embedding(tensor, self.model_config.dim, self.compute_dtype);
        }

        let default_voice = self
            .supported_voices
            .iter()
            .find(|voice| voice.as_str() == "neutral_male")
            .cloned()
            .or_else(|| self.supported_voices.first().cloned())
            .ok_or_else(|| {
                TtsError::ConfigError("Voxtral preset voices are unavailable".to_string())
            })?;
        let voice_name = request.voice.clone().unwrap_or(default_voice);
        let tensor = self
            .preset_voices
            .get(&voice_name)
            .ok_or_else(|| TtsError::UnknownVoice(voice_name.clone()))?;
        Ok(tensor.clone())
    }

    fn build_prompt_embeddings(
        &self,
        lm: &MistralLm,
        prompt_ids: &[u32],
        voice_embedding: &Tensor,
    ) -> Result<Tensor, TtsError> {
        let input_ids = Tensor::new(prompt_ids, &self.compute_device)?.unsqueeze(0)?;
        let text_embeddings = lm.embed_tokens(&input_ids)?;
        let voice_len = voice_embedding.dim(0)?;
        if voice_len == 0 {
            return Ok(text_embeddings);
        }

        let prefix = text_embeddings.narrow(1, 0, 2)?;
        let suffix_start = 2 + voice_len;
        let suffix_len = prompt_ids.len().saturating_sub(suffix_start);
        let suffix = text_embeddings.narrow(1, suffix_start, suffix_len)?;
        let voice = voice_embedding.unsqueeze(0)?;
        Ok(Tensor::cat(&[&prefix, &voice, &suffix], 1)?)
    }
}

struct MistralFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl MistralFeedForward {
    fn load(
        dim: usize,
        hidden_dim: usize,
        use_biases: bool,
        vb: VarBuilder,
    ) -> Result<Self, TtsError> {
        let w1 = candle_nn::linear_no_bias(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = linear_with_optional_bias(hidden_dim, dim, use_biases, vb.pp("w2"))?;
        let w3 = candle_nn::linear_no_bias(dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        let gate = silu(&self.w1.forward(x)?)?;
        let value = self.w3.forward(x)?;
        Ok(self.w2.forward(&gate.broadcast_mul(&value)?)?)
    }
}

struct MistralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: KvCache,
}

impl MistralAttention {
    fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self, TtsError> {
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        let q_proj =
            permuted_rope_linear(config.dim, config.n_heads, config.head_dim, vb.pp("wq"))?;
        let k_proj =
            permuted_rope_linear(config.dim, config.n_kv_heads, config.head_dim, vb.pp("wk"))?;
        let v_proj = linear_with_optional_bias(config.dim, kv_dim, false, vb.pp("wv"))?;
        let o_proj = linear_with_optional_bias(q_dim, config.dim, false, vb.pp("wo"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.n_heads,
            num_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            kv_cache: KvCache::new(),
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, TtsError> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let cos = rope_cos
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = rope_sin
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let (q, k) = apply_rotary_emb(&q, &k, &cos, &sin)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_output =
            if let Some(cpu_attn_output) = cpu_flash_attention(&q, &k, &v, mask, scale)? {
                cpu_attn_output
            } else if seq_len == 1 && mask.is_none() && matches!(x.device(), Device::Metal(_)) {
                candle_nn::ops::sdpa(&q.contiguous()?, &k, &v, None, false, scale, 1.0)?
            } else {
                let k = repeat_kv_heads(&k, self.num_heads / self.num_kv_heads)?;
                let v = repeat_kv_heads(&v, self.num_heads / self.num_kv_heads)?;
                // The fused CPU path accepts cache strides; dense CPU GEMM
                // requires packed values for multi-batch and Accelerate use.
                let v = if x.device().is_cpu() {
                    v.contiguous()?
                } else {
                    v
                };

                let k_t = k.transpose(2, 3)?.contiguous()?;
                let attn_scores = q.matmul(&k_t)?.affine(scale as f64, 0.0)?;
                let attn_scores = if let Some(mask) = mask {
                    attn_scores.broadcast_add(mask)?
                } else {
                    attn_scores
                };
                let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;
                attn_probs.matmul(&v)?
            };
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }
}

struct MistralBlock {
    attention_norm: RmsNorm,
    attention: MistralAttention,
    ffn_norm: RmsNorm,
    feed_forward: MistralFeedForward,
}

impl MistralBlock {
    fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self, TtsError> {
        let attention_norm = RmsNorm::load(config.dim, config.norm_eps, vb.pp("attention_norm"))?;
        let attention = MistralAttention::load(config, vb.pp("attention"))?;
        let ffn_norm = RmsNorm::load(config.dim, config.norm_eps, vb.pp("ffn_norm"))?;
        let feed_forward =
            MistralFeedForward::load(config.dim, config.hidden_dim, false, vb.pp("feed_forward"))?;
        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, TtsError> {
        let attn = self.attention.forward(
            &self.attention_norm.forward(x)?,
            rope_cos,
            rope_sin,
            start_pos,
            mask,
        )?;
        let hidden = x.add(&attn)?;
        let ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?;
        Ok(hidden.add(&ff)?)
    }

    fn clear_cache(&mut self) {
        self.attention.clear_cache();
    }
}

struct MistralLm {
    tok_embeddings: Embedding,
    norm: RmsNorm,
    layers: Vec<MistralBlock>,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl MistralLm {
    fn load(
        config: &VoxtralConfig,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, TtsError> {
        let tok_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.dim,
            vb.pp("mm_audio_embeddings").pp("tok_embeddings"),
        )?;
        let norm = RmsNorm::load(config.dim, config.norm_eps, vb.pp("norm"))?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for layer_index in 0..config.n_layers {
            layers.push(MistralBlock::load(
                config,
                vb.pp(format!("layers.{layer_index}")),
            )?);
        }

        let (rope_cos, rope_sin) = precompute_rope_freqs(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            tok_embeddings,
            norm,
            layers,
            rope_cos,
            rope_sin,
        })
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor, TtsError> {
        Ok(self.tok_embeddings.forward(input_ids)?)
    }

    fn forward_embeddings(
        &mut self,
        embeddings: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, TtsError> {
        let mut hidden = embeddings.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rope_cos, &self.rope_sin, start_pos, mask)?;
        }
        Ok(self.norm.forward(&hidden)?)
    }
}

struct TimeEmbedding {
    inv_freq: Tensor,
}

impl TimeEmbedding {
    fn new(dim: usize, device: &Device) -> Result<Self, TtsError> {
        let half_dim = dim / 2;
        let values: Vec<f32> = (0..half_dim)
            .map(|index| (-10000.0f64.ln() * index as f64 / half_dim as f64).exp() as f32)
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(values.as_slice(), device)?,
        })
    }

    fn forward(&self, t: f32, dtype: DType) -> Result<Tensor, TtsError> {
        let t = Tensor::new(&[t], self.inv_freq.device())?.reshape((1, 1))?;
        let emb = t.broadcast_mul(&self.inv_freq.unsqueeze(0)?)?;
        Ok(Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1)?.to_dtype(dtype)?)
    }
}

struct BidirectionalAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl BidirectionalAttention {
    fn load(args: &AcousticTransformerArgs, vb: VarBuilder) -> Result<Self, TtsError> {
        let wq = linear_with_optional_bias(
            args.dim,
            args.n_heads * args.head_dim,
            args.use_biases,
            vb.pp("wq"),
        )?;
        let wk = linear_with_optional_bias(
            args.dim,
            args.n_kv_heads * args.head_dim,
            false,
            vb.pp("wk"),
        )?;
        let wv = linear_with_optional_bias(
            args.dim,
            args.n_kv_heads * args.head_dim,
            args.use_biases,
            vb.pp("wv"),
        )?;
        let wo = linear_with_optional_bias(
            args.n_heads * args.head_dim,
            args.dim,
            args.use_biases,
            vb.pp("wo"),
        )?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let xq = self
            .wq
            .forward(x)?
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?;
        let xk =
            self.wk
                .forward(x)?
                .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?;
        let xv =
            self.wv
                .forward(x)?
                .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?;

        let xk = repeat_last_head_axis(&xk, self.n_heads / self.n_kv_heads)?;
        let xv = repeat_last_head_axis(&xv, self.n_heads / self.n_kv_heads)?;

        let q = xq.transpose(1, 2)?.contiguous()?;
        let k = xk.transpose(1, 2)?.contiguous()?;
        let v = xv.transpose(1, 2)?.contiguous()?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = if let Some(cpu_attn_output) = cpu_flash_attention(&q, &k, &v, None, scale)? {
            cpu_attn_output
        } else {
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn = q.matmul(&k_t)?.affine(scale as f64, 0.0)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            attn.matmul(&v)?
        };
        let out =
            out.transpose(1, 2)?
                .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;
        Ok(self.wo.forward(&out)?)
    }
}

struct AcousticTransformerBlock {
    attention: BidirectionalAttention,
    attention_norm: RmsNorm,
    feed_forward: MistralFeedForward,
    ffn_norm: RmsNorm,
}

impl AcousticTransformerBlock {
    fn load(args: &AcousticTransformerArgs, vb: VarBuilder) -> Result<Self, TtsError> {
        let attention = BidirectionalAttention::load(args, vb.pp("attention"))?;
        let attention_norm = RmsNorm::load(args.dim, args.norm_eps, vb.pp("attention_norm"))?;
        let feed_forward = MistralFeedForward::load(
            args.dim,
            args.hidden_dim,
            args.use_biases,
            vb.pp("feed_forward"),
        )?;
        let ffn_norm = RmsNorm::load(args.dim, args.norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention,
            attention_norm,
            feed_forward,
            ffn_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        let attn = self.attention.forward(&self.attention_norm.forward(x)?)?;
        let hidden = x.add(&attn)?;
        let ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?;
        Ok(hidden.add(&ff)?)
    }
}

struct FlowMatchingAudioTransformer {
    model_args: MultimodalAudioModelArgs,
    time_embedding: TimeEmbedding,
    input_projection: Linear,
    time_projection: Linear,
    llm_projection: Linear,
    semantic_codebook_output: Linear,
    acoustic_codebook_output: Linear,
    layers: Vec<AcousticTransformerBlock>,
    norm: RmsNorm,
    timesteps: Vec<f32>,
}

impl FlowMatchingAudioTransformer {
    fn load(model_args: &MultimodalAudioModelArgs, vb: VarBuilder) -> Result<Self, TtsError> {
        let args = &model_args.acoustic_transformer_args;
        let time_embedding = TimeEmbedding::new(args.dim, vb.device())?;
        let input_projection = candle_nn::linear_no_bias(
            model_args.n_acoustic_codebook,
            args.dim,
            vb.pp("input_projection"),
        )?;
        let time_projection =
            candle_nn::linear_no_bias(args.dim, args.dim, vb.pp("time_projection"))?;
        let llm_projection =
            candle_nn::linear_no_bias(args.input_dim, args.dim, vb.pp("llm_projection"))?;

        let semantic_vocab = model_args.get_codebook_sizes(Some(128), true)[0];
        let semantic_codebook_output = linear_with_optional_bias(
            args.dim,
            semantic_vocab,
            args.use_biases,
            vb.pp("semantic_codebook_output"),
        )?;
        let acoustic_codebook_output = candle_nn::linear_no_bias(
            args.dim,
            model_args.n_acoustic_codebook,
            vb.pp("acoustic_codebook_output"),
        )?;

        let mut layers = Vec::with_capacity(args.n_layers);
        for layer_index in 0..args.n_layers {
            layers.push(AcousticTransformerBlock::load(
                args,
                vb.pp(format!("layers.{layer_index}")),
            )?);
        }
        let norm = RmsNorm::load(args.dim, args.norm_eps, vb.pp("norm"))?;
        let timesteps = (0..ACOUSTIC_DECODE_STEPS)
            .map(|index| index as f32 / (ACOUSTIC_DECODE_STEPS - 1) as f32)
            .collect();

        Ok(Self {
            model_args: model_args.clone(),
            time_embedding,
            input_projection,
            time_projection,
            llm_projection,
            semantic_codebook_output,
            acoustic_codebook_output,
            layers,
            norm,
            timesteps,
        })
    }

    fn generate_frame(&self, llm_hidden: &Tensor) -> Result<Vec<u32>, TtsError> {
        let semantic_logits = self
            .semantic_codebook_output
            .forward(llm_hidden)?
            .to_dtype(DType::F32)?;
        let semantic_code = select_semantic_code(
            &semantic_logits.squeeze(0)?,
            self.model_args.semantic_codebook_size,
        )?;
        let mut frame = Vec::with_capacity(self.model_args.n_acoustic_codebook + 1);
        frame.push(semantic_code);
        frame.extend(self.decode_one_frame(llm_hidden, semantic_code)?);
        Ok(frame)
    }

    fn decode_one_frame(
        &self,
        llm_hidden: &Tensor,
        semantic_code: u32,
    ) -> Result<Vec<u32>, TtsError> {
        if semantic_code == END_AUDIO_TOKEN_ID {
            return Ok(std::iter::repeat_n(
                EMPTY_AUDIO_TOKEN_ID,
                self.model_args.n_acoustic_codebook,
            )
            .collect());
        }

        let n_acoustic = self.model_args.n_acoustic_codebook;
        let mut sampled = Tensor::randn(0f32, 1.0, (1, n_acoustic), llm_hidden.device())?
            .to_dtype(llm_hidden.dtype())?;
        let zero_hidden =
            Tensor::zeros(llm_hidden.shape(), llm_hidden.dtype(), llm_hidden.device())?;

        for window in self.timesteps.windows(2) {
            let time = window[0];
            let dt = window[1] - time;
            let t_emb = self.time_embedding.forward(time, llm_hidden.dtype())?;

            let x_batched = Tensor::cat(&[&sampled, &sampled], 0)?;
            let llm_batched = Tensor::cat(&[llm_hidden, &zero_hidden], 0)?;
            let t_batched = Tensor::cat(&[&t_emb, &t_emb], 0)?;

            let velocity = self.predict_velocity(&x_batched, &llm_batched, &t_batched)?;
            let cond = velocity.narrow(0, 0, 1)?;
            let uncond = velocity.narrow(0, 1, 1)?;
            let guided = cond
                .broadcast_mul(&Tensor::new(&[CFG_ALPHA], llm_hidden.device())?.reshape((1, 1))?)?
                .broadcast_add(&uncond.broadcast_mul(
                    &Tensor::new(&[1.0 - CFG_ALPHA], llm_hidden.device())?.reshape((1, 1))?,
                )?)?;
            sampled = sampled.add(&guided.affine(dt as f64, 0.0)?)?;
        }

        let sampled = sampled.clamp(-1f32, 1f32)?.to_dtype(DType::F32)?;
        let values = sampled.to_vec2::<f32>()?;
        let acoustic_levels = self.model_args.acoustic_codebook_size as f32 - 1.0;
        Ok(values[0]
            .iter()
            .map(|value| {
                let scaled = ((value + 1.0) * 0.5 * acoustic_levels)
                    .round()
                    .clamp(0.0, acoustic_levels);
                scaled as u32 + AUDIO_SPECIAL_TOKEN_COUNT
            })
            .collect())
    }

    fn predict_velocity(
        &self,
        x_t: &Tensor,
        llm_output: &Tensor,
        t_emb: &Tensor,
    ) -> Result<Tensor, TtsError> {
        let acoustic = self.input_projection.forward(x_t)?.unsqueeze(1)?;
        let time = self.time_projection.forward(t_emb)?.unsqueeze(1)?;
        let llm = self.llm_projection.forward(llm_output)?.unsqueeze(1)?;

        let mut hidden = Tensor::cat(&[&acoustic, &time, &llm], 1)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        Ok(self.acoustic_codebook_output.forward(&hidden)?)
    }
}

struct AudioCodebookEmbeddings {
    weight: Tensor,
    offsets: Vec<u32>,
}

impl AudioCodebookEmbeddings {
    fn load(
        model_args: &MultimodalAudioModelArgs,
        embedding_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self, TtsError> {
        let codebook_sizes = model_args.get_codebook_sizes(None, true);
        let total_vocab_size: usize = codebook_sizes.iter().sum();
        let padded_size = round_up_to_multiple(total_vocab_size, 128);
        let weight = vb
            .pp("embeddings")
            .get((padded_size, embedding_dim), "weight")?;

        let mut offsets = Vec::with_capacity(codebook_sizes.len());
        let mut offset = 0u32;
        for size in codebook_sizes {
            offsets.push(offset);
            offset += size as u32;
        }

        Ok(Self { weight, offsets })
    }

    fn embed_frame(&self, frame: &[u32], device: &Device) -> Result<Tensor, TtsError> {
        let ids: Vec<u32> = frame
            .iter()
            .zip(self.offsets.iter())
            .map(|(token, offset)| token + offset)
            .collect();
        let ids = Tensor::new(ids.as_slice(), device)?;
        let embeddings = self.weight.index_select(&ids, 0)?;
        Ok(embeddings.sum(0)?)
    }
}

struct SemanticCodebook {
    embedding: Tensor,
}

impl SemanticCodebook {
    fn load(codebook_size: usize, codebook_dim: usize, vb: VarBuilder) -> Result<Self, TtsError> {
        let cluster_usage = vb.get(codebook_size, "cluster_usage")?;
        let embedding_sum = vb.get((codebook_size, codebook_dim), "embedding_sum")?;
        let cluster_usage = cluster_usage
            .to_dtype(DType::F32)?
            .clamp(1e-5, f64::MAX)?
            .unsqueeze(1)?;
        let embedding = embedding_sum
            .to_dtype(DType::F32)?
            .broadcast_div(&cluster_usage)?;
        Ok(Self { embedding })
    }

    fn decode(&self, codes: &Tensor, dtype: DType) -> Result<Tensor, TtsError> {
        let frames = codes.dim(2)?;
        let codes = codes.squeeze(0)?.squeeze(0)?;
        let embeddings = self.embedding.index_select(&codes, 0)?;
        Ok(embeddings
            .reshape((frames, self.embedding.dim(1)?))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .to_dtype(dtype)?)
    }
}

struct AcousticCodebook {
    levels: usize,
}

impl AcousticCodebook {
    fn new(levels: usize) -> Self {
        Self { levels }
    }

    fn decode(&self, codes: &Tensor, dtype: DType) -> Result<Tensor, TtsError> {
        Ok(codes
            .to_dtype(dtype)?
            .affine(2.0 / (self.levels - 1) as f64, -1.0)?)
    }
}

struct CodecQuantizer {
    semantic: SemanticCodebook,
    acoustic: AcousticCodebook,
}

impl CodecQuantizer {
    fn load(args: &AudioTokenizerArgs, vb: VarBuilder) -> Result<Self, TtsError> {
        Ok(Self {
            semantic: SemanticCodebook::load(
                args.semantic_codebook_size,
                args.semantic_dim,
                vb.pp("semantic_codebook"),
            )?,
            acoustic: AcousticCodebook::new(args.acoustic_codebook_size),
        })
    }

    fn decode(&self, codes: &Tensor, dtype: DType) -> Result<Tensor, TtsError> {
        let semantic = self.semantic.decode(&codes.narrow(1, 0, 1)?, dtype)?;
        let acoustic = self
            .acoustic
            .decode(&codes.narrow(1, 1, codes.dim(1)? - 1)?, dtype)?;
        Ok(Tensor::cat(&[&semantic, &acoustic], 1)?)
    }
}

struct TokenizerConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    dilation: usize,
    groups: usize,
    pad_mode: PadMode,
    effective_kernel_size: usize,
    padding_total: usize,
}

impl TokenizerConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        pad_mode: PadMode,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self, TtsError> {
        let weight_shape = (out_channels, in_channels / groups, kernel_size);
        let weight = load_weight_norm_tensor(weight_shape, out_channels, vb.pp("conv"))?;
        let bias = if use_bias {
            Some(vb.pp("conv").get(out_channels, "bias")?)
        } else {
            None
        };
        let effective_kernel_size = (kernel_size - 1) * dilation + 1;
        let padding_total = effective_kernel_size.saturating_sub(stride);
        Ok(Self {
            weight,
            bias,
            stride,
            dilation,
            groups,
            pad_mode,
            effective_kernel_size,
            padding_total,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        let input_len = x.dim(2)?;
        let frames = (input_len.saturating_sub(self.effective_kernel_size) + self.padding_total)
            as f64
            / self.stride as f64
            + 1.0;
        let target_length = (frames.ceil().max(1.0) as usize - 1) * self.stride
            + (self.effective_kernel_size - self.padding_total);
        let extra_padding = target_length.saturating_sub(input_len);
        let x = pad1d(x, (self.padding_total, extra_padding), self.pad_mode)?;
        let out = x.conv1d(&self.weight, 0, self.stride, self.dilation, self.groups)?;
        if let Some(bias) = &self.bias {
            Ok(out.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?)?)
        } else {
            Ok(out)
        }
    }
}

struct TokenizerConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    groups: usize,
    trim_ratio: f64,
}

impl TokenizerConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        trim_ratio: f64,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self, TtsError> {
        let weight_shape = (in_channels, out_channels / groups, kernel_size);
        let weight = load_weight_norm_tensor_transpose(weight_shape, in_channels, vb.pp("conv"))?;
        let bias = if use_bias {
            Some(vb.pp("conv").get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            groups,
            trim_ratio,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        let kernel_size = self.weight.dim(2)?;
        let total_padding = kernel_size.saturating_sub(self.stride);
        let out = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, self.groups)?;
        let out = if let Some(bias) = &self.bias {
            out.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?)?
        } else {
            out
        };
        let right_padding = (total_padding as f64 * self.trim_ratio).ceil() as usize;
        let left_padding = total_padding.saturating_sub(right_padding);
        let out_len = out.dim(2)?;
        Ok(out.narrow(2, left_padding, out_len.saturating_sub(total_padding))?)
    }
}

#[derive(Clone, Copy)]
enum PadMode {
    Reflect,
    Replicate,
}

struct CodecAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    alibi_slopes: Vec<f32>,
}

impl CodecAttention {
    fn load(args: &AudioTokenizerArgs, vb: VarBuilder) -> Result<Self, TtsError> {
        let wq = candle_nn::linear_no_bias(args.dim, args.n_heads * args.head_dim, vb.pp("wq"))?;
        let wk = candle_nn::linear_no_bias(args.dim, args.n_kv_heads * args.head_dim, vb.pp("wk"))?;
        let wv = candle_nn::linear_no_bias(args.dim, args.n_kv_heads * args.head_dim, vb.pp("wv"))?;
        let wo = linear_with_optional_bias(
            args.n_heads * args.head_dim,
            args.dim,
            args.use_biases,
            vb.pp("wo"),
        )?;
        let q_norm = if args.qk_norm {
            Some(RmsNorm::load(
                args.n_heads * args.head_dim,
                args.qk_norm_eps,
                vb.pp("q_norm"),
            )?)
        } else {
            None
        };
        let k_norm = if args.qk_norm {
            Some(RmsNorm::load(
                args.n_kv_heads * args.head_dim,
                args.qk_norm_eps,
                vb.pp("k_norm"),
            )?)
        } else {
            None
        };

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
            sliding_window: args.attn_sliding_window_size,
            alibi_slopes: alibi_slopes(args.n_heads),
        })
    }

    fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor, TtsError> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let mut q = self.wq.forward(x)?;
        let mut k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }

        let q = q
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = repeat_kv_heads(&k, self.n_heads / self.n_kv_heads)?;
        let v = repeat_kv_heads(&v, self.n_heads / self.n_kv_heads)?;
        let bias = make_alibi_bias(
            seq_len,
            &self.alibi_slopes,
            self.sliding_window,
            causal,
            q.dtype(),
            q.device(),
        )?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn = q
            .matmul(&k_t)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = attn.broadcast_add(&bias)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out =
            out.transpose(1, 2)?
                .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;
        Ok(self.wo.forward(&out)?)
    }
}

struct CodecTransformerBlock {
    attention: CodecAttention,
    attention_norm: RmsNorm,
    attention_scale: Option<Tensor>,
    feed_forward: MistralFeedForward,
    ffn_norm: RmsNorm,
    ffn_scale: Option<Tensor>,
    causal: bool,
}

impl CodecTransformerBlock {
    fn load(args: &AudioTokenizerArgs, vb: VarBuilder) -> Result<Self, TtsError> {
        let attention = CodecAttention::load(args, vb.pp("attention"))?;
        let attention_norm = RmsNorm::load(args.dim, args.norm_eps, vb.pp("attention_norm"))?;
        let feed_forward = MistralFeedForward::load(
            args.dim,
            args.hidden_dim,
            args.use_biases,
            vb.pp("feed_forward"),
        )?;
        let ffn_norm = RmsNorm::load(args.dim, args.norm_eps, vb.pp("ffn_norm"))?;
        let attention_scale = if args.layer_scale {
            Some(vb.get(args.dim, "attention_scale")?)
        } else {
            None
        };
        let ffn_scale = if args.layer_scale {
            Some(vb.get(args.dim, "ffn_scale")?)
        } else {
            None
        };
        Ok(Self {
            attention,
            attention_norm,
            attention_scale,
            feed_forward,
            ffn_norm,
            ffn_scale,
            causal: args.causal,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        let mut attn = self
            .attention
            .forward(&self.attention_norm.forward(x)?, self.causal)?;
        if let Some(scale) = &self.attention_scale {
            attn = attn.broadcast_mul(scale)?;
        }
        let hidden = x.add(&attn)?;
        let mut ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?;
        if let Some(scale) = &self.ffn_scale {
            ff = ff.broadcast_mul(scale)?;
        }
        Ok(hidden.add(&ff)?)
    }
}

struct CodecTransformer {
    layers: Vec<CodecTransformerBlock>,
}

impl CodecTransformer {
    fn load(
        args: &AudioTokenizerArgs,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self, TtsError> {
        let mut layers = Vec::with_capacity(num_layers);
        for layer_index in 0..num_layers {
            layers.push(CodecTransformerBlock::load(
                args,
                vb.pp(format!("layers.{layer_index}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        let mut hidden = x.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

enum DecoderBlock {
    Conv1d(TokenizerConv1d),
    ConvTranspose1d(TokenizerConvTranspose1d),
    Transformer(CodecTransformer),
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor, TtsError> {
        match self {
            Self::Conv1d(block) => {
                let x = x.transpose(1, 2)?.contiguous()?;
                Ok(block.forward(&x)?.transpose(1, 2)?.contiguous()?)
            }
            Self::ConvTranspose1d(block) => {
                let x = x.transpose(1, 2)?.contiguous()?;
                Ok(block.forward(&x)?.transpose(1, 2)?.contiguous()?)
            }
            Self::Transformer(block) => block.forward(x),
        }
    }
}

struct VoxtralAudioDecoder {
    quantizer: CodecQuantizer,
    decoder_blocks: Vec<DecoderBlock>,
    output_proj: TokenizerConv1d,
    patch_size: usize,
    downsample_factor: usize,
    audio_device: Device,
    audio_dtype: DType,
}

impl VoxtralAudioDecoder {
    fn load(
        args: &AudioTokenizerArgs,
        vb: VarBuilder,
        audio_device: &Device,
        audio_dtype: DType,
    ) -> Result<Self, TtsError> {
        let quantizer = CodecQuantizer::load(args, vb.pp("quantizer"))?;
        let decoder_convs_kernels = args.decoder_convs_kernels()?;
        let decoder_convs_strides = args.decoder_convs_strides()?;
        let decoder_transformer_lengths = args.decoder_transformer_lengths()?;

        let mut decoder_blocks = Vec::new();
        let mut current_window = args.attn_sliding_window_size;

        decoder_blocks.push(DecoderBlock::Conv1d(TokenizerConv1d::load(
            args.semantic_dim + args.acoustic_dim,
            args.dim,
            decoder_convs_kernels[0],
            decoder_convs_strides[0],
            1,
            1,
            PadMode::Replicate,
            false,
            vb.pp("decoder_blocks.0"),
        )?));
        if args.half_attn_window_upon_downsampling && decoder_convs_strides[0] > 1 {
            current_window *= 2;
        }

        let mut module_index = 1usize;
        for (stage_index, &num_layers) in decoder_transformer_lengths.iter().enumerate() {
            let mut stage_args = args.clone();
            stage_args.attn_sliding_window_size = current_window;
            decoder_blocks.push(DecoderBlock::Transformer(CodecTransformer::load(
                &stage_args,
                num_layers,
                vb.pp(format!("decoder_blocks.{module_index}")),
            )?));
            module_index += 1;

            if stage_index + 1 != decoder_transformer_lengths.len()
                && (decoder_convs_kernels[stage_index + 1] != 1
                    || decoder_convs_strides[stage_index + 1] != 1)
            {
                decoder_blocks.push(DecoderBlock::ConvTranspose1d(
                    TokenizerConvTranspose1d::load(
                        args.dim,
                        args.dim,
                        decoder_convs_kernels[stage_index + 1],
                        decoder_convs_strides[stage_index + 1],
                        1,
                        1.0,
                        false,
                        vb.pp(format!("decoder_blocks.{module_index}")),
                    )?,
                ));
                module_index += 1;
                if args.half_attn_window_upon_downsampling
                    && decoder_convs_strides[stage_index + 1] > 1
                {
                    current_window *= 2;
                }
            }
        }

        let output_proj = TokenizerConv1d::load(
            args.dim,
            args.pretransform_patch_size,
            args.patch_proj_kernel_size,
            1,
            1,
            1,
            PadMode::Reflect,
            false,
            vb.pp("output_proj"),
        )?;
        let downsample_factor = (args.sampling_rate as f64 / args.frame_rate()?).round() as usize;

        Ok(Self {
            quantizer,
            decoder_blocks,
            output_proj,
            patch_size: args.pretransform_patch_size,
            downsample_factor,
            audio_device: audio_device.clone(),
            audio_dtype,
        })
    }

    fn decode_frames(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, TtsError> {
        let cutoff = frames
            .iter()
            .position(|frame| frame.first().copied() == Some(END_AUDIO_TOKEN_ID))
            .unwrap_or(frames.len());
        if cutoff == 0 {
            return Ok(Vec::new());
        }

        let mut samples = Vec::new();
        for chunk in frames[..cutoff].chunks(CODEC_CHUNK_FRAMES) {
            let chunk_len = chunk.len();
            let chunk_tensor = build_codec_tensor(chunk, &self.audio_device)?;
            let audio = self.decode_tensor(&chunk_tensor)?;
            let mut chunk_samples = audio.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
            chunk_samples.truncate(chunk_len * self.downsample_factor);
            samples.extend(chunk_samples);
        }
        Ok(samples)
    }

    fn decode_tensor(&self, codes: &Tensor) -> Result<Tensor, TtsError> {
        let embeddings = self.quantizer.decode(codes, self.audio_dtype)?;
        let mut hidden = embeddings.transpose(1, 2)?.contiguous()?;
        for block in &self.decoder_blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = hidden.transpose(1, 2)?.contiguous()?;
        let hidden = self.output_proj.forward(&hidden)?;
        let (_, _, frames) = hidden.dims3()?;
        let hidden = hidden.reshape((1, 1, self.patch_size, frames))?;
        Ok(hidden
            .permute((0, 1, 3, 2))?
            .reshape((1, 1, frames * self.patch_size))?)
    }
}

fn select_compute_dtype(config_dtype: ConfigDType, device: &Device) -> DType {
    if matches!(device, Device::Cpu | Device::Metal(_)) {
        DType::F32
    } else {
        config_dtype.to_candle()
    }
}

fn linear_with_optional_bias(
    in_features: usize,
    out_features: usize,
    use_bias: bool,
    vb: VarBuilder,
) -> Result<Linear, TtsError> {
    Ok(if use_bias {
        candle_nn::linear(in_features, out_features, vb)?
    } else {
        candle_nn::linear_no_bias(in_features, out_features, vb)?
    })
}

fn permuted_rope_linear(
    in_features: usize,
    num_heads: usize,
    head_dim: usize,
    vb: VarBuilder,
) -> Result<Linear, TtsError> {
    let out_features = num_heads * head_dim;
    let weight = vb.get((out_features, in_features), "weight")?;
    let weight = permute_mistral_rope_weight(&weight, num_heads, head_dim)?;
    Ok(Linear::new(weight, None))
}

fn permute_mistral_rope_weight(
    weight: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor, TtsError> {
    if head_dim % 2 != 0 {
        return Err(TtsError::WeightLoadError(format!(
            "Unsupported odd head_dim {} for Mistral RoPE permutation",
            head_dim
        )));
    }

    let (out_features, in_features) = weight.dims2()?;
    let expected_out = num_heads * head_dim;
    if out_features != expected_out {
        return Err(TtsError::WeightLoadError(format!(
            "Unexpected Mistral attention weight shape ({}, {}) for {} heads x {} head_dim",
            out_features, in_features, num_heads, head_dim
        )));
    }

    Ok(weight
        .reshape((num_heads, head_dim / 2, 2, in_features))?
        .transpose(1, 2)?
        .reshape((out_features, in_features))?
        .contiguous()?)
}

fn load_weight_var_builder(
    weights: &[ModelAsset],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>, TtsError> {
    ModelFiles::load_safetensors_vb(weights, dtype, device)
}

fn load_preset_voices(
    voices_dir: &ModelAssetDir,
    voice_names: &[String],
    hidden_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>, TtsError> {
    let mut voices = HashMap::with_capacity(voice_names.len());
    for voice_name in voice_names {
        let asset = voices_dir.load_file(&format!("{voice_name}.pt"))?;
        voices.insert(
            voice_name.clone(),
            load_voice_tensor(&asset, hidden_size, device, dtype)?,
        );
    }
    Ok(voices)
}

fn load_voice_tensor(
    asset: &ModelAsset,
    hidden_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, TtsError> {
    let bytes = asset.read_bytes()?;
    let mut archive =
        crate::models::archive::open(std::io::Cursor::new(bytes.as_ref())).map_err(|err| {
            TtsError::WeightLoadError(format!(
                "Failed to open Voxtral voice embedding '{}': {}",
                asset.display_name(),
                err
            ))
        })?;

    let data_pkl = crate::models::archive::names(&archive)
        .map_err(|e| TtsError::WeightLoadError(format!("Invalid voice ZIP names: {e}")))?
        .into_iter()
        .find(|name| name.ends_with("data.pkl"))
        .ok_or_else(|| {
            TtsError::WeightLoadError(format!(
                "Voice embedding file '{}' is missing data.pkl",
                asset.display_name()
            ))
        })?;
    let tensor_root = data_pkl.trim_end_matches("data.pkl");

    let storage_dtype = {
        let mut entry = archive.by_name(&data_pkl).map_err(|err| {
            TtsError::WeightLoadError(format!(
                "Failed to read Voxtral voice metadata '{}': {}",
                asset.display_name(),
                err
            ))
        })?;
        let mut metadata = Vec::new();
        entry.read_to_end(&mut metadata)?;
        detect_pytorch_storage_dtype(&metadata, asset.display_name().as_ref())?
    };

    let data_name = format!("{tensor_root}data/0");
    let elem_size = dtype_size_bytes(storage_dtype)?;
    let frames = {
        let entry = archive.by_name(&data_name).map_err(|err| {
            TtsError::WeightLoadError(format!(
                "Voice embedding file '{}' is missing tensor payload '{}': {}",
                asset.display_name(),
                data_name,
                err
            ))
        })?;
        let bytes = entry.size() as usize;
        let numel = bytes / elem_size;
        if numel % hidden_size != 0 {
            return Err(TtsError::WeightLoadError(format!(
                "Voice embedding '{}' has {} values, which is not divisible by hidden size {}",
                asset.display_name(),
                numel,
                hidden_size
            )));
        }
        numel / hidden_size
    };

    let mut entry = archive.by_name(&data_name).map_err(|err| {
        TtsError::WeightLoadError(format!(
            "Failed to reopen Voxtral voice tensor '{}' from '{}': {}",
            data_name,
            asset.display_name(),
            err
        ))
    })?;
    let mut raw = Vec::with_capacity(frames * hidden_size * elem_size);
    entry.read_to_end(&mut raw)?;
    let tensor =
        Tensor::from_raw_buffer(&raw, storage_dtype, &[frames, hidden_size], &Device::Cpu)?;
    normalize_voice_embedding(
        tensor.to_device(device)?.to_dtype(dtype)?,
        hidden_size,
        dtype,
    )
}

fn normalize_voice_embedding(
    tensor: Tensor,
    hidden_size: usize,
    dtype: DType,
) -> Result<Tensor, TtsError> {
    let dims = tensor.dims().to_vec();
    let tensor = match dims.as_slice() {
        [frames, dim] if *dim == hidden_size => tensor.reshape((*frames, *dim))?,
        [dim, frames] if *dim == hidden_size => tensor.transpose(0, 1)?,
        [1, frames, dim] if *dim == hidden_size => tensor.squeeze(0)?,
        [1, dim, frames] if *dim == hidden_size => tensor.squeeze(0)?.transpose(0, 1)?,
        _ => {
            return Err(TtsError::ModelError(format!(
                "Unexpected Voxtral voice embedding shape {:?}",
                dims
            )))
        }
    };
    Ok(tensor.to_dtype(dtype)?)
}

fn make_causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor, TtsError> {
    let mut mask = vec![0f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in row + 1..seq_len {
            mask[row * seq_len + col] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)?)
}

fn repeat_kv_heads(tensor: &Tensor, repeats: usize) -> Result<Tensor, TtsError> {
    if repeats == 1 {
        return Ok(tensor.clone());
    }
    let (batch_size, heads, seq_len, head_dim) = tensor.dims4()?;
    Ok(tensor
        .unsqueeze(2)?
        .repeat(&[1, 1, repeats, 1, 1])?
        .reshape((batch_size, heads * repeats, seq_len, head_dim))?)
}

fn repeat_last_head_axis(tensor: &Tensor, repeats: usize) -> Result<Tensor, TtsError> {
    if repeats == 1 {
        return Ok(tensor.clone());
    }
    let (batch_size, seq_len, heads, head_dim) = tensor.dims4()?;
    Ok(tensor
        .unsqueeze(3)?
        .repeat(&[1, 1, 1, repeats, 1])?
        .reshape((batch_size, seq_len, heads * repeats, head_dim))?)
}

fn select_semantic_code(logits: &Tensor, semantic_vocab_size: usize) -> Result<u32, TtsError> {
    let mut values = logits.to_vec1::<f32>()?;
    if !values.is_empty() {
        values[EMPTY_AUDIO_TOKEN_ID as usize] = f32::NEG_INFINITY;
    }
    for value in values
        .iter_mut()
        .skip(AUDIO_SPECIAL_TOKEN_COUNT as usize + semantic_vocab_size)
    {
        *value = f32::NEG_INFINITY;
    }
    let (index, _) = values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| TtsError::ModelError("Empty semantic logits tensor".to_string()))?;
    Ok(index as u32)
}

fn build_codec_tensor(chunk: &[Vec<u32>], device: &Device) -> Result<Tensor, TtsError> {
    let frames = chunk.len();
    let codebooks = chunk
        .first()
        .map(|frame| frame.len())
        .ok_or_else(|| TtsError::ModelError("Cannot decode an empty Voxtral chunk".to_string()))?;
    let mut values = Vec::with_capacity(frames * codebooks);
    for codebook in 0..codebooks {
        for frame in chunk {
            values.push(frame[codebook].saturating_sub(AUDIO_SPECIAL_TOKEN_COUNT));
        }
    }
    Ok(Tensor::new(values.as_slice(), device)?.reshape((1, codebooks, frames))?)
}

fn make_terminal_frame(n_acoustic_codebooks: usize) -> Vec<u32> {
    let mut frame = vec![EMPTY_AUDIO_TOKEN_ID; n_acoustic_codebooks + 1];
    frame[0] = END_AUDIO_TOKEN_ID;
    frame
}

fn log_generated_frame_summary(prompt_len: usize, frames: &[Vec<u32>]) {
    let semantic_codes: Vec<u32> = frames.iter().map(|frame| frame[0]).collect();
    let unique_semantic_codes = semantic_codes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    let first_semantic_codes = semantic_codes.iter().take(16).copied().collect::<Vec<_>>();
    let first_frame = frames.first().cloned().unwrap_or_default();
    let ended = semantic_codes
        .last()
        .copied()
        .map(|code| code == END_AUDIO_TOKEN_ID)
        .unwrap_or(false);

    eprintln!(
        "VOXTRAL_DEBUG prompt_len={} generated_frames={} unique_semantic_codes={} ended={} first_semantic_codes={:?} first_frame={:?}",
        prompt_len,
        frames.len(),
        unique_semantic_codes,
        ended,
        first_semantic_codes,
        first_frame,
    );
}

fn detect_pytorch_storage_dtype(metadata: &[u8], path: &Path) -> Result<DType, TtsError> {
    if metadata
        .windows("BFloat16Storage".len())
        .any(|window| window == b"BFloat16Storage")
    {
        return Ok(DType::BF16);
    }
    if metadata
        .windows("HalfStorage".len())
        .any(|window| window == b"HalfStorage")
    {
        return Ok(DType::F16);
    }
    if metadata
        .windows("FloatStorage".len())
        .any(|window| window == b"FloatStorage")
    {
        return Ok(DType::F32);
    }

    Err(TtsError::WeightLoadError(format!(
        "Voice embedding '{}' uses an unsupported PyTorch storage type",
        path.display()
    )))
}

fn dtype_size_bytes(dtype: DType) -> Result<usize, TtsError> {
    match dtype {
        DType::BF16 | DType::F16 => Ok(2),
        DType::F32 => Ok(4),
        _ => Err(TtsError::WeightLoadError(format!(
            "Unsupported Voxtral voice embedding dtype {:?}",
            dtype
        ))),
    }
}

fn load_weight_norm_tensor(
    shape: (usize, usize, usize),
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Tensor, TtsError> {
    let weight_g = vb.get((out_channels, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = vb.get(shape, "parametrizations.weight.original1")?;
    Ok(apply_weight_norm(&weight_v, &weight_g)?)
}

fn load_weight_norm_tensor_transpose(
    shape: (usize, usize, usize),
    in_channels: usize,
    vb: VarBuilder,
) -> Result<Tensor, TtsError> {
    let weight_g = vb.get((in_channels, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = vb.get(shape, "parametrizations.weight.original1")?;
    Ok(apply_weight_norm(&weight_v, &weight_g)?)
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    if value % multiple == 0 {
        value
    } else {
        value + (multiple - value % multiple)
    }
}

fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    fn slopes_power_of_two(num_heads: usize) -> Vec<f32> {
        let ratio = 2.0f32.powf(-8.0 / num_heads as f32);
        (0..num_heads)
            .map(|index| ratio.powi(index as i32))
            .collect()
    }

    if (num_heads as f64).log2().fract() == 0.0 {
        return slopes_power_of_two(num_heads);
    }

    let lower_power = 2usize.pow((num_heads as f64).log2().floor() as u32);
    let mut slopes = slopes_power_of_two(lower_power);
    let extra = slopes_power_of_two(lower_power * 2);
    slopes.extend(extra.into_iter().step_by(2).take(num_heads - lower_power));
    slopes
}

fn make_alibi_bias(
    seq_len: usize,
    slopes: &[f32],
    sliding_window: usize,
    causal: bool,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, TtsError> {
    let num_heads = slopes.len();
    let mut data = vec![0f32; num_heads * seq_len * seq_len];
    for (head, slope) in slopes.iter().enumerate() {
        for row in 0..seq_len {
            for col in 0..seq_len {
                let rel_pos = col as isize - row as isize;
                let mut value = *slope * rel_pos as f32;
                if causal && rel_pos > 0 {
                    value = f32::NEG_INFINITY;
                }
                let max_right = if causal { 0 } else { sliding_window as isize };
                if rel_pos < -(sliding_window as isize) || rel_pos > max_right {
                    value = f32::NEG_INFINITY;
                }
                data[head * seq_len * seq_len + row * seq_len + col] = value;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, num_heads, seq_len, seq_len), device)?.to_dtype(dtype)?)
}

fn pad1d(x: &Tensor, paddings: (usize, usize), mode: PadMode) -> Result<Tensor, TtsError> {
    let (left, right) = paddings;
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }

    let x = x.contiguous()?;

    match mode {
        PadMode::Reflect => pad_reflect(&x, left, right),
        PadMode::Replicate => pad_replicate(&x, left, right),
    }
}

fn pad_constant_zero(x: &Tensor, left: usize, right: usize) -> Result<Tensor, TtsError> {
    let (batch_size, channels, _) = x.dims3()?;
    let mut pieces = Vec::new();
    if left > 0 {
        pieces.push(Tensor::zeros(
            (batch_size, channels, left),
            x.dtype(),
            x.device(),
        )?);
    }
    pieces.push(x.clone());
    if right > 0 {
        pieces.push(Tensor::zeros(
            (batch_size, channels, right),
            x.dtype(),
            x.device(),
        )?);
    }
    let refs = pieces.iter().collect::<Vec<_>>();
    Ok(Tensor::cat(&refs, 2)?)
}

fn pad_replicate(x: &Tensor, left: usize, right: usize) -> Result<Tensor, TtsError> {
    let mut pieces = Vec::new();
    if left > 0 {
        pieces.push(x.narrow(2, 0, 1)?.repeat(&[1, 1, left])?);
    }
    pieces.push(x.clone());
    if right > 0 {
        let last_index = x.dim(2)? - 1;
        pieces.push(x.narrow(2, last_index, 1)?.repeat(&[1, 1, right])?);
    }
    let refs = pieces.iter().collect::<Vec<_>>();
    Ok(Tensor::cat(&refs, 2)?)
}

fn pad_reflect(x: &Tensor, left: usize, right: usize) -> Result<Tensor, TtsError> {
    let length = x.dim(2)?;
    let max_pad = left.max(right);
    let mut extra_pad = 0usize;
    let mut x = x.clone();
    if length <= max_pad {
        extra_pad = max_pad - length + 1;
        x = pad_constant_zero(&x, 0, extra_pad)?;
    }

    let padded_length = x.dim(2)?;
    let mut indices = Vec::with_capacity(left + padded_length + right);
    for offset in 0..left {
        indices.push((left - offset) as u32);
    }
    for index in 0..padded_length {
        indices.push(index as u32);
    }
    for offset in 0..right {
        indices.push((padded_length - 2 - offset) as u32);
    }

    let index_tensor = Tensor::new(indices.as_slice(), x.device())?;
    let padded = x.index_select(&index_tensor, 2)?;
    if extra_pad == 0 {
        Ok(padded)
    } else {
        Ok(padded.narrow(2, 0, padded.dim(2)? - extra_pad)?)
    }
}

fn same_device_kind(left: &Device, right: &Device) -> bool {
    matches!((left, right), (Device::Cpu, Device::Cpu))
        || matches!((left, right), (Device::Cuda(_), Device::Cuda(_)))
        || matches!((left, right), (Device::Metal(_), Device::Metal(_)))
}
