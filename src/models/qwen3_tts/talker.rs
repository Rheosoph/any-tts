//! Talker LM — the main autoregressive language model for Qwen3-TTS.
//!
//! The Talker generates codec tokens for **group 0** (the coarsest
//! codebook) conditioned on text, speaker, and language embeddings.
//!
//! Architecture:
//! ```text
//! text_tokens → text_embed → text_proj ─┐
//!                                        ├─► Transformer Decoder ─► codec_head ─► group-0 tokens
//! codec_tokens → codec_embed ────────────┘
//! ```

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};

use super::config::TalkerConfig;
use super::sampling::{apply_repetition_penalty, sample_logits};

pub type TalkerPredictAndSumFn<'a> =
    dyn FnMut(&Tensor, u32, &Tensor, &Device) -> Result<(Tensor, Vec<u32>)> + 'a;

pub struct TalkerGenerationConfig<'a> {
    max_tokens: usize,
    temperature: f64,
    top_k: usize,
    predict_and_sum_fn: Option<&'a mut TalkerPredictAndSumFn<'a>>,
}

impl<'a> TalkerGenerationConfig<'a> {
    pub fn new(max_tokens: usize, temperature: f64, top_k: usize) -> Self {
        Self {
            max_tokens,
            temperature,
            top_k,
            predict_and_sum_fn: None,
        }
    }

    pub fn with_predict_and_sum_fn(
        mut self,
        predict_and_sum_fn: &'a mut TalkerPredictAndSumFn<'a>,
    ) -> Self {
        self.predict_and_sum_fn = Some(predict_and_sum_fn);
        self
    }
}

/// The Talker language model.
pub struct TalkerLm {
    /// Text token embedding (text_vocab_size × text_hidden_size).
    text_embed: Embedding,
    /// Codec token embedding (codec_vocab_size × hidden_size).
    codec_embed: Embedding,
    /// Projects text embeddings into the shared hidden space.
    /// Two-layer MLP: text_hidden_size → hidden_size → hidden_size.
    text_proj_fc1: Linear,
    text_proj_fc2: Linear,
    /// Transformer decoder layers.
    layers: Vec<TransformerBlock>,
    /// Final layer norm.
    norm: RmsNorm,
    /// LM head for codec token prediction.
    codec_head: Linear,
    /// Precomputed RoPE cosine.
    rope_cos: Tensor,
    /// Precomputed RoPE sine.
    rope_sin: Tensor,
    /// Model dtype (for casting runtime-created tensors to match weights).
    dtype: DType,
    /// Config for reference.
    config: TalkerConfig,
}

impl TalkerLm {
    /// Load the Talker LM from a VarBuilder.
    ///
    /// Expected weight prefix: `talker.` (e.g. `talker.text_embed.weight`).
    pub fn load(
        config: &TalkerConfig,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let gqa_config = GqaConfig::with_head_dim(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rms_norm_eps,
        );

        // Embeddings — weights live under `talker.model.*`
        let model_vb = vb.pp("model");
        let text_embed = candle_nn::embedding(
            config.text_vocab_size,
            config.text_hidden_size,
            model_vb.pp("text_embedding"),
        )?;
        let codec_embed = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("codec_embedding"),
        )?;

        // Text projection MLP — weights under `talker.text_projection.*`
        let text_proj_fc1 = candle_nn::linear(
            config.text_hidden_size,
            config.hidden_size,
            vb.pp("text_projection.linear_fc1"),
        )?;
        let text_proj_fc2 = candle_nn::linear(
            config.hidden_size,
            config.hidden_size,
            vb.pp("text_projection.linear_fc2"),
        )?;

        // Transformer layers — weights under `talker.model.layers.*`
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::load(
                &gqa_config,
                config.intermediate_size,
                model_vb.pp(format!("layers.{}", i)),
            )?;
            layers.push(block);
        }

        // Final norm — `talker.model.norm`
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

        // Codec prediction head
        let codec_head =
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("codec_head"))?;

        // Precompute RoPE
        let (rope_cos, rope_sin) = precompute_rope_freqs(
            gqa_config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            text_embed,
            codec_embed,
            text_proj_fc1,
            text_proj_fc2,
            layers,
            norm,
            codec_head,
            rope_cos,
            rope_sin,
            dtype,
            config: config.clone(),
        })
    }

    /// Project text token IDs through the text embedding + projection.
    ///
    /// Returns tensor of shape (batch, seq_len, hidden_size).
    pub fn embed_text(&self, token_ids: &Tensor) -> Result<Tensor> {
        let text_hidden = self.text_embed.forward(token_ids)?;
        let projected = self.text_proj_fc1.forward(&text_hidden)?;
        let projected = candle_nn::Activation::Silu.forward(&projected)?;
        self.text_proj_fc2.forward(&projected)
    }

    /// Embed codec token IDs.
    ///
    /// Returns tensor of shape (batch, seq_len, hidden_size).
    pub fn embed_codec(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.codec_embed.forward(token_ids)
    }

    /// Run the transformer decoder layers on a sequence of embeddings
    /// WITHOUT applying the final RMS norm.
    ///
    /// * `embeds` — (batch, seq_len, hidden_size)
    /// * `start_pos` — position offset for incremental decoding
    ///
    /// Returns **pre-norm** hidden states (batch, seq_len, hidden_size).
    /// Use `apply_norm()` afterwards to get the post-norm version for logits.
    fn forward_prenorm(&mut self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_batch, seq_len, _) = embeds.dims3()?;

        // Build causal mask (only needed for seq_len > 1)
        let mask = if seq_len > 1 {
            // Create lower-triangular mask: 0 for attend, -inf for masked
            let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    mask_data[i * seq_len + j] = 0.0;
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), embeds.device())?;
            // Cast to the model's dtype so broadcast_add with BF16 attn_weights works.
            let mask = mask.to_dtype(self.dtype)?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            None
        };

        let mut h = embeds.clone();

        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, &self.rope_cos, &self.rope_sin, start_pos, mask.as_ref())?;
        }

        Ok(h)
    }

    /// Apply the final RMS norm to pre-norm hidden states.
    fn apply_norm(&self, h: &Tensor) -> Result<Tensor> {
        self.norm.forward(h)
    }

    /// Run the transformer decoder on a sequence of embeddings.
    ///
    /// * `embeds` — (batch, seq_len, hidden_size)
    /// * `start_pos` — position offset for incremental decoding
    ///
    /// Returns **post-norm** hidden states (batch, seq_len, hidden_size).
    pub fn forward(&mut self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let h = self.forward_prenorm(embeds, start_pos)?;
        self.apply_norm(&h)
    }

    /// Compute logits from hidden states.
    ///
    /// * `hidden` — (batch, seq_len, hidden_size)
    ///
    /// Returns (batch, seq_len, vocab_size).
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.codec_head.forward(hidden)
    }

    /// Autoregressively generate codec group-0 tokens.
    ///
    /// The generation loop interleaves with the Code Predictor to produce all
    /// 16 codebook groups at each step. The flow per step:
    /// 1. Feed `step_input` (summed codec embeddings from previous step) into transformer
    /// 2. Sample group-0 token from logits
    /// 3. Return the hidden state and token so the caller can run the Code Predictor
    /// 4. Caller computes summed embedding for the next step
    ///
    /// This method handles the full loop. Optional code-predictor feedback is
    /// passed through [`TalkerGenerationConfig`].
    ///
    /// Returns `(generated_g0_tokens, per_step_group_tokens)`.
    /// `per_step_group_tokens[i]` contains the predicted tokens for groups 1..N-1
    /// at generation step i (empty vec if no code predictor).
    pub fn generate(
        &mut self,
        text_embeds: &Tensor,
        trailing_text_hidden: &Tensor,
        mut generation: TalkerGenerationConfig<'_>,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>)> {
        let (batch, prefix_len, _) = text_embeds.dims3()?;
        if batch != 1 || prefix_len == 0 || trailing_text_hidden.dim(1)? == 0 {
            candle_core::bail!("Qwen generation requires one nonempty prompt and trailing text")
        }
        let device = text_embeds.device();
        let available = self
            .rope_cos
            .dim(0)?
            .checked_sub(prefix_len)
            .ok_or_else(|| {
                candle_core::Error::Msg("Qwen prompt exceeds context capacity".into())
            })?
            + 1;
        let max_tokens = generation.max_tokens.min(available);
        if max_tokens == 0 {
            self.clear_cache();
            return Ok((Vec::new(), Vec::new()));
        }
        let temperature = generation.temperature;
        let top_k = generation.top_k;

        // Clear KV cache from any previous generation
        self.clear_cache();

        // Feed text embeddings as prefix — the last position has codec_bos,
        // and its output logits predict the FIRST codec token.
        let h_prenorm = self.forward_prenorm(text_embeds, 0)?;
        let h = self.apply_norm(&h_prenorm)?;
        let seq_len = text_embeds.dims()[1];

        // Suppress tokens in the special range [vocab_size - 1024, vocab_size)
        // except for the EOS token (matching reference implementation).
        let suppress_start = if self.config.vocab_size > 1024 {
            self.config.vocab_size - 1024
        } else {
            self.config.vocab_size
        };

        let mut generated = Vec::new();
        let mut seen = vec![false; self.config.vocab_size];
        let mut all_group_tokens: Vec<Vec<u32>> = Vec::new();

        // Repetition penalty (reference uses 1.05)
        let repetition_penalty: f32 = 1.05;
        let trailing_len = trailing_text_hidden.dims()[1];

        // --- Predict the FIRST token from the prefill output ---
        let last_hidden = h.i((.., h.dims()[1] - 1, ..))?;

        let first_logits = self.logits(&last_hidden.unsqueeze(1)?)?.squeeze(1)?;
        let first_token = sample_codec_token(
            &first_logits,
            &seen,
            repetition_penalty,
            suppress_start,
            self.config.codec_eos_token_id,
            false,
            temperature,
            top_k,
        )?;

        // Check for immediate EOS
        if first_token == self.config.codec_eos_token_id {
            self.clear_cache();
            return Ok((generated, all_group_tokens));
        }

        generated.push(first_token);
        seen[first_token as usize] = true;

        // Build next step input using code predictor on the first token
        let g0_token_tensor = Tensor::new(&[first_token], device)?.unsqueeze(0)?;
        let g0_embed = self.embed_codec(&g0_token_tensor)?;

        let (summed_embed, step_group_tokens) =
            if let Some(ref mut predict_fn) = generation.predict_and_sum_fn {
                // Reference: past_hidden = hidden_states[:, -1:, :] where hidden_states = outputs.last_hidden_state (POST-NORM)
                predict_fn(&last_hidden, first_token, &g0_embed, device)?
            } else {
                (g0_embed, vec![])
            };
        all_group_tokens.push(step_group_tokens);

        // Add trailing text hidden for text alignment.
        // Reference: generation_step=0 for first generated token.
        // In non-streaming mode, trailing_text is just tts_pad_embed (len=1),
        // so we always use index 0 = tts_pad_embed.
        let align_idx = 0;
        let text_align = trailing_text_hidden.i((.., align_idx..align_idx + 1, ..))?;
        let mut next_step_input = summed_embed.add(&text_align)?;

        // --- Continue generating remaining tokens ---
        for step in 1..max_tokens {
            let pos = seq_len + step - 1;
            // Forward one step — get pre-norm hidden for code predictor,
            // then apply norm for logits computation.
            let h_prenorm = self.forward_prenorm(&next_step_input, pos)?;
            let h = self.apply_norm(&h_prenorm)?;
            let last_hidden = h.i((.., h.dims()[1] - 1, ..))?;

            // Read logits once; apply penalty, masking and sampling on the same buffer.
            let logits = self.logits(&last_hidden.unsqueeze(1)?)?.squeeze(1)?;
            let next_token = sample_codec_token(
                &logits,
                &seen,
                repetition_penalty,
                suppress_start,
                self.config.codec_eos_token_id,
                generated.len() >= 2,
                temperature,
                top_k,
            )?;

            // Check for EOS
            if next_token == self.config.codec_eos_token_id {
                break;
            }

            generated.push(next_token);
            seen[next_token as usize] = true;

            // Build next step input using code predictor feedback loop.
            // Reference: codec_hiddens = [last_id_hidden] + [cp_embed[i](predicted_group_i) for i in 0..14]
            //            inputs_embeds = codec_hiddens.sum(1, keepdim=True)
            let g0_token_tensor = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let g0_embed = self.embed_codec(&g0_token_tensor)?;

            let (summed_embed, step_group_tokens) =
                if let Some(ref mut predict_fn) = generation.predict_and_sum_fn {
                    // past_hidden is the POST-NORM last hidden state from the transformer
                    // (matching reference: past_hidden = hidden_states[:, -1:, :] where
                    //  hidden_states = outputs.last_hidden_state, which is after final RMS norm)
                    predict_fn(&last_hidden, next_token, &g0_embed, device)?
                } else {
                    // Fallback: just use group-0 embedding
                    (g0_embed, vec![])
                };
            all_group_tokens.push(step_group_tokens);

            // Add trailing text hidden for text alignment.
            // Reference: in non-streaming mode, trailing_text_hidden is just tts_pad_embed (len=1).
            // generation_step = step (0-indexed). For trailing_len=1, always use index 0.
            let align_idx = step.min(trailing_len - 1);
            let text_align = trailing_text_hidden.i((.., align_idx..align_idx + 1, ..))?;
            next_step_input = summed_embed.add(&text_align)?;
        }

        self.clear_cache();
        Ok((generated, all_group_tokens))
    }

    /// Clear all KV-caches (between independent sequences).
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

/// Apply all codec logits processors after a single device readback.
#[allow(clippy::too_many_arguments)]
fn sample_codec_token(
    logits: &Tensor,
    seen: &[bool],
    penalty: f32,
    suppress_start: usize,
    eos: u32,
    allow_eos: bool,
    temperature: f64,
    top_k: usize,
) -> Result<u32> {
    let mut values = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    apply_repetition_penalty(&mut values, seen, penalty);
    for (index, value) in values.iter_mut().enumerate() {
        if (index >= suppress_start && index as u32 != eos) || (index as u32 == eos && !allow_eos) {
            *value = f32::NEG_INFINITY;
        }
    }
    sample_logits(&mut values, temperature, top_k, 1.0, rand_uniform())
}

/// Simple random float in [0, 1).
fn rand_uniform() -> f32 {
    // Use xoshiro256++ for better statistical quality.
    use std::sync::Mutex;
    use std::time::SystemTime;

    static STATE: Mutex<Option<[u64; 4]>> = Mutex::new(None);

    let mut guard = STATE.lock().unwrap();
    let s = guard.get_or_insert_with(|| {
        // Seed from system time + address entropy
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xdeadbeef);
        let mut seed = [
            now,
            now.wrapping_mul(6364136223846793005),
            !now,
            now ^ 0x1234567890abcdef,
        ];
        // Warm up
        for _ in 0..8 {
            let t = seed[1] << 17;
            seed[2] ^= seed[0];
            seed[3] ^= seed[1];
            seed[1] ^= seed[2];
            seed[0] ^= seed[3];
            seed[2] ^= t;
            seed[3] = seed[3].rotate_left(45);
        }
        seed
    });

    // xoshiro256++ step
    let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
    let t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = s[3].rotate_left(45);

    (result >> 40) as f32 / (1u64 << 24) as f32
}

impl std::fmt::Debug for TalkerLm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TalkerLm")
            .field("num_layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("vocab_size", &self.config.vocab_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_sampling_masks_special_tokens_and_early_eos() -> Result<()> {
        let logits = Tensor::new(&[1f32, 2., 100., 50.], &Device::Cpu)?;
        let seen = [false; 4];
        assert_eq!(
            sample_codec_token(&logits, &seen, 1.05, 2, 3, false, 0., 50)?,
            1
        );
        assert_eq!(
            sample_codec_token(&logits, &seen, 1.05, 2, 3, true, 0., 50)?,
            3
        );
        Ok(())
    }
}
