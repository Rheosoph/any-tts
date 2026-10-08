//! Code Predictor — generates codec groups 1..N from group-0 hidden states.
//!
//! After the Talker LM produces group-0 tokens and hidden states, the Code
//! Predictor fills in the remaining codebook groups for each time step.
//!
//! Architecture:
//! ```text
//! talker_hidden ─► projection ─► Transformer ─┬─► head[1] ─► group-1 token
//!                                              ├─► head[2] ─► group-2 token
//!           prev_group_embed ─────────────────┘   …
//!                                              └─► head[N-1] ─► group-(N-1) token
//! ```

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};

use super::config::CodePredictorConfig;

/// The Code Predictor sub-model.
pub struct CodePredictor {
    /// Project from Talker hidden size to Code Predictor hidden size.
    input_proj: Linear,
    /// Per-group codec embeddings: group_embeds[g] for groups 1..N-1.
    group_embeds: Vec<Embedding>,
    /// Transformer layers (smaller than the Talker).
    layers: Vec<TransformerBlock>,
    /// Final norm.
    norm: RmsNorm,
    /// Per-group prediction heads.
    group_heads: Vec<Linear>,
    /// Precomputed RoPE cosine.
    rope_cos: Tensor,
    /// Precomputed RoPE sine.
    rope_sin: Tensor,
    /// Number of codebook groups (including group 0 from Talker).
    num_groups: usize,
    /// Config reference.
    config: CodePredictorConfig,
}

impl CodePredictor {
    /// Load the Code Predictor from a VarBuilder.
    pub fn load(
        config: &CodePredictorConfig,
        talker_hidden_size: usize,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let rope_theta = 1_000_000.0;
        let max_position_embeddings = 128; // CP only needs ~18 positions max

        let gqa_config = GqaConfig::with_head_dim(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            max_position_embeddings,
            rope_theta,
            1e-6,
        );

        // Projection from Talker hidden → Code Predictor hidden
        // Actual name: talker.code_predictor.small_to_mtp_projection.{weight,bias}
        let input_proj = candle_nn::linear(
            talker_hidden_size,
            config.hidden_size,
            vb.pp("small_to_mtp_projection"),
        )?;

        // Per-group embeddings (for groups 1..N-1)
        // Actual name: talker.code_predictor.model.codec_embedding.N.weight
        // Embed dim = talker_hidden_size (2048), NOT code predictor hidden (1024).
        // The flow is: codec_embed + talker_hidden → projection → transformer.
        let num_extra_groups = config.num_code_groups - 1;
        let model_vb = vb.pp("model");
        let mut group_embeds = Vec::with_capacity(num_extra_groups);
        for g in 0..num_extra_groups {
            let emb = candle_nn::embedding(
                config.vocab_size,
                talker_hidden_size,
                model_vb.pp(format!("codec_embedding.{}", g)),
            )?;
            group_embeds.push(emb);
        }

        // Transformer layers under talker.code_predictor.model.layers.N
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::load(
                &gqa_config,
                config.intermediate_size,
                model_vb.pp(format!("layers.{}", i)),
            )?;
            layers.push(block);
        }

        // Final norm: talker.code_predictor.model.norm
        let norm = RmsNorm::load(config.hidden_size, 1e-6, model_vb.pp("norm"))?;

        // Per-group heads: talker.code_predictor.lm_head.N.weight
        let mut group_heads = Vec::with_capacity(num_extra_groups);
        for g in 0..num_extra_groups {
            let head = candle_nn::linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp(format!("lm_head.{}", g)),
            )?;
            group_heads.push(head);
        }

        // Precompute RoPE for the code predictor transformer
        let (rope_cos, rope_sin) = precompute_rope_freqs(
            config.head_dim,
            max_position_embeddings,
            rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            input_proj,
            group_embeds,
            layers,
            norm,
            group_heads,
            rope_cos,
            rope_sin,
            num_groups: config.num_code_groups,
            config: config.clone(),
        })
    }

    /// Predict groups 1..N-1 for a single time step and return the summed
    /// embedding of all groups (used in the talker's feedback loop), plus
    /// the predicted token IDs for groups 1..N-1.
    ///
    /// Reference flow per generation step:
    /// 1. `last_id_hidden = talker.codec_embedding(group0_token)` — shape (1, 1, talker_hidden)
    /// 2. `code_predictor.generate(cat(past_hidden, last_id_hidden), max_new_tokens=N-1)`
    /// 3. `codec_hiddens = [last_id_hidden] + [cp_embed[i](predicted[i]) for i in 0..N-2]`
    /// 4. `return codec_hiddens.sum(1, keepdim=True)` — shape (1, 1, talker_hidden)
    ///
    /// * `past_hidden` — (1, talker_hidden_size) from talker's last hidden state
    /// * `g0_token` — group-0 token ID
    /// * `g0_embed` — (1, 1, talker_hidden_size) from talker.codec_embedding(g0_token)
    ///
    /// Returns `(summed_embed, predicted_tokens)`:
    /// - `summed_embed`: (1, 1, talker_hidden_size) — the summed embedding for all 16 groups
    /// - `predicted_tokens`: Vec<u32> of length num_groups-1 — predicted tokens for groups 1..N-1
    pub fn predict_step_and_sum(
        &mut self,
        past_hidden: &Tensor,
        _g0_token: u32,
        g0_embed: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Vec<u32>)> {
        let num_extra_groups = self.num_groups - 1;
        let mut predicted_tokens: Vec<u32> = Vec::with_capacity(num_extra_groups);

        // Start with group-0 embedding in the sum
        let mut summed = g0_embed.clone(); // (1, 1, hidden)

        // Clear KV-caches
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        // Prefill: [past_hidden(1,1,H), g0_embed(1,1,H)] in talker-hidden space
        let past_hidden_3d = past_hidden.unsqueeze(1)?; // (1, 1, H)
        let prefill = Tensor::cat(&[&past_hidden_3d, g0_embed], 1)?; // (1, 2, H)

        // Project to CP hidden size
        let prefill = self.input_proj.forward(&prefill)?; // (1, 2, cp_hidden)

        let mut hidden = prefill;

        // Build causal mask for 2-token prefill: position 0 can only see itself,
        // position 1 can see both (matching HuggingFace's causal mask behavior).
        let prefill_mask = {
            let mask_data = vec![0.0f32, f32::NEG_INFINITY, 0.0f32, 0.0f32];
            let mask = Tensor::from_vec(mask_data, (2, 2), device)?.to_dtype(hidden.dtype())?;
            mask.unsqueeze(0)?.unsqueeze(0)?
        };

        // Run transformer on prefill with real RoPE and causal mask
        for layer in &mut self.layers {
            hidden = layer.forward(
                &hidden,
                &self.rope_cos,
                &self.rope_sin,
                0,
                Some(&prefill_mask),
            )?;
        }

        let hidden = self.norm.forward(&hidden)?;

        // Predict group 1 from last position
        let last_h = hidden.i((.., hidden.dims()[1] - 1.., ..))?; // (1, 1, cp_hidden)
        let logits = self.group_heads[0].forward(&last_h)?;
        let mut prev_predicted_token = sample_top_k_top_p(&logits, 50, 0.8, device)?;
        predicted_tokens.push(prev_predicted_token);

        // Embed predicted group-1 token and add to sum
        let tok_tensor = Tensor::new(&[prev_predicted_token], device)?.unsqueeze(0)?;
        let tok_embed = self.group_embeds[0].forward(&tok_tensor)?; // (1, 1, talker_hidden)
        summed = summed.add(&tok_embed)?;

        // For remaining groups, use incremental decoding with KV-cache
        for gen_step in 1..num_extra_groups {
            let pos = gen_step + 1; // Two prefill positions, then one per group.
                                    // Embed the previous step's predicted token using its group embedding
            let prev_tok_tensor = Tensor::new(&[prev_predicted_token], device)?.unsqueeze(0)?;
            let embed_idx = (gen_step - 1).min(self.group_embeds.len() - 1);
            let prev_embed = self.group_embeds[embed_idx].forward(&prev_tok_tensor)?;

            // Project and run one step with real RoPE
            let step_input = self.input_proj.forward(&prev_embed)?;
            let mut h = step_input;
            for layer in &mut self.layers {
                h = layer.forward(&h, &self.rope_cos, &self.rope_sin, pos, None)?;
            }
            let h = self.norm.forward(&h)?;

            let logits = self.group_heads[gen_step].forward(&h)?;
            let predicted_token = sample_top_k_top_p(&logits, 50, 0.8, device)?;
            predicted_tokens.push(predicted_token);

            // Embed and add to sum
            let pred_tensor = Tensor::new(&[predicted_token], device)?.unsqueeze(0)?;
            let pred_embed_idx = gen_step.min(self.group_embeds.len() - 1);
            let pred_embed = self.group_embeds[pred_embed_idx].forward(&pred_tensor)?;
            summed = summed.add(&pred_embed)?;

            prev_predicted_token = predicted_token;
        }

        // Clear caches for next call
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        Ok((summed, predicted_tokens))
    }
}

impl std::fmt::Debug for CodePredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodePredictor")
            .field("num_groups", &self.num_groups)
            .field("num_layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .finish()
    }
}

/// Sample a token from logits using top-k + top-p (nucleus) filtering.
///
/// Reference: code predictor uses do_sample=True, top_k=50, top_p=0.8.
fn sample_top_k_top_p(logits: &Tensor, top_k: usize, top_p: f32, _device: &Device) -> Result<u32> {
    let mut values = logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    super::sampling::sample_logits(&mut values, 1.0, top_k, top_p, cp_rand_uniform())
}

/// Simple random float in [0, 1) for code predictor sampling.
fn cp_rand_uniform() -> f32 {
    use std::sync::Mutex;
    use std::time::SystemTime;

    static STATE: Mutex<Option<[u64; 4]>> = Mutex::new(None);

    let mut guard = STATE.lock().unwrap();
    let s = guard.get_or_insert_with(|| {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xcafebabe);
        let mut seed = [
            now ^ 0xabcdef0123456789,
            now.wrapping_mul(6364136223846793005),
            !now ^ 0x9876543210fedcba,
            now.rotate_left(17) ^ 0x1111111111111111,
        ];
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
