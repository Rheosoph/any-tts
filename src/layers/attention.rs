//! Grouped-Query Attention (GQA) layer.
//!
//! GQA is the attention variant used by Qwen 2 / Qwen 2.5 / Qwen 3
//! family models. It reduces KV-cache memory by sharing key/value heads
//! across multiple query heads.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::kv_cache::{KvCache, KvCacheState};
use crate::tensor_utils::{apply_rotary_emb, cpu_flash_attention, RmsNorm};

/// Configuration for a GQA layer, extracted from model config.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
}

impl GqaConfig {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
    ) -> Self {
        let head_dim = hidden_size / num_attention_heads;
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            attention_bias: false,
        }
    }

    /// Create a GqaConfig with an explicit head_dim (for models where
    /// head_dim != hidden_size / num_attention_heads, e.g. code predictor).
    pub fn with_head_dim(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
    ) -> Self {
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            attention_bias: false,
        }
    }

    pub fn with_attention_bias(mut self, attention_bias: bool) -> Self {
        self.attention_bias = attention_bias;
        self
    }
}

/// Grouped-Query Attention with rotary position embeddings.
pub struct GroupedQueryAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: KvCache,
}

impl GroupedQueryAttention {
    /// Load attention weights from a VarBuilder.
    ///
    /// Expected weight names under `vb`:
    /// - `q_proj.weight` — (num_heads * head_dim, hidden_size)
    /// - `k_proj.weight` — (num_kv_heads * head_dim, hidden_size)
    /// - `v_proj.weight` — (num_kv_heads * head_dim, hidden_size)
    /// - `o_proj.weight` — (hidden_size, num_heads * head_dim)
    pub fn load(config: &GqaConfig, vb: VarBuilder) -> Result<Self> {
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        let q_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, q_dim, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, q_dim, vb.pp("q_proj"))?
        };
        let k_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        };
        let v_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        };
        let o_proj = candle_nn::linear_no_bias(q_dim, config.hidden_size, vb.pp("o_proj"))?;

        // Qwen3 models have per-head QK-norm; Qwen2 models do not.
        let q_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("k_norm")).ok();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            kv_cache: KvCache::new(),
        })
    }

    /// Forward pass with rotary embeddings and optional KV-cache.
    ///
    /// * `x`     — (batch, seq_len, hidden_size)
    /// * `cos`   — (max_seq, head_dim) precomputed RoPE cos
    /// * `sin`   — (max_seq, head_dim) precomputed RoPE sin
    /// * `start_pos` — position offset for KV-cache (0 for full sequence)
    /// * `mask`  — optional causal attention mask
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        if start_pos != self.kv_cache.len() {
            candle_core::bail!(
                "attention position {start_pos} differs from KV cache length {}",
                self.kv_cache.len()
            )
        }

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, heads, seq_len, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-norm if present (Qwen3-style)
        let q = if let Some(ref qn) = self.q_norm {
            qn.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref kn) = self.k_norm {
            kn.forward(&k)?
        } else {
            k
        };

        // Apply RoPE
        let cos_slice = cos
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin_slice = sin
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let (q, k) = apply_rotary_emb(&q, &k, &cos_slice, &sin_slice)?;

        // Update KV-cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_output =
            if let Some(cpu_attn_output) = cpu_flash_attention(&q, &k, &v, mask, scale)? {
                cpu_attn_output
            } else if seq_len == 1
                && mask.is_none()
                && matches!(x.device(), candle_core::Device::Metal(_))
            {
                candle_nn::ops::sdpa(&q.contiguous()?, &k, &v, None, false, scale, 1.0)?
            } else {
                // Repeat KV heads to match Q heads (GQA expansion)
                // Use repeat_interleave semantics: [h0,h0,h1,h1,...] not repeat: [h0..h7,h0..h7]
                // so that Q head i pairs with KV head i/repeat_factor.
                let repeat_factor = self.num_heads / self.num_kv_heads;
                let kv_len = k.dim(2)?;
                let k = if repeat_factor > 1 {
                    k.unsqueeze(2)?
                        .repeat(&[1, 1, repeat_factor, 1, 1])?
                        .reshape((batch, self.num_heads, kv_len, self.head_dim))?
                } else {
                    k
                };
                let v = if repeat_factor > 1 {
                    v.unsqueeze(2)?
                        .repeat(&[1, 1, repeat_factor, 1, 1])?
                        .reshape((batch, self.num_heads, kv_len, self.head_dim))?
                } else {
                    v
                };

                // Dense CPU GEMM (including Accelerate) has stricter matrix
                // and batch stride requirements than fused attention.
                let (k, v) = if x.device().is_cpu() {
                    (k.contiguous()?, v.contiguous()?)
                } else {
                    (k, v)
                };

                let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
                let attn_weights = if let Some(mask) = mask {
                    attn_weights.broadcast_add(mask)?
                } else {
                    attn_weights
                };

                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&v)?
            };

        // Reshape back: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, hidden)
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }

    /// Clear the KV-cache (call between independent sequences).
    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// Snapshot the current KV-cache so another caller can restore it later.
    pub fn cache_state(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.tensors().ok().flatten()
    }

    /// Restore a previously captured KV-cache.
    pub fn set_cache_state(&mut self, kv_cache: Option<(Tensor, Tensor)>) {
        self.kv_cache.restore_tensors(kv_cache);
    }

    /// Capture a branch-safe prefix without copying the KV tensors.
    pub fn snapshot_cache(&self) -> Option<KvCacheState> {
        self.kv_cache.snapshot()
    }

    /// Restore a branch-safe prefix captured by [`Self::snapshot_cache`].
    pub fn restore_cache(&mut self, state: Option<KvCacheState>) {
        self.kv_cache.restore(state);
    }
}

impl std::fmt::Debug for GroupedQueryAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedQueryAttention")
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("has_cache", &!self.kv_cache.is_empty())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn incremental_and_dense_attention_match_full_causal_attention() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 32;
        let mut weights = HashMap::new();
        for (index, name) in [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
        ]
        .iter()
        .enumerate()
        {
            let values: Vec<f32> = (0..hidden * hidden)
                .map(|i| ((i + index) as f32 * 0.17).sin() * 0.05)
                .collect();
            weights.insert(
                name.to_string(),
                Tensor::from_vec(values, (hidden, hidden), &device)?,
            );
        }
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let config = GqaConfig::new(hidden, 2, 2, 128, 10000., 1e-6);
        let mut full = GroupedQueryAttention::load(&config, vb.clone())?;
        let mut dense = GroupedQueryAttention::load(&config, vb.clone())?;
        let mut incremental = GroupedQueryAttention::load(&config, vb)?;
        let (cos, sin) =
            crate::tensor_utils::precompute_rope_freqs(16, 128, 10000., &device, DType::F32)?;
        let values: Vec<f32> = (0..2 * 6 * hidden)
            .map(|i| (i as f32 * 0.1).cos())
            .collect();
        let input = Tensor::from_vec(values, (2, 6, hidden), &device)?;
        let mask: Vec<f32> = (0..6)
            .flat_map(|q| (0..6).map(move |k| if k > q { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let rank_two_mask = Tensor::from_vec(mask, (6, 6), &device)?;
        let mask = rank_two_mask.broadcast_as((2, 1, 6, 6))?.contiguous()?;
        let expected = full.forward(&input, &cos, &sin, 0, Some(&mask))?;
        // Rank-two masks deliberately select the dense fallback. Its cache
        // must work with multiple batches and with Accelerate's stride rules.
        let dense_output = dense.forward(&input, &cos, &sin, 0, Some(&rank_two_mask))?;
        let dense_difference = (&dense_output - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            dense_difference < 1e-5,
            "dense cache attention error: {dense_difference}"
        );
        let mut chunks = Vec::new();
        for position in 0..6 {
            chunks.push(incremental.forward(
                &input.narrow(1, position, 1)?,
                &cos,
                &sin,
                position,
                None,
            )?);
        }
        let actual = Tensor::cat(&chunks, 1)?;
        let difference = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(difference < 1e-5, "cached attention error: {difference}");
        assert!(incremental
            .forward(&input.narrow(1, 0, 1)?, &cos, &sin, 0, None)
            .is_err());
        Ok(())
    }
}
