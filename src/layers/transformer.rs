//! Transformer decoder block (pre-norm architecture).
//!
//! ```text
//! residual ─► RmsNorm ─► GQA Attention ─► + ─► RmsNorm ─► SiLU MLP ─► + ─► output
//!          └──────────────────────────────┘  └──────────────────────────┘
//! ```

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::attention::{GqaConfig, GroupedQueryAttention};
use crate::layers::kv_cache::KvCacheState;
use crate::layers::mlp::SiluMlp;
use crate::tensor_utils::RmsNorm;

/// A single transformer decoder block.
pub struct TransformerBlock {
    input_layernorm: RmsNorm,
    self_attn: GroupedQueryAttention,
    post_attention_layernorm: RmsNorm,
    mlp: SiluMlp,
}

impl TransformerBlock {
    /// Load one decoder layer from a VarBuilder.
    ///
    /// Expected sub-paths:
    /// - `input_layernorm.weight`
    /// - `self_attn.{q,k,v,o}_proj.weight`
    /// - `post_attention_layernorm.weight`
    /// - `mlp.{gate,up,down}_proj.weight`
    pub fn load(config: &GqaConfig, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;

        let self_attn = GroupedQueryAttention::load(config, vb.pp("self_attn"))?;

        let post_attention_layernorm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let mlp = SiluMlp::load(config.hidden_size, intermediate_size, vb.pp("mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Forward pass with residual connections.
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention with pre-norm
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, cos, sin, start_pos, mask)?;
        let h = (residual + h)?;

        // MLP with pre-norm
        let residual = &h;
        let out = self.post_attention_layernorm.forward(&h)?;
        let out = self.mlp.forward(&out)?;
        residual + out
    }

    /// Clear the attention KV-cache.
    pub fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }

    /// Snapshot the attention KV-cache for this layer.
    pub fn cache_state(&self) -> Option<(Tensor, Tensor)> {
        self.self_attn.cache_state()
    }

    /// Restore the attention KV-cache for this layer.
    pub fn set_cache_state(&mut self, cache_state: Option<(Tensor, Tensor)>) {
        self.self_attn.set_cache_state(cache_state);
    }

    pub fn snapshot_cache(&self) -> Option<KvCacheState> {
        self.self_attn.snapshot_cache()
    }

    pub fn restore_cache(&mut self, state: Option<KvCacheState>) {
        self.self_attn.restore_cache(state);
    }
}

impl std::fmt::Debug for TransformerBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerBlock").finish()
    }
}
