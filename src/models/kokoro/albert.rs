//! ALBERT (A Lite BERT) implementation for Kokoro's PL-BERT text encoder.
//!
//! ALBERT shares transformer parameters across all layers, significantly
//! reducing model size. Kokoro uses a modified ALBERT ("CustomAlbert")
//! for phoneme-level text encoding.
//!
//! Weight names follow PyTorch's `transformers.AlbertModel`:
//! - `embeddings.word_embeddings.weight`
//! - `embeddings.position_embeddings.weight`
//! - `embeddings.token_type_embeddings.weight`
//! - `embeddings.LayerNorm.weight/bias`
//! - `encoder.embedding_hidden_mapping_in.weight/bias`
//! - `encoder.albert_layer_groups.0.albert_layers.0.attention.*`
//! - `encoder.albert_layer_groups.0.albert_layers.0.ffn.*`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::config::PlbertConfig;
use crate::tensor_utils::cpu_flash_attention;

/// ALBERT model for PL-BERT text encoding.
pub struct Albert {
    word_embeddings: candle_nn::Embedding,
    position_embeddings: candle_nn::Embedding,
    token_type_embeddings: candle_nn::Embedding,
    embed_layer_norm: candle_nn::LayerNorm,
    embedding_projection: Option<candle_nn::Linear>,
    shared_layer: AlbertLayer,
    num_hidden_layers: usize,
    _num_hidden_groups: usize,
    hidden_size: usize,
}

/// A single ALBERT transformer layer (shared across all layers).
struct AlbertLayer {
    attention: AlbertAttention,
    ffn: candle_nn::Linear,
    ffn_output: candle_nn::Linear,
    attention_norm: candle_nn::LayerNorm,
    output_norm: candle_nn::LayerNorm,
}

struct AlbertAttention {
    query: candle_nn::Linear,
    key: candle_nn::Linear,
    value: candle_nn::Linear,
    dense: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Albert {
    /// Load from a VarBuilder.
    pub fn load(config: &PlbertConfig, vb: VarBuilder, _device: &Device) -> Result<Self> {
        let emb_vb = vb.pp("embeddings");

        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.embedding_size,
            emb_vb.pp("word_embeddings"),
        )?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.embedding_size,
            emb_vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            config.type_vocab_size,
            config.embedding_size,
            emb_vb.pp("token_type_embeddings"),
        )?;
        let embed_layer_norm = candle_nn::layer_norm(
            config.embedding_size,
            candle_nn::LayerNormConfig::default(),
            emb_vb.pp("LayerNorm"),
        )?;

        // Projection from embedding_size to hidden_size (if different)
        let enc_vb = vb.pp("encoder");
        let embedding_projection = if config.embedding_size != config.hidden_size {
            Some(candle_nn::linear(
                config.embedding_size,
                config.hidden_size,
                enc_vb.pp("embedding_hidden_mapping_in"),
            )?)
        } else {
            None
        };

        // Shared transformer layer
        let layer_vb = enc_vb
            .pp("albert_layer_groups")
            .pp("0")
            .pp("albert_layers")
            .pp("0");

        let shared_layer = AlbertLayer::load(config, layer_vb)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embed_layer_norm,
            embedding_projection,
            shared_layer,
            num_hidden_layers: config.num_hidden_layers,
            _num_hidden_groups: config.num_hidden_groups,
            hidden_size: config.hidden_size,
        })
    }

    /// Forward pass.
    ///
    /// `input_ids`: [batch, seq_len] — phoneme token IDs
    /// `attention_mask`: [batch, seq_len] — 1 for valid, 0 for padded
    ///
    /// Returns: [batch, seq_len, hidden_size]
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.forward_impl(input_ids, Some(attention_mask))
    }

    /// Encode an unpadded sequence without allocating a no-op attention mask.
    pub fn forward_unmasked(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_impl(input_ids, None)
    }

    fn forward_impl(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();
        let max_positions = self.position_embeddings.embeddings().dim(0)?;
        if seq_len == 0 || seq_len > max_positions {
            candle_core::bail!("ALBERT input length {seq_len} is outside 1..={max_positions}");
        }

        // Build position IDs
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        let position_ids = Tensor::new(position_ids.as_slice(), device)?
            .unsqueeze(0)?
            .broadcast_as(input_ids.shape())?;

        // Token type IDs (all zeros for single-segment)
        let token_type_ids = Tensor::zeros(input_ids.shape(), candle_core::DType::U32, device)?;

        // Embeddings
        let word_emb = self.word_embeddings.forward(input_ids)?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;
        let type_emb = self.token_type_embeddings.forward(&token_type_ids)?;

        let embeddings = word_emb.add(&pos_emb)?.add(&type_emb)?;
        let embeddings = self.embed_layer_norm.forward(&embeddings)?;

        // Project to hidden size if needed
        let mut hidden = match &self.embedding_projection {
            Some(proj) => proj.forward(&embeddings)?,
            None => embeddings,
        };

        // ALBERT is bidirectional. A mask only excludes padding positions.
        let attn_bias = attention_mask
            .map(|mask| -> Result<Tensor> {
                if mask.dims() != [batch, seq_len] {
                    candle_core::bail!("ALBERT attention mask must match input shape");
                }
                mask.to_dtype(hidden.dtype())?
                    .affine(10000.0, -10000.0)?
                    .unsqueeze(1)?
                    .unsqueeze(2)
            })
            .transpose()?;

        // Apply shared layer N times
        for _ in 0..self.num_hidden_layers {
            hidden = self.shared_layer.forward(&hidden, attn_bias.as_ref())?;
        }

        Ok(hidden)
    }

    /// Hidden size of the output.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl AlbertLayer {
    fn load(config: &PlbertConfig, vb: VarBuilder) -> Result<Self> {
        let attn_vb = vb.pp("attention");
        let attention = AlbertAttention::load(config, attn_vb)?;

        let ffn = candle_nn::linear(config.hidden_size, config.intermediate_size, vb.pp("ffn"))?;
        let ffn_output = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("ffn_output"),
        )?;

        let attention_norm = candle_nn::layer_norm(
            config.hidden_size,
            candle_nn::LayerNormConfig::default(),
            vb.pp("attention").pp("LayerNorm"),
        )?;
        let output_norm = candle_nn::layer_norm(
            config.hidden_size,
            candle_nn::LayerNormConfig::default(),
            vb.pp("full_layer_layer_norm"),
        )?;

        Ok(Self {
            attention,
            ffn,
            ffn_output,
            attention_norm,
            output_norm,
        })
    }

    fn forward(&self, hidden: &Tensor, attn_bias: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention + residual + layer norm
        let attn_out = self.attention.forward(hidden, attn_bias)?;
        let hidden = self.attention_norm.forward(&hidden.add(&attn_out)?)?;

        // FFN + residual + layer norm
        let ffn_out = self.ffn.forward(&hidden)?;
        let ffn_out = ffn_out.gelu_erf()?;
        let ffn_out = self.ffn_output.forward(&ffn_out)?;
        let hidden = self.output_norm.forward(&hidden.add(&ffn_out)?)?;

        Ok(hidden)
    }
}

impl AlbertAttention {
    fn load(config: &PlbertConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;

        let query = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("query"))?;
        let key = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("key"))?;
        let value = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("value"))?;
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;

        Ok(Self {
            query,
            key,
            value,
            dense,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden: &Tensor, attn_bias: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden.dims3()?;

        let q = self.query.forward(hidden)?;
        let k = self.key.forward(hidden)?;
        let v = self.value.forward(hidden)?;

        // Reshape to [batch, heads, seq_len, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_out =
            if let Some(cpu_attn_out) = cpu_flash_attention(&q, &k, &v, attn_bias, scale)? {
                cpu_attn_out
            } else {
                let k_t = k.transpose(2, 3)?.contiguous()?;
                let attn_weights = q.matmul(&k_t)?;
                let attn_weights = attn_weights.affine(scale as f64, 0.0)?;
                let attn_weights = match attn_bias {
                    Some(bias) => attn_weights.broadcast_add(bias)?,
                    None => attn_weights,
                };
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&v)?
            };
        let attn_out =
            attn_out
                .transpose(1, 2)?
                .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.dense.forward(&attn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unmasked_attention_matches_dense_reference() {
        let device = Device::Cpu;
        let linear = |offset: f32| {
            let values = (0..64)
                .map(|i| (i as f32 * 0.23 + offset).sin() * 0.2)
                .collect::<Vec<_>>();
            candle_nn::Linear::new(Tensor::from_vec(values, (8, 8), &device).unwrap(), None)
        };
        let attention = AlbertAttention {
            query: linear(0.0),
            key: linear(0.5),
            value: linear(1.0),
            dense: linear(1.5),
            num_heads: 2,
            head_dim: 4,
        };
        let input = Tensor::new(
            (0..40).map(|i| (i as f32 * 0.17).cos()).collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .reshape((1, 5, 8))
        .unwrap();
        let project = |linear: &candle_nn::Linear| {
            linear
                .forward(&input)
                .unwrap()
                .reshape((1, 5, 2, 4))
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let q = project(&attention.query);
        let k = project(&attention.key);
        let v = project(&attention.value);
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap()
            .affine(0.5, 0.0)
            .unwrap();
        let expected = candle_nn::ops::softmax_last_dim(&scores)
            .unwrap()
            .matmul(&v)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .reshape((1, 5, 8))
            .unwrap();
        let expected = attention.dense.forward(&expected).unwrap();
        let actual = attention.forward(&input, None).unwrap();
        let error = actual
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-6, "CPU attention parity error: {error}");
    }

    #[test]
    fn test_albert_rejects_context_overflow() {
        let mut config: PlbertConfig = serde_json::from_str("{}").unwrap();
        config.hidden_size = 8;
        config.embedding_size = 4;
        config.intermediate_size = 16;
        config.num_attention_heads = 2;
        config.num_hidden_layers = 1;
        config.max_position_embeddings = 4;
        let device = Device::Cpu;
        let model = Albert::load(
            &config,
            VarBuilder::zeros(candle_core::DType::F32, &device),
            &device,
        )
        .unwrap();
        let input = Tensor::zeros((1, 5), candle_core::DType::U32, &device).unwrap();
        assert!(model
            .forward_unmasked(&input)
            .unwrap_err()
            .to_string()
            .contains("outside 1..=4"));
    }

    #[test]
    fn test_albert_config_defaults() {
        use super::super::config::PlbertConfig;
        let config: PlbertConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.embedding_size, 128);
    }
}
