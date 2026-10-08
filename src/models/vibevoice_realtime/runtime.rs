use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::error::TtsError;
use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::models::vibevoice::generation::LayerKvCache;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};

use super::config::VibeVoiceRealtimeConfig;

#[derive(Clone)]
pub struct RealtimeDecoderState {
    next_position: usize,
    last_hidden: Tensor,
    layer_caches: Vec<LayerKvCache>,
}

impl RealtimeDecoderState {
    pub fn new(next_position: usize, last_hidden: Tensor, layer_caches: Vec<LayerKvCache>) -> Self {
        Self {
            next_position,
            last_hidden,
            layer_caches,
        }
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn last_hidden(&self) -> &Tensor {
        &self.last_hidden
    }

    pub fn layer_caches(&self) -> &[LayerKvCache] {
        &self.layer_caches
    }
}

pub struct RealtimeLanguageModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: Option<RmsNorm>,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl RealtimeLanguageModel {
    pub fn load(
        config: &VibeVoiceRealtimeConfig,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
        layer_count: usize,
        apply_final_norm: bool,
    ) -> Result<Self, TtsError> {
        let decoder = &config.decoder_config;
        let head_dim = decoder.hidden_size / decoder.num_attention_heads;
        let gqa_config = GqaConfig::with_head_dim(
            decoder.hidden_size,
            decoder.num_attention_heads,
            decoder.num_key_value_heads,
            head_dim,
            decoder.max_position_embeddings,
            decoder.rope_theta,
            decoder.rms_norm_eps,
        )
        .with_attention_bias(decoder.attention_bias);

        let embed_tokens = candle_nn::embedding(
            decoder.vocab_size,
            decoder.hidden_size,
            vb.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            layers.push(TransformerBlock::load(
                &gqa_config,
                decoder.intermediate_size,
                vb.pp(format!("layers.{index}")),
            )?);
        }

        let norm = if apply_final_norm {
            Some(RmsNorm::load(
                decoder.hidden_size,
                decoder.rms_norm_eps,
                vb.pp("norm"),
            )?)
        } else {
            None
        };
        let (rope_cos, rope_sin) = precompute_rope_freqs(
            head_dim,
            decoder.max_position_embeddings,
            decoder.rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope_cos,
            rope_sin,
        })
    }

    pub fn embed(&self, token_ids: &Tensor) -> CandleResult<Tensor> {
        self.embed_tokens.forward(token_ids)
    }

    pub fn load_cache_state(&mut self, cache_state: &[LayerKvCache]) -> Result<(), TtsError> {
        if cache_state.len() != self.layers.len() {
            return Err(TtsError::ModelError(format!(
                "VibeVoice Realtime cache state has {} layer(s), expected {}",
                cache_state.len(),
                self.layers.len(),
            )));
        }

        for (layer, cache_entry) in self.layers.iter_mut().zip(cache_state.iter()) {
            layer.restore_cache(cache_entry.clone());
        }
        Ok(())
    }

    pub fn decode_step(
        &mut self,
        input_embedding: &Tensor,
        start_pos: usize,
    ) -> Result<RealtimeDecoderState, TtsError> {
        let input_embeds = match input_embedding.rank() {
            1 => input_embedding.unsqueeze(0)?.unsqueeze(0)?,
            2 => input_embedding.unsqueeze(0)?,
            3 => input_embedding.clone(),
            _ => {
                return Err(TtsError::ModelError(
                    "Unexpected VibeVoice Realtime embedding rank while decoding incrementally"
                        .to_string(),
                ));
            }
        };

        let mut hidden = input_embeds;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rope_cos, &self.rope_sin, start_pos, None)?;
        }
        if let Some(norm) = &self.norm {
            hidden = norm.forward(&hidden)?;
        }
        let last_hidden = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?;

        Ok(RealtimeDecoderState::new(
            start_pos + 1,
            last_hidden,
            self.layers
                .iter()
                .map(TransformerBlock::snapshot_cache)
                .collect(),
        ))
    }
}

pub struct BinaryClassifier {
    fc1: Linear,
    fc2: Linear,
}

impl BinaryClassifier {
    pub fn load(hidden_size: usize, vb: VarBuilder) -> CandleResult<Self> {
        let fc1 = candle_nn::linear(hidden_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_size, 1, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, hidden: &Tensor) -> CandleResult<Tensor> {
        let hidden = self.fc1.forward(hidden)?.relu()?;
        self.fc2.forward(&hidden)
    }
}
