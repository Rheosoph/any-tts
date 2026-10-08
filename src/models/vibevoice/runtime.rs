use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::error::TtsError;
use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};

use super::config::VibeVoiceDecoderConfig;
use super::generation::{DecoderCacheState, LayerKvCache};

pub(crate) struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    pub(crate) fn load(input_dim: usize, output_dim: usize, vb: VarBuilder) -> CandleResult<Self> {
        let fc1 = candle_nn::linear(input_dim, output_dim, vb.pp("fc1"))?;
        let norm = RmsNorm::load(output_dim, 1e-6, vb.pp("norm"))?;
        let fc2 = candle_nn::linear(output_dim, output_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, norm, fc2 })
    }

    pub(crate) fn forward(&self, features: &Tensor) -> CandleResult<Tensor> {
        let original_dims = features.dims().to_vec();
        let input_dim = *original_dims.last().unwrap_or(&0);
        let features_2d = connector_input(features, &original_dims, input_dim)?;

        let hidden = self.fc1.forward(&features_2d)?;
        let hidden = self.norm.forward(&hidden)?;
        let hidden = self.fc2.forward(&hidden)?;

        reshape_connector_output(hidden, &original_dims)
    }
}

pub(crate) struct VibeVoiceLanguageModel {
    embed_tokens: Embedding,
    control_head: Tensor,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    rope_cos: Tensor,
    rope_sin: Tensor,
    dtype: DType,
}

impl VibeVoiceLanguageModel {
    pub(crate) fn load(
        config: &VibeVoiceDecoderConfig,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
        control_tokens: &[u32],
    ) -> Result<Self, TtsError> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let gqa_config = GqaConfig::with_head_dim(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rms_norm_eps,
        )
        .with_attention_bias(config.attention_bias);

        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let control_ids = Tensor::new(control_tokens, device)?;
        let control_head = embed_tokens
            .forward(&control_ids)?
            .transpose(0, 1)?
            .contiguous()?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(TransformerBlock::load(
                &gqa_config,
                config.intermediate_size,
                vb.pp(format!("layers.{}", index)),
            )?);
        }

        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let (rope_cos, rope_sin) = precompute_rope_freqs(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            control_head,
            layers,
            norm,
            rope_cos,
            rope_sin,
            dtype,
        })
    }

    pub(crate) fn embed(&self, token_ids: &Tensor) -> CandleResult<Tensor> {
        self.embed_tokens.forward(token_ids)
    }

    pub(crate) fn prefill(
        &mut self,
        input_embeds: &Tensor,
        project_logits: bool,
    ) -> Result<DecoderCacheState, TtsError> {
        let (_batch, seq_len, _hidden) = input_embeds.dims3()?;
        self.clear_cache();
        let mask = causal_mask(seq_len, input_embeds.device(), self.dtype)?;
        let hidden = self.forward_hidden(input_embeds, 0, mask.as_ref())?;
        self.capture_decode_state(hidden, seq_len, project_logits)
    }

    pub(crate) fn decode_step(
        &mut self,
        input_embedding: &Tensor,
        start_pos: usize,
        project_logits: bool,
    ) -> Result<DecoderCacheState, TtsError> {
        let input_embeds = step_input_embeddings(input_embedding)?;
        let hidden = self.forward_hidden(&input_embeds, start_pos, None)?;
        self.capture_decode_state(hidden, start_pos + 1, project_logits)
    }

    pub(crate) fn load_cache_state(
        &mut self,
        cache_state: &[LayerKvCache],
    ) -> Result<(), TtsError> {
        if cache_state.len() != self.layers.len() {
            return Err(TtsError::ModelError(format!(
                "VibeVoice cache state has {} layer(s), expected {}",
                cache_state.len(),
                self.layers.len(),
            )));
        }

        for (layer, cache_entry) in self.layers.iter_mut().zip(cache_state.iter()) {
            layer.restore_cache(cache_entry.clone());
        }
        Ok(())
    }

    pub(crate) fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    fn capture_decode_state(
        &self,
        hidden: Tensor,
        next_position: usize,
        project_logits: bool,
    ) -> Result<DecoderCacheState, TtsError> {
        let last_hidden = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?;
        let logits = if project_logits {
            Some(self.next_logits(&last_hidden)?)
        } else {
            None
        };
        Ok(DecoderCacheState::new(
            next_position,
            last_hidden,
            logits,
            self.capture_cache_state(),
        ))
    }

    fn capture_cache_state(&self) -> Vec<LayerKvCache> {
        self.layers
            .iter()
            .map(TransformerBlock::snapshot_cache)
            .collect()
    }

    fn forward_hidden(
        &mut self,
        input_embeds: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let mut hidden = input_embeds.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rope_cos, &self.rope_sin, start_pos, mask)?;
        }
        self.norm.forward(&hidden)
    }

    fn next_logits(&self, last_hidden: &Tensor) -> CandleResult<Tensor> {
        last_hidden.matmul(&self.control_head)
    }
}

fn flattened_leading_dims(dims: &[usize]) -> usize {
    if dims.len() <= 1 {
        return 1;
    }

    dims[..dims.len() - 1].iter().product::<usize>()
}

fn connector_input(
    features: &Tensor,
    original_dims: &[usize],
    input_dim: usize,
) -> CandleResult<Tensor> {
    if original_dims.len() == 2 {
        return Ok(features.clone());
    }

    let leading = flattened_leading_dims(original_dims);
    features.reshape((leading, input_dim))
}

fn connector_output_dims(original_dims: &[usize], hidden_dim: usize) -> Option<Vec<usize>> {
    if original_dims.len() == 2 {
        return None;
    }

    let mut output_dims = original_dims.to_vec();
    if let Some(last) = output_dims.last_mut() {
        *last = hidden_dim;
    }
    Some(output_dims)
}

fn reshape_connector_output(hidden: Tensor, original_dims: &[usize]) -> CandleResult<Tensor> {
    let Some(output_dims) =
        connector_output_dims(original_dims, hidden.dim(candle_core::D::Minus1)?)
    else {
        return Ok(hidden);
    };

    hidden.reshape(output_dims)
}

fn causal_mask(seq_len: usize, device: &Device, dtype: DType) -> CandleResult<Option<Tensor>> {
    if seq_len <= 1 {
        return Ok(None);
    }

    let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for row in 0..seq_len {
        for col in 0..=row {
            mask_data[row * seq_len + col] = 0.0;
        }
    }

    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    Ok(Some(mask))
}

fn step_input_embeddings(input_embedding: &Tensor) -> Result<Tensor, TtsError> {
    match input_embedding.rank() {
        1 => input_embedding
            .unsqueeze(0)?
            .unsqueeze(0)
            .map_err(Into::into),
        2 => input_embedding.unsqueeze(0).map_err(Into::into),
        3 => Ok(input_embedding.clone()),
        _ => Err(TtsError::ModelError(
            "Unexpected VibeVoice embedding rank while decoding incrementally".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_projection_matches_full_vocabulary_and_negative_states_skip_it() {
        let device = Device::Cpu;
        let config = VibeVoiceDecoderConfig {
            vocab_size: 19,
            hidden_size: 8,
            intermediate_size: 12,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            max_position_embeddings: 16,
            ..Default::default()
        };
        let variables = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let tokens = [17u32, 3, 12, 6];
        let mut model =
            VibeVoiceLanguageModel::load(&config, vb, &device, DType::F32, &tokens).unwrap();
        let input = model
            .embed(&Tensor::new(&[[1u32, 2]], &device).unwrap())
            .unwrap();
        let state = model.prefill(&input, true).unwrap();
        let full = state
            .last_hidden()
            .matmul(&model.embed_tokens.embeddings().transpose(0, 1).unwrap())
            .unwrap();
        let expected = full
            .index_select(&Tensor::new(&tokens, &device).unwrap(), 1)
            .unwrap();
        let actual = state.logits().unwrap();
        assert_eq!(actual.dims(), &[1, tokens.len()]);
        let error = actual
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "control projection error: {error}");
        let negative = model.prefill(&input, false).unwrap();
        assert!(negative.logits().is_err());
        assert_eq!(negative.last_hidden().dims(), &[1, config.hidden_size]);
    }
}
