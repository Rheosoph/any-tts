use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::config::VibeVoiceTokenizerConfig;
use crate::layers::conv::{Conv1d, ConvTranspose1d};

/// Per-request causal codec history. Each entry retains only its layer's
/// receptive field, rather than the audio/latent prefix.
#[derive(Default, Clone)]
pub(crate) struct TokenizerStreamState {
    buffers: Vec<Option<Tensor>>,
}

impl TokenizerStreamState {
    fn next(&mut self, index: &mut usize) -> &mut Option<Tensor> {
        if self.buffers.len() <= *index {
            self.buffers.push(None);
        }
        let buffer = &mut self.buffers[*index];
        *index += 1;
        buffer
    }
}

struct ConvRmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl ConvRmsNorm {
    fn load(dim: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<Self> {
        let weight = if affine {
            Some(vb.get(dim, "weight")?)
        } else {
            None
        };
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?;
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let mut out = x_normed.to_dtype(dtype)?;
        if let Some(weight) = &self.weight {
            out = out.broadcast_mul(weight)?;
        }
        out.transpose(1, 2)
    }
}

struct SConv1d {
    conv: Conv1d,
    causal: bool,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
}

impl SConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = Conv1d::load(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
            vb.pp("conv").pp("conv"),
        )?;
        Ok(Self {
            conv,
            causal,
            kernel_size,
            stride,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padding_total = (self.kernel_size - 1) * self.dilation - (self.stride - 1);
        let extra_padding =
            extra_padding_for_conv1d(x.dim(2)?, self.kernel_size, self.stride, padding_total);
        let (padding_left, padding_right) = if self.causal {
            (padding_total, extra_padding)
        } else {
            let right = padding_total / 2;
            let left = padding_total - right;
            (left, right + extra_padding)
        };
        let x = pad1d_constant(x, padding_left, padding_right)?;
        self.conv.forward(&x)
    }

    fn forward_stream(&self, x: &Tensor, history: &mut Option<Tensor>) -> Result<Tensor> {
        if !self.causal {
            candle_core::bail!("Incremental tokenization requires causal convolutions")
        }
        // Strided stages consume complete strides only. The encoder entry point
        // accepts chunks aligned to the complete codec hop so no padding from a
        // temporary chunk boundary is mistaken for real future input.
        if x.dim(2)? % self.stride != 0 {
            candle_core::bail!("Incremental codec chunk must be aligned to its stride")
        }
        let context = (self.kernel_size - 1) * self.dilation - (self.stride - 1);
        let joined = match history.as_ref() {
            Some(previous) => Tensor::cat(&[previous, x], 2)?,
            None => pad1d_constant(x, context, 0)?,
        };
        let output = self.conv.forward(&joined)?;
        // copy releases the larger temporary allocation after each step.
        *history = Some(
            joined
                .narrow(2, joined.dim(2)? - context, context)?
                .copy()?,
        );
        Ok(output)
    }
}

struct SConvTranspose1d {
    convtr: ConvTranspose1d,
    causal: bool,
    kernel_size: usize,
    stride: usize,
}

impl SConvTranspose1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let convtr = ConvTranspose1d::load(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            0,
            1,
            bias,
            vb.pp("convtr").pp("convtr"),
        )?;
        Ok(Self {
            convtr,
            causal,
            kernel_size,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padding_total = self.kernel_size - self.stride;
        let mut y = self.convtr.forward(x)?;
        if padding_total == 0 {
            return Ok(y);
        }

        let (padding_left, padding_right) = if self.causal {
            (0, padding_total)
        } else {
            let right = padding_total / 2;
            (padding_total - right, right)
        };
        y = unpad1d(&y, padding_left, padding_right)?;
        Ok(y)
    }

    fn forward_stream(&self, x: &Tensor, history: &mut Option<Tensor>) -> Result<Tensor> {
        if !self.causal {
            candle_core::bail!("Incremental tokenization requires causal convolutions")
        }
        // Re-evaluate only the input frames whose kernels overlap the new
        // output. This also applies the bias exactly once at chunk boundaries.
        let previous_len = history.as_ref().map_or(0, |value| value.dims()[2]);
        let joined = match history.as_ref() {
            Some(previous) => Tensor::cat(&[previous, x], 2)?,
            None => x.clone(),
        };
        let output = self.forward(&joined)?.narrow(
            2,
            previous_len * self.stride,
            x.dim(2)? * self.stride,
        )?;
        let context = (self.kernel_size - self.stride).div_ceil(self.stride);
        let keep = context.min(joined.dim(2)?);
        *history = Some(joined.narrow(2, joined.dim(2)? - keep, keep)?.copy()?);
        Ok(output)
    }
}

struct Ffn {
    linear1: Linear,
    linear2: Linear,
}

impl Ffn {
    fn load(dim: usize, expansion: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let linear1 = if bias {
            candle_nn::linear(dim, expansion * dim, vb.pp("linear1"))?
        } else {
            candle_nn::linear_no_bias(dim, expansion * dim, vb.pp("linear1"))?
        };
        let linear2 = if bias {
            candle_nn::linear(expansion * dim, dim, vb.pp("linear2"))?
        } else {
            candle_nn::linear_no_bias(expansion * dim, dim, vb.pp("linear2"))?
        };
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = candle_nn::Activation::Gelu.forward(&x)?;
        self.linear2.forward(&x)
    }
}

struct Block1d {
    norm: ConvRmsNorm,
    mixer: SConv1d,
    gamma: Option<Tensor>,
    ffn_norm: ConvRmsNorm,
    ffn: Ffn,
    ffn_gamma: Option<Tensor>,
}

impl Block1d {
    fn load(dim: usize, config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let norm = ConvRmsNorm::load(
            dim,
            config.layernorm_eps,
            config.layernorm_elementwise_affine,
            vb.pp("norm"),
        )?;
        let mixer = SConv1d::load(
            dim,
            dim,
            7,
            1,
            1,
            dim,
            config.conv_bias,
            config.causal,
            vb.pp("mixer").pp("conv"),
        )?;
        let gamma = vb.get(dim, "gamma").ok();
        let ffn_norm = ConvRmsNorm::load(
            dim,
            config.layernorm_eps,
            config.layernorm_elementwise_affine,
            vb.pp("ffn_norm"),
        )?;
        let ffn = Ffn::load(dim, 4, config.conv_bias, vb.pp("ffn"))?;
        let ffn_gamma = vb.get(dim, "ffn_gamma").ok();

        Ok(Self {
            norm,
            mixer,
            gamma,
            ffn_norm,
            ffn,
            ffn_gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let mut h = self.norm.forward(x)?;
        h = self.mixer.forward(&h)?;
        self.finish_forward(residual, h)
    }

    fn forward_stream(&self, x: &Tensor, history: &mut Option<Tensor>) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let h = self.mixer.forward_stream(&h, history)?;
        self.finish_forward(x, h)
    }

    fn finish_forward(&self, residual: &Tensor, mut h: Tensor) -> Result<Tensor> {
        if let Some(gamma) = &self.gamma {
            h = h.broadcast_mul(&gamma.unsqueeze(0)?.unsqueeze(2)?)?;
        }
        let h = residual.broadcast_add(&h)?;

        let residual = &h;
        let mut out = self.ffn_norm.forward(&h)?;
        out = out.transpose(1, 2)?;
        out = self.ffn.forward(&out)?;
        out = out.transpose(1, 2)?;
        if let Some(gamma) = &self.ffn_gamma {
            out = out.broadcast_mul(&gamma.unsqueeze(0)?.unsqueeze(2)?)?;
        }
        residual.broadcast_add(&out)
    }
}

struct TokenizerEncoder {
    downsample_layers: Vec<SConv1d>,
    stages: Vec<Vec<Block1d>>,
    norm: Option<ConvRmsNorm>,
    head: SConv1d,
}

impl TokenizerEncoder {
    fn load(config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let depths = config.encoder_depths();
        let mut ratios = config.encoder_ratios.clone();
        ratios.reverse();

        let mut downsample_layers = Vec::with_capacity(depths.len());
        downsample_layers.push(SConv1d::load(
            config.channels,
            config.encoder_n_filters,
            7,
            1,
            1,
            1,
            config.conv_bias,
            config.causal,
            vb.pp("downsample_layers.0.0"),
        )?);

        for (index, ratio) in ratios.iter().enumerate() {
            let in_ch = config.encoder_n_filters * (1usize << index);
            let out_ch = config.encoder_n_filters * (1usize << (index + 1));
            downsample_layers.push(SConv1d::load(
                in_ch,
                out_ch,
                ratio * 2,
                *ratio,
                1,
                1,
                config.conv_bias,
                config.causal,
                vb.pp(format!("downsample_layers.{}.0", index + 1)),
            )?);
        }

        let mut stages = Vec::with_capacity(depths.len());
        for (stage_index, depth) in depths.iter().enumerate() {
            let dim = config.encoder_n_filters * (1usize << stage_index);
            let mut blocks = Vec::with_capacity(*depth);
            for block_index in 0..*depth {
                blocks.push(Block1d::load(
                    dim,
                    config,
                    vb.pp(format!("stages.{}.{}", stage_index, block_index)),
                )?);
            }
            stages.push(blocks);
        }

        let final_dim = config.encoder_n_filters * (1usize << (depths.len() - 1));
        let norm = if config.disable_last_norm {
            None
        } else {
            Some(ConvRmsNorm::load(
                final_dim,
                config.layernorm_eps,
                config.layernorm_elementwise_affine,
                vb.pp("norm"),
            )?)
        };
        let head = SConv1d::load(
            final_dim,
            config.vae_dim,
            7,
            1,
            1,
            1,
            config.conv_bias,
            config.causal,
            vb.pp("head"),
        )?;

        Ok(Self {
            downsample_layers,
            stages,
            norm,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (downsample, stage) in self.downsample_layers.iter().zip(self.stages.iter()) {
            h = downsample.forward(&h)?;
            for block in stage {
                h = block.forward(&h)?;
            }
        }
        if let Some(norm) = &self.norm {
            h = norm.forward(&h)?;
        }
        self.head.forward(&h)
    }

    fn forward_stream(&self, x: &Tensor, state: &mut TokenizerStreamState) -> Result<Tensor> {
        let mut index = 0;
        let mut h = x.clone();
        for (downsample, stage) in self.downsample_layers.iter().zip(&self.stages) {
            h = downsample.forward_stream(&h, state.next(&mut index))?;
            for block in stage {
                h = block.forward_stream(&h, state.next(&mut index))?;
            }
        }
        if let Some(norm) = &self.norm {
            h = norm.forward(&h)?;
        }
        self.head.forward_stream(&h, state.next(&mut index))
    }
}

struct DecoderLayer {
    kind: DecoderLayerKind,
}

enum DecoderLayerKind {
    Conv(SConv1d),
    ConvTranspose(SConvTranspose1d),
}

impl DecoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.kind {
            DecoderLayerKind::Conv(layer) => layer.forward(x),
            DecoderLayerKind::ConvTranspose(layer) => layer.forward(x),
        }
    }

    fn forward_stream(&self, x: &Tensor, history: &mut Option<Tensor>) -> Result<Tensor> {
        match &self.kind {
            DecoderLayerKind::Conv(layer) => layer.forward_stream(x, history),
            DecoderLayerKind::ConvTranspose(layer) => layer.forward_stream(x, history),
        }
    }
}

struct TokenizerDecoder {
    upsample_layers: Vec<DecoderLayer>,
    stages: Vec<Vec<Block1d>>,
    norm: Option<ConvRmsNorm>,
    head: SConv1d,
}

impl TokenizerDecoder {
    fn load(config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let depths = config.decoder_depths();
        let ratios = config.decoder_ratios();
        let top_dim = config.decoder_n_filters * (1usize << (depths.len() - 1));

        let mut upsample_layers = Vec::with_capacity(depths.len());
        upsample_layers.push(DecoderLayer {
            kind: DecoderLayerKind::Conv(SConv1d::load(
                config.vae_dim,
                top_dim,
                7,
                1,
                1,
                1,
                config.conv_bias,
                config.causal,
                vb.pp("upsample_layers.0.0"),
            )?),
        });

        for (index, ratio) in ratios.iter().enumerate() {
            let in_ch = config.decoder_n_filters * (1usize << (depths.len() - 1 - index));
            let out_ch = config.decoder_n_filters * (1usize << (depths.len() - 2 - index));
            upsample_layers.push(DecoderLayer {
                kind: DecoderLayerKind::ConvTranspose(SConvTranspose1d::load(
                    in_ch,
                    out_ch,
                    ratio * 2,
                    *ratio,
                    config.conv_bias,
                    config.causal,
                    vb.pp(format!("upsample_layers.{}.0", index + 1)),
                )?),
            });
        }

        let mut stages = Vec::with_capacity(depths.len());
        for (stage_index, depth) in depths.iter().enumerate() {
            let dim = config.decoder_n_filters * (1usize << (depths.len() - 1 - stage_index));
            let mut blocks = Vec::with_capacity(*depth);
            for block_index in 0..*depth {
                blocks.push(Block1d::load(
                    dim,
                    config,
                    vb.pp(format!("stages.{}.{}", stage_index, block_index)),
                )?);
            }
            stages.push(blocks);
        }

        let final_dim = config.decoder_n_filters;
        let norm = if config.disable_last_norm {
            None
        } else {
            Some(ConvRmsNorm::load(
                final_dim,
                config.layernorm_eps,
                config.layernorm_elementwise_affine,
                vb.pp("norm"),
            )?)
        };
        let head = SConv1d::load(
            final_dim,
            config.channels,
            7,
            1,
            1,
            1,
            config.conv_bias,
            config.causal,
            vb.pp("head"),
        )?;

        Ok(Self {
            upsample_layers,
            stages,
            norm,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (upsample, stage) in self.upsample_layers.iter().zip(self.stages.iter()) {
            h = upsample.forward(&h)?;
            for block in stage {
                h = block.forward(&h)?;
            }
        }
        if let Some(norm) = &self.norm {
            h = norm.forward(&h)?;
        }
        self.head.forward(&h)
    }

    fn forward_stream(&self, x: &Tensor, state: &mut TokenizerStreamState) -> Result<Tensor> {
        let mut index = 0;
        let mut h = x.clone();
        for (upsample, stage) in self.upsample_layers.iter().zip(&self.stages) {
            h = upsample.forward_stream(&h, state.next(&mut index))?;
            for block in stage {
                h = block.forward_stream(&h, state.next(&mut index))?;
            }
        }
        if let Some(norm) = &self.norm {
            h = norm.forward(&h)?;
        }
        self.head.forward_stream(&h, state.next(&mut index))
    }
}

#[derive(Clone)]
pub struct VibeVoiceTokenizerEncoderOutput {
    pub mean: Tensor,
    pub std: Option<f64>,
}

pub struct VibeVoiceAcousticTokenizer {
    config: VibeVoiceTokenizerConfig,
    encoder: Option<TokenizerEncoder>,
    decoder: TokenizerDecoder,
}

impl VibeVoiceAcousticTokenizer {
    pub fn load(config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = TokenizerEncoder::load(config, vb.pp("encoder"))?;
        let decoder = TokenizerDecoder::load(config, vb.pp("decoder"))?;
        Ok(Self {
            config: config.clone(),
            encoder: Some(encoder),
            decoder,
        })
    }

    pub fn load_decoder_only(config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let decoder = TokenizerDecoder::load(config, vb.pp("decoder"))?;
        Ok(Self {
            config: config.clone(),
            encoder: None,
            decoder,
        })
    }

    pub fn config(&self) -> &VibeVoiceTokenizerConfig {
        &self.config
    }

    pub fn encode(&self, audio: &Tensor) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Tokenizer encoder weights are not loaded".to_string())
        })?;
        let latents = encoder.forward(audio)?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: Some(self.config.fix_std),
        })
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = if latents.dims().len() == 3 && latents.dim(1)? != self.config.vae_dim {
            latents.transpose(1, 2)?
        } else {
            latents.clone()
        };
        self.decoder.forward(&latents)
    }

    /// Decode new latents in explicit [batch, frames, latent_dim] layout.
    pub(crate) fn decode_stream(
        &self,
        latents: &Tensor,
        state: &mut TokenizerStreamState,
    ) -> Result<Tensor> {
        if !self.config.causal {
            candle_core::bail!("Incremental acoustic decoding requires a causal tokenizer")
        }
        self.decoder
            .forward_stream(&latents.transpose(1, 2)?, state)
    }
}

pub struct VibeVoiceSemanticTokenizer {
    config: VibeVoiceTokenizerConfig,
    encoder: TokenizerEncoder,
}

impl VibeVoiceSemanticTokenizer {
    pub fn load(config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = TokenizerEncoder::load(config, vb.pp("encoder"))?;
        Ok(Self {
            config: config.clone(),
            encoder,
        })
    }

    pub fn config(&self) -> &VibeVoiceTokenizerConfig {
        &self.config
    }

    pub fn encode(&self, audio: &Tensor) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let latents = self.encoder.forward(audio)?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: None,
        })
    }

    pub(crate) fn encode_stream(
        &self,
        audio: &Tensor,
        state: &mut TokenizerStreamState,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let hop = self.config.encoder_ratios.iter().product::<usize>();
        if !self.config.causal || hop == 0 || audio.dim(2)? == 0 || audio.dim(2)? % hop != 0 {
            candle_core::bail!("Incremental semantic encoding requires causal, hop-aligned audio")
        }
        let latents = self.encoder.forward_stream(audio, state)?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: None,
        })
    }
}

fn extra_padding_for_conv1d(
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let numerator = input_len as isize - kernel_size as isize + padding_total as isize;
    let frames = numerator as f64 / stride as f64 + 1.0;
    let ideal_len = ((frames.ceil() as usize).saturating_sub(1)) * stride
        + (kernel_size.saturating_sub(padding_total));
    ideal_len.saturating_sub(input_len)
}

fn pad1d_constant(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }

    let (batch, channels, _) = x.dims3()?;
    let mut parts = Vec::new();
    if left > 0 {
        parts.push(Tensor::zeros(
            (batch, channels, left),
            x.dtype(),
            x.device(),
        )?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(Tensor::zeros(
            (batch, channels, right),
            x.dtype(),
            x.device(),
        )?);
    }
    let parts_ref = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&parts_ref, 2)
}

fn unpad1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let total = x.dim(2)?;
    let keep = total.saturating_sub(left + right);
    x.narrow(2, left, keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape};
    use candle_nn::var_builder::SimpleBackend;

    struct SyntheticWeights;

    impl SimpleBackend for SyntheticWeights {
        fn get(
            &self,
            shape: Shape,
            name: &str,
            _hint: candle_nn::Init,
            dtype: DType,
            device: &Device,
        ) -> Result<Tensor> {
            let norm = name.ends_with("norm.weight") || name.ends_with("gamma");
            let salt = name
                .bytes()
                .fold(0usize, |acc, byte| acc.wrapping_add(byte as usize));
            let values: Vec<f32> = (0..shape.elem_count())
                .map(|index| {
                    if norm {
                        1.0
                    } else {
                        ((index * 7 + salt) % 29) as f32 / 100.0 - 0.14
                    }
                })
                .collect();
            Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
        }

        fn get_unchecked(&self, _name: &str, _dtype: DType, _device: &Device) -> Result<Tensor> {
            candle_core::bail!("Synthetic weights require a shape")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn config(bias: bool) -> VibeVoiceTokenizerConfig {
        VibeVoiceTokenizerConfig {
            vae_dim: 3,
            channels: 1,
            causal: true,
            encoder_n_filters: 2,
            decoder_n_filters: 2,
            encoder_ratios: vec![2, 3],
            encoder_depths: "1-1-1".to_string(),
            conv_bias: bias,
            disable_last_norm: false,
            ..Default::default()
        }
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.dims(), expected.dims());
        let error = actual
            .sub(expected)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 2e-5, "maximum codec error: {error}");
    }

    #[test]
    fn incremental_acoustic_and_semantic_codecs_match_every_full_prefix() {
        for bias in [false, true] {
            let device = Device::Cpu;
            let config = config(bias);
            let vb =
                VarBuilder::from_backend(Box::new(SyntheticWeights), DType::F32, device.clone());
            let acoustic =
                VibeVoiceAcousticTokenizer::load_decoder_only(&config, vb.pp("acoustic")).unwrap();
            let semantic = VibeVoiceSemanticTokenizer::load(&config, vb.pp("semantic")).unwrap();
            let latents = Tensor::from_vec(
                (0..27)
                    .map(|index| (index as f32 * 0.31).sin())
                    .collect::<Vec<_>>(),
                (1, 9, 3),
                &device,
            )
            .unwrap();
            let mut decoder_state = TokenizerStreamState::default();
            let mut encoder_state = TokenizerStreamState::default();
            let mut audio_chunks = Vec::new();
            let mut semantic_chunks = Vec::new();
            let mut position = 0;
            let mut retained = None;
            for chunk_len in [1, 2, 1, 3, 2] {
                let audio = acoustic
                    .decode_stream(
                        &latents.narrow(1, position, chunk_len).unwrap(),
                        &mut decoder_state,
                    )
                    .unwrap();
                semantic_chunks.push(
                    semantic
                        .encode_stream(&audio, &mut encoder_state)
                        .unwrap()
                        .mean,
                );
                audio_chunks.push(audio);
                position += chunk_len;
                let prefix_audio = acoustic
                    .decode(
                        &latents
                            .narrow(1, 0, position)
                            .unwrap()
                            .transpose(1, 2)
                            .unwrap(),
                    )
                    .unwrap();
                assert_close(&Tensor::cat(&audio_chunks, 2).unwrap(), &prefix_audio);
                assert_close(
                    &Tensor::cat(&semantic_chunks, 1).unwrap(),
                    &semantic.encode(&prefix_audio).unwrap().mean,
                );
                let buffer_size: usize = decoder_state
                    .buffers
                    .iter()
                    .chain(&encoder_state.buffers)
                    .filter_map(Option::as_ref)
                    .map(Tensor::elem_count)
                    .sum();
                if let Some(previous) = retained {
                    assert_eq!(buffer_size, previous, "stream state must remain bounded");
                }
                retained = Some(buffer_size);
            }
            let energy: f32 = Tensor::cat(&audio_chunks, 2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|value| value.abs())
                .sum();
            assert!(energy > 0.01, "parity test must exercise nonzero audio");
        }
    }

    #[test]
    fn incremental_semantic_encoder_rejects_partial_hops_before_changing_state() {
        let device = Device::Cpu;
        let vb = VarBuilder::from_backend(Box::new(SyntheticWeights), DType::F32, device.clone());
        let semantic = VibeVoiceSemanticTokenizer::load(&config(true), vb).unwrap();
        let mut state = TokenizerStreamState::default();
        let audio = Tensor::zeros((1, 1, 5), DType::F32, &device).unwrap();
        assert!(semantic.encode_stream(&audio, &mut state).is_err());
        assert!(state.buffers.is_empty());
    }
}
