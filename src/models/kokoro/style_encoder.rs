//! Style encoder for voice cloning from reference audio.
//!
//! Implements the StyleTTS2 style encoder architecture — a 2D CNN that
//! extracts a fixed-size style vector from a log-mel spectrogram.
//!
//! Kokoro uses **two** such encoders (same architecture, different weights):
//!
//! - `style_encoder` → 128-dim **acoustic** style (conditions the decoder)
//! - `predictor_encoder` → 128-dim **prosodic** style (conditions duration/F0/noise)
//!
//! The concatenation `[acoustic, prosodic]` = 256-dim is equivalent to a single
//! row of the voice pack `.pt` files.
//!
//! ## Weight availability
//!
//! The style encoders are part of the full Kokoro model but may not be
//! present in all checkpoint files. If the weights are missing, voice
//! cloning falls back to requiring pre-computed voice packs.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use tracing::{info, warn};

use crate::error::{TtsError, TtsResult};
use crate::mel::{MelConfig, MelSpectrogram};
use crate::traits::ReferenceAudio;

fn scalar_like(tensor: &Tensor, value: f32) -> candle_core::Result<Tensor> {
    Tensor::new(value, tensor.device())?.to_dtype(tensor.dtype())
}

fn scale_tensor(tensor: &Tensor, value: f32) -> candle_core::Result<Tensor> {
    tensor.broadcast_mul(&scalar_like(tensor, value)?)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Leaky ReLU with configurable negative slope.
fn leaky_relu(x: &Tensor, negative_slope: f32) -> candle_core::Result<Tensor> {
    // leaky_relu(x) = relu(x) + α · (x − relu(x))
    let relu_x = x.relu()?;
    let neg_part = (x - &relu_x)?.broadcast_mul(&scalar_like(x, negative_slope)?)?;
    &relu_x + &neg_part
}

/// Average pooling 2×2 with stride 2.
///
/// `x`: `[B, C, H, W]` → `[B, C, H/2, W/2]`
fn avg_pool2d_2x2(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let oh = h / 2;
    let ow = w / 2;
    // Reshape height: [B, C, H/2, 2, W] → mean over dim 3
    let x = x.reshape((b, c, oh, 2, w))?.mean(3)?;
    // Reshape width: [B, C, H/2, W/2, 2] → mean over dim 4
    x.reshape((b, c, oh, ow, 2))?.mean(4)
}

/// Load a Conv2d, trying `weight` first then falling back to `weight_orig`
/// (spectral-normalised checkpoints store the un-normalised weight as `weight_orig`).
fn load_conv2d(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: VarBuilder,
) -> candle_core::Result<candle_nn::Conv2d> {
    let shape = (out_c, in_c / cfg.groups, kernel, kernel);

    let ws = vb
        .get(shape, "weight")
        .or_else(|_| vb.get(shape, "weight_orig"))?;

    let bs = vb.get(out_c, "bias").ok();

    Ok(candle_nn::Conv2d::new(ws, bs, cfg))
}

/// Conv2dConfig with padding=1, stride=1 (3×3 same-padding).
fn cfg_pad1() -> candle_nn::Conv2dConfig {
    candle_nn::Conv2dConfig {
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
        ..Default::default()
    }
}

/// Conv2dConfig with stride=2, padding=1 (learned 2× downsample).
fn cfg_stride2() -> candle_nn::Conv2dConfig {
    candle_nn::Conv2dConfig {
        padding: 1,
        stride: 2,
        dilation: 1,
        groups: 1,
        ..Default::default()
    }
}

/// Conv2dConfig with no padding (5×5 final conv).
fn cfg_no_pad() -> candle_nn::Conv2dConfig {
    candle_nn::Conv2dConfig {
        padding: 0,
        stride: 1,
        dilation: 1,
        groups: 1,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// ResBlk2d
// ---------------------------------------------------------------------------

/// 2D residual block with optional downsampling (StyleTTS2 ResBlk).
///
/// ```text
/// residual: LeakyReLU → Conv(in,in,3×3) → LearnedDown → LeakyReLU → Conv(in,out,3×3)
/// shortcut: [Conv1×1 if in≠out] → AvgPool2×2
/// output:   (shortcut + residual) / √2
/// ```
struct ResBlk2d {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    downsample_conv: candle_nn::Conv2d,
    conv1x1: Option<candle_nn::Conv2d>,
}

impl ResBlk2d {
    fn load(dim_in: usize, dim_out: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv1 = load_conv2d(dim_in, dim_in, 3, cfg_pad1(), vb.pp("conv1"))?;
        let conv2 = load_conv2d(dim_in, dim_out, 3, cfg_pad1(), vb.pp("conv2"))?;
        let downsample_conv = load_conv2d(
            dim_in,
            dim_in,
            3,
            cfg_stride2(),
            vb.pp("downsample_res").pp("conv"),
        )?;

        let conv1x1 = if dim_in != dim_out {
            Some(load_conv2d(
                dim_in,
                dim_out,
                1,
                cfg_no_pad(),
                vb.pp("conv1x1"),
            )?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            conv2,
            downsample_conv,
            conv1x1,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Shortcut
        let shortcut = match &self.conv1x1 {
            Some(c) => avg_pool2d_2x2(&c.forward(x)?)?,
            None => avg_pool2d_2x2(x)?,
        };

        // Residual
        let h = leaky_relu(x, 0.2)?;
        let h = self.conv1.forward(&h)?;
        let h = self.downsample_conv.forward(&h)?;
        let h = leaky_relu(&h, 0.2)?;
        let h = self.conv2.forward(&h)?;

        // Combine
        let sum: Tensor = (shortcut + h)?;
        scale_tensor(&sum, std::f32::consts::FRAC_1_SQRT_2)
    }
}

// ---------------------------------------------------------------------------
// StyleEncoder
// ---------------------------------------------------------------------------

/// StyleTTS2 style encoder: mel spectrogram → fixed-size style vector.
///
/// Architecture:
/// ```text
/// Conv2d(1, 64, 3×3)
/// → ResBlk(64, 128, ↓2) → ResBlk(128, 256, ↓2) → ResBlk(256, 512, ↓2) → ResBlk(512, 512, ↓2)
/// → LeakyReLU → Conv2d(512, 512, 5×5) → AdaptiveAvgPool(1) → LeakyReLU
/// → Linear(512, style_dim)
/// ```
struct SingleStyleEncoder {
    initial_conv: candle_nn::Conv2d,
    res_blocks: Vec<ResBlk2d>,
    final_conv: candle_nn::Conv2d,
    fc: candle_nn::Linear,
}

impl SingleStyleEncoder {
    fn load(
        dim_in: usize,
        style_dim: usize,
        max_conv_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let vb_shared = vb.pp("shared");

        // Initial conv: Conv2d(1, dim_in, 3, 1, 1)
        let initial_conv = load_conv2d(1, dim_in, 3, cfg_pad1(), vb_shared.pp("0"))?;

        // 4 residual blocks with 2× downsampling
        let mut cur_dim = dim_in;
        let mut res_blocks = Vec::with_capacity(4);
        for i in 0..4u32 {
            let next_dim = (cur_dim * 2).min(max_conv_dim);
            let blk = ResBlk2d::load(cur_dim, next_dim, vb_shared.pp((i + 1).to_string()))?;
            res_blocks.push(blk);
            cur_dim = next_dim;
        }

        // Final conv: Conv2d(512, 512, 5, 1, 0) — index 6 in Sequential
        // (index 5 is LeakyReLU, no weights)
        let final_conv = load_conv2d(cur_dim, cur_dim, 5, cfg_no_pad(), vb_shared.pp("6"))?;

        // Linear projection: index in Python is `unshared` (not in Sequential)
        let fc = candle_nn::linear(cur_dim, style_dim, vb.pp("unshared"))?;

        Ok(Self {
            initial_conv,
            res_blocks,
            final_conv,
            fc,
        })
    }

    /// Forward pass.
    ///
    /// Input: `[B, 1, n_mels, time]` log-mel spectrogram
    /// Output: `[B, style_dim]` style vector
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = self.initial_conv.forward(x)?;

        for blk in &self.res_blocks {
            h = blk.forward(&h)?;
        }

        h = leaky_relu(&h, 0.2)?;
        h = self.final_conv.forward(&h)?;

        // Adaptive average pool to (1, 1) → flatten
        h = h.mean_keepdim(candle_core::D::Minus1)?;
        h = h.mean_keepdim(candle_core::D::Minus2)?;
        h = h.flatten_from(1)?; // [B, 512]

        self.fc.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Combined style encoder for voice cloning.
///
/// Contains two [`SingleStyleEncoder`] instances:
/// - **acoustic** (`style_encoder.*`) → conditions the decoder
/// - **prosodic** (`predictor_encoder.*`) → conditions the prosody predictor
///
/// Both share the same architecture but have independently trained weights.
pub struct StyleEncoder {
    acoustic: SingleStyleEncoder,
    prosodic: SingleStyleEncoder,
    mel: MelSpectrogram,
    target_sample_rate: u32,
}

impl StyleEncoder {
    /// Attempt to load both style encoders from a VarBuilder.
    ///
    /// Returns `Ok(Some(...))` if weights are found, `Ok(None)` if the
    /// style encoder weights are not present in the checkpoint.
    pub fn try_load(
        dim_in: usize,
        style_dim: usize,
        max_conv_dim: usize,
        sample_rate: u32,
        vb: &VarBuilder,
        device: &Device,
    ) -> TtsResult<Option<Self>> {
        // Try loading the acoustic encoder first as a probe
        let acoustic =
            match SingleStyleEncoder::load(dim_in, style_dim, max_conv_dim, vb.pp("style_encoder"))
            {
                Ok(enc) => enc,
                Err(e) => {
                    warn!(
                        "Style encoder weights not found (voice cloning unavailable): {}",
                        e
                    );
                    return Ok(None);
                }
            };

        let prosodic =
            SingleStyleEncoder::load(dim_in, style_dim, max_conv_dim, vb.pp("predictor_encoder"))
                .map_err(|e| {
                TtsError::WeightLoadError(format!(
                    "Found style_encoder but not predictor_encoder: {}",
                    e
                ))
            })?;

        let mel = MelSpectrogram::new(MelConfig::kokoro(), device)?;

        info!("Style encoders loaded — voice cloning available");

        Ok(Some(Self {
            acoustic,
            prosodic,
            mel,
            target_sample_rate: sample_rate,
        }))
    }

    /// Extract a style vector from reference audio.
    ///
    /// Returns `[1, style_dim * 2]` — concatenation of `[acoustic, prosodic]`.
    pub fn encode(&self, audio: &ReferenceAudio, dtype: DType) -> TtsResult<Tensor> {
        // Resample if necessary
        if audio.sample_rate != self.target_sample_rate {
            info!(
                "Resampling reference audio from {} Hz to {} Hz",
                audio.sample_rate, self.target_sample_rate
            );
        }
        let samples =
            crate::mel::resample_audio(&audio.samples, audio.sample_rate, self.target_sample_rate)?;

        // The extractor runs its FFT on CPU and returns mel features on the
        // configured model device, including for Metal and CUDA encoders.
        let length = samples.len();
        let audio_tensor = Tensor::from_vec(samples, length, &Device::Cpu)?;

        // Compute mel spectrogram: [1, n_mels, time]
        let mel_spec = self.mel.compute(&audio_tensor)?;
        // Add channel dim: [1, 1, n_mels, time] for 2D conv
        let mel_input = mel_spec.unsqueeze(1)?.to_dtype(dtype)?;

        // Run both encoders
        let acoustic_style = self.acoustic.forward(&mel_input)?; // [1, style_dim]
        let prosodic_style = self.prosodic.forward(&mel_input)?; // [1, style_dim]

        // Concatenate: [1, style_dim * 2]
        // Note: Kokoro convention is [acoustic, prosodic] = [decoder_style, predictor_style]
        Tensor::cat(&[&acoustic_style, &prosodic_style], 1).map_err(TtsError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu() {
        let device = Device::Cpu;
        let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &device).unwrap();
        let y = leaky_relu(&x, 0.2).unwrap();
        let vals: Vec<f32> = y.to_vec1().unwrap();
        assert!((vals[0] - (-0.4)).abs() < 1e-5);
        assert!((vals[1] - (-0.2)).abs() < 1e-5);
        assert!((vals[2] - 0.0).abs() < 1e-5);
        assert!((vals[3] - 1.0).abs() < 1e-5);
        assert!((vals[4] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_avg_pool2d_2x2() {
        let device = Device::Cpu;
        // [1, 1, 4, 4]
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = Tensor::new(data.as_slice(), &device)
            .unwrap()
            .reshape((1, 1, 4, 4))
            .unwrap();
        let y = avg_pool2d_2x2(&x).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 2]);
        let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        // (1+2+5+6)/4 = 3.5
        assert!((vals[0] - 3.5).abs() < 1e-5);
        // (3+4+7+8)/4 = 5.5
        assert!((vals[1] - 5.5).abs() < 1e-5);
        // (9+10+13+14)/4 = 11.5
        assert!((vals[2] - 11.5).abs() < 1e-5);
        // (11+12+15+16)/4 = 13.5
        assert!((vals[3] - 13.5).abs() < 1e-5);
    }
}
