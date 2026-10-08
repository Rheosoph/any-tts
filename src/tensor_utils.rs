//! Shared tensor utilities for model implementations.

use candle_core::{DType, Device, Result, Tensor};

/// RMS Layer Normalization.
///
/// Used by Qwen-family backbones and talkers.
/// Formula: `x * rsqrt(mean(x^2) + eps) * weight`
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Create a new RmsNorm layer.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Load from a `candle_nn::VarBuilder` with the given hidden size.
    pub fn load(hidden_size: usize, eps: f64, vb: candle_nn::VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self::new(weight, eps))
    }

    /// Apply RMS normalization.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(candle_core::DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let result = x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?;
        Ok(result)
    }
}

/// Apply Rotary Position Embeddings (RoPE) to query and key tensors.
///
/// This is the standard implementation used by Qwen/Llama-family models.
pub fn apply_rotary_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_embed = apply_rotary_emb_one(q, cos, sin)?;
    let k_embed = apply_rotary_emb_one(k, cos, sin)?;
    Ok((q_embed, k_embed))
}

fn apply_rotary_emb_one(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_batch, _heads, _seq_len, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;
    let result = (x.broadcast_mul(cos))?.add(&rotated.broadcast_mul(sin)?)?;
    Ok(result)
}

/// Precompute RoPE frequency tensors for a given sequence length.
///
/// The `target_dtype` parameter controls the output dtype. RoPE is always
/// computed in F32 for precision, then cast to the target dtype so that the
/// cos/sin tensors match the model's weight dtype (e.g. BF16 on GPU).
pub fn precompute_rope_freqs(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &Device,
    target_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(i as f64 / half_dim as f64) as f32)
        .collect();
    let freqs = Tensor::new(freqs.as_slice(), device)?;

    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let positions = Tensor::new(positions.as_slice(), device)?;

    // (seq_len, half_dim) outer product
    let angles = positions
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;

    let cos = angles.cos()?;
    let sin = angles.sin()?;

    // Duplicate for full head_dim: (seq_len, head_dim)
    let cos = Tensor::cat(&[&cos, &cos], candle_core::D::Minus1)?;
    let sin = Tensor::cat(&[&sin, &sin], candle_core::D::Minus1)?;

    // Cast to target dtype so RoPE matches model weight dtype (e.g. BF16).
    let cos = cos.to_dtype(target_dtype)?;
    let sin = sin.to_dtype(target_dtype)?;

    Ok((cos, sin))
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    x.mul(&sigmoid)
}

/// Run Candle's CPU fused attention helper for standard heads-first tensors.
///
/// Inputs must have shape `(batch, heads, seq, head_dim)` for `q`, `k`, and `v`.
/// The helper is only used for F32 CPU tensors in known-good attention
/// regimes; other cases fall back to the caller's existing attention
/// implementation.
fn cpu_flash_attention_shape_eligible(q_heads: usize, k_heads: usize, q_seq: usize) -> bool {
    q_heads == k_heads || q_seq == 1
}

pub fn cpu_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Option<Tensor>> {
    let (batch_size, q_heads, q_seq, _head_dim) = q.dims4()?;
    let (_k_batch, k_heads, k_seq, _k_head_dim) = k.dims4()?;

    if !matches!(q.device(), Device::Cpu)
        || !matches!(k.device(), Device::Cpu)
        || !matches!(v.device(), Device::Cpu)
    {
        return Ok(None);
    }

    if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        return Ok(None);
    }

    // Single-token incremental decode is a useful fast path even for grouped
    // attention, but full-sequence grouped-query attention fell back better in
    // OmniVoice-style runs. Keep the CPU fused path conservative.
    if !cpu_flash_attention_shape_eligible(q_heads, k_heads, q_seq) {
        return Ok(None);
    }

    if let Some(mask) = mask {
        if !matches!(mask.device(), Device::Cpu) || mask.dtype() != DType::F32 {
            return Ok(None);
        }

        let supported_mask = match mask.dims() {
            [mask_batch, mask_heads, mask_q_seq, mask_k_seq] => {
                *mask_batch == batch_size
                    && *mask_heads == 1
                    && *mask_q_seq == q_seq
                    && *mask_k_seq == k_seq
            }
            [mask_batch, mask_q_seq, mask_k_seq] => {
                *mask_batch == batch_size && *mask_q_seq == q_seq && *mask_k_seq == k_seq
            }
            _ => false,
        };

        if !supported_mask {
            return Ok(None);
        }
    }

    let q = q.transpose(1, 2)?.contiguous()?;
    // Candle's CPU kernel reads KV strides directly. Capacity-backed cache
    // prefixes need not be copied into a freshly packed allocation each step.
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;
    let attn = candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>(
        &q, &k, &v, mask, scale, None, None,
    )?;
    Ok(Some(attn))
}

/// SnakeBeta activation: x + (1/(exp(beta) + eps)) * sin²(exp(alpha) * x)
///
/// `alpha` and `beta` are the learned parameters stored in **log-space**.
/// They must be exponentiated before use (matching the Python reference).
///
/// Used in the Qwen3-TTS speech tokenizer decoder (SNAC-style vocoder).
pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let alpha = alpha.exp()?;
    let beta = beta.exp()?;
    let ax = x.broadcast_mul(&alpha)?;
    let sin_sq = ax.sin()?.sqr()?;
    let inv_beta = (&beta + 1e-9)?.recip()?;
    let correction = sin_sq.broadcast_mul(&inv_beta)?;
    x.add(&correction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_attention_accepts_strided_multi_batch_cache() -> Result<()> {
        let device = Device::Cpu;
        for q_len in [1, 3] {
            let q = Tensor::arange(0f32, (2 * 2 * q_len * 8) as f32, &device)?
                .reshape((2, 2, q_len, 8))?
                .affine(0.01, -0.2)?;
            let storage = Tensor::arange(0f32, (2 * 16 * 2 * 8) as f32, &device)?
                .reshape((2, 16, 2, 8))?
                .affine(0.003, -0.1)?;
            let k = storage.narrow(1, 0, 5)?.transpose(1, 2)?;
            let v = storage.affine(0.3, 0.2)?.narrow(1, 0, 5)?.transpose(1, 2)?;
            let actual = cpu_flash_attention(&q, &k, &v, None, 0.3535534)?.unwrap();
            let weights = q
                .matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(0.3535534, 0.)?;
            let expected = candle_nn::ops::softmax_last_dim(&weights)?.matmul(&v.contiguous()?)?;
            let error = (&actual - &expected)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error < 1e-5, "strided attention error {error}");
        }
        Ok(())
    }

    #[test]
    fn test_cpu_flash_attention_shape_policy() {
        assert!(cpu_flash_attention_shape_eligible(16, 16, 128));
        assert!(cpu_flash_attention_shape_eligible(16, 8, 1));
        assert!(!cpu_flash_attention_shape_eligible(16, 8, 128));
    }

    #[test]
    fn test_rms_norm_shape() {
        let device = Device::Cpu;
        let weight = Tensor::ones(64, candle_core::DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::randn(0f32, 1.0, (2, 8, 64), &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8, 64]);
    }

    #[test]
    fn test_silu_basic() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &device).unwrap();
        let out = silu(&x).unwrap();
        let values: Vec<f32> = out.to_vec1().unwrap();
        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        assert!((values[0]).abs() < 1e-6);
        assert!((values[1] - 0.7311).abs() < 1e-3);
        assert!((values[2] + 0.2689).abs() < 1e-3);
    }

    #[test]
    fn test_precompute_rope_freqs() {
        let device = Device::Cpu;
        let (cos, sin) = precompute_rope_freqs(64, 128, 10000.0, &device, DType::F32).unwrap();
        assert_eq!(cos.dims(), &[128, 64]);
        assert_eq!(sin.dims(), &[128, 64]);
    }
}
