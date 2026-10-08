//! Appendable attention caches with immutable-prefix snapshots.
//!
//! Extending the newest snapshot writes only unused capacity. Extending an older
//! snapshot forks its prefix once, so alternate decoding branches cannot overwrite
//! each other. CPU storage is sequence-major to avoid repacking history for fused
//! attention; accelerator storage is heads-major.

use std::sync::{Arc, Mutex};

use candle_core::{Result, Tensor};

#[derive(Debug)]
struct Buffer {
    k: Tensor,
    v: Tensor,
    written: usize,
    sequence_major: bool,
}

/// A cheap snapshot of an immutable KV prefix. Clones can be extended independently.
#[derive(Debug, Clone)]
pub struct KvCacheState {
    buffer: Arc<Mutex<Buffer>>,
    len: usize,
}

impl KvCacheState {
    /// Import heads-first `[batch, heads, sequence, head_dim]` tensors.
    pub fn from_tensors(k: Tensor, v: Tensor) -> Result<Self> {
        validate_pair(&k, &v)?;
        Ok(Self::from_legacy(k, v))
    }

    fn from_legacy(k: Tensor, v: Tensor) -> Self {
        let len = k.dims().get(2).copied().unwrap_or(0);
        Self {
            buffer: Arc::new(Mutex::new(Buffer {
                k,
                v,
                written: len,
                sequence_major: false,
            })),
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return heads-first views of this prefix, without copying its storage.
    pub fn tensors(&self) -> Result<(Tensor, Tensor)> {
        let buffer = self.buffer.lock().map_err(poisoned)?;
        views(&buffer, self.len)
    }
}

/// A growable, branch-safe KV cache.
#[derive(Debug, Default)]
pub struct KvCache {
    state: Option<KvCacheState>,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.state.as_ref().map_or(0, KvCacheState::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.state = None;
    }

    pub fn snapshot(&self) -> Option<KvCacheState> {
        self.state.clone()
    }

    pub fn restore(&mut self, state: Option<KvCacheState>) {
        self.state = state;
    }

    /// Compatibility import for the original tensor-pair cache API.
    pub fn restore_tensors(&mut self, state: Option<(Tensor, Tensor)>) {
        self.state = state.map(|(k, v)| KvCacheState::from_legacy(k, v));
    }

    pub fn tensors(&self) -> Result<Option<(Tensor, Tensor)>> {
        self.state.as_ref().map(KvCacheState::tensors).transpose()
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        validate_pair(k, v)?;
        let added = k.dim(2)?;
        let old_len = self.len();
        let next_len = old_len
            .checked_add(added)
            .ok_or_else(|| candle_core::Error::Msg("KV cache sequence length overflow".into()))?;

        if let Some(state) = &self.state {
            let mut buffer = state.buffer.lock().map_err(poisoned)?;
            let (old_k, old_v) = views(&buffer, old_len)?;
            validate_append(&old_k, k)?;
            let sequence_dim = if buffer.sequence_major { 1 } else { 2 };
            // Existing snapshots expose only the prefix, so this append cannot
            // change their values. A fork from an older prefix must allocate.
            if old_len == buffer.written && next_len <= buffer.k.dim(sequence_dim)? {
                write(&buffer, k, v, old_len)?;
                buffer.written = next_len;
                let result = views(&buffer, next_len)?;
                drop(buffer);
                self.state.as_mut().unwrap().len = next_len;
                return Ok(result);
            }
            let new_buffer = allocate(k, next_len)?;
            write(&new_buffer, &old_k, &old_v, 0)?;
            write(&new_buffer, k, v, old_len)?;
            drop(buffer);
            self.state = Some(KvCacheState {
                buffer: Arc::new(Mutex::new(new_buffer)),
                len: next_len,
            });
        } else {
            let buffer = allocate(k, next_len)?;
            write(&buffer, k, v, 0)?;
            self.state = Some(KvCacheState {
                buffer: Arc::new(Mutex::new(buffer)),
                len: next_len,
            });
        }
        self.state.as_ref().unwrap().tensors()
    }
}

fn poisoned<T>(_: std::sync::PoisonError<T>) -> candle_core::Error {
    candle_core::Error::Msg("KV cache lock was poisoned".into())
}

fn validate_pair(k: &Tensor, v: &Tensor) -> Result<()> {
    k.dims4()?;
    v.dims4()?;
    if k.dims() != v.dims() || k.dtype() != v.dtype() || !k.device().same_device(v.device()) {
        candle_core::bail!("KV cache requires matching key/value shapes, dtypes and devices")
    }
    Ok(())
}

fn validate_append(previous: &Tensor, next: &Tensor) -> Result<()> {
    let (b, h, _, d) = previous.dims4()?;
    let (nb, nh, _, nd) = next.dims4()?;
    if (b, h, d) != (nb, nh, nd)
        || previous.dtype() != next.dtype()
        || !previous.device().same_device(next.device())
    {
        candle_core::bail!("KV cache append shape, dtype or device mismatch")
    }
    Ok(())
}

fn allocate(k: &Tensor, len: usize) -> Result<Buffer> {
    let (batch, heads, _, dim) = k.dims4()?;
    let capacity = len
        .max(32)
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::Msg("KV cache capacity overflow".into()))?;
    let sequence_major = k.device().is_cpu();
    let shape = if sequence_major {
        (batch, capacity, heads, dim)
    } else {
        (batch, heads, capacity, dim)
    };
    Ok(Buffer {
        k: Tensor::zeros(shape, k.dtype(), k.device())?,
        v: Tensor::zeros(shape, k.dtype(), k.device())?,
        written: len,
        sequence_major,
    })
}

fn write(buffer: &Buffer, k: &Tensor, v: &Tensor, offset: usize) -> Result<()> {
    if buffer.sequence_major {
        buffer
            .k
            .slice_set(&k.transpose(1, 2)?.contiguous()?, 1, offset)?;
        buffer
            .v
            .slice_set(&v.transpose(1, 2)?.contiguous()?, 1, offset)?;
    } else {
        buffer.k.slice_set(&k.contiguous()?, 2, offset)?;
        buffer.v.slice_set(&v.contiguous()?, 2, offset)?;
    }
    Ok(())
}

fn views(buffer: &Buffer, len: usize) -> Result<(Tensor, Tensor)> {
    if buffer.sequence_major {
        Ok((
            buffer.k.narrow(1, 0, len)?.transpose(1, 2)?,
            buffer.v.narrow(1, 0, len)?.transpose(1, 2)?,
        ))
    } else {
        Ok((buffer.k.narrow(2, 0, len)?, buffer.v.narrow(2, 0, len)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn chunk(start: usize, len: usize) -> Tensor {
        let values: Vec<f32> = (start..start + 4 * len).map(|n| n as f32).collect();
        Tensor::from_vec(values, (1, 2, len, 2), &Device::Cpu).unwrap()
    }

    fn values(x: &Tensor) -> Vec<f32> {
        x.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn append_matches_concatenation_across_capacity_growth() -> Result<()> {
        let mut cache = KvCache::new();
        let mut expected = chunk(0, 3);
        cache.append(&expected, &expected)?;
        for step in 0..70 {
            let next = chunk(100 + step * 4, 1);
            expected = Tensor::cat(&[&expected, &next], 2)?;
            let (k, v) = cache.append(&next, &next)?;
            assert_eq!(values(&k), values(&expected));
            assert_eq!(values(&v), values(&expected));
            // Fused CPU attention can consume this view without repacking K/V.
            assert!(k.transpose(1, 2)?.is_contiguous());
        }
        Ok(())
    }

    #[test]
    fn snapshots_preserve_prefixes_and_isolate_branches() -> Result<()> {
        let mut cache = KvCache::new();
        let prefix = chunk(0, 3);
        cache.append(&prefix, &prefix)?;
        let shared = cache.snapshot();
        let a = chunk(100, 1);
        cache.append(&a, &a)?;
        let branch_a = cache.snapshot().unwrap();
        cache.restore(shared.clone());
        let b = chunk(200, 2);
        let (bk, _) = cache.append(&b, &b)?;
        assert_eq!(values(&bk), values(&Tensor::cat(&[&prefix, &b], 2)?));
        assert_eq!(
            values(&branch_a.tensors()?.0),
            values(&Tensor::cat(&[&prefix, &a], 2)?)
        );
        assert_eq!(values(&shared.unwrap().tensors()?.0), values(&prefix));
        cache.restore(Some(branch_a));
        let c = chunk(300, 1);
        let (ak, _) = cache.append(&c, &c)?;
        assert_eq!(values(&ak), values(&Tensor::cat(&[&prefix, &a, &c], 2)?));
        cache.clear();
        assert!(cache.is_empty());
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_attention_accepts_capacity_strided_cache_views() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("Metal unavailable for KV parity test: {error}");
                return Ok(());
            }
        };
        // Exercise both the vector and two-pass Metal attention kernels.
        for len in [7, 1031] {
            let values: Vec<f32> = (0..2 * len * 32)
                .map(|i| (i as f32 * 0.003).sin())
                .collect();
            let k = Tensor::from_vec(values, (1, 2, len, 32), &device)?;
            let v = k.affine(0.3, 0.1)?;
            let mut cache = KvCache::new();
            let (k, v) = cache.append(&k, &v)?;
            assert!(!k.is_contiguous());
            let q = Tensor::ones((1, 2, 1, 32), k.dtype(), &device)?;
            let actual = candle_nn::ops::sdpa(&q, &k, &v, None, false, 0.1767767, 1.)?;
            let expected = candle_nn::ops::sdpa(
                &q,
                &k.contiguous()?,
                &v.contiguous()?,
                None,
                false,
                0.1767767,
                1.,
            )?;
            let error = (&actual - &expected)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error < 1e-5, "Metal cache attention error: {error}");
        }
        Ok(())
    }
}
