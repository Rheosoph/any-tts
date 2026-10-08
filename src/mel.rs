//! Mel spectrogram extraction for voice cloning and audio analysis.
//!
//! Uses a cached CPU FFT plan for centered, reflect-padded STFT frames, then
//! applies the mel filterbank on the configured Candle device.
//!
//! # Example
//!
//! ```rust,ignore
//! use any_tts::mel::{MelConfig, MelSpectrogram};
//! use candle_core::Device;
//!
//! let mel = MelSpectrogram::new(MelConfig::kokoro(), &Device::Cpu)?;
//! let audio = candle_core::Tensor::zeros(24000, candle_core::DType::F32, &Device::Cpu)?;
//! let spectrogram = mel.compute(&audio)?;
//! // spectrogram shape: [1, 80, num_frames]
//! ```

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::error::{TtsError, TtsResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for mel spectrogram extraction.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// FFT size (number of frequency bins before taking positive half).
    pub n_fft: usize,
    /// Hop length between STFT frames, in samples.
    pub hop_length: usize,
    /// Analysis window length, in samples. May be ≤ `n_fft`.
    pub win_length: usize,
    /// Number of mel frequency bands.
    pub n_mels: usize,
    /// Expected sample rate of input audio (Hz).
    pub sample_rate: u32,
    /// Log-mel normalization mean (subtracted before dividing by `log_std`).
    pub log_mean: f64,
    /// Log-mel normalization std (divides after subtracting `log_mean`).
    pub log_std: f64,
}

impl MelConfig {
    /// Config matching Kokoro's style encoder preprocessing.
    ///
    /// ```text
    /// n_fft=2048, hop=300, win=1200, 80 mels, 24 kHz
    /// norm: (log(1e-5 + mel) - (-4)) / 4
    /// ```
    pub fn kokoro() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 300,
            win_length: 1200,
            n_mels: 80,
            sample_rate: 24000,
            log_mean: -4.0,
            log_std: 4.0,
        }
    }

    /// Number of positive frequency bins: `n_fft / 2 + 1`.
    pub fn n_freq(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

// ---------------------------------------------------------------------------
// MelSpectrogram
// ---------------------------------------------------------------------------

/// Mel spectrogram extractor.
///
/// Pre-computes the FFT plan, Hann window, and mel filterbank on
/// construction so that repeated calls to [`compute`](Self::compute) are fast.
pub struct MelSpectrogram {
    config: MelConfig,
    fft: Arc<dyn Fft<f32>>,
    /// Periodic Hann window, centered and zero-padded to `n_fft` length.
    window: Vec<f32>,
    /// Mel filterbank `[n_mels, n_freq]`.
    mel_basis: Tensor,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram extractor.
    pub fn new(config: MelConfig, device: &Device) -> TtsResult<Self> {
        if config.n_fft < 2
            || config.n_fft % 2 != 0
            || config.hop_length == 0
            || config.win_length == 0
            || config.win_length > config.n_fft
            || config.n_mels == 0
            || config.sample_rate == 0
            || !config.log_mean.is_finite()
            || !config.log_std.is_finite()
            || config.log_std <= 0.0
        {
            return Err(TtsError::ConfigError("Invalid mel spectrogram configuration: use an even FFT size, positive dimensions/sample rate, and finite normalization".into()));
        }
        let n_fft = config.n_fft;
        let n_freq = config.n_freq();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);

        // ── Hann window (centre-padded to n_fft) ─────────────────────
        let mut window_data = vec![0f32; n_fft];
        let pad_left = (n_fft - config.win_length) / 2;
        for i in 0..config.win_length {
            let w = 0.5
                * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / config.win_length as f32).cos());
            window_data[pad_left + i] = w;
        }
        let window = window_data;

        // ── Mel filterbank ────────────────────────────────────────────
        let mel_basis =
            Self::build_mel_filterbank(config.n_mels, n_freq, config.sample_rate, device)?;

        Ok(Self {
            config,
            fft,
            window,
            mel_basis,
        })
    }

    /// Compute a log-mel spectrogram from raw PCM audio.
    ///
    /// * **Input:** 1-D `[num_samples]` f32 tensor at [`MelConfig::sample_rate`].
    /// * **Output:** `[1, n_mels, num_frames]` normalised log-mel spectrogram.
    pub fn compute(&self, audio: &Tensor) -> TtsResult<Tensor> {
        if audio.rank() != 1 {
            return Err(TtsError::AudioError(
                "Mel input must be a one-dimensional mono waveform".into(),
            ));
        }
        let n_samples = audio.dim(0)?;
        let n_fft = self.config.n_fft;
        let pad = n_fft / 2;
        if n_samples <= pad {
            return Err(TtsError::AudioError(format!(
                "Mel extraction needs more than {pad} audio samples for reflection padding"
            )));
        }
        let samples = audio
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let num_frames = n_samples / self.config.hop_length + 1;
        let n_freq = self.config.n_freq();
        let mut power = vec![0.0f32; n_freq * num_frames];
        let mut frame = vec![Complex32::default(); n_fft];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];

        // Reflection matches centered STFT: the endpoint sample is not repeated.
        // Reuse the FFT input and scratch buffers across every frame.
        for frame_index in 0..num_frames {
            let start = frame_index * self.config.hop_length;
            for (index, value) in frame.iter_mut().enumerate() {
                let padded_index = start + index;
                let source = if padded_index < pad {
                    pad - padded_index
                } else if padded_index >= pad + n_samples {
                    2 * n_samples + pad - padded_index - 2
                } else {
                    padded_index - pad
                };
                *value = Complex32::new(samples[source] * self.window[index], 0.0);
            }
            self.fft.process_with_scratch(&mut frame, &mut scratch);
            for frequency in 0..n_freq {
                power[frequency * num_frames + frame_index] = frame[frequency].norm_sqr();
            }
        }
        let power = Tensor::from_vec(power, (n_freq, num_frames), self.device())?;
        let mel = self.mel_basis.matmul(&power)?;

        // ── Log compression + normalisation ───────────────────────────
        let log_mel = (mel + 1e-5)?.log()?;
        let normalised = log_mel.affine(
            1.0 / self.config.log_std,
            -self.config.log_mean / self.config.log_std,
        )?;

        // [1, n_mels, num_frames]
        normalised.unsqueeze(0).map_err(TtsError::from)
    }

    /// Get the config used by this extractor.
    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    /// Device on which mel features are returned.
    pub fn device(&self) -> &Device {
        self.mel_basis.device()
    }

    // ── Private helpers ───────────────────────────────────────────────

    /// Build a triangular mel filterbank `[n_mels, n_freq]`.
    fn build_mel_filterbank(
        n_mels: usize,
        n_freq: usize,
        sample_rate: u32,
        device: &Device,
    ) -> TtsResult<Tensor> {
        let sr = sample_rate as f32;
        let fmax = sr / 2.0;

        let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
        let mel_to_hz = |m: f32| -> f32 { 700.0 * (10.0f32.powf(m / 2595.0) - 1.0) };

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(fmax);

        // n_mels + 2 equally-spaced points in mel space
        let n_points = n_mels + 2;
        let mel_points: Vec<f32> = (0..n_points)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Convert Hz → FFT bin (fractional)
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * (n_freq as f32 - 1.0) * 2.0 / sr)
            .collect();

        // Triangular filters
        let mut filters = vec![0f32; n_mels * n_freq];
        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            for k in 0..n_freq {
                let kf = k as f32;
                if kf >= f_left && kf <= f_center && f_center > f_left {
                    filters[m * n_freq + k] = (kf - f_left) / (f_center - f_left);
                } else if kf > f_center && kf <= f_right && f_right > f_center {
                    filters[m * n_freq + k] = (f_right - kf) / (f_right - f_center);
                }
            }
        }

        Tensor::new(filters.as_slice(), device)?
            .reshape((n_mels, n_freq))
            .map_err(TtsError::from)
    }
}

/// Resample mono audio with a windowed-sinc low-pass filter.
///
/// Downsampling removes frequencies above the destination Nyquist frequency.
/// A periodic phase table avoids evaluating the filter for every output sample
/// at common audio sample rates. Zero rates and impractical conversion ratios
/// return an error. Equal rates preserve the input samples exactly.
pub fn resample_audio(samples: &[f32], src_rate: u32, dst_rate: u32) -> TtsResult<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(TtsError::AudioError(
            "Audio sample rates must be greater than zero".into(),
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(TtsError::AudioError("Audio samples must be finite".into()));
    }
    if src_rate == dst_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let output_len = (samples.len() as u128 * dst_rate as u128).div_ceil(src_rate as u128);
    let output_len = usize::try_from(output_len)
        .map_err(|_| TtsError::AudioError("Resampled audio is too large".into()))?;
    let cutoff = (dst_rate as f64 / src_rate as f64).min(1.0) * 0.95;
    let radius = (16.0 / cutoff).ceil() as usize;
    if radius > 4096 {
        return Err(TtsError::AudioError(
            "Audio resampling ratio is too small".into(),
        ));
    }
    let taps = 2 * radius + 1;
    let mut gcd = src_rate;
    let mut remainder = dst_rate;
    while remainder != 0 {
        (gcd, remainder) = (remainder, gcd % remainder);
    }
    let phases = (dst_rate / gcd) as usize;
    let kernel = |phase: usize| {
        let fractional = phase as f64 / phases as f64;
        let mut weights = Vec::with_capacity(taps);
        for index in 0..taps {
            let x = index as f64 - radius as f64 - fractional;
            let z = std::f64::consts::PI * cutoff * x;
            let sinc = if z.abs() < 1e-12 { 1.0 } else { z.sin() / z };
            let window = if x.abs() <= radius as f64 {
                let angle = std::f64::consts::PI * x / radius as f64;
                0.42 + 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
            } else {
                0.0
            };
            weights.push(cutoff * sinc * window);
        }
        let sum: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum;
        }
        weights
    };
    // Bound table memory for unusual, coprime sample rates.
    let table = if phases.saturating_mul(taps) <= 262_144 {
        Some((0..phases).map(&kernel).collect::<Vec<_>>())
    } else {
        None
    };
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|e| TtsError::AudioError(format!("Cannot allocate resampled audio: {e}")))?;
    for index in 0..output_len {
        let position = index as u128 * src_rate as u128;
        let center = (position / dst_rate as u128) as usize;
        let phase = ((position % dst_rate as u128) / gcd as u128) as usize;
        let uncached;
        let weights = match &table {
            Some(table) => &table[phase],
            None => {
                uncached = kernel(phase);
                &uncached
            }
        };
        let mut value = 0.0f64;
        for (tap, weight) in weights.iter().enumerate() {
            // Clamp the extension to preserve constant signals at clip edges.
            let source = center
                .saturating_add(tap)
                .saturating_sub(radius)
                .min(samples.len() - 1);
            value += samples[source] as f64 * weight;
        }
        output.push(value as f32);
    }
    Ok(output)
}

/// Resample audio from `src_rate` to `dst_rate` using linear interpolation.
///
/// For voice cloning, the reference audio must match the model's expected
/// sample rate. Prefer [`resample_audio`] for antialiased reference conditioning.
/// This compatibility helper returns an empty vector for a zero sample rate.
pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_idx = i as f64 / ratio;
        let idx_floor = src_idx.floor() as usize;
        let frac = (src_idx - idx_floor as f64) as f32;

        let s0 = samples[idx_floor.min(samples.len() - 1)];
        let s1 = samples[(idx_floor + 1).min(samples.len() - 1)];
        output.push(s0 + frac * (s1 - s0));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> MelConfig {
        MelConfig {
            n_fft: 16,
            hop_length: 4,
            win_length: 12,
            n_mels: 4,
            ..MelConfig::kokoro()
        }
    }

    #[test]
    fn test_fft_mel_matches_reflect_padded_direct_dft() {
        let config = small_config();
        let extractor = MelSpectrogram::new(config.clone(), &Device::Cpu).unwrap();
        let samples = (0..64)
            .map(|i| (i as f32 * 0.31).sin() + 0.2)
            .collect::<Vec<_>>();
        let audio = Tensor::new(samples.as_slice(), &Device::Cpu).unwrap();
        let actual = extractor.compute(&audio).unwrap();
        let mut padded = samples[1..=8].iter().rev().copied().collect::<Vec<_>>();
        padded.extend_from_slice(&samples);
        padded.extend(samples[samples.len() - 9..samples.len() - 1].iter().rev());
        let frames = 17;
        let mut power = vec![0.0f32; config.n_freq() * frames];
        for frame in 0..frames {
            for frequency in 0..config.n_freq() {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for index in 0..config.n_fft {
                    let angle = 2.0 * std::f32::consts::PI * frequency as f32 * index as f32
                        / config.n_fft as f32;
                    let sample =
                        padded[frame * config.hop_length + index] * extractor.window[index];
                    real += sample * angle.cos();
                    imag -= sample * angle.sin();
                }
                power[frequency * frames + frame] = real * real + imag * imag;
            }
        }
        let power = Tensor::from_vec(power, (config.n_freq(), frames), &Device::Cpu).unwrap();
        let expected = (extractor.mel_basis.matmul(&power).unwrap() + 1e-5)
            .unwrap()
            .log()
            .unwrap()
            .affine(1.0 / config.log_std, -config.log_mean / config.log_std)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let error = actual
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-4, "FFT/direct DFT mel parity error: {error}");
    }

    #[test]
    fn test_reflection_keeps_constant_signal_edges_constant() {
        let extractor = MelSpectrogram::new(small_config(), &Device::Cpu).unwrap();
        let audio = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap();
        let mel = extractor
            .compute(&audio)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for channel in mel {
            assert!(channel
                .iter()
                .all(|value| (value - channel[0]).abs() < 1e-6));
        }
        assert!(extractor
            .compute(&Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap())
            .is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_mel_metal_matches_cpu_and_returns_model_device() {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("Skipping Metal mel test: {error}");
                return;
            }
        };
        let samples = (0..64).map(|i| (i as f32 * 0.31).sin()).collect::<Vec<_>>();
        let input = Tensor::new(samples.as_slice(), &Device::Cpu).unwrap();
        let cpu = MelSpectrogram::new(small_config(), &Device::Cpu).unwrap();
        let metal = MelSpectrogram::new(small_config(), &device).unwrap();
        let expected = cpu
            .compute(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for audio in [&input, &input.to_device(&device).unwrap()] {
            let actual = metal.compute(audio).unwrap();
            assert!(matches!(actual.device(), Device::Metal(_)));
            let values = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(values
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-4));
        }
    }

    #[test]
    fn test_resampling_rejects_aliases_and_preserves_passband() {
        let tone = |frequency: f32| {
            (0..4800)
                .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / 48000.0).sin())
                .collect::<Vec<_>>()
        };
        let rms = |values: &[f32]| {
            (values[100..values.len() - 100]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                / (values.len() - 200) as f32)
                .sqrt()
        };
        let stopband = resample_audio(&tone(18000.0), 48000, 24000).unwrap();
        let passband = resample_audio(&tone(1000.0), 48000, 24000).unwrap();
        assert!(rms(&stopband) < 0.002, "Stopband RMS: {}", rms(&stopband));
        assert!((rms(&passband) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.005);
    }

    #[test]
    fn test_resampling_lengths_constant_signals_and_validation() {
        for (source, target) in [(48000, 24000), (22050, 24000), (24000, 48000)] {
            let samples = vec![0.3f32; 101];
            let output = resample_audio(&samples, source, target).unwrap();
            assert_eq!(
                output.len(),
                (samples.len() as u64 * target as u64).div_ceil(source as u64) as usize
            );
            assert!(output.iter().all(|value| (value - 0.3).abs() < 1e-6));
        }
        assert!(resample_audio(&[], 0, 24000).is_err());
        assert!(resample_audio(&[0.0], 24000, 0).is_err());
        assert!(resample_audio(&[f32::NAN], 24000, 24000).is_err());
        let samples = vec![0.1f32, -0.4, 0.5];
        assert_eq!(resample_audio(&samples, 24000, 24000).unwrap(), samples);
    }

    #[test]
    fn test_mel_config_kokoro() {
        let cfg = MelConfig::kokoro();
        assert_eq!(cfg.n_fft, 2048);
        assert_eq!(cfg.n_freq(), 1025);
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.sample_rate, 24000);
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let device = Device::Cpu;
        let cfg = MelConfig::kokoro();
        let mel = MelSpectrogram::new(cfg, &device).unwrap();

        // 1 second of audio at 24kHz
        let audio = Tensor::zeros(24000, DType::F32, &device).unwrap();
        let spec = mel.compute(&audio).unwrap();

        assert_eq!(spec.dims()[0], 1); // batch
        assert_eq!(spec.dims()[1], 80); // n_mels
                                        // num_frames ≈ (24000 + 2048) / 300 = ~86
        assert!(spec.dims()[2] > 50);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let device = Device::Cpu;
        let fb = MelSpectrogram::build_mel_filterbank(80, 1025, 24000, &device).unwrap();
        assert_eq!(fb.dims(), &[80, 1025]);
    }

    #[test]
    fn test_mel_filterbank_values() {
        let device = Device::Cpu;
        let fb = MelSpectrogram::build_mel_filterbank(80, 1025, 24000, &device).unwrap();
        let data: Vec<Vec<f32>> = fb.to_vec2().unwrap();

        // Each row should have at least some non-zero values (triangular filter)
        for row in &data {
            let sum: f32 = row.iter().sum();
            assert!(sum > 0.0, "Mel filter band has zero energy");
        }
    }

    #[test]
    fn test_resample_identity() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let out = resample_linear(&samples, 16000, 16000);
        assert_eq!(out, samples);
    }

    #[test]
    fn test_resample_upsample() {
        let samples = vec![0.0, 1.0];
        let out = resample_linear(&samples, 1, 4);
        assert_eq!(out.len(), 8);
        // Should interpolate between 0.0 and 1.0
        assert!((out[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_resample_empty() {
        let out = resample_linear(&[], 16000, 24000);
        assert!(out.is_empty());
    }
}
