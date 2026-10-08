use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use candle_core::{DType, Device, Shape, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::TtsError;

use super::config::VibeVoiceTokenizerConfig;
use super::speech_tokenizer::VibeVoiceTokenizerEncoderOutput;

const DIFFUSION_NOISE_FORMAT: &str = "vibevoice-diffusion-noise-v1";
const DEFAULT_GENERATION_SEED: u64 = 299_792_458;
const DEFAULT_PROGRESS_INTERVAL: usize = 0;
const DEFAULT_SEMANTIC_FEEDBACK_WINDOW: usize = usize::MAX;
const VIBEVOICE_NOISE_PATH_ENV: &str = "VIBEVOICE_NOISE_PATH";
const VIBEVOICE_PROGRESS_INTERVAL_ENV: &str = "VIBEVOICE_PROGRESS_INTERVAL";
const VIBEVOICE_SEED_ENV: &str = "VIBEVOICE_SEED";
const VIBEVOICE_SEMANTIC_FEEDBACK_WINDOW_ENV: &str = "VIBEVOICE_SEMANTIC_FEEDBACK_WINDOW";

pub type LayerKvCache = Option<crate::layers::kv_cache::KvCacheState>;

pub struct DecoderCacheState {
    next_position: usize,
    last_hidden: Tensor,
    logits: Option<Tensor>,
    layer_caches: Vec<LayerKvCache>,
}

impl DecoderCacheState {
    pub fn new(
        next_position: usize,
        last_hidden: Tensor,
        logits: Option<Tensor>,
        layer_caches: Vec<LayerKvCache>,
    ) -> Self {
        Self {
            next_position,
            last_hidden,
            logits,
            layer_caches,
        }
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn last_hidden(&self) -> &Tensor {
        &self.last_hidden
    }

    pub fn logits(&self) -> Result<&Tensor, TtsError> {
        self.logits.as_ref().ok_or_else(|| {
            TtsError::ModelError("This VibeVoice state has no control logits".to_string())
        })
    }

    pub fn layer_caches(&self) -> &[LayerKvCache] {
        &self.layer_caches
    }
}

pub struct GenerationParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub cfg_scale: f32,
}

pub struct GenerationArtifacts {
    pub segments: Vec<Vec<Tensor>>,
    pub trace: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionNoiseFixture {
    #[serde(default = "default_diffusion_noise_format")]
    pub format: String,
    pub latent_size: usize,
    #[serde(default)]
    pub noises: Vec<Vec<f32>>,
}

impl DiffusionNoiseFixture {
    pub fn from_file(path: &Path) -> Result<Self, TtsError> {
        let content = fs::read_to_string(path)?;
        let fixture: Self = serde_json::from_str(&content)?;
        fixture.validate()?;
        Ok(fixture)
    }

    pub fn cursor(&self) -> DiffusionNoiseCursor<'_> {
        DiffusionNoiseCursor {
            fixture: self,
            index: 0,
        }
    }

    fn validate(&self) -> Result<(), TtsError> {
        if self.format != DIFFUSION_NOISE_FORMAT {
            return Err(TtsError::ModelError(format!(
                "Unsupported VibeVoice diffusion noise fixture format: {}",
                self.format
            )));
        }

        if self.latent_size == 0 {
            return Err(TtsError::ModelError(
                "VibeVoice diffusion noise fixture must declare a positive latent size".to_string(),
            ));
        }

        for (index, noise) in self.noises.iter().enumerate() {
            if noise.len() != self.latent_size {
                return Err(TtsError::ModelError(format!(
                    "VibeVoice diffusion noise row {} has width {}, expected {}",
                    index,
                    noise.len(),
                    self.latent_size,
                )));
            }
        }

        Ok(())
    }
}

pub struct DiffusionNoiseCursor<'a> {
    fixture: &'a DiffusionNoiseFixture,
    index: usize,
}

impl<'a> DiffusionNoiseCursor<'a> {
    pub fn next_tensor(&mut self, device: &Device, dtype: DType) -> Result<Tensor, TtsError> {
        let noise = self
            .fixture
            .noises
            .get(self.index)
            .ok_or_else(|| {
                TtsError::ModelError(format!(
                    "VibeVoice diffusion noise fixture ended after {} token(s)",
                    self.index,
                ))
            })?
            .clone();
        self.index += 1;

        Tensor::new(noise.as_slice(), device)?
            .unsqueeze(0)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

pub struct TokenSequenceState {
    embedding_rows: Vec<Tensor>,
}

impl TokenSequenceState {
    pub fn from_base_embeddings(
        _token_ids: &[u32],
        base_embeddings: &Tensor,
        embedding_overrides: &HashMap<usize, Tensor>,
    ) -> Result<Self, TtsError> {
        let seq_len = validated_sequence_len(base_embeddings, _token_ids.len())?;
        Ok(Self {
            embedding_rows: embedding_rows(base_embeddings, embedding_overrides, seq_len)?,
        })
    }

    pub fn input_embeddings(&self) -> Result<Tensor, TtsError> {
        if self.embedding_rows.is_empty() {
            return Err(TtsError::ModelError(
                "Cannot build VibeVoice input embeddings for an empty sequence".to_string(),
            ));
        }

        let rows = self.embedding_rows.iter().collect::<Vec<_>>();
        Tensor::cat(&rows, 0)?.unsqueeze(0).map_err(Into::into)
    }
}

fn normalize_embedding_row(embedding: Tensor) -> Result<Tensor, TtsError> {
    match embedding.rank() {
        1 => embedding.unsqueeze(0).map_err(Into::into),
        2 => Ok(embedding),
        _ => Err(TtsError::ModelError(
            "Unexpected VibeVoice embedding rank while updating generation state".to_string(),
        )),
    }
}

fn embedding_row(
    base_embeddings: &Tensor,
    embedding_overrides: &HashMap<usize, Tensor>,
    position: usize,
) -> Result<Tensor, TtsError> {
    if let Some(override_embedding) = embedding_overrides.get(&position) {
        return normalize_embedding_row(override_embedding.clone());
    }

    base_embeddings
        .narrow(1, position, 1)?
        .squeeze(0)
        .map_err(Into::into)
}

fn embedding_rows(
    base_embeddings: &Tensor,
    embedding_overrides: &HashMap<usize, Tensor>,
    seq_len: usize,
) -> Result<Vec<Tensor>, TtsError> {
    (0..seq_len)
        .map(|position| embedding_row(base_embeddings, embedding_overrides, position))
        .collect()
}

fn validated_sequence_len(
    base_embeddings: &Tensor,
    expected_len: usize,
) -> Result<usize, TtsError> {
    let (batch, seq_len, _hidden) = base_embeddings.dims3()?;
    if batch == 1 && seq_len == expected_len {
        return Ok(seq_len);
    }

    Err(TtsError::ModelError(
        "Unexpected VibeVoice embedding shape while building generation state".to_string(),
    ))
}

fn default_diffusion_noise_format() -> String {
    DIFFUSION_NOISE_FORMAT.to_string()
}

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

pub fn random_normal_tensor<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
    rng: &mut SimpleRng,
) -> Result<Tensor, TtsError> {
    let shape = shape.into();
    let elem_count = shape.elem_count();
    let mut values = Vec::with_capacity(elem_count);
    while values.len() < elem_count {
        let u1 = rng.next_f32().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let u2 = rng.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        values.push(radius * theta.cos());
        if values.len() < elem_count {
            values.push(radius * theta.sin());
        }
    }

    Tensor::from_vec(values, shape, device)
        .and_then(|tensor| tensor.to_dtype(dtype))
        .map_err(Into::into)
}

pub fn sample_encoder_output(
    output: &VibeVoiceTokenizerEncoderOutput,
    config: &VibeVoiceTokenizerConfig,
    device: &Device,
    rng: &mut SimpleRng,
) -> Result<Tensor, TtsError> {
    if config.std_dist_type != "gaussian" {
        return Ok(output.mean.clone());
    }

    let std = output.std.unwrap_or(config.fix_std) as f32;
    let noise = random_normal_tensor(output.mean.shape().clone(), DType::F32, device, rng)?
        .to_dtype(output.mean.dtype())?;
    output
        .mean
        .broadcast_add(&noise.broadcast_mul(&Tensor::new(std, device)?)?)
        .map_err(Into::into)
}

pub fn scale_acoustic_features(
    acoustic_latents: &Tensor,
    bias_factor: f32,
    scaling_factor: f32,
    device: &Device,
) -> Result<Tensor, TtsError> {
    acoustic_latents
        .broadcast_add(&Tensor::new(bias_factor, device)?)?
        .broadcast_mul(&Tensor::new(scaling_factor, device)?)
        .map_err(Into::into)
}

pub fn stack_latents(latents: &[Tensor]) -> Result<Tensor, TtsError> {
    let pieces = latents
        .iter()
        .map(|latent| latent.unsqueeze(0))
        .collect::<Result<Vec<_>, _>>()?;
    let piece_refs = pieces.iter().collect::<Vec<_>>();
    Tensor::cat(&piece_refs, 0)?
        .unsqueeze(0)
        .map_err(Into::into)
}

pub fn valid_generated_tokens(
    speech_start_id: u32,
    speech_end_id: u32,
    speech_diffusion_id: u32,
    eos_id: u32,
    bos_id: Option<u32>,
) -> Vec<u32> {
    let mut tokens = vec![speech_start_id, speech_end_id, speech_diffusion_id, eos_id];
    if let Some(bos_id) = bos_id {
        tokens.push(bos_id);
    }
    tokens
}

pub fn sample_token(
    logits: &Tensor,
    valid_tokens: &[u32],
    temperature: f32,
    rng: &mut SimpleRng,
) -> Result<u32, TtsError> {
    let logits = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    if logits.len() != valid_tokens.len() {
        return Err(TtsError::ModelError(
            "VibeVoice control logits and token IDs differ in length".to_string(),
        ));
    }

    if temperature <= 0.0 {
        return greedy_token(&logits, valid_tokens);
    }

    sample_weighted_token(&logits, valid_tokens, temperature, rng)
}

pub fn generation_seed() -> u64 {
    std::env::var(VIBEVOICE_SEED_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_GENERATION_SEED)
}

pub fn load_diffusion_noise_fixture() -> Result<Option<DiffusionNoiseFixture>, TtsError> {
    let Some(path) = std::env::var_os(VIBEVOICE_NOISE_PATH_ENV) else {
        return Ok(None);
    };
    let fixture = DiffusionNoiseFixture::from_file(&PathBuf::from(path))?;
    Ok(Some(fixture))
}

pub fn prompt_positions(speech_input_mask: &[bool]) -> Vec<usize> {
    speech_input_mask
        .iter()
        .enumerate()
        .filter_map(|(index, is_speech)| is_speech.then_some(index))
        .collect()
}

pub fn finish_segment(current_segment: &mut Vec<Tensor>, finished_segments: &mut Vec<Vec<Tensor>>) {
    if current_segment.is_empty() {
        return;
    }
    finished_segments.push(std::mem::take(current_segment));
}

pub fn feedback_mode() -> Option<&'static str> {
    match std::env::var("VIBEVOICE_FEEDBACK_MODE").ok().as_deref() {
        Some("token") => Some("token"),
        Some("acoustic") => Some("acoustic"),
        Some("semantic") => Some("semantic"),
        _ => None,
    }
}

pub fn progress_interval() -> usize {
    std::env::var(VIBEVOICE_PROGRESS_INTERVAL_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PROGRESS_INTERVAL)
}

pub fn semantic_feedback_window() -> usize {
    std::env::var(VIBEVOICE_SEMANTIC_FEEDBACK_WINDOW_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.max(1))
        .unwrap_or(DEFAULT_SEMANTIC_FEEDBACK_WINDOW)
}

fn greedy_token(logits: &[f32], valid_tokens: &[u32]) -> Result<u32, TtsError> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| valid_tokens[index])
        .ok_or_else(|| TtsError::ModelError("No valid VibeVoice tokens available".to_string()))
}

fn sample_weighted_token(
    logits: &[f32],
    valid_tokens: &[u32],
    temperature: f32,
    rng: &mut SimpleRng,
) -> Result<u32, TtsError> {
    let max_logit = logits
        .iter()
        .map(|logit| *logit / temperature)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut cumulative = 0.0f32;
    let weights = logits
        .iter()
        .map(|logit| {
            let weight = ((*logit / temperature) - max_logit).exp();
            cumulative += weight;
            weight
        })
        .collect::<Vec<_>>();

    let threshold = rng.next_f32() * cumulative.max(f32::EPSILON);
    let mut running = 0.0f32;
    for (index, token) in valid_tokens.iter().enumerate() {
        running += weights[index];
        if running >= threshold {
            return Ok(*token);
        }
    }

    valid_tokens
        .last()
        .copied()
        .ok_or_else(|| TtsError::ModelError("No valid VibeVoice tokens available".to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

    use super::{default_diffusion_noise_format, DiffusionNoiseFixture, TokenSequenceState};

    #[test]
    fn restricted_control_logits_map_back_to_original_token_ids() {
        let logits = Tensor::new(&[1f32, 8.0, 2.0, 0.0], &Device::Cpu).unwrap();
        let tokens = [151_652, 151_653, 151_654, 151_643];
        let mut rng = super::SimpleRng::new(42);
        assert_eq!(
            super::sample_token(&logits, &tokens, 0.0, &mut rng).unwrap(),
            tokens[1]
        );
        assert!(super::sample_token(&logits, &tokens[..2], 0.0, &mut rng).is_err());
    }

    #[test]
    fn token_sequence_state_applies_overrides() {
        let device = Device::Cpu;
        let base = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 3, 2), &device).unwrap();
        let mut overrides = HashMap::new();
        overrides.insert(
            1usize,
            Tensor::from_vec(vec![30f32, 40.], 2, &device).unwrap(),
        );

        let state =
            TokenSequenceState::from_base_embeddings(&[10, 11, 12], &base, &overrides).unwrap();
        let rows = state
            .input_embeddings()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert_eq!(
            rows,
            vec![vec![vec![1.0, 2.0], vec![30.0, 40.0], vec![5.0, 6.0]]]
        );
    }

    #[test]
    fn diffusion_noise_fixture_validates_row_widths() {
        let fixture = DiffusionNoiseFixture {
            format: default_diffusion_noise_format(),
            latent_size: 4,
            noises: vec![vec![0.0; 4], vec![1.0; 4]],
        };
        let device = Device::Cpu;
        let mut cursor = fixture.cursor();
        let tensor = cursor.next_tensor(&device, DType::F32).unwrap();

        assert_eq!(tensor.dims(), &[1, 4]);
    }
}
