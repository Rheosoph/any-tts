//! Core TTS trait and request/response types.

use std::path::Path;

use crate::audio::AudioSamples;
use crate::config::TtsConfig;
use crate::error::TtsError;

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Human-readable model name.
    pub name: String,
    /// Model version or variant.
    pub variant: String,
    /// Approximate parameter count.
    pub parameters: u64,
    /// Output audio sample rate in Hz.
    pub sample_rate: u32,
    /// Supported languages (ISO 639-1 codes or language names).
    pub languages: Vec<String>,
    /// Available voice/speaker names.
    pub voices: Vec<String>,
}

/// A request to synthesize speech from text.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// The text to synthesize.
    pub text: String,
    /// Target language (ISO code or name). If `None`, auto-detect.
    pub language: Option<String>,
    /// Voice/speaker name. If `None`, use default.
    pub voice: Option<String>,
    /// Style instruction (model-dependent, e.g. "angry", "whisper").
    pub instruct: Option<String>,
    /// Maximum number of tokens to generate. If `None`, use model default.
    pub max_tokens: Option<usize>,
    /// Sampling temperature. If `None`, use model default.
    pub temperature: Option<f64>,
    /// Speaking-rate multiplier for backends that support it (currently Kokoro).
    /// `1.0` is the normal rate; this is independent of sampling temperature.
    pub speed: Option<f64>,
    /// Guidance scale for backends that support classifier-free guidance.
    pub cfg_scale: Option<f64>,
    /// Reference audio for zero-shot voice cloning.
    ///
    /// When provided, the model will attempt to match the voice characteristics
    /// of this audio instead of using a named voice. Not all models support this;
    /// models that don't will return [`TtsError::ModelError`].
    pub reference_audio: Option<ReferenceAudio>,
    /// Pre-extracted voice embedding for voice cloning.
    ///
    /// Use this to re-use an embedding extracted via [`VoiceCloning::extract_voice`]
    /// without re-processing the reference audio each time.
    pub voice_embedding: Option<VoiceEmbedding>,
}

impl SynthesisRequest {
    /// Create a new synthesis request with the given text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
            voice: None,
            instruct: None,
            max_tokens: None,
            temperature: None,
            speed: None,
            cfg_scale: None,
            reference_audio: None,
            voice_embedding: None,
        }
    }

    /// Set the target language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set the voice/speaker.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Set a style instruction.
    pub fn with_instruct(mut self, instruct: impl Into<String>) -> Self {
        self.instruct = Some(instruct.into());
        self
    }

    /// Set the maximum number of tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the speaking-rate multiplier (currently supported by Kokoro).
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Set the classifier-free guidance scale.
    pub fn with_cfg_scale(mut self, cfg_scale: f64) -> Self {
        self.cfg_scale = Some(cfg_scale);
        self
    }

    /// Set reference audio for zero-shot voice cloning.
    ///
    /// The model will extract speaker characteristics from this audio and
    /// use them to condition the synthesis. Overrides any named voice.
    pub fn with_reference_audio(mut self, audio: ReferenceAudio) -> Self {
        self.reference_audio = Some(audio);
        self
    }

    /// Set a pre-extracted voice embedding.
    ///
    /// Useful for caching: extract once with [`VoiceCloning::extract_voice`],
    /// then reuse across multiple synthesis calls.
    pub fn with_voice_embedding(mut self, embedding: VoiceEmbedding) -> Self {
        self.voice_embedding = Some(embedding);
        self
    }
}

/// The core trait that all TTS model backends must implement.
///
/// This provides a unified interface for text-to-speech synthesis regardless
/// of the underlying model architecture.
pub trait TtsModel: Send + Sync {
    /// Load model weights and initialize the model from configuration.
    ///
    /// This may involve reading safetensors files, parsing config JSON,
    /// and moving weights to the target device.
    fn load(config: TtsConfig) -> Result<Self, TtsError>
    where
        Self: Sized;

    /// Synthesize speech from a text request.
    ///
    /// Returns raw f32 PCM audio samples at the model's native sample rate.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError>;

    /// Deliver consecutive audio chunks in playback order.
    ///
    /// The callback runs synchronously and provides backpressure. Returning an
    /// error cancels generation immediately; chunks already delivered remain
    /// valid. Chunks use the model's native sample rate and contain no repeated
    /// overlap. Implementations without incremental synthesis deliver one
    /// complete chunk. Query [`Self::supports_streaming`] to distinguish them.
    fn synthesize_stream(
        &self,
        request: &SynthesisRequest,
        on_chunk: &mut dyn FnMut(AudioSamples) -> Result<(), TtsError>,
    ) -> Result<(), TtsError> {
        on_chunk(self.synthesize(request)?)
    }

    /// Whether this backend can deliver audio before the full request finishes.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Return the native output sample rate in Hz (e.g. 24000).
    fn sample_rate(&self) -> u32;

    /// Return the list of supported language identifiers.
    fn supported_languages(&self) -> Vec<String>;

    /// Return the list of available voice/speaker names.
    fn supported_voices(&self) -> Vec<String>;

    /// Return metadata about this model.
    fn model_info(&self) -> ModelInfo;
}

// ---------------------------------------------------------------------------
// Voice cloning types
// ---------------------------------------------------------------------------

/// Raw audio data used as a reference for voice cloning.
///
/// The model will extract speaker characteristics from this audio and use
/// them to condition speech synthesis. For best results:
///
/// - Use 3–10 seconds of clean speech (single speaker, no background noise)
/// - Match the model's native sample rate (e.g. 24 kHz for Kokoro)
///   or the library will resample automatically
///
/// # Example
///
/// ```rust
/// use any_tts::ReferenceAudio;
///
/// let audio = ReferenceAudio::new(vec![0.0f32; 24000], 24000);
/// assert_eq!(audio.duration_secs(), 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct ReferenceAudio {
    /// Raw f32 PCM audio samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate of the audio in Hz.
    pub sample_rate: u32,
}

impl ReferenceAudio {
    /// Create a new reference audio from raw samples.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Duration of the reference audio in seconds.
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Whether the audio is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// An extracted voice embedding that can be saved, loaded, and reused.
///
/// Voice embeddings are model-specific opaque data. An embedding extracted
/// from one model type cannot be used with a different model type.
///
/// # Persistence
///
/// Embeddings can be saved to and loaded from JSON files:
///
/// ```rust,ignore
/// // Extract once
/// let embedding = model.extract_voice(&reference)?;
/// embedding.save("my_voice.json")?;
///
/// // Reuse later
/// let embedding = VoiceEmbedding::load("my_voice.json")?;
/// let request = SynthesisRequest::new("Hello!")
///     .with_voice_embedding(embedding);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoiceEmbedding {
    /// Raw embedding data as flattened f32 values.
    data: Vec<f32>,
    /// Shape of the embedding tensor.
    shape: Vec<usize>,
    /// Model type identifier (e.g. "kokoro", "qwen3-tts").
    model_type: String,
}

impl VoiceEmbedding {
    /// Create a new voice embedding from raw data.
    pub fn new(data: Vec<f32>, shape: Vec<usize>, model_type: impl Into<String>) -> Self {
        Self {
            data,
            shape,
            model_type: model_type.into(),
        }
    }

    /// Reconstruct the embedding as a candle [`Tensor`](candle_core::Tensor).
    pub fn to_tensor(&self, device: &candle_core::Device) -> Result<candle_core::Tensor, TtsError> {
        candle_core::Tensor::new(self.data.as_slice(), device)?
            .reshape(self.shape.as_slice())
            .map_err(TtsError::from)
    }

    /// The model type this embedding was extracted for.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Shape of the embedding tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Save the embedding to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), TtsError> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load an embedding from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        let json = std::fs::read_to_string(path)?;
        let embedding: Self = serde_json::from_str(&json)?;
        Ok(embedding)
    }
}

/// Trait for TTS models that support voice cloning from reference audio.
///
/// Not all models support voice cloning. Check [`VoiceCloning::supports_voice_cloning`]
/// before calling other methods.
///
/// # Example
///
/// ```rust,ignore
/// use any_tts::{VoiceCloning, ReferenceAudio, SynthesisRequest};
///
/// if model.supports_voice_cloning() {
///     let reference = ReferenceAudio::new(samples, 24000);
///     let embedding = model.extract_voice(&reference)?;
///     let request = SynthesisRequest::new("Hello in the cloned voice!")
///         .with_voice_embedding(embedding);
///     let audio = model.synthesize(&request)?;
/// }
/// ```
pub trait VoiceCloning: TtsModel {
    /// Whether voice cloning is currently available.
    ///
    /// Returns `false` if the model doesn't support voice cloning or if
    /// the required encoder weights were not found during loading.
    fn supports_voice_cloning(&self) -> bool;

    /// Extract a reusable voice embedding from reference audio.
    ///
    /// The returned embedding encodes the speaker's voice characteristics
    /// and can be saved to disk for later use.
    fn extract_voice(&self, audio: &ReferenceAudio) -> Result<VoiceEmbedding, TtsError>;

    /// Synthesize speech conditioned on a pre-extracted voice embedding.
    fn synthesize_with_voice(
        &self,
        request: &SynthesisRequest,
        voice: &VoiceEmbedding,
    ) -> Result<AudioSamples, TtsError>;
}
