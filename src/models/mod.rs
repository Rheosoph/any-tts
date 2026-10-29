//! Model backends for TTS synthesis.

mod archive;

#[cfg(feature = "kokoro")]
pub mod kokoro;

#[cfg(feature = "omnivoice")]
pub mod omnivoice;

#[cfg(feature = "qwen3-tts")]
pub mod qwen3_tts;

#[cfg(feature = "vibevoice")]
pub mod vibevoice;

#[cfg(feature = "vibevoice")]
pub mod vibevoice_realtime;

#[cfg(feature = "voxtral")]
pub mod voxtral;

/// A documented model asset requirement or optional asset pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAssetRequirement {
    pub pattern: &'static str,
    pub required: bool,
    pub purpose: &'static str,
}

const KOKORO_ASSETS: &[ModelAssetRequirement] = &[
    ModelAssetRequirement {
        pattern: "config.json",
        required: true,
        purpose: "Model architecture and phoneme vocabulary.",
    },
    ModelAssetRequirement {
        pattern: "model.safetensors | *.pth",
        required: true,
        purpose: "Main Kokoro weights.",
    },
    ModelAssetRequirement {
        pattern: "voices/*.pt",
        required: false,
        purpose: "Preset voice packs for named-voice synthesis.",
    },
];

const OMNIVOICE_ASSETS: &[ModelAssetRequirement] = &[
    ModelAssetRequirement {
        pattern: "config.json",
        required: true,
        purpose: "Main OmniVoice config.",
    },
    ModelAssetRequirement {
        pattern: "tokenizer.json",
        required: true,
        purpose: "Text tokenizer.",
    },
    ModelAssetRequirement {
        pattern: "model.safetensors | model-*-of-*.safetensors",
        required: true,
        purpose: "Main OmniVoice weights.",
    },
    ModelAssetRequirement {
        pattern: "audio_tokenizer/config.json",
        required: true,
        purpose: "Codec decoder config.",
    },
    ModelAssetRequirement {
        pattern: "audio_tokenizer/model.safetensors | audio_tokenizer/model-*-of-*.safetensors",
        required: true,
        purpose: "Codec decoder weights.",
    },
];

const QWEN3_TTS_ASSETS: &[ModelAssetRequirement] = &[
    ModelAssetRequirement {
        pattern: "config.json",
        required: true,
        purpose: "Main talker/code-predictor config.",
    },
    ModelAssetRequirement {
        pattern: "tokenizer.json",
        required: true,
        purpose: "Text tokenizer.",
    },
    ModelAssetRequirement {
        pattern: "model.safetensors | model-*-of-*.safetensors",
        required: true,
        purpose: "Main Qwen3-TTS weights.",
    },
    ModelAssetRequirement {
        pattern: "speech_tokenizer/model.safetensors | speech_tokenizer/model-*-of-*.safetensors",
        required: true,
        purpose: "Speech-tokenizer decoder weights.",
    },
    ModelAssetRequirement {
        pattern: "speech_tokenizer/config.json",
        required: false,
        purpose: "Optional speech-tokenizer config when it is stored beside the main assets.",
    },
];

const VIBEVOICE_ASSETS: &[ModelAssetRequirement] = &[
    ModelAssetRequirement {
        pattern: "config.json",
        required: true,
        purpose: "Main VibeVoice config.",
    },
    ModelAssetRequirement {
        pattern: "tokenizer.json",
        required: true,
        purpose: "Text tokenizer.",
    },
    ModelAssetRequirement {
        pattern: "model.safetensors | model-*-of-*.safetensors",
        required: true,
        purpose: "Unified VibeVoice weights.",
    },
    ModelAssetRequirement {
        pattern: "preprocessor_config.json",
        required: false,
        purpose: "Published preprocessing defaults.",
    },
];

const VIBEVOICE_REALTIME_ASSETS: &[ModelAssetRequirement] = &[
    ModelAssetRequirement {
        pattern: "config.json",
        required: true,
        purpose: "Main VibeVoice Realtime config.",
    },
    ModelAssetRequirement {
        pattern: "tokenizer.json",
        required: true,
        purpose: "Text tokenizer.",
    },
    ModelAssetRequirement {
        pattern: "model.safetensors",
        required: true,
        purpose: "Realtime VibeVoice weights.",
    },
    ModelAssetRequirement {
        pattern: "preprocessor_config.json",
        required: false,
        purpose: "Published preprocessing defaults.",
    },
    ModelAssetRequirement {
        pattern: "voices/*.pt",
        required: false,
        purpose: "Optional cached-prompt voice presets from the upstream demo bundle.",
    },
];

const VOXTRAL_ASSETS: &[ModelAssetRequirement] = &[
    ModelAssetRequirement {
        pattern: "params.json",
        required: true,
        purpose: "Main Voxtral config.",
    },
    ModelAssetRequirement {
        pattern: "tekken.json",
        required: true,
        purpose: "Tekken tokenizer.",
    },
    ModelAssetRequirement {
        pattern: "consolidated.safetensors",
        required: true,
        purpose: "Main Voxtral weights.",
    },
    ModelAssetRequirement {
        pattern: "voice_embedding/*.pt",
        required: true,
        purpose: "Preset voice embeddings.",
    },
];

/// Supported model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Kokoro-82M: 82M parameter StyleTTS2 model with ISTFTNet decoder.
    Kokoro,
    /// OmniVoice: native Candle implementation for omnilingual zero-shot TTS.
    OmniVoice,
    /// Qwen3-TTS-12Hz-1.7B-CustomVoice: 1.7B multi-codebook LM.
    Qwen3Tts,
    /// VibeVoice-1.5B: native Candle implementation with diffusion speech tokens.
    VibeVoice,
    /// VibeVoice-Realtime-0.5B: native Candle implementation with cached prompt presets.
    VibeVoiceRealtime,
    /// Voxtral-4B-TTS-2603: native Candle implementation.
    Voxtral,
}

impl ModelType {
    /// Return the documented asset layout for this backend.
    pub fn asset_requirements(self) -> &'static [ModelAssetRequirement] {
        match self {
            Self::Kokoro => KOKORO_ASSETS,
            Self::OmniVoice => OMNIVOICE_ASSETS,
            Self::Qwen3Tts => QWEN3_TTS_ASSETS,
            Self::VibeVoice => VIBEVOICE_ASSETS,
            Self::VibeVoiceRealtime => VIBEVOICE_REALTIME_ASSETS,
            Self::Voxtral => VOXTRAL_ASSETS,
        }
    }
}
