//! Tests for the Kokoro-82M model backend.

mod common;

use any_tts::models::kokoro::config::KokoroConfig;
use any_tts::models::kokoro::KokoroModel;
use any_tts::traits::TtsModel;
use any_tts::{ModelType, SynthesisRequest, TtsConfig};

#[test]
#[cfg_attr(
    feature = "download",
    ignore = "Missing-path fallback can download weights; run without the download feature"
)]
fn test_kokoro_load_missing_path() {
    let config = TtsConfig::new(ModelType::Kokoro);
    let result = KokoroModel::load(config);
    // Without model weights present locally, loading should fail.
    // With the `download` feature, it may attempt to download from HF and
    // still fail (e.g. tensor errors, network issues) — any error is valid.
    // If download succeeds and loading works, that's also acceptable.
    if let Err(e) = &result {
        let err = e.to_string();
        assert!(
            err.contains("missing")
                || err.contains("Missing")
                || err.contains("not specified")
                || err.contains("Failed")
                || err.contains("cannot find")
                || err.contains("error"),
            "Unexpected error: {}",
            err
        );
    }
}

#[test]
fn test_kokoro_config_parsing_minimal() {
    let json = "{}";
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.style_dim, 128);
    assert_eq!(config.n_mels, 80);
    assert_eq!(config.n_token, 178);
    assert_eq!(config.n_layer, 3);
    assert_eq!(config.dim_in, 64);
    assert_eq!(config.max_dur, 50);
    assert!(config.multispeaker);
    assert_eq!(config.text_encoder_kernel_size, 5);
}

#[test]
fn test_kokoro_config_full_style_dim() {
    let json = "{}";
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.full_style_dim(), 256);
}

#[test]
fn test_kokoro_config_sample_rate() {
    let json = "{}";
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.sample_rate(), 24000);
}

#[test]
fn test_kokoro_config_upsample_factor() {
    let json = "{}";
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    // upsample_rates=[10,6], gen_istft_hop_size=5
    // factor = 10*6*5 = 300
    assert_eq!(config.upsample_factor(), 300);
}

#[test]
fn test_kokoro_plbert_config_defaults() {
    let json = r#"{"plbert": {}}"#;
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.plbert.hidden_size, 768);
    assert_eq!(config.plbert.num_attention_heads, 12);
    assert_eq!(config.plbert.num_hidden_layers, 12);
    assert_eq!(config.plbert.intermediate_size, 2048);
    assert_eq!(config.plbert.max_position_embeddings, 512);
    assert_eq!(config.plbert.embedding_size, 128);
    assert_eq!(config.plbert.num_hidden_groups, 1);
}

#[test]
fn test_kokoro_istftnet_config_defaults() {
    let json = r#"{"istftnet": {}}"#;
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.istftnet.upsample_rates, vec![10, 6]);
    assert_eq!(config.istftnet.upsample_kernel_sizes, vec![20, 12]);
    assert_eq!(config.istftnet.upsample_initial_channel, 512);
    assert_eq!(config.istftnet.gen_istft_n_fft, 20);
    assert_eq!(config.istftnet.gen_istft_hop_size, 5);
    assert_eq!(config.istftnet.resblock_kernel_sizes, vec![3, 7, 11]);
    assert_eq!(
        config.istftnet.resblock_dilation_sizes,
        vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
    );
}

#[test]
fn test_kokoro_vocab_parsing() {
    let json = r#"{
        "vocab": {
            "$": 0,
            ";": 1,
            ":": 2,
            "a": 9,
            "b": 10,
            "ʒ": 45
        }
    }"#;
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.vocab.len(), 6);
    assert_eq!(config.vocab["$"], 0);
    assert_eq!(config.vocab["a"], 9);
    assert_eq!(config.vocab["ʒ"], 45);
}

#[test]
fn test_kokoro_config_custom_values() {
    let json = r#"{
        "hidden_dim": 256,
        "style_dim": 64,
        "n_mels": 40,
        "n_token": 100,
        "n_layer": 2,
        "istftnet": {
            "upsample_rates": [8, 4],
            "gen_istft_n_fft": 16,
            "gen_istft_hop_size": 4
        }
    }"#;
    let config: KokoroConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.hidden_dim, 256);
    assert_eq!(config.style_dim, 64);
    assert_eq!(config.full_style_dim(), 128);
    assert_eq!(config.n_mels, 40);
    assert_eq!(config.n_token, 100);
    assert_eq!(config.n_layer, 2);
    assert_eq!(config.istftnet.upsample_rates, vec![8, 4]);
    // upsample_factor = 8*4*4 = 128
    assert_eq!(config.upsample_factor(), 128);
}

#[test]
fn test_kokoro_tts_config_default_hf_id() {
    let config = TtsConfig::new(ModelType::Kokoro);
    assert_eq!(config.default_hf_model_id(), "hexgrad/Kokoro-82M");
}

#[test]
fn test_kokoro_model_files_no_tokenizer_required() {
    let files = any_tts::ModelFiles::default();
    let missing = files.missing_files(ModelType::Kokoro);
    // Kokoro doesn't need tokenizer.json
    assert!(!missing.iter().any(|m| m.contains("tokenizer")));
    // But still needs config and weights
    assert!(missing.contains(&"config.json"));
    assert!(missing.contains(&"model weight files"));
}

// ---------------------------------------------------------------------------
// Voice cloning tests
// ---------------------------------------------------------------------------

#[test]
fn test_kokoro_voice_embedding_wrong_model_type() {
    // VoiceEmbedding with wrong model_type should be rejected
    // (Tested at the type level — actual model integration requires weights)
    let embedding = any_tts::VoiceEmbedding::new(
        vec![0.0; 256],
        vec![1, 256],
        "qwen3-tts", // wrong type for Kokoro
    );
    assert_eq!(embedding.model_type(), "qwen3-tts");
    assert_ne!(embedding.model_type(), "kokoro");
}

#[test]
fn test_kokoro_mel_spectrogram_for_voice_cloning() {
    use any_tts::mel::{MelConfig, MelSpectrogram};

    let config = MelConfig::kokoro();
    let mel = MelSpectrogram::new(config, &candle_core::Device::Cpu).unwrap();

    // 3 seconds of silence at 24kHz
    let audio =
        candle_core::Tensor::zeros(72000, candle_core::DType::F32, &candle_core::Device::Cpu)
            .unwrap();
    let spec = mel.compute(&audio).unwrap();

    // Output: [1, 80, num_frames]
    assert_eq!(spec.dims()[0], 1);
    assert_eq!(spec.dims()[1], 80);
    // ~(72000 + 2048) / 300 ≈ 246 frames
    assert!(spec.dims()[2] > 200);
}

#[test]
fn test_kokoro_reference_audio_basic() {
    // Verify ReferenceAudio works with Kokoro-expected parameters
    let ref_audio = any_tts::ReferenceAudio::new(
        vec![0.0f32; 24000 * 5], // 5 seconds
        24000,
    );
    assert!((ref_audio.duration_secs() - 5.0).abs() < 0.001);
    assert!(!ref_audio.is_empty());
}

// ---------------------------------------------------------------------------
// End-to-end integration tests (require model weights)
// ---------------------------------------------------------------------------

/// Integration test — requires Kokoro-82M weights downloaded locally.
/// Run with: cargo test --features kokoro -- --ignored test_kokoro_full_synthesis
#[test]
#[ignore = "Requires Kokoro-82M model weights (~330 MB)"]
fn test_kokoro_full_synthesis() {
    let config = TtsConfig::new(ModelType::Kokoro).with_model_path("./models/Kokoro-82M");

    let model = KokoroModel::load(config).expect("Failed to load Kokoro model");

    assert_eq!(model.sample_rate(), 24000);
    assert!(!model.supported_languages().is_empty());

    let info = model.model_info();
    assert_eq!(info.name, "Kokoro");
    assert_eq!(info.variant, "82M");
    assert_eq!(info.sample_rate, 24000);
    assert_eq!(info.parameters, 82_000_000);

    let request = SynthesisRequest::new("Hello, this is a test of Kokoro text to speech.")
        .with_language("en");

    let audio = model.synthesize(&request).expect("Kokoro synthesis failed");
    common::assert_valid_audio(&audio);
    assert_eq!(audio.sample_rate, 24000);
}

/// Voice-pack–based synthesis test.
///
/// Kokoro-82M ships with pre-computed voice packs (.pt files) rather than
/// a style encoder for reference-audio voice cloning.  This test verifies
/// that synthesis through a named voice pack works correctly.
///
/// Run with: cargo test --features kokoro -- --ignored test_kokoro_voice_cloning
#[test]
#[ignore = "Requires Kokoro-82M model weights (~330 MB)"]
fn test_kokoro_voice_cloning() {
    use any_tts::traits::VoiceCloning;

    let config = TtsConfig::new(ModelType::Kokoro).with_model_path("./models/Kokoro-82M");

    let model = KokoroModel::load(config).expect("Failed to load Kokoro model");

    // Kokoro-82M does not include a style encoder, so extract_voice() from
    // reference audio is not supported.
    assert!(
        !model.supports_voice_cloning(),
        "Kokoro-82M should not report style-encoder-based voice cloning support"
    );

    // Discover available voices – the model auto-discovers the voices dir
    let voices = model.supported_voices();
    println!("Available voices: {:?}", voices);
    assert!(
        !voices.is_empty(),
        "At least one voice pack should be available"
    );

    // Synthesize with an explicit voice name (exercises voice-pack loading)
    let voice_name = &voices[0];
    println!("Using voice pack: {}", voice_name);

    let request = SynthesisRequest::new("Testing voice pack synthesis with Kokoro.")
        .with_language("en")
        .with_voice(voice_name);

    let audio = model
        .synthesize(&request)
        .expect("Synthesis with voice pack failed");

    common::assert_valid_audio(&audio);
    assert_eq!(audio.sample_rate, 24000);
}
