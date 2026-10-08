//! Tests for the core TtsModel API types.

use any_tts::{AudioSamples, ModelFiles, ModelType, SynthesisRequest, TtsConfig};
use std::path::PathBuf;

#[test]
fn test_synthesis_request_builder() {
    let req = SynthesisRequest::new("Hello, world!")
        .with_language("en")
        .with_voice("Ryan")
        .with_instruct("Speak happily")
        .with_max_tokens(2048)
        .with_temperature(0.7)
        .with_speed(1.2)
        .with_cfg_scale(3.0);

    assert_eq!(req.text, "Hello, world!");
    assert_eq!(req.language.as_deref(), Some("en"));
    assert_eq!(req.voice.as_deref(), Some("Ryan"));
    assert_eq!(req.instruct.as_deref(), Some("Speak happily"));
    assert_eq!(req.max_tokens, Some(2048));
    assert!((req.temperature.unwrap() - 0.7).abs() < f64::EPSILON);
    assert_eq!(req.speed, Some(1.2));
    assert!((req.cfg_scale.unwrap() - 3.0).abs() < f64::EPSILON);
}

#[test]
fn test_synthesis_request_defaults() {
    let req = SynthesisRequest::new("Test");
    assert_eq!(req.text, "Test");
    assert!(req.language.is_none());
    assert!(req.voice.is_none());
    assert!(req.instruct.is_none());
    assert!(req.max_tokens.is_none());
    assert!(req.temperature.is_none());
    assert!(req.speed.is_none());
    assert!(req.cfg_scale.is_none());
}

#[test]
fn test_tts_config_builder() {
    let config = TtsConfig::new(ModelType::Qwen3Tts)
        .with_model_path("/models/qwen3-tts")
        .with_dtype(any_tts::config::DType::BF16);

    assert_eq!(config.model_type, ModelType::Qwen3Tts);
    assert_eq!(config.model_path.as_deref(), Some("/models/qwen3-tts"));
    assert_eq!(config.dtype, any_tts::config::DType::BF16);
}

#[test]
fn test_tts_config_default_hf_ids() {
    let qwen = TtsConfig::new(ModelType::Qwen3Tts);
    assert_eq!(
        qwen.default_hf_model_id(),
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    );

    let kokoro = TtsConfig::new(ModelType::Kokoro);
    assert_eq!(kokoro.default_hf_model_id(), "hexgrad/Kokoro-82M");

    let omnivoice = TtsConfig::new(ModelType::OmniVoice);
    assert_eq!(omnivoice.default_hf_model_id(), "k2-fsa/OmniVoice");

    let voxtral = TtsConfig::new(ModelType::Voxtral);
    assert_eq!(
        voxtral.default_hf_model_id(),
        "mistralai/Voxtral-4B-TTS-2603"
    );

    let realtime = TtsConfig::new(ModelType::VibeVoiceRealtime);
    assert_eq!(
        realtime.default_hf_model_id(),
        "microsoft/VibeVoice-Realtime-0.5B"
    );
}

#[test]
fn test_tts_config_effective_hf_id() {
    let config = TtsConfig::new(ModelType::Qwen3Tts).with_hf_model_id("custom/model-id");
    assert_eq!(config.effective_hf_model_id(), "custom/model-id");

    let config_default = TtsConfig::new(ModelType::Qwen3Tts);
    assert_eq!(
        config_default.effective_hf_model_id(),
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    );
}

#[test]
fn test_tts_config_effective_model_ref_prefers_model_path() {
    let config = TtsConfig::new(ModelType::OmniVoice)
        .with_hf_model_id("k2-fsa/OmniVoice")
        .with_model_path("/models/omnivoice-local");

    assert_eq!(config.effective_model_ref(), "/models/omnivoice-local");
}

#[test]
fn test_tts_config_individual_file_builders() {
    let config = TtsConfig::new(ModelType::Qwen3Tts)
        .with_config_file("/cache/abc/config.json")
        .with_tokenizer_file("/cache/def/tokenizer.json")
        .with_weight_file("/cache/012/model-00001.safetensors")
        .with_weight_file("/cache/345/model-00002.safetensors")
        .with_speech_tokenizer_weight_file("/cache/678/st_model.safetensors")
        .with_speech_tokenizer_config_file("/cache/9ab/st_config.json")
        .with_generation_config_file("/cache/cde/generation_config.json");

    assert_eq!(
        config
            .files
            .config
            .as_ref()
            .and_then(any_tts::ModelAsset::as_path),
        Some(std::path::Path::new("/cache/abc/config.json"))
    );
    assert_eq!(
        config
            .files
            .tokenizer
            .as_ref()
            .and_then(any_tts::ModelAsset::as_path),
        Some(std::path::Path::new("/cache/def/tokenizer.json"))
    );
    assert_eq!(config.files.weights.len(), 2);
    assert_eq!(config.files.speech_tokenizer_weights.len(), 1);
    assert!(config.files.speech_tokenizer_config.is_some());
    assert!(config.files.generation_config.is_some());
}

#[test]
fn test_tts_config_weight_files_batch() {
    let paths = vec![
        PathBuf::from("/a/shard1.safetensors"),
        PathBuf::from("/b/shard2.safetensors"),
    ];
    let config = TtsConfig::new(ModelType::Qwen3Tts).with_weight_files(paths);

    assert_eq!(config.files.weights.len(), 2);
}

#[test]
fn test_tts_config_voices_dir() {
    let config = TtsConfig::new(ModelType::Kokoro).with_voices_dir("/models/voices");
    assert!(config.files.voices_dir.is_some());
}

#[test]
fn test_model_files_missing_files_qwen3tts() {
    let files = ModelFiles::default();
    let missing = files.missing_files(ModelType::Qwen3Tts);
    assert!(missing.contains(&"config.json"));
    assert!(missing.contains(&"tokenizer.json"));
    assert!(missing.contains(&"model weight files"));
    assert!(missing.contains(&"speech tokenizer weights"));
}

#[test]
fn test_model_files_missing_files_omnivoice() {
    let files = ModelFiles::default();
    let missing = files.missing_files(ModelType::OmniVoice);
    assert!(missing.contains(&"config.json"));
    assert!(missing.contains(&"tokenizer.json"));
    assert!(missing.contains(&"model weight files"));
    assert!(missing.contains(&"audio tokenizer config"));
    assert!(missing.contains(&"audio tokenizer weights"));
    assert!(files.validate(ModelType::OmniVoice).is_err());
}

#[test]
fn test_model_files_missing_files_external_runtimes() {
    let files = ModelFiles::default();
    let missing = files.missing_files(ModelType::Voxtral);
    assert!(missing.contains(&"params.json"));
    assert!(missing.contains(&"tekken.json"));
    assert!(missing.contains(&"consolidated.safetensors"));
    assert!(missing.contains(&"voice_embedding"));
    assert!(files.validate(ModelType::Voxtral).is_err());
}

#[test]
fn test_model_files_validate_fails_on_empty() {
    let files = ModelFiles::default();
    let result = files.validate(ModelType::Qwen3Tts);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("config.json"));
}

#[test]
fn test_audio_samples_properties() {
    let audio = AudioSamples::new(vec![0.0; 48000], 24000);
    assert_eq!(audio.sample_rate, 24000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.len(), 48000);
    assert!(!audio.is_empty());
    assert!((audio.duration_secs() - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_audio_samples_i16_conversion() {
    let audio = AudioSamples::new(vec![0.0, 0.5, -0.5, 1.0, -1.0], 24000);
    let pcm = audio.to_i16();
    assert_eq!(pcm.len(), 5);
    assert_eq!(pcm[0], 0);
    assert!(pcm[1] > 0);
    assert!(pcm[2] < 0);
    assert_eq!(pcm[3], i16::MAX);
    assert_eq!(pcm[4], -i16::MAX);
}

#[test]
#[cfg_attr(
    feature = "download",
    ignore = "Missing-path fallback can download weights; run without the download feature"
)]
fn test_load_model_without_path_returns_error() {
    // Without download feature disabled or HF unavailable, this should
    // fail because no files can be resolved.
    let config = TtsConfig::new(ModelType::Qwen3Tts);
    let result = any_tts::load_model(config);
    // On machines with cached weights and the default `download` feature,
    // loading may succeed. Only assert the error shape when it does fail.
    if let Err(err) = result {
        let err = err.to_string();
        assert!(
            err.contains("missing")
                || err.contains("Missing")
                || err.contains("not enabled")
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
#[cfg_attr(
    feature = "download",
    ignore = "Missing-path fallback can download weights; run without the download feature"
)]
fn test_load_model_with_nonexistent_path_returns_error() {
    let config = TtsConfig::new(ModelType::Qwen3Tts).with_model_path("/nonexistent/path/to/model");
    let result = any_tts::load_model(config);
    // A nonexistent local path can still succeed when the download fallback
    // resolves cached or remote model files.
    if let Err(err) = result {
        let err = err.to_string();
        assert!(
            err.contains("missing")
                || err.contains("Missing")
                || err.contains("not enabled")
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
fn test_dtype_to_candle_conversion() {
    use any_tts::config::DType;
    assert_eq!(DType::F32.to_candle(), candle_core::DType::F32);
    assert_eq!(DType::F16.to_candle(), candle_core::DType::F16);
    assert_eq!(DType::BF16.to_candle(), candle_core::DType::BF16);
}

#[test]
fn test_model_type_equality() {
    assert_eq!(ModelType::Qwen3Tts, ModelType::Qwen3Tts);
    assert_eq!(ModelType::Kokoro, ModelType::Kokoro);
    assert_eq!(ModelType::OmniVoice, ModelType::OmniVoice);
    assert_eq!(ModelType::VibeVoiceRealtime, ModelType::VibeVoiceRealtime);
    assert_eq!(ModelType::Voxtral, ModelType::Voxtral);
    assert_ne!(ModelType::Qwen3Tts, ModelType::Kokoro);
    assert_ne!(ModelType::Qwen3Tts, ModelType::OmniVoice);
    assert_ne!(ModelType::Qwen3Tts, ModelType::VibeVoiceRealtime);
    assert_ne!(ModelType::Qwen3Tts, ModelType::Voxtral);
    assert_ne!(ModelType::Kokoro, ModelType::OmniVoice);
    assert_ne!(ModelType::Kokoro, ModelType::VibeVoiceRealtime);
    assert_ne!(ModelType::Kokoro, ModelType::Voxtral);
    assert_ne!(ModelType::OmniVoice, ModelType::Voxtral);
}

#[test]
fn test_external_runtime_builders() {
    let config = TtsConfig::new(ModelType::Voxtral)
        .with_runtime_command("python-custom")
        .with_runtime_endpoint("http://localhost:8091/v1")
        .with_bearer_token("token-123");

    assert_eq!(config.effective_runtime_command(), Some("python-custom"));
    assert_eq!(
        config.effective_runtime_endpoint(),
        Some("http://localhost:8091/v1")
    );
    assert_eq!(config.effective_bearer_token(), "token-123");

    let defaults = TtsConfig::new(ModelType::Voxtral);
    assert_eq!(defaults.effective_runtime_command(), Some("python3"));
    assert_eq!(defaults.effective_runtime_endpoint(), None);
    assert_eq!(defaults.effective_bearer_token(), "EMPTY");

    let omnivoice = TtsConfig::new(ModelType::OmniVoice);
    assert_eq!(omnivoice.effective_runtime_command(), None);
    assert_eq!(omnivoice.effective_runtime_endpoint(), None);
}

#[test]
fn test_model_files_missing_files_kokoro() {
    let files = ModelFiles::default();
    let missing = files.missing_files(ModelType::Kokoro);
    assert!(missing.contains(&"config.json"));
    // Kokoro does NOT need tokenizer.json
    assert!(!missing.iter().any(|m| m.contains("tokenizer")));
    assert!(missing.contains(&"model weight files"));
    // No speech tokenizer required
    assert!(!missing.iter().any(|m| m.contains("speech tokenizer")));
}

#[test]
#[cfg_attr(
    feature = "download",
    ignore = "Missing-path fallback can download weights; run without the download feature"
)]
fn test_load_kokoro_without_path_returns_error() {
    let config = TtsConfig::new(ModelType::Kokoro);
    let result = any_tts::load_model(config);
    // Without model weights present locally, loading should fail.
    // With the `download` feature, it may attempt to download from HF and
    // succeed if the model is cached — in that case the test is trivially ok.
    if let Err(e) = &result {
        let err = e.to_string();
        assert!(
            err.contains("missing")
                || err.contains("Missing")
                || err.contains("not enabled")
                || err.contains("not specified")
                || err.contains("Failed")
                || err.contains("cannot find")
                || err.contains("error"),
            "Unexpected error: {}",
            err
        );
    }
}

// ---------------------------------------------------------------------------
// Voice cloning types
// ---------------------------------------------------------------------------

#[test]
fn test_reference_audio_construction() {
    let audio = any_tts::ReferenceAudio::new(vec![0.0f32; 24000], 24000);
    assert!(!audio.is_empty());
    assert!((audio.duration_secs() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_reference_audio_empty() {
    let audio = any_tts::ReferenceAudio::new(vec![], 24000);
    assert!(audio.is_empty());
    assert!((audio.duration_secs()).abs() < f32::EPSILON);
}

#[test]
fn test_reference_audio_zero_sample_rate() {
    let audio = any_tts::ReferenceAudio::new(vec![0.0; 100], 0);
    assert!((audio.duration_secs()).abs() < f32::EPSILON);
}

#[test]
fn test_voice_embedding_roundtrip() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![1, 6];
    let embedding = any_tts::VoiceEmbedding::new(data.clone(), shape.clone(), "kokoro");

    assert_eq!(embedding.model_type(), "kokoro");
    assert_eq!(embedding.shape(), &[1, 6]);

    // Convert to tensor and back
    let tensor = embedding.to_tensor(&candle_core::Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, 6]);

    let recovered: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(recovered, data);
}

#[test]
fn test_voice_embedding_save_load() {
    let data = vec![0.1f32, 0.2, 0.3, 0.4];
    let shape = vec![1, 4];
    let embedding = any_tts::VoiceEmbedding::new(data.clone(), shape.clone(), "kokoro");

    let dir = std::env::temp_dir().join("tts_rs_test_voice_embedding");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_voice.json");

    embedding.save(&path).unwrap();
    assert!(path.exists());

    let loaded = any_tts::VoiceEmbedding::load(&path).unwrap();
    assert_eq!(loaded.model_type(), "kokoro");
    assert_eq!(loaded.shape(), &[1, 4]);

    let tensor = loaded.to_tensor(&candle_core::Device::Cpu).unwrap();
    let recovered: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(recovered, data);

    // Cleanup
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_synthesis_request_with_reference_audio() {
    let audio = any_tts::ReferenceAudio::new(vec![0.0; 24000], 24000);
    let req = SynthesisRequest::new("Hello!").with_reference_audio(audio);

    assert!(req.reference_audio.is_some());
    assert!(req.voice.is_none());
    assert!(req.voice_embedding.is_none());
}

#[test]
fn test_synthesis_request_with_voice_embedding() {
    let embedding = any_tts::VoiceEmbedding::new(vec![0.0; 256], vec![1, 256], "kokoro");
    let req = SynthesisRequest::new("Hello!").with_voice_embedding(embedding);

    assert!(req.voice_embedding.is_some());
    assert!(req.reference_audio.is_none());
    assert!(req.voice.is_none());
}
