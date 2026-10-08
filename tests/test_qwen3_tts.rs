//! Tests for the Qwen3-TTS model backend.

mod common;

use any_tts::models::qwen3_tts::Qwen3TtsModel;
use any_tts::traits::TtsModel;
use any_tts::{ModelType, SynthesisRequest, TtsConfig};

#[test]
#[cfg_attr(
    feature = "download",
    ignore = "Missing-path fallback can download weights; run without the download feature"
)]
fn test_qwen3tts_load_missing_path() {
    let config = TtsConfig::new(ModelType::Qwen3Tts);
    let result = Qwen3TtsModel::load(config);
    // With cached weights and the default `download` feature, loading may
    // succeed on developer machines. Only validate the error text if it fails.
    if let Err(err) = result {
        let err = err.to_string();
        assert!(
            err.contains("missing")
                || err.contains("Missing")
                || err.contains("not specified")
                || err.contains("Failed"),
            "Unexpected error: {}",
            err
        );
    }
}

#[test]
fn test_qwen3tts_config_parsing() {
    use any_tts::models::qwen3_tts::config::Qwen3TtsConfig;

    let json = r#"{
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "tts_pad_token_id": 151671,
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "talker_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 8960,
            "vocab_size": 2048,
            "text_vocab_size": 152064,
            "text_hidden_size": 1536,
            "num_code_groups": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "spk_id": {
                "Vivian": 0,
                "Serena": 1,
                "Ryan": 2,
                "Aiden": 3
            },
            "codec_language_id": {
                "Chinese": 0,
                "English": 1,
                "Japanese": 2,
                "Korean": 3,
                "German": 4,
                "French": 5,
                "auto": 6
            }
        }
    }"#;

    let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.model_type, "qwen3_tts");
    assert_eq!(config.tts_model_type, "custom_voice");
    assert_eq!(config.talker_config.hidden_size, 1536);
    assert_eq!(config.talker_config.num_code_groups, 16);
    assert_eq!(config.talker_config.spk_id.len(), 4);
    assert_eq!(config.talker_config.spk_id["Ryan"], 2);
    assert_eq!(config.tts_bos_token_id, 151672);

    let speakers = config.speakers();
    assert!(speakers.contains(&"Vivian".to_string()));
    assert!(speakers.contains(&"Ryan".to_string()));

    let languages = config.languages();
    assert!(languages.contains(&"English".to_string()));
    assert!(languages.contains(&"Chinese".to_string()));
}

#[test]
fn test_qwen3tts_config_with_code_predictor() {
    use any_tts::models::qwen3_tts::config::Qwen3TtsConfig;

    let json = r#"{
        "talker_config": {
            "code_predictor_config": {
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 2048,
                "num_code_groups": 16
            }
        }
    }"#;

    let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
    let cp = config.talker_config.code_predictor_config.unwrap();
    assert_eq!(cp.hidden_size, 512);
    assert_eq!(cp.num_hidden_layers, 4);
    assert_eq!(cp.num_code_groups, 16);
}

#[test]
fn test_qwen3tts_language_dialect_filtering() {
    use any_tts::models::qwen3_tts::config::Qwen3TtsConfig;

    let json = r#"{
        "talker_config": {
            "codec_language_id": {
                "Chinese": 0,
                "English": 1,
                "Chinese_dialect_sichuan": 10,
                "Chinese_dialect_beijing": 11
            }
        }
    }"#;

    let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
    let languages = config.languages();

    // Should include Chinese and English but not dialect variants
    assert!(languages.contains(&"Chinese".to_string()));
    assert!(languages.contains(&"English".to_string()));
    assert!(!languages.iter().any(|l| l.contains("dialect")));
    assert_eq!(languages.len(), 2);
}

/// Integration test — requires model weights downloaded locally.
/// Run with: cargo test --features qwen3-tts -- --ignored test_qwen3tts_full
#[test]
#[ignore = "Requires Qwen3-TTS model weights (~4.5 GB)"]
fn test_qwen3tts_full_synthesis() {
    let config = TtsConfig::new(ModelType::Qwen3Tts)
        .with_model_path("./models/Qwen3-TTS-12Hz-1.7B-CustomVoice");

    let model = Qwen3TtsModel::load(config).expect("Failed to load model");

    assert_eq!(model.sample_rate(), 24000);
    assert!(model.files().config.is_some());

    let info = model.model_info();
    assert!(info.name.contains("Qwen3"));
    assert_eq!(info.sample_rate, 24000);

    let request = SynthesisRequest::new("Hello, this is a test.")
        .with_language("English")
        .with_voice("Ryan");

    let audio = model.synthesize(&request).expect("Synthesis failed");
    common::assert_valid_audio(&audio);
    assert_eq!(audio.sample_rate, 24000);
}

/// Integration test for multi-language synthesis.
#[test]
#[ignore = "Requires Qwen3-TTS model weights (~4.5 GB)"]
fn test_qwen3tts_multilingual() {
    let config = TtsConfig::new(ModelType::Qwen3Tts)
        .with_model_path("./models/Qwen3-TTS-12Hz-1.7B-CustomVoice");

    let model = Qwen3TtsModel::load(config).expect("Failed to load model");

    let test_cases = vec![
        ("Hello world!", "English", "Ryan"),
        ("你好世界", "Chinese", "Vivian"),
        ("こんにちは世界", "Japanese", "Ono_Anna"),
        ("Hallo Welt!", "German", "Ryan"),
    ];

    for (text, lang, voice) in test_cases {
        let request = SynthesisRequest::new(text)
            .with_language(lang)
            .with_voice(voice);

        let audio = model
            .synthesize(&request)
            .unwrap_or_else(|_| panic!("Synthesis failed for text='{}' lang='{}'", text, lang));
        common::assert_valid_audio(&audio);
    }
}
