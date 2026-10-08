//! Tests for the OmniVoice model backend.

#![cfg(all(feature = "omnivoice", feature = "download"))]

mod common;

use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};

#[test]
#[cfg(feature = "download")]
#[ignore = "Requires OmniVoice model weights or Hugging Face download access"]
fn test_omnivoice_smoke_synthesis() {
    let model =
        load_model(TtsConfig::new(ModelType::OmniVoice)).expect("Failed to load OmniVoice model");

    assert_eq!(model.sample_rate(), 24000);

    let request = SynthesisRequest::new("Hello! This is a short English OmniVoice smoke test.")
        .with_language("English")
        .with_instruct("female, moderate pitch, american accent")
        .with_cfg_scale(2.0)
        .with_max_tokens(24);

    let audio = model
        .synthesize(&request)
        .expect("OmniVoice smoke synthesis failed");

    common::assert_valid_audio(&audio);
    assert_eq!(audio.sample_rate, 24000);
    assert!(
        audio.duration_secs() < 5.0,
        "Smoke output should stay short, got {}s",
        audio.duration_secs()
    );
}
