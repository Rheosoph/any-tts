//! Generate a sample WAV file with the native OmniVoice backend.
//!
//! Run with:
//!   cargo run --example generate_omnivoice --release --no-default-features --features omnivoice,download
//!
//! Add `metal` on Apple builds or `cuda` on NVIDIA builds to enable faster
//! backends; this example will pick the best compiled and available backend
//! with CPU fallback.

use any_tts::models::omnivoice::preferred_runtime_choice;
use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct OmniVoiceExampleConfig {
    text: String,
    language: String,
    instruct: String,
    cfg_scale: f64,
    output: String,
}

fn load_example_config() -> OmniVoiceExampleConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/omnivoice_example.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse {}: {err}", path.display()))
}

fn load_runtime_config() -> OmniVoiceExampleConfig {
    let mut example = load_example_config();

    if let Some(text_path) = env_override("OMNIVOICE_TEXT_FILE") {
        example.text = fs::read_to_string(&text_path)
            .unwrap_or_else(|err| panic!("Failed to read {}: {err}", text_path));
    }
    if let Some(text) = env_override("OMNIVOICE_TEXT") {
        example.text = text;
    }
    if let Some(language) = env_override("OMNIVOICE_LANGUAGE") {
        example.language = language;
    }
    if let Some(instruct) = env_override("OMNIVOICE_INSTRUCT") {
        example.instruct = instruct;
    }
    if let Some(cfg_scale) = env_override("OMNIVOICE_CFG_SCALE") {
        example.cfg_scale = cfg_scale
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("Invalid OMNIVOICE_CFG_SCALE value: {cfg_scale}"));
    }
    if let Some(output) = env_override("OMNIVOICE_OUTPUT") {
        example.output = output;
    }

    example
}

fn env_override(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn main() {
    let example = load_runtime_config();
    let runtime = preferred_runtime_choice();
    let model = load_model(
        TtsConfig::new(ModelType::OmniVoice)
            .with_device(runtime.device)
            .with_dtype(runtime.dtype),
    )
    .expect("Failed to initialize OmniVoice");

    let request = SynthesisRequest::new(&example.text)
        .with_language(&example.language)
        .with_instruct(&example.instruct)
        .with_cfg_scale(example.cfg_scale);

    let audio = model
        .synthesize(&request)
        .expect("OmniVoice synthesis failed");

    let output_path = PathBuf::from(&example.output);
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("Failed to create {}: {err}", parent.display()));
    }
    audio
        .save_wav(Path::new(&example.output))
        .expect("Failed to write WAV");

    println!("  Language   : {}", example.language);
    println!("  Instruct   : {}", example.instruct);
    println!("  Text chars : {}", example.text.chars().count());
    println!(
        "Saved OmniVoice sample to {} using {} ({} samples @ {} Hz)",
        example.output,
        runtime.label(),
        audio.len(),
        audio.sample_rate
    );
}
