//! Generate sample WAV files from Qwen3-TTS-1.7B.
//!
//! Run with:
//!   cargo run --example generate_qwen3_tts --release
//!
//! Force CPU output into a dedicated folder:
//!   QWEN3_TTS_DEVICE=cpu QWEN3_TTS_OUTPUT_DIR=output/qwen3_tts/cpu cargo run --example generate_qwen3_tts --release
//!
//! Force Metal output into a dedicated folder:
//!   QWEN3_TTS_DEVICE=metal QWEN3_TTS_OUTPUT_DIR=output/qwen3_tts/metal cargo run --example generate_qwen3_tts --release --features metal
//!
//! Render one custom request with a named speaker:
//!   QWEN3_TTS_TEXT_FILE=output/demo.txt QWEN3_TTS_LANGUAGE=German QWEN3_TTS_SPEAKER=dylan \
//!   QWEN3_TTS_OUTPUT=output/qwen3_tts/demo_dylan.wav \
//!   cargo run --example generate_qwen3_tts --release --no-default-features --features qwen3-tts,download,metal
//!
//! ⚠ Requires ~4.5 GB of model weights. They will be downloaded from
//!   HuggingFace on first run if the `download` feature is enabled.
//!
//! Output goes to `output/qwen3_tts/` in the project root.

use any_tts::models::qwen3_tts::Qwen3TtsModel;
use any_tts::traits::TtsModel;
use any_tts::DeviceSelection;
use any_tts::{ModelType, SynthesisRequest, TtsConfig};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct CliArgs {
    device: DeviceSelection,
    output_dir: PathBuf,
    model_path: Option<String>,
    custom_request: Option<CustomRequest>,
}

#[derive(Debug)]
struct CustomRequest {
    text: String,
    language: String,
    speaker: Option<String>,
    instruct: Option<String>,
    output: PathBuf,
}

type SampleSpec = (&'static str, &'static str, &'static str, &'static str);

const SAMPLE_SPECS: &[SampleSpec] = &[
    (
        "English",
        "english_hello",
        "Hello! This is a test of the Qwen3 text to speech model, running entirely in Rust.",
        "ryan",
    ),
    (
        "German",
        "german_hallo",
        "Hallo! Dies ist ein Test der Qwen3 Sprachsynthese, vollständig in Rust implementiert.",
        "dylan",
    ),
    (
        "German",
        "german_long",
        "Die Entwicklung von Sprachsynthese-Systemen hat in den letzten Jahren enorme \
         Fortschritte gemacht. Neuronale Netzwerke ermöglichen eine natürlich klingende \
         Ausgabe, die kaum von menschlicher Sprache zu unterscheiden ist.",
        "dylan",
    ),
    (
        "Chinese",
        "chinese_nihao",
        "你好！这是Qwen3文本转语音模型的测试，完全用Rust实现。",
        "dylan",
    ),
    (
        "Japanese",
        "japanese_konnichiwa",
        "こんにちは！これはQwen3テキスト読み上げモデルのテストです。",
        "dylan",
    ),
    (
        "Korean",
        "korean_annyeong",
        "안녕하세요! 이것은 Qwen3 텍스트 음성 변환 모델의 테스트입니다.",
        "dylan",
    ),
];

fn load_options() -> CliArgs {
    let device = env::var("QWEN3_TTS_DEVICE")
        .ok()
        .map(|value| parse_device(&value))
        .unwrap_or(DeviceSelection::Auto);
    let output_dir = env::var("QWEN3_TTS_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("output/qwen3_tts"));
    let model_path = env::var("QWEN3_TTS_MODEL_PATH").ok();
    let custom_text = load_custom_text();
    let custom_language = env_override("QWEN3_TTS_LANGUAGE");
    let custom_speaker = env_override("QWEN3_TTS_SPEAKER");
    let custom_instruct = env_override("QWEN3_TTS_INSTRUCT");
    let custom_output = env_override("QWEN3_TTS_OUTPUT").map(PathBuf::from);

    let custom_request = if custom_text.is_some()
        || custom_language.is_some()
        || custom_speaker.is_some()
        || custom_instruct.is_some()
        || custom_output.is_some()
    {
        Some(CustomRequest {
            text: custom_text.unwrap_or_else(|| {
                panic!("Set QWEN3_TTS_TEXT_FILE or QWEN3_TTS_TEXT to use custom request mode")
            }),
            language: custom_language.unwrap_or_else(|| "auto".to_string()),
            speaker: custom_speaker,
            instruct: custom_instruct,
            output: custom_output.unwrap_or_else(|| output_dir.join("qwen3tts_custom.wav")),
        })
    } else {
        None
    };

    CliArgs {
        device,
        output_dir,
        model_path,
        custom_request,
    }
}

fn load_custom_text() -> Option<String> {
    if let Some(text_path) = env_override("QWEN3_TTS_TEXT_FILE") {
        return Some(
            fs::read_to_string(&text_path)
                .unwrap_or_else(|err| panic!("Failed to read {}: {err}", text_path)),
        );
    }

    env_override("QWEN3_TTS_TEXT")
}

fn env_override(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_device(value: &str) -> DeviceSelection {
    if value.eq_ignore_ascii_case("auto") {
        return DeviceSelection::Auto;
    }
    if value.eq_ignore_ascii_case("cpu") {
        return DeviceSelection::Cpu;
    }
    if let Some(ordinal) = value.strip_prefix("metal:") {
        return DeviceSelection::Metal(
            ordinal
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Invalid metal device ordinal: {value}")),
        );
    }
    if value.eq_ignore_ascii_case("metal") {
        return DeviceSelection::Metal(0);
    }
    if let Some(ordinal) = value.strip_prefix("cuda:") {
        return DeviceSelection::Cuda(
            ordinal
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Invalid cuda device ordinal: {value}")),
        );
    }
    if value.eq_ignore_ascii_case("cuda") {
        return DeviceSelection::Cuda(0);
    }

    panic!("Unsupported device '{value}'. Expected auto, cpu, metal[:N], or cuda[:N]");
}

fn load_model(args: &CliArgs) -> Qwen3TtsModel {
    let mut config = TtsConfig::new(ModelType::Qwen3Tts).with_device(args.device);
    if let Some(model_path) = args.model_path.as_deref() {
        config = config.with_model_path(model_path);
    }

    Qwen3TtsModel::load(config).expect("Failed to load model")
}

fn print_model_summary(model: &Qwen3TtsModel, output_dir: &Path) {
    let info = model.model_info();
    println!("  Model       : {}", info.name);
    println!("  Device      : {:?}", model.device());
    println!("  Sample rate : {} Hz", model.sample_rate());
    println!("  Voices      : {:?}", model.supported_voices());
    println!("  Languages   : {:?}", model.supported_languages());
    println!("  Output dir  : {}", output_dir.display());
    println!();
}

fn render_samples(model: &Qwen3TtsModel, output_dir: &Path) {
    // let voices = model.supported_voices();

    for (lang, name, text, speaker) in SAMPLE_SPECS {
        let stem = format!("qwen3tts_{name}");
        println!("▸ [{lang}] {stem}");
        println!("  \"{text}\"");

        let mut request = SynthesisRequest::new(*text).with_language(*lang);
        request = request.with_voice(speaker.to_string());

        match model.synthesize(&request) {
            Ok(audio) => {
                println!(
                    "  {:.2}s  ({} samples @ {} Hz)",
                    audio.duration_secs(),
                    audio.len(),
                    audio.sample_rate
                );

                let wav_path = output_dir.join(format!("{stem}.wav"));
                audio.save_wav(&wav_path).expect("Failed to write WAV");
                println!("  ✓ {}", wav_path.display());
            }
            Err(e) => eprintln!("  ✗ {e}"),
        }
        println!();
    }
}

fn render_custom_request(model: &Qwen3TtsModel, request: &CustomRequest) {
    println!("▸ [Custom] qwen3tts_custom");
    if let Some(speaker) = request.speaker.as_deref() {
        println!("  Speaker  : {}", speaker);
    }
    println!("  Language : {}", request.language);
    if let Some(instruct) = request.instruct.as_deref() {
        println!("  Instruct : {}", instruct);
    }

    let mut synthesis_request =
        SynthesisRequest::new(&request.text).with_language(&request.language);
    if let Some(speaker) = request.speaker.as_deref() {
        synthesis_request = synthesis_request.with_voice(speaker);
    }
    if let Some(instruct) = request.instruct.as_deref() {
        synthesis_request = synthesis_request.with_instruct(instruct);
    }

    let audio = model
        .synthesize(&synthesis_request)
        .expect("Failed to synthesize custom Qwen3-TTS request");

    if let Some(parent) = request
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("Failed to create {}: {err}", parent.display()));
    }
    audio
        .save_wav(&request.output)
        .expect("Failed to write custom Qwen3-TTS WAV");

    println!(
        "  {:.2}s  ({} samples @ {} Hz)",
        audio.duration_secs(),
        audio.len(),
        audio.sample_rate
    );
    println!("  ✓ {}", request.output.display());
    println!();
}

fn main() {
    let args = load_options();
    fs::create_dir_all(&args.output_dir)
        .unwrap_or_else(|err| panic!("Failed to create {}: {err}", args.output_dir.display()));

    println!("═══════════════════════════════════════════════════════");
    println!("  Qwen3-TTS-1.7B  —  Sample Audio Generation");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("Loading Qwen3-TTS-1.7B (this may take a while) …");

    let model = load_model(&args);
    print_model_summary(&model, &args.output_dir);

    if let Some(custom_request) = args.custom_request.as_ref() {
        render_custom_request(&model, custom_request);
    } else {
        render_samples(&model, &args.output_dir);
    }

    println!("Done! Check {}", args.output_dir.display());
}
