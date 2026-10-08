//! Local-asset latency benchmark, including time to first audio and warm runs.
//!
//! cargo run --release --example benchmark_tts -- --model kokoro \
//!   --model-path ./models/Kokoro-82M --device cpu --iterations 5
//! Add --phonemizer-only to measure Kokoro preprocessing without loading weights.

use std::time::Instant;

use any_tts::{load_model, DeviceSelection, ModelType, SynthesisRequest, TtsConfig};
use serde_json::{json, Value};

fn option(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == key)
        .map(|pair| pair[1].clone())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let key = option(&args, "--model").unwrap_or_else(|| "kokoro".into());
    let model_type = match key.as_str() {
        "kokoro" => ModelType::Kokoro,
        "omnivoice" => ModelType::OmniVoice,
        "qwen3-tts" => ModelType::Qwen3Tts,
        "vibevoice" => ModelType::VibeVoice,
        "vibevoice-realtime" => ModelType::VibeVoiceRealtime,
        "voxtral" => ModelType::Voxtral,
        _ => anyhow::bail!("unknown model: {key}"),
    };
    let model_path = option(&args, "--model-path")
        .ok_or_else(|| anyhow::anyhow!("--model-path is required; benchmarks use local assets"))?;
    let device = match option(&args, "--device").as_deref().unwrap_or("cpu") {
        "cpu" => DeviceSelection::Cpu,
        "metal" => DeviceSelection::Metal(0),
        "cuda" => DeviceSelection::Cuda(0),
        "auto" => DeviceSelection::Auto,
        value => anyhow::bail!("unknown device: {value}"),
    };
    let iterations: usize = option(&args, "--iterations")
        .unwrap_or_else(|| "5".into())
        .parse()?;
    anyhow::ensure!(iterations > 0, "--iterations must be positive");
    let text = option(&args, "--text").unwrap_or_else(|| {
        "I read a book yesterday. They record a record. The total is $12.50.".into()
    });
    let language = option(&args, "--language").unwrap_or_else(|| "en".into());
    let mut request = SynthesisRequest::new(&text).with_language(&language);
    if let Some(voice) = option(&args, "--voice") {
        request = request.with_voice(voice);
    }
    if let Some(tokens) = option(&args, "--max-tokens") {
        request = request.with_max_tokens(tokens.parse()?);
    }
    if let Some(temperature) = option(&args, "--temperature") {
        request = request.with_temperature(temperature.parse()?);
    }
    if let Some(speed) = option(&args, "--speed") {
        request = request.with_speed(speed.parse()?);
    }

    let mut result = json!({
        "model": key, "device": device.label(), "model_path": model_path,
        "text": text, "language": language, "iterations": iterations,
        "voice": request.voice, "temperature": request.temperature,
        "speed": request.speed, "max_tokens": request.max_tokens,
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
    });
    if model_type == ModelType::Kokoro {
        result["phonemizer"] = benchmark_phonemizer(&model_path, &request, iterations)?;
    }
    if args.iter().any(|arg| arg == "--phonemizer-only") {
        anyhow::ensure!(
            model_type == ModelType::Kokoro,
            "--phonemizer-only requires Kokoro"
        );
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    let started = Instant::now();
    let config = TtsConfig::new(model_type)
        .with_model_path(&model_path)
        .with_device(device);
    // Do not silently benchmark downloads as model load time.
    let mut local_files = config.files.clone();
    local_files.fill_from_directory(std::path::Path::new(&model_path));
    local_files.validate(model_type)?;
    let model = load_model(config)?;
    result["load_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    result["incremental_audio"] = json!(model.supports_streaming());
    let mut warm = Vec::new();
    for index in 0..=iterations {
        let started = Instant::now();
        let mut first_audio_ms = None;
        let mut samples = 0usize;
        let mut chunks = 0usize;
        model.synthesize_stream(&request, &mut |chunk| {
            if !chunk.samples.is_empty() {
                first_audio_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
                samples += chunk.samples.len();
                chunks += 1;
            }
            Ok(())
        })?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let duration = samples as f64 / model.sample_rate() as f64;
        let run = json!({
            "total_ms": elapsed_ms, "first_audio_ms": first_audio_ms,
            "audio_seconds": duration, "chunks": chunks,
            "realtime_factor": if duration > 0.0 { Some(elapsed_ms / 1000.0 / duration) } else { None },
        });
        if index == 0 {
            result["first_synthesis"] = run;
        } else {
            warm.push(run);
        }
    }
    result["warm_median_ms"] = json!(percentile(&warm, "total_ms", 0.5));
    result["warm_p95_ms"] = json!(percentile(&warm, "total_ms", 0.95));
    result["warm_first_audio_median_ms"] = json!(percentile(&warm, "first_audio_ms", 0.5));
    result["warm_runs"] = json!(warm);
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn percentile(runs: &[Value], field: &str, quantile: f64) -> Option<f64> {
    let mut values: Vec<f64> = runs.iter().filter_map(|run| run[field].as_f64()).collect();
    values.sort_by(f64::total_cmp);
    let index = (values.len() as f64 * quantile).ceil() as usize;
    values.get(index.saturating_sub(1)).copied()
}

#[cfg(feature = "kokoro")]
fn benchmark_phonemizer(
    path: &str,
    request: &SynthesisRequest,
    iterations: usize,
) -> anyhow::Result<Value> {
    use any_tts::models::kokoro::{config::KokoroConfig, phonemizer};
    let bytes = std::fs::read(std::path::Path::new(path).join("config.json"))?;
    let config = KokoroConfig::from_bytes(&bytes)?;
    let language = request.language.as_deref().unwrap_or("en");
    let mut runs = Vec::new();
    for _ in 0..=iterations {
        let started = Instant::now();
        let output = phonemizer::phonemize(&request.text, language, &config.vocab)?;
        runs.push(json!({ "ms": started.elapsed().as_secs_f64() * 1000.0, "tokens": output.chars().count() }));
    }
    Ok(json!({"cold": runs[0], "warm": runs[1..]}))
}

#[cfg(not(feature = "kokoro"))]
fn benchmark_phonemizer(_: &str, _: &SynthesisRequest, _: usize) -> anyhow::Result<Value> {
    anyhow::bail!("enable the kokoro feature for phonemizer benchmarks")
}
