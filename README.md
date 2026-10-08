<p align="center">
  <img src="https://flow-like.com/favicon.svg" alt="Flow-Like icon in use here" width="84" height="84">
</p>

# any-tts

<p align="center">
  Rust-native text-to-speech and speech synthesis for modern open-weight models.
</p>

<p align="center">
  <a href="https://github.com/TM9657/any-tts"><img src="https://img.shields.io/badge/repo-TM9657%2Fany--tts-111111" alt="Repository"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/crate_license-MIT%20OR%20Apache--2.0-2d6cdf" alt="Crate license"></a>
    <img src="https://img.shields.io/badge/models-6_public%20backends-0a7f5a" alt="Public models">
  <img src="https://img.shields.io/badge/runtime-Candle%20CPU%20%7C%20CUDA%20%7C%20Metal%20%7C%20Accelerate-8a4fff" alt="Backends">
</p>

any-tts is a Rust TTS library built around Candle with one trait-based API for Kokoro, OmniVoice, Qwen3-TTS, VibeVoice, VibeVoice Realtime, and Voxtral. It is aimed at developers who want local speech synthesis, multilingual TTS, reference-audio prompting, preset-voice workflows, or low-latency voice agents without rewriting their application around each model family.

You can point it at local files, hand it explicit paths from your own cache, feed it named in-memory byte assets from an object store, or let it resolve missing assets from Hugging Face while keeping the synthesis call site unchanged.

For Flow-like specifically: every public backend can now load from relative-path byte assets, so `object_store` reads can go straight into `TtsConfig` without writing temp files first.

## Jump to

- [Supported TTS models](#supported-tts-models)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples in this repo](#examples-in-this-repo)
- [Model guide](#model-guide)

## Why this repo exists

- One API for Kokoro, OmniVoice, Qwen3-TTS, VibeVoice, VibeVoice Realtime, and Voxtral.
- Native Rust backends across the public model surface.
- Local path loading, in-memory byte bundles, per-file wiring, or Hugging Face fallback.
- GPU first when available through CUDA or Metal, with CPU fallback and optional Accelerate support for Apple CPU builds.
- Request-level control for `language`, `voice`, `reference_audio`, `instruct`, `max_tokens`, `temperature`, `speed`, and `cfg_scale`.
- WAV output everywhere, with built-in WAV and MP3 input decoding for cleanup and reference-audio workflows.

## Supported TTS models

| Model | Status in any-tts | Default upstream | Best at | Main tradeoff | Model license |
| --- | --- | --- | --- | --- | --- |
| Kokoro-82M | Public, native, lightweight | `hexgrad/Kokoro-82M` | Fast local TTS with small weights | Uses an in-tree pure-Rust phonemizer compatible with Kokoro's current public language set; parity tuning is still ongoing | Apache-2.0 |
| OmniVoice | Public, native | `k2-fsa/OmniVoice` | Huge language coverage and instruct-driven voice design | The current Rust backend does not yet expose upstream zero-shot cloning | Apache-2.0 |
| Qwen3-TTS-12Hz-1.7B | Public, native | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Strong multilingual control, named speakers, and instruct handling | Heavy weights and extra speech-tokenizer assets | Apache-2.0 |
| VibeVoice-1.5B | Public, native | `microsoft/VibeVoice-1.5B` | Long-form multi-speaker speech diffusion with native Rust inference | Still early and currently optimized for single-request parity work rather than streaming performance | MIT |
| VibeVoice-Realtime-0.5B | Public, native | `microsoft/VibeVoice-Realtime-0.5B` | Low-latency preset-voice TTS with native Rust inference | English-first upstream and depends on cached `voices/*.pt` presets instead of reference audio | MIT |
| Voxtral-4B-TTS-2603 | Public, native | `mistralai/Voxtral-4B-TTS-2603` | Production-style voice agents, preset voices, low-latency oriented stack | Largest backend here and not commercially permissive | CC BY-NC 4.0 |

Important: the Rust crate is dual licensed under `MIT OR Apache-2.0`. The model weights are not. Always check the model-specific license before shipping. Voxtral is the one that changes the deployment story the most because its published checkpoint is `CC BY-NC 4.0`.

All six models above are exposed through the public `ModelType` enum and `load_model()` API.

## Choose a backend

- Use `Kokoro` when you want the smallest local deployment and simple preset-voice TTS.
- Use `OmniVoice` when language coverage and `instruct` matter more than named voices.
- Use `Qwen3Tts` when you want the strongest public request-control surface and named speakers.
- Use `VibeVoice` when you need long-form or reference-audio-conditioned generation.
- Use `VibeVoiceRealtime` when you want cached-prompt preset voices and faster time-to-first-audio.
- Use `Voxtral` when you want preset-voice, voice-agent-style deployment and can accept its model license.

## Installation

For CPU-only builds, a recent stable Rust toolchain is enough. For GPU builds, compile the feature set that matches your machine. Kokoro no longer requires a system `espeak-ng` install: the repo now ships an in-tree pure-Rust phonemizer with an `espeak-rs`-compatible interface for the language set exposed by the current Kokoro backend.

English phonemization preserves sentence context and Kokoro's trained phoneme
symbols, with separate American and British dictionaries. Unknown words use a
bounded, cached Rust fallback; no external executable or runtime download is
used. Numbers, decimals, currencies and percentages are normalized before
non-English sentence splitting. Non-English pronunciation still uses the
in-tree rule frontend; it does not promise complete native eSpeak parity.
Embedded G2P sources and dictionaries retain their upstream licenses and pinned
provenance in [`src/models/kokoro/g2p/NOTICE`](src/models/kokoro/g2p/NOTICE).

Add the crate from crates.io:

```toml
[dependencies]
any-tts = "0.1"
```

Or opt into a smaller feature set:

```toml
[dependencies]
any-tts = { version = "0.1", default-features = false, features = ["kokoro", "download", "metal"] }
```

### Feature flags

By default the crate enables `qwen3-tts`, `kokoro`, `omnivoice`, `vibevoice`, `voxtral`, and `download`. The `vibevoice` feature exposes both `ModelType::VibeVoice` and `ModelType::VibeVoiceRealtime`.

| Feature | What it does |
| --- | --- |
| `kokoro` | Enables the Kokoro backend. |
| `omnivoice` | Enables the native OmniVoice backend. |
| `qwen3-tts` | Enables the Qwen3-TTS backend. |
| `vibevoice` | Enables the native VibeVoice and VibeVoice Realtime backends. |
| `voxtral` | Enables the native Voxtral backend. |
| `download` | Allows missing model files to be pulled from Hugging Face Hub through the crate's built-in downloader. |
| `cuda` | Builds Candle with CUDA support. |
| `metal` | Builds Candle with Metal support for Apple GPUs. |
| `accelerate` | Enables Apple Accelerate support for CPU-heavy Apple builds. |

### Backend selection

- `DeviceSelection::Auto` tries CUDA first, then Metal, then CPU.
- `DeviceSelection::Cpu`, `DeviceSelection::Cuda(0)`, and `DeviceSelection::Metal(0)` let you force the runtime target.
- `preferred_runtime_choice(ModelType::...)` returns the fastest safe device and dtype for the current machine.
- `TtsConfig::with_preferred_runtime()` applies that runtime choice in one builder call.
- `DType` can be set to `F32`, `F16`, or `BF16`.
- On CPU, models that cannot safely run BF16 fall back to `F32`.
- The native OmniVoice helper prefers `cuda:0 (bf16)`, then `metal:0 (f32)`, then `cpu (f32)`.

## Quick start

```rust,no_run
use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_model(
        TtsConfig::new(ModelType::Qwen3Tts)
            .with_model_path("./models/Qwen3-TTS")
    )?;

    let audio = model.synthesize(
        &SynthesisRequest::new("Hello from Rust TTS.")
            .with_language("English")
            .with_voice("Ryan")
            .with_instruct("Calm, clear, slightly upbeat."),
    )?;

    audio.save_wav("hello.wav")?;
    Ok(())
}
```

## Byte-first loading

If your runtime already has model artifacts in memory, use `ModelAssetBundle` or the `with_*_bytes()` builders instead of writing them to disk first.

```rust,no_run
use any_tts::{load_model, ModelAssetBundle, ModelType, SynthesisRequest, TtsConfig};

fn read_object(_key: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
  Ok(Vec::new())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let assets = ModelAssetBundle::new()
    .with_bytes("config.json", read_object("config.json")?)
    .with_bytes("tokenizer.json", read_object("tokenizer.json")?)
    .with_bytes("model.safetensors", read_object("model.safetensors")?)
    .with_bytes(
      "speech_tokenizer/model.safetensors",
      read_object("speech_tokenizer/model.safetensors")?,
    );

  let model = load_model(
    TtsConfig::new(ModelType::Qwen3Tts)
      .with_asset_bundle(assets)
  )?;

  let audio = model.synthesize(&SynthesisRequest::new("Hello from byte-backed assets."))?;
  let wav_bytes = audio.get_wav();
  let _ = wav_bytes;
  Ok(())
}
```

The relative paths in the asset bundle should match the model layout documented below, for example `config.json`, `audio_tokenizer/model.safetensors`, or `voice_embedding/Aurora.pt`.

## Audio bytes

Generated audio already comes back as `AudioSamples`, so output does not need filesystem paths either.

- `audio.get_wav()` returns a complete WAV file as `Vec<u8>` for every backend.
- `audio.save_wav()` is a convenience helper on top of the same byte encoder.

## Audio cleanup

```rust,no_run
use any_tts::{AudioSamples, DenoiseOptions};
use std::io::Cursor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let input = std::fs::read("speech-with-music.mp3")?;
  let cleaned = AudioSamples::denoise_audio_stream(
    Cursor::new(input),
    DenoiseOptions::default(),
  )?;

  cleaned.save_wav("speech-cleaned.wav")?;
  Ok(())
}
```

The denoiser auto-detects WAV and MP3 input streams and applies a speech-band
filter plus a short-time spectral gate. It is useful for attenuating steady
background noise and background music, but it is not a full voice-isolation or
source-separation model.

## File resolution flow

any-tts resolves model assets in four tiers, in this order:

1. Explicit files you set on `TtsConfig` with methods like `with_config_file()` or `with_weight_file()`.
2. Auto-discovery from `with_asset_bundle()` or `with_asset_bytes()` using model-relative paths.
3. Auto-discovery from `with_model_path()` using the expected filenames for that backend.
4. Hugging Face fallback through the `download` feature.

That means you can mix strategies. A service with its own artifact cache can hand over a few exact files and let the crate discover or download the rest.

## Model asset layouts

You can inspect the documented manifest programmatically through `ModelType::asset_requirements()`. The expected relative paths are:

| Model | Required asset patterns | Optional asset patterns |
| --- | --- | --- |
| Kokoro | `config.json`, `model.safetensors` or `*.pth` | `voices/*.pt` |
| OmniVoice | `config.json`, `tokenizer.json`, `model.safetensors` or `model-*-of-*.safetensors`, `audio_tokenizer/config.json`, `audio_tokenizer/model.safetensors` or `audio_tokenizer/model-*-of-*.safetensors` | `generation_config.json` |
| Qwen3-TTS | `config.json`, `tokenizer.json`, `model.safetensors` or `model-*-of-*.safetensors`, `speech_tokenizer/model.safetensors` or `speech_tokenizer/model-*-of-*.safetensors` | `speech_tokenizer/config.json`, `generation_config.json` |
| VibeVoice | `config.json`, `tokenizer.json`, `model.safetensors` or `model-*-of-*.safetensors` | `preprocessor_config.json`, `generation_config.json` |
| VibeVoice Realtime | `config.json`, `tokenizer.json`, `model.safetensors` | `preprocessor_config.json`, `voices/*.pt` |
| Voxtral | `params.json`, `tekken.json`, `consolidated.safetensors`, `voice_embedding/*.pt` | none |

Using these exact relative paths makes the byte-based API foolproof because the same names are what `with_model_path()` auto-discovery expects on disk.

## Request controls

`SynthesisRequest` keeps the per-call control surface stable across models.

| Field | Purpose | Notes |
| --- | --- | --- |
| `text` | Input text to synthesize | Required for every backend. |
| `language` | Language tag or model-specific language name | Supports ISO tags in several backends and `auto` where available. |
| `voice` | Named speaker or preset voice | Works for Kokoro, Qwen3 CustomVoice, VibeVoice Realtime, and Voxtral. OmniVoice rejects named voices, and full VibeVoice expects `reference_audio` instead. |
| `instruct` | Natural-language style control | Most useful on OmniVoice and Qwen3. |
| `max_tokens` | Upper bound on generated codec/audio tokens | Helpful for latency testing and smoke tests. |
| `temperature` | Sampling temperature | Supported where the backend uses it. |
| `speed` | Speaking-rate multiplier | Kokoro only; finite values from `0.1` through `5.0`, default `1.0`. |
| `cfg_scale` | Classifier-free guidance scale | Used by OmniVoice and other backends that expose CFG-like control. |
| `reference_audio` | Reference clip for voice cloning or prompt conditioning | Used by backends such as VibeVoice when conditioning from speech; unsupported backends return an explicit error. |
| `voice_embedding` | Precomputed embedding payload | Currently reusable with backends that accept embeddings directly. |

Kokoro callers that previously used `with_temperature()` to change speaking
rate should switch to `with_speed()`. Temperature no longer changes Kokoro's
duration prediction. Other backends explicitly reject `speed` requests.

## Incremental audio

`TtsModel::synthesize_stream()` delivers consecutive PCM chunks through a
synchronous callback. Kokoro splits long inputs at phoneme-counted sentence,
clause or word boundaries, keeping each chunk within the encoder context.
VibeVoice Realtime emits audio as speech latents are generated using a stateful
causal decoder. `synthesize()` collects the same chunks into one `AudioSamples`.

```rust,no_run
use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_model(
        TtsConfig::new(ModelType::Kokoro).with_model_path("./models/Kokoro-82M")
    )?;
    let request = SynthesisRequest::new("Your text here.").with_speed(1.1);
    model.synthesize_stream(&request, &mut |chunk| {
        // Send chunk.samples to your playback device or network stream here.
        println!("{} samples at {} Hz", chunk.samples.len(), chunk.sample_rate);
        Ok(())
    })?;
    Ok(())
}
```

The callback provides backpressure; returning an error stops generation before
the next chunk. Delivered chunks contain no repeated overlap. Check
`supports_streaming()` before relying on early output: other backends and
noncausal Realtime codec configurations deliver a complete buffered chunk.
Realtime's `max_tokens` limits newly generated speech tokens; text conditioning
and cached prompts consume context capacity but do not consume this budget.

## Measuring latency

Use a release build and local model assets:

```bash
cargo run --release --example benchmark_tts -- \
  --model kokoro --model-path ./models/Kokoro-82M --device cpu --iterations 5
```

The JSON report records model loading, the first synthesis, warm median/p95
latency, time to first audio, audio duration and real-time factor. For Kokoro it
also measures cold and warm phonemization separately; this initializes the G2P
dictionaries before the synthesis measurements. Add `--phonemizer-only` to
benchmark text processing using just the model's `config.json`. Use `--text`,
`--language`, `--voice`, `--temperature`, `--speed`, and `--max-tokens` to hold the workload
constant. Compare short and long utterances and listen to generated audio when
evaluating performance changes.

Applications can move dictionary initialization out of their first request with
`any_tts::models::kokoro::phonemizer::prewarm("en")?` (or `"en-gb"` / `"ja"`).

## Examples in this repo

These are the example entry points that match the current public crate surface:

```bash
cargo run --example generate_kokoro --release
cargo run --example generate_qwen3_tts --release
cargo run --example generate_vibevoice --release --no-default-features --features vibevoice,download,metal
cargo run --example generate_vibevoice_realtime --release --no-default-features --features vibevoice,download,metal
cargo run --example generate_voxtral --release
cargo run --example generate_omnivoice --release --no-default-features --features omnivoice,download,metal
cargo run --example generate_comparison_suite --release --features metal -- --runtime all
cargo run --example benchmark_omnivoice --release --no-default-features --features omnivoice,download,metal -- --warmup 1 --iterations 3
```

Outputs are written under `output/` by the example binaries.

`generate_vibevoice` keeps writing the main raw render to the configured
`VIBEVOICE_OUTPUT` path and also writes `*_base.wav`,
`*_denoised_default.wav`, and `*_denoised_aggressive.wav` under
`output/denoise/` by default. You can override that folder with
`VIBEVOICE_DENOISE_DIR`.

`generate_vibevoice_realtime` targets `ModelType::VibeVoiceRealtime` and expects cached prompt presets under `models/VibeVoice-Realtime-0.5B/voices/` by default. Use `VIBEVOICE_REALTIME_MODEL_PATH`, `VIBEVOICE_REALTIME_VOICES_DIR`, `VIBEVOICE_REALTIME_VOICE`, `VIBEVOICE_REALTIME_DEVICE`, and `VIBEVOICE_REALTIME_OUTPUT` to override the defaults.

`generate_comparison_suite` writes a shared English and German comparison set under `output/model_comparison/cpu/` and `output/model_comparison/metal/`, plus `report.json` files with per-model load time, per-sample synthesis time, audio duration, and realtime factor. It loads one model at a time so the full suite can run sequentially on tighter memory budgets.

## Model guide

### Kokoro-82M

**What it is**

Kokoro is the compact option in this repo: an 82M-parameter StyleTTS2 plus ISTFTNet stack with Apache-licensed weights. In practice, it is the backend you reach for when you want a fast local model, simple deployment, and a much smaller download than the larger multilingual checkpoints.

**What works in any-tts today**

- Native Rust inference.
- Default output at 24 kHz.
- Named preset voices discovered from the `voices/` directory.
- Language tags exposed by the current backend: `en`, `en-gb`, `ja`, `zh`, `ko`, `fr`, `de`, `it`, `pt`, `es`, `hi`.
- Long-input chunking and callback streaming within the model's context limit.
- Optional voice-cloning support only when a checkpoint includes style-encoder weights.

**Pros**

- Small enough to be the practical local-first choice.
- Apache-2.0 model license makes deployment straightforward.
- Good fit for desktop apps, tools, and low-latency local generation.
- Simple model layout relative to the larger codebook-based stacks.

**Cons**

- Non-English pronunciation uses a rule frontend whose parity tuning is ongoing.
- The common open release is mostly about preset voice packs, not raw zero-shot cloning.
- Less expressive control than the bigger instruct-heavy model families.

**License**

- Upstream model weights: Apache-2.0.
- Crate code using the model: `MIT OR Apache-2.0`.

### OmniVoice

**What it is**

OmniVoice is the ambition play in this repo. Upstream, it is a diffusion language model TTS stack aimed at omnilingual zero-shot speech generation with voice design and massive language coverage.

**What works in any-tts today**

- Native Candle backend.
- `language`, `instruct`, `cfg_scale`, and `max_tokens` request controls.
- Automatic runtime preference selection for CPU, CUDA, or Metal.
- Repo-exposed language set: `auto`, `en`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `pt`, `ru`, `it`.

**What does not work yet in the Rust backend**

- Named voices.
- Reference-audio voice cloning.
- Reusable voice embeddings.

The code returns explicit errors for those cases instead of silently falling back to Python.

**Pros**

- Strong upstream story for language coverage.
- Good fit for instruction-driven voice design.
- Benchmark helper in this repo already makes backend comparisons easy.
- Apache-2.0 model license.

**Cons**

- The current Rust implementation exposes less than the upstream model card promises.
- If your main requirement is zero-shot cloning from reference audio, this backend is not there yet in this crate.
- Heavier than Kokoro and less turnkey than the small local-first path.

**License**

- Upstream model weights: Apache-2.0.
- Crate code using the model: `MIT OR Apache-2.0`.

### Qwen3-TTS

**What it is**

Qwen3-TTS is the control-heavy multilingual option. It uses a discrete multi-codebook language model plus a speech-tokenizer decoder and is designed for named speakers, instruction-following, and multiple TTS operating modes.

**What works in any-tts today**

- Native Rust backend.
- Default path points to `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.
- Named speaker generation for CustomVoice checkpoints.
- VoiceDesign checkpoints also work when selected through `with_hf_model_id()` or local files.
- 24 kHz output with the extra speech-tokenizer weights resolved alongside the main model.
- Repo-level language support tracks the current checkpoint config and includes `auto`.

**Example: switch from CustomVoice to VoiceDesign**

```rust,no_run
use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::Qwen3Tts)
        .with_hf_model_id("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
)?;

let audio = model.synthesize(
    &SynthesisRequest::new("This voice should sound sharp, precise, and quietly confident.")
        .with_language("English")
        .with_instruct("Female presenter voice, low warmth, clear diction, subtle authority."),
)?;
```

**Pros**

- Best overall control surface in the current public crate API.
- Strong multilingual coverage: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian upstream.
- Named speakers are easy to use from a single request builder.
- VoiceDesign gives you a second mode without changing the main crate API.

**Cons**

- Large download and memory footprint compared with Kokoro.
- Requires the separate speech-tokenizer decoder assets in addition to the main weights.
- Upstream Base-model voice cloning exists, but reference-audio cloning is not implemented in this crate yet.

**License**

- Upstream model weights: Apache-2.0.
- Crate code using the model: `MIT OR Apache-2.0`.

### VibeVoice-1.5B

**What it is**

VibeVoice-1.5B is the long-form diffusion backend in this crate. It is the VibeVoice option to choose when you want native Rust inference with reference-audio-conditioned prompting instead of named preset voices.

**What works in any-tts today**

- Native Rust backend.
- `reference_audio`, `language`, `instruct`, `cfg_scale`, `max_tokens`, and `temperature` request controls.
- 24 kHz output with automatic runtime selection across CPU, CUDA, or Metal.
- Public `ModelType::VibeVoice` loading through the shared `vibevoice` feature.

**What does not work yet in the Rust backend**

- Named voice presets.
- Pre-extracted reusable voice embeddings.
- Low-latency streaming generation; the current path is still optimized for correctness and parity work.

**Pros**

- Best fit in this repo for long-form, reference-audio-conditioned synthesis.
- Native Rust inference for a model family that is often driven from Python examples upstream.
- Shares the same trait-based API as the smaller and faster backends.

**Cons**

- Heavier and slower than Kokoro or VibeVoice Realtime.
- Still an early backend compared with the simpler local-first paths.
- Not the right choice if your application needs preset voice names or fast startup latency.

**License**

- Upstream model weights: MIT.
- Crate code using the model: `MIT OR Apache-2.0`.

### VibeVoice-Realtime-0.5B

**What it is**

VibeVoice Realtime is the smaller low-latency VibeVoice variant. In this crate it is exposed as `ModelType::VibeVoiceRealtime` and centers on cached-prompt voice presets rather than reference-audio cloning.

**What works in any-tts today**

- Native Rust backend.
- Cached-prompt voice presets discovered from `voices/*.pt`.
- Public example coverage through `generate_vibevoice_realtime`.
- 24 kHz output with the same Candle runtime selection flow used by the rest of the crate.
- Voice selection through `SynthesisRequest::with_voice()` when matching preset files are present.
- Incremental PCM output through `synthesize_stream()` with causal codec checkpoints.

**What does not work yet in the Rust backend**

- Reference-audio input.
- Pre-extracted voice embeddings.
- Arbitrary speaker cloning without upstream preset cache files.

**Pros**

- Best fit here for low-latency preset-voice generation.
- Smaller checkpoint than the full VibeVoice-1.5B model.
- Good developer path for apps that reuse approved voices and want predictable startup behavior.

**Cons**

- Depends on `voices/*.pt` preset caches and fails explicitly if they are missing.
- The upstream model card is English-first and research-oriented.
- Less flexible than full VibeVoice when you need reference-audio conditioning or multi-speaker behavior.

**License**

- Upstream model weights: MIT.
- Crate code using the model: `MIT OR Apache-2.0`.

### Voxtral-4B-TTS-2603

**What it is**

Voxtral is the biggest public backend in the repo and the most obviously voice-agent-oriented. It pairs a language model with acoustic generation and preset voice embeddings, and the published checkpoint is tuned for multilingual, low-latency TTS deployment scenarios.

**What works in any-tts today**

- Native Rust backend.
- Preset voice selection from the checkpoint's `voice_embedding/` assets.
- Optional direct `voice_embedding` reuse when you already have a compatible embedding.
- Repo-exposed languages: `en`, `fr`, `es`, `de`, `it`, `pt`, `nl`, `ar`, `hi`.
- Default sample rate is resolved from the model config and outputs at 24 kHz with the published checkpoint.

**Pros**

- Best fit here for production-style voice-agent workloads.
- Upstream model card emphasizes streaming and low time-to-first-audio.
- Comes with preset voices and a clear multilingual story.

**Cons**

- The open checkpoint does not ship reference-audio encoder weights, so raw voice cloning is unavailable.
- This is the heaviest public backend in the crate.
- The published model license is `CC BY-NC 4.0`, so commercial deployment needs extra care or a different model choice.

**License**

- Upstream model weights: CC BY-NC 4.0.
- Crate code using the model: `MIT OR Apache-2.0`.

## Usage patterns that hold across models

### Local directory loading

```rust,no_run
use any_tts::{load_model, ModelType, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::Kokoro)
        .with_model_path("./models/Kokoro-82M")
)?;
```

### Explicit file-path loading

```rust,no_run
use any_tts::{load_model, ModelType, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::Qwen3Tts)
        .with_config_file("/cache/config.json")
        .with_tokenizer_file("/cache/tokenizer.json")
        .with_weight_file("/cache/model-00001-of-00002.safetensors")
        .with_weight_file("/cache/model-00002-of-00002.safetensors")
        .with_speech_tokenizer_weight_file("/cache/speech-tokenizer.safetensors")
)?;
```

### Device and dtype selection

```rust,no_run
use any_tts::config::DType;
use any_tts::{load_model, DeviceSelection, ModelType, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::OmniVoice)
        .with_device(DeviceSelection::Metal(0))
        .with_dtype(DType::F16)
)?;
```

## Repo health files

This repo now includes the standard GitHub community files you would expect for an active project:

- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Support Guide](SUPPORT.md)
- Issue templates under `.github/ISSUE_TEMPLATE/`
- A pull request template at `.github/PULL_REQUEST_TEMPLATE.md`

## Contributing

If you are adding a backend, model variant, or new loading flow, keep the public story honest: unsupported features should fail explicitly, examples should match exported API, and docs should separate experimental repo work from supported top-level surfaces.

The short version is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Model weights, runtime backends, and artifact loading all change the risk profile of TTS systems. Please read [SECURITY.md](SECURITY.md) before disclosing a vulnerability publicly.

## License

The crate metadata declares `MIT OR Apache-2.0` for this repository's Rust code. That does not supersede the terms attached to any model weights you download and run through it.

## Status

This repo now has a clear public shape: six native backends, trait-based loading, byte-first asset support, and example coverage for local, multilingual, long-form, preset-voice, and realtime TTS workflows. The right way to think about it is not "a single-model wrapper" but "a Rust TTS platform layer that lets one application target multiple open model ecosystems without hiding their differences."
