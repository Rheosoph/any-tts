//! Quick test for Qwen3-TTS audio quality.
use any_tts::models::qwen3_tts::Qwen3TtsModel;
use any_tts::traits::TtsModel;
use any_tts::{ModelType, SynthesisRequest, TtsConfig};

fn main() {
    let out_dir = std::path::Path::new("output/qwen3_tts");
    std::fs::create_dir_all(out_dir).ok();

    println!("Loading Qwen3-TTS...");
    let config = TtsConfig::new(ModelType::Qwen3Tts).with_model_path("./models/Qwen3-TTS");
    let model = Qwen3TtsModel::load(config).expect("Failed to load model");
    println!("Model loaded.");

    // Verify tokenization: check that special tokens are correctly identified
    {
        use any_tts::tokenizer::TextTokenizer;
        let files = model.files();
        let tok_path = files.tokenizer.as_ref().unwrap();
        let tok = TextTokenizer::from_asset(tok_path).unwrap();

        let template = "<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>assistant\n";
        let ids = tok.encode(template).unwrap();
        println!("Template tokens ({} total): {:?}", ids.len(), ids);
        println!("First 3 (role): {:?}", &ids[..3.min(ids.len())]);
        if ids.len() > 8 {
            println!("Text content [3..-5]: {:?}", &ids[3..ids.len() - 5]);
        }
        println!("Last 5: {:?}", &ids[ids.len().saturating_sub(5)..]);

        // Check specific tokens
        for name in [
            "<|im_start|>",
            "<|im_end|>",
            "<|tts_pad|>",
            "<|tts_bos|>",
            "<|tts_eos|>",
        ] {
            if let Some(id) = tok.token_to_id(name) {
                println!("  {} = {}", name, id);
            } else {
                println!("  {} = NOT FOUND", name);
            }
        }
    }

    println!("\nGenerating...");
    let request = SynthesisRequest::new("Hello.")
        .with_language("en")
        .with_voice("serena")
        .with_max_tokens(300);

    match model.synthesize(&request) {
        Ok(audio) => {
            println!(
                "Generated {:.2}s ({} samples @ {} Hz)",
                audio.duration_secs(),
                audio.len(),
                audio.sample_rate
            );
            let wav_path = out_dir.join("quick_test.wav");
            audio.save_wav(&wav_path).expect("Failed to write WAV");
            println!("Saved to {}", wav_path.display());
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}
