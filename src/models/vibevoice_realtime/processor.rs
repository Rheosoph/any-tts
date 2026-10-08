use crate::error::TtsError;
use crate::tokenizer::TextTokenizer;
use crate::traits::SynthesisRequest;

use crate::models::vibevoice::config::VibeVoicePreprocessorConfig;
use crate::models::vibevoice::processor::VibeVoiceTokenizerSpec;

pub struct VibeVoiceRealtimeProcessor {
    tokenizer: TextTokenizer,
}

impl VibeVoiceRealtimeProcessor {
    pub fn new(
        tokenizer: TextTokenizer,
        _tokenizer_spec: VibeVoiceTokenizerSpec,
        _config: VibeVoicePreprocessorConfig,
    ) -> Self {
        Self { tokenizer }
    }

    pub fn prepare_text(&self, request: &SynthesisRequest) -> Result<Vec<u32>, TtsError> {
        self.tokenizer.encode(&format!("{}\n", request.text.trim()))
    }
}
