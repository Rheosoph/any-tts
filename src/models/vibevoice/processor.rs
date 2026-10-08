use std::collections::BTreeMap;

use candle_core::{Device, Tensor};

use crate::error::TtsError;
use crate::mel::resample_audio;
use crate::tokenizer::TextTokenizer;
use crate::traits::{ReferenceAudio, SynthesisRequest};

use super::config::VibeVoicePreprocessorConfig;

const SYSTEM_PROMPT: &str = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";

type VoicePrompt = (Vec<u32>, Vec<bool>, Vec<f32>);

#[derive(Debug, Clone)]
pub struct VibeVoiceTokenizerSpec {
    pub speech_start_id: u32,
    pub speech_end_id: u32,
    pub speech_diffusion_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    pub bos_id: Option<u32>,
}

impl VibeVoiceTokenizerSpec {
    pub fn from_tokenizer(tokenizer: &TextTokenizer) -> Result<Self, TtsError> {
        let speech_start_id = tokenizer.token_to_id("<|vision_start|>").ok_or_else(|| {
            TtsError::TokenizerError("Missing <|vision_start|> token".to_string())
        })?;
        let speech_end_id = tokenizer
            .token_to_id("<|vision_end|>")
            .ok_or_else(|| TtsError::TokenizerError("Missing <|vision_end|> token".to_string()))?;
        let speech_diffusion_id = tokenizer
            .token_to_id("<|vision_pad|>")
            .ok_or_else(|| TtsError::TokenizerError("Missing <|vision_pad|> token".to_string()))?;
        let eos_id = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| TtsError::TokenizerError("Missing <|endoftext|> token".to_string()))?;
        let pad_id = tokenizer.token_to_id("<|image_pad|>").unwrap_or(eos_id);
        let bos_id = tokenizer.token_to_id("<|begin_of_text|>");

        Ok(Self {
            speech_start_id,
            speech_end_id,
            speech_diffusion_id,
            eos_id,
            pad_id,
            bos_id,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ParsedScriptLine {
    pub speaker_id: usize,
    pub text: String,
}

#[derive(Debug)]
pub struct PreparedSpeechInputs {
    pub speech_tensors: Tensor,
    pub speech_masks: Tensor,
}

#[derive(Debug)]
pub struct PreparedVibeVoiceInput {
    pub input_ids: Vec<u32>,
    pub speech_input_mask: Vec<bool>,
    pub speech_inputs: Option<PreparedSpeechInputs>,
    pub parsed_script: Vec<ParsedScriptLine>,
    pub all_speakers: Vec<usize>,
}

pub struct VibeVoiceProcessor {
    config: VibeVoicePreprocessorConfig,
    tokenizer: TextTokenizer,
    tokenizer_spec: VibeVoiceTokenizerSpec,
}

impl VibeVoiceProcessor {
    pub fn new(
        tokenizer: TextTokenizer,
        tokenizer_spec: VibeVoiceTokenizerSpec,
        config: VibeVoicePreprocessorConfig,
    ) -> Self {
        Self {
            config,
            tokenizer,
            tokenizer_spec,
        }
    }

    pub fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    pub fn tokenizer_spec(&self) -> &VibeVoiceTokenizerSpec {
        &self.tokenizer_spec
    }

    pub fn config(&self) -> &VibeVoicePreprocessorConfig {
        &self.config
    }

    pub fn prepare_request(
        &self,
        request: &SynthesisRequest,
        device: &Device,
    ) -> Result<PreparedVibeVoiceInput, TtsError> {
        if request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice does not yet accept pre-extracted voice embeddings".to_string(),
            ));
        }

        let parsed_script = self.parse_script(&request.text);
        let all_speakers = parsed_script
            .iter()
            .map(|line| line.speaker_id)
            .collect::<Vec<_>>();

        let mut input_ids = self.tokenizer.encode(SYSTEM_PROMPT)?;
        let mut speech_input_mask = vec![false; input_ids.len()];
        let mut speech_inputs = Vec::new();

        if let Some(reference_audio) = &request.reference_audio {
            let (voice_tokens, voice_masks, voice_audio) =
                self.create_voice_prompt(reference_audio)?;
            input_ids.extend(voice_tokens);
            speech_input_mask.extend(voice_masks);
            speech_inputs.push(voice_audio);
        }

        let text_input_tokens = self.tokenizer.encode(" Text input:\n")?;
        speech_input_mask.extend(vec![false; text_input_tokens.len()]);
        input_ids.extend(text_input_tokens);

        for line in &parsed_script {
            let line_tokens = self
                .tokenizer
                .encode(&format!(" Speaker {}:{}\n", line.speaker_id, line.text))?;
            speech_input_mask.extend(vec![false; line_tokens.len()]);
            input_ids.extend(line_tokens);
        }

        let speech_output_tokens = self.tokenizer.encode(" Speech output:\n")?;
        speech_input_mask.extend(vec![false; speech_output_tokens.len() + 1]);
        input_ids.extend(speech_output_tokens);
        input_ids.push(self.tokenizer_spec.speech_start_id);

        let speech_inputs = if speech_inputs.is_empty() {
            None
        } else {
            Some(self.prepare_speech_inputs(&speech_inputs, device)?)
        };

        Ok(PreparedVibeVoiceInput {
            input_ids,
            speech_input_mask,
            speech_inputs,
            parsed_script,
            all_speakers,
        })
    }

    fn create_voice_prompt(
        &self,
        reference_audio: &ReferenceAudio,
    ) -> Result<VoicePrompt, TtsError> {
        let mut tokens = self.tokenizer.encode(" Voice input:\n")?;
        let mut masks = vec![false; tokens.len()];
        let prefix_tokens = self.tokenizer.encode(" Speaker 0:")?;
        let normalized_audio = self.normalize_reference_audio(reference_audio)?;
        let vae_len = normalized_audio
            .len()
            .div_ceil(self.config.speech_tok_compress_ratio);

        let mut speaker_tokens = prefix_tokens.clone();
        speaker_tokens.push(self.tokenizer_spec.speech_start_id);
        speaker_tokens.extend(std::iter::repeat_n(
            self.tokenizer_spec.speech_diffusion_id,
            vae_len,
        ));
        speaker_tokens.push(self.tokenizer_spec.speech_end_id);
        speaker_tokens.extend(self.tokenizer.encode("\n")?);

        let mut speaker_mask = vec![false; prefix_tokens.len() + 1];
        speaker_mask.extend(std::iter::repeat_n(true, vae_len));
        speaker_mask.extend(vec![false; 2]);

        tokens.extend(speaker_tokens);
        masks.extend(speaker_mask);

        Ok((tokens, masks, normalized_audio))
    }

    fn prepare_speech_inputs(
        &self,
        speech_inputs: &[Vec<f32>],
        device: &Device,
    ) -> Result<PreparedSpeechInputs, TtsError> {
        let max_samples = speech_inputs.iter().map(Vec::len).max().unwrap_or(0);
        let max_tokens = speech_inputs
            .iter()
            .map(|samples| {
                samples
                    .len()
                    .div_ceil(self.config.speech_tok_compress_ratio)
            })
            .max()
            .unwrap_or(0);

        let mut padded = vec![0f32; speech_inputs.len() * max_samples];
        let mut masks = vec![0u8; speech_inputs.len() * max_tokens];

        for (row, samples) in speech_inputs.iter().enumerate() {
            let start = row * max_samples;
            padded[start..start + samples.len()].copy_from_slice(samples);

            let token_len = samples
                .len()
                .div_ceil(self.config.speech_tok_compress_ratio);
            let mask_start = row * max_tokens;
            for value in &mut masks[mask_start..mask_start + token_len] {
                *value = 1;
            }
        }

        let speech_tensors = Tensor::from_vec(padded, (speech_inputs.len(), max_samples), device)?;
        let speech_masks = Tensor::from_vec(masks, (speech_inputs.len(), max_tokens), device)?;

        Ok(PreparedSpeechInputs {
            speech_tensors,
            speech_masks,
        })
    }

    pub fn normalize_reference_audio(&self, audio: &ReferenceAudio) -> Result<Vec<f32>, TtsError> {
        let resampled = resample_audio(
            &audio.samples,
            audio.sample_rate,
            self.config.audio_processor.sampling_rate,
        )?;

        if !self.config.db_normalize {
            return Ok(resampled);
        }

        Ok(normalize_dbfs(
            &resampled,
            self.config.audio_processor.target_d_b_fs,
            self.config.audio_processor.eps,
        ))
    }

    pub fn parse_script(&self, script: &str) -> Vec<ParsedScriptLine> {
        let mut parsed = Vec::new();
        let mut raw_ids = Vec::new();

        for line in script.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some((speaker_id, text)) = parse_speaker_line(trimmed) {
                raw_ids.push(speaker_id);
                parsed.push((speaker_id, text));
            } else {
                parsed.push((0, trimmed.to_string()));
                raw_ids.push(0);
            }
        }

        if parsed.is_empty() {
            parsed.push((0, script.trim().to_string()));
            raw_ids.push(0);
        }

        let mut mapping = BTreeMap::new();
        for raw_id in raw_ids {
            let next = mapping.len();
            mapping.entry(raw_id).or_insert(next);
        }

        parsed
            .into_iter()
            .map(|(speaker_id, text)| ParsedScriptLine {
                speaker_id: *mapping.get(&speaker_id).unwrap_or(&speaker_id),
                text: format!(" {}", text.trim()),
            })
            .collect()
    }
}

fn parse_speaker_line(line: &str) -> Option<(usize, String)> {
    let rest = line.strip_prefix("Speaker ")?;
    let (speaker_id, text) = rest.split_once(':')?;
    let speaker_id = speaker_id.trim().parse::<usize>().ok()?;
    Some((speaker_id, text.trim().to_string()))
}

fn normalize_dbfs(samples: &[f32], target_db_fs: f32, eps: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let rms =
        (samples.iter().map(|value| value * value).sum::<f32>() / samples.len() as f32).sqrt();
    let scalar = 10f32.powf(target_db_fs / 20.0) / (rms + eps);
    let mut normalized = samples
        .iter()
        .map(|value| value * scalar)
        .collect::<Vec<_>>();

    let peak = normalized
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    if peak > 1.0 {
        let scale = peak + eps;
        for value in &mut normalized {
            *value /= scale;
        }
    }

    normalized
}
