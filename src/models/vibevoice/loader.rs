use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tracing::info;

use crate::config::ModelFiles;
use crate::error::TtsError;
use crate::tokenizer::TextTokenizer;

use super::config::{VibeVoiceConfig, VibeVoicePreprocessorConfig};
use super::diffusion::VibeVoiceDiffusionHead;
use super::generation::valid_generated_tokens;
use super::processor::{VibeVoiceProcessor, VibeVoiceTokenizerSpec};
use super::runtime::{SpeechConnector, VibeVoiceLanguageModel};
use super::speech_tokenizer::{VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer};

pub(super) struct LoadedVibeVoiceComponents {
    pub language_model: VibeVoiceLanguageModel,
    pub acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    pub semantic_tokenizer: VibeVoiceSemanticTokenizer,
    pub acoustic_connector: SpeechConnector,
    pub semantic_connector: SpeechConnector,
    pub prediction_head: VibeVoiceDiffusionHead,
    pub speech_scaling_factor: f32,
    pub speech_bias_factor: f32,
}

pub(super) fn resolve_runtime_dtype(device: &Device, requested: DType) -> DType {
    if matches!(device, Device::Cpu) && requested == DType::BF16 {
        info!("BF16 is not supported on CPU; falling back to F32 for VibeVoice");
        return DType::F32;
    }

    if matches!(device, Device::Metal(_)) {
        info!(
            "VibeVoice uses F32 on Metal to match the Python reference path and avoid unstable first-token behavior"
        );
        return DType::F32;
    }

    requested
}

pub(super) fn load_preprocessor_config(
    files: &ModelFiles,
) -> Result<VibeVoicePreprocessorConfig, TtsError> {
    if let Some(asset) = &files.preprocessor_config {
        let bytes = asset.read_bytes()?;
        return VibeVoicePreprocessorConfig::from_bytes(bytes.as_ref());
    }
    Ok(VibeVoicePreprocessorConfig::default())
}

pub(super) fn build_processor(
    files: &ModelFiles,
    preprocessor_config: &VibeVoicePreprocessorConfig,
) -> Result<VibeVoiceProcessor, TtsError> {
    let tokenizer = TextTokenizer::from_asset(
        files
            .tokenizer
            .as_ref()
            .expect("validated by resolve_files"),
    )?;
    let tokenizer_spec = VibeVoiceTokenizerSpec::from_tokenizer(&tokenizer)?;
    Ok(VibeVoiceProcessor::new(
        tokenizer,
        tokenizer_spec,
        preprocessor_config.clone(),
    ))
}

pub(super) fn load_components(
    files: &ModelFiles,
    model_config: &VibeVoiceConfig,
    device: &Device,
    dtype: DType,
    tokenizer_spec: &VibeVoiceTokenizerSpec,
) -> Result<LoadedVibeVoiceComponents, TtsError> {
    let vb = ModelFiles::load_safetensors_vb(&files.weights, dtype, device)?;
    let model_vb = vb.pp("model");
    let language_model = VibeVoiceLanguageModel::load(
        &model_config.decoder_config,
        model_vb.pp("language_model"),
        device,
        dtype,
        &valid_generated_tokens(
            tokenizer_spec.speech_start_id,
            tokenizer_spec.speech_end_id,
            tokenizer_spec.speech_diffusion_id,
            tokenizer_spec.eos_id,
            tokenizer_spec.bos_id,
        ),
    )?;
    let acoustic_tokenizer = VibeVoiceAcousticTokenizer::load(
        &model_config.acoustic_tokenizer_config,
        model_vb.pp("acoustic_tokenizer"),
    )?;
    let semantic_tokenizer = VibeVoiceSemanticTokenizer::load(
        &model_config.semantic_tokenizer_config,
        model_vb.pp("semantic_tokenizer"),
    )?;
    let acoustic_connector = SpeechConnector::load(
        model_config.acoustic_vae_dim,
        model_config.decoder_config.hidden_size,
        model_vb.pp("acoustic_connector"),
    )?;
    let semantic_connector = SpeechConnector::load(
        model_config.semantic_vae_dim,
        model_config.decoder_config.hidden_size,
        model_vb.pp("semantic_connector"),
    )?;
    let prediction_head = VibeVoiceDiffusionHead::load(
        &model_config.diffusion_head_config,
        model_vb.pp("prediction_head"),
    )?;

    Ok(LoadedVibeVoiceComponents {
        language_model,
        acoustic_tokenizer,
        semantic_tokenizer,
        acoustic_connector,
        semantic_connector,
        prediction_head,
        speech_scaling_factor: load_scalar(&model_vb, "speech_scaling_factor")?,
        speech_bias_factor: load_scalar(&model_vb, "speech_bias_factor")?,
    })
}

fn load_scalar(model_vb: &VarBuilder, name: &str) -> Result<f32, TtsError> {
    let tensor = model_vb.get(1, name).or_else(|_| model_vb.get((), name))?;
    scalar_from_tensor(&tensor)
}

fn scalar_from_tensor(tensor: &Tensor) -> Result<f32, TtsError> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| TtsError::ModelError("Expected scalar tensor".to_string()))
}
