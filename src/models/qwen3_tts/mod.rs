//! Qwen3-TTS-12Hz model implementation.
//!
//! Architecture: Talker LM + Code Predictor → Speech Tokenizer Decoder → 24kHz audio
//! Discrete multi-codebook language model approach.

pub mod code_predictor;
pub mod config;
pub mod model;
mod sampling;
pub mod speech_tokenizer;
pub mod talker;

pub use model::Qwen3TtsModel;
