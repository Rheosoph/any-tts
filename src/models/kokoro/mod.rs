//! Kokoro-82M model implementation.
//!
//! Architecture: PL-BERT → Text Encoder → Prosody Predictor → ISTFTNet Decoder → 24kHz audio
//! StyleTTS2-based approach with lightweight 82M parameters.
//!
//! Voice cloning is supported when the checkpoint includes the style encoder
//! weights (`style_encoder.*` and `predictor_encoder.*` prefixes).

pub mod albert;
pub mod config;
pub mod decoder;
mod english_g2p;
pub mod espeak_compat;
mod g2p;
pub mod model;
pub mod phonemizer;
pub mod prosody;
pub mod style_encoder;
pub mod text_encoder;
mod text_normalization;

pub use model::KokoroModel;
