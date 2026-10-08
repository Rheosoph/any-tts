//! Shared neural-network building blocks used by model backends.
//!
//! Qwen-family backends use grouped-query attention and SiLU-gated MLPs,
//! while Kokoro uses ALBERT + LSTM + ISTFTNet. This module provides reusable
//! implementations so the model-specific code only has to deal with
//! architecture differences.

pub mod attention;
pub mod conv;
pub mod kv_cache;
pub mod lstm;
pub mod mlp;
pub mod transformer;

pub use attention::GroupedQueryAttention;
pub use conv::{AdaIn1d, AdaLayerNorm, Conv1d, ConvTranspose1d, InstanceNorm1d, LinearNorm};
pub use lstm::Lstm;
pub use mlp::SiluMlp;
pub use transformer::TransformerBlock;
