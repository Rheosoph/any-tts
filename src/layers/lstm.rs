//! LSTM (Long Short-Term Memory) implementation for candle.
//!
//! Provides inference-only LSTM layers compatible with PyTorch weight
//! format. Used by Kokoro's text encoder and prosody predictor.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// A single LSTM layer (unidirectional or bidirectional).
///
/// Compatible with PyTorch's `nn.LSTM` weight naming:
/// - `weight_ih_l0` shape `[4 * hidden, input]`
/// - `weight_hh_l0` shape `[4 * hidden, hidden]`
/// - `bias_ih_l0` shape `[4 * hidden]`
/// - `bias_hh_l0` shape `[4 * hidden]`
/// - For reverse: `weight_ih_l0_reverse`, etc.
pub struct Lstm {
    layers: Vec<LstmLayer>,
    bidirectional: bool,
    hidden_size: usize,
}

struct LstmLayer {
    forward_weights: LstmWeights,
    reverse_weights: Option<LstmWeights>,
    hidden_size: usize,
}

/// Inference weights in the layout consumed by matrix multiplication.
struct LstmWeights {
    weight_ih_t: Tensor,
    weight_hh_t: Tensor,
    bias: Tensor,
}

impl LstmWeights {
    fn new(weight_ih: Tensor, weight_hh: Tensor, bias_ih: Tensor, bias_hh: Tensor) -> Result<Self> {
        Ok(Self {
            weight_ih_t: weight_ih.t()?.contiguous()?,
            weight_hh_t: weight_hh.t()?.contiguous()?,
            bias: (bias_ih + bias_hh)?,
        })
    }
}

impl Lstm {
    /// Load an LSTM from a VarBuilder.
    ///
    /// `num_layers`: number of stacked LSTM layers
    /// `input_size`: input feature dimension
    /// `hidden_size`: hidden state dimension
    /// `bidirectional`: whether to use bidirectional LSTM
    pub fn load(
        num_layers: usize,
        input_size: usize,
        hidden_size: usize,
        bidirectional: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let dir_factor = if bidirectional { 2 } else { 1 };

        for layer_idx in 0..num_layers {
            let in_size = if layer_idx == 0 {
                input_size
            } else {
                hidden_size * dir_factor
            };

            let weight_ih = vb.get(
                (4 * hidden_size, in_size),
                &format!("weight_ih_l{}", layer_idx),
            )?;
            let weight_hh = vb.get(
                (4 * hidden_size, hidden_size),
                &format!("weight_hh_l{}", layer_idx),
            )?;
            let bias_ih = vb.get(4 * hidden_size, &format!("bias_ih_l{}", layer_idx))?;
            let bias_hh = vb.get(4 * hidden_size, &format!("bias_hh_l{}", layer_idx))?;

            let reverse_weights = if bidirectional {
                let w_ih = vb.get(
                    (4 * hidden_size, in_size),
                    &format!("weight_ih_l{}_reverse", layer_idx),
                )?;
                let w_hh = vb.get(
                    (4 * hidden_size, hidden_size),
                    &format!("weight_hh_l{}_reverse", layer_idx),
                )?;
                let b_ih = vb.get(4 * hidden_size, &format!("bias_ih_l{}_reverse", layer_idx))?;
                let b_hh = vb.get(4 * hidden_size, &format!("bias_hh_l{}_reverse", layer_idx))?;
                Some(LstmWeights::new(w_ih, w_hh, b_ih, b_hh)?)
            } else {
                None
            };

            layers.push(LstmLayer {
                forward_weights: LstmWeights::new(weight_ih, weight_hh, bias_ih, bias_hh)?,
                reverse_weights,
                hidden_size,
            });
        }

        Ok(Self {
            layers,
            bidirectional,
            hidden_size,
        })
    }

    /// Forward pass through the LSTM.
    ///
    /// `input`: tensor of shape `[batch, seq_len, input_size]`
    ///
    /// Returns output of shape `[batch, seq_len, hidden_size * num_directions]`
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut current = input.clone();

        for layer in &self.layers {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    /// Output size (hidden_size * num_directions).
    pub fn output_size(&self) -> usize {
        self.hidden_size * if self.bidirectional { 2 } else { 1 }
    }
}

impl LstmLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let fwd = self.forward_direction(input, &self.forward_weights, false)?;

        if let Some(weights) = &self.reverse_weights {
            let rev = self.forward_direction(input, weights, true)?;
            Tensor::cat(&[&fwd, &rev], 2)
        } else {
            Ok(fwd)
        }
    }

    fn forward_direction(
        &self,
        input: &Tensor,
        weights: &LstmWeights,
        reverse: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, input_size) = input.dims3()?;
        let device = input.device();
        let dtype = input.dtype();

        if seq_len == 0 {
            return Tensor::zeros((batch, 0, self.hidden_size), dtype, device);
        }

        // Every input projection is independent of the recurrent state. One
        // matrix multiplication handles all positions and batch elements.
        let input_gates = input
            .reshape((batch * seq_len, input_size))?
            .matmul(&weights.weight_ih_t)?
            .broadcast_add(&weights.bias)?
            .reshape((batch, seq_len, 4 * self.hidden_size))?;

        let mut h = Tensor::zeros((batch, self.hidden_size), dtype, device)?;
        let mut c = Tensor::zeros((batch, self.hidden_size), dtype, device)?;
        let mut outputs = Vec::with_capacity(seq_len);

        for step in 0..seq_len {
            let t = if reverse { seq_len - 1 - step } else { step };
            let gates = input_gates
                .narrow(1, t, 1)?
                .squeeze(1)?
                .add(&h.matmul(&weights.weight_hh_t)?)?;

            // Split into 4 gates: input, forget, cell, output
            let gate_size = self.hidden_size;
            let i_gate = candle_nn::ops::sigmoid(&gates.narrow(1, 0, gate_size)?)?;
            let f_gate = candle_nn::ops::sigmoid(&gates.narrow(1, gate_size, gate_size)?)?;
            let g_gate = gates.narrow(1, 2 * gate_size, gate_size)?.tanh()?;
            let o_gate = candle_nn::ops::sigmoid(&gates.narrow(1, 3 * gate_size, gate_size)?)?;

            c = f_gate.mul(&c)?.add(&i_gate.mul(&g_gate)?)?;
            h = o_gate.mul(&c.tanh()?)?;

            outputs.push(h.unsqueeze(1)?);
        }

        // If reversed, reverse the outputs back to original order
        if reverse {
            outputs.reverse();
        }

        Tensor::cat(&outputs, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    // Original per-timestep equations provide an independent reference for
    // batching, bias folding, reverse order, and stacked bidirectional layers.
    fn reference_direction(
        input: &Tensor,
        weights: &HashMap<String, Tensor>,
        layer: usize,
        reverse: bool,
    ) -> Result<Tensor> {
        let suffix = if reverse { "_reverse" } else { "" };
        let weight = |kind: &str| &weights[&format!("{kind}_l{layer}{suffix}")];
        let (batch, seq, _) = input.dims3()?;
        let hidden = weight("weight_hh").dim(1)?;
        let mut h = Tensor::zeros((batch, hidden), input.dtype(), input.device())?;
        let mut c = h.clone();
        let mut outputs = Vec::new();
        for step in 0..seq {
            let t = if reverse { seq - step - 1 } else { step };
            let gates = input
                .narrow(1, t, 1)?
                .squeeze(1)?
                .contiguous()?
                .matmul(&weight("weight_ih").t()?.contiguous()?)?
                .broadcast_add(weight("bias_ih"))?
                .add(
                    &h.matmul(&weight("weight_hh").t()?.contiguous()?)?
                        .broadcast_add(weight("bias_hh"))?,
                )?;
            let i = candle_nn::ops::sigmoid(&gates.narrow(1, 0, hidden)?)?;
            let f = candle_nn::ops::sigmoid(&gates.narrow(1, hidden, hidden)?)?;
            let g = gates.narrow(1, 2 * hidden, hidden)?.tanh()?;
            let o = candle_nn::ops::sigmoid(&gates.narrow(1, 3 * hidden, hidden)?)?;
            c = (f.mul(&c)? + i.mul(&g)?)?;
            h = o.mul(&c.tanh()?)?;
            outputs.push(h.unsqueeze(1)?);
        }
        if reverse {
            outputs.reverse();
        }
        Tensor::cat(&outputs, 1)
    }

    #[test]
    fn test_prepacked_lstm_matches_timestep_reference() {
        let device = Device::Cpu;
        let hidden = 3;
        for bidirectional in [false, true] {
            let mut weights = HashMap::new();
            for layer in 0..2 {
                let input_size = if layer == 0 {
                    4
                } else {
                    hidden * if bidirectional { 2 } else { 1 }
                };
                for suffix in if bidirectional {
                    vec!["", "_reverse"]
                } else {
                    vec![""]
                } {
                    for (kind, width) in [("ih", input_size), ("hh", hidden)] {
                        let values = (0..4 * hidden * width)
                            .map(|i| ((i + layer * 3 + suffix.len()) as f32 * 0.37).sin() * 0.2)
                            .collect::<Vec<_>>();
                        weights.insert(
                            format!("weight_{kind}_l{layer}{suffix}"),
                            Tensor::from_vec(values, (4 * hidden, width), &device).unwrap(),
                        );
                        let values = (0..4 * hidden)
                            .map(|i| (i as f32 * 0.13).cos() * 0.05)
                            .collect::<Vec<_>>();
                        weights.insert(
                            format!("bias_{kind}_l{layer}{suffix}"),
                            Tensor::new(values.as_slice(), &device).unwrap(),
                        );
                    }
                }
            }
            let input = Tensor::new(
                (0..40).map(|i| (i as f32 * 0.17).sin()).collect::<Vec<_>>(),
                &device,
            )
            .unwrap()
            .reshape((2, 4, 5))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
            let model = Lstm::load(
                2,
                4,
                hidden,
                bidirectional,
                VarBuilder::from_tensors(weights.clone(), input.dtype(), &device),
            )
            .unwrap();
            let actual = model.forward(&input).unwrap();
            let mut expected = input;
            for layer in 0..2 {
                let forward = reference_direction(&expected, &weights, layer, false).unwrap();
                expected = if bidirectional {
                    Tensor::cat(
                        &[
                            forward,
                            reference_direction(&expected, &weights, layer, true).unwrap(),
                        ],
                        2,
                    )
                    .unwrap()
                } else {
                    forward
                };
            }
            let error = actual
                .sub(&expected)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(error < 1e-6, "LSTM parity error: {error}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_prepacked_lstm_metal_matches_cpu() {
        let metal = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("Skipping Metal LSTM test: {error}");
                return;
            }
        };
        let mut weights = HashMap::new();
        for suffix in ["", "_reverse"] {
            for (kind, input) in [("ih", 6), ("hh", 4)] {
                let values = (0..16 * input)
                    .map(|i| (i as f32 * 0.19).sin() * 0.1)
                    .collect::<Vec<_>>();
                weights.insert(
                    format!("weight_{kind}_l0{suffix}"),
                    Tensor::from_vec(values, (16, input), &Device::Cpu).unwrap(),
                );
                weights.insert(
                    format!("bias_{kind}_l0{suffix}"),
                    Tensor::ones(16, candle_core::DType::F32, &Device::Cpu)
                        .unwrap()
                        .affine(0.03, 0.0)
                        .unwrap(),
                );
            }
        }
        let input = Tensor::new(
            (0..60).map(|i| (i as f32 * 0.17).cos()).collect::<Vec<_>>(),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((2, 5, 6))
        .unwrap();
        let cpu = Lstm::load(
            1,
            6,
            4,
            true,
            VarBuilder::from_tensors(weights.clone(), input.dtype(), &Device::Cpu),
        )
        .unwrap();
        let gpu = Lstm::load(
            1,
            6,
            4,
            true,
            VarBuilder::from_tensors(weights, input.dtype(), &metal),
        )
        .unwrap();
        let expected = cpu.forward(&input).unwrap();
        let actual = gpu
            .forward(&input.to_device(&metal).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let error = actual
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "CPU/Metal LSTM parity error: {error}");
    }

    #[test]
    fn test_lstm_output_shape() {
        let device = Device::Cpu;
        let dtype = candle_core::DType::F32;

        let hidden = 32;
        let input_size = 16;

        // Create mock weights
        let weight_ih = Tensor::randn(0f32, 0.1, (4 * hidden, input_size), &device).unwrap();
        let weight_hh = Tensor::randn(0f32, 0.1, (4 * hidden, hidden), &device).unwrap();
        let bias_ih = Tensor::zeros(4 * hidden, dtype, &device).unwrap();
        let bias_hh = Tensor::zeros(4 * hidden, dtype, &device).unwrap();

        let layer = LstmLayer {
            forward_weights: LstmWeights::new(weight_ih, weight_hh, bias_ih, bias_hh).unwrap(),
            reverse_weights: None,
            hidden_size: hidden,
        };

        let lstm = Lstm {
            layers: vec![layer],
            bidirectional: false,
            hidden_size: hidden,
        };

        let input = Tensor::randn(0f32, 1.0, (2, 10, input_size), &device).unwrap();
        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 10, hidden]);
    }
}
