//! Native Burn inference graph for the official Timestral checkpoints.
//!
//! Architecture reference: <https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral>.

use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Bool, TensorData, activation, module::attention, ops::AttentionModuleOptions};

use crate::{ModelError, config::TimestralConfig};

#[derive(Module, Debug)]
pub(crate) struct Network {
    representation: Representation,
    transformer: Transformer,
    head: QuantileHead,
}

#[derive(Module, Debug)]
struct Representation {
    patch_proj: ResidualProjection,
    forecast_query: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct Transformer {
    layers: Vec<Block>,
    out_norm: RmsNorm,
}

#[derive(Module, Debug)]
struct QuantileHead {
    projection: ResidualProjection,
}

#[derive(Module, Debug)]
struct ResidualProjection {
    hidden_layer: Linear,
    output_layer: Linear,
    residual_layer: Linear,
}

impl ResidualProjection {
    fn new(input: usize, hidden: usize, output: usize, bias: bool, device: &Device) -> Self {
        Self {
            hidden_layer: linear(input, hidden, bias, device),
            output_layer: linear(hidden, output, bias, device),
            residual_layer: linear(input, output, bias, device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.output_layer
            .forward(activation::silu(self.hidden_layer.forward(x.clone())))
            + self.residual_layer.forward(x)
    }
}

#[derive(Module, Debug)]
struct Block {
    attn_norms: Vec<RmsNorm>,
    attns: Vec<Attention>,
    mlp_norm: RmsNorm,
    mlp: SwiGlu,
}

#[derive(Module, Debug)]
struct SwiGlu {
    w12: Linear,
    w3: Linear,
}

impl SwiGlu {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let projected = self.w12.forward(x);
        let hidden = projected.dims()[2] / 2;
        let gate = projected.clone().slice_dim(2, 0..hidden);
        let up = projected.slice_dim(2, hidden..2 * hidden);
        self.w3.forward(gate * activation::silu(up))
    }
}

#[derive(Module, Debug)]
struct Attention {
    w_qkv: Linear,
    w_o: Linear,
    // Keep the checkpoint's parameter name for the optional sigmoid gate.
    _gate: Option<Linear>,
    is_time: bool,
}

struct AttentionInputs {
    time_mask: Tensor<4, Bool>,
    variate_mask: Tensor<4, Bool>,
    cos: Tensor<4>,
    sin: Tensor<4>,
    heads: usize,
    split_time: bool,
}

impl Attention {
    fn new(config: &TimestralConfig, is_time: bool, device: &Device) -> Self {
        let d = config.d_model;
        let gate = if is_time {
            &config.time_attn_gate
        } else {
            &config.variate_attn_gate
        };
        Self {
            w_qkv: linear(d, 3 * d, config.attn_bias, device),
            w_o: linear(d, d, config.attn_bias, device),
            _gate: (gate == "sdpa_sigmoid").then(|| linear(d, d, config.attn_bias, device)),
            is_time,
        }
    }

    fn forward(&self, x: Tensor<3>, inputs: &AttentionInputs) -> Tensor<3> {
        let x = if self.is_time { x } else { x.swap_dims(0, 1) };
        let [batch, seq, d] = x.dims();
        let heads = inputs.heads;
        let head_dim = d / heads;
        // The fused projection stores head index as the innermost axis, not head dimension.
        let qkv = self
            .w_qkv
            .forward(x.clone())
            .reshape([batch, seq, 3, head_dim, heads]);
        let split = |i| {
            qkv.clone()
                .slice_dim(2, i..i + 1)
                .squeeze_dim::<4>(2)
                .permute([0, 3, 1, 2])
        };
        let (mut q, mut k, v) = (split(0), split(1), split(2));
        if self.is_time {
            q = rotate(q, &inputs.cos, &inputs.sin);
            k = rotate(k, &inputs.cos, &inputs.sin);
        }

        let out = if self.is_time && inputs.split_time {
            let enc = seq / 2;
            let mask = &inputs.time_mask;
            // Encoder queries cannot attend to decoder keys. Avoid that empty quadrant.
            let context_mask = mask
                .clone()
                .slice_dim(2, 0..enc)
                .slice_dim(3, 0..enc)
                .expand([batch, heads, enc, enc]);
            let context = attention(
                q.clone().slice_dim(2, 0..enc),
                k.clone().slice_dim(2, 0..enc),
                v.clone().slice_dim(2, 0..enc),
                Some(context_mask),
                None,
                AttentionModuleOptions::default(),
            );
            let decoder_mask =
                mask.clone()
                    .slice_dim(2, enc..seq)
                    .expand([batch, heads, seq - enc, seq]);
            let decoder = attention(
                q.slice_dim(2, enc..seq),
                k,
                v,
                Some(decoder_mask),
                None,
                AttentionModuleOptions::default(),
            );
            Tensor::cat(vec![context, decoder], 2)
        } else {
            let mask = if self.is_time {
                &inputs.time_mask
            } else {
                &inputs.variate_mask
            };
            attention(
                q,
                k,
                v,
                Some(mask.clone().expand([batch, heads, seq, seq])),
                None,
                AttentionModuleOptions::default(),
            )
        };
        let mut out = out.swap_dims(1, 2).reshape([batch, seq, d]);
        if let Some(gate) = &self._gate {
            out = out * activation::sigmoid(gate.forward(x));
        }
        self.w_o.forward(if self.is_time {
            out
        } else {
            out.swap_dims(0, 1)
        })
    }
}

fn linear(input: usize, output: usize, bias: bool, device: &Device) -> Linear {
    LinearConfig::new(input, output)
        .with_bias(bias)
        .init(device)
}

fn rotate(x: Tensor<4>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<4> {
    let dim = x.dims()[3];
    let half = dim / 2;
    let left = x.clone().slice_dim(3, 0..half);
    let right = x.clone().slice_dim(3, half..dim);
    x * cos.clone() + Tensor::cat(vec![right.neg(), left], 3) * sin.clone()
}

fn rope(positions: Tensor<2>, config: &TimestralConfig) -> (Tensor<4>, Tensor<4>) {
    let dim = config.rope_dim();
    // RoPE pairs coordinates across the two halves of the full head dimension.
    let frequencies: Vec<f32> = (0..config.head_dim() / 2)
        .map(|i| {
            if 2 * i < dim {
                config.rope_theta.powf(2.0 * i as f32 / dim as f32).recip()
            } else {
                0.0
            }
        })
        .collect();
    let half = frequencies.len();
    let frequencies = Tensor::<3>::from_data(
        TensorData::new(frequencies, [1, 1, half]),
        &positions.device(),
    );
    let angles = positions.unsqueeze_dim::<3>(2) * frequencies;
    let angles = Tensor::cat(vec![angles.clone(), angles], 2).unsqueeze_dim::<4>(1);
    (angles.clone().cos(), angles.sin())
}

fn monotone_quantiles(raw: Tensor<3>, config: &TimestralConfig, m: usize) -> Tensor<3> {
    let q = config.num_quantiles();
    let median = raw.clone().slice_dim(2, m..m + 1);
    let knots = &config.quantile_knots;
    let gap = |pair: &[f32]| pair[1] - pair[0];
    let mut widths = Vec::with_capacity(q);
    widths.extend(knots[..=m].windows(2).map(gap));
    widths.push(0.0);
    widths.extend(knots[m..].windows(2).map(gap));
    let widths = Tensor::<3>::from_data(TensorData::new(widths, [1, 1, q]), &raw.device());
    let increments = (widths * config.quantile_width_factor) * activation::softplus(raw, 1.0);
    let mut parts = Vec::with_capacity(3);
    if m > 0 {
        let down = increments
            .clone()
            .slice_dim(2, 0..m)
            .flip([2])
            .cumsum(2)
            .flip([2]);
        parts.push(median.clone() - down);
    }
    parts.push(median.clone());
    if m + 1 < q {
        parts.push(median + increments.slice_dim(2, m + 1..q).cumsum(2));
    }
    Tensor::cat(parts, 2).clamp(-f32::MAX, f32::MAX)
}

impl Network {
    pub(crate) fn new(config: &TimestralConfig, device: &Device) -> Result<Self, ModelError> {
        let d = config.d_model;
        let norm = || {
            RmsNormConfig::new(d)
                .with_epsilon(config.norm_eps.into())
                .init(device)
        };
        let layers = config
            .block_schedule
            .iter()
            .map(|kind| {
                let axes: &[bool] = match kind.as_str() {
                    "time" => &[true],
                    "variate" => &[false],
                    "time_variate" => &[true, false],
                    "variate_time" => &[false, true],
                    _ => {
                        return Err(ModelError::InvalidInput(format!(
                            "unknown layer kind {kind}"
                        )));
                    }
                };
                Ok(Block {
                    attn_norms: axes.iter().map(|_| norm()).collect(),
                    attns: axes
                        .iter()
                        .map(|&is_time| Attention::new(config, is_time, device))
                        .collect(),
                    mlp_norm: norm(),
                    mlp: SwiGlu {
                        w12: linear(d, 2 * config.mlp_hidden_dim, config.mlp_bias, device),
                        w3: linear(config.mlp_hidden_dim, d, config.mlp_bias, device),
                    },
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            representation: Representation {
                patch_proj: ResidualProjection::new(2 * config.patch_size, 4 * d, d, true, device),
                forecast_query: Initializer::Zeros.init([d], device),
            },
            transformer: Transformer {
                layers,
                out_norm: norm(),
            },
            head: QuantileHead {
                projection: ResidualProjection::new(
                    d,
                    4 * d,
                    config.patch_size * config.num_quantiles(),
                    false,
                    device,
                ),
            },
        })
    }

    pub(crate) fn forward(
        &self,
        config: &TimestralConfig,
        scaled: Tensor<2>,
        observed: Tensor<2>,
        time_mask: Tensor<4, Bool>,
        variate_mask: Tensor<4, Bool>,
        positions: Tensor<2>,
    ) -> Result<Tensor<3>, ModelError> {
        let median = config.median_index().ok_or_else(|| {
            ModelError::InvalidInput("configuration has no median quantile".into())
        })?;
        let [rows, window] = scaled.dims();
        let ps = config.patch_size;
        let patches = window / ps;
        let values = scaled.reshape([rows, patches, ps]);
        let missing = (observed.neg() + 1.0).reshape([rows, patches, ps]);
        let enc = self
            .representation
            .patch_proj
            .forward(Tensor::cat(vec![values, missing], 2));
        let dec = self
            .representation
            .forecast_query
            .val()
            .reshape([1, 1, config.d_model])
            .expand([rows, patches, config.d_model]);
        let mut x = Tensor::cat(vec![enc, dec], 1);
        let (cos, sin) = rope(positions, config);
        let inputs = AttentionInputs {
            time_mask,
            variate_mask,
            cos,
            sin,
            heads: config.num_heads,
            split_time: config.split_context_horizon_attn,
        };
        for block in &self.transformer.layers {
            for (norm, attn) in block.attn_norms.iter().zip(&block.attns) {
                x = x.clone() + attn.forward(norm.forward(x), &inputs);
            }
            x = x.clone() + block.mlp.forward(block.mlp_norm.forward(x));
        }
        let last = x.slice_dim(1, 2 * patches - 1..2 * patches);
        let raw = self
            .head
            .projection
            .forward(self.transformer.out_norm.forward(last))
            .reshape([rows, ps, config.num_quantiles()]);
        Ok(monotone_quantiles(raw, config, median))
    }
}
