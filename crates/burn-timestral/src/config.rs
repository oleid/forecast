use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::{ModelError, error::checked_buffer_len};

#[cfg(test)]
#[path = "../tests/unit/test_config.rs"]
mod config_tests;

const LAYER_KINDS: &[&str] = &["time", "variate", "time_variate", "variate_time"];
const ATTENTION_GATES: &[&str] = &["none", "sdpa_sigmoid"];

pub(crate) fn validate_quantile_levels(levels: &[f32]) -> Result<usize, String> {
    if levels.iter().any(|k| !(0.0..=1.0).contains(k)) {
        return Err("quantile levels must be finite probabilities in [0, 1]".into());
    }
    if levels.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("quantile levels must be strictly increasing".into());
    }
    levels
        .iter()
        .position(|&k| (k - 0.5).abs() < 1e-6)
        .ok_or_else(|| "quantile levels must contain 0.5".into())
}

fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rope_partial_factor() -> f32 {
    0.5
}
fn default_norm_eps() -> f32 {
    1e-6
}
fn default_false() -> bool {
    false
}
fn default_sdpa_sigmoid() -> String {
    "sdpa_sigmoid".to_string()
}
fn default_minimum_scale() -> f32 {
    1e-6
}
fn default_true() -> bool {
    true
}
fn default_quantile_width_factor() -> f32 {
    2.0
}

/// Model architecture and inference settings for Timestral.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimestralConfig {
    pub(crate) patch_size: usize,
    pub(crate) d_model: usize,
    pub(crate) num_heads: usize,
    pub(crate) mlp_hidden_dim: usize,
    pub(crate) block_schedule: Vec<String>,
    pub(crate) quantile_knots: Vec<f32>,
    pub(crate) block_size: usize,
    pub(crate) context_length: usize,

    #[serde(default = "default_rope_theta")]
    pub(crate) rope_theta: f32,

    #[serde(default = "default_rope_partial_factor")]
    pub(crate) rope_partial_factor: f32,

    #[serde(default = "default_norm_eps")]
    pub(crate) norm_eps: f32,

    #[serde(default = "default_false")]
    pub(crate) attn_bias: bool,

    #[serde(default = "default_false")]
    pub(crate) mlp_bias: bool,

    #[serde(default = "default_sdpa_sigmoid")]
    pub(crate) time_attn_gate: String,

    #[serde(default = "default_sdpa_sigmoid")]
    pub(crate) variate_attn_gate: String,

    #[serde(default = "default_minimum_scale")]
    pub(crate) minimum_scale: f32,

    #[serde(default = "default_true")]
    pub(crate) use_asinh: bool,

    #[serde(default = "default_false")]
    pub(crate) stabilize_with_global: bool,

    #[serde(default = "default_true")]
    pub(crate) covariate_enabled: bool,

    #[serde(default = "default_true")]
    pub(crate) split_context_horizon_attn: bool,

    #[serde(default = "default_true")]
    pub(crate) attend_missing_patches: bool,

    #[serde(default = "default_quantile_width_factor")]
    pub(crate) quantile_width_factor: f32,
}

impl TimestralConfig {
    /// Checks that the configuration is supported by the Rust inference engine.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if [
            self.patch_size,
            self.d_model,
            self.num_heads,
            self.mlp_hidden_dim,
            self.block_size,
            self.context_length,
        ]
        .contains(&0)
        {
            return Err("model dimensions and lengths must be positive".to_string());
        }
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err(format!(
                "d_model={} not divisible by num_heads={}",
                self.d_model, self.num_heads
            ));
        }
        let rope_dim = self.rope_dim();
        if !(0.0..=1.0).contains(&self.rope_partial_factor)
            || !self.head_dim().is_multiple_of(2)
            || rope_dim == 0
            || !rope_dim.is_multiple_of(2)
            || rope_dim > self.head_dim()
        {
            return Err(format!(
                "rope_partial_factor={} gives invalid rope_dim={}",
                self.rope_partial_factor, rope_dim
            ));
        }
        for (name, value) in [
            ("norm_eps", self.norm_eps),
            ("rope_theta", self.rope_theta),
            ("minimum_scale", self.minimum_scale),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        if !self.quantile_width_factor.is_finite() || self.quantile_width_factor < 0.0 {
            return Err("quantile_width_factor must be finite and non-negative".to_string());
        }
        for kind in &self.block_schedule {
            if !LAYER_KINDS.contains(&kind.as_str()) {
                return Err(format!(
                    "unknown layer kind '{}'; expected one of {:?}",
                    kind, LAYER_KINDS
                ));
            }
        }
        for gate in [&self.time_attn_gate, &self.variate_attn_gate] {
            if !ATTENTION_GATES.contains(&gate.as_str()) {
                return Err(format!(
                    "unknown attention gate '{}'; expected one of {:?}",
                    gate, ATTENTION_GATES
                ));
            }
        }
        validate_quantile_levels(&self.quantile_knots)?;
        let d = self.d_model;
        let p = self.patch_size;
        for (name, shape) in [
            ("patch projection", [2, p, 4, d]),
            ("residual projection", [4, d, d, 1]),
            ("QKV projection", [3, d, d, 1]),
            ("MLP projection", [2, d, self.mlp_hidden_dim, 1]),
            ("quantile projection", [4, d, p, self.num_quantiles()]),
        ] {
            checked_buffer_len::<f32>(name, &shape)
                .map_err(|_| format!("{name} dimensions exceed buffer limits: {shape:?}"))?;
        }
        if self.stabilize_with_global {
            return Err(
                "the advancing refine frame requires stabilize_with_global=false".to_string(),
            );
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    #[inline]
    pub(crate) fn rope_dim(&self) -> usize {
        (self.head_dim() as f32 * self.rope_partial_factor) as usize
    }

    #[inline]
    pub(crate) fn num_quantiles(&self) -> usize {
        self.quantile_knots.len()
    }

    #[inline]
    pub(crate) fn median_index(&self) -> Option<usize> {
        self.quantile_knots
            .iter()
            .position(|&k| (k - 0.5).abs() < 1e-6)
    }

    /// Returns whether the layer schedule can propagate known future covariates to target variates.
    pub(crate) fn supports_known_covariates(&self) -> bool {
        if !self.covariate_enabled {
            return false;
        }
        let mut ops: Vec<bool> = Vec::new();
        for kind in &self.block_schedule {
            match kind.as_str() {
                "time" => ops.push(true),
                "variate" => ops.push(false),
                "time_variate" => {
                    ops.push(true);
                    ops.push(false);
                }
                "variate_time" => {
                    ops.push(false);
                    ops.push(true);
                }
                _ => {}
            }
        }
        let mut later_time = false;
        let mut later_variate = false;
        for is_time in ops.into_iter().rev() {
            if is_time {
                if later_time && later_variate {
                    return true;
                }
                later_time = true;
            } else {
                later_variate = true;
            }
        }
        false
    }

    /// Returns the default configuration for Timestral-209M.
    #[cfg(test)]
    pub(crate) fn default_209m() -> Self {
        let mut block_schedule = Vec::with_capacity(24);
        for _ in 0..6 {
            block_schedule.push("time".to_string());
            block_schedule.push("time".to_string());
            block_schedule.push("time".to_string());
            block_schedule.push("time_variate".to_string());
        }

        Self {
            patch_size: 32,
            d_model: 768,
            num_heads: 12,
            mlp_hidden_dim: 2048,
            block_schedule,
            quantile_knots: vec![
                0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
            ],
            block_size: 32,
            context_length: 32768,
            rope_theta: 10000.0,
            rope_partial_factor: 0.5,
            norm_eps: 1e-6,
            attn_bias: false,
            mlp_bias: false,
            time_attn_gate: "sdpa_sigmoid".to_string(),
            variate_attn_gate: "sdpa_sigmoid".to_string(),
            minimum_scale: 1e-6,
            use_asinh: true,
            stabilize_with_global: false,
            covariate_enabled: true,
            split_context_horizon_attn: true,
            attend_missing_patches: true,
            quantile_width_factor: 3.0,
        }
    }

    #[cfg(test)]
    fn from_json_str(s: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        config.validate()?;
        Ok(config)
    }

    #[cfg(test)]
    fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub(crate) fn load_from_file(path: &Path) -> Result<Self, ModelError> {
        let contents = fs::read(path).map_err(|source| ModelError::ConfigRead {
            path: path.into(),
            source,
        })?;
        let config: Self =
            serde_json::from_slice(&contents).map_err(|source| ModelError::ConfigParse {
                path: path.into(),
                source,
            })?;
        config
            .validate()
            .map_err(|reason| ModelError::InvalidConfig {
                path: path.into(),
                reason,
            })?;
        Ok(config)
    }

    #[cfg(test)]
    pub(crate) fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let json = self
            .to_json_string()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::TimestralConfig;

    #[test]
    fn rejects_invalid_scaling_and_quantiles() {
        let config = TimestralConfig::default_209m();
        for minimum_scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                TimestralConfig {
                    minimum_scale,
                    ..config.clone()
                }
                .validate()
                .is_err()
            );
        }
        for quantile_knots in [vec![0.5, f32::NAN], vec![-0.1, 0.5], vec![0.5, 1.1]] {
            assert!(
                TimestralConfig {
                    quantile_knots,
                    ..config.clone()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            TimestralConfig {
                quantile_width_factor: -1.0,
                ..config
            }
            .validate()
            .is_err()
        );
    }
}
