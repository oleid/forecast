//! Timestral model wrapper and two-stream inference integration.
//!
//! Encapsulates the native Burn model core, providing
//! causal two-stream time and variate mask generation, RoPE position embeddings,
//! role differentiation (Target, PastCovariate, KnownCovariate), and anchor-frame denormalization.

use std::path::Path;

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::checkpoint::{CONFIG, TimestralVariant, WEIGHTS};
use crate::config::TimestralConfig;
use crate::masks::{
    build_two_stream_positions, build_two_stream_time_mask, build_two_stream_variate_mask,
    compute_enc_key_validity, group_first_observed_patch,
};
use crate::network::Network;
use crate::scaler::{
    apply_affine_asinh_masked, causal_patch_loc_scale, denormalize_quantiles, known_cov_scaling,
};
use crate::{ModelError, error::checked_buffer_len};

#[cfg(test)]
#[path = "../tests/unit/test_model.rs"]
mod tests;

/// Role of a time series row in multivariate / covariate forecasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub(crate) enum RowRole {
    /// Target variable to forecast.
    #[default]
    Target = 0,
    /// Past-only covariate known over the context window only.
    PastCovariate = 1,
    /// Known-future covariate known over both context and horizon.
    KnownCovariate = 2,
}

#[cfg(test)]
impl RowRole {
    /// Returns `true` if this role is a Target variate.
    #[inline]
    const fn is_target(&self) -> bool {
        matches!(self, RowRole::Target)
    }

    /// Returns `true` if this role is a PastCovariate.
    #[inline]
    const fn is_past_covariate(&self) -> bool {
        matches!(self, RowRole::PastCovariate)
    }

    /// Returns `true` if this role is a KnownCovariate.
    #[inline]
    const fn is_known_covariate(&self) -> bool {
        matches!(self, RowRole::KnownCovariate)
    }

    /// Returns the integer code corresponding to this role.
    #[inline]
    const fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Returns the usize index corresponding to this role.
    #[inline]
    const fn as_usize(&self) -> usize {
        *self as usize
    }
}

#[cfg(test)]
const ROLE_TARGET: usize = 0;
#[cfg(test)]
const ROLE_PAST_COV: usize = 1;
#[cfg(test)]
const ROLE_KNOWN_COV: usize = 2;

#[cfg(test)]
impl From<RowRole> for usize {
    #[inline]
    fn from(role: RowRole) -> Self {
        role.as_usize()
    }
}

#[cfg(test)]
impl TryFrom<usize> for RowRole {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RowRole::Target),
            1 => Ok(RowRole::PastCovariate),
            2 => Ok(RowRole::KnownCovariate),
            _ => Err(ModelError::InvalidInput(format!(
                "invalid RowRole value: {value}, expected 0 (Target), 1 (PastCov), or 2 (KnownCov)"
            ))),
        }
    }
}

/// High-level Timestral model wrapper.
///
/// Manages configuration, underlying neural network forward pass,
/// causal scaling and instance normalization, mask construction, and denormalization.
#[derive(Clone)]
pub(crate) struct TimestralModel {
    config: TimestralConfig,
    model: Network,
    device: burn::tensor::Device,
}

impl TimestralModel {
    /// Creates a new model instance with initialized/random weights.
    #[cfg(test)]
    pub(crate) fn new(config: TimestralConfig, device: &Device) -> Result<Self, ModelError> {
        config.validate().map_err(ModelError::InvalidInput)?;
        let model = Network::new(&config, device)?;
        Ok(Self {
            config,
            model,
            device: device.clone(),
        })
    }

    /// Loads a local checkpoint, or resolves `variant` from the Hugging Face cache/Hub.
    ///
    /// When `model_dir` is given, the local configuration determines the architecture;
    /// `variant` is ignored and no network access is attempted. `None` requires the `hub` feature.
    pub(crate) fn load(
        variant: TimestralVariant,
        model_dir: Option<&Path>,
        device: &burn::tensor::Device,
    ) -> Result<Self, ModelError> {
        if let Some(dir) = model_dir {
            return Self::from_dir(dir, device);
        }
        #[cfg(feature = "hub")]
        {
            let (config, weights) = crate::checkpoint::hub_files(variant)?;
            Self::load_files(&config, &weights, device)
        }
        #[cfg(not(feature = "hub"))]
        {
            Err(ModelError::HubDisabled {
                repo: variant.repo_id(),
            })
        }
    }

    /// Loads the original `config.json` and `model.safetensors` from a local directory.
    pub(crate) fn from_dir(
        dir: impl AsRef<Path>,
        device: &burn::tensor::Device,
    ) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        Self::load_files(&dir.join(CONFIG), &dir.join(WEIGHTS), device)
    }

    fn load_files(config: &Path, weights: &Path, device: &Device) -> Result<Self, ModelError> {
        let parsed = TimestralConfig::load_from_file(config)?;
        Self::from_safetensors(parsed, weights, device)
    }

    /// Loads original Safetensors weights with an explicit model configuration.
    fn from_safetensors(
        config: TimestralConfig,
        file: impl AsRef<Path>,
        device: &burn::tensor::Device,
    ) -> Result<Self, ModelError> {
        let path_ref = file.as_ref();
        let mut model = Network::new(&config, device)?;
        let mut store = SafetensorsStore::from_file(path_ref)
            .with_key_remapping(r"\.scale$", ".gamma")
            .with_key_remapping(r"\.wQKV\.", ".w_qkv.")
            .with_key_remapping(r"\.wO\.", ".w_o.")
            .with_from_adapter(PyTorchToBurnAdapter)
            // Inspect the complete report below; no partial checkpoint is accepted.
            .allow_partial(true)
            .validate(false);
        let result = model
            .load_from(&mut store)
            .map_err(|source| ModelError::CheckpointRead {
                path: path_ref.into(),
                source: Box::new(source),
            })?;
        if !result.missing.is_empty()
            || !result.unused.is_empty()
            || !result.skipped.is_empty()
            || !result.errors.is_empty()
        {
            return Err(ModelError::IncompatibleCheckpoint {
                path: path_ref.into(),
                reason: format!(
                    "missing={:?}, unused={:?}, skipped={:?}, errors={:?}",
                    result.missing, result.unused, result.skipped, result.errors
                ),
            });
        }
        Ok(Self {
            config,
            model,
            device: device.clone(),
        })
    }

    /// Returns a reference to the active configuration.
    #[inline]
    pub(crate) fn config(&self) -> &TimestralConfig {
        &self.config
    }

    /// Direct execution of the native Burn core forward graph.
    ///
    /// # Arguments
    /// * `scaled`: `(R, W)` normalized inputs.
    /// * `observed`: `(R, W)` observation indicators (1.0 = observed, 0.0 = missing).
    /// * `time_mask`: `(R, 1, S, S)` additive attention mask.
    /// * `variate_mask`: `(S, 1, R, R)` additive variate attention mask.
    /// * `positions`: `(R, S)` RoPE position indices.
    ///
    /// # Returns
    /// Normalized quantiles tensor `(R, patch_size, num_quantiles)`.
    fn forward(
        &self,
        scaled: Tensor<2>,
        observed: Tensor<2>,
        time_mask: Tensor<4, burn::tensor::Bool>,
        variate_mask: Tensor<4, burn::tensor::Bool>,
        positions: Tensor<2>,
    ) -> Result<Tensor<3>, ModelError> {
        self.model.forward(
            &self.config,
            scaled,
            observed,
            time_mask,
            variate_mask,
            positions,
        )
    }

    /// Computes causal scaling and instance normalization for inputs `(R, T)`.
    ///
    /// Returns `(scaled, loc, scale)` flat buffers of length `R * T`.
    fn scale(
        &self,
        inputs: &[f32],
        observed: &[bool],
        role_ids: Option<&[RowRole]>,
        r: usize,
        t: usize,
    ) -> (Vec<f32>, Vec<f64>, Vec<f64>) {
        assert_eq!(inputs.len(), r * t, "inputs buffer length mismatch");
        assert_eq!(observed.len(), r * t, "observed buffer length mismatch");

        let inputs_f64: Vec<f64> = inputs.iter().map(|&x| x as f64).collect();
        let ps = self.config.patch_size;
        let min_scale = self.config.minimum_scale as f64;
        let stabilize = self.config.stabilize_with_global;

        let mut loc = Vec::with_capacity(r * t);
        let mut scale = Vec::with_capacity(r * t);
        if t != 0 {
            for (row_in, row_obs) in inputs_f64.chunks_exact(t).zip(observed.chunks_exact(t)) {
                let (r_loc, r_scale) =
                    causal_patch_loc_scale(row_in, row_obs, ps, min_scale, stabilize);
                loc.extend_from_slice(&r_loc);
                scale.extend_from_slice(&r_scale);
            }
        }

        let scaled =
            apply_affine_asinh_masked(&inputs_f64, &loc, &scale, observed, self.config.use_asinh);

        if self.config.covariate_enabled
            && let Some(roles) = role_ids
        {
            assert_eq!(roles.len(), r, "role_ids length mismatch");
            let known_rows: Vec<bool> = roles
                .iter()
                .map(|&r| r == RowRole::KnownCovariate)
                .collect();
            if known_rows.iter().any(|&k| k) {
                let (k_scaled, k_loc, k_scale) = known_cov_scaling(
                    inputs,
                    observed,
                    &scaled,
                    &loc,
                    &scale,
                    &known_rows,
                    r,
                    t,
                    self.config.minimum_scale,
                    self.config.use_asinh,
                );
                return (k_scaled, k_loc, k_scale);
            }
        }

        (scaled, loc, scale)
    }

    /// Forecasts one block of `horizon_patches` (currently exactly one patch) after context.
    ///
    /// Performs two-stream scaling, mask and position generation, model forward,
    /// and anchor-frame denormalization.
    ///
    /// # Note on 1-Patch Contract & Python Reference Parity:
    /// The Rust inference implementation outputs exactly one patch per forward pass.
    /// In contrast, Python PyTorch reference defaults to `per_shot_horizon_patches = 32`.
    /// Longer horizons in Rust are orchestrated autoregressively at the forecaster level.
    ///
    /// `horizon_patches` must strictly equal `1`. Unsupported values return `Err(ModelError::InvalidInput)`
    /// and are never silently capped or altered.
    ///
    /// # Arguments
    /// * `inputs`: Context values `[R * T]`, row-major `(R, T)` where `T` is a multiple of `patch_size`.
    /// * `observed`: Observation mask `[R * T]` (true = observed, false = missing).
    /// * `group_ids`: Series group identifier per row `[R]`.
    /// * `horizon_patches`: Number of patches to forecast in this block (must be 1).
    /// * `role_ids`: Optional row roles `[R]`. If `None`, all rows are `RowRole::Target`.
    /// * `known_future`: Optional known-future covariate values `[R * patch_size]`.
    /// * `known_future_mask`: Optional observation mask for `known_future`.
    /// * `r`: Number of rows.
    /// * `t`: Context length (number of time steps).
    ///
    /// # Returns
    /// Raw-space denormalized quantiles buffer of length `R * patch_size * Q`,
    /// or `Err(ModelError::InvalidInput)` if input shapes or arguments are invalid.
    #[expect(
        clippy::too_many_arguments,
        reason = "the block operates on aligned value, mask, and role buffers"
    )]
    pub(crate) fn predict_block(
        &self,
        inputs: &[f32],
        observed: &[bool],
        group_ids: &[usize],
        horizon_patches: usize,
        role_ids: Option<&[RowRole]>,
        known_future: Option<&[f32]>,
        known_future_mask: Option<&[bool]>,
        r: usize,
        t: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let ps = self.config.patch_size;
        if r == 0 {
            return Err(ModelError::InvalidInput(
                "number of rows r must be > 0".to_string(),
            ));
        }
        if t == 0 || !t.is_multiple_of(ps) {
            return Err(ModelError::InvalidInput(format!(
                "context length t ({t}) must be a positive multiple of patch_size ({ps})"
            )));
        }
        let context_len = checked_buffer_len::<f32>("block context", &[r, t])?;
        if inputs.len() != context_len {
            return Err(ModelError::InvalidInput(format!(
                "inputs length mismatch: expected {context_len} (r={r} * t={t}), got {}",
                inputs.len()
            )));
        }
        if observed.len() != context_len {
            return Err(ModelError::InvalidInput(format!(
                "observed length mismatch: expected {context_len} (r={r} * t={t}), got {}",
                observed.len()
            )));
        }
        if group_ids.len() != r {
            return Err(ModelError::InvalidInput(format!(
                "group_ids length mismatch: expected r={r}, got {}",
                group_ids.len()
            )));
        }
        if horizon_patches != 1 {
            return Err(ModelError::InvalidInput(format!(
                "horizon_patches must be 1 for the single-block forward pass (got {horizon_patches}). \
                 Multi-patch horizons must be autoregressively predicted block-by-block."
            )));
        }

        let horizon_len = horizon_patches * ps;
        let w = t.checked_add(horizon_len).ok_or_else(|| {
            ModelError::InvalidInput("block window length overflows usize".into())
        })?;
        let n = w / ps;
        let window_len = checked_buffer_len::<f64>("block window", &[r, w])?;
        let q = self.config.num_quantiles();
        let output_len = checked_buffer_len::<f32>("block output", &[r, horizon_len, q])?;
        let s = 2 * n;
        let ll = t / ps;
        let time_mask_len = checked_buffer_len::<bool>("time mask", &[r, s, s])?;
        checked_buffer_len::<bool>("variate mask", &[s, r, r])?;
        checked_buffer_len::<f32>("hidden states", &[r, s, 4, self.config.d_model])?;
        checked_buffer_len::<f32>("MLP activations", &[r, s, 2, self.config.mlp_hidden_dim])?;
        checked_buffer_len::<f32>("time attention", &[self.config.num_heads, time_mask_len])?;
        checked_buffer_len::<f32>("variate attention", &[self.config.num_heads, s, r, r])?;

        // Check known_future and known_future_mask presence and lengths
        match (known_future, known_future_mask) {
            (Some(kf), Some(kfm)) => {
                if kf.len() != r * horizon_len {
                    return Err(ModelError::InvalidInput(format!(
                        "known_future length mismatch: expected {} (r={r} * horizon_len={horizon_len}), got {}",
                        r * horizon_len,
                        kf.len()
                    )));
                }
                if kfm.len() != r * horizon_len {
                    return Err(ModelError::InvalidInput(format!(
                        "known_future_mask length mismatch: expected {} (r={r} * horizon_len={horizon_len}), got {}",
                        r * horizon_len,
                        kfm.len()
                    )));
                }
            }
            (Some(_), None) => {
                return Err(ModelError::InvalidInput(
                    "known_future_mask is required alongside known_future".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(ModelError::InvalidInput(
                    "known_future is required alongside known_future_mask".to_string(),
                ));
            }
            (None, None) => {}
        }

        if let Some(roles) = role_ids
            && roles.len() != r
        {
            return Err(ModelError::InvalidInput(format!(
                "role_ids length mismatch: expected r={r}, got {}",
                roles.len()
            )));
        }

        // When covariates are enabled, distinguish roles; otherwise treat all rows as Target
        let (known_rows, target_rows) = if self.config.covariate_enabled {
            match role_ids {
                Some(roles) => {
                    let kr: Vec<bool> = roles
                        .iter()
                        .map(|&role| role == RowRole::KnownCovariate)
                        .collect();
                    let tr: Vec<bool> = roles.iter().map(|&role| role == RowRole::Target).collect();
                    (kr, tr)
                }
                None => (vec![false; r], vec![true; r]),
            }
        } else {
            (vec![false; r], vec![true; r])
        };

        // Construct full_inputs (R, W) and full_observed (R, W)
        // Known future is copied only for known-covariate rows (when covariate_enabled is true).
        let mut full_inputs = vec![0.0f32; window_len];
        let mut full_observed = vec![false; window_len];

        let full_rows = full_inputs
            .chunks_exact_mut(w)
            .zip(full_observed.chunks_exact_mut(w));
        let context_rows = inputs.chunks_exact(t).zip(observed.chunks_exact(t));
        for ((full_values, full_mask), (context_values, context_mask)) in
            full_rows.zip(context_rows)
        {
            full_values[..t].copy_from_slice(context_values);
            full_mask[..t].copy_from_slice(context_mask);
        }

        if self.config.covariate_enabled
            && let (Some(future), Some(future_mask)) = (known_future, known_future_mask)
        {
            let full_rows = full_inputs
                .chunks_exact_mut(w)
                .zip(full_observed.chunks_exact_mut(w));
            let future_rows = future
                .chunks_exact(horizon_len)
                .zip(future_mask.chunks_exact(horizon_len));
            for (((full_values, full_mask), (future_values, future_mask)), &is_known) in
                full_rows.zip(future_rows).zip(&known_rows)
            {
                if is_known {
                    full_values[t..].copy_from_slice(future_values);
                    full_mask[t..].copy_from_slice(future_mask);
                }
            }
        }

        // De-duplicated scaling call on the full window
        let (final_scaled, final_loc, final_scale) = self.scale(
            &full_inputs,
            &full_observed,
            if self.config.covariate_enabled {
                role_ids
            } else {
                None
            },
            r,
            w,
        );

        // Extract anchor statistics for each row from the last context patch (step t - 1)
        let anchors = final_loc.chunks_exact(w).zip(final_scale.chunks_exact(w));

        // obs_per_patch: Vec<Vec<bool>> (R x n) and key_valid: Vec<Vec<bool>> (R x n)
        let mut obs_per_patch = Vec::with_capacity(r);
        let mut all_key_valid = Vec::with_capacity(r);
        let mut time_mask_vec = Vec::with_capacity(time_mask_len);

        for (row, &is_known) in full_observed.chunks_exact(w).zip(&known_rows) {
            let row_obs: Vec<_> = row
                .chunks_exact(ps)
                .map(|patch| patch.iter().any(|&observed| observed))
                .collect();
            let row_key_valid =
                compute_enc_key_validity(&row_obs, self.config.attend_missing_patches);
            let row_time_mask =
                build_two_stream_time_mask(ll, n, self.config.block_size, &row_key_valid, is_known);
            time_mask_vec.extend_from_slice(&row_time_mask);
            obs_per_patch.push(row_obs);
            all_key_valid.push(row_key_valid);
        }

        let variate_mask_vec = build_two_stream_variate_mask(
            group_ids,
            &all_key_valid,
            n,
            Some(&target_rows),
            Some(&known_rows),
        );

        let group_first_obs = group_first_observed_patch(&obs_per_patch, group_ids);
        let mut positions_vec = Vec::with_capacity(r * s);
        for first in group_first_obs {
            let pos_row = build_two_stream_positions(n, first);
            positions_vec.extend_from_slice(&pos_row);
        }

        // Convert to Burn Tensors
        let scaled_tensor: Tensor<2> =
            Tensor::from_data(TensorData::new(final_scaled, [r, w]), &self.device);

        let obs_f32: Vec<f32> = full_observed
            .iter()
            .map(|&o| if o { 1.0f32 } else { 0.0f32 })
            .collect();
        let observed_tensor: Tensor<2> =
            Tensor::from_data(TensorData::new(obs_f32, [r, w]), &self.device);

        let time_mask_tensor: Tensor<4, burn::tensor::Bool> =
            Tensor::from_data(TensorData::new(time_mask_vec, [r, 1, s, s]), &self.device);

        let variate_mask_tensor: Tensor<4, burn::tensor::Bool> = match variate_mask_vec {
            Some(vm) => Tensor::from_data(TensorData::new(vm, [s, 1, r, r]), &self.device),
            None => Tensor::zeros([s, 1, r, r], &self.device),
        };

        let positions_tensor: Tensor<2> =
            Tensor::from_data(TensorData::new(positions_vec, [r, s]), &self.device);

        // Model Forward Pass -> (R, ps, Q)
        let norm_preds_tensor = self.forward(
            scaled_tensor,
            observed_tensor,
            time_mask_tensor,
            variate_mask_tensor,
            positions_tensor,
        )?;

        let norm_preds_data = norm_preds_tensor.into_data();
        let norm_preds = norm_preds_data.as_slice::<f32>().map_err(|err| {
            ModelError::InvalidOutput(format!("reading prediction tensor: {err:?}"))
        })?;
        if norm_preds.len() != output_len {
            return Err(ModelError::InvalidOutput(format!(
                "prediction buffer has {} elements; expected {output_len}",
                norm_preds.len()
            )));
        }

        // Denormalize predictions using anchor stats for each row
        let mut out_preds = Vec::with_capacity(output_len);
        for (row_norm_preds, (row_loc, row_scale)) in
            norm_preds.chunks_exact(horizon_len * q).zip(anchors)
        {
            let row_denorm = denormalize_quantiles(
                row_norm_preds,
                q,
                row_loc[t - 1] as f32,
                row_scale[t - 1] as f32,
                self.config.use_asinh,
            );
            out_preds.extend_from_slice(&row_denorm);
        }

        Ok(out_preds)
    }

    /// Forecasts one block and returns the raw-space quantiles as a Burn `Tensor<3>`.
    #[cfg(test)]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror predict_block in tensor-shape tests"
    )]
    fn predict_block_tensor(
        &self,
        inputs: &[f32],
        observed: &[bool],
        group_ids: &[usize],
        horizon_patches: usize,
        role_ids: Option<&[RowRole]>,
        known_future: Option<&[f32]>,
        known_future_mask: Option<&[bool]>,
        r: usize,
        t: usize,
    ) -> Result<Tensor<3>, ModelError> {
        let preds = self.predict_block(
            inputs,
            observed,
            group_ids,
            horizon_patches,
            role_ids,
            known_future,
            known_future_mask,
            r,
            t,
        )?;
        let q = self.config.num_quantiles();
        let h_len = horizon_patches * self.config.patch_size;
        Ok(Tensor::from_data(
            TensorData::new(preds, [r, h_len, q]),
            &self.device,
        ))
    }
}
