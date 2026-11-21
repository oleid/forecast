//! Autoregressive Forecaster Engine with Covariate Support and Clampsym.
//!
//! Provides the [`TimestralForecaster`] pipeline, which performs block-autoregressive
//! probabilistic forecasting for univariate and multivariate time series with support for:
//! - Past-only numeric and categorical covariates (`past_covariates`, `categorical_past_covariates`).
//! - Known-future numeric and categorical covariates (`known_covariates`, `categorical_known_covariates`).
//! - Smoothed mean target-encoding for categorical features (`target_encode`).
//! - Two-stream causal and variate firewall masking with RoPE position tracking.
//! - Clampsym test-time augmentation (symmetric averaging and non-negative context clipping).
//! - Quantile real-space bounding (`quantile_real_cap_k`).

use std::path::Path;

use burn::prelude::*;
use burn::tensor::TensorData;
use forecast_core::{ForecastRequest, Forecaster, QuantileForecaster};

use crate::config::{TimestralConfig, validate_quantile_levels};
use crate::covariates::target_encode;
use crate::error::checked_buffer_len;
use crate::model::{RowRole, TimestralModel};
use crate::scaler::causal_patch_loc_scale;
use crate::{ModelError, TimestralVariant};

#[cfg(test)]
#[path = "../tests/unit/test_forecaster.rs"]
mod tests;

/// Default smoothing parameter for mean target-encoding of categorical covariates (1.0).
const DEFAULT_TARGET_ENCODE_SMOOTH: f64 = 1.0;

/// Container for flat numeric covariate data `(B, C, L)`.
#[derive(Debug, Clone, PartialEq)]
struct CovariateData {
    /// Flat row-major values of shape `(B, C, L)`.
    values: Vec<f32>,
    /// Number of covariate channels C.
    num_channels: usize,
}

impl CovariateData {
    /// Creates a new numeric covariate container.
    fn new(values: Vec<f32>, num_channels: usize) -> Self {
        Self {
            values,
            num_channels,
        }
    }

    /// Returns a new `CovariateData` with all values negated.
    fn negate(&self) -> Self {
        let negated = self.values.iter().map(|&x| -x).collect();
        Self {
            values: negated,
            num_channels: self.num_channels,
        }
    }
}

/// Container for flat categorical category codes `(B, C, L)`.
#[derive(Debug, Clone, PartialEq)]
struct CategoricalData {
    /// Flat row-major category codes of shape `(B, C, L)`.
    codes: Vec<f64>,
    /// Number of categorical channels C.
    num_channels: usize,
}

impl CategoricalData {
    /// Creates a new categorical codes container.
    fn new(codes: Vec<f64>, num_channels: usize) -> Self {
        Self {
            codes,
            num_channels,
        }
    }
}

/// Options configuring the forecasting pass.
#[derive(Debug, Clone)]
pub struct ForecastOptions {
    /// Optional observation mask shaped `(B, V, T)`. True = observed, False = missing.
    observation_mask: Option<Vec<bool>>,
    /// Past-only numeric covariates shaped `(B, C_past, T)`.
    past_covariates: Option<CovariateData>,
    /// Known-future numeric covariates shaped `(B, C_known, T + H)`.
    known_covariates: Option<CovariateData>,
    /// Past-only categorical category codes shaped `(B, C_cat_past, T)`.
    categorical_past_covariates: Option<CategoricalData>,
    /// Known-future categorical category codes shaped `(B, C_cat_known, T + H)`.
    categorical_known_covariates: Option<CategoricalData>,
    /// Symmetric averaging (reflects and negates context + covariates, averages with original).
    symmetric_averaging: bool,
    /// Non-negative clipping (clamps to >= 0 for series with non-negative observed context).
    clip_nonnegative: bool,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            observation_mask: None,
            past_covariates: None,
            known_covariates: None,
            categorical_past_covariates: None,
            categorical_known_covariates: None,
            symmetric_averaging: true,
            clip_nonnegative: true,
        }
    }
}

impl ForecastOptions {
    /// Sets a row-major `(B, V, T)` mask. Only finite values marked `true` are observed.
    pub fn with_observation_mask(mut self, mask: Vec<bool>) -> Self {
        self.observation_mask = Some(mask);
        self
    }

    /// Sets row-major `(B, C, T)` past-only numeric covariates.
    pub fn with_past_covariates(mut self, values: Vec<f32>, num_channels: usize) -> Self {
        self.past_covariates = Some(CovariateData::new(values, num_channels));
        self
    }

    /// Sets row-major `(B, C, T + H)` known-future numeric covariates.
    pub fn with_known_covariates(mut self, values: Vec<f32>, num_channels: usize) -> Self {
        self.known_covariates = Some(CovariateData::new(values, num_channels));
        self
    }

    /// Sets row-major `(B, C, T)` past-only category codes. Use `NaN` for missing codes.
    pub fn with_categorical_past_covariates(
        mut self,
        codes: Vec<f64>,
        num_channels: usize,
    ) -> Self {
        self.categorical_past_covariates = Some(CategoricalData::new(codes, num_channels));
        self
    }

    /// Sets row-major `(B, C, T + H)` known-future category codes. Use `NaN` for missing codes.
    pub fn with_categorical_known_covariates(
        mut self,
        codes: Vec<f64>,
        num_channels: usize,
    ) -> Self {
        self.categorical_known_covariates = Some(CategoricalData::new(codes, num_channels));
        self
    }

    /// Sets whether to perform symmetric averaging.
    pub fn with_symmetric_averaging(mut self, enabled: bool) -> Self {
        self.symmetric_averaging = enabled;
        self
    }

    /// Sets whether to clamp non-negative contexts to >= 0.
    pub fn with_clip_nonnegative(mut self, enabled: bool) -> Self {
        self.clip_nonnegative = enabled;
        self
    }
}

/// Container for quantile forecast results.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    quantiles: Vec<f32>,
    quantile_levels: Vec<f32>,
    batch_size: usize,
    num_variates: usize,
    prediction_length: usize,
    median_index: usize,
}

impl Forecast {
    /// Creates a forecast from a flat `(B, V, H, Q)` buffer and sorted levels including 0.5.
    pub fn new(
        quantiles: Vec<f32>,
        quantile_levels: Vec<f32>,
        batch_size: usize,
        num_variates: usize,
        prediction_length: usize,
    ) -> Result<Self, ModelError> {
        if batch_size == 0 || num_variates == 0 || prediction_length == 0 {
            return Err(ModelError::InvalidInput(
                "forecast dimensions must be positive".into(),
            ));
        }
        let median_index =
            validate_quantile_levels(&quantile_levels).map_err(ModelError::InvalidInput)?;
        let num_quantiles = quantile_levels.len();
        let expected_len = checked_size(
            "quantiles",
            [batch_size, num_variates, prediction_length, num_quantiles],
        )?;
        if quantiles.len() != expected_len {
            return Err(ModelError::InvalidInput(format!(
                "quantiles length {} does not match expected B*V*H*Q = {}*{}*{}*{} = {}",
                quantiles.len(),
                batch_size,
                num_variates,
                prediction_length,
                num_quantiles,
                expected_len
            )));
        }
        Ok(Self {
            quantiles,
            quantile_levels,
            batch_size,
            num_variates,
            prediction_length,
            median_index,
        })
    }

    /// Returns the shape `(B, V, H, Q)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (
            self.batch_size,
            self.num_variates,
            self.prediction_length,
            self.quantile_levels.len(),
        )
    }

    /// Returns the flat row-major `(B, V, H, Q)` buffer.
    pub fn predictions(&self) -> &[f32] {
        &self.quantiles
    }

    /// Returns the quantile probability levels in buffer order.
    pub fn quantile_levels(&self) -> &[f32] {
        &self.quantile_levels
    }

    /// Returns the `(B, V, H)` median forecast.
    pub fn median(&self) -> Vec<f32> {
        self.quantile_by_index(self.median_index)
    }

    /// Returns the `(B, V, H)` forecast for the specified quantile level (e.g. 0.1, 0.5, 0.9).
    pub fn quantile(&self, level: f32) -> Result<Vec<f32>, ModelError> {
        let idx = self
            .quantile_levels
            .iter()
            .position(|&lvl| (lvl - level).abs() < 1e-5)
            .ok_or_else(|| {
                ModelError::InvalidInput(format!(
                    "quantile level {} not found in quantile_levels: {:?}",
                    level, self.quantile_levels
                ))
            })?;
        Ok(self.quantile_by_index(idx))
    }

    fn quantile_by_index(&self, q_idx: usize) -> Vec<f32> {
        debug_assert!(q_idx < self.quantile_levels.len());
        self.quantiles
            .chunks_exact(self.quantile_levels.len())
            .map(|row| row[q_idx])
            .collect()
    }

    /// Returns a single quantile value, or `None` if any index is out of bounds.
    #[inline]
    pub fn get(&self, b: usize, v: usize, h: usize, q: usize) -> Option<f32> {
        let n_quantiles = self.quantile_levels.len();
        if b >= self.batch_size
            || v >= self.num_variates
            || h >= self.prediction_length
            || q >= n_quantiles
        {
            return None;
        }
        let idx = ((b * self.num_variates + v) * self.prediction_length + h) * n_quantiles + q;
        self.quantiles.get(idx).copied()
    }

    /// Converts the forecast quantiles to a Burn `Tensor<4>`.
    pub fn into_tensor(self, device: &burn::tensor::Device) -> Tensor<4> {
        let shape = burn::tensor::Shape::new([
            self.batch_size,
            self.num_variates,
            self.prediction_length,
            self.quantile_levels.len(),
        ]);
        Tensor::from_data(TensorData::new(self.quantiles, shape), device)
    }
}

/// Forward-fill unobserved positions; leading unobserved positions become 0.0.
fn ffill_with_mask(inputs: &[f32], mask: &[bool], num_rows: usize, length: usize) -> Vec<f32> {
    assert_eq!(inputs.len(), num_rows * length);
    assert_eq!(mask.len(), num_rows * length);

    let mut out = Vec::with_capacity(inputs.len());
    for r in 0..num_rows {
        let row_in = &inputs[r * length..(r + 1) * length];
        let row_mask = &mask[r * length..(r + 1) * length];

        let mut last_val = 0.0f32;
        let mut has_seen_obs = false;

        for t in 0..length {
            if row_mask[t] {
                let v = if row_in[t].is_nan() { 0.0 } else { row_in[t] };
                last_val = v;
                has_seen_obs = true;
                out.push(v);
            } else if has_seen_obs {
                out.push(last_val);
            } else {
                out.push(0.0);
            }
        }
    }
    out
}

/// Left-pads inputs and observation mask to a multiple of `patch_size`.
/// Padding values are 0.0 and unobserved (`false`).
fn pad_to_patch(
    inputs: &[f32],
    mask: &[bool],
    num_rows: usize,
    length: usize,
    patch_size: usize,
) -> (Vec<f32>, Vec<bool>, usize) {
    assert_eq!(inputs.len(), num_rows * length);
    assert_eq!(mask.len(), num_rows * length);

    assert!(patch_size > 0, "patch_size must be positive");
    if length.is_multiple_of(patch_size) {
        return (inputs.to_vec(), mask.to_vec(), length);
    }

    let pad_len = patch_size - (length % patch_size);
    let padded_len = length + pad_len;
    let mut out_inputs = Vec::with_capacity(num_rows * padded_len);
    let mut out_mask = Vec::with_capacity(num_rows * padded_len);

    for r in 0..num_rows {
        // Left pad with 0.0 / false
        out_inputs.extend(std::iter::repeat_n(0.0, pad_len));
        out_mask.extend(std::iter::repeat_n(false, pad_len));
        // Append original row
        out_inputs.extend_from_slice(&inputs[r * length..(r + 1) * length]);
        out_mask.extend_from_slice(&mask[r * length..(r + 1) * length]);
    }

    (out_inputs, out_mask, padded_len)
}

/// Replaces non-finite values or values with magnitude >= `threshold` with `replacement`.
#[inline]
#[cfg(test)]
fn replace_extreme_values(x: &mut [f32], threshold: f32, replacement: f32) {
    for v in x.iter_mut() {
        if !v.is_finite() || v.abs() >= threshold {
            *v = replacement;
        }
    }
}

/// Determines for each series `(B, V)` whether every observed context value is non-negative (and at least one exists).
fn nonnegative_context(
    inputs: &[f32],
    observation_mask: &[bool],
    batch_size: usize,
    num_variates: usize,
    length: usize,
) -> Vec<bool> {
    assert_eq!(inputs.len(), batch_size * num_variates * length);
    assert_eq!(observation_mask.len(), batch_size * num_variates * length);

    let num_series = batch_size * num_variates;
    let mut eligible = Vec::with_capacity(num_series);

    for s in 0..num_series {
        let row_in = &inputs[s * length..(s + 1) * length];
        let row_mask = &observation_mask[s * length..(s + 1) * length];

        let mut has_observed = false;
        let mut all_nonneg = true;

        for t in 0..length {
            let obs = row_in[t].is_finite() && row_mask[t];
            if obs {
                has_observed = true;
                if row_in[t] < 0.0 {
                    all_nonneg = false;
                    break;
                }
            }
        }

        eligible.push(has_observed && all_nonneg);
    }

    eligible
}

/// High-level autoregressive probabilistic forecaster.
#[derive(Clone)]
pub struct TimestralForecaster {
    model: TimestralModel,
    max_context: usize,
    quantile_real_cap_k: f32,
    extreme_value_threshold: f32,
    quantile_levels: Vec<f32>,
    median_index: usize,
}

impl TimestralForecaster {
    /// Creates a new `TimestralForecaster` instance.
    ///
    /// # Note on `per_shot_horizon_patches` and reference parity
    /// In the Python PyTorch reference implementation, `per_shot_horizon_patches` defaults to `32`
    /// (decoding up to 32 patches = 1024 time steps in a single shot).
    ///
    /// The Rust model evaluates exactly one patch per forward pass (`per_shot_horizon_patches = 1`).
    /// Longer horizons are decoded autoregressively patch-by-patch with median feedback
    /// and known-covariate slicing.
    ///
    /// The requested `per_shot_horizon_patches` must explicitly be `1`. If any other value is passed,
    /// this constructor returns [`ModelError::InvalidInput`]. Parity tests against Python should
    /// configure `per_shot_horizon_patches=1`
    /// (or `per_shot=1`). If provided, `max_context` must be positive.
    fn new(
        model: TimestralModel,
        max_context: Option<usize>,
        per_shot_horizon_patches: usize,
        quantile_real_cap_k: f32,
        extreme_value_threshold: f32,
    ) -> Result<Self, ModelError> {
        if per_shot_horizon_patches != 1 {
            return Err(ModelError::InvalidInput(format!(
                "per_shot_horizon_patches must be 1 for the Rust inference engine (got {per_shot_horizon_patches}). \
                 Multi-block horizons are autoregressively decoded 1 patch per step, matching Python reference configured with per_shot_horizon_patches=1."
            )));
        }

        let max_ctx = max_context.unwrap_or(model.config().context_length);
        let quantile_levels = model.config().quantile_knots.clone();
        let median_index = model.config().median_index().ok_or_else(|| {
            ModelError::InvalidInput("configuration has no median quantile".into())
        })?;

        Self {
            model,
            max_context: max_ctx,
            quantile_real_cap_k,
            extreme_value_threshold,
            quantile_levels,
            median_index,
        }
        .with_max_context(max_ctx)
    }

    /// Creates a forecaster around a model with default settings (`per_shot_horizon_patches = 1`).
    fn from_model(model: TimestralModel) -> Result<Self, ModelError> {
        Self::new(model, None, 1, 1e4, 1e30)
    }

    /// Loads a forecaster from a local directory, or from the Hugging Face cache/Hub.
    ///
    /// An explicit `model_dir` takes precedence over `variant` and never uses the network.
    /// Passing `None` requires the `hub` feature and resolves the variant's pinned snapshot.
    pub fn load(
        variant: TimestralVariant,
        model_dir: Option<&Path>,
        device: &burn::tensor::Device,
    ) -> Result<Self, ModelError> {
        Self::from_model(TimestralModel::load(variant, model_dir, device)?)
    }

    /// Loads a forecaster from a local directory containing the original checkpoint files.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        device: &burn::tensor::Device,
    ) -> Result<Self, ModelError> {
        Self::from_model(TimestralModel::from_dir(dir, device)?)
    }

    /// Sets the maximum retained context in time steps. A zero limit is invalid.
    pub fn with_max_context(mut self, max_context: usize) -> Result<Self, ModelError> {
        if max_context == 0 {
            return Err(ModelError::InvalidInput(
                "max_context must be positive".into(),
            ));
        }
        self.max_context = max_context;
        Ok(self)
    }

    /// Returns a reference to the active configuration.
    #[inline]
    fn config(&self) -> &TimestralConfig {
        self.model.config()
    }

    /// Returns the patch size.
    #[inline]
    pub fn patch_size(&self) -> usize {
        self.config().patch_size
    }

    /// Returns the number of quantile knots (e.g. 21).
    #[inline]
    pub fn num_quantiles(&self) -> usize {
        self.quantile_levels.len()
    }

    /// Returns whether the model supports covariates.
    #[inline]
    pub fn supports_covariates(&self) -> bool {
        self.config().covariate_enabled
    }

    /// Returns whether the model supports known-future covariates.
    #[inline]
    pub fn supports_known_covariates(&self) -> bool {
        self.config().supports_known_covariates()
    }

    /// Returns the maximum context length in time steps.
    #[inline]
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Returns the slice of quantile probability knots.
    #[inline]
    pub fn quantile_levels(&self) -> &[f32] {
        &self.quantile_levels
    }

    /// Forecasts `prediction_length` steps ahead given context `(B, V, T)` and options.
    ///
    /// # Arguments
    /// * `context`: Flat row-major array `(B, V, T)`. NaN marks missing values.
    /// * `batch_size`: Batch size B.
    /// * `num_variates`: Number of target variates V.
    /// * `context_length`: Context length T.
    /// * `prediction_length`: Forecasting horizon H.
    /// * `options`: Optional configuration (covariates, symmetric averaging, non-negative clipping).
    ///
    /// # Returns
    /// [`Forecast`] containing quantiles `(B, V, H, Q)`.
    pub fn forecast(
        &self,
        context: &[f32],
        batch_size: usize,
        num_variates: usize,
        context_length: usize,
        prediction_length: usize,
        options: Option<&ForecastOptions>,
    ) -> Result<Forecast, ModelError> {
        let request = ForecastRequest::new(
            context,
            batch_size,
            num_variates,
            context_length,
            prediction_length,
        )?;
        let quantiles = self.forecast_quantiles_with_options(&request, options)?;

        Forecast::new(
            quantiles,
            self.quantile_levels.clone(),
            batch_size,
            num_variates,
            prediction_length,
        )
    }

    fn forecast_quantiles_with_options(
        &self,
        request: &ForecastRequest<'_>,
        options: Option<&ForecastOptions>,
    ) -> Result<Vec<f32>, ModelError> {
        request.quantile_output_len(self.num_quantiles())?;
        let context = request.context();
        let batch_size = request.batch_size();
        let num_variates = request.num_variates();
        let context_length = request.context_length();
        let prediction_length = request.prediction_length();

        let full_length = context_length
            .checked_add(prediction_length)
            .ok_or_else(|| {
                ModelError::InvalidInput(
                    "context_length + prediction_length overflows usize".into(),
                )
            })?;
        let patch_size = self.patch_size();
        checked_size(
            "padded forecast",
            [
                batch_size,
                num_variates,
                prediction_length.div_ceil(patch_size),
                patch_size,
                self.num_quantiles(),
            ],
        )?;

        let default_opt = ForecastOptions::default();
        let opt = options.unwrap_or(&default_opt);

        let (inputs, mask) = self._as_context(
            context,
            opt.observation_mask.as_deref(),
            batch_size,
            num_variates,
            context_length,
        )?;

        let past = self._validate_covariates(
            opt.past_covariates.as_ref(),
            batch_size,
            context_length,
            "past_covariates",
        )?;
        let known = self._validate_covariates(
            opt.known_covariates.as_ref(),
            batch_size,
            full_length,
            "known_covariates",
        )?;

        let cat_past = self._validate_categorical(
            opt.categorical_past_covariates.as_ref(),
            batch_size,
            context_length,
            "categorical_past_covariates",
        )?;
        let cat_known = self._validate_categorical(
            opt.categorical_known_covariates.as_ref(),
            batch_size,
            full_length,
            "categorical_known_covariates",
        )?;

        let (mut final_past, mut final_known) = (past, known);

        if cat_past.is_some() || cat_known.is_some() {
            let target_context =
                self._encoding_target(&inputs, &mask, batch_size, num_variates, context_length);
            let enc_past = self._encode_categorical(
                cat_past.as_ref(),
                target_context.as_deref(),
                batch_size,
                context_length,
                context_length,
            )?;
            let enc_known = self._encode_categorical(
                cat_known.as_ref(),
                target_context.as_deref(),
                batch_size,
                context_length,
                full_length,
            )?;

            final_past = _cat_covariates(final_past, enc_past, batch_size, context_length)?;
            final_known = _cat_covariates(final_known, enc_known, batch_size, full_length)?;
        }

        if final_known.is_some() && !self.supports_known_covariates() {
            return Err(ModelError::InvalidInput(
                "this checkpoint cannot condition on known-future covariates".to_string(),
            ));
        }

        if opt.symmetric_averaging {
            let levels = &self.quantile_levels;
            let n_knots = levels.len();
            for i in 0..n_knots {
                let lo = levels[i];
                let hi = levels[n_knots - 1 - i];
                if (lo + hi - 1.0).abs() >= 1e-4 {
                    return Err(ModelError::InvalidInput(format!(
                        "symmetric averaging requires quantile levels symmetric about 0.5, got {:?}",
                        levels
                    )));
                }
            }
        }

        let mut quantiles = self._forecast_raw(
            &inputs,
            &mask,
            batch_size,
            num_variates,
            context_length,
            prediction_length,
            final_past.as_ref(),
            final_known.as_ref(),
        )?;

        if opt.symmetric_averaging {
            let n_knots = self.num_quantiles();

            let neg_inputs: Vec<f32> = inputs.iter().map(|&x| -x).collect();
            let neg_past = final_past.as_ref().map(|p| p.negate());
            let neg_known = final_known.as_ref().map(|k| k.negate());

            let mirrored = self._forecast_raw(
                &neg_inputs,
                &mask,
                batch_size,
                num_variates,
                context_length,
                prediction_length,
                neg_past.as_ref(),
                neg_known.as_ref(),
            )?;

            for (quantile_set, mirrored_set) in quantiles
                .chunks_exact_mut(n_knots)
                .zip(mirrored.chunks_exact(n_knots))
            {
                for (quantile, &mirr_val) in quantile_set.iter_mut().zip(mirrored_set.iter().rev())
                {
                    let orig_val = *quantile;
                    let difference = orig_val - mirr_val;
                    *quantile = if difference.is_infinite() {
                        0.5 * orig_val - 0.5 * mirr_val
                    } else {
                        0.5 * difference
                    };
                }
            }
        }

        if opt.clip_nonnegative {
            let eligible =
                nonnegative_context(&inputs, &mask, batch_size, num_variates, context_length);
            let n_knots = self.num_quantiles();
            let quantiles_per_series = prediction_length * n_knots;
            for (&is_eligible, series_quantiles) in eligible
                .iter()
                .zip(quantiles.chunks_exact_mut(quantiles_per_series))
            {
                if is_eligible {
                    for quantile_set in series_quantiles.chunks_exact_mut(n_knots) {
                        for quantile in quantile_set {
                            if *quantile < 0.0 {
                                *quantile = 0.0;
                            }
                        }
                    }
                }
            }
        }

        Ok(quantiles)
    }

    /// Convenience forecasting method for 1D univariate series `(T,)`.
    pub fn forecast_1d(
        &self,
        context: &[f32],
        prediction_length: usize,
        options: Option<&ForecastOptions>,
    ) -> Result<Forecast, ModelError> {
        self.forecast(context, 1, 1, context.len(), prediction_length, options)
    }

    /// Convenience forecasting method for Burn `Tensor<3>` context `(B, V, T)`.
    pub fn forecast_tensor(
        &self,
        context: Tensor<3>,
        prediction_length: usize,
        options: Option<&ForecastOptions>,
    ) -> Result<Forecast, ModelError> {
        let [b, v, t] = context.dims();
        let data = context.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            ModelError::InvalidInput(format!(
                "failed to convert context tensor to f32 slice: {e:?}"
            ))
        })?;
        self.forecast(slice, b, v, t, prediction_length, options)
    }

    fn _as_context(
        &self,
        context: &[f32],
        observation_mask: Option<&[bool]>,
        batch_size: usize,
        num_variates: usize,
        context_length: usize,
    ) -> Result<(Vec<f32>, Vec<bool>), ModelError> {
        let total_len = batch_size * num_variates * context_length;
        if let Some(user_mask) = observation_mask
            && user_mask.len() != total_len
        {
            return Err(ModelError::InvalidInput(format!(
                "observation_mask length {} does not match context length {total_len}",
                user_mask.len(),
            )));
        }
        let mut mask = Vec::with_capacity(total_len);
        let mut inputs = Vec::with_capacity(total_len);

        for (i, &val) in context.iter().enumerate() {
            let is_obs = val.is_finite() && observation_mask.is_none_or(|user_mask| user_mask[i]);
            mask.push(is_obs);
            inputs.push(if is_obs { val } else { 0.0 });
        }

        Ok((inputs, mask))
    }

    fn _validate_covariates(
        &self,
        cov: Option<&CovariateData>,
        batch_size: usize,
        expected_len: usize,
        name: &str,
    ) -> Result<Option<CovariateData>, ModelError> {
        match cov {
            None => Ok(None),
            Some(c) => {
                if !self.supports_covariates() {
                    return Err(ModelError::InvalidInput(format!(
                        "this checkpoint was not trained with covariates; got {name}"
                    )));
                }
                if c.num_channels == 0 {
                    return Ok(None);
                }
                let expected_total =
                    checked_size(name, [batch_size, c.num_channels, expected_len])?;
                if c.values.len() != expected_total {
                    return Err(ModelError::InvalidInput(format!(
                        "{name} buffer length {} does not match expected B*C*L = {}*{}*{} = {}",
                        c.values.len(),
                        batch_size,
                        c.num_channels,
                        expected_len,
                        expected_total
                    )));
                }
                Ok(Some(c.clone()))
            }
        }
    }

    fn _validate_categorical(
        &self,
        cat: Option<&CategoricalData>,
        batch_size: usize,
        expected_len: usize,
        name: &str,
    ) -> Result<Option<CategoricalData>, ModelError> {
        match cat {
            None => Ok(None),
            Some(c) => {
                if !self.supports_covariates() {
                    return Err(ModelError::InvalidInput(format!(
                        "this checkpoint was not trained with covariates; got {name}"
                    )));
                }
                if c.num_channels == 0 {
                    return Ok(None);
                }
                let expected_total =
                    checked_size(name, [batch_size, c.num_channels, expected_len])?;
                if c.codes.len() != expected_total {
                    return Err(ModelError::InvalidInput(format!(
                        "{name} buffer length {} does not match expected B*C*L = {}*{}*{} = {}",
                        c.codes.len(),
                        batch_size,
                        c.num_channels,
                        expected_len,
                        expected_total
                    )));
                }
                for &code in &c.codes {
                    if code.is_infinite() {
                        return Err(ModelError::InvalidInput(format!(
                            "{name} must hold finite codes or NaN; got an infinite value"
                        )));
                    }
                }
                Ok(Some(c.clone()))
            }
        }
    }

    fn _encoding_target(
        &self,
        inputs: &[f32],
        mask: &[bool],
        batch_size: usize,
        num_variates: usize,
        length: usize,
    ) -> Option<Vec<f64>> {
        if num_variates != 1 {
            return None;
        }

        let ffilled = ffill_with_mask(inputs, mask, batch_size, length);
        let mut target_context = Vec::with_capacity(batch_size * length);

        for b in 0..batch_size {
            let row_mask = &mask[b * length..(b + 1) * length];
            let row_ffilled = &ffilled[b * length..(b + 1) * length];

            let first_obs = row_mask.iter().position(|&obs| obs);
            let window_start = length.saturating_sub(self.max_context);

            for (t, &value) in row_ffilled.iter().enumerate() {
                let started = match first_obs {
                    Some(first) => t >= first,
                    None => false,
                };
                let in_window = t >= window_start;

                if started && in_window {
                    target_context.push(f64::from(value));
                } else {
                    target_context.push(f64::NAN);
                }
            }
        }

        Some(target_context)
    }

    fn _encode_categorical(
        &self,
        codes: Option<&CategoricalData>,
        target_context: Option<&[f64]>,
        batch_size: usize,
        context_length: usize,
        total_length: usize,
    ) -> Result<Option<CovariateData>, ModelError> {
        let Some(cat) = codes else { return Ok(None) };
        let c = cat.num_channels;

        let ctx = match target_context {
            None => {
                // Multi-variate case: no single target, return raw codes as f32
                let raw_f32: Vec<f32> = cat.codes.iter().map(|&x| x as f32).collect();
                return Ok(Some(CovariateData::new(raw_f32, c)));
            }
            Some(t) => t,
        };

        let mut out = Vec::with_capacity(batch_size * c * total_length);

        for b in 0..batch_size {
            let ctx_slice = &ctx[b * context_length..(b + 1) * context_length];

            for j in 0..c {
                let code_slice =
                    &cat.codes[(b * c + j) * total_length..(b * c + j + 1) * total_length];
                let past_codes = &code_slice[..context_length];
                let future_codes = if total_length > context_length {
                    Some(&code_slice[context_length..])
                } else {
                    None
                };

                let (past_enc, future_enc) = target_encode(
                    ctx_slice,
                    past_codes,
                    future_codes,
                    DEFAULT_TARGET_ENCODE_SMOOTH,
                )?;

                for t in 0..context_length {
                    if ctx_slice[t].is_nan() && past_codes[t].is_nan() {
                        out.push(f32::NAN);
                    } else {
                        out.push(past_enc[t]);
                    }
                }

                if let Some(f_enc) = future_enc {
                    out.extend_from_slice(&f_enc);
                }
            }
        }

        Ok(Some(CovariateData::new(out, c)))
    }

    #[allow(clippy::too_many_arguments)]
    fn _forecast_raw(
        &self,
        inputs: &[f32],
        observation_mask: &[bool],
        batch_size: usize,
        num_variates: usize,
        context_length: usize,
        prediction_length: usize,
        past_covariates: Option<&CovariateData>,
        known_covariates: Option<&CovariateData>,
    ) -> Result<Vec<f32>, ModelError> {
        let assembly = self._assemble_rows(
            inputs,
            observation_mask,
            batch_size,
            num_variates,
            context_length,
            prediction_length,
            past_covariates,
            known_covariates,
        )?;

        let mut quantiles = self._block_ar_decode(
            &assembly.rows,
            &assembly.mask,
            batch_size,
            assembly.v_all,
            assembly.padded_context_length,
            prediction_length,
            assembly.role_ids.as_deref(),
            assembly.known_future.as_deref(),
            assembly.known_future_mask.as_deref(),
            assembly.n_tgt,
        )?;

        if self.quantile_real_cap_k > 0.0 {
            _cap_quantiles(
                &mut quantiles,
                &assembly.rows,
                &assembly.mask,
                self.config(),
                batch_size,
                assembly.n_tgt,
                assembly.v_all,
                assembly.padded_context_length,
                prediction_length,
                self.quantile_real_cap_k,
            );
        }

        if quantiles.iter().any(|val| !val.is_finite()) {
            return Err(ModelError::InvalidOutput(
                "forecast produced non-finite values".into(),
            ));
        }

        Ok(quantiles)
    }

    #[allow(clippy::too_many_arguments)]
    fn _assemble_rows(
        &self,
        inputs: &[f32],
        observation_mask: &[bool],
        batch_size: usize,
        num_variates: usize,
        context_length: usize,
        prediction_length: usize,
        past_covariates: Option<&CovariateData>,
        known_covariates: Option<&CovariateData>,
    ) -> Result<AssemblyResult, ModelError> {
        let ps = self.patch_size();
        let b = batch_size;
        let n_tgt = num_variates;
        let ctx0 = context_length;
        let horizon_patches = prediction_length.div_ceil(ps);
        let horizon = horizon_patches * ps;

        let n_past = past_covariates.map(|c| c.num_channels).unwrap_or(0);
        let n_known = known_covariates.map(|c| c.num_channels).unwrap_or(0);
        let v_all = n_tgt
            .checked_add(n_past)
            .and_then(|count| count.checked_add(n_known))
            .ok_or_else(|| ModelError::InvalidInput("total series count overflows usize".into()))?;
        let unpadded_size = checked_size("combined context", [b, v_all, ctx0])?;
        checked_size(
            "padded forecast including covariates",
            [b, v_all, horizon, self.num_quantiles()],
        )?;

        let ffilled_targets = ffill_with_mask(inputs, observation_mask, b * n_tgt, ctx0);

        let mut unpadded_rows = Vec::with_capacity(unpadded_size);
        let mut unpadded_mask = Vec::with_capacity(unpadded_size);

        let mut known_future_vals = if n_known > 0 {
            Some(Vec::with_capacity(b * n_known * prediction_length))
        } else {
            None
        };
        let mut known_future_obs = if n_known > 0 {
            Some(Vec::with_capacity(b * n_known * prediction_length))
        } else {
            None
        };

        for i in 0..b {
            // 1. Targets
            unpadded_rows
                .extend_from_slice(&ffilled_targets[i * n_tgt * ctx0..(i + 1) * n_tgt * ctx0]);
            unpadded_mask
                .extend_from_slice(&observation_mask[i * n_tgt * ctx0..(i + 1) * n_tgt * ctx0]);

            // 2. Past covariates
            if let Some(past) = past_covariates {
                let past_slice = &past.values[i * n_past * ctx0..(i + 1) * n_past * ctx0];
                for &val in past_slice {
                    let is_obs = !val.is_nan();
                    unpadded_mask.push(is_obs);
                    unpadded_rows.push(if is_obs { val } else { 0.0 });
                }
            }

            // 3. Known covariates
            if let Some(known) = known_covariates {
                let total_k_len = ctx0 + prediction_length;
                let known_slice =
                    &known.values[i * n_known * total_k_len..(i + 1) * n_known * total_k_len];

                for ch in 0..n_known {
                    let ch_data = &known_slice[ch * total_k_len..(ch + 1) * total_k_len];
                    let ctx_part = &ch_data[..ctx0];
                    let fut_part = &ch_data[ctx0..];

                    for &val in ctx_part {
                        let is_obs = !val.is_nan();
                        unpadded_mask.push(is_obs);
                        unpadded_rows.push(if is_obs { val } else { 0.0 });
                    }

                    if let (Some(kf_vals), Some(kf_obs)) =
                        (&mut known_future_vals, &mut known_future_obs)
                    {
                        for &val in fut_part {
                            let is_obs = !val.is_nan();
                            kf_obs.push(is_obs);
                            kf_vals.push(if is_obs { val } else { 0.0 });
                        }
                    }
                }
            }
        }

        // Context truncation to max_context
        let (sliced_rows, sliced_mask, sliced_len) = if ctx0 > self.max_context {
            let keep = self.max_context;
            let mut s_rows = Vec::with_capacity(b * v_all * keep);
            let mut s_mask = Vec::with_capacity(b * v_all * keep);
            for r in 0..(b * v_all) {
                s_rows.extend_from_slice(&unpadded_rows[r * ctx0 + (ctx0 - keep)..(r + 1) * ctx0]);
                s_mask.extend_from_slice(&unpadded_mask[r * ctx0 + (ctx0 - keep)..(r + 1) * ctx0]);
            }
            (s_rows, s_mask, keep)
        } else {
            (unpadded_rows, unpadded_mask, ctx0)
        };

        // Left-pad to a multiple of the configured patch size.
        checked_size("padded context", [b, v_all, sliced_len.div_ceil(ps), ps])?;
        let (rows, mask, padded_len) =
            pad_to_patch(&sliced_rows, &sliced_mask, b * v_all, sliced_len, ps);

        if n_past == 0 && n_known == 0 {
            return Ok(AssemblyResult {
                rows,
                mask,
                role_ids: None,
                known_future: None,
                known_future_mask: None,
                v_all,
                padded_context_length: padded_len,
                n_tgt,
            });
        }

        let mut role_ids = Vec::with_capacity(b * v_all);
        for _ in 0..b {
            role_ids.extend(std::iter::repeat_n(RowRole::Target, n_tgt));
            role_ids.extend(std::iter::repeat_n(RowRole::PastCovariate, n_past));
            role_ids.extend(std::iter::repeat_n(RowRole::KnownCovariate, n_known));
        }

        let mut known_future = None;
        let mut known_future_mask = None;

        if let (Some(kf_vals), Some(kf_obs)) = (known_future_vals, known_future_obs) {
            let mut kf = vec![0.0f32; b * v_all * horizon];
            let mut kfm = vec![false; b * v_all * horizon];
            let k0 = n_tgt + n_past;

            for i in 0..b {
                for ch in 0..n_known {
                    let row_idx = i * v_all + k0 + ch;
                    let src_idx = (i * n_known + ch) * prediction_length;
                    for step in 0..prediction_length {
                        kf[row_idx * horizon + step] = kf_vals[src_idx + step];
                        kfm[row_idx * horizon + step] = kf_obs[src_idx + step];
                    }
                }
            }

            known_future = Some(kf);
            known_future_mask = Some(kfm);
        }

        Ok(AssemblyResult {
            rows,
            mask,
            role_ids: Some(role_ids),
            known_future,
            known_future_mask,
            v_all,
            padded_context_length: padded_len,
            n_tgt,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn _block_ar_decode(
        &self,
        inputs: &[f32],
        observation_mask: &[bool],
        batch_size: usize,
        v_all: usize,
        ctx_len: usize,
        prediction_length: usize,
        role_ids: Option<&[RowRole]>,
        known_future: Option<&[f32]>,
        known_future_mask: Option<&[bool]>,
        n_tgt: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let ps = self.patch_size();
        let q = self.num_quantiles();
        let r = batch_size * v_all;
        let horizon_patches = prediction_length.div_ceil(ps);
        let horizon = horizon_patches * ps;

        let mut context = inputs.to_vec();
        let mut mask = observation_mask.to_vec();
        let mut cur_ctx_len = ctx_len;

        let mut group_ids = Vec::with_capacity(r);
        for b in 0..batch_size {
            for _ in 0..v_all {
                group_ids.push(b);
            }
        }

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut remaining = horizon_patches;
        let mut offset = 0;

        while remaining > 0 {
            let block = 1;
            let blen = block * ps;

            let kf_block = known_future.map(|kf| {
                let mut slice = Vec::with_capacity(r * blen);
                for row in 0..r {
                    slice.extend_from_slice(
                        &kf[row * horizon + offset..row * horizon + offset + blen],
                    );
                }
                slice
            });

            let kfm_block = known_future_mask.map(|kfm| {
                let mut slice = Vec::with_capacity(r * blen);
                for row in 0..r {
                    slice.extend_from_slice(
                        &kfm[row * horizon + offset..row * horizon + offset + blen],
                    );
                }
                slice
            });

            let block_quantiles = self.model.predict_block(
                &context,
                &mask,
                &group_ids,
                block,
                role_ids,
                kf_block.as_deref(),
                kfm_block.as_deref(),
                r,
                cur_ctx_len,
            )?;

            remaining -= block;
            offset += blen;

            if remaining == 0 {
                blocks.push(block_quantiles);
                break;
            }

            // Assemble autoregressive feedback for next block
            let mut feedback = vec![0.0f32; r * blen];
            let mut feedback_mask = vec![true; r * blen];

            for row in 0..r {
                let role = role_ids.map(|roles| roles[row]).unwrap_or(RowRole::Target);

                match role {
                    RowRole::KnownCovariate => {
                        if let (Some(kfb), Some(kfmb)) = (&kf_block, &kfm_block) {
                            for step in 0..blen {
                                feedback[row * blen + step] = kfb[row * blen + step];
                                feedback_mask[row * blen + step] = kfmb[row * blen + step];
                            }
                        }
                    }
                    RowRole::PastCovariate => {
                        for step in 0..blen {
                            feedback[row * blen + step] = 0.0;
                            feedback_mask[row * blen + step] = false;
                        }
                    }
                    RowRole::Target => {
                        for step in 0..blen {
                            let val = block_quantiles[(row * blen + step) * q + self.median_index];
                            let clean_val =
                                if !val.is_finite() || val.abs() >= self.extreme_value_threshold {
                                    0.0
                                } else {
                                    val
                                };
                            feedback[row * blen + step] = clean_val;
                            feedback_mask[row * blen + step] = true;
                        }
                    }
                }
            }

            // Append feedback to context and mask per row
            let new_ctx_len = cur_ctx_len + blen;
            let size = checked_size("autoregressive context", [r, new_ctx_len])?;
            let mut new_context = Vec::with_capacity(size);
            let mut new_mask = Vec::with_capacity(size);

            for row in 0..r {
                new_context.extend_from_slice(&context[row * cur_ctx_len..(row + 1) * cur_ctx_len]);
                new_context.extend_from_slice(&feedback[row * blen..(row + 1) * blen]);

                new_mask.extend_from_slice(&mask[row * cur_ctx_len..(row + 1) * cur_ctx_len]);
                new_mask.extend_from_slice(&feedback_mask[row * blen..(row + 1) * blen]);
            }

            cur_ctx_len = new_ctx_len;

            // Context window cap: drop oldest steps past max_context
            if cur_ctx_len > self.max_context {
                let keep = ps.max((self.max_context / ps) * ps);
                let drop = cur_ctx_len - keep;
                let mut truncated_context = Vec::with_capacity(r * keep);
                let mut truncated_mask = Vec::with_capacity(r * keep);

                for row in 0..r {
                    truncated_context.extend_from_slice(
                        &new_context[row * cur_ctx_len + drop..(row + 1) * cur_ctx_len],
                    );
                    truncated_mask.extend_from_slice(
                        &new_mask[row * cur_ctx_len + drop..(row + 1) * cur_ctx_len],
                    );
                }

                context = truncated_context;
                mask = truncated_mask;
                cur_ctx_len = keep;
            } else {
                context = new_context;
                mask = new_mask;
            }
            blocks.push(block_quantiles);
        }

        // Concatenate blocks along time dimension and unpack target rows: (B, n_tgt, H, Q)
        let total_decoded = blocks.iter().map(|b| b.len() / (r * q)).sum::<usize>();
        assert!(total_decoded >= prediction_length);

        let mut target_quantiles = Vec::with_capacity(batch_size * n_tgt * prediction_length * q);

        for b in 0..batch_size {
            for v in 0..n_tgt {
                let row_idx = b * v_all + v;

                let mut step = 0;
                for blk in &blocks {
                    let blk_steps = blk.len() / (r * q);
                    for s in 0..blk_steps {
                        if step < prediction_length {
                            let idx = (row_idx * blk_steps + s) * q;
                            target_quantiles.extend_from_slice(&blk[idx..idx + q]);
                            step += 1;
                        }
                    }
                }
            }
        }

        Ok(target_quantiles)
    }
}

impl Forecaster for TimestralForecaster {
    type Error = ModelError;

    fn forecast_into(
        &self,
        request: &ForecastRequest<'_>,
        output: &mut [f32],
    ) -> Result<(), Self::Error> {
        request.validate_point_output(output)?;
        let num_quantiles = self.num_quantiles();
        let quantiles = self.forecast_quantiles_with_options(request, None)?;

        for (point, quantile_set) in output.iter_mut().zip(quantiles.chunks_exact(num_quantiles)) {
            *point = quantile_set[self.median_index];
        }
        Ok(())
    }
}

impl QuantileForecaster for TimestralForecaster {
    fn quantile_levels(&self) -> &[f32] {
        TimestralForecaster::quantile_levels(self)
    }

    fn forecast_quantiles_into(
        &self,
        request: &ForecastRequest<'_>,
        output: &mut [f32],
    ) -> Result<(), Self::Error> {
        request.validate_quantile_output(self.num_quantiles(), output)?;
        let quantiles = self.forecast_quantiles_with_options(request, None)?;
        output.copy_from_slice(&quantiles);
        Ok(())
    }
}

struct AssemblyResult {
    rows: Vec<f32>,
    mask: Vec<bool>,
    role_ids: Option<Vec<RowRole>>,
    known_future: Option<Vec<f32>>,
    known_future_mask: Option<Vec<bool>>,
    v_all: usize,
    padded_context_length: usize,
    n_tgt: usize,
}

fn checked_size<const N: usize>(name: &str, dimensions: [usize; N]) -> Result<usize, ModelError> {
    checked_buffer_len::<f32>(name, &dimensions)
}

fn _cat_covariates(
    a: Option<CovariateData>,
    b: Option<CovariateData>,
    batch_size: usize,
    length: usize,
) -> Result<Option<CovariateData>, ModelError> {
    match (a, b) {
        (None, None) => Ok(None),
        (Some(cov), None) | (None, Some(cov)) => Ok(Some(cov)),
        (Some(cov_a), Some(cov_b)) => {
            let ca = cov_a.num_channels;
            let cb = cov_b.num_channels;
            let total_c = ca.checked_add(cb).ok_or_else(|| {
                ModelError::InvalidInput("total covariate count overflows usize".into())
            })?;
            let mut out = Vec::with_capacity(checked_size(
                "combined covariates",
                [batch_size, total_c, length],
            )?);

            for i in 0..batch_size {
                out.extend_from_slice(&cov_a.values[i * ca * length..(i + 1) * ca * length]);
                out.extend_from_slice(&cov_b.values[i * cb * length..(i + 1) * cb * length]);
            }

            Ok(Some(CovariateData::new(out, total_c)))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn _cap_quantiles(
    quantiles: &mut [f32],
    inputs: &[f32],
    mask: &[bool],
    config: &TimestralConfig,
    batch_size: usize,
    num_targets: usize,
    v_all: usize,
    context_length: usize,
    prediction_length: usize,
    cap_k: f32,
) {
    let ps = config.patch_size;
    let min_scale = config.minimum_scale as f64;
    let stabilize = config.stabilize_with_global;
    let q = config.num_quantiles();

    for b in 0..batch_size {
        for v in 0..num_targets {
            let row_idx = b * v_all + v;
            let row_in = &inputs[row_idx * context_length..(row_idx + 1) * context_length];
            let row_mask = &mask[row_idx * context_length..(row_idx + 1) * context_length];

            let mut ctx_max = f32::NEG_INFINITY;
            let mut ctx_min = f32::INFINITY;
            let mut has_observed = false;

            for t in 0..context_length {
                if row_mask[t] && row_in[t].is_finite() {
                    has_observed = true;
                    if row_in[t] > ctx_max {
                        ctx_max = row_in[t];
                    }
                    if row_in[t] < ctx_min {
                        ctx_min = row_in[t];
                    }
                }
            }

            let final_max = if has_observed { ctx_max } else { 0.0 };
            let final_min = if has_observed { ctx_min } else { 0.0 };

            let row_in_f64: Vec<f64> = row_in.iter().map(|&x| x as f64).collect();
            let (_, scales) =
                causal_patch_loc_scale(&row_in_f64, row_mask, ps, min_scale, stabilize);
            let anchor_scale = (*scales.last().unwrap_or(&min_scale) as f32) * cap_k;

            let lower_bound = final_min - anchor_scale;
            let upper_bound = final_max + anchor_scale;

            for h in 0..prediction_length {
                let base_idx = ((b * num_targets + v) * prediction_length + h) * q;
                for k in 0..q {
                    let val = quantiles[base_idx + k];
                    quantiles[base_idx + k] = val.clamp(lower_bound, upper_bound);
                }
            }
        }
    }
}
