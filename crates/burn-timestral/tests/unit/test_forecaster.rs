use super::*;
use crate::test_support::{pretrained_209m, tiny_model};
use forecast_core::{ForecastRequest, Forecaster, QuantileForecaster};

#[test]
fn rejects_overflowing_forecast_dimensions() {
    let device = burn::tensor::Device::flex();
    let forecaster = TimestralForecaster::from_model(tiny_model(&device)).unwrap();
    assert!(matches!(
        forecaster.forecast(&[], usize::MAX, 2, 1, 1, None),
        Err(ModelError::InvalidInput(_))
    ));
    assert!(matches!(
        forecaster.forecast_1d(&[1.0], usize::MAX, None),
        Err(ModelError::InvalidInput(_))
    ));
    assert!(matches!(
        Forecast::new(Vec::new(), vec![0.5], usize::MAX, 2, 1),
        Err(ModelError::InvalidInput(_))
    ));
}

#[test]
fn shared_forecasting_traits_write_into_preallocated_buffers() {
    let device = burn::tensor::Device::flex();
    let forecaster = TimestralForecaster::from_model(tiny_model(&device)).unwrap();
    let context = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let request = ForecastRequest::new(&context, 2, 1, 12, 3).unwrap();
    let expected = forecaster.forecast(&context, 2, 1, 12, 3, None).unwrap();

    let quantile_len = request
        .quantile_output_len(forecaster.num_quantiles())
        .unwrap();
    let mut quantiles = vec![f32::NAN; quantile_len];
    forecaster
        .forecast_quantiles_into(&request, &mut quantiles)
        .unwrap();
    assert_eq!(quantiles, expected.predictions());

    let mut points = vec![f32::NAN; request.point_output_len()];
    forecaster.forecast_into(&request, &mut points).unwrap();
    assert_eq!(points, expected.median());

    let mut invalid_points = vec![42.0; request.point_output_len() - 1];
    assert!(
        forecaster
            .forecast_into(&request, &mut invalid_points)
            .is_err()
    );
    assert_eq!(invalid_points, vec![42.0; request.point_output_len() - 1]);

    let mut invalid_quantiles = vec![42.0; quantile_len - 1];
    assert!(
        forecaster
            .forecast_quantiles_into(&request, &mut invalid_quantiles)
            .is_err()
    );
    assert_eq!(invalid_quantiles, vec![42.0; quantile_len - 1]);
}

#[test]
fn shared_forecasting_traits_handle_multiple_variates_and_patches() {
    let device = burn::tensor::Device::flex();
    let forecaster = TimestralForecaster::from_model(tiny_model(&device)).unwrap();
    let context = (0..48).map(|value| value as f32).collect::<Vec<_>>();
    let request = ForecastRequest::new(&context, 2, 2, 12, 33).unwrap();
    let expected = forecaster.forecast(&context, 2, 2, 12, 33, None).unwrap();

    let quantile_len = request
        .quantile_output_len(forecaster.num_quantiles())
        .unwrap();
    let mut quantiles = vec![f32::NAN; quantile_len];
    forecaster
        .forecast_quantiles_into(&request, &mut quantiles)
        .unwrap();
    assert_eq!(quantiles, expected.predictions());

    let mut points = vec![f32::NAN; request.point_output_len()];
    forecaster.forecast_into(&request, &mut points).unwrap();
    assert_eq!(points, expected.median());
}

#[test]
fn rejects_zero_max_context() {
    let device = burn::tensor::Device::flex();
    assert!(matches!(
        TimestralForecaster::new(tiny_model(&device), Some(0), 1, 1e4, 1e30),
        Err(ModelError::InvalidInput(_))
    ));
}

#[test]
fn test_ffill_with_mask_basic_and_leading_nan() {
    // Row 0: [NaN, NaN, 10.0, NaN, 20.0, NaN]
    // Mask:  [false, false, true, false, true, false]
    let inputs = vec![f32::NAN, f32::NAN, 10.0, f32::NAN, 20.0, f32::NAN];
    let mask = vec![false, false, true, false, true, false];

    let filled = ffill_with_mask(&inputs, &mask, 1, 6);
    assert_eq!(filled, vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0]);
}

#[test]
fn test_pad_to_patch_unaligned_and_aligned() {
    // Unaligned: length 5 with patch_size 4 -> padded to 8 (3 leading zeros)
    let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = vec![true, true, true, true, true];
    let (pad_in, pad_m, new_len) = pad_to_patch(&inputs, &mask, 1, 5, 4);
    assert_eq!(new_len, 8);
    assert_eq!(pad_in, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(
        pad_m,
        vec![false, false, false, true, true, true, true, true]
    );

    // Aligned: length 8 with patch_size 4 -> unchanged
    let (same_in, same_m, same_len) = pad_to_patch(&pad_in, &pad_m, 1, 8, 4);
    assert_eq!(same_len, 8);
    assert_eq!(same_in, pad_in);
    assert_eq!(same_m, pad_m);
}

#[test]
fn test_replace_extreme_values() {
    let mut data = vec![1.0, f32::NAN, 1e35, -1e32, f32::INFINITY, -5.0];
    replace_extreme_values(&mut data, 1e30, 0.0);
    assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 0.0, -5.0]);
}

#[test]
fn test_nonnegative_context_criteria() {
    // Four series of length 4:
    // Series 0: [0.0, 1.0, 2.0, 3.0] -> true
    // Series 1: [NaN, 1.0, 2.0, NaN] -> true
    // Series 2: [1.0, -0.5, 2.0, 3.0] -> false
    // Series 3: [NaN, NaN, NaN, NaN] -> false (no observations)
    let inputs = vec![
        0.0,
        1.0,
        2.0,
        3.0,
        f32::NAN,
        1.0,
        2.0,
        f32::NAN,
        1.0,
        -0.5,
        2.0,
        3.0,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
    ];
    let mask = vec![
        true, true, true, true, false, true, true, false, true, true, true, true, false, false,
        false, false,
    ];

    let eligible = nonnegative_context(&inputs, &mask, 4, 1, 4);
    assert_eq!(eligible, vec![true, true, false, false]);
}

#[test]
fn test_forecast_options_builder() {
    let opt = ForecastOptions::default()
        .with_symmetric_averaging(false)
        .with_clip_nonnegative(false)
        .with_observation_mask(vec![true, false, true])
        .with_past_covariates(vec![1.0, 2.0, 3.0], 1)
        .with_known_covariates(vec![4.0, 5.0, 6.0, 7.0], 1)
        .with_categorical_past_covariates(vec![0.0, 1.0, 0.0], 1)
        .with_categorical_known_covariates(vec![1.0, 2.0, 1.0, 2.0], 1);

    assert!(!opt.symmetric_averaging);
    assert!(!opt.clip_nonnegative);
    assert_eq!(opt.observation_mask.unwrap(), vec![true, false, true]);
    assert_eq!(opt.past_covariates.unwrap().values, vec![1.0, 2.0, 3.0]);
    assert_eq!(
        opt.known_covariates.unwrap().values,
        vec![4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(
        opt.categorical_past_covariates.unwrap().codes,
        vec![0.0, 1.0, 0.0]
    );
    assert_eq!(
        opt.categorical_known_covariates.unwrap().codes,
        vec![1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn test_forecast_struct_accessors() {
    // B=1, V=1, H=2, Q=3 (levels: 0.1, 0.5, 0.9)
    let levels = vec![0.1, 0.5, 0.9];
    let quantiles = vec![
        1.0, 2.0, 3.0, // h=0
        4.0, 5.0, 6.0, // h=1
    ];
    let forecast = Forecast::new(quantiles, levels, 1, 1, 2).unwrap();

    assert_eq!(forecast.shape(), (1, 1, 2, 3));
    assert_eq!(forecast.median(), vec![2.0, 5.0]);
    assert_eq!(forecast.quantile(0.1).unwrap(), vec![1.0, 4.0]);
    assert_eq!(forecast.quantile(0.9).unwrap(), vec![3.0, 6.0]);
    assert!(forecast.quantile(0.7).is_err());

    assert_eq!(forecast.get(0, 0, 0, 0), Some(1.0));
    assert_eq!(forecast.get(0, 0, 0, 1), Some(2.0));
    assert_eq!(forecast.get(0, 0, 1, 2), Some(6.0));

    let device = burn::tensor::Device::flex();
    let tensor = forecast.into_tensor(&device);
    assert_eq!(tensor.shape().dims(), [1, 1, 2, 3]);
}

#[test]
fn test_forecaster_input_validation() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);

    // Constructor validation for per_shot_horizon_patches
    // per_shot == 0 fails
    let err_zero = TimestralForecaster::new(model.clone(), None, 0, 1e4, 1e30);
    assert!(err_zero.is_err());

    // per_shot > 1 fails due to the single-patch inference contract.
    let err_multi = TimestralForecaster::new(model.clone(), None, 2, 1e4, 1e30);
    assert!(err_multi.is_err());
    let err_msg = format!("{}", err_multi.err().unwrap());
    assert!(err_msg.contains("per_shot_horizon_patches must be 1"));

    // per_shot == 1 succeeds
    let ok_forecaster = TimestralForecaster::new(model.clone(), None, 1, 1e4, 1e30);
    assert!(ok_forecaster.is_ok());

    let forecaster = ok_forecaster.unwrap();
    let ctx = vec![1.0; 64];

    // Prediction length 0 fails
    assert!(forecaster.forecast(&ctx, 1, 1, 64, 0, None).is_err());

    // Context length mismatch fails
    assert!(forecaster.forecast(&ctx, 1, 1, 32, 10, None).is_err());

    // Batch size 0 fails
    assert!(forecaster.forecast(&ctx, 0, 1, 64, 10, None).is_err());

    // Past covariates length mismatch fails
    let opt_bad_past = ForecastOptions::default().with_past_covariates(vec![1.0; 30], 1);
    assert!(
        forecaster
            .forecast(&ctx, 1, 1, 64, 10, Some(&opt_bad_past))
            .is_err()
    );

    // Known covariates length mismatch fails (needs B * C * (T + H) = 1 * 1 * (64 + 10) = 74)
    let opt_bad_known = ForecastOptions::default().with_known_covariates(vec![1.0; 64], 1);
    assert!(
        forecaster
            .forecast(&ctx, 1, 1, 64, 10, Some(&opt_bad_known))
            .is_err()
    );

    // Infinite categorical codes are rejected.
    let opt_inf_cat =
        ForecastOptions::default().with_categorical_past_covariates(vec![f64::INFINITY; 64], 1);
    assert!(
        forecaster
            .forecast(&ctx, 1, 1, 64, 10, Some(&opt_inf_cat))
            .is_err()
    );
}

#[test]
fn test_forecast_shapes_and_finite() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);
    let forecaster = TimestralForecaster::new(model, Some(256), 1, 3.0, 1e30).unwrap();

    let cases = [
        (1usize, 100usize, 12usize), // one block
        (3, 300, 50), // context past the cap (300 > max_context 256) / multi-variate multi-block
        (2, 37, 70),  // three blocks after a short unaligned context
    ];

    let batch = 2usize;

    for (variates, length, horizon) in cases {
        let mut x = Vec::with_capacity(batch * variates * length);
        for b in 0..batch {
            for v in 0..variates {
                for t in 0..length {
                    let base = 10.0 * (2.0 * std::f32::consts::PI * (t as f32) / 24.0).sin()
                        + (b as f32) * 1.5
                        + (v as f32) * 2.0;
                    x.push(base);
                }
            }
        }

        // Insert NaNs into batch 0, variate 0 at steps 5..9
        x[5..9].fill(f32::NAN);

        let out = forecaster
            .forecast(&x, batch, variates, length, horizon, None)
            .unwrap();

        assert_eq!(out.shape(), (batch, variates, horizon, 21));
        for &q in &out.quantiles {
            assert!(q.is_finite());
        }

        assert_monotone_quantiles(&out);
    }
}

fn assert_monotone_quantiles(forecast: &Forecast) {
    let (_, variates, horizon, num_quantiles) = forecast.shape();
    for (row, quantiles) in forecast
        .predictions()
        .chunks_exact(num_quantiles)
        .enumerate()
    {
        let h = row % horizon;
        let series = row / horizon;
        let v = series % variates;
        let b = series / variates;
        for (k, pair) in quantiles.windows(2).enumerate() {
            let (q_curr, q_next) = (pair[0], pair[1]);
            assert!(
                q_next >= q_curr - 1e-5,
                "quantiles must be sorted: {q_curr} > {q_next} at b={b}, v={v}, h={h}, k={k}"
            );
        }
    }
}

#[test]
fn test_forecast_univariate_short_and_multi_block_horizons() {
    let device = burn::tensor::Device::flex();
    let forecaster = TimestralForecaster::from_model(tiny_model(&device)).unwrap();

    let t = 96;
    let mut ctx = Vec::with_capacity(t);
    for i in 0..t {
        ctx.push(((i as f32) * 0.1).sin() * 10.0);
    }
    // Insert a NaN gap
    ctx[5] = f32::NAN;
    ctx[6] = f32::NAN;

    // Horizon 12 (less than 1 block = 32 steps)
    let out_short = forecaster.forecast_1d(&ctx, 12, None).unwrap();
    assert_eq!(out_short.shape(), (1, 1, 12, 21));
    for &q in &out_short.quantiles {
        assert!(q.is_finite());
    }
    assert_monotone_quantiles(&out_short);

    // Horizon 70 (3 blocks: 32 + 32 + 6)
    let out_long = forecaster.forecast_1d(&ctx, 70, None).unwrap();
    assert_eq!(out_long.shape(), (1, 1, 70, 21));
    for &q in &out_long.quantiles {
        assert!(q.is_finite());
    }
    assert_monotone_quantiles(&out_long);
}

#[test]
fn test_forecast_multivariate_joint_forecast() {
    let device = burn::tensor::Device::flex();
    let forecaster = TimestralForecaster::from_model(tiny_model(&device)).unwrap();

    let b = 1;
    let v = 2;
    let t = 64;
    let h = 32;

    let mut ctx = Vec::with_capacity(b * v * t);
    for var in 0..v {
        for step in 0..t {
            ctx.push(((step as f32 + var as f32 * 10.0) * 0.1).cos() * 5.0);
        }
    }

    let out = forecaster.forecast(&ctx, b, v, t, h, None).unwrap();
    assert_eq!(out.shape(), (b, v, h, 21));

    let median = out.median();
    assert_eq!(median.len(), b * v * h);
    for &m in &median {
        assert!(m.is_finite());
    }
}

#[test]
fn test_symmetric_averaging_is_odd_with_covariates() {
    let device = burn::tensor::Device::flex();
    let forecaster = TimestralForecaster::from_model(tiny_model(&device)).unwrap();

    let b = 2;
    let t = 96;
    let h = 24;

    let mut x = Vec::with_capacity(b * t);
    let mut past = Vec::with_capacity(b * 2 * t);
    let mut known = Vec::with_capacity(b * (t + h));

    for i in 0..b * t {
        x.push(((i as f32) * 0.15).sin() * 8.0);
    }
    for i in 0..b * 2 * t {
        past.push(((i as f32) * 0.08).cos() * 4.0);
    }
    for i in 0..b * (t + h) {
        known.push(((i as f32) * 0.12).sin() * 3.0);
    }

    let opt_pos = ForecastOptions::default()
        .with_past_covariates(past.clone(), 2)
        .with_known_covariates(known.clone(), 1)
        .with_clip_nonnegative(false)
        .with_symmetric_averaging(true);

    let neg_x: Vec<f32> = x.iter().map(|&v| -v).collect();
    let neg_past: Vec<f32> = past.iter().map(|&v| -v).collect();
    let neg_known: Vec<f32> = known.iter().map(|&v| -v).collect();

    let opt_neg = ForecastOptions::default()
        .with_past_covariates(neg_past, 2)
        .with_known_covariates(neg_known, 1)
        .with_clip_nonnegative(false)
        .with_symmetric_averaging(true);

    let pos_out = forecaster.forecast(&x, b, 1, t, h, Some(&opt_pos)).unwrap();
    let neg_out = forecaster
        .forecast(&neg_x, b, 1, t, h, Some(&opt_neg))
        .unwrap();

    let n_knots = 21;
    let mut max_diff = 0.0f32;
    for i in 0..b * h {
        for q in 0..n_knots {
            let pos_val = pos_out.quantiles[i * n_knots + q];
            let neg_val = neg_out.quantiles[i * n_knots + (n_knots - 1 - q)];
            let diff = (neg_val - (-pos_val)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    assert!(
        max_diff < 1e-4,
        "symmetric averaging odd symmetry violation: max diff = {}",
        max_diff
    );
}

#[test]
#[ignore = "requires the official Timestral-209M checkpoint"]
fn test_categorical_covariates_are_target_encoded() {
    let forecaster = TimestralForecaster::from_model(pretrained_209m().clone()).unwrap();

    let b = 2;
    let t = 96;
    let h = 24;

    // Positive target series
    let mut x = Vec::with_capacity(b * t);
    for i in 0..b * t {
        x.push(((i as f32) * 0.1).sin().abs() + 2.0);
    }

    // Categorical known codes: integers 0..3 with NaNs
    let mut cat_known = Vec::with_capacity(b * (t + h));
    for i in 0..b * (t + h) {
        let code = (i % 4) as f64;
        if (10..20).contains(&i) {
            cat_known.push(f64::NAN);
        } else {
            cat_known.push(code);
        }
    }

    let opt_cat = ForecastOptions::default()
        .with_categorical_known_covariates(cat_known.clone(), 1)
        .with_symmetric_averaging(false)
        .with_clip_nonnegative(false);

    let cat_out = forecaster.forecast(&x, b, 1, t, h, Some(&opt_cat)).unwrap();

    // 1. Perform manual target encoding with Python default smooth=1.0
    let mut manual_encoded = Vec::with_capacity(b * (t + h));
    for i in 0..b {
        let ctx_slice: Vec<f64> = x[i * t..(i + 1) * t].iter().map(|&v| v as f64).collect();
        let code_slice = &cat_known[i * (t + h)..(i + 1) * (t + h)];
        let past_codes = &code_slice[..t];
        let future_codes = &code_slice[t..];
        let (p_enc, f_enc) = target_encode(
            &ctx_slice,
            past_codes,
            Some(future_codes),
            DEFAULT_TARGET_ENCODE_SMOOTH,
        )
        .unwrap();

        // Verify encoded values against Python golden master (smooth=1.0)
        if i == 0 {
            #[expect(
                clippy::excessive_precision,
                reason = "Preserve the decimal literals from the Python reference"
            )]
            let golden_p_enc = [
                2.5946043, 2.5976629, 2.5838199, 2.5812271, 2.5946043, 2.5976629,
            ];
            for (step, &expected) in golden_p_enc.iter().enumerate() {
                assert!(
                    (p_enc[step] - expected).abs() < 1e-4,
                    "target_encode past mismatch at step {step}: actual={}, expected={expected}",
                    p_enc[step]
                );
            }
            let f_slice = f_enc.as_ref().unwrap();
            #[expect(
                clippy::excessive_precision,
                reason = "Preserve the decimal literals from the Python reference"
            )]
            let golden_f_enc = [
                2.5946043, 2.5976629, 2.5838199, 2.5812271, 2.5946043, 2.5976629,
            ];
            for (step, &expected) in golden_f_enc.iter().enumerate() {
                assert!(
                    (f_slice[step] - expected).abs() < 1e-4,
                    "target_encode future mismatch at step {step}: actual={}, expected={expected}",
                    f_slice[step]
                );
            }
        }

        manual_encoded.extend_from_slice(&p_enc);
        manual_encoded.extend_from_slice(&f_enc.unwrap());
    }

    let opt_direct = ForecastOptions::default()
        .with_known_covariates(manual_encoded, 1)
        .with_symmetric_averaging(false)
        .with_clip_nonnegative(false);

    let direct_out = forecaster
        .forecast(&x, b, 1, t, h, Some(&opt_direct))
        .unwrap();

    // 2. Parity check between internal categorical target-encoding and manual target_encode
    let mut max_diff = 0.0f32;
    for (a, b) in cat_out.quantiles.iter().zip(direct_out.quantiles.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(
        max_diff < 1e-5,
        "categorical target encoding parity diff too large: {}",
        max_diff
    );

    // 3. Absolute numeric check against Python reference (per_shot_horizon_patches=1, smooth=1.0)
    let b0_expected_median = [2.1590164, 2.2523024, 2.3460462, 2.4375281, 2.5270722];
    for (step, &expected) in b0_expected_median.iter().enumerate() {
        let actual = cat_out.get(0, 0, step, 10).unwrap();
        assert!(
            (actual - expected).abs() < 1e-4,
            "Series 0 median mismatch at step {step}: actual={actual}, expected={expected}"
        );
    }

    #[expect(
        clippy::excessive_precision,
        reason = "Preserve the decimal literals from the Python reference"
    )]
    let b1_expected_median = [2.3424904, 2.4326506, 2.5261805, 2.6054728, 2.6831081];
    for (step, &expected) in b1_expected_median.iter().enumerate() {
        let actual = cat_out.get(1, 0, step, 10).unwrap();
        assert!(
            (actual - expected).abs() < 1e-4,
            "Series 1 median mismatch at step {step}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
#[ignore = "requires the official Timestral-209M checkpoint"]
fn test_clip_nonnegative_behavior() {
    let forecaster = TimestralForecaster::from_model(pretrained_209m().clone()).unwrap();

    let b = 2;
    let t = 64;
    let h = 32;

    // Series 0: small positive values where unclipped lower quantiles drop below 0.0
    // Series 1: negative context (-10.0)
    let mut ctx = Vec::with_capacity(b * t);
    for step in 0..t {
        ctx.push(((step as f32) * 5.0 / 63.0).sin() * 0.1 + 0.1);
    }
    for step in 0..t {
        ctx.push(((step as f32) * 0.1).sin() - 10.0);
    }

    let opt_clipped = ForecastOptions::default()
        .with_clip_nonnegative(true)
        .with_symmetric_averaging(false);

    let opt_unclipped = ForecastOptions::default()
        .with_clip_nonnegative(false)
        .with_symmetric_averaging(false);

    let out_clipped = forecaster
        .forecast(&ctx, b, 1, t, h, Some(&opt_clipped))
        .unwrap();
    let out_unclipped = forecaster
        .forecast(&ctx, b, 1, t, h, Some(&opt_unclipped))
        .unwrap();

    // Verify Series 0 without clipping has negative lower quantiles (< 0.0)
    let mut series0_unclipped_min = f32::INFINITY;
    for step in 0..h {
        for q in 0..21 {
            let val = out_unclipped.get(0, 0, step, q).unwrap();
            if val < series0_unclipped_min {
                series0_unclipped_min = val;
            }
        }
    }
    assert!(
        series0_unclipped_min < -0.01,
        "Series 0 unclipped should have lower quantiles below 0, got {series0_unclipped_min}"
    );

    // Verify Series 0 with clipping is strictly >= 0.0
    let mut series0_clipped_min = f32::INFINITY;
    for step in 0..h {
        for q in 0..21 {
            let val = out_clipped.get(0, 0, step, q).unwrap();
            if val < series0_clipped_min {
                series0_clipped_min = val;
            }
            assert!(
                val >= 0.0,
                "eligible non-negative series produced negative quantile: {val} at h={step}, q={q}"
            );
        }
    }
    assert_eq!(
        series0_clipped_min, 0.0,
        "Series 0 clipped min must be exactly 0.0"
    );

    // Series 1 has negative context and retains negative predictions with clipping enabled.
    let mut series1_clipped_has_neg = false;
    for step in 0..h {
        for q in 0..21 {
            let val = out_clipped.get(1, 0, step, q).unwrap();
            if val < 0.0 {
                series1_clipped_has_neg = true;
                break;
            }
        }
    }
    assert!(
        series1_clipped_has_neg,
        "Series 1 with negative context was unexpectedly clamped"
    );
}

#[test]
#[ignore = "requires the official Timestral-209M checkpoint"]
fn test_multi_block_golden_numeric_parity() {
    let forecaster = TimestralForecaster::from_model(pretrained_209m().clone()).unwrap();

    let t = 96;
    let h = 70;

    let mut x = Vec::with_capacity(t);
    for i in 0..t {
        let val = ((i as f32) * 10.0 / 95.0).sin();
        x.push(val);
    }

    let mut known = Vec::with_capacity(t + h);
    for i in 0..(t + h) {
        let val = ((i as f32) * 15.0 / 165.0).cos();
        known.push(val);
    }

    let opt = ForecastOptions::default()
        .with_known_covariates(known, 1)
        .with_symmetric_averaging(true)
        .with_clip_nonnegative(false);

    let out = forecaster.forecast(&x, 1, 1, t, h, Some(&opt)).unwrap();
    assert_eq!(out.shape(), (1, 1, 70, 21));

    let median = out.median();

    // Reference values computed from official Python TimestralForecaster (per_shot_horizon_patches=1)
    // Block 1 start: [-0.6347688, -0.7160572, -0.7866592, -0.8508362, -0.9064128]
    let ref_block1_start = [
        -0.63476884,
        -0.7160572,
        -0.78665924,
        -0.85083616,
        -0.9064128,
    ];
    for (i, &expected) in ref_block1_start.iter().enumerate() {
        let actual = median[i];
        assert!(
            (actual - expected).abs() < 1e-4,
            "Block 1 step {i} mismatch: actual={actual}, expected={expected}"
        );
    }

    // Block 2 boundary (steps 30..35):
    // [0.6073347, 0.6849894, 0.7522281, 0.8130311, 0.8635278]
    let ref_block2_boundary = [
        (30, 0.60733473),
        (31, 0.6849894),
        (32, 0.7522281),
        (33, 0.8130311),
        (34, 0.8635278),
    ];
    for &(step, expected) in &ref_block2_boundary {
        let actual = median[step];
        assert!(
            (actual - expected).abs() < 1e-4,
            "Block boundary step {step} mismatch: actual={actual}, expected={expected}"
        );
    }

    // Block 3 boundary (steps 62..66):
    // [-0.7722707, -0.8234061, -0.8725418, -0.9113342]
    let ref_block3_boundary = [
        (62, -0.77227074),
        (63, -0.8234061),
        (64, -0.87254184),
        (65, -0.91133416),
    ];
    for &(step, expected) in &ref_block3_boundary {
        let actual = median[step];
        assert!(
            (actual - expected).abs() < 1e-4,
            "Block boundary step {step} mismatch: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn test_max_context_truncation_and_sliding_window() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);
    let forecaster = TimestralForecaster::new(model, Some(128), 1, 3.0, 1e30).unwrap();

    assert_eq!(forecaster.max_context(), 128);

    // Context T=300 > 128 (will be truncated to 128)
    // Horizon H=70 -> 3 AR steps (32 + 32 + 6)
    // During AR steps: 128 + 32 = 160 > 128 -> sliding window caps to keep=128
    let t = 300;
    let h = 70;
    let mut x = Vec::with_capacity(t);
    for i in 0..t {
        x.push(((i as f32) * 0.1).sin() * 5.0 + 1.0);
    }

    let out = forecaster.forecast_1d(&x, h, None).unwrap();
    assert_eq!(out.shape(), (1, 1, 70, 21));
    for &q in &out.quantiles {
        assert!(q.is_finite());
    }

    assert_monotone_quantiles(&out);
}
