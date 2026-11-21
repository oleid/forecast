use super::*;
use crate::test_support::{pretrained_209m, tiny_config, tiny_model};

#[test]
fn test_row_role_enum_and_conversions() {
    let t = RowRole::Target;
    let p = RowRole::PastCovariate;
    let k = RowRole::KnownCovariate;

    assert!(t.is_target());
    assert!(!t.is_past_covariate());
    assert!(!t.is_known_covariate());
    assert_eq!(t.as_u8(), 0);
    assert_eq!(t.as_usize(), ROLE_TARGET);
    assert_eq!(usize::from(t), 0);

    assert!(!p.is_target());
    assert!(p.is_past_covariate());
    assert!(!p.is_known_covariate());
    assert_eq!(p.as_u8(), 1);
    assert_eq!(p.as_usize(), ROLE_PAST_COV);
    assert_eq!(usize::from(p), 1);

    assert!(!k.is_target());
    assert!(!k.is_past_covariate());
    assert!(k.is_known_covariate());
    assert_eq!(k.as_u8(), 2);
    assert_eq!(k.as_usize(), ROLE_KNOWN_COV);
    assert_eq!(usize::from(k), 2);

    assert_eq!(RowRole::try_from(0).unwrap(), RowRole::Target);
    assert_eq!(RowRole::try_from(1).unwrap(), RowRole::PastCovariate);
    assert_eq!(RowRole::try_from(2).unwrap(), RowRole::KnownCovariate);
    assert!(RowRole::try_from(3).is_err());
    assert!(RowRole::try_from(99).is_err());
}

#[test]
fn test_model_error_display() {
    let err_in = ModelError::InvalidInput("bad shape".to_string());
    assert_eq!(format!("{err_in}"), "Invalid model input: bad shape");
}

#[test]
fn test_local_load_reports_missing_configuration_and_weights() {
    let device = burn::tensor::Device::flex();
    let dir = tempfile::tempdir().unwrap();
    let load = || TimestralModel::load(TimestralVariant::M5, Some(dir.path()), &device);
    assert!(
        matches!(load(), Err(ModelError::ConfigRead { path, .. }) if path.ends_with("config.json"))
    );
    tiny_config()
        .save_to_file(dir.path().join("config.json"))
        .unwrap();
    assert!(
        matches!(load(), Err(ModelError::CheckpointRead { path, .. }) if path.ends_with("model.safetensors"))
    );
}

#[test]
fn test_local_load_reports_corrupt_checkpoint() {
    let device = burn::tensor::Device::flex();
    let dir = tempfile::tempdir().unwrap();
    tiny_config()
        .save_to_file(dir.path().join("config.json"))
        .unwrap();
    std::fs::write(
        dir.path().join("model.safetensors"),
        b"corrupted checkpoint",
    )
    .unwrap();
    let result = TimestralModel::from_dir(dir.path(), &device);
    assert!(
        matches!(result, Err(ModelError::CheckpointRead { path, .. }) if path.ends_with("model.safetensors"))
    );
}

#[test]
fn test_local_load_rejects_missing_parameters() {
    let device = burn::tensor::Device::flex();
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("model.safetensors");
    std::fs::write(&weights, b"\x02\x00\x00\x00\x00\x00\x00\x00{}").unwrap();
    let result = TimestralModel::from_safetensors(tiny_config(), weights, &device);
    assert!(
        matches!(result, Err(ModelError::IncompatibleCheckpoint { reason, .. }) if reason.contains("missing"))
    );
}

#[test]
fn test_local_load_rejects_incompatible_tensor_shape() {
    let device = burn::tensor::Device::flex();
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("model.safetensors");
    let header = br#"{"representation.patch_proj.hidden_layer.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = [&(header.len() as u64).to_le_bytes()[..], header, &[0; 4]].concat();
    std::fs::write(&weights, file).unwrap();

    let result = TimestralModel::from_safetensors(tiny_config(), weights, &device);
    assert!(
        matches!(result, Err(ModelError::IncompatibleCheckpoint { reason, .. })
        if reason.contains("representation.patch_proj.hidden_layer.weight") && !reason.ends_with("errors=[]"))
    );
}

#[cfg(not(feature = "hub"))]
#[test]
fn test_missing_model_path_without_hub_reports_how_to_load() {
    let result = TimestralModel::load(TimestralVariant::M5, None, &burn::tensor::Device::flex());
    assert!(
        matches!(result, Err(ModelError::HubDisabled { repo }) if repo == TimestralVariant::M5.repo_id())
    );
}

#[test]
#[ignore = "requires the official Timestral-209M checkpoint"]
fn test_model_loading_and_properties() {
    let model = pretrained_209m();
    assert_eq!(model.config().patch_size, 32);
    assert_eq!(model.config().num_quantiles(), 21);
    assert_eq!(model.config().quantile_width_factor, 3.0);
    assert!(model.config().covariate_enabled);
}

#[test]
fn test_model_scale_method() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);

    let r = 2;
    let t = 64; // 2 patches
    let mut inputs = vec![10.0f32; r * t];
    // Row 0 is continuous target, Row 1 is binary known covariate
    for step in 0..t {
        inputs[t + step] = (step % 2) as f32; // 0, 1, 0, 1...
    }
    let observed = vec![true; r * t];
    let roles = vec![RowRole::Target, RowRole::KnownCovariate];

    let (scaled, loc, scale) = model.scale(&inputs, &observed, Some(&roles), r, t);

    assert_eq!(scaled.len(), r * t);
    assert_eq!(loc.len(), r * t);
    assert_eq!(scale.len(), r * t);

    // Row 0: constant 10.0 -> loc = 10.0, scale = minimum_scale (1e-6)
    assert!((loc[0] - 10.0).abs() < 1e-4);
    assert_eq!(scaled[0], 0.0);

    // Row 1: binary indicator known covariate -> loc = 0.0, scale = 1.0, binary bypass
    assert_eq!(loc[t], 0.0);
    assert_eq!(scale[t], 1.0);
    // Values should match binary input directly without asinh
    assert_eq!(scaled[t], 0.0);
    assert_eq!(scaled[t + 1], 1.0);
}

#[test]
fn test_predict_block_input_validation() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);

    let r = 1;
    let t = 64;
    let inputs = vec![1.0f32; r * t];
    let observed = vec![true; r * t];
    let group_ids = vec![0];

    // Case 1: r == 0
    let err = model.predict_block(&[], &[], &[], 1, None, None, None, 0, t);
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 2: t == 0
    let err = model.predict_block(&[], &[], &group_ids, 1, None, None, None, r, 0);
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 3: t not multiple of 32
    let bad_t = 37;
    let bad_inputs = vec![1.0f32; bad_t];
    let bad_obs = vec![true; bad_t];
    let err = model.predict_block(
        &bad_inputs,
        &bad_obs,
        &group_ids,
        1,
        None,
        None,
        None,
        r,
        bad_t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 4: inputs length mismatch
    let err = model.predict_block(
        &inputs[..30],
        &observed,
        &group_ids,
        1,
        None,
        None,
        None,
        r,
        t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 5: observed length mismatch
    let err = model.predict_block(
        &inputs,
        &observed[..30],
        &group_ids,
        1,
        None,
        None,
        None,
        r,
        t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 6: group_ids length mismatch
    let err = model.predict_block(&inputs, &observed, &[], 1, None, None, None, r, t);
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 7: role_ids length mismatch
    let roles = vec![RowRole::Target, RowRole::Target]; // len 2 != r=1
    let err = model.predict_block(
        &inputs,
        &observed,
        &group_ids,
        1,
        Some(&roles),
        None,
        None,
        r,
        t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 8: horizon_patches != 1
    let err = model.predict_block(&inputs, &observed, &group_ids, 2, None, None, None, r, t);
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 9: known_future provided without known_future_mask
    let kf = vec![1.0f32; 32];
    let err = model.predict_block(
        &inputs,
        &observed,
        &group_ids,
        1,
        None,
        Some(&kf),
        None,
        r,
        t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 10: known_future_mask provided without known_future
    let kfm = vec![true; 32];
    let err = model.predict_block(
        &inputs,
        &observed,
        &group_ids,
        1,
        None,
        None,
        Some(&kfm),
        r,
        t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 11: known_future length mismatch
    let bad_kf = vec![1.0f32; 10];
    let err = model.predict_block(
        &inputs,
        &observed,
        &group_ids,
        1,
        None,
        Some(&bad_kf),
        Some(&kfm),
        r,
        t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));

    // Case 12: known_future_mask length mismatch
    let bad_kfm = vec![true; 10];
    let err = model.predict_block(
        &inputs,
        &observed,
        &group_ids,
        1,
        None,
        Some(&kf),
        Some(&bad_kfm),
        r,
        t,
    );
    assert!(matches!(err, Err(ModelError::InvalidInput(_))));
}

#[test]
fn test_predict_block_univariate_monotonic_quantiles() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);

    let r = 1;
    let t = 64; // 2 patches context (32 * 2)
    let inputs: Vec<f32> = (0..t)
        .map(|i| (i as f32 * 0.1).sin() * 5.0 + 10.0)
        .collect();
    let observed = vec![true; t];
    let group_ids = vec![0];
    let horizon_patches = 1;

    let preds = model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            horizon_patches,
            None,
            None,
            None,
            r,
            t,
        )
        .expect("predict_block failed");

    let ps = 32;
    let q = 21;
    assert_eq!(preds.len(), r * ps * q);

    // Check finite and monotonic across quantiles
    for (step, q_row) in preds.chunks_exact(q).enumerate() {
        for (k, &value) in q_row.iter().enumerate() {
            assert!(
                value.is_finite(),
                "step {} quantile {} is not finite: {}",
                step,
                k,
                value
            );
        }
        for (k, pair) in q_row.windows(2).enumerate() {
            let (q_curr, q_next) = (pair[0], pair[1]);
            assert!(
                q_next >= q_curr - 1e-6,
                "step {step} non-monotonic quantile: Q[{k}] = {q_curr} > Q[{}] = {q_next}",
                k + 1
            );
        }
    }
}

#[test]
fn test_predict_block_tensor_matches_flat_predict_block() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);

    let r = 1;
    let t = 96; // 3 patches context
    let inputs: Vec<f32> = (0..t).map(|i| 20.0 + (i as f32) * 0.2).collect();
    let observed = vec![true; t];
    let group_ids = vec![0];
    let horizon_patches = 1;

    let flat_preds = model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            horizon_patches,
            None,
            None,
            None,
            r,
            t,
        )
        .expect("flat predict_block failed");

    let tensor_preds = model
        .predict_block_tensor(
            &inputs,
            &observed,
            &group_ids,
            horizon_patches,
            None,
            None,
            None,
            r,
            t,
        )
        .expect("predict_block_tensor failed");

    assert_eq!(tensor_preds.shape().dims(), [1, 32, 21]);
    let tensor_data = tensor_preds.into_data();
    assert_eq!(
        flat_preds.as_slice(),
        tensor_data.as_slice::<f32>().unwrap()
    );
}

#[test]
#[ignore = "requires the official Timestral-209M checkpoint"]
fn test_predict_block_univariate_golden_reference_numeric_parity() {
    let model = pretrained_209m();

    let r = 1;
    let t = 96; // 3 patches ramp: 20.0, 20.2, 20.4, ...
    let inputs: Vec<f32> = (0..t).map(|i| 20.0 + (i as f32) * 0.2).collect();
    let observed = vec![true; t];
    let group_ids = vec![0];

    let preds = model
        .predict_block(&inputs, &observed, &group_ids, 1, None, None, None, r, t)
        .expect("predict_block failed");

    let q = 21;
    let m = 10; // median index

    // Hard-coded golden reference values from Python reference implementation
    #[expect(
        clippy::excessive_precision,
        reason = "Preserve the decimal literals from the Python reference"
    )]
    let expected_median_first_8 = [
        39.226433f32,
        39.433050f32,
        39.625507f32,
        39.838420f32,
        40.020683f32,
        40.236465f32,
        40.426140f32,
        40.612780f32,
    ];

    for (step, &expected) in expected_median_first_8.iter().enumerate() {
        let actual = preds[step * q + m];
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-4,
            "step {step} median parity mismatch: actual={actual}, expected={expected}, diff={diff}"
        );
    }

    // Step 0 all quantiles check
    #[expect(
        clippy::excessive_precision,
        reason = "Preserve the decimal literals from the Python reference"
    )]
    let step0_expected = [
        39.11465f32,
        39.14779,
        39.16489,
        39.17474,
        39.18370,
        39.19160,
        39.19825,
        39.20511,
        39.21230,
        39.21920,
        39.22643,
        39.23269,
        39.23853,
        39.24519,
        39.25306,
        39.26083,
        39.27020,
        39.28101,
        39.29449,
        39.31694,
        39.35443,
    ];

    for (k, &expected) in step0_expected.iter().enumerate() {
        let actual = preds[k];
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-4,
            "step 0 quantile {k} parity mismatch: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
#[ignore = "requires the official Timestral-209M checkpoint"]
fn test_predict_block_covariates_golden_reference_numeric_parity() {
    let model = pretrained_209m();

    let r = 3;
    let t = 64;
    let ps = 32;
    let q = 21;
    let m = 10;

    let roles = vec![
        RowRole::Target,
        RowRole::PastCovariate,
        RowRole::KnownCovariate,
    ];
    let group_ids = vec![0, 0, 0];

    let mut inputs = vec![0.0f32; r * t];
    let mut observed = vec![true; r * t];

    for i in 0..t {
        inputs[i] = 100.0 + (i as f32);
        if i % 10 == 0 {
            observed[t + i] = false;
        } else {
            inputs[t + i] = (i as f32) * 0.5;
        }
        inputs[2 * t + i] = 5.0;
    }

    let mut known_future = vec![0.0f32; r * ps];
    let mut known_future_mask = vec![false; r * ps];
    for i in 0..ps {
        known_future[2 * ps + i] = 5.0;
        known_future_mask[2 * ps + i] = true;
    }

    let preds = model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            1,
            Some(&roles),
            Some(&known_future),
            Some(&known_future_mask),
            r,
            t,
        )
        .expect("predict_block failed");

    // Hard-coded golden reference values for Target row from Python reference
    let expected_target_median_first_4 = [164.12872f32, 165.12895f32, 166.07727f32, 167.14624f32];

    for (step, &expected) in expected_target_median_first_4.iter().enumerate() {
        let actual = preds[step * q + m];
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-4,
            "target step {step} median mismatch: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_variate_firewall_and_covariate_causality() {
    let device = burn::tensor::Device::flex();
    let model = tiny_model(&device);

    let r = 3;
    let t = 64;
    let ps = 32;
    let q = 21;
    let m = 10;

    let roles = vec![
        RowRole::Target,
        RowRole::PastCovariate,
        RowRole::KnownCovariate,
    ];
    let group_ids = vec![0, 0, 0];

    let mut inputs = vec![0.0f32; r * t];
    let observed = vec![true; r * t];

    for i in 0..t {
        inputs[i] = 100.0 + (i as f32);
        inputs[t + i] = (i as f32) * 0.5;
        inputs[2 * t + i] = 5.0;
    }

    let mut known_future = vec![0.0f32; r * ps];
    let mut known_future_mask = vec![false; r * ps];
    for i in 0..ps {
        known_future[2 * ps + i] = 5.0;
        known_future_mask[2 * ps + i] = true;
    }

    // Baseline forecast
    let base_preds = model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            1,
            Some(&roles),
            Some(&known_future),
            Some(&known_future_mask),
            r,
            t,
        )
        .expect("baseline predict_block failed");

    // Test 1: Firewall Check.
    // Modifying Target context values must NOT change the covariate predictions (rows 1 and 2),
    // because the variate firewall blocks attention from covariate queries to target keys.
    let mut inputs_modified_target = inputs.clone();
    for value in &mut inputs_modified_target[..t] {
        *value += 5000.0; // Large shift to target
    }

    let mod_target_preds = model
        .predict_block(
            &inputs_modified_target,
            &observed,
            &group_ids,
            1,
            Some(&roles),
            Some(&known_future),
            Some(&known_future_mask),
            r,
            t,
        )
        .expect("modified target predict_block failed");

    // Target forecast MUST change significantly
    assert!((mod_target_preds[m] - base_preds[m]).abs() > 100.0);

    // Past covariate (row 1) and Known covariate (row 2) predictions MUST be identical
    for step in 0..ps {
        for k in 0..q {
            let row1_base = base_preds[ps * q + step * q + k];
            let row1_mod = mod_target_preds[ps * q + step * q + k];
            assert!(
                (row1_base - row1_mod).abs() < 1e-5,
                "variate firewall violation on row 1 (past cov) step {step} quantile {k}"
            );

            let row2_base = base_preds[2 * ps * q + step * q + k];
            let row2_mod = mod_target_preds[2 * ps * q + step * q + k];
            assert!(
                (row2_base - row2_mod).abs() < 1e-5,
                "variate firewall violation on row 2 (known cov) step {step} quantile {k}"
            );
        }
    }

    // Test 2: Known Future Causality Check.
    // Modifying Known-Future covariate values in the horizon MUST influence Target predictions.
    let mut known_future_changed = known_future.clone();
    for i in 0..ps {
        known_future_changed[2 * ps + i] = 50.0; // Changed from 5.0 to 50.0
    }

    let changed_kf_preds = model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            1,
            Some(&roles),
            Some(&known_future_changed),
            Some(&known_future_mask),
            r,
            t,
        )
        .expect("changed kf predict_block failed");

    // Target forecast must change
    let target_base = base_preds[m];
    let target_changed = changed_kf_preds[m];
    assert!(
        (target_base - target_changed).abs() > 1e-3,
        "Target forecast should change when known future covariates change: base={target_base}, changed={target_changed}"
    );
}

#[test]
fn test_predict_block_disabled_covariates_ignores_roles_and_future() {
    let device = burn::tensor::Device::flex();
    let mut config = tiny_config();
    config.covariate_enabled = false;
    let model = TimestralModel::new(config.clone(), &device).unwrap();

    let r = 2;
    let t = 64;
    let ps = 32;
    let mut inputs = vec![0.0f32; r * t];
    let mut observed = vec![true; r * t];
    for i in 0..t {
        inputs[i] = (i as f32 * 0.3).sin() * 4.0 + 20.0;
        inputs[t + i] = i as f32 * 0.5 + 3.0;
    }
    observed[t + 10] = false;
    inputs[t + 10] = 0.0;
    let group_ids = vec![0, 0];
    let roles = vec![RowRole::Target, RowRole::KnownCovariate];
    let known_future: Vec<f32> = (0..r * ps)
        .map(|i| {
            if i < ps {
                0.0
            } else {
                ((i - ps) as f32 * 0.7).cos() * 2.0 + 5.0
            }
        })
        .collect();
    let known_future_mask = vec![true; r * ps];

    let baseline = model
        .predict_block(&inputs, &observed, &group_ids, 1, None, None, None, r, t)
        .expect("baseline predict_block failed");
    let with_roles_and_future = model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            1,
            Some(&roles),
            Some(&known_future),
            Some(&known_future_mask),
            r,
            t,
        )
        .expect("predict_block with roles and future failed");
    let with_roles_only = model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            1,
            Some(&roles),
            None,
            None,
            r,
            t,
        )
        .expect("predict_block with roles only failed");

    assert_eq!(baseline.len(), r * ps * config.num_quantiles());
    assert_eq!(
        with_roles_and_future, baseline,
        "disabled covariates must ignore roles and known future"
    );
    assert_eq!(
        with_roles_only, baseline,
        "disabled covariates must ignore roles without known future"
    );

    let cov_model = tiny_model(&device);
    let cov_baseline = cov_model
        .predict_block(&inputs, &observed, &group_ids, 1, None, None, None, r, t)
        .expect("covariate-enabled baseline predict_block failed");
    let cov_preds = cov_model
        .predict_block(
            &inputs,
            &observed,
            &group_ids,
            1,
            Some(&roles),
            Some(&known_future),
            Some(&known_future_mask),
            r,
            t,
        )
        .expect("predict_block with covariates enabled failed");
    let max_diff = cov_preds
        .iter()
        .zip(cov_baseline.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "covariate-enabled forecast must react to roles and known future, max_diff={max_diff}"
    );
}
