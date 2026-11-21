use super::{
    advancing_block_frames, apply_affine_asinh_masked, causal_patch_loc_scale, clamp_nonfinite_row,
    compute_causal_statistics, cumulative_welford, denormalize_quantiles, instance_norm_stats,
    invert_affine_asinh, known_cov_scaling, welford_combine,
};

#[test]
fn test_compute_causal_statistics_constant() {
    let data = vec![5.0, 5.0, 5.0, 5.0];
    let mask = vec![true, true, true, true];
    let (means, scales) = compute_causal_statistics(&data, &mask, 1e-6, false, 10.0);

    for &m in &means {
        assert!((m - 5.0).abs() < 1e-6);
    }
    for &s in &scales {
        assert!((s - 1e-6).abs() < 1e-6); // Zero standard deviation is clamped to minimum_scale.
    }
}

#[test]
fn test_compute_causal_statistics_analytical() {
    // Step 0: x=10 -> mean=10, variance=0/1=0 -> scale=1e-6
    // Step 1: x=20 -> cum_w=2, cum_x=30, mean=15
    //   increment = (20 - 10)*(20 - 15) = 10 * 5 = 50
    //   m2 = 50, variance = 50 / (2-1) = 50 -> scale = sqrt(50) = 7.0710678...
    // Step 2: x=30 -> cum_w=3, cum_x=60, mean=20
    //   increment = (30 - 15)*(30 - 20) = 15 * 10 = 150
    //   m2 = 50 + 150 = 200, variance = 200 / (3-1) = 100 -> scale = sqrt(100) = 10.0
    let data = vec![10.0, 20.0, 30.0];
    let mask = vec![true, true, true];
    let (means, scales) = compute_causal_statistics(&data, &mask, 1e-6, false, 10.0);

    assert!((means[0] - 10.0).abs() < 1e-6);
    assert!((means[1] - 15.0).abs() < 1e-6);
    assert!((means[2] - 20.0).abs() < 1e-6);

    assert!((scales[0] - 1e-6).abs() < 1e-6);
    assert!((scales[1] - 50.0f64.sqrt()).abs() < 1e-6);
    assert!((scales[2] - 10.0).abs() < 1e-6);
}

#[test]
fn test_compute_causal_statistics_with_unobserved() {
    let data = vec![10.0, 999.0, 20.0];
    let mask = vec![true, false, true];
    let (means, scales) = compute_causal_statistics(&data, &mask, 1e-6, false, 10.0);

    assert!((means[0] - 10.0).abs() < 1e-6);
    assert!((means[1] - 10.0).abs() < 1e-6); // The unobserved step leaves the mean unchanged.
    assert!((means[2] - 15.0).abs() < 1e-6);

    assert!((scales[0] - 1e-6).abs() < 1e-6);
    assert!((scales[1] - 1e-6).abs() < 1e-6);
    assert!((scales[2] - 50.0f64.sqrt()).abs() < 1e-6);
}

#[test]
fn test_causal_patch_loc_scale() {
    let patch_size = 2;
    let data = vec![10.0, 20.0, 30.0, 40.0];
    let mask = vec![true, true, true, true];
    let (locs, scales) = causal_patch_loc_scale(&data, &mask, patch_size, 1e-6, false);

    assert_eq!(locs.len(), 4);
    assert_eq!(scales.len(), 4);

    // Patch 0 ends at index 1 (mean=15.0, scale=sqrt(50)).
    assert!((locs[0] - 15.0).abs() < 1e-6);
    assert!((locs[1] - 15.0).abs() < 1e-6);
    assert!((scales[0] - 50.0f64.sqrt()).abs() < 1e-6);
    assert!((scales[1] - 50.0f64.sqrt()).abs() < 1e-6);

    // Patch 1 ends at index 3 (x=[10,20,30,40], mean=25.0, var=(225+25+25+225)/3 = 166.6667).
    let expected_scale_p1 = (500.0f64 / 3.0).sqrt();
    assert!((locs[2] - 25.0).abs() < 1e-6);
    assert!((locs[3] - 25.0).abs() < 1e-6);
    assert!((scales[2] - expected_scale_p1).abs() < 1e-6);
    assert!((scales[3] - expected_scale_p1).abs() < 1e-6);
}

#[test]
fn test_instance_norm_stats_continuous() {
    let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let mask = vec![true; 8];
    let (loc, scale, is_binary) = instance_norm_stats(&values, &mask, 1e-6);

    // mean = 40/8 = 5.0
    // var = ((9 + 1 + 1 + 1 + 0 + 0 + 4 + 16)/8) = 32/8 = 4.0 -> std = 2.0
    assert!((loc - 5.0).abs() < 1e-6);
    assert!((scale - 2.0).abs() < 1e-6);
    assert!(!is_binary);
}

#[test]
fn test_instance_norm_stats_uses_f32_arithmetic() {
    let values = [16_777_216.0_f32, 1.0, 1.0];
    let mask = [true; 3];
    let (loc, _, is_binary) = instance_norm_stats(&values, &mask, 1e-6);
    let expected_f32 = values.iter().sum::<f32>() / values.len() as f32;
    let mean_f64 = values.iter().map(|&value| f64::from(value)).sum::<f64>() / 3.0;

    assert_eq!(loc, expected_f32);
    assert_ne!(loc, mean_f64 as f32);
    assert!(!is_binary);
}

#[test]
fn test_instance_norm_stats_binary_indicator() {
    let values = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let mask = vec![true, true, true, false, true];
    let (loc, scale, is_binary) = instance_norm_stats(&values, &mask, 1e-6);

    assert_eq!(loc, 0.0);
    assert_eq!(scale, 1.0);
    assert!(is_binary);
}

#[test]
fn test_instance_norm_stats_non_binary_fractional() {
    let values = vec![0.0, 0.5, 1.0];
    let mask = vec![true, true, true];
    let (_loc, _scale, is_binary) = instance_norm_stats(&values, &mask, 1e-6);

    assert!(!is_binary);
}

#[test]
fn test_apply_and_invert_affine_asinh() {
    let data = vec![10.0, 20.0, 30.0];
    let loc = vec![5.0, 5.0, 5.0];
    let scale = vec![2.0, 2.0, 2.0];
    let mask = vec![true, false, true];

    let scaled = apply_affine_asinh_masked(&data, &loc, &scale, &mask, true);
    assert_eq!(scaled.len(), 3);
    assert_eq!(scaled[1], 0.0f32); // Masked -> 0.0

    let val0 = ((10.0 - 5.0) / 2.0f64).asinh() as f32;
    assert!((scaled[0] - val0).abs() < 1e-6);

    // Invert the transform.
    let inv0 = invert_affine_asinh(scaled[0] as f64, loc[0], scale[0], true);
    assert!((inv0 - data[0]).abs() < 1e-5);
}

#[test]
fn test_clamp_nonfinite_row() {
    let mut row = vec![f32::NEG_INFINITY, 2.0, 5.0, f32::INFINITY];
    clamp_nonfinite_row(&mut row);

    assert_eq!(row[0], 2.0);
    assert_eq!(row[1], 2.0);
    assert_eq!(row[2], 5.0);
    assert_eq!(row[3], 5.0);
}

#[test]
fn test_clamp_nonfinite_all_inf() {
    let mut row = vec![f32::INFINITY, f32::NEG_INFINITY];
    clamp_nonfinite_row(&mut row);

    assert_eq!(row[0], 0.0);
    assert_eq!(row[1], 0.0);
}

#[test]
fn test_denormalize_quantiles() {
    // Three quantiles
    let preds = vec![-1.0f32, 0.0, 1.0];
    let num_q = 3;
    let loc = 100.0f32;
    let scale = 10.0f32;

    let denorm = denormalize_quantiles(&preds, num_q, loc, scale, false);
    assert_eq!(denorm.len(), 3);
    assert!((denorm[0] - 90.0).abs() < 1e-5);
    assert!((denorm[1] - 100.0).abs() < 1e-5);
    assert!((denorm[2] - 110.0).abs() < 1e-5);
}

#[test]
fn test_welford_combine() {
    // Group A: [10, 20] -> cnt=2, mean=15, m2=50
    // Group B: [30, 40] -> cnt=2, mean=35, m2=50
    // Combined: [10, 20, 30, 40] -> cnt=4, mean=25, m2 = (225+25+25+225) = 500
    let (cnt, mean, m2) = welford_combine(2.0, 15.0, 50.0, 2.0, 35.0, 50.0);

    assert_eq!(cnt, 4.0);
    assert!((mean - 25.0).abs() < 1e-6);
    assert!((m2 - 500.0).abs() < 1e-6);
}

#[test]
fn test_cumulative_welford() {
    let w = vec![1.0, 1.0, 1.0];
    let mean = vec![10.0, 20.0, 30.0];
    let m2 = vec![0.0, 0.0, 0.0];

    let (cw, cm, c_m2) = cumulative_welford(&w, &mean, &m2);
    assert_eq!(cw, vec![1.0, 2.0, 3.0]);
    assert!((cm[0] - 10.0).abs() < 1e-6);
    assert!((cm[1] - 15.0).abs() < 1e-6);
    assert!((cm[2] - 20.0).abs() < 1e-6);
    assert!((c_m2[2] - 200.0).abs() < 1e-6);
}

#[test]
fn test_advancing_block_frames() {
    let patch_size = 2;
    let context = vec![10.0, 20.0];
    let mask = vec![true, true];
    let sinh_median = [0.0, 0.0]; // One forecast patch.
    let (locs, scales) = advancing_block_frames(&context, &mask, &sinh_median, patch_size, 1, 1e-6);

    assert_eq!(locs.len(), 1);
    assert_eq!(scales.len(), 1);
    assert!((locs[0] - 15.0).abs() < 1e-6);
    assert!((scales[0] - 50.0f64.sqrt()).abs() < 1e-6);
}

#[test]
fn test_advancing_block_frames_multiblock() {
    // Two context patches (32 steps each): patch 0 = 10.0, patch 1 = 20.0.
    // Three forecast patches with sinh_median: patch 0 = 1.0, patch 1 = 0.5, patch 2 = -0.5.
    let ps = 32;
    let mut context = Vec::with_capacity(64);
    context.extend(std::iter::repeat_n(10.0, ps));
    context.extend(std::iter::repeat_n(20.0, ps));
    let mask = vec![true; 64];

    let mut sinh_median = Vec::with_capacity(3 * ps);
    sinh_median.extend(std::iter::repeat_n(1.0, ps));
    sinh_median.extend(std::iter::repeat_n(0.5, ps));
    sinh_median.extend(std::iter::repeat_n(-0.5, ps));

    let (locs, scales) = advancing_block_frames(&context, &mask, &sinh_median, ps, 3, 1e-6);

    assert_eq!(locs.len(), 3);
    assert_eq!(scales.len(), 3);

    let expected_locs = [15.0, 16.679842100808293, 17.27336510871302];
    let expected_scales = [5.039526305634568, 4.748184079860647, 4.234346300409778];

    for i in 0..3 {
        assert!(
            (locs[i] - expected_locs[i]).abs() < 1e-5,
            "loc mismatch at {}: got {}, expected {}",
            i,
            locs[i],
            expected_locs[i]
        );
        assert!(
            (scales[i] - expected_scales[i]).abs() < 1e-5,
            "scale mismatch at {}: got {}, expected {}",
            i,
            scales[i],
            expected_scales[i]
        );
    }
}

#[test]
fn test_known_cov_scaling_selective() {
    // Four rows of four time steps each: target and continuous, binary, and unobserved covariates.
    // The observed continuous values [10, 30, 40] give loc=80/3 and scale=sqrt(1400)/3.
    // The missing step must be set to zero. The observed binary values are not transformed.
    // The last known covariate is entirely unobserved and gets loc=0, scale=min.
    let inputs = vec![
        10.0, 20.0, 30.0, 40.0, // target
        10.0, 20.0, 30.0, 40.0, // continuous known covariate
        0.0, 1.0, 99.0, 0.0, // binary known covariate; 99 is unobserved
        5.0, 6.0, 7.0, 8.0, // all-unobserved known covariate
    ];
    let observed = vec![
        true, true, true, true, // target
        true, false, true, true, // continuous known covariate
        true, true, false, true, // binary known covariate
        false, false, false, false, // all-unobserved known covariate
    ];
    let scaled_init = vec![
        1.0, 2.0, 3.0, 4.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    ];
    let loc_init = vec![
        10.0, 10.0, 10.0, 10.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
        99.0,
    ];
    let scale_init = vec![
        2.0, 2.0, 2.0, 2.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    ];
    let known_rows = vec![false, true, true, true];

    let (scaled_out, loc_out, scale_out) = known_cov_scaling(
        &inputs,
        &observed,
        &scaled_init,
        &loc_init,
        &scale_init,
        &known_rows,
        4,
        4,
        1e-6,
        true,
    );

    // Row 0: the target stays unchanged.
    assert_eq!(&scaled_out[0..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&loc_out[0..4], &[10.0, 10.0, 10.0, 10.0]);
    assert_eq!(&scale_out[0..4], &[2.0, 2.0, 2.0, 2.0]);

    // Row 1: continuous, with statistics based only on observed values.
    let continuous_loc_f32 = 80.0_f32 / 3.0;
    let continuous_scale_f32 = ([10.0_f32, 30.0, 40.0]
        .iter()
        .map(|&value| (value - continuous_loc_f32).powi(2))
        .sum::<f32>()
        / 3.0)
        .sqrt();
    let continuous_loc = f64::from(continuous_loc_f32);
    let continuous_scale = f64::from(continuous_scale_f32);
    let continuous_start = 4;
    let expected_continuous = [10.0_f32, 0.0, 30.0, 40.0].map(|value| {
        if value == 0.0 {
            0.0
        } else {
            ((value - continuous_loc_f32) / continuous_scale_f32).asinh()
        }
    });
    for t in 0..4 {
        assert!((scaled_out[continuous_start + t] - expected_continuous[t]).abs() < 1e-6);
        assert!((loc_out[continuous_start + t] - continuous_loc).abs() < 1e-12);
        assert!((scale_out[continuous_start + t] - continuous_scale).abs() < 1e-12);
    }

    // Row 2: binary without asinh; the unobserved step stays at zero.
    assert_eq!(&scaled_out[8..12], &[0.0, 1.0, 0.0, 0.0]);
    assert_eq!(&loc_out[8..12], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&scale_out[8..12], &[1.0, 1.0, 1.0, 1.0]);

    // Row 3: entirely unobserved -> zero values, loc=0, scale=minimum_scale.
    assert_eq!(&scaled_out[12..16], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&loc_out[12..16], &[0.0, 0.0, 0.0, 0.0]);
    let minimum_scale = f64::from(1e-6_f32);
    assert_eq!(&scale_out[12..16], &[minimum_scale; 4]);
}

#[test]
fn test_compute_causal_statistics_leading_padding() {
    let data = vec![999.0, 999.0, 10.0, 20.0, 999.0, 30.0];
    let mask = vec![false, false, true, true, false, true];
    let (means, scales) = compute_causal_statistics(&data, &mask, 1e-6, false, 10.0);

    // Indices 0 and 1: leading padding -> unobserved
    assert_eq!(means[0], 0.0);
    assert_eq!(means[1], 0.0);
    assert_eq!(scales[0], 1e-6);
    assert_eq!(scales[1], 1e-6);

    // Index 2: x=10 -> mean=10, scale=min_scale
    assert_eq!(means[2], 10.0);
    assert_eq!(scales[2], 1e-6);

    // Index 3: x=20 -> mean=15, scale=sqrt(50)
    assert_eq!(means[3], 15.0);
    assert!((scales[3] - 50.0f64.sqrt()).abs() < 1e-6);

    // Index 4: unobserved -> statistics stay unchanged
    assert_eq!(means[4], 15.0);
    assert!((scales[4] - 50.0f64.sqrt()).abs() < 1e-6);

    // Index 5: x=30 -> mean=20, scale=10.0
    assert_eq!(means[5], 20.0);
    assert!((scales[5] - 10.0).abs() < 1e-6);
}

#[test]
fn test_compute_causal_statistics_stabilize_with_global() {
    let data = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let mask = vec![true, true, true, true, true];
    let (_means, scales) = compute_causal_statistics(&data, &mask, 1e-6, true, 2.0);

    // All values are identical -> global scale = sqrt(0 + 1e-6) = 1e-3
    // Floor = max(1e-6, 1e-3 * 10^-2) = 1e-5
    // Ceil = 1e-3 * 10^2 = 1e-1
    for &s in &scales {
        assert!((s - 1e-5).abs() < 1e-12);
    }

    // For nonconstant data, the scale derived from the global population variance sets a floor
    // that raises the first causal scale above minimum_scale.
    let data = vec![0.0, 10.0, 20.0];
    let mask = vec![true, true, true];
    let (_means, scales) = compute_causal_statistics(&data, &mask, 1e-6, true, 2.0);
    let global_mean = 10.0;
    let global_var = ((0.0f64 - global_mean).powi(2)
        + (10.0f64 - global_mean).powi(2)
        + (20.0f64 - global_mean).powi(2))
        / 3.0;
    let global_scale = (global_var + 1e-6f64).sqrt();
    let expected_floor = (1e-6f64).max(global_scale * 10.0f64.powf(-2.0));
    let expected_scales = [expected_floor, 50.0f64.sqrt(), 10.0];
    for (actual, expected) in scales.iter().zip(expected_scales) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(scales[0] > 1e-6);
}

#[test]
fn test_instance_norm_stats_masked_and_all_unobserved() {
    // Nonbinary values with a mask:
    let values = vec![10.0, 20.0, 30.0];
    let mask = vec![true, true, false];
    let (loc, scale, is_binary) = instance_norm_stats(&values, &mask, 1e-6);

    assert_eq!(loc, 15.0);
    assert!((scale - 5.0).abs() < 1e-6);
    assert!(!is_binary);

    // All values are masked:
    let unobs_values = vec![10.0, 20.0];
    let unobs_mask = vec![false, false];
    let (loc_u, scale_u, is_bin_u) = instance_norm_stats(&unobs_values, &unobs_mask, 1e-6);

    assert_eq!(loc_u, 0.0);
    assert_eq!(scale_u, 1e-6);
    assert!(!is_bin_u);
}

#[test]
fn test_apply_affine_asinh_masked_no_asinh() {
    let data = vec![10.0, 20.0];
    let loc = vec![5.0, 5.0];
    let scale = vec![2.0, 2.0];
    let mask = vec![true, false];

    let scaled = apply_affine_asinh_masked(&data, &loc, &scale, &mask, false);
    assert_eq!(scaled.len(), 2);
    assert_eq!(scaled[0], 2.5f32); // (10 - 5) / 2 = 2.5 without asinh
    assert_eq!(scaled[1], 0.0f32); // Masked -> 0.0
}
