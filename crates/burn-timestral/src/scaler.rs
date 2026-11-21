//! Causal scaling, Welford statistics, instance normalization, and denormalization.

#[cfg(test)]
#[path = "../tests/unit/test_scaler.rs"]
mod tests;

/// Computes causal running mean and standard deviation in `f64` using Welford statistics.
fn compute_causal_statistics(
    data: &[f64],
    mask: &[bool],
    minimum_scale: f64,
    stabilize_with_global: bool,
    scale_factor_exponent: f64,
) -> (Vec<f64>, Vec<f64>) {
    let t = data.len();
    assert_eq!(
        t,
        mask.len(),
        "data length ({}) and mask length ({}) must match",
        t,
        mask.len()
    );

    let mut means = vec![0.0f64; t];
    let mut scales = vec![minimum_scale; t];
    if t == 0 {
        return (means, scales);
    }

    let mut cum_w = 0.0f64;
    let mut cum_x = 0.0f64;
    let mut m2 = 0.0f64;
    let mut prev_mean = 0.0f64;

    for ((&value, &observed), (mean, scale)) in
        data.iter().zip(mask).zip(means.iter_mut().zip(&mut scales))
    {
        let w = if observed { 1.0f64 } else { 0.0f64 };
        let x = if observed { value } else { 0.0f64 };
        cum_w += w;
        cum_x += w * x;
        let denominator = cum_w.max(1.0);
        let cur_mean = cum_x / denominator;
        *mean = cur_mean;

        let increment = (x - prev_mean) * (x - cur_mean) * w;
        m2 += increment;

        let var_den = (denominator - 1.0).max(1.0);
        let variance = m2 / var_den;
        *scale = variance.sqrt().max(minimum_scale);

        prev_mean = cur_mean;
    }

    if stabilize_with_global {
        let lo = 10.0f64.powf(-scale_factor_exponent);
        let hi = 10.0f64.powf(scale_factor_exponent);
        let global_den = cum_w.max(1.0);
        let global_mean = cum_x / global_den;
        let mut global_var = 0.0f64;
        for (&value, &observed) in data.iter().zip(mask) {
            if observed {
                global_var += (value - global_mean).powi(2);
            }
        }
        global_var /= global_den;
        let global_scale = (global_var + minimum_scale).sqrt();
        let floor = minimum_scale.max(global_scale * lo);
        let ceil = global_scale * hi;
        for scale in &mut scales {
            *scale = scale.clamp(floor, ceil);
        }
    }

    (means, scales)
}

/// Computes causal location and scale per patch and repeats them at each step of the patch.
pub(crate) fn causal_patch_loc_scale(
    data: &[f64],
    mask: &[bool],
    patch_size: usize,
    minimum_scale: f64,
    stabilize_with_global: bool,
) -> (Vec<f64>, Vec<f64>) {
    let t = data.len();
    assert_eq!(t, mask.len());
    assert!(
        patch_size > 0 && t.is_multiple_of(patch_size),
        "length {} is not a multiple of patch_size {}",
        t,
        patch_size
    );

    let (means, scales) =
        compute_causal_statistics(data, mask, minimum_scale, stabilize_with_global, 10.0);

    let mut patch_locs = Vec::with_capacity(t);
    let mut patch_scales = Vec::with_capacity(t);

    for (patch_means, patch_scales_in) in means
        .chunks_exact(patch_size)
        .zip(scales.chunks_exact(patch_size))
    {
        patch_locs.extend(std::iter::repeat_n(
            *patch_means.last().unwrap(),
            patch_size,
        ));
        patch_scales.extend(std::iter::repeat_n(
            *patch_scales_in.last().unwrap(),
            patch_size,
        ));
    }

    (patch_locs, patch_scales)
}

/// Computes noncausal instance normalization statistics for known future covariates.
///
/// Indicator covariates (all observed values are `0` or `1`) retain `loc = 0.0`
/// and `scale = 1.0` and are marked as binary.
fn instance_norm_stats(values: &[f32], mask: &[bool], minimum_scale: f32) -> (f32, f32, bool) {
    assert_eq!(values.len(), mask.len());
    let mut n = 0.0f32;
    let mut sum = 0.0f32;
    let mut has_observed = false;
    let mut is_zero_one = true;

    for (&v, &observed) in values.iter().zip(mask) {
        if observed {
            has_observed = true;
            n += 1.0;
            sum += v;
            if v != 0.0 && v != 1.0 {
                is_zero_one = false;
            }
        }
    }

    let is_binary = is_zero_one && has_observed;
    if is_binary {
        return (0.0, 1.0, true);
    }

    let n_clamped = n.max(1.0);
    let loc = sum / n_clamped;
    let mut var = 0.0f32;
    for (&value, &observed) in values.iter().zip(mask) {
        if observed {
            var += (value - loc).powi(2);
        }
    }
    var /= n_clamped;
    let scale = var.sqrt().max(minimum_scale);
    (loc, scale, false)
}

/// Scales observed values, optionally applies `asinh`, and returns zero at missing positions.
pub(crate) fn apply_affine_asinh_masked(
    data: &[f64],
    loc: &[f64],
    scale: &[f64],
    mask: &[bool],
    use_asinh: bool,
) -> Vec<f32> {
    let t = data.len();
    assert_eq!(t, loc.len());
    assert_eq!(t, scale.len());
    assert_eq!(t, mask.len());

    let mut out = Vec::with_capacity(t);
    for ((&value, &observed), (&loc, &scale)) in data.iter().zip(mask).zip(loc.iter().zip(scale)) {
        if !observed {
            out.push(0.0f32);
        } else {
            let s = scale.max(f64::EPSILON);
            let val = (value - loc) / s;
            let res = if use_asinh { val.asinh() } else { val };
            out.push(res as f32);
        }
    }
    out
}

/// Inverts affine scaling, applying `sinh` first if `asinh` was used.
#[inline]
fn invert_affine_asinh(x: f64, loc: f64, scale: f64, use_asinh: bool) -> f64 {
    let val = if use_asinh { x.sinh() } else { x };
    val * scale + loc
}

/// Replaces infinities with the finite extrema in the quantile row, or zero if none exist.
fn clamp_nonfinite_row(vals: &mut [f32]) {
    let mut max_finite = f32::NEG_INFINITY;
    let mut min_finite = f32::INFINITY;
    let mut has_finite = false;

    for &v in vals.iter() {
        if v.is_finite() {
            has_finite = true;
            if v > max_finite {
                max_finite = v;
            }
            if v < min_finite {
                min_finite = v;
            }
        }
    }

    for v in vals.iter_mut() {
        if *v == f32::INFINITY {
            *v = if has_finite { max_finite } else { 0.0 };
        } else if *v == f32::NEG_INFINITY {
            *v = if has_finite { min_finite } else { 0.0 };
        }
    }
}

/// Denormalizes quantile predictions and clamps infinities within each row.
pub(crate) fn denormalize_quantiles(
    preds: &[f32],
    num_quantiles: usize,
    loc: f32,
    scale: f32,
    use_asinh: bool,
) -> Vec<f32> {
    assert_eq!(preds.len() % num_quantiles, 0);
    let mut out = Vec::with_capacity(preds.len());

    for chunk in preds.chunks(num_quantiles) {
        let mut row: Vec<f32> = chunk
            .iter()
            .map(|&p| invert_affine_asinh(p as f64, loc as f64, scale as f64, use_asinh) as f32)
            .collect();
        clamp_nonfinite_row(&mut row);
        out.extend_from_slice(&row);
    }
    out
}

/// Combines two Welford states `(weight, mean, m2)`.
#[cfg(test)]
fn welford_combine(
    w_a: f64,
    mean_a: f64,
    m2_a: f64,
    w_b: f64,
    mean_b: f64,
    m2_b: f64,
) -> (f64, f64, f64) {
    let w = w_a + w_b;
    let w_safe = w.max(f64::MIN_POSITIVE);
    let delta = mean_b - mean_a;
    let mut mean = mean_a + delta * w_b / w_safe;
    let mut m2 = m2_a + m2_b + delta.powi(2) * w_a * w_b / w_safe;
    if w == 0.0 {
        mean = 0.0;
        m2 = 0.0;
    }
    (w, mean, m2)
}

/// Returns the cumulative Welford state at each step.
#[cfg(test)]
fn cumulative_welford(w: &[f64], mean: &[f64], m2: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = w.len();
    assert_eq!(n, mean.len());
    assert_eq!(n, m2.len());

    let mut out_w = Vec::with_capacity(n);
    let mut out_mean = Vec::with_capacity(n);
    let mut out_m2 = Vec::with_capacity(n);

    let mut cur_w = 0.0f64;
    let mut cur_mean = 0.0f64;
    let mut cur_m2 = 0.0f64;

    for ((&weight, &mean), &m2) in w.iter().zip(mean).zip(m2) {
        let (nw, nm, n2) = welford_combine(cur_w, cur_mean, cur_m2, weight, mean, m2);
        cur_w = nw;
        cur_mean = nm;
        cur_m2 = n2;
        out_w.push(cur_w);
        out_mean.push(cur_mean);
        out_m2.push(cur_m2);
    }
    (out_w, out_mean, out_m2)
}

/// Computes the advancing location and scale for a decoded block.
#[cfg(test)]
fn advancing_block_frames(
    context: &[f64],
    mask: &[bool],
    sinh_median: &[f64],
    patch_size: usize,
    n_pred: usize,
    minimum_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let t = context.len();
    assert_eq!(t, mask.len());
    assert!(patch_size > 0 && t.is_multiple_of(patch_size));
    assert_eq!(sinh_median.len(), n_pred * patch_size);

    let n_ctx = t / patch_size;
    let mut patch_cnt = Vec::with_capacity(n_ctx);
    let mut patch_mean = Vec::with_capacity(n_ctx);
    let mut patch_m2 = Vec::with_capacity(n_ctx);

    for (values, observed) in context
        .chunks_exact(patch_size)
        .zip(mask.chunks_exact(patch_size))
    {
        let mut cnt = 0.0f64;
        let mut sum = 0.0f64;
        for (&value, &is_observed) in values.iter().zip(observed) {
            if is_observed {
                cnt += 1.0;
                sum += value;
            }
        }
        let mean = if cnt > 0.0 { sum / cnt } else { 0.0 };
        let mut m2 = 0.0f64;
        for (&value, &is_observed) in values.iter().zip(observed) {
            if is_observed {
                m2 += (value - mean).powi(2);
            }
        }
        patch_cnt.push(cnt);
        patch_mean.push(mean);
        patch_m2.push(m2);
    }

    let (cum_cnt, cum_mean, cum_m2) = cumulative_welford(&patch_cnt, &patch_mean, &patch_m2);
    let mut cnt = *cum_cnt.last().unwrap_or(&0.0);
    let mut mean = *cum_mean.last().unwrap_or(&0.0);
    let mut m2 = *cum_m2.last().unwrap_or(&0.0);

    let offset = mean;
    let clip = (f32::MAX.sqrt() as f64) / 1000.0;

    let mut locs = Vec::with_capacity(n_pred);
    let mut scales = Vec::with_capacity(n_pred);

    for patch in sinh_median.chunks_exact(patch_size) {
        let var = (m2 / (cnt - 1.0).max(1.0)).max(0.0);
        let scale = var.sqrt().max(minimum_scale);
        locs.push(mean);
        scales.push(scale);

        let mut folded = Vec::with_capacity(patch_size);
        for &sm in patch {
            let raw = sm * scale + mean - offset;
            let val = raw.clamp(-clip, clip) + offset;
            folded.push(val);
        }

        let b_mean = folded.iter().sum::<f64>() / (patch_size as f64);
        let b_m2 = folded.iter().map(|&x| (x - b_mean).powi(2)).sum::<f64>();
        let b_cnt = patch_size as f64;

        let (nc, nm, n_m2) = welford_combine(cnt, mean, m2, b_cnt, b_mean, b_m2);
        cnt = nc;
        mean = nm;
        m2 = n_m2;
    }

    (locs, scales)
}

/// Applies instance normalization to known future covariate rows.
///
/// For rows marked in `known_rows`, computes noncausal statistics over the full row
/// and applies `(x - loc) / scale`. When `use_asinh` is enabled, only observed
/// values in non-indicator rows receive the transform. Binary indicator rows
/// retain `loc = 0.0` and `scale = 1.0` and skip `asinh`.
///
/// For all other rows, `scaled`, `loc`, and `scale` are left unchanged.
#[expect(
    clippy::too_many_arguments,
    reason = "scales aligned value, mask, and statistics buffers"
)]
pub(crate) fn known_cov_scaling(
    inputs: &[f32],
    observed: &[bool],
    scaled: &[f32],
    loc: &[f64],
    scale: &[f64],
    known_rows: &[bool],
    num_rows: usize,
    seq_len: usize,
    minimum_scale: f32,
    use_asinh: bool,
) -> (Vec<f32>, Vec<f64>, Vec<f64>) {
    let total_len = num_rows * seq_len;
    assert_eq!(inputs.len(), total_len);
    assert_eq!(observed.len(), total_len);
    assert_eq!(scaled.len(), total_len);
    assert_eq!(loc.len(), total_len);
    assert_eq!(scale.len(), total_len);
    assert_eq!(known_rows.len(), num_rows);

    if seq_len == 0 || !known_rows.iter().any(|&k| k) {
        return (scaled.to_vec(), loc.to_vec(), scale.to_vec());
    }

    let mut out_scaled = scaled.to_vec();
    let mut out_loc = loc.to_vec();
    let mut out_scale = scale.to_vec();

    let input_rows = inputs
        .chunks_exact(seq_len)
        .zip(observed.chunks_exact(seq_len));
    let stats_rows = out_loc
        .chunks_exact_mut(seq_len)
        .zip(out_scale.chunks_exact_mut(seq_len));
    let output_rows = out_scaled.chunks_exact_mut(seq_len).zip(stats_rows);
    for ((&is_known, (inp_r, obs_r)), (scaled_r, (loc_r, scale_r))) in
        known_rows.iter().zip(input_rows).zip(output_rows)
    {
        if is_known {
            let (loc_c, scale_c, is_binary) = instance_norm_stats(inp_r, obs_r, minimum_scale);
            let s_clamped = scale_c.max(f32::EPSILON);

            loc_r.fill(f64::from(loc_c));
            scale_r.fill(f64::from(scale_c));
            for (scaled, (&value, &observed)) in scaled_r.iter_mut().zip(inp_r.iter().zip(obs_r)) {
                *scaled = if observed {
                    let val = (value - loc_c) / s_clamped;
                    if use_asinh && !is_binary {
                        val.asinh()
                    } else {
                        val
                    }
                } else {
                    0.0
                };
            }
        }
    }

    (out_scaled, out_loc, out_scale)
}
