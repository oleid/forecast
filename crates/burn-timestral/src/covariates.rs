//! Target encoding for categorical covariates.
//!
//! Before inference, categorical covariates are converted to numeric values using
//! smoothed mean target encoding over the target series context:
//!
//! `encoded = (smooth * item_mean + sum_{c}) / (smooth + count_{c})`
//!
//! - Missing codes (`NaN`) have their own slot.
//! - Categories not seen in the context are encoded as `item_mean` in the forecast horizon.
//! - Non-finite values in `target_context` do not contribute to sums or counts.

use crate::ModelError;

#[cfg(test)]
#[path = "../tests/unit/test_covariates.rs"]
mod tests;

type EncodedCovariate = (Vec<f32>, Option<Vec<f32>>);

#[inline]
fn normalize_code(c: f64) -> f64 {
    if c == 0.0 { 0.0 } else { c }
}

/// Encodes past and optional future categories from the observed target context.
pub(crate) fn target_encode(
    target_context: &[f64],
    past_codes: &[f64],
    future_codes: Option<&[f64]>,
    smooth: f64,
) -> Result<EncodedCovariate, ModelError> {
    if target_context.len() != past_codes.len() {
        return Err(ModelError::InvalidInput(format!(
            "target_context length ({}) and past_codes length ({}) must align",
            target_context.len(),
            past_codes.len()
        )));
    }
    if past_codes
        .iter()
        .chain(future_codes.into_iter().flatten())
        .any(|c| c.is_infinite())
    {
        return Err(ModelError::InvalidInput(
            "category codes must be finite or NaN".into(),
        ));
    }
    if !smooth.is_finite() || smooth <= 0.0 {
        return Err(ModelError::InvalidInput(
            "target encoding smoothing must be finite and positive".into(),
        ));
    }

    let mut observed_count = 0usize;
    let mut target_sum = 0.0f64;

    for &t in target_context {
        if t.is_finite() {
            observed_count += 1;
            target_sum += t;
        }
    }

    let item_mean = if observed_count > 0 {
        target_sum / (observed_count as f64)
    } else {
        0.0
    };

    // Collect unique, sorted finite past codes. Normalize -0.0 to 0.0 for consistent comparisons.
    let mut categories: Vec<f64> = past_codes
        .iter()
        .copied()
        .filter(|c| c.is_finite())
        .map(normalize_code)
        .collect();
    categories.sort_unstable_by(f64::total_cmp);
    categories.dedup();

    let nan_slot = categories.len();
    let total_slots = nan_slot + 1; // 0..nan_slot for categories; nan_slot for NaN.

    let past_slots = past_codes
        .iter()
        .map(|&code| {
            get_slot(code, &categories, nan_slot).ok_or_else(|| {
                ModelError::InvalidInput(format!("past category {code} has no slot"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut sums = vec![0.0f64; total_slots];
    let mut counts = vec![0.0f64; total_slots];

    for (&t, &slot) in target_context.iter().zip(&past_slots) {
        if t.is_finite() {
            sums[slot] += t;
            counts[slot] += 1.0;
        }
    }

    // Store smoothed means for seen categories and use the item mean for unseen categories.
    let mut lookup = Vec::with_capacity(total_slots + 1);
    for (sum, count) in sums.into_iter().zip(counts) {
        let smoothed = (smooth * item_mean + sum) / (smooth + count);
        lookup.push(smoothed);
    }
    lookup.push(item_mean); // unseen slot = nan_slot + 1

    let encoded_past: Vec<f32> = past_slots.iter().map(|&s| lookup[s] as f32).collect();

    let encoded_future = future_codes.map(|f_codes| {
        let unseen_slot = nan_slot + 1;
        f_codes
            .iter()
            .map(|&code| {
                let slot = get_slot(code, &categories, nan_slot).unwrap_or(unseen_slot);
                lookup[slot] as f32
            })
            .collect()
    });

    Ok((encoded_past, encoded_future))
}

fn get_slot(code: f64, categories: &[f64], nan_slot: usize) -> Option<usize> {
    if code.is_nan() {
        return Some(nan_slot);
    }
    let norm = normalize_code(code);
    categories.binary_search_by(|c| c.total_cmp(&norm)).ok()
}
