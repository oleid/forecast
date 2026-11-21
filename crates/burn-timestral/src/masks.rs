//! Attention masks and RoPE positions for Timestral.
//!
//! Builds causal and two-stream time masks, cross-variate masks, and positions
//! anchored at the first observed patch in each series group.

use std::{collections::HashMap, iter::repeat_n};

#[cfg(test)]
#[path = "../tests/unit/test_masks.rs"]
mod tests;

/// Computes encoder key validity for one stream.
///
/// If `attend_missing` is true, all patches from the first observation onward
/// are valid keys. Leading padding patches remain invalid.
pub(crate) fn compute_enc_key_validity(obs_per_patch: &[bool], attend_missing: bool) -> Vec<bool> {
    if !attend_missing {
        return obs_per_patch.to_vec();
    }
    let mut started = false;
    let mut key_valid = Vec::with_capacity(obs_per_patch.len());
    for &obs in obs_per_patch {
        if obs {
            started = true;
        }
        key_valid.push(started);
    }
    key_valid
}

/// Builds a causal encoder mask (`L × L`) over valid keys. The diagonal is always allowed.
#[cfg(test)]
fn enc_causal_mask(key_valid: &[bool]) -> Vec<bool> {
    let l = key_valid.len();
    let mut mask = Vec::with_capacity(l * l);
    for i in 0..l {
        for (j, &valid) in key_valid.iter().enumerate() {
            let allowed = (j <= i && valid) || (i == j);
            mask.push(allowed);
        }
    }
    mask
}

/// Builds a time mask for a block without covariates (`(L + H) × (L + H)`).
///
/// - Encoder attention is causal over valid keys.
/// - Decoder tokens attend to all valid context keys and causally to other decoder tokens.
/// - Context tokens never attend to the forecast horizon.
#[cfg(test)]
fn build_forecast_time_mask(context_len: usize, horizon: usize, key_valid: &[bool]) -> Vec<bool> {
    assert_eq!(context_len, key_valid.len());
    let total_len = context_len + horizon;
    let mut mask = Vec::with_capacity(total_len * total_len);

    for i in 0..total_len {
        for (j, valid) in key_valid
            .iter()
            .copied()
            .chain(repeat_n(false, horizon))
            .enumerate()
        {
            let allowed = if i < context_len {
                if j < context_len {
                    (j <= i && valid) || (i == j)
                } else {
                    false
                }
            } else if j < context_len {
                valid
            } else {
                j <= i
            };
            mask.push(allowed);
        }
    }
    mask
}

/// Builds RoPE positions for the context and horizon, anchored at the first observed patch.
#[cfg(test)]
fn build_forecast_positions(context_len: usize, horizon: usize, first_obs: i64) -> Vec<f32> {
    let total = context_len + horizon;
    (0..total).map(|p| (p as i64 - first_obs) as f32).collect()
}

/// Computes the block anchor and activity flag for each decoder position.
fn compute_block_layout(
    min_context: usize,
    n_patches: usize,
    block_size: usize,
) -> (Vec<usize>, Vec<bool>) {
    let mut anchor = Vec::with_capacity(n_patches);
    let mut active = Vec::with_capacity(n_patches);

    for p in 0..n_patches {
        let is_active = p >= min_context;
        active.push(is_active);
        if is_active {
            let offset = p - min_context;
            let anc = min_context + block_size * (offset / block_size);
            anchor.push(anc);
        } else {
            anchor.push(0);
        }
    }

    (anchor, active)
}

/// Builds a two-stream Boolean time mask for `[Enc; Dec]` (`2N × 2N`), where `true` blocks attention.
///
/// - Enc to Enc: Causal over valid keys, or bidirectional for known future covariates.
/// - Enc to Dec: Never allowed.
/// - Dec to Enc: A decoder token can attend to valid encoder keys before its block anchor.
/// - Dec to Dec: Causal within the block; each decoder token can attend to itself.
pub(crate) fn build_two_stream_time_mask(
    min_context: usize,
    n_patches: usize,
    block_size: usize,
    key_valid: &[bool],
    is_bidirectional: bool,
) -> Vec<bool> {
    assert_eq!(n_patches, key_valid.len());
    let (anchor, active) = compute_block_layout(min_context, n_patches, block_size);
    let total = 2 * n_patches;
    let mut mask = Vec::with_capacity(total * total);

    for i in 0..total {
        for (j, valid) in key_valid
            .iter()
            .copied()
            .chain(repeat_n(false, n_patches))
            .enumerate()
        {
            let allowed = if i < n_patches {
                if j < n_patches {
                    // Enc -> Enc
                    if is_bidirectional {
                        valid || (i == j)
                    } else {
                        (j <= i && valid) || (i == j)
                    }
                } else {
                    // Enc -> Dec (never allowed)
                    false
                }
            } else {
                let dec_i = i - n_patches;
                if j < n_patches {
                    // Dec -> Enc
                    active[dec_i] && (j < anchor[dec_i]) && valid
                } else {
                    // Dec -> Dec
                    let dec_j = j - n_patches;
                    let same_block = anchor[dec_i] == anchor[dec_j];
                    (same_block && (dec_j <= dec_i) && active[dec_i] && active[dec_j])
                        || (dec_i == dec_j)
                }
            };
            mask.push(!allowed);
        }
    }
    mask
}

/// Identifies blocked `(query row, key row)` pairs across variates (`R × R`).
///
/// - Exogenous covariate rows cannot attend to target rows.
/// - Bidirectional known-future rows cannot attend to causal rows, preventing leakage.
fn variate_firewall(
    num_rows: usize,
    target_rows: Option<&[bool]>,
    bidirectional_rows: Option<&[bool]>,
) -> Option<Vec<bool>> {
    if num_rows == 0 || (target_rows.is_none() && bidirectional_rows.is_none()) {
        return None;
    }

    let mut forbid = vec![false; num_rows * num_rows];
    let mut any_forbid = false;

    for (q, row) in forbid.chunks_exact_mut(num_rows).enumerate() {
        let q_is_cov = target_rows.is_some_and(|targets| !targets[q]);
        let q_is_bidir = bidirectional_rows.is_some_and(|bidir| bidir[q]);
        for (k, blocked) in row.iter_mut().enumerate() {
            let k_is_tgt = target_rows.is_some_and(|targets| targets[k]);
            let k_is_causal = bidirectional_rows.is_some_and(|bidir| !bidir[k]);
            *blocked = (q_is_cov && k_is_tgt) || (q_is_bidir && k_is_causal);
            any_forbid |= *blocked;
        }
    }

    if any_forbid { Some(forbid) } else { None }
}

/// Builds a two-stream Boolean variate mask (`S × R × R`), where `true` blocks attention.
///
/// A row can attend to another row in the same group if the key is valid at
/// that slot and no firewall rule blocks the pair. The diagonal is always allowed.
pub(crate) fn build_two_stream_variate_mask(
    group_ids: &[usize],
    enc_key_valid: &[Vec<bool>],
    n_dec: usize,
    target_rows: Option<&[bool]>,
    bidirectional_rows: Option<&[bool]>,
) -> Option<Vec<bool>> {
    let r = group_ids.len();
    if r <= 1 {
        return None;
    }

    assert_eq!(r, enc_key_valid.len());
    let n_enc = enc_key_valid[0].len();
    let total_slots = n_enc + n_dec;
    let forbid = variate_firewall(r, target_rows, bidirectional_rows);

    let mut mask = Vec::with_capacity(total_slots * r * r);

    for t in 0..total_slots {
        for (q, &query_group) in group_ids.iter().enumerate() {
            let blocked_keys = forbid.as_ref().map(|f| &f[q * r..(q + 1) * r]);
            for (k, (&key_group, key_validity)) in group_ids.iter().zip(enc_key_valid).enumerate() {
                let row_eq = query_group == key_group;
                let k_valid = t >= n_enc || key_validity[t];
                let blocked = blocked_keys.is_some_and(|keys| keys[k]);
                let allowed = q == k || (row_eq && k_valid && !blocked);
                mask.push(!allowed);
            }
        }
    }

    Some(mask)
}

/// Builds RoPE positions for `[Enc; Dec]` (length `2N`).
/// Encoder and decoder tokens for the same patch share a position.
pub(crate) fn build_two_stream_positions(n_patches: usize, first_obs: i64) -> Vec<f32> {
    let mut positions = Vec::with_capacity(2 * n_patches);
    for _ in 0..2 {
        for p in 0..n_patches {
            positions.push((p as i64 - first_obs) as f32);
        }
    }
    positions
}

/// For each row, returns the first observed patch in its group (zero if unobserved).
pub(crate) fn group_first_observed_patch(
    obs_per_patch: &[Vec<bool>],
    group_ids: &[usize],
) -> Vec<i64> {
    debug_assert_eq!(obs_per_patch.len(), group_ids.len());
    let mut group_first = HashMap::<usize, usize>::new();
    for (&group, row) in group_ids.iter().zip(obs_per_patch) {
        if let Some(patch) = row.iter().position(|&observed| observed) {
            group_first
                .entry(group)
                .and_modify(|first| *first = (*first).min(patch))
                .or_insert(patch);
        }
    }

    group_ids
        .iter()
        .map(|group| group_first.get(group).copied().unwrap_or(0) as i64)
        .collect()
}
