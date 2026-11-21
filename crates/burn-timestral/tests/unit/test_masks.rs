use super::{
    build_forecast_positions, build_forecast_time_mask, build_two_stream_positions,
    build_two_stream_time_mask, build_two_stream_variate_mask, compute_block_layout,
    compute_enc_key_validity, enc_causal_mask, group_first_observed_patch, variate_firewall,
};

#[test]
fn test_compute_enc_key_validity() {
    // 4 Patches: [false, true, false, false]
    // Patch 0 is leading padding and is invalid.
    // Patch 1 is observed and is valid.
    // Patches 2 and 3 follow the first observation and are valid only when `attend_missing` is true.
    let obs = vec![false, true, false, false];

    let valid_attend = compute_enc_key_validity(&obs, true);
    assert_eq!(valid_attend, vec![false, true, true, true]);

    let valid_no_attend = compute_enc_key_validity(&obs, false);
    assert_eq!(valid_no_attend, vec![false, true, false, false]);

    // Entirely unobserved row:
    let all_unobserved = vec![false, false, false];
    let valid_all_unobs = compute_enc_key_validity(&all_unobserved, true);
    assert_eq!(valid_all_unobs, vec![false, false, false]);
}

#[test]
fn test_enc_causal_mask() {
    // key_valid = [false, true, true] (L=3)
    // Expected 3x3 mask:
    // Row 0: the diagonal is allowed, but later keys are blocked -> [true, false, false]
    // Row 1: key 0 is invalid and key 2 is later -> [false, true, false]
    // Row 2: key 0 is invalid and key 1 is valid and causal -> [false, true, true]
    let key_valid = vec![false, true, true];
    let mask = enc_causal_mask(&key_valid);

    assert_eq!(
        mask,
        vec![true, false, false, false, true, false, false, true, true]
    );
}

#[test]
fn test_build_forecast_time_mask() {
    // L=2, H=2, key_valid = [true, true]
    // 4x4 overall
    let key_valid = vec![true, true];
    let mask = build_forecast_time_mask(2, 2, &key_valid);

    // Enc -> Enc:
    // [true, false]
    // [true, true]
    // Enc -> Dec: all false
    // Dec -> Enc: all true (all encoder keys are valid)
    // Dec -> Dec: causal
    assert_eq!(mask.len(), 16);

    // Row 0 (Enc 0)
    assert_eq!(&mask[0..4], &[true, false, false, false]);
    // Row 1 (Enc 1)
    assert_eq!(&mask[4..8], &[true, true, false, false]);
    // Row 2 (Dec 0)
    assert_eq!(&mask[8..12], &[true, true, true, false]);
    // Row 3 (Dec 1)
    assert_eq!(&mask[12..16], &[true, true, true, true]);
}

#[test]
fn test_compute_block_layout() {
    // min_context = 3, n_patches = 8, block_size = 2
    // p=0,1,2: active=false, anchor=0
    // p=3: active=true, offset=0, anchor=3 + 2*(0/2) = 3
    // p=4: active=true, offset=1, anchor=3 + 2*(1/2) = 3
    // p=5: active=true, offset=2, anchor=3 + 2*(2/2) = 5
    // p=6: active=true, offset=3, anchor=3 + 2*(3/2) = 5
    // p=7: active=true, offset=4, anchor=3 + 2*(4/2) = 7
    let (anchor, active) = compute_block_layout(3, 8, 2);

    assert_eq!(
        active,
        vec![false, false, false, true, true, true, true, true]
    );
    assert_eq!(anchor, vec![0, 0, 0, 3, 3, 5, 5, 7]);
}

#[test]
fn test_build_two_stream_time_mask_full_10x10_causal_and_bidirectional() {
    // Full 10x10 mask for N=5, min_context=2, block_size=2:
    // key_valid = [false, true, true, false, true]
    // Layout:
    // Decoders:
    // p=0,1 (inactive, anchor=0)
    // p=2,3 (active, block 0, anchor=2)
    // p=4   (active, block 1, anchor=4)
    let key_valid = vec![false, true, true, false, true];

    // 1. Causal (is_bidirectional = false):
    let mask_causal = build_two_stream_time_mask(2, 5, 2, &key_valid, false);
    assert_eq!(mask_causal.len(), 100);
    let allowed_causal: Vec<_> = mask_causal.iter().map(|&blocked| !blocked).collect();

    #[rustfmt::skip]
    let expected_causal: Vec<bool> = vec![
        // Enc -> Enc                      | Enc -> Dec
        true,  false, false, false, false,   false, false, false, false, false, // Enc 0 (diag)
        false, true,  false, false, false,   false, false, false, false, false, // Enc 1 (diag/valid)
        false, true,  true,  false, false,   false, false, false, false, false, // Enc 2
        false, true,  true,  true,  false,   false, false, false, false, false, // Enc 3 (diag)
        false, true,  true,  false, true,    false, false, false, false, false, // Enc 4
        // Dec -> Enc                      | Dec -> Dec
        false, false, false, false, false,   true,  false, false, false, false, // Dec 0 (inactive)
        false, false, false, false, false,   false, true,  false, false, false, // Dec 1 (inactive)
        false, true,  false, false, false,   false, false, true,  false, false, // Dec 2 (anchor=2, reads Enc[j<2])
        false, true,  false, false, false,   false, false, true,  true,  false, // Dec 3 (anchor=2, reads Enc[j<2], Dec 2)
        false, true,  true,  false, false,   false, false, false, false, true,  // Dec 4 (anchor=4, reads Enc[j<4])
    ];
    assert_eq!(allowed_causal, expected_causal);

    // 2. Bidirectional (is_bidirectional = true):
    let mask_bidir = build_two_stream_time_mask(2, 5, 2, &key_valid, true);
    assert_eq!(mask_bidir.len(), 100);
    let allowed_bidir: Vec<_> = mask_bidir.iter().map(|&blocked| !blocked).collect();

    #[rustfmt::skip]
    let expected_bidir: Vec<bool> = vec![
        // Enc -> Enc                      | Enc -> Dec
        true,  true,  true,  false, true,    false, false, false, false, false, // Enc 0 (full valid + diag)
        false, true,  true,  false, true,    false, false, false, false, false, // Enc 1
        false, true,  true,  false, true,    false, false, false, false, false, // Enc 2
        false, true,  true,  true,  true,    false, false, false, false, false, // Enc 3 (diag)
        false, true,  true,  false, true,    false, false, false, false, false, // Enc 4
        // Dec -> Enc                      | Dec -> Dec
        false, false, false, false, false,   true,  false, false, false, false, // Dec 0 (inactive)
        false, false, false, false, false,   false, true,  false, false, false, // Dec 1 (inactive)
        false, true,  false, false, false,   false, false, true,  false, false, // Dec 2 (anchor=2)
        false, true,  false, false, false,   false, false, true,  true,  false, // Dec 3 (anchor=2)
        false, true,  true,  false, false,   false, false, false, false, true,  // Dec 4 (anchor=4)
    ];
    assert_eq!(allowed_bidir, expected_bidir);
}

#[test]
fn test_build_two_stream_time_mask_blocks_attention_with_true() {
    let key_valid = vec![true, true];
    let mask = build_two_stream_time_mask(1, 2, 1, &key_valid, false);
    assert!(!mask[0]);
    assert!(mask[1]);
}

#[test]
fn test_variate_firewall_all_combinations() {
    // Three rows:
    // Row 0: target (target = true, bidirectional = false)
    // Row 1: past covariate (target = false, bidirectional = false)
    // Row 2: known covariate (target = false, bidirectional = true)
    let target_rows = vec![true, false, false];
    let bidirectional_rows = vec![false, false, true];

    let forbid = variate_firewall(3, Some(&target_rows), Some(&bidirectional_rows))
        .expect("should have forbid rules");

    // target -> target (0 -> 0): allowed
    assert!(!forbid[0]);
    // target -> past (0 -> 1): allowed
    assert!(!forbid[1]);
    // target -> known (0 -> 2): allowed
    assert!(!forbid[2]);

    // past -> target (1 -> 0): blocked (a covariate cannot attend to the target)
    assert!(forbid[3]);
    // past -> past (1 -> 1): allowed
    assert!(!forbid[4]);
    // past -> known (1 -> 2): allowed
    assert!(!forbid[5]);

    // known -> target (2 -> 0): blocked (a covariate cannot attend to the target)
    assert!(forbid[6]);
    // known -> past (2 -> 1): blocked (a bidirectional row cannot attend to a causal row)
    assert!(forbid[7]);
    // known -> known (2 -> 2): allowed
    assert!(!forbid[8]);
}

#[test]
fn test_build_two_stream_variate_mask_with_firewalls_and_slots() {
    // Three rows in the same group [0, 0, 0]:
    // Row 0: target
    // Row 1: past covariate
    // Row 2: known covariate
    let group_ids = vec![0, 0, 0];
    let target_rows = vec![true, false, false];
    let bidirectional_rows = vec![false, false, true];

    // N_enc = 2 Patches:
    // Row 0 key_valid: [false, true]  (patch 0 is invalid)
    // Row 1 key_valid: [true, true]
    // Row 2 key_valid: [true, false]  (patch 1 is invalid)
    let enc_key_valid = vec![vec![false, true], vec![true, true], vec![true, false]];
    let n_dec = 1; // One decoder slot -> three total slots (Enc 0, Enc 1, Dec 0).

    let mask = build_two_stream_variate_mask(
        &group_ids,
        &enc_key_valid,
        n_dec,
        Some(&target_rows),
        Some(&bidirectional_rows),
    )
    .expect("variate mask exists");
    let allowed: Vec<_> = mask.iter().map(|&blocked| !blocked).collect();

    assert_eq!(mask.len(), 3 * 3 * 3); // Three slots, each with a 3x3 matrix.

    // Slot 0 (Enc 0): key validity is the first entry of each row -> [false, true, true].
    // Row 0 (target):
    //   k=0: the diagonal is allowed even when the key is invalid.
    //   k=1: valid key and no firewall rule -> allowed.
    //   k=2: valid key and no firewall rule -> allowed.
    assert_eq!(&allowed[0..3], &[true, true, true]);
    // Row 1 (past covariate):
    //   k=0: the firewall blocks past -> target.
    //   k=1: the diagonal is allowed.
    //   k=2: valid key -> allowed.
    assert_eq!(&allowed[3..6], &[false, true, true]);
    // Row 2 (known covariate):
    //   k=0: the firewall blocks known -> target.
    //   k=1: the firewall blocks known -> past.
    //   k=2: the diagonal is allowed.
    assert_eq!(&allowed[6..9], &[false, false, true]);

    // Slot 1 (Enc 1): Row 2 is an invalid key, so Target -> Known is blocked.
    // At decoder slot 2, the same pair is allowed because decoder keys are valid.
    assert!(!allowed[9 + 2]);
    assert!(allowed[18 + 2]);

    // Slot 2 (Dec 0): all decoder keys are valid.
    // Row 0 (target): attends to all rows.
    assert_eq!(&allowed[18..21], &[true, true, true]);
    // Row 1 (past covariate): key 0 is blocked; keys 1 and 2 are allowed.
    assert_eq!(&allowed[21..24], &[false, true, true]);
    // Row 2 (known covariate): keys 0 and 1 are blocked; key 2 is allowed.
    assert_eq!(&allowed[24..27], &[false, false, true]);
}

#[test]
fn test_build_two_stream_variate_mask_isolates_groups_at_every_slot() {
    let group_ids = vec![0, 0, 1];
    let enc_key_valid = vec![vec![true, false], vec![true, true], vec![false, true]];
    let n_dec = 1;

    let mask = build_two_stream_variate_mask(&group_ids, &enc_key_valid, n_dec, None, None)
        .expect("variate mask exists for multiple rows");
    let rows = group_ids.len();
    let slots = enc_key_valid[0].len() + n_dec;

    for slot in 0..slots {
        let slot_start = slot * rows * rows;
        // Rows 0 and 2 belong to different series and cannot attend to one another.
        assert!(mask[slot_start + 2], "0 -> 2 at slot {slot}");
        assert!(mask[slot_start + 2 * rows], "2 -> 0 at slot {slot}");

        // The diagonal is always open, even when an encoder key is invalid.
        for row in 0..rows {
            assert!(
                !mask[slot_start + row * rows + row],
                "diagonal at slot {slot}, row {row}"
            );
        }
    }
}

#[test]
fn test_build_two_stream_variate_mask_single_row_returns_none() {
    let group_ids = vec![0];
    let enc_key_valid = vec![vec![true, true]];
    assert!(build_two_stream_variate_mask(&group_ids, &enc_key_valid, 1, None, None).is_none());
}

#[test]
fn test_build_two_stream_positions() {
    // n_patches = 3, first_obs = 1
    // Positions are [0 - 1, 1 - 1, 2 - 1] = [-1.0, 0.0, 1.0] for both streams.
    let pos = build_two_stream_positions(3, 1);
    assert_eq!(pos, vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
}

#[test]
fn test_build_forecast_positions() {
    let pos = build_forecast_positions(2, 2, 1);
    assert_eq!(pos, vec![-1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_group_first_observed_patch() {
    // Two groups, three rows:
    // Row 0: group 0, patches [false, true, true] -> first_obs = 1
    // Row 1: group 0, patches [false, false, true]
    // Row 2: group 1, patches [true, true, true] -> first_obs = 0
    let obs = vec![
        vec![false, true, true],
        vec![false, false, true],
        vec![true, true, true],
    ];
    let group_ids = vec![0, 0, 1];

    let group_first = group_first_observed_patch(&obs, &group_ids);
    assert_eq!(group_first, vec![1, 1, 0]);
}

#[test]
fn test_group_first_observed_patch_accepts_sparse_group_ids() {
    let obs = vec![vec![false, true], vec![false, false]];
    let group_ids = vec![usize::MAX, 5];
    assert_eq!(group_first_observed_patch(&obs, &group_ids), vec![1, 0]);
}
