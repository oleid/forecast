use super::{ModelError, target_encode};

#[test]
fn test_target_encode_basic() {
    // Three time steps in the context:
    // Target: [10.0, 20.0, 30.0] -> item_mean = 20.0
    // Past codes: [0.0, 0.0, 1.0]
    // Category 0.0: target values [10.0, 20.0], sum = 30.0, count = 2
    // Category 1.0: target values [30.0], sum = 30.0, count = 1
    // Encoding slots with smooth = 1.0:
    // Slot 0 (category 0.0): (1.0 * 20.0 + 30.0) / (1.0 + 2.0) = 50.0 / 3.0 = 16.666666...
    // Slot 1 (category 1.0): (1.0 * 20.0 + 30.0) / (1.0 + 1.0) = 50.0 / 2.0 = 25.0
    // Slot 2 (NaN):          (1.0 * 20.0 + 0.0) / (1.0 + 0.0) = 20.0 / 1.0 = 20.0
    // Slot 3 (unseen):       20.0
    let target = [10.0, 20.0, 30.0];
    let past_codes = [0.0, 0.0, 1.0];
    let future_codes = [0.0, 1.0, 2.0, f64::NAN]; // 2.0 is unseen.

    let (encoded_past, encoded_future) =
        target_encode(&target, &past_codes, Some(&future_codes), 1.0).unwrap();

    let fut = encoded_future.expect("future should be encoded");

    assert_eq!(encoded_past.len(), 3);
    assert!((encoded_past[0] - (50.0 / 3.0) as f32).abs() < 1e-6);
    assert!((encoded_past[1] - (50.0 / 3.0) as f32).abs() < 1e-6);
    assert!((encoded_past[2] - 25.0f32).abs() < 1e-6);

    assert_eq!(fut.len(), 4);
    assert!((fut[0] - (50.0 / 3.0) as f32).abs() < 1e-6);
    assert!((fut[1] - 25.0f32).abs() < 1e-6);
    // The unseen category falls back to item_mean.
    assert!((fut[2] - 20.0f32).abs() < 1e-6);
    // With no context observations in the NaN slot, (1 * 20 + 0) / 1 = 20.
    assert!((fut[3] - 20.0f32).abs() < 1e-6);
}

#[test]
fn test_target_encode_with_nans_in_target_and_past() {
    // Target: [10.0, NaN, 30.0, 40.0]; observed values: [10.0, 30.0, 40.0].
    // sum = 80, count = 3, item_mean = 80/3 = 26.666667
    // Past codes: [0.0, 0.0, f64::NAN, 1.0]
    // Categories: [0.0, 1.0], nan_slot = 2
    // Index 0: category 0.0, target 10.0 -> sum[0]=10, count[0]=1
    // Index 1: category 0.0, target NaN -> ignored
    // Index 2: NaN code, target 30.0 -> sum[nan]=30, count[nan]=1
    // Index 3: category 1.0, target 40.0 -> sum[1]=40, count[1]=1
    // Encoding slots with smooth = 1.0:
    // Slot 0 (category 0.0): (1.0 * 26.666667 + 10.0) / 2.0 = 18.333333
    // Slot 1 (category 1.0): (1.0 * 26.666667 + 40.0) / 2.0 = 33.333333
    // Slot 2 (NaN):          (1.0 * 26.666667 + 30.0) / 2.0 = 28.333333
    // Slot 3 (unseen):       26.666667
    let target = [10.0, f64::NAN, 30.0, 40.0];
    let past_codes = [0.0, 0.0, f64::NAN, 1.0];
    let future_codes = [f64::NAN, 99.0];

    let (encoded_past, encoded_future) =
        target_encode(&target, &past_codes, Some(&future_codes), 1.0).unwrap();
    let fut = encoded_future.unwrap();

    let mean = 80.0 / 3.0;
    let expected_slot0 = ((1.0 * mean + 10.0) / 2.0) as f32;
    let expected_slot1 = ((1.0 * mean + 40.0) / 2.0) as f32;
    let expected_slot_nan = ((1.0 * mean + 30.0) / 2.0) as f32;
    let expected_unseen = mean as f32;

    assert!((encoded_past[0] - expected_slot0).abs() < 1e-5);
    assert!((encoded_past[1] - expected_slot0).abs() < 1e-5);
    assert!((encoded_past[2] - expected_slot_nan).abs() < 1e-5);
    assert!((encoded_past[3] - expected_slot1).abs() < 1e-5);

    assert!((fut[0] - expected_slot_nan).abs() < 1e-5);
    assert!((fut[1] - expected_unseen).abs() < 1e-5);
}

#[test]
fn test_target_encode_no_future() {
    let target = [1.0, 2.0];
    let past_codes = [0.0, 1.0];
    let (encoded_past, encoded_future) = target_encode(&target, &past_codes, None, 1.0).unwrap();
    assert_eq!(encoded_past.len(), 2);
    assert!(encoded_future.is_none());
}

#[test]
fn test_target_encode_complex_parity() {
    let tgt = [12.5, f64::NAN, -3.2, 8.0, 15.1, 0.0, -1.5, 4.2];
    let past = [1.0, 2.0, f64::NAN, 1.0, 3.0, 2.0, 1.0, f64::NAN];
    let fut = [1.0, 4.0, f64::NAN, 2.0]; // 4.0 is unseen

    let (p_enc, f_enc) = target_encode(&tgt, &past, Some(&fut), 1.5).unwrap();
    let f_enc = f_enc.unwrap();

    let expected_p = [
        5.893651f32,
        3.0085714,
        2.4346938,
        5.893651,
        9.048572,
        3.0085714,
        5.893651,
        2.4346938,
    ];
    let expected_f = [5.893651f32, 5.0142856, 2.4346938, 3.0085714];

    assert_eq!(p_enc.len(), expected_p.len());
    for (i, (&actual, &exp)) in p_enc.iter().zip(&expected_p).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "p_enc mismatch at index {i}: got {actual}, expected {exp}"
        );
    }

    assert_eq!(f_enc.len(), expected_f.len());
    for (i, (&actual, &exp)) in f_enc.iter().zip(&expected_f).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "f_enc mismatch at index {i}: got {actual}, expected {exp}"
        );
    }
}

#[test]
fn test_target_encode_signed_zero_regression() {
    // Regression test for handling -0.0 and +0.0:
    // target = [1.0, 2.0, 3.0, 10.0]
    // past_codes = [-1.0, -0.0, 0.0, 5.0]
    // future_codes = [0.0, -0.0]
    // Python returns:
    // past: [2.5, 3.0, 3.0, 7.0]
    // future: [3.0, 3.0]
    let tgt = [1.0, 2.0, 3.0, 10.0];
    let past = [-1.0, -0.0, 0.0, 5.0];
    let fut = [0.0, -0.0];

    let (p_enc, f_enc) = target_encode(&tgt, &past, Some(&fut), 1.0).unwrap();
    let f_enc = f_enc.unwrap();

    let expected_p = [2.5f32, 3.0, 3.0, 7.0];
    let expected_f = [3.0f32, 3.0];

    assert_eq!(p_enc.len(), expected_p.len());
    for (i, (&actual, &exp)) in p_enc.iter().zip(&expected_p).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "signed-zero past mismatch at {i}: got {actual}, expected {exp}"
        );
    }

    assert_eq!(f_enc.len(), expected_f.len());
    for (i, (&actual, &exp)) in f_enc.iter().zip(&expected_f).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "signed-zero future mismatch at {i}: got {actual}, expected {exp}"
        );
    }
}

#[test]
fn test_target_encode_all_nan_past_codes() {
    // All past codes are NaN.
    let tgt = [10.0, 20.0];
    let past = [f64::NAN, f64::NAN];
    let fut = [1.0, f64::NAN]; // 1.0 is unseen.

    let (p_enc, f_enc) = target_encode(&tgt, &past, Some(&fut), 1.0).unwrap();
    let f_enc = f_enc.unwrap();

    // item_mean = 15.0
    // nan_slot: (1.0 * 15.0 + 30.0) / (1.0 + 2.0) = 45.0 / 3.0 = 15.0
    // unseen: 15.0
    assert_eq!(p_enc, vec![15.0f32, 15.0f32]);
    assert_eq!(f_enc, vec![15.0f32, 15.0f32]);
}

#[test]
fn test_target_encode_python_reference_golden_master() {
    // Independent golden reference test computed from the official Python Timestral reference:
    // target = [5.5, -2.0, NaN, 14.0, -8.5, 3.0, 0.0, -1.0, 9.0, NaN] (mean = 2.5)
    // past_codes = [1.0, 2.0, 1.0, NaN, 2.0, 3.0, 1.0, NaN, 2.0, 3.0]
    // future_codes = [1.0, 2.0, 3.0, 4.0, NaN, 0.0] (4.0 and 0.0 are unseen)
    // smooth = 1.0 (Python default)
    let target = [
        5.5,
        -2.0,
        f64::NAN,
        14.0,
        -8.5,
        3.0,
        0.0,
        -1.0,
        9.0,
        f64::NAN,
    ];
    let past_codes = [1.0, 2.0, 1.0, f64::NAN, 2.0, 3.0, 1.0, f64::NAN, 2.0, 3.0];
    let future_codes = [1.0, 2.0, 3.0, 4.0, f64::NAN, 0.0];

    let (p_enc, f_enc) = target_encode(&target, &past_codes, Some(&future_codes), 1.0).unwrap();
    let f_enc = f_enc.expect("future should be encoded");

    let expected_past: [f32; 10] = [
        2.6666667, 0.25, 2.6666667, 5.1666665, 0.25, 2.75, 2.6666667, 5.1666665, 0.25, 2.75,
    ];
    let expected_future: [f32; 6] = [2.6666667, 0.25, 2.75, 2.5, 5.1666665, 2.5];

    assert_eq!(p_enc.len(), expected_past.len());
    for (i, (&actual, &expected)) in p_enc.iter().zip(&expected_past).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Python golden past mismatch at index {i}: got {actual}, expected {expected}"
        );
    }

    assert_eq!(f_enc.len(), expected_future.len());
    for (i, (&actual, &expected)) in f_enc.iter().zip(&expected_future).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Python golden future mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_target_encode_rejects_length_mismatch() {
    let tgt = [1.0, 2.0];
    let past = [1.0];
    assert!(matches!(
        target_encode(&tgt, &past, None, 1.0),
        Err(ModelError::InvalidInput(_))
    ));
}
