use super::TimestralConfig;

#[test]
fn test_config_rejects_invalid_graph_dimensions() {
    let valid = TimestralConfig::default_209m();
    for invalid in [
        TimestralConfig {
            num_heads: 0,
            ..valid.clone()
        },
        TimestralConfig {
            patch_size: 0,
            ..valid.clone()
        },
        TimestralConfig {
            patch_size: usize::MAX,
            ..valid.clone()
        },
        TimestralConfig {
            rope_partial_factor: 2.0,
            ..valid.clone()
        },
        TimestralConfig {
            norm_eps: 0.0,
            ..valid
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn test_default_209m_config_validation() {
    let cfg = TimestralConfig::default_209m();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.patch_size, 32);
    assert_eq!(cfg.d_model, 768);
    assert_eq!(cfg.num_heads, 12);
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.rope_dim(), 32);
    assert_eq!(cfg.num_quantiles(), 21);
    assert_eq!(cfg.median_index(), Some(10));
    assert!(cfg.supports_known_covariates());
    assert_eq!(cfg.quantile_width_factor, 3.0);
}

#[test]
fn test_config_json_roundtrip() {
    let cfg = TimestralConfig::default_209m();
    let json_str = cfg.to_json_string().expect("serialization should succeed");
    let loaded = TimestralConfig::from_json_str(&json_str).expect("deserialization should succeed");
    assert_eq!(cfg, loaded);
}

#[test]
fn test_supports_known_covariates() {
    let mut cfg = TimestralConfig::default_209m();
    assert!(cfg.supports_known_covariates());

    // Known covariates are unsupported when `covariate_enabled` is false.
    cfg.covariate_enabled = false;
    assert!(!cfg.supports_known_covariates());

    // With only `time` layers, information cannot pass between variates.
    cfg.covariate_enabled = true;
    cfg.block_schedule = vec!["time".into(), "time".into()];
    assert!(!cfg.supports_known_covariates());

    // After the first `time` layer, the schedule contains both `variate` and `time` layers.
    cfg.block_schedule = vec!["time".into(), "variate".into(), "time".into()];
    assert!(cfg.supports_known_covariates());

    cfg.block_schedule = vec!["variate_time".into(), "time".into()];
    assert!(!cfg.supports_known_covariates());
    cfg.block_schedule = vec!["time_variate".into(), "time".into()];
    assert!(cfg.supports_known_covariates());
    cfg.block_schedule = vec!["time".into(), "time_variate".into()];
    assert!(cfg.supports_known_covariates());
}

#[test]
fn test_config_validation_errors() {
    let mut cfg = TimestralConfig::default_209m();

    // `d_model` is not divisible by `num_heads`.
    cfg.d_model = 100;
    cfg.num_heads = 12;
    assert!(cfg.validate().is_err());

    // The quantile knots omit 0.5.
    cfg = TimestralConfig::default_209m();
    cfg.quantile_knots = vec![0.1, 0.9];
    assert!(cfg.validate().is_err());

    // The quantile knots are out of order.
    cfg = TimestralConfig::default_209m();
    cfg.quantile_knots = vec![0.5, 0.2];
    assert!(cfg.validate().is_err());

    // stabilize_with_global = true
    cfg = TimestralConfig::default_209m();
    cfg.stabilize_with_global = true;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_serde_defaults() {
    // Minimal JSON with no optional fields.
    let json_str = r#"{
        "patch_size": 32,
        "d_model": 64,
        "num_heads": 2,
        "mlp_hidden_dim": 128,
        "block_schedule": ["time", "variate"],
        "quantile_knots": [0.1, 0.5, 0.9],
        "block_size": 16,
        "context_length": 1024
    }"#;

    let cfg = TimestralConfig::from_json_str(json_str).expect("minimal valid config should load");
    // The Python dataclass defaults `quantile_width_factor` to 2.0.
    assert_eq!(cfg.quantile_width_factor, 2.0);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.rope_partial_factor, 0.5);
    assert_eq!(cfg.minimum_scale, 1e-6);
    assert!(cfg.use_asinh);
    assert!(cfg.covariate_enabled);
}

#[test]
fn test_config_rejects_unknown_fields() {
    // The JSON contains an unknown field, `unknown_extra_field`.
    let json_str = r#"{
        "patch_size": 32,
        "d_model": 64,
        "num_heads": 2,
        "mlp_hidden_dim": 128,
        "block_schedule": ["time"],
        "quantile_knots": [0.1, 0.5, 0.9],
        "block_size": 16,
        "context_length": 1024,
        "unknown_extra_field": 123
    }"#;

    let res = TimestralConfig::from_json_str(json_str);
    assert!(res.is_err(), "should reject unknown fields in config");
}

#[test]
fn test_config_from_json_validates_automatically() {
    // The JSON is well formed, but `d_model = 65` is not divisible by `num_heads = 2`.
    let json_str = r#"{
        "patch_size": 32,
        "d_model": 65,
        "num_heads": 2,
        "mlp_hidden_dim": 128,
        "block_schedule": ["time"],
        "quantile_knots": [0.1, 0.5, 0.9],
        "block_size": 16,
        "context_length": 1024
    }"#;

    let res = TimestralConfig::from_json_str(json_str);
    assert!(
        res.is_err(),
        "from_json_str must call validate and reject invalid config"
    );
}
