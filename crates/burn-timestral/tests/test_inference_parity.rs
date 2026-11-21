//! End-to-end numerical parity between the Rust inference pipeline (`TimestralForecaster`)
//! and deterministic golden fixtures from the official Python reference (`mistralai/timestral`).
//!
//! Only the fixture metadata and integrity check runs by default without a checkpoint.
//! See `fixtures/README.md` for instructions on running the model tests.

use burn_timestral::{ForecastOptions, TimestralForecaster, TimestralVariant};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct ParityFixtures {
    metadata: FixtureMetadata,
    scenarios: Vec<ParityScenario>,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    generator: String,
    checkpoint: String,
    checkpoint_revision: String,
    model_safetensors_sha256: String,
    config_json_sha256: String,
    timestral_git_rev: String,
    timestral_git_dirty: bool,
    timestral_source_url: String,
    torch_version: String,
    per_shot_horizon_patches: usize,
    patch_size: usize,
    quantile_knots: Vec<f32>,
    num_quantiles: usize,
    median_index: usize,
}

#[derive(Debug, Deserialize)]
struct CovariateValues {
    num_covariates: usize,
    values: Vec<Option<f32>>,
}

#[derive(Debug, Deserialize)]
struct CategoricalCovariateCodes {
    num_covariates: usize,
    codes: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct ParityScenario {
    name: String,
    description: String,
    batch: usize,
    variates: usize,
    context_length: usize,
    prediction_length: usize,
    max_context: Option<usize>,
    context: Vec<Option<f32>>,
    observation_mask: Option<Vec<bool>>,
    past_covariates: Option<CovariateValues>,
    known_covariates: Option<CovariateValues>,
    categorical_past_covariates: Option<CategoricalCovariateCodes>,
    categorical_known_covariates: Option<CategoricalCovariateCodes>,
    symmetric_averaging: bool,
    clip_nonnegative: bool,
    expected_shape: Vec<usize>,
    expected_quantiles: Vec<f32>,
}

const COMPACT_SCENARIOS: &[&str] = &[
    "univariate_unaligned_padding",
    "multi_block_autoregressive",
    "numeric_covariates_with_nan_gaps",
];

struct Fixture {
    variant: TimestralVariant,
    file: &'static str,
    scenarios: &'static [&'static str],
}

const M5: Fixture = Fixture {
    variant: TimestralVariant::M5,
    file: "parity_scenarios_5m.json",
    scenarios: COMPACT_SCENARIOS,
};
const M28: Fixture = Fixture {
    variant: TimestralVariant::M28,
    file: "parity_scenarios_28m.json",
    scenarios: COMPACT_SCENARIOS,
};
const M123: Fixture = Fixture {
    variant: TimestralVariant::M123,
    file: "parity_scenarios_123m.json",
    scenarios: COMPACT_SCENARIOS,
};
const M209: Fixture = Fixture {
    variant: TimestralVariant::M209,
    file: "parity_scenarios.json",
    scenarios: &[
        "univariate_short_horizon",
        "univariate_single_block",
        "univariate_unaligned_padding",
        "univariate_nan_gaps",
        "multivariate_3variates",
        "batched_multivariate",
        "multi_block_autoregressive",
        "numeric_past_and_known_covariates",
        "binary_indicator_known_covariate",
        "categorical_known_covariates_target_encoded",
        "categorical_past_and_known_covariates",
        "clip_nonnegative_clamping",
        "symmetric_averaging_disabled",
        "max_context_truncation_sliding_categorical",
        "explicit_observation_mask",
        "numeric_covariates_with_nan_gaps",
        "leading_and_all_nan_targets",
    ],
};

fn load_fixtures(fixture: &Fixture) -> ParityFixtures {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture.file);
    let file = File::open(&path).unwrap_or_else(|e| panic!("Opening {}: {e}", path.display()));
    serde_json::from_reader(BufReader::new(file))
        .unwrap_or_else(|e| panic!("Parsing {}: {e}", path.display()))
}

fn verify_scenario(default_forecaster: &TimestralForecaster, scenario: &ParityScenario) {
    const ATOL: f32 = 1e-4;
    const RTOL: f32 = 1e-4;
    let num_quantiles = scenario.expected_shape[3];

    let context: Vec<f32> = scenario
        .context
        .iter()
        .map(|opt| opt.unwrap_or(f32::NAN))
        .collect();

    let mut opt = ForecastOptions::default()
        .with_symmetric_averaging(scenario.symmetric_averaging)
        .with_clip_nonnegative(scenario.clip_nonnegative);

    if let Some(obs) = &scenario.observation_mask {
        opt = opt.with_observation_mask(obs.clone());
    }

    if let Some(past) = &scenario.past_covariates {
        let values: Vec<f32> = past
            .values
            .iter()
            .map(|opt| opt.unwrap_or(f32::NAN))
            .collect();
        opt = opt.with_past_covariates(values, past.num_covariates);
    }

    if let Some(known) = &scenario.known_covariates {
        let values: Vec<f32> = known
            .values
            .iter()
            .map(|opt| opt.unwrap_or(f32::NAN))
            .collect();
        opt = opt.with_known_covariates(values, known.num_covariates);
    }

    if let Some(cat_past) = &scenario.categorical_past_covariates {
        let codes: Vec<f64> = cat_past
            .codes
            .iter()
            .map(|c| c.unwrap_or(f64::NAN))
            .collect();
        opt = opt.with_categorical_past_covariates(codes, cat_past.num_covariates);
    }

    if let Some(cat_known) = &scenario.categorical_known_covariates {
        let codes: Vec<f64> = cat_known
            .codes
            .iter()
            .map(|c| c.unwrap_or(f64::NAN))
            .collect();
        opt = opt.with_categorical_known_covariates(codes, cat_known.num_covariates);
    }

    // Reuse the loaded model even when a scenario overrides max_context.
    let custom_forecaster;
    let forecaster = match scenario.max_context {
        Some(mc) if mc != default_forecaster.max_context() => {
            custom_forecaster = default_forecaster
                .clone()
                .with_max_context(mc)
                .expect("Failed to apply context limit");
            &custom_forecaster
        }
        _ => default_forecaster,
    };

    let forecast = forecaster
        .forecast(
            &context,
            scenario.batch,
            scenario.variates,
            scenario.context_length,
            scenario.prediction_length,
            Some(&opt),
        )
        .unwrap_or_else(|e| panic!("Scenario '{}' execution failed: {e}", scenario.name));

    // Validate output shape and ensure the element-wise comparison covers the full buffer.
    let actual_shape = forecast.shape();
    let expected_shape = (
        scenario.expected_shape[0],
        scenario.expected_shape[1],
        scenario.expected_shape[2],
        scenario.expected_shape[3],
    );
    assert_eq!(
        actual_shape, expected_shape,
        "Scenario '{}' shape mismatch: actual={:?}, expected={:?}",
        scenario.name, actual_shape, expected_shape
    );
    assert_eq!(
        forecast.predictions().len(),
        scenario.expected_quantiles.len(),
        "Scenario '{}' quantile count mismatch",
        scenario.name
    );

    // Numerical parity comparison across all (B, V, H, Q) quantile elements
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;

    for (elem_idx, (&actual, &expected)) in forecast
        .predictions()
        .iter()
        .zip(scenario.expected_quantiles.iter())
        .enumerate()
    {
        assert!(
            actual.is_finite(),
            "Scenario '{}' produced non-finite quantile at index {elem_idx}: {actual}",
            scenario.name
        );

        let abs_diff = (actual - expected).abs();
        let tol = ATOL + RTOL * expected.abs();

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(abs_diff / expected.abs().max(1e-5));

        if abs_diff > tol {
            let q = elem_idx % num_quantiles;
            let rem = elem_idx / num_quantiles;
            let h = rem % scenario.prediction_length;
            let rem2 = rem / scenario.prediction_length;
            let v = rem2 % scenario.variates;
            let b = rem2 / scenario.variates;

            panic!(
                "Scenario '{}' parity violation at (b={b}, v={v}, h={h}, q={q}): actual={actual:.7}, expected={expected:.7}, diff={abs_diff:.6e} > tol={tol:.6e}",
                scenario.name
            );
        }
    }

    // Monotonicity check across quantiles
    for (row, quantiles) in forecast
        .predictions()
        .chunks_exact(num_quantiles)
        .enumerate()
    {
        let h = row % scenario.prediction_length;
        let series = row / scenario.prediction_length;
        let v = series % scenario.variates;
        let b = series / scenario.variates;
        for (q, pair) in quantiles.windows(2).enumerate() {
            let (q_curr, q_next) = (pair[0], pair[1]);
            assert!(
                q_next >= q_curr - 1e-5,
                "Scenario '{}' quantile monotonicity violation at (b={b}, v={v}, h={h}): Q[{q}]={q_curr:.7} > Q[{}]={q_next:.7}",
                scenario.name,
                q + 1
            );
        }
    }

    println!(
        "Scenario '{}': PASSED (max_abs_diff={:.2e}, max_rel_diff={:.2e})",
        scenario.name, max_abs_diff, max_rel_diff
    );
}

fn is_lower_hex_of_len(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_fixtures(fixture: &Fixture, fixtures: &ParityFixtures) {
    let m = &fixtures.metadata;
    assert_eq!(m.generator, "tools/generate_parity_fixtures.py");
    assert_eq!(m.checkpoint, fixture.variant.repo_id());
    assert_eq!(m.checkpoint_revision, fixture.variant.revision());
    assert_eq!(m.per_shot_horizon_patches, 1);
    assert_eq!(m.patch_size, 32);
    assert_eq!(m.num_quantiles, 21);
    assert_eq!(m.median_index, 10);
    assert_eq!(
        m.quantile_knots,
        [
            0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
            0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
        ],
        "{}: unexpected quantile knots",
        fixture.file
    );
    assert!(!m.torch_version.is_empty());
    assert!(is_lower_hex_of_len(&m.timestral_git_rev, 40));
    assert!(!m.timestral_git_dirty);
    assert_eq!(
        m.timestral_source_url,
        "https://github.com/mistralai/timestral"
    );
    assert!(is_lower_hex_of_len(&m.model_safetensors_sha256, 64));
    assert!(is_lower_hex_of_len(&m.config_json_sha256, 64));

    let names: Vec<_> = fixtures.scenarios.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names.as_slice(),
        fixture.scenarios,
        "{}: scenario coverage",
        fixture.file
    );

    for s in &fixtures.scenarios {
        assert!(
            !s.description.is_empty(),
            "Scenario '{}' has no description",
            s.name
        );
        let shape = [s.batch, s.variates, s.prediction_length, m.num_quantiles];
        assert!(
            s.context_length > 0 && !shape.contains(&0),
            "Scenario '{}' has a zero dimension",
            s.name
        );
        assert_eq!(
            s.expected_shape, shape,
            "Scenario '{}' shape must match (B, V, H, Q)",
            s.name
        );
        assert_eq!(
            s.expected_quantiles.len(),
            shape.iter().product::<usize>(),
            "Scenario '{}' expected_quantiles count mismatch",
            s.name
        );
        assert!(
            s.expected_quantiles.iter().all(|x| x.is_finite()),
            "Scenario '{}' contains non-finite expected quantile",
            s.name
        );
        let context_count = s.batch * s.variates * s.context_length;
        assert_eq!(
            s.context.len(),
            context_count,
            "Scenario '{}' context count mismatch",
            s.name
        );
        if let Some(mask) = &s.observation_mask {
            assert_eq!(
                mask.len(),
                context_count,
                "Scenario '{}' mask count mismatch",
                s.name
            );
        }
    }
}

#[test]
fn test_parity_metadata_and_integrity() {
    for fixture in [&M5, &M28, &M123, &M209] {
        validate_fixtures(fixture, &load_fixtures(fixture));
    }
}

fn verify_variant(fixture: Fixture) {
    let fixtures = load_fixtures(&fixture);
    validate_fixtures(&fixture, &fixtures);
    let model_dir = std::env::var_os("TIMESTRAL_MODEL_DIR").map(PathBuf::from);
    let device = burn::tensor::Device::default();
    let forecaster = TimestralForecaster::load(fixture.variant, model_dir.as_deref(), &device)
        .unwrap_or_else(|e| {
            panic!(
                "Loading {} (TIMESTRAL_MODEL_DIR={model_dir:?}): {e}",
                fixture.variant.repo_id()
            )
        });
    assert_eq!(forecaster.num_quantiles(), fixtures.metadata.num_quantiles);
    assert_eq!(
        forecaster.quantile_levels(),
        fixtures.metadata.quantile_knots.as_slice()
    );
    for scenario in &fixtures.scenarios {
        verify_scenario(&forecaster, scenario);
    }
}

#[test]
#[ignore = "requires official Timestral-5M checkpoint (Hub or TIMESTRAL_MODEL_DIR)"]
fn test_parity_5m() {
    verify_variant(M5);
}

#[test]
#[ignore = "requires official Timestral-28M checkpoint (Hub or TIMESTRAL_MODEL_DIR)"]
fn test_parity_28m() {
    verify_variant(M28);
}

#[test]
#[ignore = "requires official Timestral-123M checkpoint (Hub or TIMESTRAL_MODEL_DIR)"]
fn test_parity_123m() {
    verify_variant(M123);
}

#[test]
#[ignore = "requires official Timestral-209M checkpoint (Hub or TIMESTRAL_MODEL_DIR)"]
fn test_parity_209m() {
    verify_variant(M209);
}
