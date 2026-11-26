//! Database/CSV loading, time-series preparation, and model inference.
//!
//! All entry points run synchronously on the calling thread. Async callers
//! should offload them to a blocking worker if the executor must stay responsive.

#[cfg(feature = "csv")]
mod csv_input;
mod database;
mod series;
mod series_config;

use std::{num::NonZeroUsize, path::PathBuf, time::Duration as StdDuration};

use anyhow::{Context, Result, ensure};
use burn::tensor::Device;
use burn_timestral::{ForecastOptions, TimestralForecaster};
use chrono::{DateTime, Utc};
use clap::ValueEnum;
use series::{RasterOptions, RasterizedData, RawSeries};
use series_config::{SeriesSpec, Source, resolve_specs};

pub use burn_timestral::{Forecast, TimestralVariant};
pub use database::QueryWindowSuccess;
pub use series::{Aggregation, FillMethod, GridSpec, SeriesData, TimePoint, parse_duration};
pub use series_config::{SeriesArg, SourceOptions, parse_series_arg};

/// Length of the forecast horizon.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HorizonSpec {
    /// A positive number of grid steps.
    Steps(NonZeroUsize),
    /// A duration that must be a positive multiple of the timestamp grid interval.
    Duration(StdDuration),
}

impl HorizonSpec {
    fn resolve(self, grid: Option<&GridSpec>) -> Result<usize> {
        match self {
            Self::Steps(steps) => Ok(steps.get()),
            Self::Duration(duration) => grid
                .context("A duration HORIZON requires timestamped series; use a step count for row-indexed CSV")?
                .steps_for_duration(duration),
        }
    }
}

/// Compute device for model inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ForecastBackend {
    /// CPU inference using the Flex backend.
    Cpu,
    #[cfg(feature = "wgpu")]
    /// GPU inference using WGPU.
    Wgpu,
}

/// Input and inference settings for one end-to-end forecast run.
#[derive(Clone, Debug)]
pub struct ForecastConfig {
    /// Default CSV input, if any.
    pub input: Option<PathBuf>,
    /// Length of the forecast horizon.
    pub horizon: HorizonSpec,
    /// Whether the default CSV starts with a header row.
    pub headers: bool,
    /// Optional directory containing model weights.
    pub model_dir: Option<PathBuf>,
    /// Model variant; `None` selects the model library's default.
    pub variant: Option<TimestralVariant>,
    /// Inference backend; `None` selects the default device.
    pub backend: Option<ForecastBackend>,
    /// Default aggregation for timestamped inputs.
    pub aggregation: Aggregation,
    /// Target selections, in output order; empty selects default CSV columns.
    pub values: Vec<SeriesArg>,
    /// Historical covariate selections, in model input order.
    pub covariates: Vec<SeriesArg>,
    /// Shared source, time window, and gap-filling settings.
    pub source: SourceOptions,
}

/// Diagnostic for a completed database query window.
#[derive(Debug)]
pub struct QueryProgress {
    /// Label of the input source.
    pub source: String,
    /// Bounds and elapsed time of the successful query.
    pub window: QueryWindowSuccess,
}

/// Model inputs and diagnostics returned by data preparation.
#[derive(Debug)]
pub struct PreparedInputs {
    /// Target history in variate-major order.
    pub targets: SeriesData,
    /// Optional covariate history aligned to the target context grid.
    pub past_covariates: Option<SeriesData>,
    /// Names in target variate order.
    pub target_names: Vec<String>,
    /// Whether each target history value was imputed.
    pub target_filled: Vec<bool>,
    /// Observed targets in the backtest horizon. Empty for live forecasts.
    pub target_actual: Vec<Option<f32>>,
    /// Grid for target and covariate history.
    pub context_grid: GridSpec,
    /// Forecast horizon grid (also the actuals grid for backtests).
    pub actual_grid: GridSpec,
    /// Number of forecast steps per target.
    pub horizon_steps: usize,
    /// Successfully loaded database query windows.
    pub query_progress: Vec<QueryProgress>,
    /// Nonfatal preparation diagnostics.
    pub warnings: Vec<String>,
    /// Requested and effective end times when a backtest window was shortened.
    pub adjusted_window: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

/// Model forecast together with its prepared inputs and diagnostics.
#[derive(Debug)]
pub struct ForecastOutput {
    /// Inputs used for inference.
    pub inputs: PreparedInputs,
    /// Model predictions for the forecast horizon.
    pub forecast: Forecast,
}

/// Runs a backtest, holding out the last horizon steps as observed actuals.
pub fn forecast(config: ForecastConfig) -> Result<ForecastOutput> {
    let inputs = prepare_inputs(&config)?;
    forecast_prepared(config, inputs)
}

/// Forecasts from timestamped history without loading observations in the forecast horizon.
/// `config.source.to` is the exclusive end of history and the first forecast timestamp.
/// `config.horizon` determines the length of the forecast grid independently of the history.
pub fn forecast_live(config: ForecastConfig) -> Result<ForecastOutput> {
    let inputs = prepare_for_mode(&config, ForecastMode::Live)?;
    forecast_prepared(config, inputs)
}

/// Loads and prepares backtest inputs without running model inference.
pub fn prepare_inputs(config: &ForecastConfig) -> Result<PreparedInputs> {
    prepare_for_mode(config, ForecastMode::Backtest)
}

/// Runs inference for previously prepared inputs without reloading source data.
/// Only the model directory, variant, and backend are read from `config`.
pub fn forecast_prepared(config: ForecastConfig, inputs: PreparedInputs) -> Result<ForecastOutput> {
    let has_missing_targets = inputs.targets.values.iter().any(|value| !value.is_finite());
    let mut forecast_options = ForecastOptions::default();
    let has_past_covariates = inputs.past_covariates.is_some();
    if let Some(past) = &inputs.past_covariates {
        forecast_options =
            forecast_options.with_past_covariates(past.values.clone(), past.num_variates);
    }
    if has_missing_targets {
        forecast_options = forecast_options.with_observation_mask(
            inputs
                .targets
                .values
                .iter()
                .map(|value| value.is_finite())
                .collect(),
        );
    }
    let forecast_options = (has_past_covariates || has_missing_targets).then_some(forecast_options);

    let device = match config.backend {
        None => Device::default(),
        Some(ForecastBackend::Cpu) => Device::flex(),
        #[cfg(feature = "wgpu")]
        Some(ForecastBackend::Wgpu) => Device::wgpu_options()
            .init()
            .context("Failed to initialize the WGPU backend")?,
    };
    let forecaster = TimestralForecaster::load(
        config.variant.unwrap_or_default(),
        config.model_dir.as_deref(),
        &device,
    )?;
    let forecast = forecaster.forecast(
        &inputs.targets.values,
        1,
        inputs.targets.num_variates,
        inputs.targets.context_length,
        inputs.horizon_steps,
        forecast_options.as_ref(),
    )?;

    Ok(ForecastOutput { inputs, forecast })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ForecastMode {
    Backtest,
    Live,
}

fn prepare_for_mode(config: &ForecastConfig, mode: ForecastMode) -> Result<PreparedInputs> {
    let database_config = config
        .source
        .config_path
        .as_deref()
        .map(database::load_config)
        .transpose()?;
    prepare_inputs_with_config(config, database_config.as_ref(), mode)
}

fn prepare_inputs_with_config(
    config: &ForecastConfig,
    database_config: Option<&database::ConfigFile>,
    mode: ForecastMode,
) -> Result<PreparedInputs> {
    let target_specs = resolve_specs(&config.values, database_config)?;
    let covariate_specs = resolve_specs(&config.covariates, database_config)?;
    let default_csv = config.input.as_deref().map(|path| {
        Source::default_csv(
            path,
            config.headers,
            target_specs
                .iter()
                .find_map(|(_, spec)| spec.positional_csv_time_column(path))
                .or(config.source.time_column.as_deref()),
        )
    });

    let resolve_sources = |specs: Vec<_>, role| {
        specs
            .into_iter()
            .map(|(name, spec)| {
                let source =
                    SeriesSpec::source(spec, default_csv.as_ref(), &config.source, database_config)
                        .with_context(|| format!("Invalid {role} {name:?}"))?;
                Ok((Some((name, spec)), source))
            })
            .collect::<Result<Vec<_>>>()
    };
    let mut target_sources = resolve_sources(target_specs, "target series")?;
    let covariates = resolve_sources(covariate_specs, "covariate")?;
    if target_sources.is_empty() {
        target_sources.push((
            None,
            default_csv.context("Database input requires --value")?,
        ));
    }

    let timestamped = target_sources[0].1.is_timestamped();
    let all_sources = || {
        target_sources
            .iter()
            .chain(&covariates)
            .map(|(_, source)| source)
    };
    ensure!(
        all_sources().all(|source| source.is_timestamped() == timestamped),
        "Cannot mix absolute timestamps and row indices across target and covariate series"
    );
    ensure!(
        timestamped || mode == ForecastMode::Backtest,
        "Live forecasts require timestamped series"
    );
    let mut adjusted_window = None;
    let grid = if timestamped {
        let missing = config.source.missing_window_options();
        ensure!(
            missing.is_empty(),
            "Timestamped series require {}",
            missing.join(", ")
        );
        let from = config.source.from.as_deref().expect("validated window");
        let to = config.source.to.as_deref().expect("validated window");
        let step = config.source.step.as_deref().expect("validated window");
        let (grid, effective_to) = GridSpec::from_window(from, to, step)?;
        let requested_to = DateTime::parse_from_rfc3339(to)?.with_timezone(&Utc);
        if effective_to != requested_to {
            ensure!(
                mode == ForecastMode::Backtest,
                "Live forecast start (--to) must align with --from and --step"
            );
            adjusted_window = Some((requested_to, effective_to));
        }
        Some(grid)
    } else {
        ensure!(
            config.source.missing_window_options().len() == 3,
            "--from, --to, and --step require timestamped series"
        );
        None
    };
    ensure!(
        config.source.query_interval.is_none()
            || all_sources().any(|source| matches!(source, Source::Database(_))),
        "--query-interval requires a database target or covariate"
    );

    let horizon = config.horizon.resolve(grid.as_ref())?;
    let grids = grid
        .as_ref()
        .map(|grid| match mode {
            ForecastMode::Backtest => grid.split_tail(horizon),
            ForecastMode::Live => Ok((grid.clone(), grid.following(horizon)?)),
        })
        .transpose()?;
    let mut query_progress = Vec::new();
    let mut warnings = Vec::new();
    let mut target_raw = Vec::new();
    for (selection, source) in target_sources {
        let excluded = if selection.is_none() {
            covariates
                .iter()
                .filter(|(_, cov_source)| source.shares_csv(cov_source))
                .filter_map(|(selection, _)| {
                    selection.map(|(_, spec)| spec.effective_value_column(&config.source))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let (raw, raster) = load_series(
            source,
            selection,
            &excluded,
            grid.as_ref(),
            config,
            &mut query_progress,
        )?;
        target_raw.push((selection.map(|(_, spec)| spec), raw, raster));
    }
    let (context_grid, actual_grid) = match grids {
        Some(grids) => grids,
        None => {
            let rows = target_raw[0].1[0].observations.len();
            ensure!(
                target_raw
                    .iter()
                    .flat_map(|(_, raw, _)| raw)
                    .all(|series| series.observations.len() == rows),
                "Row-indexed target series must have the same number of rows"
            );
            GridSpec::index(rows)?.split_tail(horizon)?
        }
    };
    let mut targets = SeriesData {
        values: Vec::new(),
        num_variates: 0,
        context_length: context_grid.len(),
    };
    let mut target_names = Vec::new();
    let mut target_filled = Vec::new();
    let mut target_actual = Vec::new();
    for (spec, raw, raster) in target_raw {
        let prepared = prepare_context(
            &raw,
            spec,
            &context_grid,
            &raster,
            &config.source,
            &mut warnings,
        )?;
        targets.values.extend(prepared.data.values);
        target_filled.extend(prepared.filled);
        if mode == ForecastMode::Backtest {
            target_actual.extend(series::rasterize_optional(&raw, &actual_grid, &raster)?);
        }
        targets.num_variates += prepared.data.num_variates;
        target_names.extend(raw.into_iter().map(|raw| raw.name));
    }
    let mut past = SeriesData {
        values: Vec::new(),
        num_variates: 0,
        context_length: targets.context_length,
    };
    for (selection, source) in covariates {
        let (raw, raster) = load_series(
            source,
            selection,
            &[],
            Some(&context_grid),
            config,
            &mut query_progress,
        )?;
        let spec = selection.map(|(_, spec)| spec);
        let prepared = prepare_context(
            &raw,
            spec,
            &context_grid,
            &raster,
            &config.source,
            &mut warnings,
        )?;
        past.values.extend(prepared.data.values);
        past.num_variates += prepared.data.num_variates;
    }

    Ok(PreparedInputs {
        targets,
        past_covariates: (past.num_variates > 0).then_some(past),
        target_names,
        target_filled,
        target_actual,
        context_grid,
        actual_grid,
        horizon_steps: horizon,
        query_progress,
        warnings,
        adjusted_window,
    })
}

fn load_series(
    source: Source<'_>,
    selection: Option<(&str, &SeriesSpec)>,
    excluded: &[&str],
    grid: Option<&GridSpec>,
    config: &ForecastConfig,
    query_progress: &mut Vec<QueryProgress>,
) -> Result<(Vec<RawSeries>, RasterOptions)> {
    let (label, spec) = selection.map_or(("CSV targets", None), |(name, spec)| (name, Some(spec)));
    let (raster, window) = RasterOptions::for_input(
        grid,
        spec.and_then(|spec| spec.aggregation)
            .unwrap_or(config.aggregation),
        spec.and_then(|spec| spec.max_hold.as_deref()),
    )
    .with_context(|| format!("Invalid raster options for {label:?}"))?;
    let columns = spec
        .map(|spec| {
            spec.effective_value_column(&config.source)
                .map(str::to_owned)
        })
        .transpose()?
        .into_iter()
        .collect::<Vec<_>>();
    let mut raw = source
        .load(
            &columns,
            excluded,
            window,
            config.source.query_interval,
            &mut |window| {
                query_progress.push(QueryProgress {
                    source: label.to_owned(),
                    window,
                });
                Ok(())
            },
        )
        .with_context(|| format!("Failed to load {label:?}"))?;
    ensure!(!raw.is_empty(), "No series were selected for {label:?}");
    if spec.is_some() {
        ensure!(raw.len() == 1, "Expected exactly one series for {label:?}");
        raw[0].name = label.to_owned();
    }
    Ok((raw, raster))
}

fn prepare_context(
    raw: &[RawSeries],
    spec: Option<&SeriesSpec>,
    grid: &GridSpec,
    raster: &RasterOptions,
    options: &SourceOptions,
    warnings: &mut Vec<String>,
) -> Result<RasterizedData> {
    let fill = spec
        .and_then(|spec| spec.fill)
        .or(options.fill)
        .unwrap_or_default();
    let limit = spec
        .and_then(|spec| spec.max_fill_gap.as_deref())
        .or(options.max_fill_gap.as_deref())
        .map(|value| series::max_fill_gap_steps(value, grid))
        .transpose()?;
    let fills = raw.iter().map(|raw| (raw.name.clone(), fill)).collect();
    let limits = raw
        .iter()
        .filter_map(|raw| limit.map(|limit| (raw.name.clone(), limit)))
        .collect();
    let prepared = series::rasterize_with_gap_limits(raw, grid, &fills, &limits, raster)?;
    for gaps in &prepared.unfilled_gaps {
        let reasons = gaps
            .reasons
            .iter()
            .map(|reason| format!("{} ({} gaps)", reason.reason, reason.gap_count))
            .collect::<Vec<_>>()
            .join(", ");
        warnings.push(format!(
            "series {:?} has {} unfilled gaps ({} steps total); longest {} ({} steps); reasons: {}",
            gaps.series_name,
            gaps.gap_count,
            gaps.total_steps,
            gaps.longest_range,
            gaps.longest_steps,
            reasons
        ));
    }
    Ok(prepared)
}

#[cfg(all(test, feature = "csv"))]
mod tests {
    use super::*;

    fn prepare_live_inputs(config: &ForecastConfig) -> Result<PreparedInputs> {
        prepare_for_mode(config, ForecastMode::Live)
    }

    #[test]
    fn csv_input_adapter_preserves_column_major_context_order() {
        let raw =
            csv_input::load(std::io::Cursor::new("10,1\n20,2\n30,3\n"), false, None, &[]).unwrap();
        let grid = GridSpec::index(3).unwrap();
        let options = RasterOptions::for_grid(&grid, Aggregation::TimeWeightedMean, None).unwrap();
        let context = series::rasterize(&raw, &grid, &Default::default(), &options).unwrap();

        assert_eq!(context.num_variates, 2);
        assert_eq!(context.context_length, 3);
        assert_eq!(context.values, vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0]);
    }

    const START: &str = "2026-01-02T00:00:00Z";
    const CUTOFF: &str = "2026-01-02T00:10:00Z";

    fn config() -> ForecastConfig {
        ForecastConfig {
            input: Some(
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/live_inputs.csv"),
            ),
            horizon: HorizonSpec::Duration(StdDuration::from_secs(15 * 60)),
            headers: true,
            model_dir: None,
            variant: None,
            backend: None,
            aggregation: Aggregation::Last,
            values: vec![],
            covariates: vec![parse_series_arg("temperature={value_column='temperature'}").unwrap()],
            source: SourceOptions {
                config_path: None,
                connection: None,
                time_column: Some("time".into()),
                value_column: None,
                from: Some(START.into()),
                to: Some(CUTOFF.into()),
                step: Some("5m".into()),
                query_interval: None,
                fill: None,
                max_fill_gap: None,
            },
        }
    }

    #[test]
    fn live_uses_only_history_and_forecasts_from_the_exclusive_cutoff() {
        let inputs = prepare_live_inputs(&config()).unwrap();
        assert_eq!(inputs.targets.context_length, 2);
        assert_eq!(inputs.target_names, ["load"]);
        assert_eq!(inputs.targets.values, [10.0, 20.0]);
        assert_eq!(inputs.past_covariates.unwrap().values, [1.0, 2.0]);
        assert!(inputs.target_actual.is_empty());
        assert!(inputs.adjusted_window.is_none());
        assert_eq!(inputs.horizon_steps, 3);
        assert_eq!(inputs.actual_grid.len(), 3);
        let (_, context_end) = inputs.context_grid.time_bounds().unwrap();
        let (forecast_start, forecast_end) = inputs.actual_grid.time_bounds().unwrap();
        assert_eq!(forecast_start, context_end);
        assert_eq!(forecast_start, CUTOFF.parse::<DateTime<Utc>>().unwrap());
        assert_eq!(
            forecast_end,
            "2026-01-02T00:25:00Z".parse::<DateTime<Utc>>().unwrap()
        );
    }

    #[test]
    fn backtest_still_reserves_the_tail_for_actuals() {
        let mut config = config();
        config.source.to = Some("2026-01-02T00:20:00Z".into());
        config.horizon = HorizonSpec::Steps(NonZeroUsize::new(2).unwrap());
        let inputs = prepare_inputs(&config).unwrap();
        assert_eq!(inputs.targets.values, [10.0, 20.0]);
        assert_eq!(inputs.past_covariates.unwrap().values, [1.0, 2.0]);
        assert_eq!(inputs.target_actual, [Some(30.0), Some(40.0)]);
    }

    #[test]
    fn live_rejects_invalid_windows_and_horizons() {
        let mut config = config();
        config.source.to = Some("2026-01-02T00:11:00Z".into());
        assert!(
            prepare_live_inputs(&config)
                .unwrap_err()
                .to_string()
                .contains("must align")
        );
        config.source.to = Some(START.into());
        assert!(prepare_live_inputs(&config).is_err());
        config.source.to = Some(CUTOFF.into());
        config.horizon = HorizonSpec::Steps(NonZeroUsize::MAX);
        assert!(
            prepare_live_inputs(&config)
                .unwrap_err()
                .to_string()
                .contains("Grid is too large")
        );
        config.source.time_column = None;
        assert!(
            prepare_live_inputs(&config)
                .unwrap_err()
                .to_string()
                .contains("require timestamped")
        );
    }
}
