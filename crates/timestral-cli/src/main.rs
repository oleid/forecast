mod plot;

use anyhow::{Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser, ValueEnum, error::ErrorKind};
use forecast_pipeline::{
    self as pipeline, Aggregation, ForecastBackend as Backend, ForecastConfig, GridSpec,
    HorizonSpec, PreparedInputs, QueryWindowSuccess, SeriesArg, SourceOptions, TimestralVariant,
    parse_duration, parse_series_arg,
};
use std::{
    ffi::OsString,
    io::{self, Write},
    num::NonZeroUsize,
    path::PathBuf,
};

fn main() -> Result<()> {
    let options =
        CliOptions::try_parse_args(std::env::args_os()).unwrap_or_else(|error| error.exit());
    run(options)
}

fn run(options: CliOptions) -> Result<()> {
    let config = pipeline_config(&options);
    let inputs = pipeline::prepare_inputs(&config)?;
    log_prepared_inputs(&inputs, &mut io::stderr().lock())?;
    writeln!(
        io::stderr().lock(),
        "INFO: Starting forecast at {} with horizon {} steps",
        forecast_start(
            &options,
            inputs.targets.context_length,
            inputs.horizon_steps
        )?,
        inputs.horizon_steps
    )?;
    let output = pipeline::forecast_prepared(config, inputs)?;

    plot::write_html(
        &options.out_plot,
        &output.inputs.target_names,
        &output.inputs.targets,
        &output.inputs.target_filled,
        &output.inputs.target_actual,
        &output.inputs.context_grid,
        &output.inputs.actual_grid,
        &output.forecast,
        options.plot_days,
    )?;
    writeln!(
        io::stderr().lock(),
        "INFO: Wrote plot to {}",
        options.out_plot.display()
    )?;
    Ok(())
}

fn pipeline_config(options: &CliOptions) -> ForecastConfig {
    ForecastConfig {
        input: options.input.clone(),
        horizon: options.horizon.expect("CLI arguments are validated"),
        headers: options.headers,
        model_dir: options.model_dir.clone(),
        variant: options.variant.map(TimestralVariant::from),
        backend: options.backend,
        aggregation: options.aggregation,
        values: options.values.clone(),
        covariates: options.covariates.clone(),
        source: options.source.clone(),
    }
}

fn forecast_start(
    options: &CliOptions,
    context_length: usize,
    horizon_steps: usize,
) -> Result<String> {
    match (
        options.source.from.as_deref(),
        options.source.to.as_deref(),
        options.source.step.as_deref(),
    ) {
        (Some(from), Some(to), Some(step)) => {
            let (grid, _) = GridSpec::from_window(from, to, step)?;
            let (context, _) = grid.split_tail(horizon_steps)?;
            let (_, cutoff) = context
                .time_bounds()
                .context("A timestamp grid is required for the backtest")?;
            Ok(cutoff.to_rfc3339())
        }
        _ => Ok(format!(
            "row index {context_length} (after {context_length} context steps)"
        )),
    }
}

fn write_query_progress(
    stderr: &mut impl Write,
    source: &str,
    progress: &QueryWindowSuccess,
) -> io::Result<()> {
    writeln!(
        stderr,
        "INFO: Loaded {source} query window [{}, {}) in {:?}",
        progress.from.to_rfc3339(),
        progress.to.to_rfc3339(),
        progress.elapsed
    )
}

fn log_prepared_inputs(inputs: &PreparedInputs, stderr: &mut impl Write) -> io::Result<()> {
    if let Some((requested_to, effective_to)) = inputs.adjusted_window {
        writeln!(
            stderr,
            "INFO: Adjusting end to nearest preceding interval step: {} -> {}",
            requested_to.to_rfc3339(),
            effective_to.to_rfc3339()
        )?;
    }
    for progress in &inputs.query_progress {
        write_query_progress(stderr, &progress.source, &progress.window)?;
    }
    for warning in &inputs.warnings {
        writeln!(stderr, "WARN: {warning}")?;
    }
    Ok(())
}

#[cfg(test)]
fn prepare_inputs(options: &CliOptions, stderr: &mut impl Write) -> Result<PreparedInputs> {
    let inputs = pipeline::prepare_inputs(&pipeline_config(options))?;
    log_prepared_inputs(&inputs, stderr)?;
    Ok(inputs)
}

/// Forecast time series from CSV, InfluxDB 3, or TimescaleDB.
#[derive(Debug, Parser)]
#[command(
    version,
    after_help = r#"CSV input:
  timestral-cli series.csv 24 --headers
  timestral-cli series.csv 24 --headers --time-column timestamp --value-column temperature \
      --value 'temperature={}' \
      --from 2026-01-01T00:00:00Z --to 2026-01-04T00:00:00Z --step 1h

Database example:
  timestral-cli 24 --connection solar-prod --config "$HOME/src/solar-brain/tools/config.toml" \
      --time-column time --value-column value \
      --value 'household={table="W",filters={entity_id="household"}}' \
      --from 2026-01-01T00:00:00Z --to 2026-11-24T00:00:00Z --step 1h

Use --value and --covariate repeatedly with names under [series] in TOML or 'name={...}' inline definitions.
--time-column and --value-column are defaults for all series; explicit columns in a series definition override them.
Without a positional CSV file, provide only HORIZON and a connection in each database series or as --connection.
HORIZON accepts a positive step count or a duration such as 72h; durations require timestamped series and must be a multiple of --step.
Series can override --fill and --max-fill-gap with fill and max_fill_gap. Gaps longer than the limit
are left missing with a warning. Without a fill rule, gaps are never filled and remain masked with a warning;
unfillable gaps also remain masked. The per-series max_hold setting controls raw values before aggregation.
Plots are saved as standalone HTML; use --out-plot to change the path and --plot-days to adjust the initial time view."#
)]
struct CliOptions {
    /// CSV file path; omit for database input.
    input: Option<PathBuf>,
    /// Held-out steps or duration (e.g. 72h; durations require timestamped series).
    #[arg(value_name = "HORIZON", value_parser = parse_horizon)]
    horizon: Option<HorizonSpec>,
    /// Path for the standalone HTML backtest plot.
    #[arg(long = "out-plot", default_value = "timestral_forecast.html")]
    out_plot: PathBuf,
    /// Number of context days shown initially; 0 displays the full timestamped context.
    #[arg(long, default_value_t = 7)]
    plot_days: u64,
    /// Skip the first row of the CSV file as a header.
    #[arg(long)]
    headers: bool,
    /// Directory containing a local checkpoint.
    #[arg(long, conflicts_with = "variant")]
    model_dir: Option<PathBuf>,
    /// Hub model size when no local checkpoint is specified [default: 209M].
    #[arg(long, value_enum, ignore_case = true)]
    variant: Option<Variant>,
    /// Inference backend (WGPU requires the `wgpu` Cargo feature).
    #[arg(long, value_enum, ignore_case = true)]
    backend: Option<Backend>,
    /// Default aggregation for each grid interval.
    #[arg(long, value_enum, default_value = "time-weighted-mean")]
    aggregation: Aggregation,
    /// Target series: a TOML name or name={...} inline definition (repeatable).
    #[arg(long = "value", value_parser = parse_series_arg)]
    values: Vec<SeriesArg>,
    /// Historical covariate: a TOML name or name={...} inline definition (repeatable).
    #[arg(long = "covariate", value_parser = parse_series_arg)]
    covariates: Vec<SeriesArg>,
    #[command(flatten, next_help_heading = "Source and grid options")]
    source: SourceOptions,
}

impl CliOptions {
    fn try_parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Self, clap::Error> {
        let mut command = Self::command();
        let matches = command.try_get_matches_from_mut(args)?;
        let mut options = Self::from_arg_matches(&matches)?;
        if options.source.connection.is_some()
            || (options.horizon.is_none() && !options.values.is_empty())
        {
            if options.headers {
                return Err(command.error(
                    ErrorKind::ArgumentConflict,
                    "--headers can only be used with CSV input",
                ));
            }
            if options.horizon.is_some() {
                return Err(command.error(
                    ErrorKind::ArgumentConflict,
                    "database input accepts only HORIZON; remove the CSV INPUT argument",
                ));
            }
            let Some(input) = options.input.take() else {
                return Err(command.error(
                    ErrorKind::MissingRequiredArgument,
                    "database input requires HORIZON",
                ));
            };
            let missing = options
                .source
                .missing_db_options(!options.values.is_empty());
            if !missing.is_empty() {
                return Err(command.error(
                    ErrorKind::MissingRequiredArgument,
                    format!("database input requires {}", missing.join(", ")),
                ));
            }
            let horizon = input.as_os_str().to_str().ok_or_else(|| {
                command.error(ErrorKind::InvalidUtf8, "HORIZON must be valid UTF-8")
            })?;
            options.horizon = Some(
                parse_horizon(horizon)
                    .map_err(|error| command.error(ErrorKind::ValueValidation, error))?,
            );
        } else {
            if options.input.is_none() || options.horizon.is_none() {
                return Err(command.error(
                    ErrorKind::MissingRequiredArgument,
                    "CSV input requires both INPUT and HORIZON",
                ));
            }
            let missing = options.source.missing_window_options();
            if !missing.is_empty() && missing.len() != 3 {
                return Err(command.error(
                    ErrorKind::ArgumentConflict,
                    "--from, --to, and --step must be provided together",
                ));
            }
            if options.values.is_empty()
                && missing.is_empty()
                && options.source.time_column.is_none()
            {
                return Err(command.error(
                    ErrorKind::ArgumentConflict,
                    "timestamped CSV grid options require --time-column or a --value definition with time_column",
                ));
            }
        }
        Ok(options)
    }

    #[cfg(test)]
    fn is_database(&self) -> bool {
        self.input.is_none()
    }
}

fn parse_horizon(value: &str) -> std::result::Result<HorizonSpec, String> {
    match value.parse::<usize>() {
        Ok(steps) => NonZeroUsize::new(steps)
            .map(HorizonSpec::Steps)
            .ok_or_else(|| "HORIZON must be a positive step count or duration".to_owned()),
        Err(_) => parse_duration(value, "HORIZON")
            .map(HorizonSpec::Duration)
            .map_err(|error| error.to_string()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Variant {
    #[value(name = "5M")]
    M5,
    #[value(name = "28M")]
    M28,
    #[value(name = "123M")]
    M123,
    #[value(name = "209M")]
    M209,
}

impl From<Variant> for TimestralVariant {
    fn from(variant: Variant) -> Self {
        match variant {
            Variant::M5 => Self::M5,
            Variant::M28 => Self::M28,
            Variant::M123 => Self::M123,
            Variant::M209 => Self::M209,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Aggregation, Backend, CliOptions, HorizonSpec, NonZeroUsize, Variant};
    use clap::error::ErrorKind;
    use forecast_pipeline::{FillMethod, TimestralVariant};
    use std::{
        ffi::OsString,
        path::{Path, PathBuf},
    };

    fn parse(arguments: &[&str]) -> Result<CliOptions, clap::Error> {
        CliOptions::try_parse_args(
            std::iter::once("timestral-cli")
                .chain(arguments.iter().copied())
                .map(OsString::from),
        )
    }

    #[test]
    fn prepares_backtest_context_and_covariates_in_selection_order() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/indexed.csv"),
            "2",
            "--headers",
            "--config",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/series.toml"),
            "--covariate",
            "air={value_column='temperature',aggregation='last'}",
            "--covariate",
            "pv",
        ])
        .unwrap();
        assert!(!options.is_database());
        let inputs = super::prepare_inputs(&options, &mut Vec::new()).unwrap();
        assert_eq!(inputs.targets.values, [1.0]);
        assert_eq!(inputs.targets.num_variates, 1);
        let past = inputs.past_covariates.unwrap();
        assert_eq!(past.values, [21.0, 10.0]);
        assert_eq!(past.num_variates, 2);
        assert_eq!(past.context_length, inputs.targets.context_length);
        assert_eq!(inputs.target_names, ["load"]);
        assert_eq!(inputs.target_actual, [Some(2.0), Some(3.0)]);
        assert_eq!(inputs.target_filled, [false]);
    }

    #[test]
    fn prepares_named_and_inline_targets_in_selection_order() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/indexed.csv"),
            "2",
            "--headers",
            "--config",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/series.toml"),
            "--value",
            "pv",
            "--value",
            "house={value_column='load',fill='none'}",
            "--covariate",
            "air={value_column='temperature'}",
        ])
        .unwrap();
        let inputs = super::prepare_inputs(&options, &mut std::io::sink()).unwrap();
        assert_eq!(inputs.target_names, ["pv", "house"]);
        assert_eq!(inputs.targets.values, [10.0, 1.0]);
        assert_eq!(
            inputs.target_actual,
            [None, Some(14.0), Some(2.0), Some(3.0)]
        );
        assert_eq!(inputs.past_covariates.unwrap().values, [21.0]);
    }

    #[test]
    fn selected_timestamped_csv_series_use_global_column_defaults() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/irregular.csv"),
            "1",
            "--headers",
            "--value",
            "target={aggregation='last'}",
            "--time-column",
            "time",
            "--value-column",
            "target",
            "--from",
            "2026-01-02T00:00:00Z",
            "--to",
            "2026-01-02T00:03:00Z",
            "--step",
            "1m",
            "--aggregation",
            "last",
        ])
        .unwrap();
        let inputs = super::prepare_inputs(&options, &mut std::io::sink()).unwrap();

        assert_eq!(inputs.target_names, ["target"]);
        assert_eq!(inputs.targets.values, [6.0, 8.0]);
        assert_eq!(inputs.target_actual, [Some(10.0)]);
    }

    #[test]
    fn global_time_column_keeps_automatic_csv_target_selection() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/irregular.csv"),
            "1",
            "--headers",
            "--time-column",
            "time",
            "--from",
            "2026-01-02T00:00:00Z",
            "--to",
            "2026-01-02T00:03:00Z",
            "--step",
            "1m",
            "--aggregation",
            "last",
        ])
        .unwrap();
        let inputs = super::prepare_inputs(&options, &mut std::io::sink()).unwrap();

        assert_eq!(inputs.target_names, ["target", "pv"]);
        assert_eq!(inputs.targets.values, [6.0, 8.0, 20.0, 40.0]);
        assert_eq!(inputs.target_actual, [Some(10.0), Some(50.0)]);
    }

    #[test]
    fn timestamped_csv_splits_context_from_actual_values_at_the_cutoff() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/irregular.csv"),
            "2m",
            "--headers",
            "--value",
            "target={value_column='target',time_column='time',fill='hold'}",
            "--from",
            "2026-01-02T00:00:00Z",
            "--to",
            "2026-01-02T00:04:00Z",
            "--step",
            "1m",
            "--aggregation",
            "last",
        ])
        .unwrap();
        let inputs = super::prepare_inputs(&options, &mut Vec::new()).unwrap();
        assert_eq!(inputs.targets.values, [6.0, 8.0]);
        assert_eq!(inputs.targets.context_length, 2);
        assert_eq!(inputs.target_names, ["target"]);
        assert_eq!(inputs.target_filled, [false, false]);
        assert_eq!(inputs.target_actual, [Some(10.0), Some(1000.0)]);
    }

    #[test]
    fn rejects_duration_horizon_for_row_indexed_csv_and_non_multiple_durations() {
        let row_indexed = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/indexed.csv"),
            "2h",
            "--headers",
        ])
        .unwrap();
        assert!(
            super::prepare_inputs(&row_indexed, &mut Vec::new())
                .unwrap_err()
                .to_string()
                .contains("requires timestamped series")
        );

        let timestamped = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/irregular.csv"),
            "90s",
            "--headers",
            "--value",
            "target={value_column='target',time_column='time'}",
            "--from",
            "2026-01-02T00:00:00Z",
            "--to",
            "2026-01-02T00:04:00Z",
            "--step",
            "1m",
        ])
        .unwrap();
        assert!(
            super::prepare_inputs(&timestamped, &mut Vec::new())
                .unwrap_err()
                .to_string()
                .contains("exact multiple of --step")
        );
    }

    #[test]
    fn rejects_a_horizon_that_consumes_the_entire_csv_context() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/indexed.csv"),
            "3",
            "--headers",
        ])
        .unwrap();
        let error = super::prepare_inputs(&options, &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("at least one context step"));
    }

    #[test]
    fn rejects_invalid_covariates_before_loading_target_data() {
        for (extra, diagnostic) in [
            (
                vec!["--covariate", "unlisted"],
                "Unknown series \"unlisted\"",
            ),
            (
                vec!["--covariate", "x={value_column='1'}", "--covariate", "x"],
                "Duplicate series \"x\"",
            ),
            (
                vec![
                    "--covariate",
                    "x={value_column='1',csv='other.csv',connection='db'}",
                ],
                "mutually exclusive",
            ),
            (
                vec!["--covariate", "x={value_column='2',time_column='1'}"],
                "Cannot mix absolute timestamps and row indices",
            ),
        ] {
            let options = parse(&[vec!["nonexistent.csv", "2"], extra].concat()).unwrap();
            let error = super::prepare_inputs(&options, &mut Vec::new()).unwrap_err();
            assert!(format!("{error:#}").contains(diagnostic), "{error:#}");
        }
        assert!(parse(&["series.csv", "2", "--covariate", "x={value_colum='1'}"]).is_err());
    }

    #[test]
    fn masks_unfilled_target_and_covariate_gaps_by_default() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/irregular.csv"),
            "1",
            "--headers",
            "--value",
            "target={value_column='target',time_column='time'}",
            "--from",
            "2026-01-02T00:00:00Z",
            "--to",
            "2026-01-02T00:03:00Z",
            "--step",
            "1m",
            "--covariate",
            "solar={value_column='pv'}",
        ])
        .unwrap();
        let inputs = super::prepare_inputs(&options, &mut Vec::new()).unwrap();
        assert!(inputs.targets.values.iter().any(|value| value.is_nan()));
        assert!(
            inputs
                .past_covariates
                .unwrap()
                .values
                .iter()
                .any(|value| value.is_nan())
        );
    }

    #[test]
    fn fill_none_overrides_a_global_fill_method() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/irregular.csv"),
            "1",
            "--headers",
            "--value",
            "target={value_column='target',time_column='time',fill='none'}",
            "--covariate",
            "solar={value_column='pv'}",
            "--from",
            "2026-01-02T00:00:00Z",
            "--to",
            "2026-01-02T00:03:00Z",
            "--step",
            "1m",
            "--fill",
            "hold",
        ])
        .unwrap();
        let inputs = super::prepare_inputs(&options, &mut Vec::new()).unwrap();
        let context_length = inputs.targets.context_length;

        assert!(inputs.targets.values[context_length - 1].is_nan());
        assert!(inputs.past_covariates.unwrap().values[context_length - 1].is_finite());
    }

    #[test]
    fn applies_global_fill_defaults_and_series_gap_limit_overrides() {
        let options = parse(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/irregular.csv"),
            "1",
            "--headers",
            "--value",
            "target={value_column='target',time_column='time',max_fill_gap='1m'}",
            "--value",
            "masked={value_column='pv'}",
            "--covariate",
            "solar={value_column='pv'}",
            "--from",
            "2026-01-02T00:00:00Z",
            "--to",
            "2026-01-02T00:03:00Z",
            "--step",
            "1m",
            "--fill",
            "hold",
            "--max-fill-gap",
            "30s",
        ])
        .unwrap();
        let inputs = super::prepare_inputs(&options, &mut Vec::new()).unwrap();

        assert_eq!(inputs.target_filled, [false, true, false, false]);
        assert!(inputs.targets.values[3].is_nan());
        let past = inputs.past_covariates.unwrap();
        assert!(past.values[1].is_nan());
    }

    #[test]
    fn parses_cli_arguments_and_rejects_invalid_horizons() {
        let options = parse(&["series.csv", "12"]).unwrap();
        assert_eq!(options.input.as_deref(), Some(Path::new("series.csv")));
        assert_eq!(
            options.horizon,
            NonZeroUsize::new(12).map(HorizonSpec::Steps)
        );
        assert!(!options.headers);
        assert!(options.variant.is_none());
        assert_eq!(options.backend, None);
        assert!(options.model_dir.is_none());
        assert_eq!(options.aggregation, Aggregation::TimeWeightedMean);

        assert!(parse(&["series.csv", "12", "--headers"]).unwrap().headers);
        assert_eq!(
            parse(&["series.csv"]).unwrap_err().kind(),
            ErrorKind::MissingRequiredArgument
        );
        for horizon in ["0", "many"] {
            assert_eq!(
                parse(&["series.csv", horizon]).unwrap_err().kind(),
                ErrorKind::ValueValidation
            );
        }
    }

    #[test]
    fn accepts_timestamped_csv_with_shared_grid_options() {
        let options = parse(&[
            "series.csv",
            "24",
            "--headers",
            "--value",
            "temperature={value_column='temperature',time_column='timestamp'}",
            "--from",
            "2026-01-01T00:00:00Z",
            "--to",
            "2026-01-02T00:00:00Z",
            "--step",
            "1h",
            "--fill",
            "hold",
        ])
        .unwrap();
        assert!(!options.is_database());
        assert_eq!(options.values.len(), 1);
        assert_eq!(options.source.fill, Some(FillMethod::Hold));
        assert_eq!(
            parse(&[
                "series.csv",
                "24",
                "--from",
                "2026-01-01T00:00:00Z",
                "--to",
                "2026-01-02T00:00:00Z",
                "--step",
                "1h",
            ])
            .unwrap_err()
            .kind(),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn configures_shared_defaults_and_per_series_max_hold() {
        let options = parse(&[
            "series.csv",
            "24",
            "--value",
            "series={value_column='value',time_column='time',max_hold='10m'}",
            "--from",
            "2026-01-01T00:00:00Z",
            "--to",
            "2026-01-02T00:00:00Z",
            "--step",
            "5m",
            "--aggregation",
            "last",
            "--fill",
            "linear",
            "--max-fill-gap",
            "30m",
        ])
        .unwrap();
        assert_eq!(options.aggregation, Aggregation::Last);
        assert_eq!(options.source.fill, Some(FillMethod::Linear));
        assert_eq!(options.source.max_fill_gap.as_deref(), Some("30m"));
        assert_eq!(
            parse(&["series.csv", "24", "--max-hold", "10m"])
                .unwrap_err()
                .kind(),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn provides_help_and_version_without_input() {
        let help = parse(&["--help"]).unwrap_err();
        assert_eq!(help.kind(), ErrorKind::DisplayHelp);
        assert!(help.to_string().contains("--value <VALUES>"));
        assert!(!help.to_string().contains("--value-columns"));
        assert!(help.to_string().contains("--time-column <TIME_COLUMN>"));
        assert!(help.to_string().contains("--value-column <VALUE_COLUMN>"));
        assert!(help.to_string().contains("--query-interval"));
        assert!(
            help.to_string()
                .contains("timestral-cli 24 --connection solar-prod")
        );
        assert!(
            help.to_string()
                .contains("Without a positional CSV file, provide only HORIZON")
        );
        assert_eq!(
            parse(&["--version"]).unwrap_err().kind(),
            ErrorKind::DisplayVersion
        );
    }

    #[test]
    fn validates_query_interval_and_rejects_it_for_csv_only() {
        let options = parse(&["series.csv", "24", "--query-interval", "1d"]).unwrap();
        assert_eq!(
            options.source.query_interval,
            Some(chrono::Duration::days(1))
        );
        let error = super::pipeline::prepare_inputs(&super::pipeline_config(&options)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "--query-interval requires a database target or covariate"
        );
        for interval in ["0s", "2", "3weeks", "18446744073709551615d"] {
            assert_eq!(
                parse(&["series.csv", "24", "--query-interval", interval])
                    .unwrap_err()
                    .kind(),
                ErrorKind::ValueValidation
            );
        }
    }

    #[test]
    fn parses_database_source_options() {
        let args = [
            "24",
            "--config",
            "config.toml",
            "--connection",
            "timescale",
            "--value",
            "temperature={table='public.measurements',time_column='recorded_at',value_column='temperature',filters={location='kitchen',floor='2'}}",
            "--from",
            "2026-01-01T00:00:00Z",
            "--to",
            "2026-01-02T00:00:00Z",
            "--step",
            "1h",
            "--fill",
            "hold",
        ];
        let options = parse(&args).unwrap();

        assert!(options.is_database());
        assert!(options.input.is_none());
        assert_eq!(
            options.horizon,
            NonZeroUsize::new(24).map(HorizonSpec::Steps)
        );
        assert_eq!(
            options.source.config_path,
            Some(PathBuf::from("config.toml"))
        );
        assert_eq!(options.source.connection.as_deref(), Some("timescale"));
        assert_eq!(options.values.len(), 1);
        assert_eq!(options.source.fill, Some(FillMethod::Hold));

        let mut missing_horizon = args.to_vec();
        missing_horizon.remove(0);
        let missing_horizon = parse(&missing_horizon).unwrap_err();
        assert_eq!(missing_horizon.kind(), ErrorKind::MissingRequiredArgument);
        assert!(missing_horizon.to_string().contains("HORIZON"));

        let mut args = args.to_vec();
        args.push("--headers");
        assert_eq!(
            parse(&args).unwrap_err().kind(),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn selects_database_source_from_connection_with_horizon_only() {
        let options = parse(&[
            "--connection",
            "solar-prod",
            "--config",
            "config.toml",
            "--value",
            "temperature={table='measurements',time_column='time',value_column='temperature'}",
            "--from",
            "2026-01-01T00:00:00Z",
            "--to",
            "2026-01-02T00:00:00Z",
            "--step",
            "1h",
            "24",
            "--backend",
            "cpu",
        ])
        .unwrap();

        assert!(options.is_database());
        assert_eq!(
            options.horizon,
            NonZeroUsize::new(24).map(HorizonSpec::Steps)
        );
        assert_eq!(options.source.connection.as_deref(), Some("solar-prod"));
        assert_eq!(options.backend, Some(Backend::Cpu));
    }

    #[test]
    fn treats_db_as_a_csv_path_without_connection() {
        let options = parse(&["db", "12"]).unwrap();
        assert!(!options.is_database());
        assert_eq!(options.input, Some(PathBuf::from("db")));
        assert_eq!(
            options.horizon,
            NonZeroUsize::new(12).map(HorizonSpec::Steps)
        );
    }

    #[test]
    fn accepts_duration_horizons_in_csv_and_database_positions() {
        let csv = parse(&[
            "series.csv",
            "72h",
            "--from",
            "2026-01-01T00:00:00Z",
            "--to",
            "2026-01-10T00:00:00Z",
            "--step",
            "1h",
            "--time-column",
            "timestamp",
        ])
        .unwrap();
        assert!(matches!(csv.horizon, Some(super::HorizonSpec::Duration(_))));

        let database = parse(&[
            "72h",
            "--config",
            "config.toml",
            "--connection",
            "influx",
            "--value",
            "target={table='measurements'}",
            "--time-column",
            "time",
            "--value-column",
            "value",
            "--from",
            "2026-01-01T00:00:00Z",
            "--to",
            "2026-01-10T00:00:00Z",
            "--step",
            "1h",
        ])
        .unwrap();
        assert!(database.is_database());
        assert!(matches!(
            database.horizon,
            Some(super::HorizonSpec::Duration(_))
        ));

        for invalid in ["0", "0h", "three days"] {
            assert_eq!(
                parse(&["series.csv", invalid]).unwrap_err().kind(),
                ErrorKind::ValueValidation
            );
        }
    }

    #[test]
    fn requires_all_database_fields_and_rejects_removed_source_flags() {
        let missing = parse(&["12", "--connection", "solar-prod"]).unwrap_err();
        assert_eq!(missing.kind(), ErrorKind::MissingRequiredArgument);
        for flag in ["--config", "--value", "--from", "--to", "--step"] {
            assert!(missing.to_string().contains(flag), "{missing}");
        }
        assert_eq!(
            parse(&["12", "--connection", "solar-prod", "--config"])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
        for (flag, value) in [
            ("--table", "sensors"),
            ("--filter", "x=y"),
            ("--value-columns", "x"),
        ] {
            assert_eq!(
                parse(&["series.csv", "12", flag, value])
                    .unwrap_err()
                    .kind(),
                ErrorKind::UnknownArgument
            );
        }
        assert!(parse(&["series.csv", "12", "--fill", "hold"]).is_ok());
    }

    #[test]
    fn rejects_csv_input_when_connection_selects_database() {
        let error = parse(&["series.csv", "12", "--connection", "solar-prod"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
        assert!(error.to_string().contains("remove the CSV INPUT argument"));
    }

    #[test]
    fn parses_backend_selection_and_rejects_invalid_values() {
        assert_eq!(
            parse(&["series.csv", "4", "--backend", "cpu"])
                .unwrap()
                .backend,
            Some(Backend::Cpu)
        );
        assert_eq!(
            parse(&["series.csv", "4", "--backend"]).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            parse(&["series.csv", "4", "--backend", "flex"])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
        #[cfg(feature = "wgpu")]
        assert_eq!(
            parse(&["series.csv", "4", "--backend", "WGPU"])
                .unwrap()
                .backend,
            Some(Backend::Wgpu)
        );
        #[cfg(not(feature = "wgpu"))]
        assert_eq!(
            parse(&["series.csv", "4", "--backend", "wgpu"])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn parses_either_a_local_directory_or_a_hub_variant() {
        let local = parse(&["series.csv", "4", "--model-dir", "checkpoint", "--headers"]).unwrap();
        assert_eq!(local.model_dir.as_deref(), Some(Path::new("checkpoint")));
        assert!(local.headers);
        let hub = parse(&["series.csv", "4", "--variant", "5m"]).unwrap();
        assert_eq!(hub.variant, Some(Variant::M5));
        assert_eq!(
            hub.variant.map(TimestralVariant::from),
            Some(TimestralVariant::M5)
        );
        for args in [
            &["--variant", "unknown"][..],
            &["--variant", "--headers"],
            &["--model-dir"],
        ] {
            assert_eq!(
                parse(&[&["series.csv", "4"], args].concat())
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidValue
            );
        }
        assert_eq!(
            parse(&[
                "series.csv",
                "4",
                "--model-dir",
                "checkpoint",
                "--variant",
                "5M"
            ])
            .unwrap_err()
            .kind(),
            ErrorKind::ArgumentConflict
        );
    }

    #[cfg(unix)]
    #[test]
    fn preserves_non_utf8_paths_and_rejects_non_utf8_horizon() {
        use std::os::unix::ffi::OsStringExt;

        let csv_path = OsString::from_vec(b"series-\xff.csv".to_vec());
        let model_dir = OsString::from_vec(b"checkpoint-\xfe".to_vec());
        let options = CliOptions::try_parse_args([
            "timestral-cli".into(),
            csv_path.clone(),
            "4".into(),
            "--model-dir".into(),
            model_dir.clone(),
        ])
        .unwrap();
        assert_eq!(options.input, Some(PathBuf::from(&csv_path)));
        assert_eq!(options.model_dir.as_deref(), Some(Path::new(&model_dir)));

        let invalid_horizon = OsString::from_vec(b"1\xff".to_vec());
        assert_eq!(
            CliOptions::try_parse_args(["timestral-cli".into(), csv_path, invalid_horizon])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidUtf8
        );
    }
}
