use std::collections::{BTreeMap, HashSet};
#[cfg(feature = "csv")]
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use chrono::Duration;
use clap::Args;
use serde::Deserialize;

#[cfg(feature = "csv")]
use crate::csv_input;
use crate::database::{ConfigFile, DatabaseSource, QueryWindowSuccess};
use crate::series::{Aggregation, FillMethod, RawSeries, TimeWindow, parse_duration};

#[derive(Clone, Debug, Args)]
/// Shared defaults for resolving and loading target and covariate sources.
pub struct SourceOptions {
    /// TOML file with connection and named series settings.
    #[arg(long = "config")]
    pub config_path: Option<PathBuf>,
    /// Default connection name for database series.
    #[arg(long)]
    pub connection: Option<String>,
    /// Default time column for targets and covariates.
    #[arg(long)]
    pub time_column: Option<String>,
    /// Default value column for targets and covariates.
    #[arg(long)]
    pub value_column: Option<String>,
    /// Inclusive start of a timestamped input window (RFC 3339).
    #[arg(long)]
    pub from: Option<String>,
    /// Exclusive end of a timestamped input window (RFC 3339).
    #[arg(long)]
    pub to: Option<String>,
    /// Grid interval for timestamped sources (e.g. 30s, 5m, 1h).
    #[arg(long)]
    pub step: Option<String>,
    /// Maximum database query interval (e.g. 1d); defaults to the full window, shrinking on file limits.
    #[arg(long, value_parser = parse_query_interval)]
    pub query_interval: Option<Duration>,
    /// Default gap fill rule for targets and covariates.
    #[arg(long, value_enum)]
    pub fill: Option<FillMethod>,
    /// Maximum contiguous gap to fill (duration for timestamped sources, steps for row-indexed CSV).
    #[arg(long = "max-fill-gap")]
    pub max_fill_gap: Option<String>,
}

impl SourceOptions {
    /// Returns the names of required database input options that are missing.
    pub fn missing_db_options(&self, has_values: bool) -> Vec<&'static str> {
        let mut missing = [
            (self.config_path.is_none(), "--config"),
            (!has_values, "--value"),
        ]
        .into_iter()
        .filter_map(|(missing, flag)| missing.then_some(flag))
        .collect::<Vec<_>>();
        missing.extend(self.missing_window_options());
        missing
    }

    /// Returns the names of missing timestamp window options.
    pub fn missing_window_options(&self) -> Vec<&'static str> {
        [
            (self.from.is_none(), "--from"),
            (self.to.is_none(), "--to"),
            (self.step.is_none(), "--step"),
        ]
        .into_iter()
        .filter_map(|(missing, flag)| missing.then_some(flag))
        .collect()
    }
}

fn parse_query_interval(input: &str) -> Result<Duration, String> {
    let duration = parse_duration(input, "--query-interval").map_err(|error| error.to_string())?;
    Duration::from_std(duration)
        .map_err(|_| "--query-interval exceeds the supported duration".into())
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesSpec {
    csv: Option<PathBuf>,
    headers: Option<bool>,
    connection: Option<String>,
    table: Option<String>,
    time_column: Option<String>,
    pub value_column: Option<String>,
    #[serde(default)]
    filters: BTreeMap<String, String>,
    #[serde(default)]
    pub aggregation: Option<Aggregation>,
    pub max_hold: Option<String>,
    pub fill: Option<FillMethod>,
    pub max_fill_gap: Option<String>,
}

#[derive(Debug, Clone)]
/// A named series selection or an inline series definition.
pub struct SeriesArg {
    name: String,
    inline: Option<SeriesSpec>,
}

/// Parses a configured series name or an inline `name={...}` TOML definition.
pub fn parse_series_arg(value: &str) -> Result<SeriesArg, String> {
    let (name, inline) = value
        .split_once('=')
        .map_or((value.trim(), None), |(name, spec)| {
            (name.trim(), Some(spec.trim()))
        });
    if name.is_empty() {
        return Err("a series name is required".into());
    }
    let inline = inline
        .map(|definition| {
            let text = format!("series = {definition}");
            parse_inline(&text).map_err(|_| format!("invalid inline TOML for series {name:?}"))
        })
        .transpose()?;
    Ok(SeriesArg {
        name: name.into(),
        inline,
    })
}

fn parse_inline(text: &str) -> Result<SeriesSpec, toml::de::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InlineSpec {
        series: SeriesSpec,
    }
    toml::from_str::<InlineSpec>(text).map(|entry| entry.series)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsvSource<'a> {
    path: &'a Path,
    headers: bool,
    time_column: Option<&'a str>,
}

pub enum Source<'a> {
    Csv(CsvSource<'a>),
    Database(DatabaseSource<'a>),
}

impl<'a> Source<'a> {
    pub fn default_csv(path: &'a Path, headers: bool, time_column: Option<&'a str>) -> Self {
        Self::Csv(CsvSource {
            path,
            headers,
            time_column,
        })
    }

    pub fn is_timestamped(&self) -> bool {
        match self {
            Self::Csv(csv) => csv.time_column.is_some(),
            Self::Database(_) => true,
        }
    }

    pub fn load(
        self,
        columns: &[String],
        exclude_columns: &[&str],
        window: Option<TimeWindow>,
        query_interval: Option<Duration>,
        on_query_success: &mut impl FnMut(QueryWindowSuccess) -> std::io::Result<()>,
    ) -> Result<Vec<RawSeries>> {
        #[cfg(not(feature = "csv"))]
        let _ = exclude_columns;
        match self {
            Self::Csv(csv) => {
                #[cfg(feature = "csv")]
                {
                    let file = File::open(csv.path)
                        .with_context(|| format!("Failed to open {}", csv.path.display()))?;
                    csv_input::load_window(
                        file,
                        csv.headers,
                        csv.time_column,
                        columns,
                        exclude_columns,
                        window,
                    )
                }
                #[cfg(not(feature = "csv"))]
                {
                    let _ = csv;
                    Err(anyhow::anyhow!(
                        "CSV support is disabled; enable the `csv` feature"
                    ))
                }
            }
            Self::Database(database) => Ok(database.load(
                columns,
                window.context("Database input requires a timestamp grid")?,
                query_interval,
                on_query_success,
            )?),
        }
    }

    pub fn shares_csv(&self, other: &Self) -> bool {
        matches!((self, other), (Self::Csv(left), Self::Csv(right)) if left == right)
    }
}

pub fn resolve_specs<'a>(
    selections: &'a [SeriesArg],
    config: Option<&'a ConfigFile>,
) -> Result<Vec<(&'a str, &'a SeriesSpec)>> {
    let mut names = HashSet::new();
    selections
        .iter()
        .map(|selection| {
            let name = selection.name.as_str();
            ensure!(names.insert(name), "Duplicate series {name:?}");
            let spec = selection
                .inline
                .as_ref()
                .or_else(|| {
                    config.and_then(|config| {
                        config
                            .series
                            .get(name)
                            .or_else(|| config.covariates.get(name))
                    })
                })
                .with_context(|| {
                    format!("Unknown series {name:?}; define it in --config or inline")
                })?;
            Ok((name, spec))
        })
        .collect()
}

impl SeriesSpec {
    pub fn effective_value_column<'a>(&'a self, options: &'a SourceOptions) -> Result<&'a str> {
        let column = self
            .value_column
            .as_deref()
            .or(options.value_column.as_deref())
            .context("A series requires value_column or --value-column")?;
        ensure!(!column.trim().is_empty(), "value_column cannot be empty");
        Ok(column)
    }

    pub fn effective_time_column<'a>(&'a self, options: &'a SourceOptions) -> Option<&'a str> {
        self.time_column
            .as_deref()
            .or(options.time_column.as_deref())
    }

    pub fn positional_csv_time_column(&self, path: &Path) -> Option<&str> {
        if self.connection.is_none() && self.csv.as_deref().is_none_or(|csv| csv == path) {
            self.time_column.as_deref()
        } else {
            None
        }
    }

    pub fn source<'a>(
        &'a self,
        default_csv: Option<&Source<'a>>,
        options: &'a SourceOptions,
        config: Option<&'a ConfigFile>,
    ) -> Result<Source<'a>> {
        let value_column = self.effective_value_column(options)?;
        ensure!(
            self.csv.is_none() || self.connection.is_none(),
            "csv and connection are mutually exclusive"
        );
        let target_csv = match default_csv {
            Some(Source::Csv(csv)) => Some(*csv),
            _ => None,
        };
        let source = if self.csv.is_some() || (self.connection.is_none() && target_csv.is_some()) {
            ensure!(
                self.table.is_none() && self.filters.is_empty(),
                "CSV sources do not accept table or filters"
            );
            let inherited =
                target_csv.filter(|csv| self.csv.as_deref().is_none_or(|path| path == csv.path));
            let path = self
                .csv
                .as_deref()
                .or(inherited.map(|csv| csv.path))
                .context("Missing CSV file")?;
            ensure!(!path.as_os_str().is_empty(), "csv cannot be empty");
            let time_column = self
                .effective_time_column(options)
                .or(inherited.and_then(|csv| csv.time_column));
            ensure!(
                time_column.is_none_or(|column| !column.trim().is_empty()),
                "time_column cannot be empty"
            );
            Source::Csv(CsvSource {
                path,
                headers: self
                    .headers
                    .unwrap_or_else(|| inherited.is_some_and(|csv| csv.headers)),
                time_column,
            })
        } else {
            ensure!(
                self.headers.is_none(),
                "Database sources do not accept headers"
            );
            let connection = self
                .connection
                .as_deref()
                .or(options.connection.as_deref())
                .context("A database connection is required")?;
            let table = self
                .table
                .as_deref()
                .context("A database series requires table")?;
            let time_column = self
                .effective_time_column(options)
                .context("A database series requires time_column or --time-column")?;
            ensure!(
                !time_column.trim().is_empty(),
                "time_column cannot be empty"
            );
            let value_columns = [value_column.to_owned()];
            Source::Database(DatabaseSource::new(
                config.context("Database series require --config")?,
                connection,
                table,
                time_column,
                &value_columns,
                self.filters
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            )?)
        };

        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::{SeriesSpec, SourceOptions, parse_series_arg};

    fn source_options(time_column: Option<&str>, value_column: Option<&str>) -> SourceOptions {
        SourceOptions {
            config_path: None,
            connection: None,
            time_column: time_column.map(str::to_owned),
            value_column: value_column.map(str::to_owned),
            from: None,
            to: None,
            step: None,
            query_interval: None,
            fill: None,
            max_fill_gap: None,
        }
    }

    #[test]
    fn series_inherits_global_column_defaults() {
        let selection = parse_series_arg("temperature={table='weather'}").unwrap();
        let spec = selection.inline.unwrap();
        let options = source_options(Some("observed_at"), Some("reading"));

        assert_eq!(spec.effective_time_column(&options), Some("observed_at"));
        assert_eq!(spec.effective_value_column(&options).unwrap(), "reading");
    }

    #[test]
    fn explicit_series_columns_override_global_defaults() {
        let selection = parse_series_arg(
            "temperature={table='weather',time_column='recorded_at',value_column='celsius'}",
        )
        .unwrap();
        let spec = selection.inline.unwrap();
        let options = source_options(Some("observed_at"), Some("reading"));

        assert_eq!(spec.effective_time_column(&options), Some("recorded_at"));
        assert_eq!(spec.effective_value_column(&options).unwrap(), "celsius");
    }

    #[test]
    fn selected_series_without_value_column_requires_global_default() {
        let selection = parse_series_arg("temperature={table='weather'}").unwrap();
        let spec: SeriesSpec = selection.inline.unwrap();
        let options = source_options(None, None);

        assert_eq!(
            spec.effective_value_column(&options)
                .unwrap_err()
                .to_string(),
            "A series requires value_column or --value-column"
        );
    }
}
