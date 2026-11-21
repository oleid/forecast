use burn::tensor::Device;
use burn_timestral::{Forecast, ModelError, TimestralForecaster, TimestralVariant};
use std::{
    ffi::OsString,
    fs::File,
    io::{self, Read, Write},
    num::{ParseFloatError, ParseIntError},
    path::PathBuf,
    process::ExitCode,
};
use thiserror::Error;

#[derive(Debug, Error)]
enum CliError {
    #[error(
        "Usage: burn-timestral <file.csv> <horizon> [--headers] [--backend <cpu|wgpu>] [--model-dir <path> | --variant <5M|28M|123M|209M>]"
    )]
    Usage,
    #[error("Invalid forecast horizon: {value:?}")]
    InvalidHorizon {
        value: OsString,
        #[source]
        source: ParseIntError,
    },
    #[error("Forecast horizon must be greater than zero")]
    ZeroHorizon,
    #[error("--model-dir requires a path")]
    MissingModelDir,
    #[error("--variant requires a model size")]
    MissingVariant,
    #[error("--backend requires a backend name (cpu or wgpu)")]
    MissingBackend,
    #[error("Unknown model variant: {0:?}")]
    UnknownVariant(OsString),
    #[error("Unknown backend: {0:?}; expected cpu or wgpu")]
    UnknownBackend(OsString),
    #[cfg(not(feature = "wgpu"))]
    #[error("The WGPU backend requires the `wgpu` Cargo feature")]
    WgpuRequiresFeature,
    #[cfg(feature = "wgpu")]
    #[error("Failed to initialize the WGPU backend: {0}")]
    WgpuInitialization(#[from] burn::tensor::wgpu::WgpuInitError),
    #[error("Unexpected argument: {0:?}")]
    UnexpectedArgument(OsString),
    #[error("--variant is only used when --model-dir is not supplied")]
    ConflictingModelOptions,
    #[error("Failed to open {}: {source}", .path.display())]
    OpenCsv {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Invalid CSV row {row}: {source}")]
    ReadCsv {
        row: usize,
        #[source]
        source: csv::Error,
    },
    #[error("CSV file is empty")]
    EmptyCsv,
    #[error("CSV file has no data rows")]
    NoDataRows,
    #[error("CSV header has no columns")]
    EmptyHeader,
    #[error("CSV row {row} has no columns")]
    EmptyRow { row: usize },
    #[error("CSV row {row} has {actual} columns; expected {expected}")]
    WrongColumnCount {
        row: usize,
        actual: usize,
        expected: usize,
    },
    #[error("Invalid number at CSV row {row}, column {column}: {field:?}")]
    InvalidNumber {
        row: usize,
        column: usize,
        field: String,
        #[source]
        source: ParseFloatError,
    },
    #[error("Non-finite number at CSV row {row}, column {column}")]
    NonFiniteNumber { row: usize, column: usize },
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("Failed to write forecast to stdout: {0}")]
    WriteForecast(#[from] io::Error),
}

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(io::stderr().lock(), "{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<(), CliError> {
    let options = parse_args(arguments)?;
    let file = File::open(&options.csv_path).map_err(|source| CliError::OpenCsv {
        path: options.csv_path,
        source,
    })?;
    let context = parse_csv(file, options.has_headers)?;

    let device = match options.backend {
        None => Device::default(),
        Some(Backend::Cpu) => Device::flex(),
        #[cfg(feature = "wgpu")]
        Some(Backend::Wgpu) => Device::wgpu_options().init()?,
    };
    let forecaster =
        TimestralForecaster::load(options.variant, options.model_dir.as_deref(), &device)?;
    let forecast = forecaster.forecast(
        &context.values,
        1,
        context.num_variates,
        context.context_length,
        options.horizon,
        None,
    )?;

    write_forecast(io::stdout().lock(), &forecast)
}

struct CliOptions {
    csv_path: PathBuf,
    horizon: usize,
    has_headers: bool,
    model_dir: Option<PathBuf>,
    variant: TimestralVariant,
    backend: Option<Backend>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

struct CsvContext {
    values: Vec<f32>,
    num_variates: usize,
    context_length: usize,
}

fn parse_args(arguments: impl IntoIterator<Item = OsString>) -> Result<CliOptions, CliError> {
    let mut flags = arguments.into_iter();
    let csv_path = flags.next().ok_or(CliError::Usage)?;
    let horizon = flags.next().ok_or(CliError::Usage)?;
    let horizon = horizon
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|source| CliError::InvalidHorizon {
            value: horizon,
            source,
        })?;
    if horizon == 0 {
        return Err(CliError::ZeroHorizon);
    }

    let mut has_headers = false;
    let mut model_dir = None;
    let mut variant = None;
    let mut backend = None;
    while let Some(flag) = flags.next() {
        match flag.to_str() {
            Some("--headers") if !has_headers => has_headers = true,
            Some("--model-dir") if model_dir.is_none() => {
                let dir = flags
                    .next()
                    .filter(|value| !value.as_encoded_bytes().starts_with(b"--"))
                    .ok_or(CliError::MissingModelDir)?;
                model_dir = Some(PathBuf::from(dir));
            }
            Some("--variant") if variant.is_none() => {
                let size = flags
                    .next()
                    .filter(|value| !value.as_encoded_bytes().starts_with(b"--"))
                    .ok_or(CliError::MissingVariant)?;
                variant = Some(match size.to_ascii_uppercase().to_str() {
                    Some("5M") => TimestralVariant::M5,
                    Some("28M") => TimestralVariant::M28,
                    Some("123M") => TimestralVariant::M123,
                    Some("209M") => TimestralVariant::M209,
                    _ => return Err(CliError::UnknownVariant(size)),
                });
            }
            Some("--backend") if backend.is_none() => {
                let name = flags
                    .next()
                    .filter(|value| !value.as_encoded_bytes().starts_with(b"--"))
                    .ok_or(CliError::MissingBackend)?;
                backend = Some(match name.to_ascii_lowercase().to_str() {
                    Some("cpu") => Backend::Cpu,
                    #[cfg(feature = "wgpu")]
                    Some("wgpu") => Backend::Wgpu,
                    #[cfg(not(feature = "wgpu"))]
                    Some("wgpu") => return Err(CliError::WgpuRequiresFeature),
                    _ => return Err(CliError::UnknownBackend(name)),
                });
            }
            _ => return Err(CliError::UnexpectedArgument(flag)),
        }
    }
    if model_dir.is_some() && variant.is_some() {
        return Err(CliError::ConflictingModelOptions);
    }
    Ok(CliOptions {
        csv_path: PathBuf::from(csv_path),
        horizon,
        has_headers,
        model_dir,
        variant: variant.unwrap_or_default(),
        backend,
    })
}

fn parse_csv<R: Read>(reader: R, has_headers: bool) -> Result<CsvContext, CliError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(reader);
    let mut columns = None;
    let mut context_length = 0;

    for (row_index, record) in reader.records().enumerate() {
        let row = row_index + 1;
        let is_header = row_index == 0 && has_headers;
        let record = record.map_err(|source| CliError::ReadCsv { row, source })?;
        if record.is_empty() {
            return Err(if is_header {
                CliError::EmptyHeader
            } else {
                CliError::EmptyRow { row }
            });
        }
        let columns = columns.get_or_insert_with(|| vec![Vec::new(); record.len()]);
        if is_header {
            continue;
        }
        if record.len() != columns.len() {
            return Err(CliError::WrongColumnCount {
                row,
                actual: record.len(),
                expected: columns.len(),
            });
        }

        for (column_index, (values, field)) in columns.iter_mut().zip(record.iter()).enumerate() {
            let column = column_index + 1;
            let value = field
                .trim()
                .parse::<f32>()
                .map_err(|source| CliError::InvalidNumber {
                    row,
                    column,
                    field: field.to_owned(),
                    source,
                })?;
            if !value.is_finite() {
                return Err(CliError::NonFiniteNumber { row, column });
            }
            values.push(value);
        }
        context_length += 1;
    }

    let columns = columns.ok_or(CliError::EmptyCsv)?;
    if context_length == 0 {
        return Err(CliError::NoDataRows);
    }

    let num_variates = columns.len();
    Ok(CsvContext {
        values: columns.into_iter().flatten().collect(),
        num_variates,
        context_length,
    })
}

fn write_forecast(mut output: impl Write, forecast: &Forecast) -> Result<(), CliError> {
    writeln!(output, "variable,step,quantile,value")?;
    let (_, num_variates, prediction_length, num_quantiles) = forecast.shape();
    let first_batch = forecast
        .predictions()
        .chunks_exact(prediction_length * num_quantiles)
        .take(num_variates);
    for (variable, series) in first_batch.enumerate() {
        for (step, quantiles) in series.chunks_exact(num_quantiles).enumerate() {
            for (level, value) in forecast.quantile_levels().iter().zip(quantiles) {
                writeln!(
                    output,
                    "{},{},{level:.3},{value:.6}",
                    variable + 1,
                    step + 1
                )?;
            }
        }
    }
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Backend, CliError, CliOptions, parse_args, parse_csv, write_forecast};
    use burn_timestral::{Forecast, TimestralVariant};
    use std::{
        ffi::OsString,
        io::{self, Cursor, Write},
        path::Path,
    };

    fn parse(arguments: &[&str]) -> Result<CliOptions, CliError> {
        parse_args(arguments.iter().copied().map(OsString::from))
    }

    #[test]
    fn parses_numeric_csv_as_series_by_column() {
        let context = parse_csv(Cursor::new("10,1\n20,2\n30,3\n"), false).unwrap();

        assert_eq!(context.num_variates, 2);
        assert_eq!(context.context_length, 3);
        assert_eq!(context.values, vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn skips_csv_header_when_requested() {
        let context = parse_csv(Cursor::new("temperature,humidity\n10,1\n20,2\n"), true).unwrap();

        assert_eq!(context.num_variates, 2);
        assert_eq!(context.context_length, 2);
        assert_eq!(context.values, vec![10.0, 20.0, 1.0, 2.0]);
    }

    #[test]
    fn rejects_empty_ragged_and_non_numeric_csv() {
        assert!(matches!(
            parse_csv(Cursor::new(""), false),
            Err(CliError::EmptyCsv)
        ));
        assert!(matches!(
            parse_csv(Cursor::new("temperature,humidity\n"), true),
            Err(CliError::NoDataRows)
        ));
        assert!(matches!(
            parse_csv(Cursor::new("1,2\n3\n"), false),
            Err(CliError::WrongColumnCount {
                row: 2,
                actual: 1,
                expected: 2
            })
        ));
        assert!(matches!(
            parse_csv(Cursor::new("1,not-a-number\n"), false),
            Err(CliError::InvalidNumber {
                row: 1,
                column: 2,
                ..
            })
        ));
        assert!(matches!(
            parse_csv(Cursor::new("NaN\n"), false),
            Err(CliError::NonFiniteNumber { row: 1, column: 1 })
        ));
    }

    #[test]
    fn preserves_csv_reader_error() {
        assert!(matches!(
            parse_csv(Cursor::new(b"1\n\xff\n"), false),
            Err(CliError::ReadCsv { row: 2, source })
                if matches!(source.kind(), csv::ErrorKind::Utf8 { .. })
        ));
    }

    #[test]
    fn parses_cli_arguments_and_rejects_invalid_horizons() {
        let options = parse(&["series.csv", "12"]).unwrap();
        assert_eq!(options.csv_path, Path::new("series.csv"));
        assert_eq!(options.horizon, 12);
        assert!(!options.has_headers);
        assert_eq!(options.variant, TimestralVariant::M209);
        assert_eq!(options.backend, None);
        assert!(options.model_dir.is_none());

        let with_headers = parse(&["series.csv", "12", "--headers"]).unwrap();
        assert!(with_headers.has_headers);
        assert!(matches!(parse(&["series.csv"]), Err(CliError::Usage)));
        assert!(matches!(
            parse(&["series.csv", "0"]),
            Err(CliError::ZeroHorizon)
        ));
        assert!(matches!(
            parse(&["series.csv", "many"]),
            Err(CliError::InvalidHorizon { .. })
        ));
    }

    #[test]
    fn parses_backend_selection_and_rejects_invalid_values() {
        assert_eq!(
            parse(&["series.csv", "4", "--backend", "cpu"])
                .unwrap()
                .backend,
            Some(Backend::Cpu)
        );
        assert!(matches!(
            parse(&["series.csv", "4", "--backend"]),
            Err(CliError::MissingBackend)
        ));
        assert!(matches!(
            parse(&["series.csv", "4", "--backend", "flex"]),
            Err(CliError::UnknownBackend(_))
        ));
        #[cfg(feature = "wgpu")]
        assert_eq!(
            parse(&["series.csv", "4", "--backend", "WGPU"])
                .unwrap()
                .backend,
            Some(Backend::Wgpu)
        );
        #[cfg(not(feature = "wgpu"))]
        assert!(matches!(
            parse(&["series.csv", "4", "--backend", "wgpu"]),
            Err(CliError::WgpuRequiresFeature)
        ));
    }

    #[test]
    fn parses_either_a_local_directory_or_a_hub_variant() {
        let local = parse(&["series.csv", "4", "--model-dir", "checkpoint", "--headers"]).unwrap();
        assert_eq!(local.model_dir.as_deref(), Some(Path::new("checkpoint")));
        assert!(local.has_headers);
        assert_eq!(
            parse(&["series.csv", "4", "--variant", "5m"])
                .unwrap()
                .variant,
            TimestralVariant::M5
        );
        assert!(matches!(
            parse(&["series.csv", "4", "--variant", "unknown"]),
            Err(CliError::UnknownVariant(_))
        ));
        assert!(matches!(
            parse(&["series.csv", "4", "--variant", "--headers"]),
            Err(CliError::MissingVariant)
        ));
        assert!(matches!(
            parse(&["series.csv", "4", "--model-dir"]),
            Err(CliError::MissingModelDir)
        ));
        assert!(matches!(
            parse(&[
                "series.csv",
                "4",
                "--model-dir",
                "checkpoint",
                "--variant",
                "5M"
            ]),
            Err(CliError::ConflictingModelOptions)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn preserves_non_utf8_paths_and_rejects_non_utf8_horizon() {
        use std::os::unix::ffi::OsStringExt;

        let csv_path = OsString::from_vec(b"series-\xff.csv".to_vec());
        let model_dir = OsString::from_vec(b"checkpoint-\xfe".to_vec());
        let options = parse_args([
            csv_path.clone(),
            "4".into(),
            "--model-dir".into(),
            model_dir.clone(),
        ])
        .unwrap();
        assert_eq!(options.csv_path.as_os_str(), csv_path.as_os_str());
        assert_eq!(options.model_dir.as_deref(), Some(Path::new(&model_dir)));

        let invalid_horizon = OsString::from_vec(b"1\xff".to_vec());
        assert!(matches!(
            parse_args([csv_path, invalid_horizon.clone()]),
            Err(CliError::InvalidHorizon { value, .. }) if value == invalid_horizon
        ));
    }

    #[test]
    fn formats_each_series_step_and_quantile() {
        let predictions = (1..=16).map(|value| value as f32).collect();
        let forecast = Forecast::new(predictions, vec![0.1, 0.5], 2, 2, 2).unwrap();
        let mut output = Vec::new();

        write_forecast(&mut output, &forecast).unwrap();
        assert_eq!(
            output.as_slice(),
            b"variable,step,quantile,value\n\
              1,1,0.100,1.000000\n1,1,0.500,2.000000\n1,2,0.100,3.000000\n1,2,0.500,4.000000\n\
              2,1,0.100,5.000000\n2,1,0.500,6.000000\n2,2,0.100,7.000000\n2,2,0.500,8.000000\n"
        );
    }

    #[test]
    fn propagates_output_write_and_flush_failures() {
        struct BrokenOutput {
            fail_on_write: bool,
        }

        impl Write for BrokenOutput {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.fail_on_write {
                    return Err(io::ErrorKind::BrokenPipe.into());
                }
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }

        let forecast = Forecast::new(vec![1.0], vec![0.5], 1, 1, 1).unwrap();
        for fail_on_write in [true, false] {
            assert!(matches!(
                write_forecast(BrokenOutput { fail_on_write }, &forecast),
                Err(CliError::WriteForecast(source)) if source.kind() == io::ErrorKind::BrokenPipe
            ));
        }
    }
}
