use std::collections::HashSet;
use std::io::Read;

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use csv::{ReaderBuilder, StringRecord};

use crate::series::{RawObservation, RawSeries, TimePoint, TimeWindow};

#[cfg(test)]
pub fn load(
    input: impl Read,
    has_headers: bool,
    time_column: Option<&str>,
    value_columns: &[String],
) -> Result<Vec<RawSeries>> {
    load_window(input, has_headers, time_column, value_columns, &[], None)
}

pub fn load_window(
    input: impl Read,
    has_headers: bool,
    time_column: Option<&str>,
    value_columns: &[String],
    exclude_columns: &[&str],
    window: Option<TimeWindow>,
) -> Result<Vec<RawSeries>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(input);
    let mut records = reader.records();
    let first = records
        .next()
        .context("CSV file is empty")?
        .context("Invalid CSV row 1")?;
    ensure!(!first.is_empty(), "CSV row 1 has no columns");
    let (headers, first_data, first_data_row) = if has_headers {
        (Some(first), None, 2)
    } else {
        (None, Some(first), 1)
    };

    let column_count = headers
        .as_ref()
        .or(first_data.as_ref())
        .map(StringRecord::len)
        .context("CSV file has no columns")?;
    let time_index = time_column
        .map(|column| resolve_column(column, headers.as_ref(), column_count))
        .transpose()?;
    let selected_columns = if value_columns.is_empty() {
        let exclusions = exclude_columns
            .iter()
            .map(|name| resolve_column(name, headers.as_ref(), column_count))
            .collect::<Result<Vec<_>>>()?;
        (0..column_count)
            .filter(|index| Some(*index) != time_index && !exclusions.contains(index))
            .collect::<Vec<_>>()
    } else {
        value_columns
            .iter()
            .map(|column| resolve_column(column, headers.as_ref(), column_count))
            .collect::<Result<Vec<_>>>()?
    };
    ensure!(
        !selected_columns.is_empty(),
        "CSV contains no value columns"
    );
    ensure!(
        selected_columns
            .iter()
            .all(|index| Some(*index) != time_index),
        "The time column cannot also be selected as a value column"
    );

    let usable_headers = headers.as_ref().filter(|headers| {
        let mut names = HashSet::new();
        headers
            .iter()
            .all(|name| !name.trim().is_empty() && names.insert(name))
    });
    let mut series = selected_columns
        .iter()
        .map(|index| RawSeries {
            name: usable_headers
                .and_then(|headers| headers.get(*index))
                .map(str::to_owned)
                .unwrap_or_else(|| (index + 1).to_string()),
            observations: Vec::new(),
        })
        .collect::<Vec<_>>();
    for (row_offset, record) in first_data
        .into_iter()
        .map(Ok::<_, csv::Error>)
        .chain(records)
        .enumerate()
    {
        let row_number = first_data_row + row_offset;
        let record = record.with_context(|| format!("Invalid CSV row {row_number}"))?;
        ensure!(
            record.len() == column_count,
            "CSV row {row_number} has {} columns; expected {column_count}",
            record.len()
        );
        let time = if let Some(index) = time_index {
            let value = record.get(index).unwrap_or_default().trim();
            let timestamp = DateTime::parse_from_rfc3339(value)
                .with_context(|| {
                    format!(
                        "Invalid RFC 3339 timestamp at CSV row {row_number}, column {}",
                        index + 1
                    )
                })?
                .with_timezone(&Utc);
            if window.is_some_and(|(from, to)| timestamp < from || timestamp >= to) {
                continue;
            }
            TimePoint::Timestamp(timestamp)
        } else {
            TimePoint::Index((row_number - first_data_row) as i64)
        };
        for (series, column) in series.iter_mut().zip(&selected_columns) {
            let field = record.get(*column).unwrap_or_default().trim();
            let value = if field.is_empty() || field.eq_ignore_ascii_case("null") {
                None
            } else {
                let value = field.parse::<f32>().with_context(|| {
                    format!(
                        "Invalid number at CSV row {row_number}, column {}: {field:?}",
                        column + 1
                    )
                })?;
                ensure!(
                    value.is_finite(),
                    "Non-finite number at CSV row {row_number}, column {}",
                    column + 1
                );
                Some(value)
            };
            series.observations.push(RawObservation {
                time: time.clone(),
                value,
            });
        }
    }
    Ok(series)
}

fn resolve_column(
    selector: &str,
    headers: Option<&StringRecord>,
    column_count: usize,
) -> Result<usize> {
    if let Some(headers) = headers {
        let mut matches = headers
            .iter()
            .enumerate()
            .filter_map(|(i, name)| (name == selector).then_some(i));
        if let Some(index) = matches.next() {
            ensure!(
                matches.next().is_none(),
                "CSV column name {selector:?} is ambiguous; use a 1-based column index"
            );
            return Ok(index);
        }
    }
    let index = selector
        .parse::<usize>()
        .ok()
        .and_then(|index| index.checked_sub(1))
        .filter(|index| *index < column_count);
    index.ok_or_else(|| {
        anyhow::anyhow!(
            "CSV column {selector:?} was not found; use a header name or a 1-based column index"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::load;
    use crate::series::TimePoint;
    use std::io::Cursor;

    #[test]
    fn reads_untimed_csv_as_indexed_named_series() {
        let series = load(
            Cursor::new("temperature,humidity\n10,1\n20,2\n"),
            true,
            None,
            &[],
        )
        .unwrap();
        assert_eq!(series.len(), 2);
        assert_eq!(series[0].name, "temperature");
        assert_eq!(series[0].observations[1].time, TimePoint::Index(1));
        assert_eq!(series[1].observations[0].value, Some(1.0));
    }

    #[test]
    fn reads_timestamps_and_missing_values_and_selects_columns() {
        let series = load(
            Cursor::new(
                "time,pv,temperature\n2026-01-02T00:00:00Z,,10\n2026-01-02T00:05:00Z,2,11\n",
            ),
            true,
            Some("time"),
            &["pv".into()],
        )
        .unwrap();
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].name, "pv");
        assert_eq!(series[0].observations[0].value, None);
        assert_eq!(
            series[0].observations[1].time,
            TimePoint::Timestamp(
                chrono::DateTime::parse_from_rfc3339("2026-01-02T00:05:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc)
            )
        );
    }

    #[test]
    fn selects_no_header_columns_by_one_based_index() {
        let series = load(
            Cursor::new("2026-01-02T00:00:00Z,7\n"),
            false,
            Some("1"),
            &["2".into()],
        )
        .unwrap();
        assert_eq!(series[0].name, "2");
        assert_eq!(series[0].observations[0].value, Some(7.0));
    }

    #[test]
    fn rejects_invalid_timestamps_values_and_selection() {
        assert!(load(Cursor::new("time,x\nnot-time,1\n"), true, Some("time"), &[]).is_err());
        assert!(load(Cursor::new("x\nnope\n"), true, None, &[]).is_err());
        assert!(load(Cursor::new("1,2\n"), false, None, &["3".into()]).is_err());
    }

    #[test]
    fn preserves_legacy_csv_with_duplicate_or_empty_header_labels() {
        let csv = "x,x,\n1,2,3\n4,5,6\n";
        let series = load(Cursor::new(csv), true, None, &[]).unwrap();
        assert_eq!(
            series.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["1", "2", "3"]
        );
        assert_eq!(series[1].observations[0].value, Some(2.0));
        assert!(load(Cursor::new(csv), true, None, &["x".into()]).is_err());
    }

    #[test]
    fn preserves_csv_read_error_diagnostics() {
        for (input, diagnostic) in [
            ("", "CSV file is empty"),
            ("1,2\n3\n", "CSV row 2 has 1 columns; expected 2"),
            ("NaN\n", "Non-finite number at CSV row 1, column 1"),
        ] {
            let error = load(Cursor::new(input), false, None, &[]).unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error:#}");
        }
        let error = load(Cursor::new(b"1\n\xff\n"), false, None, &[]).unwrap_err();
        assert_eq!(error.to_string(), "Invalid CSV row 2");
        assert!(matches!(
            error.downcast_ref::<csv::Error>().map(csv::Error::kind),
            Some(csv::ErrorKind::Utf8 { .. })
        ));
    }
}
