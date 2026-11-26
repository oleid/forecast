use std::io;
#[cfg(feature = "influx3")]
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Duration, NaiveDateTime, TimeZone, Utc};
#[cfg(feature = "postgres")]
use postgres::{error::SqlState, types::ToSql};
use serde::Deserialize;
use thiserror::Error;

use crate::series::{RawObservation, RawSeries, TimePoint, TimeWindow};
use crate::series_config::SeriesSpec;

#[derive(Debug, Error)]
pub enum DatabaseError {
    #[error("Failed to read database configuration {path}: {source}")]
    ReadConfig {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Invalid TOML database configuration {0}")]
    ParseConfig(PathBuf),
    #[error("No database connection named {0:?} exists in the configuration")]
    UnknownConnection(String),
    #[error("Connection {0:?} has an unsupported provider; expected influxdb3 or timescaledb")]
    UnsupportedProvider(String),
    #[error("Database URL or database name is missing from the selected connection")]
    IncompleteConnection,
    #[error("Invalid SQL identifier: {0:?}")]
    InvalidIdentifier(String),
    #[error("Database connection failed")]
    Connection,
    #[error("Database request failed")]
    Request,
    #[error("Database returned HTTP status {0}")]
    HttpStatus(u16),
    #[error("Database query exceeded a file limit")]
    QueryFileLimit,
    #[error("Database query exceeded a file limit and [{from}, {to}) cannot be split further")]
    QueryFileLimitAtMinimum {
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    },
    #[error("Failed to write database query progress")]
    Progress(#[from] io::Error),
    #[error("Database returned invalid or unexpected query data")]
    InvalidResponse,
}

#[derive(Clone, Debug)]
/// Timing and bounds of a successfully loaded database window.
pub struct QueryWindowSuccess {
    /// Inclusive lower bound of the query window.
    pub from: DateTime<Utc>,
    /// Exclusive upper bound of the query window.
    pub to: DateTime<Utc>,
    /// Time spent executing and loading the query.
    pub elapsed: std::time::Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Provider {
    Unsupported,
    #[cfg(feature = "influx3")]
    InfluxDb3,
    #[cfg(feature = "postgres")]
    TimescaleDb,
}

#[derive(Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    connections: Vec<ConnectionConfig>,
    #[serde(default)]
    pub series: std::collections::BTreeMap<String, SeriesSpec>,
    #[serde(default)]
    pub covariates: std::collections::BTreeMap<String, SeriesSpec>,
    #[cfg(feature = "influx3")]
    influx_url: Option<String>,
    #[cfg(feature = "influx3")]
    influx_db: Option<String>,
    #[cfg(feature = "influx3")]
    influx_token: Option<String>,
}

#[derive(Deserialize)]
struct ConnectionConfig {
    name: String,
    provider: String,
    url: String,
    #[cfg(feature = "influx3")]
    database: Option<String>,
    #[cfg(feature = "influx3")]
    token: Option<String>,
}

struct Connection<'a> {
    provider: Provider,
    url: &'a str,
    #[cfg(feature = "influx3")]
    database: &'a str,
    #[cfg(feature = "influx3")]
    token: Option<&'a str>,
}

struct QuerySpec<'a> {
    table: &'a str,
    time_column: &'a str,
    value_columns: &'a [String],
    filters: &'a [(String, String)],
}

struct Observation {
    timestamp: DateTime<Utc>,
    values: Vec<Option<f32>>,
}

pub struct DatabaseSource<'a> {
    connection: Connection<'a>,
    table: &'a str,
    time_column: &'a str,
    filters: Vec<(String, String)>,
}

impl<'a> DatabaseSource<'a> {
    pub fn new(
        config: &'a ConfigFile,
        connection: &str,
        table: &'a str,
        time_column: &'a str,
        columns: &[String],
        filters: Vec<(String, String)>,
    ) -> Result<Self, DatabaseError> {
        for identifier in [table, time_column]
            .into_iter()
            .chain(columns.iter().map(String::as_str))
            .chain(filters.iter().map(|(column, _)| column.as_str()))
        {
            quote_identifier(identifier)?;
        }
        Ok(Self {
            connection: resolve_connection(config, connection)?,
            table,
            time_column,
            filters,
        })
    }

    pub fn load(
        self,
        columns: &[String],
        window: TimeWindow,
        query_interval: Option<Duration>,
        on_query_success: &mut impl FnMut(QueryWindowSuccess) -> io::Result<()>,
    ) -> Result<Vec<RawSeries>, DatabaseError> {
        let spec = QuerySpec {
            table: self.table,
            time_column: self.time_column,
            value_columns: columns,
            filters: &self.filters,
        };
        let observations = match self.connection.provider {
            Provider::Unsupported => Err(DatabaseError::UnsupportedProvider(
                "database feature disabled".into(),
            )),
            #[cfg(feature = "influx3")]
            Provider::InfluxDb3 => {
                let agent = ureq::Agent::new();
                query_in_windows(
                    window,
                    query_interval,
                    Duration::nanoseconds(1),
                    on_query_success,
                    |from, to| query_influx(&agent, &self.connection, &spec, from, to),
                )
            }
            #[cfg(feature = "postgres")]
            Provider::TimescaleDb => {
                let mut client = connect_timescale(self.connection.url)?;
                let statement = client
                    .prepare(&build_timescale_query(&spec)?)
                    .map_err(|_| DatabaseError::Request)?;
                query_in_windows(
                    window,
                    query_interval,
                    Duration::microseconds(1),
                    on_query_success,
                    |from, to| query_timescale(&mut client, &statement, &spec, from, to),
                )
            }
        }?;
        Ok(observations_to_series(columns, observations))
    }
}

fn query_in_windows(
    (mut from, to): TimeWindow,
    query_interval: Option<Duration>,
    precision: Duration,
    mut on_query_success: impl FnMut(QueryWindowSuccess) -> io::Result<()>,
    mut query: impl FnMut(DateTime<Utc>, DateTime<Utc>) -> Result<Vec<Observation>, DatabaseError>,
) -> Result<Vec<Observation>, DatabaseError> {
    let total = to - from;
    let mut width = query_interval.map_or(total, |interval| interval.min(total));
    let mut reported_width = None;
    let mut observations = Vec::new();
    while from < to {
        let duration = width.min(to - from);
        let end = from + duration;
        let started = Instant::now();
        match query(from, end) {
            Ok(rows) => {
                if duration == width && reported_width != Some(width) {
                    on_query_success(QueryWindowSuccess {
                        from,
                        to: end,
                        elapsed: started.elapsed(),
                    })?;
                    reported_width = Some(width);
                }
                observations.extend(rows);
                from = end;
            }
            Err(DatabaseError::QueryFileLimit) => {
                width = duration / 2;
                if width < precision {
                    return Err(DatabaseError::QueryFileLimitAtMinimum { from, to: end });
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(observations)
}

fn observations_to_series(columns: &[String], observations: Vec<Observation>) -> Vec<RawSeries> {
    let mut series = columns
        .iter()
        .map(|name| RawSeries {
            name: name.clone(),
            observations: Vec::with_capacity(observations.len()),
        })
        .collect::<Vec<_>>();
    for observation in observations {
        for (index, value) in observation.values.into_iter().enumerate() {
            series[index].observations.push(RawObservation {
                time: TimePoint::Timestamp(observation.timestamp),
                value,
            });
        }
    }
    series
}

pub fn load_config(path: &Path) -> Result<ConfigFile, DatabaseError> {
    let contents = std::fs::read_to_string(path).map_err(|source| DatabaseError::ReadConfig {
        path: path.into(),
        source,
    })?;
    toml::from_str(&contents).map_err(|_| DatabaseError::ParseConfig(path.into()))
}

fn resolve_connection<'a>(
    config: &'a ConfigFile,
    name: &str,
) -> Result<Connection<'a>, DatabaseError> {
    if let Some(connection) = config.connections.iter().find(|item| item.name == name) {
        let provider = match connection.provider.to_ascii_lowercase().as_str() {
            #[cfg(feature = "influx3")]
            "influxdb3" | "influx" => Provider::InfluxDb3,
            #[cfg(not(feature = "influx3"))]
            "influxdb3" | "influx" => Provider::Unsupported,
            #[cfg(feature = "postgres")]
            "timescaledb" | "timescale" | "postgres" | "postgresql" => Provider::TimescaleDb,
            #[cfg(not(feature = "postgres"))]
            "timescaledb" | "timescale" | "postgres" | "postgresql" => Provider::Unsupported,
            _ => Provider::Unsupported,
        };
        if provider == Provider::Unsupported {
            return Err(DatabaseError::UnsupportedProvider(connection.name.clone()));
        }
        #[cfg(feature = "influx3")]
        let database = connection.database.as_deref().unwrap_or_default();
        if connection.url.trim().is_empty() {
            return Err(DatabaseError::IncompleteConnection);
        }
        #[cfg(feature = "influx3")]
        if provider == Provider::InfluxDb3 && database.trim().is_empty() {
            return Err(DatabaseError::IncompleteConnection);
        }
        return Ok(Connection {
            provider,
            url: &connection.url,
            #[cfg(feature = "influx3")]
            database,
            #[cfg(feature = "influx3")]
            token: connection.token.as_deref(),
        });
    }

    #[cfg(feature = "influx3")]
    if name == "influx" {
        if let (Some(url), Some(database)) =
            (config.influx_url.as_deref(), config.influx_db.as_deref())
        {
            if url.trim().is_empty() || database.trim().is_empty() {
                return Err(DatabaseError::IncompleteConnection);
            }
            return Ok(Connection {
                provider: Provider::InfluxDb3,
                url,
                database,
                token: config.influx_token.as_deref(),
            });
        }
    }
    Err(DatabaseError::UnknownConnection(name.to_owned()))
}

fn quote_identifier(identifier: &str) -> Result<String, DatabaseError> {
    let parts = identifier.split('.').collect::<Vec<_>>();
    if parts
        .iter()
        .any(|part| part.trim().is_empty() || part.contains('\0'))
    {
        return Err(DatabaseError::InvalidIdentifier(identifier.to_owned()));
    }
    Ok(parts
        .into_iter()
        .map(|part| format!("\"{}\"", part.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join("."))
}

#[cfg(any(feature = "influx3", feature = "postgres"))]
fn build_select(spec: &QuerySpec<'_>, provider: Provider) -> Result<String, DatabaseError> {
    let table = quote_identifier(spec.table)?;
    let time = quote_identifier(spec.time_column)?;
    let time = match provider {
        Provider::Unsupported => {
            return Err(DatabaseError::UnsupportedProvider(
                "database feature disabled".into(),
            ));
        }
        #[cfg(feature = "influx3")]
        Provider::InfluxDb3 => time,
        #[cfg(feature = "postgres")]
        Provider::TimescaleDb => format!("CAST({time} AS TEXT)"),
    };
    let mut fields = vec![format!("{time} AS \"__timestral_time\"")];
    for (index, column) in spec.value_columns.iter().enumerate() {
        let column = quote_identifier(column)?;
        let column = match provider {
            Provider::Unsupported => {
                return Err(DatabaseError::UnsupportedProvider(
                    "database feature disabled".into(),
                ));
            }
            #[cfg(feature = "influx3")]
            Provider::InfluxDb3 => column,
            #[cfg(feature = "postgres")]
            Provider::TimescaleDb => format!("CAST({column} AS DOUBLE PRECISION)"),
        };
        fields.push(format!("{column} AS \"__timestral_value_{index}\""));
    }
    Ok(format!("SELECT {} FROM {table}", fields.join(", ")))
}

#[cfg(feature = "influx3")]
fn build_influx_query(spec: &QuerySpec<'_>) -> Result<(String, serde_json::Value), DatabaseError> {
    let mut query = build_select(spec, Provider::InfluxDb3)?;
    let mut clauses = vec![
        format!("{} >= $from", quote_identifier(spec.time_column)?),
        format!("{} < $to", quote_identifier(spec.time_column)?),
    ];
    let mut parameters = serde_json::Map::new();
    for (index, (column, value)) in spec.filters.iter().enumerate() {
        clauses.push(format!(
            "CAST({} AS VARCHAR) = $filter_{index}",
            quote_identifier(column)?
        ));
        parameters.insert(
            format!("filter_{index}"),
            serde_json::Value::String(value.clone()),
        );
    }
    query.push_str(" WHERE ");
    query.push_str(&clauses.join(" AND "));
    query.push_str(&format!(
        " ORDER BY {} ASC",
        quote_identifier(spec.time_column)?
    ));
    parameters.insert("from".into(), serde_json::Value::String(String::new()));
    parameters.insert("to".into(), serde_json::Value::String(String::new()));
    Ok((query, serde_json::Value::Object(parameters)))
}

#[cfg(feature = "influx3")]
fn query_influx(
    agent: &ureq::Agent,
    connection: &Connection<'_>,
    spec: &QuerySpec<'_>,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<Vec<Observation>, DatabaseError> {
    let (query, mut params) = build_influx_query(spec)?;
    let params = params
        .as_object_mut()
        .ok_or(DatabaseError::InvalidResponse)?;
    params.insert("from".into(), serde_json::Value::String(from.to_rfc3339()));
    params.insert("to".into(), serde_json::Value::String(to.to_rfc3339()));
    let endpoint = format!("{}/api/v3/query_sql", connection.url.trim_end_matches('/'));
    let body = serde_json::json!({
        "db": connection.database,
        "q": query,
        "params": params,
        "format": "csv",
    });
    let mut request = agent
        .post(&endpoint)
        .set("Content-Type", "application/json");
    if let Some(token) = connection.token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = match request.send_json(body) {
        Ok(response) => response,
        Err(ureq::Error::Status(status, response)) => {
            return Err(if status == 500 && influx_file_limit(response) {
                DatabaseError::QueryFileLimit
            } else {
                DatabaseError::HttpStatus(status)
            });
        }
        Err(_) => return Err(DatabaseError::Request),
    };
    let body = response
        .into_string()
        .map_err(|_| DatabaseError::InvalidResponse)?;
    parse_influx_csv(&body, spec.value_columns.len())
}

#[cfg(feature = "influx3")]
fn influx_file_limit(response: ureq::Response) -> bool {
    let mut body = Vec::new();
    response
        .into_reader()
        .take(16 * 1024)
        .read_to_end(&mut body)
        .is_ok()
        && is_parquet_file_limit(&String::from_utf8_lossy(&body))
}

#[cfg(any(feature = "influx3", feature = "postgres"))]
fn is_parquet_file_limit(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        ("query would exceed file limit of ", "parquet files"),
        (
            "query would scan ",
            "parquet files, exceeding the file limit",
        ),
        (
            "query would process ",
            "parquet files, which exceeds the limit",
        ),
    ]
    .into_iter()
    .any(|(prefix, suffix)| {
        message
            .split_once(prefix)
            .and_then(|(_, tail)| tail.split_once(' '))
            .is_some_and(|(count, rest)| {
                !count.is_empty()
                    && count.bytes().all(|byte| byte.is_ascii_digit())
                    && rest.starts_with(suffix)
            })
    })
}

#[cfg(feature = "influx3")]
fn parse_influx_csv(body: &str, num_values: usize) -> Result<Vec<Observation>, DatabaseError> {
    let mut reader = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .from_reader(body.as_bytes());
    let headers = reader
        .headers()
        .map_err(|_| DatabaseError::InvalidResponse)?
        .clone();
    if headers.is_empty() {
        return Ok(Vec::new());
    }
    let time_index = headers
        .iter()
        .position(|header| header == "__timestral_time")
        .ok_or(DatabaseError::InvalidResponse)?;
    let value_indices = (0..num_values)
        .map(|index| {
            headers
                .iter()
                .position(|header| header == format!("__timestral_value_{index}"))
                .ok_or(DatabaseError::InvalidResponse)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut rows = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|_| DatabaseError::InvalidResponse)?;
        let timestamp =
            parse_db_timestamp(row.get(time_index).ok_or(DatabaseError::InvalidResponse)?)?;
        let values = value_indices
            .iter()
            .map(|index| {
                parse_optional_number(row.get(*index).ok_or(DatabaseError::InvalidResponse)?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(Observation { timestamp, values });
    }
    Ok(rows)
}

#[cfg(feature = "postgres")]
fn connect_timescale(url: &str) -> Result<postgres::Client, DatabaseError> {
    let tls = native_tls::TlsConnector::new().map_err(|_| DatabaseError::Connection)?;
    let connector = postgres_native_tls::MakeTlsConnector::new(tls);
    let mut client =
        postgres::Client::connect(url, connector).map_err(|_| DatabaseError::Connection)?;
    client
        .batch_execute("SET TIME ZONE 'UTC'")
        .map_err(|_| DatabaseError::Connection)?;
    Ok(client)
}

#[cfg(feature = "postgres")]
fn query_timescale(
    client: &mut postgres::Client,
    statement: &postgres::Statement,
    spec: &QuerySpec<'_>,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<Vec<Observation>, DatabaseError> {
    let params = timescale_parameters(&from, &to, spec.filters);
    let rows = client.query(statement, &params).map_err(|error| {
        if error.as_db_error().is_some_and(|error| {
            is_timescale_file_limit(error.code(), error.message())
                || error.detail().is_some_and(is_parquet_file_limit)
        }) {
            DatabaseError::QueryFileLimit
        } else {
            DatabaseError::Request
        }
    })?;
    let mut observations = Vec::with_capacity(rows.len());
    for row in rows {
        let timestamp = row
            .try_get::<_, String>(0)
            .map_err(|_| DatabaseError::InvalidResponse)?;
        let timestamp = parse_db_timestamp(&timestamp)?;
        let values = (0..spec.value_columns.len())
            .map(|index| {
                row.try_get::<_, Option<f64>>(index + 1)
                    .map_err(|_| DatabaseError::InvalidResponse)
                    .and_then(|value| match value {
                        Some(value) if value.is_finite() && (value as f32).is_finite() => {
                            Ok(Some(value as f32))
                        }
                        Some(_) => Err(DatabaseError::InvalidResponse),
                        None => Ok(None),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        observations.push(Observation { timestamp, values });
    }
    Ok(observations)
}

#[cfg(feature = "postgres")]
fn is_timescale_file_limit(code: &SqlState, message: &str) -> bool {
    (code == &SqlState::CONFIGURATION_LIMIT_EXCEEDED
        && message.starts_with("temporary file size exceeds temp_file_limit ("))
        || is_parquet_file_limit(message)
}

#[cfg(feature = "postgres")]
fn timescale_parameters<'a>(
    from: &'a DateTime<Utc>,
    to: &'a DateTime<Utc>,
    filters: &'a [(String, String)],
) -> Vec<&'a (dyn ToSql + Sync)> {
    let mut params: Vec<&(dyn ToSql + Sync)> = vec![from, to];
    params.extend(
        filters
            .iter()
            .map(|(_, value)| value as &(dyn ToSql + Sync)),
    );
    params
}

#[cfg(feature = "postgres")]
fn build_timescale_query(spec: &QuerySpec<'_>) -> Result<String, DatabaseError> {
    let mut query = build_select(spec, Provider::TimescaleDb)?;
    let mut clauses = vec![
        format!("{} >= $1", quote_identifier(spec.time_column)?),
        format!("{} < $2", quote_identifier(spec.time_column)?),
    ];
    for (column, _) in spec.filters {
        let placeholder = clauses.len() + 1;
        clauses.push(format!(
            "CAST({} AS TEXT) = ${placeholder}",
            quote_identifier(column)?
        ));
    }
    query.push_str(" WHERE ");
    query.push_str(&clauses.join(" AND "));
    query.push_str(&format!(
        " ORDER BY {} ASC",
        quote_identifier(spec.time_column)?
    ));
    Ok(query)
}

fn parse_db_timestamp(value: &str) -> Result<DateTime<Utc>, DatabaseError> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return Ok(timestamp.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%d %H:%M:%S%.f%:z", "%Y-%m-%d %H:%M:%S%.f%z"] {
        if let Ok(timestamp) = DateTime::parse_from_str(value, format) {
            return Ok(timestamp.with_timezone(&Utc));
        }
    }
    for format in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(timestamp) = NaiveDateTime::parse_from_str(value, format) {
            return Ok(Utc.from_utc_datetime(&timestamp));
        }
    }
    Err(DatabaseError::InvalidResponse)
}

#[cfg(feature = "influx3")]
fn parse_optional_number(value: &str) -> Result<Option<f32>, DatabaseError> {
    if value.is_empty() || value.eq_ignore_ascii_case("null") {
        return Ok(None);
    }
    let value = value
        .parse::<f32>()
        .map_err(|_| DatabaseError::InvalidResponse)?;
    if !value.is_finite() {
        return Err(DatabaseError::InvalidResponse);
    }
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "influx3", feature = "postgres"))]
    use super::Provider;
    #[cfg(any(feature = "influx3", feature = "postgres"))]
    use super::is_parquet_file_limit;
    use super::{DatabaseError, QuerySpec, parse_db_timestamp, query_in_windows, quote_identifier};
    #[cfg(feature = "postgres")]
    use super::{SqlState, build_timescale_query, is_timescale_file_limit};
    #[cfg(feature = "influx3")]
    use super::{build_influx_query, parse_influx_csv};
    use chrono::Duration;
    #[cfg(feature = "influx3")]
    use std::io::{Read, Write};
    #[cfg(feature = "influx3")]
    use std::net::TcpListener;
    #[cfg(feature = "influx3")]
    use std::thread;

    #[test]
    #[cfg(feature = "influx3")]
    fn builds_quoted_sql_and_parameterized_influx_filters() {
        let columns = vec!["temperature".to_owned()];
        let filters = [("entity_id".to_owned(), "kitchen=west".to_owned())];
        let spec = QuerySpec {
            table: "home.measurements",
            time_column: "time",
            value_columns: &columns,
            filters: &filters,
        };
        let (query, params) = build_influx_query(&spec).unwrap();

        assert!(query.contains("FROM \"home\".\"measurements\""));
        assert!(query.contains("CAST(\"entity_id\" AS VARCHAR) = $filter_0"));
        assert_eq!(params["filter_0"], "kitchen=west");
        assert!(query.contains("$from"));
        assert!(query.contains("$to"));
    }

    #[test]
    #[cfg(feature = "postgres")]
    fn builds_a_parameterized_timescale_query_and_casts_values() {
        let columns = vec!["temperature".to_owned()];
        let filters = [("entity_id".to_owned(), "kitchen".to_owned())];
        let spec = QuerySpec {
            table: "public.measurements",
            time_column: "recorded_at",
            value_columns: &columns,
            filters: &filters,
        };
        let query = build_timescale_query(&spec).unwrap();

        assert!(query.contains("CAST(\"recorded_at\" AS TEXT)"));
        assert!(query.contains("CAST(\"temperature\" AS DOUBLE PRECISION)"));
        assert!(query.contains("CAST(\"entity_id\" AS TEXT) = $3"));
        assert!(query.contains("ORDER BY \"recorded_at\" ASC"));
        assert!(query.contains("\"recorded_at\" >= $1 AND \"recorded_at\" < $2"));
        let grid = crate::series::GridSpec::from_window(
            "2000-01-01T00:00:00Z",
            "2000-01-01T00:03:01Z",
            "1m",
        )
        .unwrap()
        .0;
        let (_, window) = crate::series::RasterOptions::for_input(
            Some(&grid),
            crate::series::Aggregation::default(),
            Some("30s"),
        )
        .unwrap();
        let (from, to) = window.unwrap();
        let params = super::timescale_parameters(&from, &to, &filters);
        use postgres::types::Type;
        assert_eq!(params.len(), 3);
        for ((parameter, kind), expected) in params
            .iter()
            .zip([Type::TIMESTAMPTZ, Type::TIMESTAMPTZ, Type::TEXT])
            .zip([
                &(-30_000_000_i64).to_be_bytes()[..],
                &180_000_000_i64.to_be_bytes(),
                b"kitchen",
            ])
        {
            let mut encoded = Default::default();
            parameter.to_sql_checked(&kind, &mut encoded).unwrap();
            assert_eq!(&encoded[..], expected);
        }
    }

    #[test]
    #[cfg(all(feature = "influx3", feature = "csv"))]
    fn prepares_equal_target_and_covariate_buffers_from_csv_and_influx() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let targets = [
                influx_exchange_response(
                    &listener,
                    500,
                    "Query would exceed file limit of 432 parquet files.",
                ),
                influx_exchange_response(
                    &listener,
                    500,
                    "Query would scan 500 Parquet files, exceeding the file limit.",
                ),
                influx_exchange(&listener, "2026-01-01T23:59:30Z,2\n"),
                influx_exchange(
                    &listener,
                    "2026-01-02T00:00:15Z,6\n2026-01-02T00:00:45Z,6\n",
                ),
                influx_exchange(&listener, "2026-01-02T00:01:10Z,\n2026-01-02T00:01:40Z,8\n"),
                influx_exchange(&listener, "2026-01-02T00:02:00Z,10\n"),
            ];
            let pv_before_one =
                "2026-01-01T23:59:30Z,10\n2026-01-02T00:00:15Z,20\n2026-01-02T00:00:45Z,20\n";
            let pv_after_one =
                "2026-01-02T00:01:10Z,null\n2026-01-02T00:01:40Z,40\n2026-01-02T00:02:00Z,50\n";
            let covariate = influx_exchange(&listener, &format!("{pv_before_one}{pv_after_one}"));
            let mixed_covariate = [
                influx_exchange_response(&listener, 200, ""),
                influx_exchange(&listener, pv_before_one),
            ];
            let live = [
                influx_exchange(
                    &listener,
                    "2026-01-01T23:59:30Z,2\n2026-01-02T00:00:15Z,6\n",
                ),
                influx_exchange(&listener, pv_before_one),
            ];
            (targets, covariate, mixed_covariate, live)
        });
        let config: super::ConfigFile = toml::from_str(&format!(
            r#"
influx_url = "http://{address}"
influx_db = "metrics"
influx_token = "test-token"

[[connections]]
name = "solar-db"
provider = "influxdb3"
url = "http://{address}"
database = "solar"
token = "test-token"

[covariates.pv]
connection = "solar-db"
table = "measurements"
filters = {{ entity_id = "pv" }}
fill = "linear"
max_hold = "2m"
"#
        ))
        .unwrap();
        let prepare = |input: Option<&str>, target: &str, covariate: &str, query_interval, mode| {
            let request = crate::ForecastConfig {
                input: input.map(std::path::PathBuf::from),
                horizon: crate::HorizonSpec::Duration(std::time::Duration::from_secs(120)),
                headers: input.is_some(),
                model_dir: None,
                variant: None,
                backend: None,
                aggregation: crate::Aggregation::TimeWeightedMean,
                values: vec![crate::parse_series_arg(target).unwrap()],
                covariates: vec![crate::parse_series_arg(covariate).unwrap()],
                source: crate::SourceOptions {
                    config_path: None,
                    connection: None,
                    time_column: Some("time".into()),
                    value_column: Some("value".into()),
                    from: Some("2026-01-02T00:00:00Z".into()),
                    to: Some(match mode {
                        crate::ForecastMode::Backtest => "2026-01-02T00:03:01Z".into(),
                        crate::ForecastMode::Live => "2026-01-02T00:01:00Z".into(),
                    }),
                    step: Some("1m".into()),
                    query_interval,
                    fill: None,
                    max_fill_gap: None,
                },
            };
            crate::prepare_inputs_with_config(&request, Some(&config), mode).unwrap()
        };
        let influx_target = "target={connection='influx',table='measurements',filters={entity_id='kitchen'},fill='linear'}";
        let database = prepare(
            None,
            influx_target,
            "pv",
            None,
            crate::ForecastMode::Backtest,
        );
        let csv = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../timestral-cli/tests/fixtures/irregular.csv"
        );
        let csv_inputs = |covariate, query_interval| {
            prepare(
                Some(csv),
                "target={value_column='target',time_column='time',fill='linear'}",
                covariate,
                query_interval,
                crate::ForecastMode::Backtest,
            )
        };
        let csv_result = csv_inputs("pv={value_column='pv',fill='linear',max_hold='2m'}", None);
        let chunked_csv_result = csv_inputs("pv", Some(Duration::seconds(90)));
        for result in [&csv_result, &chunked_csv_result] {
            assert_eq!(database.targets, result.targets);
            assert_eq!(database.past_covariates, result.past_covariates);
            assert_eq!(database.target_filled, result.target_filled);
            assert_eq!(database.target_actual, result.target_actual);
        }
        assert_eq!(database.targets.values, [5.0]);
        assert_eq!(database.past_covariates.as_ref().unwrap().values, [17.5]);

        let live = prepare(None, influx_target, "pv", None, crate::ForecastMode::Live);
        assert_eq!(database.targets, live.targets);
        assert_eq!(database.past_covariates, live.past_covariates);
        assert!(live.target_actual.is_empty());

        let (targets, covariate, mixed_covariate, live_queries) = server.join().unwrap();
        let assert_request = |request: &serde_json::Value, db, filter, from: &str, to: &str| {
            assert_eq!(request["db"], db);
            assert_eq!(request["params"]["filter_0"], filter);
            assert!(request["params"].get("filter_1").is_none());
            assert_eq!(
                request["params"]["from"],
                parse_db_timestamp(from).unwrap().to_rfc3339()
            );
            assert_eq!(
                request["params"]["to"],
                parse_db_timestamp(to).unwrap().to_rfc3339()
            );
            assert_eq!(request["format"], "csv");
            assert!(
                request["q"]
                    .as_str()
                    .unwrap()
                    .contains("\"time\" >= $from AND \"time\" < $to")
            );
            assert!(
                request["q"]
                    .as_str()
                    .unwrap()
                    .contains("\"value\" AS \"__timestral_value_0\"")
            );
            assert!(request["q"].as_str().unwrap().contains("$filter_0"));
        };
        let times = [
            "2026-01-01T23:58:00Z",
            "2026-01-01T23:59:00Z",
            "2026-01-01T23:59:30Z",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:01:00Z",
            "2026-01-02T00:02:00Z",
            "2026-01-02T00:02:30Z",
            "2026-01-02T00:03:00Z",
        ];
        for (request, (from, to)) in
            targets
                .iter()
                .zip([(1, 7), (1, 4), (1, 3), (3, 4), (4, 5), (5, 7)])
        {
            assert_request(request, "metrics", "kitchen", times[from], times[to]);
        }
        assert_request(&covariate, "solar", "pv", times[0], times[4]);
        assert_eq!(targets[0]["q"], covariate["q"]);
        for (request, (from, to)) in mixed_covariate.iter().zip([(0, 2), (2, 4)]) {
            assert_request(request, "solar", "pv", times[from], times[to]);
            assert_eq!(request["q"], covariate["q"]);
        }
        assert_request(&live_queries[0], "metrics", "kitchen", times[1], times[4]);
        assert_request(&live_queries[1], "solar", "pv", times[0], times[4]);
    }

    #[test]
    #[cfg(feature = "influx3")]
    fn does_not_retry_an_unrelated_influx_http_500() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            influx_exchange_response(&listener, 500, "Internal error containing test-token")
        });
        let config = toml::from_str(&format!(
            "influx_url = 'http://{address}'\ninflux_db = 'metrics'\ninflux_token = 'test-token'"
        ))
        .unwrap();
        let columns = ["value".to_owned()];
        let source =
            super::DatabaseSource::new(&config, "influx", "sensors", "time", &columns, vec![])
                .unwrap();
        let from = parse_db_timestamp("2026-01-02T00:00:00Z").unwrap();
        let error = source
            .load(
                &columns,
                (from, from + Duration::hours(2)),
                Some(Duration::hours(1)),
                &mut |_| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(error, DatabaseError::HttpStatus(500)));
        assert_eq!(error.to_string(), "Database returned HTTP status 500");
        assert_eq!(
            server.join().unwrap()["params"]["to"],
            "2026-01-02T01:00:00+00:00"
        );
    }

    #[test]
    fn a_later_file_limit_reduces_the_remaining_query_windows() {
        let start = parse_db_timestamp("2026-01-02T00:00:00Z").unwrap();
        let mut windows = Vec::new();
        let mut reported = Vec::new();
        let observations = query_in_windows(
            (start, start + Duration::seconds(9)),
            Some(Duration::seconds(4)),
            Duration::microseconds(1),
            |event| {
                reported.push((
                    (event.from - start).num_seconds(),
                    (event.to - start).num_seconds(),
                ));
                Ok(())
            },
            |from, to| {
                let offsets = ((from - start).num_seconds(), (to - start).num_seconds());
                windows.push(offsets);
                if offsets == (4, 8) {
                    Err(DatabaseError::QueryFileLimit)
                } else {
                    Ok(Vec::new())
                }
            },
        )
        .unwrap();
        assert!(observations.is_empty());
        assert_eq!(windows, [(0, 4), (4, 8), (4, 6), (6, 8), (8, 9)]);
        assert_eq!(reported, [(0, 4), (4, 6)]);
    }

    #[test]
    fn stops_splitting_at_the_database_timestamp_resolution() {
        let start = parse_db_timestamp("2026-01-02T00:00:00Z").unwrap();
        for precision in [Duration::nanoseconds(1), Duration::microseconds(1)] {
            let mut windows = Vec::new();
            let error = query_in_windows(
                (start, start + precision * 3),
                None,
                precision,
                |_| Ok(()),
                |from, to| {
                    windows.push((from, to));
                    Err(DatabaseError::QueryFileLimit)
                },
            );
            assert!(
                matches!(error, Err(DatabaseError::QueryFileLimitAtMinimum { from, to }) if from == start && (to - from) == precision * 3 / 2)
            );
            assert_eq!(
                windows,
                [
                    (start, start + precision * 3),
                    (start, start + precision * 3 / 2)
                ]
            );
        }
    }

    #[test]
    #[cfg(any(feature = "influx3", feature = "postgres"))]
    fn classifies_specific_database_file_limit_messages_only() {
        assert!(is_parquet_file_limit(
            "Error while planning query: Query would process 432 parquet files, which exceeds the limit of 100."
        ));
        assert!(!is_parquet_file_limit("Parquet files failed to load"));
        assert!(!is_parquet_file_limit(
            "Query would exceed file limit of 5 open descriptors"
        ));
    }

    #[test]
    #[cfg(feature = "postgres")]
    fn classifies_timescale_file_limit_messages_only() {
        let message = "temporary file size exceeds temp_file_limit (1024kB)";
        assert!(is_timescale_file_limit(
            &SqlState::CONFIGURATION_LIMIT_EXCEEDED,
            message
        ));
        assert!(!is_timescale_file_limit(
            &SqlState::CONFIGURATION_LIMIT_EXCEEDED,
            "another configuration limit exceeded"
        ));
        assert!(!is_timescale_file_limit(&SqlState::QUERY_CANCELED, message));
    }

    #[cfg(all(feature = "influx3", feature = "csv"))]
    fn influx_exchange(listener: &TcpListener, rows: &str) -> serde_json::Value {
        let response = format!("__timestral_time,__timestral_value_0\n{rows}");
        influx_exchange_response(listener, 200, &response)
    }

    #[cfg(feature = "influx3")]
    fn influx_exchange_response(
        listener: &TcpListener,
        status: u16,
        response: &str,
    ) -> serde_json::Value {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        let (header_end, content_length) = loop {
            let bytes_read = stream.read(&mut buffer).unwrap();
            assert_ne!(bytes_read, 0);
            request.extend_from_slice(&buffer[..bytes_read]);
            if let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .and_then(|length| length.parse::<usize>().ok())
                    })
                    .unwrap();
                if request.len() >= header_end + 4 + content_length {
                    break (header_end, content_length);
                }
            }
        };
        write!(
            stream,
            "HTTP/1.1 {status} Mock response\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.len(),
            response
        )
        .unwrap();
        let headers = String::from_utf8_lossy(&request[..header_end]);
        assert!(headers.starts_with("POST /api/v3/query_sql HTTP/1.1"));
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("authorization: bearer test-token")
        );
        let body = &request[header_end + 4..header_end + 4 + content_length];
        serde_json::from_slice(body).unwrap()
    }

    #[test]
    fn quotes_identifiers_and_rejects_empty_qualified_parts() {
        assert_eq!(
            quote_identifier("schema.table").unwrap(),
            "\"schema\".\"table\""
        );
        assert_eq!(quote_identifier("a\"b").unwrap(), "\"a\"\"b\"");
        assert!(matches!(
            quote_identifier("schema."),
            Err(DatabaseError::InvalidIdentifier(_))
        ));
    }

    #[test]
    #[cfg(feature = "influx3")]
    fn parses_steps_timestamps_and_influx_csv() {
        assert_eq!(
            parse_db_timestamp("2026-01-02T03:04:05Z")
                .unwrap()
                .to_rfc3339(),
            "2026-01-02T03:04:05+00:00"
        );
        let rows = parse_influx_csv(
            "#datatype,string,double\n__timestral_time,__timestral_value_0\n2026-01-02T03:04:05Z,4.5\n",
            1,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values, vec![Some(4.5)]);
    }

    #[test]
    #[cfg(all(feature = "influx3", feature = "postgres"))]
    fn resolves_legacy_and_named_toml_connections() {
        let config: super::ConfigFile = toml::from_str(
            "influx_url = 'http://localhost:8181'\ninflux_db = 'metrics'\ninflux_token = 'token'\n\n[[connections]]\nname = 'timescale'\nprovider = 'timescaledb'\nurl = 'postgres://localhost/metrics'\n",
        )
        .unwrap();
        let influx = super::resolve_connection(&config, "influx").unwrap();
        assert_eq!(influx.provider, Provider::InfluxDb3);
        assert_eq!(influx.database, "metrics");
        let timescale = super::resolve_connection(&config, "timescale").unwrap();
        assert_eq!(timescale.provider, Provider::TimescaleDb);
        assert_eq!(timescale.url, "postgres://localhost/metrics");
    }

    #[test]
    #[cfg(feature = "influx3")]
    fn database_source_rejects_unknown_connections_and_invalid_filters() {
        let config =
            toml::from_str("influx_url = 'http://localhost'\ninflux_db = 'metrics'").unwrap();
        assert!(matches!(
            super::DatabaseSource::new(&config, "other", "sensors", "time", &["value".into()], vec![]),
            Err(DatabaseError::UnknownConnection(name)) if name == "other"
        ));
        assert!(matches!(
            super::DatabaseSource::new(
                &config,
                "influx",
                "sensors",
                "time",
                &["value".into()],
                vec![("bad.".into(), "x".into())]
            ),
            Err(DatabaseError::InvalidIdentifier(_))
        ));
    }
}
