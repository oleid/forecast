# timestral-cli

Command-line inference for the `burn-timestral` library. The CLI accepts CSV input, InfluxDB 3, or TimescaleDB and saves a self-contained interactive HTML backtest plot.

## CSV

```sh
cargo run --package timestral-cli --release -- series.csv 48 --headers --variant 5M
cargo run --package timestral-cli --release --no-default-features -- series.csv 48 --headers --model-dir /path/to/checkpoint
cargo run --package timestral-cli --release --features wgpu -- series.csv 48 --backend wgpu --variant 5M
```

For a numeric CSV without timestamps, each column is a target series and each row is one time step. Use `--headers` only when the file has a header row; this preserves the existing row-index input behavior. `HORIZON` reserves the final rows for backtesting, so at least one preceding row must remain as context. To select a target column explicitly, use repeatable `--value` with the same series syntax used for database sources. For timestamped CSV, `--time-column` supplies a default time column, while `--value-column` supplies a default for selected series that omit `value_column`. A timestamped file uses the same half-open time window and grid as a database source:

```sh
cargo run --package timestral-cli --release -- series.csv 24 --headers \
  --time-column timestamp --value-column temperature \
  --value 'temperature={}' \
  --from 2026-01-01T00:00:00Z --to 2026-01-04T00:00:00Z --step 1h
```

Timestamp fields must be RFC 3339 and include a timezone. Select columns by header name, or by 1-based position when the CSV has no header. Empty value fields are treated as missing observations. When `--value` is omitted for CSV, all value columns are targets; selected covariate columns from that same file are excluded from this implicit target list. If neither `--variant` nor `--model-dir` is specified, the CLI loads 209M. A local directory must contain the original `config.json` and `model.safetensors`; its configuration determines the variant.

Without `--backend`, the CLI uses Burn's `Device::default()`: Flex in a standard build, WGPU when built with `--features wgpu`, unless Burn's `BURN_DEVICE` environment variable overrides the default. Use `--backend cpu` to force Flex or `--backend wgpu` to explicitly initialize WGPU.

## Databases

Database credentials are read from a TOML file, not command-line arguments. Keep that file private because it can contain access tokens or passwords. The original Influx settings are supported directly:

```toml
influx_url = "http://localhost:8181"
influx_db = "homeassistant"
influx_token = "replace-with-a-private-token"
```

For multiple connections, use named entries. Both providers can appear in the same file:

```toml
[[connections]]
name = "influx"
provider = "influxdb3"
url = "http://localhost:8181"
database = "homeassistant"
token = "replace-with-a-private-token"

[[connections]]
name = "timescale"
provider = "timescaledb"
url = "postgresql://user:password@localhost:5432/homeassistant"
```

`--connection <name>` supplies the default connection for database series. Each series can set its own `connection` instead. With the legacy `influx_url` / `influx_db` settings, the connection name is `influx`. A TimescaleDB connection uses a PostgreSQL connection URL. The CLI does not write database data; it issues `SELECT` queries and sets the TimescaleDB session time zone to UTC for consistent timestamp handling.

For database input, pass only a positive horizon as a positional argument; do not pass a `db` marker or CSV path. The horizon accepts either a positive number of grid steps or a duration: with `--step 1h`, `96` and `96h` are equivalent; with `--step 15m`, `72h` is 288 steps. A duration must be an exact multiple of `--step` and is supported for database and timestamped CSV sources. For CSV without timestamps, continue to use a step count because each row is one step. The horizon is reserved from the end of the selected window, so at least one earlier step must remain for model context. Timestamps must be RFC 3339 and include a timezone (`Z` means UTC). When a config path starts with `~`, leave the tilde unquoted or use `"$HOME/..."`; quoting `~` prevents the shell from expanding it.

Use repeatable `--value` definitions for targets. A definition contains the table and optional exact-match filters; timestamp and value columns can use the global defaults or be set per series. The same format is used for `--covariate`:

```sh
cargo run --package timestral-cli --release -- 48 \
  --config "$HOME/src/solar-brain/tools/config.toml" --connection influx \
  --time-column time --value-column value \
  --value 'temperature={table="measurements",filters={entity_id="kitchen_temperature"}}' \
  --value 'humidity={table="measurements",filters={entity_id="kitchen_humidity"}}' \
  --from 2026-01-01T00:00:00Z --to 2026-01-02T00:03:00Z --step 5m \
  --fill linear --max-fill-gap 30m
```

Use the same options with `--connection timescale` to query TimescaleDB. SQL identifiers are quoted; filter values are bound as query parameters. Global `--connection` is the default for all database series; a series definition can select another connection. `--time-column` and `--value-column` are shared defaults for every target and covariate, and explicit `time_column` or `value_column` fields take precedence. `--table` and `--filter` are defined per series; the old `--value-columns` option is replaced by repeated `--value` definitions.

Each database target or covariate starts with its full query window, including any `max_hold` lookback. If the database reports a recognized file limit, the CLI halves the window at the same start time and keeps the smaller interval for the rest of that source, halving again if needed. InfluxDB 3's Parquet file limit and TimescaleDB's PostgreSQL `temp_file_limit` are recognized; generic HTTP 500 errors and timeouts are not retried. Optionally set `--query-interval 1d` to cap the initial interval for all database sources in the invocation. This does not change `--step`: the CLI combines the raw rows from consecutive half-open query windows before aggregation or gap filling.

## Grid and missing values

The requested window is `[--from, --to)`. Its effective length is rounded down to whole `--step` intervals, anchored at `--from`; an incomplete trailing interval is discarded with an `INFO` message on stderr. The final `HORIZON` intervals are the backtest window, and the preceding intervals form the model context. A window must contain at least one full context step before the horizon. Supported durations are positive integers in `ms`, `s`, `m`, `h`, or `d`; a duration such as `90m` is rejected with `--step 1h` rather than rounded.

`time-weighted-mean` is the default aggregation for all sources. Each value is treated as constant until the next observation, but no longer than that series' `max_hold`; by default, `max_hold` equals one `--step`. Set it in the series definition, for example `max_hold="14d"`. An interval produces a mean only when it is fully covered. For example, with `--step 5m`, readings at `00:00`, `00:01`, and `00:04` contribute for one, three, and one minute respectively; a reading at `00:05` starts the next interval. Database queries and timestamped CSV processing can use observations from up to `max_hold` before `--from` to cover the beginning of the first interval. The backtest context is cut off before aggregation and filling, so observations from the held-out horizon cannot affect model inputs. Use `--aggregation last` to select the last observation in each interval instead.

`--fill hold` and `--fill linear` set a global fill default for targets and covariates. Without `--fill`, nothing is filled: every gap remains missing, is warned about, and is masked for the model. A series can override a global fill with `fill = "hold"`, `fill = "linear"`, or `fill = "none"` (disable filling). `--max-fill-gap` limits the length of each contiguous gap that an explicitly selected fill rule may fill; it is unlimited by default. For timestamped sources, use a duration such as `2h`; for row-indexed CSV, use a positive number of steps such as `8`. The limit itself is inclusive. A longer gap produces a warning and remains missing in the model context and plot. Gaps a selected method cannot fill (for example, leading gaps with `hold` or boundary gaps with `linear`) are also warned about and remain masked, never an error. Warnings are summarized once per affected series with the number and total size of gaps, the longest gap's range, and counts by reason.

- `--fill hold` carries the previous aggregated value forward; a leading gap remains missing because there is no previous value.
- `--fill linear` interpolates between known aggregated values; leading or trailing gaps remain missing because they lack a value on both sides.

An empty CSV value or database `NULL` ends the previous observation's validity. With `last`, a missing last observation makes the whole interval missing. Equal timestamps are rejected as ambiguous. `max_hold` and `max-fill-gap` control different stages: the former limits raw-value validity before aggregation; the latter limits filling after aggregation. Move the old `--max-hold` flag and `--fill COLUMN=METHOD` rules into the matching series definitions as `max_hold` and `fill`.

## Backtest plot

Every run is a backtest; there is no live mode. The interactive, self-contained HTML output defaults to `timestral_forecast.html`; use `--out-plot path.html` to choose another path. The plot gives each target series its own selectable view and shows the context, marked filled context points, the forecast start, median, central 90% band, and actual values with missing intervals left as visible gaps. Plotly JavaScript is embedded, so the file works offline. On timestamped data, the initial view shows the last seven context days; use `--plot-days 0` to show the full context. For untimestamped CSV, the full context is shown on a step axis. Changing this display range never changes model inputs. The CLI reports successful database query-window sizes and the forecast start on stderr, then prints the plot path after writing it.

## Historical covariates

Repeat `--covariate` for as many historical numeric inputs as needed. `--value` and `--covariate` share the same parser and source definition; only the option determines whether a series is a target or a model input. Each argument is either a name under `[series]` in `--config` (the legacy `[covariates]` lookup remains accepted), or an inline TOML definition. Only target series appear in the forecast plot. Covariates use the target context grid and are supplied to the model in argument order; they do not include held-out or future values.

Use another column of the same CSV as a covariate:

```sh
timestral-cli signals.csv 24 --headers --value 'load={value_column="load"}' \
  --covariate 'pv={value_column="pv",fill="linear"}'
```

Without `--value`, all columns are targets except selected covariate columns from the same CSV. Untimestamped CSV sources align by data row index and cannot be combined with timestamped sources.

For a database target, a named covariate can use a separate filter on the same table. Add this to the TOML file:

```toml
[series.pv]
table = "measurements"
filters = { sensor_id = "pv" }
max_hold = "15m"
```

```sh
timestral-cli 24 --config sources.toml --connection influx \
  --time-column time --value-column value \
  --value 'load={table="measurements",filters={sensor_id="load"}}' \
  --from 2026-01-01T00:00:00Z --to 2026-01-02T00:00:00Z --step 5m --covariate pv
```

Without an explicit source, a CSV series uses the positional CSV and a database series uses the global `--connection`. A database series must define its own `table`; its time and value columns come from `--time-column` and `--value-column` unless set explicitly in that series. Database filters are **not** inherited. Set `connection` to select another database connection. All series inherit the global `--fill` and `--max-fill-gap` defaults unless the definition sets its own `fill` or `max_fill_gap`. Each series can set `aggregation = "last"` and `max_hold = "15m"`; the latter limits raw-value validity before aggregation and is not the filling limit.

Sources can be mixed as long as their time axes match. For example, combine a timestamped CSV target with a TimescaleDB covariate from the `timescale` connection defined above:

```sh
timestral-cli load.csv 24 --config sources.toml --headers \
  --value 'load={value_column="load",time_column="timestamp"}' \
  --from 2026-01-01T00:00:00Z --to 2026-01-02T00:00:00Z --step 5m \
  --covariate 'air={connection="timescale",table="weather",time_column="recorded_at",value_column="temperature"}'
```

Use `csv = "weather.csv"` instead of `connection` for a separate CSV. For a timestamped file with headers, also set `headers = true` and either a global `--time-column` or a per-series `time_column`. Relative CSV paths are interpreted from the working directory. A series using the positional CSV inherits its timestamp column from the global `--time-column` or a target definition for that file if it does not specify its own.

## Options

Run `cargo run --package timestral-cli -- --help` to see all options. CSV input uses `<INPUT> <HORIZON>`; database input uses only `<HORIZON>`, with `--connection <name>` as a default or `connection` in each series definition. A horizon can be a positive step count or, for timestamped sources, a duration such as `4h`; at `--step 5m`, both `48` and `4h` reserve the same backtest window. Database input additionally requires `--config`, at least one `--value`, `--from`, `--to`, and `--step`. `--out-plot`, `--plot-days`, `--backend`, `--model-dir`, and `--variant` work for both inputs.