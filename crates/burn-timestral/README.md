# burn-timestral

Native Burn inference for Mistral's Timestral time-series forecasting models: **5M, 28M, 123M and 209M**. The Rust build needs neither Python nor ONNX, and does not download any model files.

### CLI

The command-line application is a separate workspace package, [`timestral-cli`](../timestral-cli/README.md). It supports CSV files, InfluxDB 3, and TimescaleDB while this crate remains a library.

### Library

```rust
use burn::tensor::Device;
use burn_timestral::{TimestralForecaster, TimestralVariant};

let device = Device::default();
let forecaster = TimestralForecaster::load(TimestralVariant::M5, None, &device)?;
let forecast = forecaster.forecast_1d(&[1.0, 2.0, 3.0, 4.0], 2, None)?;
let median = forecast.median();
```

The public API is exposed from the crate root. Load a checkpoint with `TimestralForecaster::load` or `from_dir`, and configure a request with the `ForecastOptions` builders. To reuse loaded weights with a different context limit, clone the forecaster and call `with_max_context`. For batched univariate data, call `forecast` with `num_variates = 1`.

Use `Device::flex()` to select Flex explicitly. Enable the `wgpu` feature on `burn-timestral` to use WGPU; `Device::wgpu_options().init()?` initializes it and reports adapter errors.

For a backend-independent interface, import `ForecastRequest` and `Forecaster` from `burn_timestral`. The request validates the row-major `(B, V, T)` context and reports the `(B, V, H)` output-buffer length; `QuantileForecaster` provides the equivalent `(B, V, H, Q)` interface for probabilistic models. These APIs write into caller-provided slices, though model inference may still allocate temporary storage internally.

Forecast buffers are read-only: use `shape`, `predictions`, and `quantile_levels` for the flat `(B, V, H, Q)` layout. `Forecast::get` returns `None` for an out-of-range index. Loading and forecasting return `ModelError`, which distinguishes invalid inputs, configuration errors, incompatible checkpoints, and Hub failures. I/O and parsing errors retain their underlying causes.

With the default `hub` feature, `None` reuses the requested variant's pinned Hugging Face snapshot or downloads the missing files on first load. Downloads are synchronous. The cache location is selected in this order (unset or empty variables are skipped):

1. `HF_HUB_CACHE`, or the legacy `HUGGINGFACE_HUB_CACHE`, directly.
2. `$HF_HOME/hub`.
3. `$XDG_CACHE_HOME/huggingface/hub`.
4. `~/.cache/huggingface/hub`.

Set `HF_HUB_OFFLINE=1` to use the cache without network access. To disable Hub support entirely, set `default-features = false` on the dependency and pass `Some(dir)` to `load`, or use `TimestralForecaster::from_dir(dir, &device)`. Explicit local paths never access the network and never fall back to the Hub.

The old `.bpk` APIs were removed. Use the original checkpoint files instead; no conversion is needed. The weights are provided by Mistral under the terms in the respective [Hugging Face repository](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/LICENSE), not bundled with this crate.

### Tests

Regular `cargo test` runs without model files or network calls at runtime. GitHub Actions also runs the [reference parity tests](crates/burn-timestral/tests/fixtures/README.md) against all four original checkpoints. Python is only needed if you want to regenerate the reference forecasts.
