use anyhow::{Context, Result, ensure};
use chrono::{Duration, SecondsFormat};
use forecast_pipeline::Forecast;
use serde_json::{Value, json};
use std::path::Path;

use forecast_pipeline::{GridSpec, SeriesData, TimePoint};

pub fn write_html(
    path: &Path,
    target_names: &[String],
    context: &SeriesData,
    filled: &[bool],
    actual: &[Option<f32>],
    context_grid: &GridSpec,
    actual_grid: &GridSpec,
    forecast: &Forecast,
    plot_days: u64,
) -> Result<()> {
    let html = render_html(
        target_names,
        context,
        filled,
        actual,
        context_grid,
        actual_grid,
        forecast,
        plot_days,
    )?;
    std::fs::write(path, html)
        .with_context(|| format!("Failed to write plot to {}", path.display()))
}

fn render_html(
    target_names: &[String],
    context: &SeriesData,
    filled: &[bool],
    actual: &[Option<f32>],
    context_grid: &GridSpec,
    actual_grid: &GridSpec,
    forecast: &Forecast,
    plot_days: u64,
) -> Result<String> {
    let (batch_size, num_variates, horizon, _) = forecast.shape();
    ensure!(batch_size == 1, "Plotting requires a single forecast batch");
    ensure!(
        num_variates == context.num_variates,
        "Forecast and target series counts differ"
    );
    ensure!(
        target_names.len() == num_variates,
        "Target series names do not match the model output"
    );
    let expected_context_len = num_variates
        .checked_mul(context.context_length)
        .context("Target context shape is too large")?;
    let expected_actual_len = num_variates
        .checked_mul(horizon)
        .context("Actual series shape is too large")?;
    ensure!(
        context.values.len() == expected_context_len,
        "Target context has an invalid shape"
    );
    ensure!(
        filled.len() == context.values.len(),
        "Target fill mask has an invalid shape"
    );
    ensure!(
        actual.len() == expected_actual_len,
        "Actual series do not match the forecast horizon"
    );
    ensure!(horizon > 0, "Forecast horizon must be positive");
    ensure!(
        context_grid.coordinate_at(context.context_length)? == actual_grid.coordinate_at(0)?,
        "Context and actual grids do not share a cutoff"
    );

    let levels = forecast.quantile_levels();
    let missing_levels = [0.05_f32, 0.95_f32]
        .into_iter()
        .filter(|needed| {
            !levels
                .iter()
                .any(|available| (available - needed).abs() < 1e-5)
        })
        .collect::<Vec<_>>();
    ensure!(
        missing_levels.is_empty(),
        "Forecast plot requires quantiles 0.05 and 0.95; missing {missing_levels:?}, available {levels:?}"
    );
    let lower = forecast
        .quantile(0.05)
        .context("Forecast plot requires quantile 0.05")?;
    let upper = forecast
        .quantile(0.95)
        .context("Forecast plot requires quantile 0.95")?;
    let median = forecast.median();
    let context_x = coordinates(context_grid, context.context_length)?;
    let forecast_x = coordinates(actual_grid, horizon)?;
    let cutoff = point_value(&actual_grid.coordinate_at(0)?);
    let mut views = String::new();
    let mut options = String::new();

    for (variable, name) in target_names.iter().enumerate() {
        let context_start = variable * context.context_length;
        let actual_start = variable * horizon;
        let context_values = &context.values[context_start..context_start + context.context_length];
        let context_filled = &filled[context_start..context_start + context.context_length];
        let actual_values = &actual[actual_start..actual_start + horizon];
        let median_values = &median[actual_start..actual_start + horizon];
        let lower_values = &lower[actual_start..actual_start + horizon];
        let upper_values = &upper[actual_start..actual_start + horizon];
        let filled_x = context_x
            .iter()
            .zip(context_filled)
            .filter_map(|(x, filled)| filled.then_some(x.clone()))
            .collect::<Vec<_>>();
        let filled_values = context_values
            .iter()
            .zip(context_filled)
            .filter_map(|(value, filled)| filled.then_some(*value))
            .collect::<Vec<_>>();
        let plot_id = format!("series-plot-{variable}");
        let layout = layout(name, &cutoff, context_grid, actual_grid, plot_days)?;
        let figure = json!([
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Context",
                "x": context_x,
                "y": context_values,
                "line": {"color": "#355c7d", "width": 1.5},
                "connectgaps": false
            },
            {
                "type": "scatter",
                "mode": "markers",
                "name": "Filled context",
                "x": filled_x,
                "y": filled_values,
                "marker": {"color": "#e67e22", "size": 7, "symbol": "circle-open"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Actual",
                "x": forecast_x,
                "y": actual_values,
                "line": {"color": "#2c7a4b", "width": 1.5},
                "connectgaps": false
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "90% interval lower",
                "x": forecast_x,
                "y": lower_values,
                "line": {"color": "rgba(65, 105, 160, 0.15)", "width": 0},
                "hoverinfo": "skip",
                "showlegend": false
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Central 90% interval",
                "x": forecast_x,
                "y": upper_values,
                "line": {"color": "rgba(65, 105, 160, 0.15)", "width": 0},
                "fill": "tonexty",
                "fillcolor": "rgba(65, 105, 160, 0.22)"
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Median forecast",
                "x": forecast_x,
                "y": median_values,
                "line": {"color": "#4169a0", "width": 2},
                "connectgaps": false
            }
        ]);
        let config = json!({"responsive": true, "displaylogo": false});
        let html_json = format!(
            "Plotly.newPlot({},{},{},{});",
            safe_json(&json!(plot_id)),
            safe_json(&figure),
            safe_json(&layout),
            safe_json(&config)
        );
        options.push_str(&format!(
            "<option value=\"{plot_id}\">{}</option>",
            escape_html(name)
        ));
        views.push_str(&format!(
            "<section class=\"plot-view\" id=\"{plot_id}\"{}><div class=\"plot\"></div><script>{html_json}</script></section>",
            if variable == 0 { "" } else { " hidden" }
        ));
    }

    Ok(format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Timestral forecast backtest</title><style>body{{font:14px system-ui,sans-serif;margin:1.5rem;color:#222}}label{{margin-right:.5rem}}select{{font:inherit;padding:.25rem}}.plot-view{{width:100%;height:min(75vh,720px);min-height:360px}}.plot{{width:100%;height:100%}}</style>{}<script>document.addEventListener('DOMContentLoaded',()=>{{const selector=document.getElementById('series-selector');selector.addEventListener('change',()=>{{document.querySelectorAll('.plot-view').forEach(view=>{{view.hidden=view.id!==selector.value;if(!view.hidden)Plotly.Plots.resize(view.querySelector('.plot'));}});}});}});</script></head><body><label for=\"series-selector\">Target series:</label><select id=\"series-selector\">{options}</select>{views}</body></html>",
        plotly::Plot::offline_js_sources()
    ))
}

fn coordinates(grid: &GridSpec, length: usize) -> Result<Vec<Value>> {
    (0..length)
        .map(|index| grid.coordinate_at(index).map(|point| point_value(&point)))
        .collect()
}

fn point_value(point: &TimePoint) -> Value {
    match point {
        TimePoint::Timestamp(timestamp) => {
            json!(timestamp.to_rfc3339_opts(SecondsFormat::AutoSi, true))
        }
        TimePoint::Index(index) => json!(index),
    }
}

fn layout(
    name: &str,
    cutoff: &Value,
    context_grid: &GridSpec,
    actual_grid: &GridSpec,
    plot_days: u64,
) -> Result<Value> {
    let timed = matches!(context_grid.coordinate_at(0)?, TimePoint::Timestamp(_));
    let mut x_axis = json!({
        "title": if timed { "Time (UTC)" } else { "Step" },
        "type": if timed { "date" } else { "linear" }
    });
    if timed && plot_days > 0 {
        let days = i64::try_from(plot_days).context("--plot-days exceeds the supported range")?;
        let duration =
            Duration::try_days(days).context("--plot-days exceeds the supported duration")?;
        let cutoff_time = match actual_grid.coordinate_at(0)? {
            TimePoint::Timestamp(timestamp) => timestamp,
            TimePoint::Index(_) => unreachable!(),
        };
        let context_start = match context_grid.coordinate_at(0)? {
            TimePoint::Timestamp(timestamp) => timestamp,
            TimePoint::Index(_) => unreachable!(),
        };
        let display_start = cutoff_time
            .checked_sub_signed(duration)
            .context("--plot-days moves the plot outside the supported timestamp range")?
            .max(context_start);
        let (_, display_end) = actual_grid
            .time_bounds()
            .context("A timestamp grid is required for a time-range plot")?;
        x_axis["range"] = json!([
            display_start.to_rfc3339_opts(SecondsFormat::AutoSi, true),
            display_end.to_rfc3339_opts(SecondsFormat::AutoSi, true)
        ]);
    }
    Ok(json!({
        "title": {"text": name},
        "xaxis": x_axis,
        "yaxis": {"title": {"text": "Value"}},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.12},
        "margin": {"l": 65, "r": 20, "t": 75, "b": 55},
        "shapes": [{
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": cutoff,
            "x1": cutoff,
            "y0": 0,
            "y1": 1,
            "line": {"color": "#555", "width": 1, "dash": "dash"}
        }],
        "annotations": [{
            "x": cutoff,
            "y": 1,
            "xref": "x",
            "yref": "paper",
            "text": "Forecast start",
            "showarrow": false,
            "xanchor": "left",
            "yanchor": "bottom"
        }]
    }))
}

fn safe_json(value: &Value) -> String {
    serde_json::to_string(value)
        .expect("JSON values are serializable")
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::render_html;
    use forecast_pipeline::{Forecast, GridSpec, SeriesData};

    #[test]
    fn renders_a_self_contained_multi_series_backtest_plot() {
        let (grid, _) =
            GridSpec::from_window("2026-01-01T00:00:00Z", "2026-01-12T00:00:00Z", "1d").unwrap();
        let (context_grid, actual_grid) = grid.split_tail(2).unwrap();
        let forecast = Forecast::new(
            vec![
                8.0, 10.0, 10.0, 12.0, 12.0, 14.0, 18.0, 20.0, 20.0, 22.0, 22.0, 24.0,
            ],
            vec![0.05, 0.5, 0.95],
            1,
            2,
            2,
        )
        .unwrap();
        let context = SeriesData {
            values: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                70.0, 80.0, 90.0,
            ],
            num_variates: 2,
            context_length: 9,
        };
        let filled = [
            false, false, false, true, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false,
        ];
        let actual = [Some(9.0), None, Some(19.0), Some(21.0)];
        let names = ["load".to_owned(), "temperature".to_owned()];
        let html = render_html(
            &names,
            &context,
            &filled,
            &actual,
            &context_grid,
            &actual_grid,
            &forecast,
            7,
        )
        .unwrap();

        assert!(html.contains("Plotly.newPlot"));
        assert!(html.contains("Central 90% interval"));
        assert!(html.contains("Filled context"));
        assert!(html.contains("connectgaps"));
        assert!(html.contains("2026-01-03T00:00:00Z"));
        assert!(html.contains("2026-01-12T00:00:00Z"));
        assert!(html.contains("<option value=\"series-plot-1\">temperature</option>"));
        assert!(!html.contains("<script src=\"https://"));
    }

    #[test]
    fn renders_masked_context_gaps_as_null_values() {
        let (grid, _) =
            GridSpec::from_window("2026-01-01T00:00:00Z", "2026-01-04T00:00:00Z", "1d").unwrap();
        let (context_grid, actual_grid) = grid.split_tail(1).unwrap();
        let context = SeriesData {
            values: vec![1.0, f32::NAN],
            num_variates: 1,
            context_length: 2,
        };
        let forecast = Forecast::new(vec![1.0, 2.0, 3.0], vec![0.05, 0.5, 0.95], 1, 1, 1).unwrap();
        let html = render_html(
            &["load".to_owned()],
            &context,
            &[false, false],
            &[None],
            &context_grid,
            &actual_grid,
            &forecast,
            0,
        )
        .unwrap();

        assert!(html.contains("\"y\":[1.0,null]"));
        assert!(html.contains("connectgaps\":false"));
    }

    #[test]
    fn embeds_series_names_safely_and_requires_the_interval_quantiles() {
        let (grid, _) =
            GridSpec::from_window("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", "1d").unwrap();
        let (context_grid, actual_grid) = grid.split_tail(1).unwrap();
        let context = SeriesData {
            values: vec![1.0],
            num_variates: 1,
            context_length: 1,
        };
        let names = ["</script><img src=x onerror=alert(1)> & \"load\"".to_owned()];
        let forecast = Forecast::new(vec![2.0, 3.0], vec![0.1, 0.5], 1, 1, 1).unwrap();
        let error = render_html(
            &names,
            &context,
            &[false],
            &[Some(2.0)],
            &context_grid,
            &actual_grid,
            &forecast,
            7,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires quantiles 0.05 and 0.95")
        );

        let forecast = Forecast::new(vec![2.0, 3.0, 4.0], vec![0.05, 0.5, 0.95], 1, 1, 1).unwrap();
        let html = render_html(
            &names,
            &context,
            &[false],
            &[Some(2.0)],
            &context_grid,
            &actual_grid,
            &forecast,
            0,
        )
        .unwrap();
        assert!(
            html.contains(
                "&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt; &amp; &quot;load&quot;"
            )
        );
        assert!(html.contains("\\u003c/script\\u003e\\u003cimg"));
    }

    #[test]
    fn row_index_plots_keep_the_full_context_and_use_a_step_axis() {
        let grid = GridSpec::index(4).unwrap();
        let (context_grid, actual_grid) = grid.split_tail(1).unwrap();
        let context = SeriesData {
            values: vec![1.0, 2.0, 3.0],
            num_variates: 1,
            context_length: 3,
        };
        let forecast = Forecast::new(vec![2.0, 3.0, 4.0], vec![0.05, 0.5, 0.95], 1, 1, 1).unwrap();
        let html = render_html(
            &["load".to_owned()],
            &context,
            &[false, false, false],
            &[Some(4.0)],
            &context_grid,
            &actual_grid,
            &forecast,
            7,
        )
        .unwrap();
        assert!(html.contains("\"xaxis\":{\"title\":\"Step\",\"type\":\"linear\"}"));
    }
}
