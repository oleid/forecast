use std::collections::{HashMap, HashSet};
use std::time::Duration as StdDuration;

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Duration, Utc};
use clap::ValueEnum;
use serde::Deserialize;

pub(crate) type TimeWindow = (DateTime<Utc>, DateTime<Utc>);

/// Coordinate of an observation on a timestamp or row-index grid.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TimePoint {
    /// Absolute timestamp in UTC.
    Timestamp(DateTime<Utc>),
    /// Zero-based row index.
    Index(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawObservation {
    pub time: TimePoint,
    pub value: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawSeries {
    pub name: String,
    pub observations: Vec<RawObservation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GridStep {
    Time(i64),
    Index(i64),
}

/// A regularly spaced sequence of timestamps or row indices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GridSpec {
    origin: TimePoint,
    step: GridStep,
    length: usize,
}

impl GridSpec {
    /// Builds a grid from RFC 3339 bounds and returns its effective exclusive end.
    /// A trailing partial interval is dropped.
    pub fn from_window(from: &str, to: &str, step: &str) -> Result<(Self, DateTime<Utc>)> {
        let from = DateTime::parse_from_rfc3339(from)
            .context("Invalid --from timestamp; expected RFC 3339 with a timezone")?
            .with_timezone(&Utc);
        let to = DateTime::parse_from_rfc3339(to)
            .context("Invalid --to timestamp; expected RFC 3339 with a timezone")?
            .with_timezone(&Utc);
        let step = parse_duration(step, "--step")?;
        let range_ns = to
            .signed_duration_since(from)
            .num_nanoseconds()
            .context("Invalid time window")?;
        let step_ns = i64::try_from(step.as_nanos()).context("Invalid --step duration")?;
        ensure!(range_ns > 0 && step_ns > 0, "Invalid time window");
        let length = usize::try_from(range_ns / step_ns).context("Invalid time window")?;
        ensure!(
            length > 0,
            "The time window must contain at least one full --step"
        );
        let effective_ns = i64::try_from(length)
            .ok()
            .and_then(|length| length.checked_mul(step_ns))
            .context("Invalid time window")?;
        let effective_to = from + Duration::nanoseconds(effective_ns);
        Ok((
            Self {
                origin: TimePoint::Timestamp(from),
                step: GridStep::Time(step_ns),
                length,
            },
            effective_to,
        ))
    }

    /// Builds a nonempty grid of zero-based row indices.
    pub fn index(length: usize) -> Result<Self> {
        ensure!(length > 0, "Series has no data rows");
        Ok(Self {
            origin: TimePoint::Index(0),
            step: GridStep::Index(1),
            length,
        })
    }

    /// Returns the number of grid steps.
    pub fn len(&self) -> usize {
        self.length
    }

    pub(crate) fn steps_for_duration(&self, duration: StdDuration) -> Result<usize> {
        let GridStep::Time(step_ns) = self.step else {
            anyhow::bail!(
                "A duration HORIZON requires timestamped series; use a step count for row-indexed CSV"
            );
        };
        let step_ns = u128::try_from(step_ns)?;
        let duration_ns = duration.as_nanos();
        ensure!(
            duration_ns % step_ns == 0,
            "HORIZON duration must be an exact multiple of --step"
        );
        let steps = usize::try_from(duration_ns / step_ns).context("HORIZON is too large")?;
        ensure!(steps > 0, "HORIZON must be positive");
        Ok(steps)
    }

    /// Splits the grid into a nonempty prefix and a tail of `tail_length` steps.
    pub fn split_tail(&self, tail_length: usize) -> Result<(Self, Self)> {
        ensure!(tail_length > 0, "HORIZON must be positive");
        ensure!(
            tail_length < self.length,
            "HORIZON must leave at least one context step"
        );
        let context_length = self.length - tail_length;
        let offset = self
            .step_size()
            .checked_mul(i128::try_from(context_length).context("Grid is too large")?)
            .context("Grid is too large")?;
        let cutoff = match (&self.origin, self.step) {
            (TimePoint::Timestamp(origin), GridStep::Time(_)) => {
                let offset = i64::try_from(offset).context("Grid is too large")?;
                TimePoint::Timestamp(
                    origin
                        .checked_add_signed(Duration::nanoseconds(offset))
                        .context("Grid is outside the supported timestamp range")?,
                )
            }
            (TimePoint::Index(origin), GridStep::Index(_)) => {
                let offset = i64::try_from(offset).context("Grid is too large")?;
                TimePoint::Index(
                    origin
                        .checked_add(offset)
                        .context("Grid is outside the supported index range")?,
                )
            }
            _ => unreachable!("grid origin and step use matching coordinate types"),
        };
        Ok((
            Self {
                origin: self.origin.clone(),
                step: self.step,
                length: context_length,
            },
            Self {
                origin: cutoff,
                step: self.step,
                length: tail_length,
            },
        ))
    }

    pub(crate) fn following(&self, length: usize) -> Result<Self> {
        ensure!(length > 0, "HORIZON must be positive");
        let grid = Self {
            origin: self.coordinate_at(self.length)?,
            step: self.step,
            length,
        };
        grid.coordinate_at(length)?;
        Ok(grid)
    }

    /// Returns the inclusive start and exclusive end, or `None` for a row-index grid.
    pub fn time_bounds(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        let (TimePoint::Timestamp(from), GridStep::Time(step_ns)) = (&self.origin, self.step)
        else {
            return None;
        };
        let total_ns = i64::try_from(self.length).ok()?.checked_mul(step_ns)?;
        Some((*from, *from + Duration::nanoseconds(total_ns)))
    }

    /// Returns the coordinate at the given zero-based position.
    pub fn coordinate_at(&self, index: usize) -> Result<TimePoint> {
        ensure!(index <= self.length, "Grid coordinate is outside the grid");
        let offset = self
            .step_size()
            .checked_mul(i128::try_from(index).context("Grid is too large")?)
            .context("Grid is too large")?;
        match (&self.origin, self.step) {
            (TimePoint::Timestamp(origin), GridStep::Time(_)) => {
                let offset = i64::try_from(offset).context("Grid is too large")?;
                Ok(TimePoint::Timestamp(
                    origin
                        .checked_add_signed(Duration::nanoseconds(offset))
                        .context("Grid is outside the supported timestamp range")?,
                ))
            }
            (TimePoint::Index(origin), GridStep::Index(_)) => {
                let offset = i64::try_from(offset).context("Grid is too large")?;
                Ok(TimePoint::Index(
                    origin
                        .checked_add(offset)
                        .context("Grid is outside the supported index range")?,
                ))
            }
            _ => unreachable!("grid origin and step use matching coordinate types"),
        }
    }

    fn query_start(&self, max_hold: StdDuration) -> Result<DateTime<Utc>> {
        let (from, _) = self
            .time_bounds()
            .context("A timestamp grid is required for database input")?;
        let max_hold_ns = i64::try_from(max_hold.as_nanos())
            .context("max_hold exceeds the supported duration")?;
        from.checked_sub_signed(Duration::nanoseconds(max_hold_ns))
            .context("max_hold moves the query start outside the supported timestamp range")
    }

    fn coordinate(&self, time: &TimePoint) -> Result<Option<i128>> {
        match (&self.origin, self.step, time) {
            (TimePoint::Timestamp(origin), GridStep::Time(_), TimePoint::Timestamp(time)) => {
                Ok(Some(i128::from(
                    time.signed_duration_since(*origin)
                        .num_nanoseconds()
                        .context("Timestamp is outside the supported range")?,
                )))
            }
            (TimePoint::Index(origin), GridStep::Index(_), TimePoint::Index(time)) => {
                Ok(Some(i128::from(*time) - i128::from(*origin)))
            }
            _ => Ok(None),
        }
    }

    fn step_size(&self) -> i128 {
        match self.step {
            GridStep::Time(step) | GridStep::Index(step) => i128::from(step),
        }
    }
}

/// Rule for filling missing grid values in historical inputs.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum FillMethod {
    #[default]
    #[serde(alias = "reject")]
    #[value(alias = "reject")]
    /// Leave missing values unfilled.
    None,
    /// Carry forward the last available value.
    Hold,
    /// Interpolate linearly between available values.
    Linear,
}

/// Rule for combining observations within a timestamp grid interval.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Aggregation {
    #[default]
    #[value(name = "time-weighted-mean")]
    /// Average values weighted by the time for which they are held.
    TimeWeightedMean,
    /// Use the last observation in the interval.
    Last,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RasterOptions {
    aggregation: Aggregation,
    max_hold: i128,
    max_hold_duration: Option<StdDuration>,
}

impl RasterOptions {
    #[cfg(test)]
    pub fn for_grid(
        grid: &GridSpec,
        aggregation: Aggregation,
        max_hold: Option<&str>,
    ) -> Result<Self> {
        Self::for_step(grid.step, aggregation, max_hold)
    }

    pub fn for_input(
        grid: Option<&GridSpec>,
        aggregation: Aggregation,
        max_hold: Option<&str>,
    ) -> Result<(Self, Option<TimeWindow>)> {
        let raster = Self::for_step(
            grid.map_or(GridStep::Index(1), |grid| grid.step),
            aggregation,
            max_hold,
        )?;
        let window = match grid {
            Some(grid) => raster.time_window(grid)?,
            None => None,
        };
        Ok((raster, window))
    }

    fn for_step(step: GridStep, aggregation: Aggregation, max_hold: Option<&str>) -> Result<Self> {
        let (max_hold, max_hold_duration) = match max_hold {
            Some(value) => {
                ensure!(
                    matches!(step, GridStep::Time(_)),
                    "max_hold requires a timestamped input"
                );
                let duration = parse_duration(value, "max_hold")?;
                let nanos = i128::try_from(duration.as_nanos())
                    .context("max_hold exceeds the supported duration")?;
                ensure!(nanos <= i128::from(i64::MAX), "max_hold is too large");
                (nanos, Some(duration))
            }
            None => match step {
                GridStep::Time(step_ns) => (
                    i128::from(step_ns),
                    Some(StdDuration::from_nanos(step_ns as u64)),
                ),
                GridStep::Index(step) => (i128::from(step), None),
            },
        };
        ensure!(max_hold > 0, "max_hold must be positive");
        Ok(Self {
            aggregation,
            max_hold,
            max_hold_duration,
        })
    }

    fn time_window(&self, grid: &GridSpec) -> Result<Option<TimeWindow>> {
        let Some((from, to)) = grid.time_bounds() else {
            return Ok(None);
        };
        let from = match (self.aggregation, self.max_hold_duration) {
            (Aggregation::TimeWeightedMean, Some(hold)) => grid.query_start(hold)?,
            _ => from,
        };
        Ok(Some((from, to)))
    }
}

/// Grid-aligned model data stored in variate-major order.
#[derive(Debug, PartialEq)]
pub struct SeriesData {
    /// Consecutive context values for each variate.
    pub values: Vec<f32>,
    /// Number of variates in the buffer.
    pub num_variates: usize,
    /// Number of context steps per variate.
    pub context_length: usize,
}

#[derive(Debug, PartialEq)]
pub struct RasterizedData {
    pub data: SeriesData,
    pub filled: Vec<bool>,
    pub unfilled_gaps: Vec<UnfilledGapSummary>,
}

#[derive(Debug, PartialEq)]
pub struct UnfilledGapSummary {
    pub series_name: String,
    pub gap_count: usize,
    pub total_steps: usize,
    pub longest_range: String,
    pub longest_steps: usize,
    pub reasons: Vec<GapReasonCount>,
}

#[derive(Debug, PartialEq)]
pub struct GapReasonCount {
    pub reason: String,
    pub gap_count: usize,
}

#[cfg(test)]
pub fn rasterize(
    series: &[RawSeries],
    grid: &GridSpec,
    fill_rules: &HashMap<String, FillMethod>,
    options: &RasterOptions,
) -> Result<SeriesData> {
    Ok(rasterize_with_mask(series, grid, fill_rules, options)?.data)
}

#[cfg(test)]
fn rasterize_with_mask(
    series: &[RawSeries],
    grid: &GridSpec,
    fill_rules: &HashMap<String, FillMethod>,
    options: &RasterOptions,
) -> Result<RasterizedData> {
    rasterize_with_gap_limits(series, grid, fill_rules, &HashMap::new(), options)
}

pub fn rasterize_with_gap_limits(
    series: &[RawSeries],
    grid: &GridSpec,
    fill_rules: &HashMap<String, FillMethod>,
    max_fill_gaps: &HashMap<String, usize>,
    options: &RasterOptions,
) -> Result<RasterizedData> {
    validate_series(series)?;
    let seen_names = series
        .iter()
        .map(|raw| raw.name.as_str())
        .collect::<HashSet<_>>();
    for name in fill_rules.keys() {
        ensure!(
            seen_names.contains(name.as_str()),
            "Fill rule references unknown series {name:?}"
        );
    }

    let capacity = series
        .len()
        .checked_mul(grid.length)
        .context("Grid is too large")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .context("Grid is too large")?;
    let mut filled = Vec::new();
    filled
        .try_reserve_exact(capacity)
        .context("Grid is too large")?;
    let mut unfilled_gaps = Vec::new();
    for raw in series {
        let mut grid_values = aggregate(raw, grid, options)?;
        let method = fill_rules.get(&raw.name).copied().unwrap_or_default();
        let (series_filled, gaps) = fill_missing(
            &mut grid_values,
            method,
            raw,
            grid,
            max_fill_gaps.get(&raw.name).copied(),
        )?;
        filled.extend(series_filled);
        if let Some(gaps) = gaps {
            unfilled_gaps.push(gaps);
        }
        values.extend(
            grid_values
                .into_iter()
                .map(|value| value.unwrap_or(f32::NAN)),
        );
    }

    Ok(RasterizedData {
        data: SeriesData {
            values,
            num_variates: series.len(),
            context_length: grid.length,
        },
        filled,
        unfilled_gaps,
    })
}

pub fn rasterize_optional(
    series: &[RawSeries],
    grid: &GridSpec,
    options: &RasterOptions,
) -> Result<Vec<Option<f32>>> {
    validate_series(series)?;
    let capacity = series
        .len()
        .checked_mul(grid.length)
        .context("Grid is too large")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .context("Grid is too large")?;
    for raw in series {
        values.extend(aggregate(raw, grid, options)?);
    }
    Ok(values)
}

fn validate_series(series: &[RawSeries]) -> Result<()> {
    ensure!(!series.is_empty(), "No series were selected");
    let mut seen_names = HashSet::new();
    for raw in series {
        ensure!(
            seen_names.insert(raw.name.as_str()),
            "Duplicate series name {:?}",
            raw.name
        );
    }
    Ok(())
}

fn aggregate(
    series: &RawSeries,
    grid: &GridSpec,
    options: &RasterOptions,
) -> Result<Vec<Option<f32>>> {
    match options.aggregation {
        Aggregation::TimeWeightedMean => aggregate_time_weighted(series, grid, options.max_hold),
        Aggregation::Last => aggregate_last(series, grid),
    }
}

fn sorted_coordinates<'a>(
    series: &'a RawSeries,
    grid: &GridSpec,
) -> Result<Vec<(i128, &'a RawObservation)>> {
    let mut observations = series
        .observations
        .iter()
        .map(|observation| {
            let coordinate = grid.coordinate(&observation.time)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "Series {:?} mixes absolute timestamps and row indices",
                    series.name
                )
            })?;
            Ok((coordinate, observation))
        })
        .collect::<Result<Vec<_>>>()?;
    observations.sort_by_key(|(coordinate, _)| *coordinate);
    Ok(observations)
}

fn aggregate_last(series: &RawSeries, grid: &GridSpec) -> Result<Vec<Option<f32>>> {
    let step_size = grid.step_size();
    let extent = step_size * grid.length as i128;
    let mut values = vec![None; grid.length];
    let mut last_coordinate = None;
    for (coordinate, observation) in sorted_coordinates(series, grid)? {
        if coordinate < 0 || coordinate >= extent {
            continue;
        }
        ensure!(
            last_coordinate != Some(coordinate),
            "Series {:?} has duplicate timestamp/index {:?}",
            series.name,
            observation.time
        );
        last_coordinate = Some(coordinate);
        let bucket =
            usize::try_from(coordinate / step_size).context("Observation is outside the grid")?;
        values[bucket] = observation.value;
    }
    Ok(values)
}

fn aggregate_time_weighted(
    series: &RawSeries,
    grid: &GridSpec,
    max_hold: i128,
) -> Result<Vec<Option<f32>>> {
    let step_size = grid.step_size();
    let extent = step_size * grid.length as i128;
    let observations = sorted_coordinates(series, grid)?;
    let mut sums = vec![0.0_f64; grid.length];
    let mut coverage = vec![0_i128; grid.length];

    for index in 0..observations.len() {
        let (start, observation) = observations[index];
        if start >= extent {
            break;
        }
        if start < -max_hold {
            continue;
        }
        ensure!(
            index == 0 || observations[index - 1].0 != start,
            "Series {:?} has duplicate timestamp/index {:?}",
            series.name,
            observation.time
        );
        let expiration = start
            .checked_add(max_hold)
            .context("--max-hold overflows the supported time range")?;
        let next = observations
            .get(index + 1)
            .map_or(expiration, |(time, _)| expiration.min(*time));
        let clipped_start = start.max(0);
        let clipped_end = next.min(extent);
        let Some(value) = observation.value else {
            continue;
        };
        if clipped_start >= clipped_end {
            continue;
        }

        let first_bucket = usize::try_from(clipped_start / step_size)
            .context("Observation is outside the grid")?;
        let last_bucket = usize::try_from((clipped_end - 1) / step_size)
            .context("Observation is outside the grid")?;
        for bucket in first_bucket..=last_bucket {
            let bucket_start = bucket as i128 * step_size;
            let overlap_start = clipped_start.max(bucket_start);
            let overlap_end = clipped_end.min(bucket_start + step_size);
            let overlap = overlap_end - overlap_start;
            if overlap > 0 {
                sums[bucket] += f64::from(value) * overlap as f64;
                coverage[bucket] += overlap;
            }
        }
    }

    Ok(sums
        .into_iter()
        .zip(coverage)
        .map(|(sum, covered)| (covered == step_size).then_some((sum / step_size as f64) as f32))
        .collect())
}

fn fill_missing(
    values: &mut [Option<f32>],
    method: FillMethod,
    series: &RawSeries,
    grid: &GridSpec,
    max_fill_gap: Option<usize>,
) -> Result<(Vec<bool>, Option<UnfilledGapSummary>)> {
    let mut filled = vec![false; values.len()];
    let mut gaps = None;
    let mut start = 0;
    while start < values.len() {
        if values[start].is_some() {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < values.len() && values[end].is_none() {
            end += 1;
        }
        let gap_length = end - start;
        if let Some(limit) = max_fill_gap
            && gap_length > limit
        {
            record_unfilled_gap(
                &mut gaps,
                series,
                grid,
                start,
                end,
                gap_length,
                &format!("exceeds --max-fill-gap ({limit} steps)"),
            )?;
            start = end;
            continue;
        }
        match method {
            FillMethod::None => {
                record_unfilled_gap(
                    &mut gaps,
                    series,
                    grid,
                    start,
                    end,
                    gap_length,
                    "no fill rule configured",
                )?;
            }
            FillMethod::Hold => {
                if let Some(previous) = start.checked_sub(1).and_then(|index| values[index]) {
                    for index in start..end {
                        values[index] = Some(previous);
                        filled[index] = true;
                    }
                } else {
                    record_unfilled_gap(
                        &mut gaps,
                        series,
                        grid,
                        start,
                        end,
                        gap_length,
                        "hold requires a preceding value",
                    )?;
                }
            }
            FillMethod::Linear => {
                let left_value = start.checked_sub(1).and_then(|index| values[index]);
                let right_value = values.get(end).copied().flatten();
                if let (Some(left_value), Some(right_value)) = (left_value, right_value) {
                    let span = (end - start + 1) as f32;
                    for index in start..end {
                        let ratio = (index - start + 1) as f32 / span;
                        values[index] = Some(left_value + (right_value - left_value) * ratio);
                        filled[index] = true;
                    }
                } else {
                    record_unfilled_gap(
                        &mut gaps,
                        series,
                        grid,
                        start,
                        end,
                        gap_length,
                        "linear interpolation requires values on both sides",
                    )?;
                }
            }
        }
        start = end;
    }
    Ok((filled, gaps))
}

fn record_unfilled_gap(
    summary: &mut Option<UnfilledGapSummary>,
    series: &RawSeries,
    grid: &GridSpec,
    start: usize,
    end: usize,
    length: usize,
    reason: &str,
) -> Result<()> {
    let range = describe_range(grid, start, end)?;
    let summary = summary.get_or_insert_with(|| UnfilledGapSummary {
        series_name: series.name.clone(),
        gap_count: 0,
        total_steps: 0,
        longest_range: range.clone(),
        longest_steps: 0,
        reasons: Vec::new(),
    });
    summary.gap_count += 1;
    summary.total_steps += length;
    if length > summary.longest_steps {
        summary.longest_range = range;
        summary.longest_steps = length;
    }
    if let Some(reason_count) = summary
        .reasons
        .iter_mut()
        .find(|reason_count| reason_count.reason == reason)
    {
        reason_count.gap_count += 1;
    } else {
        summary.reasons.push(GapReasonCount {
            reason: reason.to_owned(),
            gap_count: 1,
        });
    }
    Ok(())
}

fn describe_range(grid: &GridSpec, start: usize, end: usize) -> Result<String> {
    let start_point = grid.coordinate_at(start)?;
    let end_point = grid.coordinate_at(end)?;
    Ok(match (start_point, end_point) {
        (TimePoint::Timestamp(start), TimePoint::Timestamp(end)) => {
            format!("[{}, {})", start.to_rfc3339(), end.to_rfc3339())
        }
        (TimePoint::Index(start), TimePoint::Index(end)) => {
            format!("rows [{}..{})", start + 1, end + 1)
        }
        _ => unreachable!("grid coordinates use one coordinate type"),
    })
}

/// Parses a positive duration with a unit suffix (for example `30s` or `5m`).
/// `option` labels the input in error messages.
pub fn parse_duration(value: &str, option: &str) -> Result<StdDuration> {
    let (number, unit) = ["ms", "s", "m", "h", "d"]
        .into_iter()
        .find_map(|unit| value.strip_suffix(unit).map(|number| (number, unit)))
        .with_context(|| {
            format!("Invalid {option}; use a positive duration such as 30s, 5m, 1h, or 1d")
        })?;
    let amount = number
        .parse::<u64>()
        .with_context(|| format!("Invalid {option}; duration must be a positive integer"))?;
    ensure!(amount > 0, "Invalid {option}; duration must be positive");
    match unit {
        "ms" => Ok(StdDuration::from_millis(amount)),
        "s" => Ok(StdDuration::from_secs(amount)),
        "m" => Ok(StdDuration::from_secs(
            amount
                .checked_mul(60)
                .with_context(|| format!("Invalid {option} duration"))?,
        )),
        "h" => Ok(StdDuration::from_secs(
            amount
                .checked_mul(60 * 60)
                .with_context(|| format!("Invalid {option} duration"))?,
        )),
        "d" => Ok(StdDuration::from_secs(
            amount
                .checked_mul(24 * 60 * 60)
                .with_context(|| format!("Invalid {option} duration"))?,
        )),
        _ => unreachable!(),
    }
}

pub fn max_fill_gap_steps(value: &str, grid: &GridSpec) -> Result<usize> {
    match grid.step {
        GridStep::Time(step_ns) => {
            let duration = parse_duration(value, "--max-fill-gap")?;
            let steps = duration.as_nanos() / u128::try_from(step_ns)?;
            usize::try_from(steps).context("--max-fill-gap is too large")
        }
        GridStep::Index(_) => {
            let steps = value.parse::<usize>().context(
                "For row-indexed CSV, --max-fill-gap must be a positive number of steps",
            )?;
            ensure!(steps > 0, "--max-fill-gap must be positive");
            Ok(steps)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Aggregation, FillMethod, GridSpec, RasterOptions, RawObservation, RawSeries, TimePoint,
        max_fill_gap_steps, rasterize, rasterize_optional, rasterize_with_gap_limits,
        rasterize_with_mask,
    };
    use chrono::{DateTime, Utc};
    use std::{collections::HashMap, time::Duration as StdDuration};

    fn timestamp(value: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(value)
            .unwrap()
            .with_timezone(&Utc)
    }

    fn series(name: &str, observations: &[(&str, Option<f32>)]) -> RawSeries {
        RawSeries {
            name: name.into(),
            observations: observations
                .iter()
                .map(|(time, value)| RawObservation {
                    time: TimePoint::Timestamp(timestamp(time)),
                    value: *value,
                })
                .collect(),
        }
    }

    #[test]
    fn splits_timestamp_and_index_grids_at_the_requested_horizon() {
        let (timed, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T05:00:00Z", "1h").unwrap();
        let (context, actual) = timed.split_tail(2).unwrap();
        assert_eq!(
            context.time_bounds(),
            Some((
                timestamp("2026-01-02T00:00:00Z"),
                timestamp("2026-01-02T03:00:00Z"),
            ))
        );
        assert_eq!(
            actual.time_bounds(),
            Some((
                timestamp("2026-01-02T03:00:00Z"),
                timestamp("2026-01-02T05:00:00Z"),
            ))
        );

        let (context, actual) = GridSpec::index(5).unwrap().split_tail(2).unwrap();
        assert_eq!(context.origin, TimePoint::Index(0));
        assert_eq!(context.length, 3);
        assert_eq!(actual.origin, TimePoint::Index(3));
        assert_eq!(actual.length, 2);
    }

    #[test]
    fn converts_duration_horizons_to_exact_grid_steps() {
        let (hourly, _) =
            GridSpec::from_window("2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z", "1h").unwrap();
        assert_eq!(
            hourly
                .steps_for_duration(StdDuration::from_secs(72 * 60 * 60))
                .unwrap(),
            72
        );
        assert!(
            hourly
                .steps_for_duration(StdDuration::from_secs(90 * 60))
                .unwrap_err()
                .to_string()
                .contains("exact multiple of --step")
        );

        let (quarter_hourly, _) =
            GridSpec::from_window("2026-01-01T00:00:00Z", "2026-01-04T00:00:00Z", "15m").unwrap();
        assert_eq!(
            quarter_hourly
                .steps_for_duration(StdDuration::from_secs(72 * 60 * 60))
                .unwrap(),
            288
        );
        assert!(
            quarter_hourly
                .split_tail(
                    quarter_hourly
                        .steps_for_duration(StdDuration::from_secs(4 * 24 * 60 * 60))
                        .unwrap()
                )
                .unwrap_err()
                .to_string()
                .contains("at least one context step")
        );
        assert!(
            GridSpec::index(10)
                .unwrap()
                .steps_for_duration(StdDuration::from_secs(60))
                .unwrap_err()
                .to_string()
                .contains("requires timestamped series")
        );
    }

    #[test]
    fn rejects_a_horizon_that_leaves_no_context() {
        let grid = GridSpec::index(2).unwrap();
        assert!(
            grid.split_tail(0)
                .unwrap_err()
                .to_string()
                .contains("positive")
        );
        assert!(
            grid.split_tail(2)
                .unwrap_err()
                .to_string()
                .contains("at least one context step")
        );
    }

    #[test]
    fn keeps_actual_gaps_optional_and_marks_context_fill() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T04:00:00Z", "1h").unwrap();
        let (context, actual) = grid.split_tail(2).unwrap();
        let input = series(
            "target",
            &[
                ("2026-01-02T00:00:00Z", Some(1.0)),
                ("2026-01-02T01:00:00Z", None),
                ("2026-01-02T02:00:00Z", Some(3.0)),
                ("2026-01-02T03:00:00Z", None),
                ("2026-01-02T04:00:00Z", Some(99.0)),
            ],
        );
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let fills = HashMap::from([("target".into(), FillMethod::Hold)]);
        let context_values =
            rasterize_with_mask(&[input.clone()], &context, &fills, &options).unwrap();
        assert_eq!(context_values.data.values, [1.0, 1.0]);
        assert_eq!(context_values.filled, [false, true]);
        assert_eq!(
            rasterize_optional(&[input], &actual, &options).unwrap(),
            [Some(3.0), None]
        );
    }

    #[test]
    fn fills_gaps_at_the_limit_and_masks_longer_timestamp_gaps() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T09:00:00Z", "1h").unwrap();
        let input = series(
            "target",
            &[
                ("2026-01-02T00:00:00Z", Some(0.0)),
                ("2026-01-02T03:00:00Z", Some(3.0)),
                ("2026-01-02T04:00:00Z", Some(4.0)),
                ("2026-01-02T08:00:00Z", Some(8.0)),
            ],
        );
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let fills = HashMap::from([("target".into(), FillMethod::Linear)]);
        let limits = HashMap::from([("target".into(), max_fill_gap_steps("2h", &grid).unwrap())]);
        let prepared =
            rasterize_with_gap_limits(&[input], &grid, &fills, &limits, &options).unwrap();

        assert_eq!(prepared.data.values[..4], [0.0, 1.0, 2.0, 3.0]);
        assert!(prepared.data.values[4].is_finite());
        assert!(
            prepared.data.values[5..8]
                .iter()
                .all(|value| value.is_nan())
        );
        assert_eq!(prepared.data.values[8], 8.0);
        assert_eq!(
            prepared.filled,
            [false, true, true, false, false, false, false, false, false]
        );
        assert_eq!(prepared.unfilled_gaps.len(), 1);
        assert_eq!(prepared.unfilled_gaps[0].gap_count, 1);
        assert_eq!(prepared.unfilled_gaps[0].total_steps, 3);
        assert_eq!(prepared.unfilled_gaps[0].longest_steps, 3);
        assert!(
            prepared.unfilled_gaps[0].reasons[0]
                .reason
                .contains("max-fill-gap (2 steps)")
        );
    }

    #[test]
    fn parses_max_fill_gap_in_grid_units() {
        let (timed, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T05:00:00Z", "1h").unwrap();
        assert_eq!(max_fill_gap_steps("2h", &timed).unwrap(), 2);
        assert_eq!(max_fill_gap_steps("150m", &timed).unwrap(), 2);
        assert!(max_fill_gap_steps("2", &timed).is_err());

        let indexed = GridSpec::index(5).unwrap();
        assert_eq!(max_fill_gap_steps("3", &indexed).unwrap(), 3);
        assert!(max_fill_gap_steps("3h", &indexed).is_err());
        assert!(max_fill_gap_steps("0", &indexed).is_err());
    }

    #[test]
    fn masks_longer_gaps_on_row_indexed_grids() {
        let grid = GridSpec::index(9).unwrap();
        let input = RawSeries {
            name: "csv-series".into(),
            observations: [
                (0, Some(0.0)),
                (3, Some(3.0)),
                (4, Some(4.0)),
                (8, Some(8.0)),
            ]
            .into_iter()
            .map(|(index, value)| RawObservation {
                time: TimePoint::Index(index),
                value,
            })
            .collect(),
        };
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let fills = HashMap::from([("csv-series".into(), FillMethod::Hold)]);
        let limits = HashMap::from([("csv-series".into(), 2)]);
        let prepared =
            rasterize_with_gap_limits(&[input], &grid, &fills, &limits, &options).unwrap();

        assert_eq!(prepared.data.values[..4], [0.0, 0.0, 0.0, 3.0]);
        assert!(
            prepared.data.values[5..8]
                .iter()
                .all(|value| value.is_nan())
        );
        assert_eq!(
            prepared.filled,
            [false, true, true, false, false, false, false, false, false]
        );
        assert_eq!(prepared.unfilled_gaps.len(), 1);
        assert_eq!(prepared.unfilled_gaps[0].longest_range, "rows [6..9)");
    }

    #[test]
    fn summarizes_multiple_unfilled_gaps_per_series_and_omits_complete_series() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T08:00:00Z", "1h").unwrap();
        let incomplete = series(
            "incomplete",
            &[
                ("2026-01-02T00:00:00Z", Some(0.0)),
                ("2026-01-02T02:00:00Z", Some(2.0)),
                ("2026-01-02T03:00:00Z", Some(3.0)),
                ("2026-01-02T07:00:00Z", Some(7.0)),
            ],
        );
        let complete = series(
            "complete",
            &[
                ("2026-01-02T00:00:00Z", Some(0.0)),
                ("2026-01-02T01:00:00Z", Some(1.0)),
                ("2026-01-02T02:00:00Z", Some(2.0)),
                ("2026-01-02T03:00:00Z", Some(3.0)),
                ("2026-01-02T04:00:00Z", Some(4.0)),
                ("2026-01-02T05:00:00Z", Some(5.0)),
                ("2026-01-02T06:00:00Z", Some(6.0)),
                ("2026-01-02T07:00:00Z", Some(7.0)),
            ],
        );
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let limits = HashMap::from([("incomplete".into(), 2)]);
        let prepared = rasterize_with_gap_limits(
            &[incomplete, complete],
            &grid,
            &HashMap::new(),
            &limits,
            &options,
        )
        .unwrap();

        assert_eq!(prepared.unfilled_gaps.len(), 1);
        let summary = &prepared.unfilled_gaps[0];
        assert_eq!(summary.series_name, "incomplete");
        assert_eq!(summary.gap_count, 2);
        assert_eq!(summary.total_steps, 4);
        assert_eq!(summary.longest_steps, 3);
        assert_eq!(
            summary.longest_range,
            "[2026-01-02T04:00:00+00:00, 2026-01-02T07:00:00+00:00)"
        );
        assert_eq!(summary.reasons.len(), 2);
        assert_eq!(summary.reasons[0].reason, "no fill rule configured");
        assert_eq!(summary.reasons[0].gap_count, 1);
        assert!(
            summary.reasons[1]
                .reason
                .contains("exceeds --max-fill-gap (2 steps)")
        );
        assert_eq!(summary.reasons[1].gap_count, 1);
        assert!(
            prepared.data.values[8..]
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn context_linear_fill_cannot_use_an_actual_window_value_and_masks_the_gap() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T04:00:00Z", "1h").unwrap();
        let (context, _) = grid.split_tail(2).unwrap();
        let input = series(
            "target",
            &[
                ("2026-01-02T00:00:00Z", Some(1.0)),
                ("2026-01-02T01:00:00Z", None),
                ("2026-01-02T02:00:00Z", Some(3.0)),
                ("2026-01-02T03:00:00Z", Some(4.0)),
            ],
        );
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let fills = HashMap::from([("target".into(), FillMethod::Linear)]);
        let prepared = rasterize_with_mask(&[input], &context, &fills, &options).unwrap();
        assert_eq!(prepared.data.values[0], 1.0);
        assert!(prepared.data.values[1].is_nan());
        assert_eq!(prepared.filled, [false, false]);
        assert_eq!(prepared.unfilled_gaps.len(), 1);
        assert_eq!(prepared.unfilled_gaps[0].series_name, "target");
        assert_eq!(
            prepared.unfilled_gaps[0].reasons[0].reason,
            "linear interpolation requires values on both sides"
        );
    }

    #[test]
    fn hold_fill_masks_a_leading_gap_without_a_previous_value() {
        let grid = GridSpec::index(3).unwrap();
        let input = RawSeries {
            name: "target".into(),
            observations: [None, Some(2.0), None]
                .into_iter()
                .enumerate()
                .map(|(index, value)| RawObservation {
                    time: TimePoint::Index(index as i64),
                    value,
                })
                .collect(),
        };
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let fills = HashMap::from([("target".into(), FillMethod::Hold)]);
        let prepared = rasterize_with_mask(&[input], &grid, &fills, &options).unwrap();

        assert!(prepared.data.values[0].is_nan());
        assert_eq!(prepared.data.values[1..], [2.0, 2.0]);
        assert_eq!(prepared.filled, [false, false, true]);
        assert_eq!(prepared.unfilled_gaps.len(), 1);
        assert_eq!(
            prepared.unfilled_gaps[0].reasons[0].reason,
            "hold requires a preceding value"
        );
    }

    #[test]
    fn changing_actual_values_does_not_change_the_model_context() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T04:00:00Z", "1h").unwrap();
        let (context, _) = grid.split_tail(2).unwrap();
        let before_cutoff = [
            ("2026-01-02T00:00:00Z", Some(1.0)),
            ("2026-01-02T01:00:00Z", None),
        ];
        let first = series(
            "target",
            &[
                before_cutoff[0],
                before_cutoff[1],
                ("2026-01-02T02:00:00Z", Some(3.0)),
                ("2026-01-02T03:00:00Z", Some(4.0)),
            ],
        );
        let second = series(
            "target",
            &[
                before_cutoff[0],
                before_cutoff[1],
                ("2026-01-02T02:00:00Z", Some(300.0)),
                ("2026-01-02T03:00:00Z", Some(400.0)),
            ],
        );
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let fills = HashMap::from([("target".into(), FillMethod::Hold)]);
        let first = rasterize_with_mask(&[first], &context, &fills, &options).unwrap();
        let second = rasterize_with_mask(&[second], &context, &fills, &options).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn valid_raw_value_before_the_cutoff_can_cover_actual_steps() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T04:00:00Z", "1h").unwrap();
        let (_, actual) = grid.split_tail(2).unwrap();
        let input = series("target", &[("2026-01-02T01:00:00Z", Some(5.0))]);
        let options =
            RasterOptions::for_grid(&grid, Aggregation::TimeWeightedMean, Some("2h")).unwrap();
        assert_eq!(
            rasterize_optional(&[input], &actual, &options).unwrap(),
            [Some(5.0), None]
        );
    }

    #[test]
    fn last_rasterization_sorts_observations_and_honors_exclusive_end() {
        let (grid, effective_to) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:10:00Z", "5m").unwrap();
        assert_eq!(effective_to, timestamp("2026-01-02T00:10:00Z"));
        let input = series(
            "temperature",
            &[
                ("2026-01-02T00:10:00Z", Some(99.0)),
                ("2026-01-02T00:04:00Z", Some(4.0)),
                ("2026-01-02T00:01:00Z", Some(1.0)),
                ("2026-01-02T00:06:00Z", Some(6.0)),
            ],
        );
        let filled = HashMap::from([("temperature".into(), FillMethod::Hold)]);
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let result = rasterize(&[input], &grid, &filled, &options).unwrap();
        assert_eq!(result.values, vec![4.0, 6.0]);
    }

    #[test]
    fn last_value_can_be_missing_and_leading_gaps_are_masked() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:10:00Z", "5m").unwrap();
        let input = series(
            "pv",
            &[
                ("2026-01-02T00:01:00Z", Some(2.0)),
                ("2026-01-02T00:04:00Z", None),
            ],
        );
        let options = RasterOptions::for_grid(&grid, Aggregation::Last, None).unwrap();
        let prepared = rasterize_with_mask(&[input], &grid, &HashMap::new(), &options).unwrap();
        assert!(prepared.data.values.iter().all(|value| value.is_nan()));
        assert_eq!(prepared.unfilled_gaps.len(), 1);
        assert_eq!(prepared.unfilled_gaps[0].series_name, "pv");
        assert!(
            prepared.unfilled_gaps[0]
                .longest_range
                .contains("2026-01-02T00:00:00+00:00")
        );
    }

    #[test]
    fn rejects_duplicate_times_and_mixing_time_axes() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:05:00Z", "5m").unwrap();
        let duplicate = series(
            "x",
            &[
                ("2026-01-02T00:01:00Z", Some(1.0)),
                ("2026-01-02T01:01:00+01:00", Some(2.0)),
            ],
        );
        for aggregation in [Aggregation::Last, Aggregation::TimeWeightedMean] {
            let options = RasterOptions::for_grid(&grid, aggregation, None).unwrap();
            let error = rasterize(
                std::slice::from_ref(&duplicate),
                &grid,
                &HashMap::new(),
                &options,
            )
            .unwrap_err();
            assert!(error.to_string().contains("duplicate timestamp"));
        }
        let indexed = RawSeries {
            name: "index".into(),
            observations: vec![RawObservation {
                time: TimePoint::Index(0),
                value: Some(1.0),
            }],
        };
        let options = RasterOptions::for_grid(&grid, Aggregation::default(), None).unwrap();
        assert!(rasterize(&[indexed], &grid, &HashMap::new(), &options).is_err());
    }

    #[test]
    fn parses_per_series_fill_methods() {
        for (value, method) in [("linear", FillMethod::Linear), ("none", FillMethod::None)] {
            let spec: crate::series_config::SeriesSpec =
                toml::from_str(&format!("value_column = 'temperature'\nfill = '{value}'")).unwrap();
            assert_eq!(spec.fill, Some(method));
        }
        assert!(
            toml::from_str::<crate::series_config::SeriesSpec>(
                "value_column = 'temperature'\nfill = 'nearest'"
            )
            .is_err()
        );
    }

    #[test]
    fn preserves_untimed_csv_row_indices() {
        let grid = GridSpec::index(2).unwrap();
        let input = RawSeries {
            name: "1".into(),
            observations: vec![
                RawObservation {
                    time: TimePoint::Index(0),
                    value: Some(1.0),
                },
                RawObservation {
                    time: TimePoint::Index(1),
                    value: Some(2.0),
                },
            ],
        };
        let options = RasterOptions::for_grid(&grid, Aggregation::TimeWeightedMean, None).unwrap();
        let result = rasterize(&[input], &grid, &HashMap::new(), &options).unwrap();
        assert_eq!(result.values, vec![1.0, 2.0]);
    }

    #[test]
    fn time_weighted_mean_integrates_irregular_observations_and_preroll() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:05:00Z", "5m").unwrap();
        let input = series(
            "power",
            &[
                ("2026-01-01T23:59:00Z", Some(2.0)),
                ("2026-01-02T00:02:00Z", Some(8.0)),
            ],
        );
        let options =
            RasterOptions::for_grid(&grid, Aggregation::TimeWeightedMean, Some("5m")).unwrap();
        let result = rasterize(&[input], &grid, &HashMap::new(), &options).unwrap();
        assert!((result.values[0] - 5.6).abs() < 0.0001);
    }

    #[test]
    fn max_hold_can_cover_an_interval_exactly_but_partial_coverage_is_a_gap() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:05:00Z", "5m").unwrap();
        let options =
            RasterOptions::for_grid(&grid, Aggregation::TimeWeightedMean, Some("5m")).unwrap();
        let exact = series("x", &[("2026-01-02T00:00:00Z", Some(3.0))]);
        assert_eq!(
            rasterize(&[exact], &grid, &HashMap::new(), &options)
                .unwrap()
                .values,
            vec![3.0]
        );

        let partial = series("x", &[("2026-01-02T00:01:00Z", Some(3.0))]);
        let result = rasterize(&[partial], &grid, &HashMap::new(), &options).unwrap();
        assert!(result.values[0].is_nan());
    }

    #[test]
    fn a_missing_observation_ends_the_previous_value_and_fill_runs_after_aggregation() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:10:00Z", "5m").unwrap();
        let input = series(
            "x",
            &[
                ("2026-01-02T00:00:00Z", Some(2.0)),
                ("2026-01-02T00:05:00Z", None),
            ],
        );
        let options =
            RasterOptions::for_grid(&grid, Aggregation::TimeWeightedMean, Some("10m")).unwrap();
        let without_fill = rasterize(
            std::slice::from_ref(&input),
            &grid,
            &HashMap::new(),
            &options,
        )
        .unwrap();
        assert_eq!(without_fill.values[0], 2.0);
        assert!(without_fill.values[1].is_nan());
        let fills = HashMap::from([("x".into(), FillMethod::Hold)]);
        let result = rasterize(&[input], &grid, &fills, &options).unwrap();
        assert_eq!(result.values, vec![2.0, 2.0]);
    }

    #[test]
    fn query_preroll_is_bounded_by_max_hold() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:10:00Z", "5m").unwrap();
        assert_eq!(
            grid.query_start(std::time::Duration::from_secs(300))
                .unwrap(),
            timestamp("2026-01-01T23:55:00Z")
        );
    }

    #[test]
    fn linear_fill_masks_a_gap_without_a_right_value_before_the_exclusive_end() {
        let (grid, _) =
            GridSpec::from_window("2026-01-02T00:00:00Z", "2026-01-02T00:11:00Z", "5m").unwrap();
        let input = series(
            "x",
            &[
                ("2026-01-02T00:00:00Z", Some(2.0)),
                ("2026-01-02T00:10:00Z", Some(4.0)),
            ],
        );
        let options = RasterOptions::for_grid(&grid, Aggregation::default(), None).unwrap();
        let fills = HashMap::from([("x".into(), FillMethod::Linear)]);
        let prepared = rasterize_with_mask(&[input], &grid, &fills, &options).unwrap();
        assert_eq!(prepared.data.values[0], 2.0);
        assert!(prepared.data.values[1].is_nan());
        assert_eq!(prepared.unfilled_gaps.len(), 1);
        assert_eq!(
            prepared.unfilled_gaps[0].longest_range,
            "[2026-01-02T00:05:00+00:00, 2026-01-02T00:10:00+00:00)"
        );
    }
}
