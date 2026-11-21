//! Backend-independent traits and request types for time-series forecasting.

/// A validated row-major forecast context and its requested output dimensions.
#[derive(Clone, Copy, Debug)]
pub struct ForecastRequest<'a> {
    context: &'a [f32],
    batch_size: usize,
    num_variates: usize,
    context_length: usize,
    prediction_length: usize,
    point_output_len: usize,
}

impl<'a> ForecastRequest<'a> {
    /// Creates a request for context `(B, V, T)` and horizon `H`.
    pub fn new(
        context: &'a [f32],
        batch_size: usize,
        num_variates: usize,
        context_length: usize,
        prediction_length: usize,
    ) -> Result<Self, ForecastError> {
        if batch_size == 0 || num_variates == 0 || context_length == 0 || prediction_length == 0 {
            return Err(ForecastError::InvalidInput(
                "batch_size, num_variates, context_length, and prediction_length must be > 0"
                    .into(),
            ));
        }

        let expected_context_len =
            checked_buffer_len("context", &[batch_size, num_variates, context_length])?;
        if context.len() != expected_context_len {
            return Err(ForecastError::InvalidInput(format!(
                "context length {} does not match expected B*V*T = {}*{}*{} = {}",
                context.len(),
                batch_size,
                num_variates,
                context_length,
                expected_context_len
            )));
        }

        let point_output_len = checked_buffer_len(
            "point forecast",
            &[batch_size, num_variates, prediction_length],
        )?;

        Ok(Self {
            context,
            batch_size,
            num_variates,
            context_length,
            prediction_length,
            point_output_len,
        })
    }

    /// Returns the flat row-major `(B, V, T)` context.
    pub fn context(&self) -> &'a [f32] {
        self.context
    }

    /// Returns the batch size `B`.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the number of target variates `V`.
    pub fn num_variates(&self) -> usize {
        self.num_variates
    }

    /// Returns the context length `T`.
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// Returns the requested forecast horizon `H`.
    pub fn prediction_length(&self) -> usize {
        self.prediction_length
    }

    /// Returns the required flat row-major `(B, V, H)` point forecast length.
    pub fn point_output_len(&self) -> usize {
        self.point_output_len
    }

    /// Returns the required flat row-major `(B, V, H, Q)` quantile forecast length.
    pub fn quantile_output_len(&self, num_quantiles: usize) -> Result<usize, ForecastError> {
        if num_quantiles == 0 {
            return Err(ForecastError::InvalidInput(
                "num_quantiles must be > 0".into(),
            ));
        }
        checked_buffer_len(
            "quantile forecast",
            &[
                self.batch_size,
                self.num_variates,
                self.prediction_length,
                num_quantiles,
            ],
        )
    }

    /// Checks that a caller-provided point forecast buffer has the required length.
    pub fn validate_point_output(&self, output: &[f32]) -> Result<(), ForecastError> {
        validate_output_len("point forecast", self.point_output_len, output.len())
    }

    /// Checks that a caller-provided quantile forecast buffer has the required length.
    pub fn validate_quantile_output(
        &self,
        num_quantiles: usize,
        output: &[f32],
    ) -> Result<(), ForecastError> {
        let expected = self.quantile_output_len(num_quantiles)?;
        validate_output_len("quantile forecast", expected, output.len())
    }
}

/// An invalid forecast request or output buffer.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ForecastError {
    /// An input dimension or context buffer is invalid.
    #[error("{0}")]
    InvalidInput(String),
    /// An output buffer does not have the required length.
    #[error("{kind} output length {actual} does not match expected {expected}")]
    OutputBufferLength {
        /// Description of the output data.
        kind: &'static str,
        /// Required number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
    },
}

/// Produces point forecasts in a caller-provided row-major `(B, V, H)` buffer.
pub trait Forecaster {
    /// Error returned by this model; it can represent shared request validation failures.
    type Error: From<ForecastError>;

    /// Writes point forecasts for `request` into `output`.
    ///
    /// Implementations must reject an output whose length differs from
    /// [`ForecastRequest::point_output_len`] before modifying it.
    fn forecast_into(
        &self,
        request: &ForecastRequest<'_>,
        output: &mut [f32],
    ) -> Result<(), Self::Error>;
}

/// Produces probabilistic forecasts in a caller-provided row-major `(B, V, H, Q)` buffer.
pub trait QuantileForecaster: Forecaster {
    /// Returns the quantile probability levels in output-buffer order.
    fn quantile_levels(&self) -> &[f32];

    /// Writes quantiles for `request` into `output`.
    ///
    /// Implementations must reject an output whose length differs from
    /// `B * V * H * self.quantile_levels().len()` before modifying it.
    fn forecast_quantiles_into(
        &self,
        request: &ForecastRequest<'_>,
        output: &mut [f32],
    ) -> Result<(), Self::Error>;
}

fn validate_output_len(
    kind: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), ForecastError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ForecastError::OutputBufferLength {
            kind,
            expected,
            actual,
        })
    }
}

fn checked_buffer_len(kind: &str, dimensions: &[usize]) -> Result<usize, ForecastError> {
    dimensions
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .filter(|&len| {
            len.checked_mul(std::mem::size_of::<f32>())
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| {
            ForecastError::InvalidInput(format!("{kind} dimensions exceed buffer limits"))
        })
}

#[cfg(test)]
mod tests {
    use super::{ForecastError, ForecastRequest};
    use std::error::Error;

    #[test]
    fn errors_implement_std_error_and_preserve_display_messages() {
        let invalid_input = ForecastError::InvalidInput("invalid dimensions".into());
        assert_eq!(invalid_input.to_string(), "invalid dimensions");
        assert!(Error::source(&invalid_input).is_none());

        let output_length = ForecastError::OutputBufferLength {
            kind: "point forecast",
            expected: 4,
            actual: 3,
        };
        assert_eq!(
            output_length.to_string(),
            "point forecast output length 3 does not match expected 4"
        );
        assert!(Error::source(&output_length).is_none());
    }

    #[test]
    fn validates_context_shape_and_output_lengths() {
        let request = ForecastRequest::new(&[1.0; 24], 2, 3, 4, 5).unwrap();
        assert_eq!(request.point_output_len(), 30);
        assert_eq!(request.quantile_output_len(7).unwrap(), 210);
        request.validate_point_output(&[0.0; 30]).unwrap();
        request.validate_quantile_output(7, &[0.0; 210]).unwrap();
    }

    #[test]
    fn rejects_zero_dimensions_and_incorrect_context_length() {
        assert!(matches!(
            ForecastRequest::new(&[], 0, 1, 1, 1),
            Err(ForecastError::InvalidInput(_))
        ));
        assert!(matches!(
            ForecastRequest::new(&[0.0], 1, 1, 2, 1),
            Err(ForecastError::InvalidInput(_))
        ));
    }

    #[test]
    fn rejects_overflowing_dimensions_and_wrong_output_lengths() {
        assert!(matches!(
            ForecastRequest::new(&[], usize::MAX, 2, 1, 1),
            Err(ForecastError::InvalidInput(_))
        ));

        let request = ForecastRequest::new(&[1.0], 1, 1, 1, 1).unwrap();
        assert_eq!(
            request.validate_point_output(&[]),
            Err(ForecastError::OutputBufferLength {
                kind: "point forecast",
                expected: 1,
                actual: 0,
            })
        );
        assert!(matches!(
            request.quantile_output_len(usize::MAX),
            Err(ForecastError::InvalidInput(_))
        ));
    }
}
