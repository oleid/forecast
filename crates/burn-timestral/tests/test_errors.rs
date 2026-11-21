use std::error::Error as _;

use burn_timestral::{Forecast, ModelError, TimestralForecaster};

#[test]
fn rejects_forecast_without_median() {
    assert!(matches!(
        Forecast::new(vec![1.0], vec![0.9], 1, 1, 1),
        Err(ModelError::InvalidInput(_))
    ));
}

#[test]
fn rejects_forecast_dimensions_beyond_the_buffer_limit() {
    assert!(matches!(
        Forecast::new(vec![], vec![0.5], isize::MAX as usize, 1, 1),
        Err(ModelError::InvalidInput(message)) if message.contains("buffer limits")
    ));
}

#[test]
fn out_of_bounds_forecast_access_does_not_panic() {
    let forecast = Forecast::new(vec![1.0, 2.0], vec![0.5], 1, 1, 2).unwrap();
    for (b, v, h, q) in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 2, 0), (0, 0, 0, 1)] {
        assert_eq!(forecast.get(b, v, h, q), None);
    }
}

#[test]
fn configuration_io_and_json_errors_retain_their_sources() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    let device = Default::default();

    let missing = TimestralForecaster::from_dir(dir.path(), &device)
        .err()
        .unwrap();
    assert_eq!(
        missing
            .source()
            .and_then(|err| err.downcast_ref::<std::io::Error>())
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::NotFound)
    );
    std::fs::write(&path, "not JSON").unwrap();
    let malformed = TimestralForecaster::from_dir(dir.path(), &device)
        .err()
        .unwrap();
    assert!(
        malformed
            .source()
            .is_some_and(|err| err.is::<serde_json::Error>())
    );
    assert!(malformed.to_string().contains(&*path.to_string_lossy()));
}
