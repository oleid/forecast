mod checkpoint;
mod config;
mod covariates;
mod error;
mod forecaster;
mod masks;
mod model;
mod network;
mod scaler;

pub use checkpoint::TimestralVariant;
pub use error::ModelError;
pub use forecast_core::{ForecastError, ForecastRequest, Forecaster, QuantileForecaster};
pub use forecaster::{Forecast, ForecastOptions, TimestralForecaster};

#[cfg(test)]
mod test_support {
    use crate::{TimestralVariant, config::TimestralConfig, model::TimestralModel};
    use burn::tensor::Device;
    use std::sync::OnceLock;

    pub(crate) fn tiny_config() -> TimestralConfig {
        TimestralConfig {
            d_model: 24,
            num_heads: 3,
            mlp_hidden_dim: 48,
            block_schedule: vec!["time".into(), "time_variate".into(), "time".into()],
            ..TimestralConfig::default_209m()
        }
    }

    pub(crate) fn tiny_model(device: &Device) -> TimestralModel {
        static MODEL: OnceLock<TimestralModel> = OnceLock::new();
        MODEL
            .get_or_init(|| {
                device.seed(42);
                TimestralModel::new(tiny_config(), device).unwrap()
            })
            .clone()
    }

    pub(crate) fn pretrained_209m() -> &'static TimestralModel {
        static MODEL: OnceLock<TimestralModel> = OnceLock::new();
        MODEL.get_or_init(|| {
            let dir = std::env::var_os("TIMESTRAL_MODEL_DIR").map(std::path::PathBuf::from);
            TimestralModel::load(TimestralVariant::M209, dir.as_deref(), &Device::default())
                .expect("failed to load the official Timestral-209M checkpoint")
        })
    }
}
