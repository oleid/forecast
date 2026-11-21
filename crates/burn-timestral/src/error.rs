use std::{error::Error, io, path::PathBuf};

use forecast_core::ForecastError;

/// An error while loading a Timestral checkpoint or producing a forecast.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// Forecast data, options, or dimensions are invalid.
    #[error("Invalid model input: {0}")]
    InvalidInput(String),
    /// The checkpoint configuration could not be read.
    #[error("reading configuration {path}: {source}")]
    ConfigRead {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// The checkpoint configuration is not valid Timestral JSON.
    #[error("parsing configuration {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    /// The configuration describes an unsupported architecture or setting.
    #[error("invalid configuration {path}: {reason}")]
    InvalidConfig { path: PathBuf, reason: String },
    /// The weights file could not be opened or decoded.
    #[error("loading checkpoint {path}: {source}")]
    CheckpointRead {
        path: PathBuf,
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
    /// The checkpoint tensors do not match the configured architecture.
    #[error("incompatible checkpoint {path}: {reason}")]
    IncompatibleCheckpoint { path: PathBuf, reason: String },
    /// A local directory is required when the Hub feature is disabled.
    #[error("provide a model directory for {repo} or enable the `hub` feature")]
    HubDisabled { repo: &'static str },
    /// No default cache path was available and no override was provided.
    #[error("cannot determine the Hugging Face cache directory; set HF_HUB_CACHE")]
    HubCacheUnavailable,
    /// A pinned snapshot is incomplete and downloads were disabled.
    #[error("{repo} is not fully cached at {snapshot} and HF_HUB_OFFLINE is set")]
    HubOffline {
        repo: &'static str,
        snapshot: PathBuf,
    },
    /// The Hugging Face client could not be initialized.
    #[error("initializing Hugging Face client: {0}")]
    HubClient(#[source] Box<dyn Error + Send + Sync>),
    /// A required file could not be downloaded.
    #[error("downloading {repo}/{file} at {revision}: {source}")]
    HubDownload {
        repo: &'static str,
        file: &'static str,
        revision: &'static str,
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
    /// Inference did not produce a usable forecast.
    #[error("invalid model output: {0}")]
    InvalidOutput(String),
}

impl From<ForecastError> for ModelError {
    fn from(error: ForecastError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

pub(crate) fn checked_buffer_len<T>(name: &str, dimensions: &[usize]) -> Result<usize, ModelError> {
    dimensions
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .filter(|&len| {
            len.checked_mul(size_of::<T>())
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| ModelError::InvalidInput(format!("{name} dimensions exceed buffer limits")))
}
