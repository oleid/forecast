#[cfg(feature = "hub")]
use std::path::PathBuf;

/// Official Mistral checkpoints available via Hugging Face.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TimestralVariant {
    M5,
    M28,
    M123,
    #[default]
    M209,
}

impl TimestralVariant {
    pub const fn repo_id(self) -> &'static str {
        match self {
            Self::M5 => "mistralai/Timestral-5M",
            Self::M28 => "mistralai/Timestral-28M",
            Self::M123 => "mistralai/Timestral-123M",
            Self::M209 => "mistralai/Timestral-209M",
        }
    }

    pub const fn revision(self) -> &'static str {
        match self {
            Self::M5 => "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
            Self::M28 => "49b26409ac87b5935539a5d8ffe7f397b720db47",
            Self::M123 => "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
            Self::M209 => "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
        }
    }
}

pub(crate) const CONFIG: &str = "config.json";
pub(crate) const WEIGHTS: &str = "model.safetensors";

#[cfg(feature = "hub")]
fn hub_cache_dir() -> Result<PathBuf, crate::ModelError> {
    let env_path = |name| {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    // hf-hub 0.4 only honors HF_HOME. Match the documented Hub cache precedence here.
    env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .or_else(|| env_path("HF_HOME").map(|path| path.join("hub")))
        .or_else(|| env_path("XDG_CACHE_HOME").map(|path| path.join("huggingface/hub")))
        .or_else(|| std::env::home_dir().map(|path| path.join(".cache/huggingface/hub")))
        .ok_or(crate::ModelError::HubCacheUnavailable)
}

#[cfg(feature = "hub")]
pub(crate) fn hub_files(
    variant: TimestralVariant,
) -> Result<(PathBuf, PathBuf), crate::ModelError> {
    use crate::ModelError;
    use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};

    let cache = Cache::new(hub_cache_dir()?);
    let repo = Repo::with_revision(
        variant.repo_id().into(),
        RepoType::Model,
        variant.revision().into(),
    );
    // Other Hub clients may have populated the pinned snapshot without a refs entry.
    let snapshot = cache
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(variant.revision());
    let config = snapshot.join(CONFIG);
    let weights = snapshot.join(WEIGHTS);
    if config.is_file() && weights.is_file() {
        return Ok((config, weights));
    }
    if std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        matches!(
            value.to_ascii_uppercase().as_str(),
            "1" | "ON" | "YES" | "TRUE"
        )
    }) {
        return Err(ModelError::HubOffline {
            repo: variant.repo_id(),
            snapshot,
        });
    }

    let api = ApiBuilder::from_cache(cache)
        .with_retries(2)
        .build()
        .map_err(|error| ModelError::HubClient(Box::new(error)))?;
    let repo = api.repo(repo);
    let resolve = |path: PathBuf, name: &'static str| {
        if path.is_file() {
            return Ok(path);
        }
        repo.download(name)
            .map_err(|source| ModelError::HubDownload {
                repo: variant.repo_id(),
                file: name,
                revision: variant.revision(),
                source: Box::new(source),
            })
    };
    Ok((resolve(config, CONFIG)?, resolve(weights, WEIGHTS)?))
}
