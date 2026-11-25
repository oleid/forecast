#![cfg(feature = "hub")]

use std::path::Path;
use std::process::Command;

use burn::tensor::Device;
use burn_timestral::{TimestralForecaster, TimestralVariant};

const CACHE_VARS: &[&str] = &[
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HOME",
    "XDG_CACHE_HOME",
];

fn offline_error(overrides: &[(&str, &Path)]) -> String {
    let mut command = Command::new(std::env::current_exe().unwrap());
    for name in CACHE_VARS {
        command.env_remove(name);
    }
    let output = command
        .envs(overrides.iter().copied())
        .env("HF_HUB_OFFLINE", "true")
        .env("TIMESTRAL_HUB_CACHE_CHILD", "1")
        .args(["--exact", "hub_cache_child_load", "--nocapture"])
        .output()
        .unwrap();

    assert!(output.status.success());
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn hub_cache_child_load() {
    if std::env::var_os("TIMESTRAL_HUB_CACHE_CHILD").is_none() {
        return;
    }
    let error = match TimestralForecaster::load(TimestralVariant::M5, None, &Device::flex()) {
        Ok(_) => panic!("the test child unexpectedly loaded a checkpoint"),
        Err(error) => error,
    };
    println!("{error}");
}

#[test]
fn resolves_cache_overrides_in_order_without_network() {
    let dir = tempfile::tempdir().unwrap();

    let explicit = dir.path().join("explicit-cache");
    let legacy = dir.path().join("legacy-cache");
    let hf_home = dir.path().join("hf-home");
    let xdg = dir.path().join("xdg-cache");
    let overrides = [
        ("HF_HUB_CACHE", explicit.as_path()),
        ("HUGGINGFACE_HUB_CACHE", legacy.as_path()),
        ("HF_HOME", hf_home.as_path()),
        ("XDG_CACHE_HOME", xdg.as_path()),
    ];
    let expected = [
        explicit.clone(),
        legacy.clone(),
        hf_home.join("hub"),
        xdg.join("huggingface/hub"),
    ];
    for (index, cache) in expected.iter().enumerate() {
        let error = offline_error(&overrides[index..]);
        assert!(error.contains("HF_HUB_OFFLINE is set"), "{error}");
        assert!(error.contains(cache.to_str().unwrap()), "{error}");
    }
    let error = offline_error(&[("HF_HUB_CACHE", Path::new("")), overrides[3]]);
    assert!(error.contains(expected[3].to_str().unwrap()), "{error}");
}

#[test]
fn reuses_pinned_snapshot_without_a_ref_when_offline() {
    let dir = tempfile::tempdir().unwrap();

    let snapshot = dir
        .path()
        .join("models--mistralai--Timestral-5M/snapshots")
        .join(TimestralVariant::M5.revision());
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.json"), "not JSON").unwrap();

    let overrides = [("HF_HUB_CACHE", dir.path())];
    let partial = offline_error(&overrides);
    assert!(partial.contains("HF_HUB_OFFLINE is set"), "{partial}");

    std::fs::write(snapshot.join("model.safetensors"), []).unwrap();
    let cached = offline_error(&overrides);
    assert!(cached.contains("parsing configuration"), "{cached}");
    assert!(cached.contains(snapshot.to_str().unwrap()), "{cached}");
}
