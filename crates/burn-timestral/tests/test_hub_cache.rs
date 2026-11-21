#![cfg(feature = "hub")]

use std::fs;
use std::path::Path;
use std::process::Command;

use burn_timestral::TimestralVariant;

const CACHE_VARS: &[&str] = &[
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HOME",
    "XDG_CACHE_HOME",
];

fn offline_error(csv: &Path, overrides: &[(&str, &Path)]) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_burn-timestral"));
    for name in CACHE_VARS {
        command.env_remove(name);
    }
    let output = command
        .envs(overrides.iter().copied())
        .env("HF_HUB_OFFLINE", "true")
        .arg(csv)
        .args(["1", "--variant", "5M"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    String::from_utf8(output.stderr).unwrap()
}

#[test]
fn resolves_cache_overrides_in_order_without_network() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("input.csv");
    fs::write(&csv, "1\n2\n").unwrap();

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
        let error = offline_error(&csv, &overrides[index..]);
        assert!(error.contains("HF_HUB_OFFLINE is set"), "{error}");
        assert!(error.contains(cache.to_str().unwrap()), "{error}");
    }
    let error = offline_error(&csv, &[("HF_HUB_CACHE", Path::new("")), overrides[3]]);
    assert!(error.contains(expected[3].to_str().unwrap()), "{error}");
}

#[test]
fn reuses_pinned_snapshot_without_a_ref_when_offline() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("input.csv");
    fs::write(&csv, "1\n2\n").unwrap();

    let snapshot = dir
        .path()
        .join("models--mistralai--Timestral-5M/snapshots")
        .join(TimestralVariant::M5.revision());
    fs::create_dir_all(&snapshot).unwrap();
    fs::write(snapshot.join("config.json"), "not JSON").unwrap();

    let overrides = [("HF_HUB_CACHE", dir.path())];
    let partial = offline_error(&csv, &overrides);
    assert!(partial.contains("HF_HUB_OFFLINE is set"), "{partial}");

    fs::write(snapshot.join("model.safetensors"), []).unwrap();
    let cached = offline_error(&csv, &overrides);
    assert!(cached.contains("parsing configuration"), "{cached}");
    assert!(cached.contains(snapshot.to_str().unwrap()), "{cached}");
}
