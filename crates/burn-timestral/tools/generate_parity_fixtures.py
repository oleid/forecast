#!/usr/bin/env python3
"""
tools/generate_parity_fixtures.py

Generate deterministic parity fixtures from the official Timestral Python reference
(mistralai/timestral). See tests/fixtures/README.md for commands and model revisions.

- Sets per_shot_horizon_patches=1 to match autoregressive inference in Rust.
- Writes full (B, V, H, Q) quantile tensors for end-to-end parity checks.
- Covers uni- and multivariate forecasts, NaNs (including leading NaNs), padding,
  multi-block autoregression, max_context truncation and sliding windows, numeric
  covariates with NaN gaps, binary covariates, categorical target encoding with
  NaNs and unseen categories in the forecast horizon, explicit observation masks,
  symmetric averaging, and nonnegative clipping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import torch

from timestral import TimestralForecaster
from timestral.loading import load_model


TIMESTRAL_SOURCE_URL = "https://github.com/mistralai/timestral"
CHECKPOINT_REVISIONS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def git_head_of(checkout: Path) -> tuple[str | None, bool | None]:
    """Measure HEAD and dirtiness of a git checkout. Returns (None, None) if not measurable."""
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        return None, None
    return rev, bool(status)


def get_timestral_provenance(checkout_override: str | None = None) -> dict[str, Any]:
    """Measure the revision of the installed timestral package.

    source_url records the upstream project rather than an absolute local path.
    git_rev is a measured commit: from vcs_info.commit_id, or git rev-parse HEAD in the
    local checkout (file:// direct_url or --timestral-checkout). Otherwise "unknown".
    No arbitrary cache directories are scanned.
    """
    source_url: str | None = None
    commit_id: str | None = None
    checkout: Path | None = None
    try:
        dist = importlib.metadata.distribution("timestral")
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            data = json.loads(direct_url_text)
            source_url = data.get("url")
            commit_id = (data.get("vcs_info") or {}).get("commit_id")
            if source_url and source_url.startswith("file://"):
                checkout = Path(urllib.parse.unquote(urllib.parse.urlparse(source_url).path))
    except Exception:
        pass

    if checkout_override:
        checkout = Path(checkout_override).expanduser()

    git_rev: str | None = None
    git_dirty: bool | None = None
    if checkout is not None:
        git_rev, git_dirty = git_head_of(checkout)
    elif commit_id and re.fullmatch(r"[0-9a-f]{40}", commit_id):
        git_rev = commit_id
        git_dirty = False

    return {
        "source_url": TIMESTRAL_SOURCE_URL,
        "git_rev": git_rev or "unknown",
        "git_dirty": git_dirty,
    }


def make_series(
    batch: int,
    variates: int,
    length: int,
    seed: int = 0,
    positive: bool = False,
    scale: float = 10.0,
    period: float = 24.0,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    base = scale * torch.sin(2 * math.pi * t / period)
    noise = torch.randn(batch, variates, length, generator=g).cumsum(-1) * 0.1
    x = base.unsqueeze(0).unsqueeze(0).expand(batch, variates, length) + noise
    return (x.abs() + 1.0) if positive else x


def generate_full_scenarios(forecaster: TimestralForecaster, model: Any) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    # Scenario 1: Univariate short horizon (T=96, H=12, symmetric_averaging=True)
    print("Generating scenario 01: univariate_short_horizon...")
    x1 = make_series(1, 1, 96, seed=42)
    out1 = forecaster.forecast(x1, prediction_length=12, symmetric_averaging=True, clip_nonnegative=False)
    scenarios.append({
        "name": "univariate_short_horizon",
        "description": "B=1, V=1, T=96, H=12, symmetric_averaging=True, standard sine series",
        "batch": 1,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 12,
        "max_context": None,
        "context": x1.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out1.quantiles.shape),
        "expected_quantiles": out1.quantiles.reshape(-1).tolist(),
    })

    # Scenario 2: Univariate full block (T=64, H=32, symmetric_averaging=True)
    print("Generating scenario 02: univariate_single_block...")
    x2 = make_series(1, 1, 64, seed=101)
    out2 = forecaster.forecast(x2, prediction_length=32, symmetric_averaging=True, clip_nonnegative=False)
    scenarios.append({
        "name": "univariate_single_block",
        "description": "B=1, V=1, T=64, H=32, symmetric_averaging=True, exactly 1 patch horizon",
        "batch": 1,
        "variates": 1,
        "context_length": 64,
        "prediction_length": 32,
        "max_context": None,
        "context": x2.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out2.quantiles.shape),
        "expected_quantiles": out2.quantiles.reshape(-1).tolist(),
    })

    scenarios.append(univariate_unaligned_padding(forecaster))

    # Scenario 4: Univariate with NaN gaps in context (T=96, H=24)
    print("Generating scenario 04: univariate_nan_gaps...")
    x4 = make_series(1, 1, 96, seed=88)
    x4[0, 0, 10:16] = float("nan")
    x4[0, 0, 50:56] = float("nan")
    out4 = forecaster.forecast(x4, prediction_length=24, symmetric_averaging=True, clip_nonnegative=False)
    scenarios.append({
        "name": "univariate_nan_gaps",
        "description": "B=1, V=1, T=96, H=24, with NaN blocks at steps 10..15 and 50..55",
        "batch": 1,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": [None if math.isnan(v) else v for v in x4.reshape(-1).tolist()],
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out4.quantiles.shape),
        "expected_quantiles": out4.quantiles.reshape(-1).tolist(),
    })

    # Scenario 5: Multivariate 3 variates (B=1, V=3, T=128, H=32)
    print("Generating scenario 05: multivariate_3variates...")
    x5 = make_series(1, 3, 128, seed=12)
    out5 = forecaster.forecast(x5, prediction_length=32, symmetric_averaging=True, clip_nonnegative=False)
    scenarios.append({
        "name": "multivariate_3variates",
        "description": "B=1, V=3, T=128, H=32, joint multivariate forecast under variate firewall",
        "batch": 1,
        "variates": 3,
        "context_length": 128,
        "prediction_length": 32,
        "max_context": None,
        "context": x5.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out5.quantiles.shape),
        "expected_quantiles": out5.quantiles.reshape(-1).tolist(),
    })

    # Scenario 6: Batched Multivariate (B=2, V=2, T=64, H=24)
    print("Generating scenario 06: batched_multivariate...")
    x6 = make_series(2, 2, 64, seed=33)
    out6 = forecaster.forecast(x6, prediction_length=24, symmetric_averaging=True, clip_nonnegative=False)
    scenarios.append({
        "name": "batched_multivariate",
        "description": "B=2, V=2, T=64, H=24, batch dimension and multi-variates combined",
        "batch": 2,
        "variates": 2,
        "context_length": 64,
        "prediction_length": 24,
        "max_context": None,
        "context": x6.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out6.quantiles.shape),
        "expected_quantiles": out6.quantiles.reshape(-1).tolist(),
    })

    scenarios.append(multi_block_autoregressive(forecaster))

    # Scenario 8: Numeric past and known covariates (B=2, V=1, T=96, H=24, C_past=2, C_known=1)
    print("Generating scenario 08: numeric_past_and_known_covariates...")
    x8 = make_series(2, 1, 96, seed=123)
    past8 = make_series(2, 2, 96, seed=234, scale=4.0)
    known8 = make_series(2, 1, 96 + 24, seed=345, scale=3.0)
    out8 = forecaster.forecast(
        x8,
        prediction_length=24,
        past_covariates=past8,
        known_covariates=known8,
        symmetric_averaging=True,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "numeric_past_and_known_covariates",
        "description": "B=2, V=1, T=96, H=24, C_past=2, C_known=1 numeric covariates",
        "batch": 2,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": x8.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": {
            "num_covariates": 2,
            "values": past8.reshape(-1).tolist(),
        },
        "known_covariates": {
            "num_covariates": 1,
            "values": known8.reshape(-1).tolist(),
        },
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out8.quantiles.shape),
        "expected_quantiles": out8.quantiles.reshape(-1).tolist(),
    })

    # Scenario 9: Binary 0/1 indicator known covariate (T=64, H=32, C_known=1)
    print("Generating scenario 09: binary_indicator_known_covariate...")
    x9 = make_series(1, 1, 64, seed=77)
    bin_known = torch.zeros(1, 1, 64 + 32, dtype=torch.float32)
    bin_known[0, 0, ::4] = 1.0
    bin_known[0, 0, 1::4] = 1.0
    out9 = forecaster.forecast(
        x9,
        prediction_length=32,
        known_covariates=bin_known,
        symmetric_averaging=True,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "binary_indicator_known_covariate",
        "description": "B=1, V=1, T=64, H=32, known covariate with exact 0.0 and 1.0 values (instance norm bypass)",
        "batch": 1,
        "variates": 1,
        "context_length": 64,
        "prediction_length": 32,
        "max_context": None,
        "context": x9.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": {
            "num_covariates": 1,
            "values": bin_known.reshape(-1).tolist(),
        },
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out9.quantiles.shape),
        "expected_quantiles": out9.quantiles.reshape(-1).tolist(),
    })

    # Scenario 10: Categorical known covariates with target encoding, NaNs and unseen category (B=2, V=1, T=96, H=24)
    print("Generating scenario 10: categorical_known_covariates_target_encoded...")
    x10 = make_series(2, 1, 96, seed=888, positive=True)
    g10 = torch.Generator().manual_seed(999)
    cat_known10 = torch.randint(0, 4, (2, 1, 96 + 24), generator=g10).float()
    cat_known10[0, 0, 10:20] = float("nan")
    cat_known10[1, 0, 96 + 5 : 96 + 10] = 99.0

    out10 = forecaster.forecast(
        x10,
        prediction_length=24,
        categorical_known_covariates=cat_known10,
        symmetric_averaging=False,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "categorical_known_covariates_target_encoded",
        "description": "B=2, V=1, T=96, H=24, categorical codes with NaNs and unseen category (99.0) in horizon",
        "batch": 2,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": x10.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": {
            "num_covariates": 1,
            "codes": [None if math.isnan(v) else v for v in cat_known10.reshape(-1).tolist()],
        },
        "symmetric_averaging": False,
        "clip_nonnegative": False,
        "expected_shape": list(out10.quantiles.shape),
        "expected_quantiles": out10.quantiles.reshape(-1).tolist(),
    })

    # Scenario 11: Categorical past and known covariates combined (B=1, V=1, T=96, H=24)
    print("Generating scenario 11: categorical_past_and_known_covariates...")
    x11 = make_series(1, 1, 96, seed=512, positive=True)
    g11 = torch.Generator().manual_seed(1024)
    cat_past11 = torch.randint(0, 3, (1, 1, 96), generator=g11).float()
    cat_past11[0, 0, 5:10] = float("nan")
    cat_known11 = torch.randint(0, 4, (1, 1, 96 + 24), generator=g11).float()
    cat_known11[0, 0, 20:25] = float("nan")

    out11 = forecaster.forecast(
        x11,
        prediction_length=24,
        categorical_past_covariates=cat_past11,
        categorical_known_covariates=cat_known11,
        symmetric_averaging=True,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "categorical_past_and_known_covariates",
        "description": "B=1, V=1, T=96, H=24, both past and known categorical covariates with NaNs",
        "batch": 1,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": x11.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": {
            "num_covariates": 1,
            "codes": [None if math.isnan(v) else v for v in cat_past11.reshape(-1).tolist()],
        },
        "categorical_known_covariates": {
            "num_covariates": 1,
            "codes": [None if math.isnan(v) else v for v in cat_known11.reshape(-1).tolist()],
        },
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out11.quantiles.shape),
        "expected_quantiles": out11.quantiles.reshape(-1).tolist(),
    })

    # Scenario 12: Non-negativity Clamping (B=2, V=1, T=64, H=32)
    print("Generating scenario 12: clip_nonnegative_clamping...")
    t12 = 64
    s0 = (torch.sin(torch.linspace(0, 5, t12)) * 0.1 + 0.1).view(1, 1, t12)
    s1 = (torch.sin(torch.linspace(0, 10, t12)) - 10.0).view(1, 1, t12)
    x12 = torch.cat([s0, s1], dim=0)
    out12 = forecaster.forecast(
        x12,
        prediction_length=32,
        symmetric_averaging=False,
        clip_nonnegative=True,
    )
    scenarios.append({
        "name": "clip_nonnegative_clamping",
        "description": "B=2, V=1, T=64, H=32, series 0 clamped at >= 0.0, series 1 with negative context unclamped",
        "batch": 2,
        "variates": 1,
        "context_length": 64,
        "prediction_length": 32,
        "max_context": None,
        "context": x12.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": False,
        "clip_nonnegative": True,
        "expected_shape": list(out12.quantiles.shape),
        "expected_quantiles": out12.quantiles.reshape(-1).tolist(),
    })

    # Scenario 13: Symmetric averaging disabled with known covariates (B=1, V=2, T=96, H=24)
    print("Generating scenario 13: symmetric_averaging_disabled...")
    x13 = make_series(1, 2, 96, seed=2048)
    known13 = make_series(1, 1, 96 + 24, seed=4096, scale=4.0)
    out13 = forecaster.forecast(
        x13,
        prediction_length=24,
        known_covariates=known13,
        symmetric_averaging=False,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "symmetric_averaging_disabled",
        "description": "B=1, V=2, T=96, H=24, symmetric_averaging=False with 1 known covariate",
        "batch": 1,
        "variates": 2,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": x13.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": {
            "num_covariates": 1,
            "values": known13.reshape(-1).tolist(),
        },
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": False,
        "clip_nonnegative": False,
        "expected_shape": list(out13.quantiles.shape),
        "expected_quantiles": out13.quantiles.reshape(-1).tolist(),
    })

    # Scenario 14: max_context truncation, sliding window & categorical target-encoding (B=1, V=1, T=300, H=70, max_context=128)
    print("Generating scenario 14: max_context_truncation_sliding_categorical...")
    fc_128 = TimestralForecaster(model, max_context=128, per_shot_horizon_patches=1)
    g14 = torch.Generator().manual_seed(1414)
    t14 = torch.arange(300, dtype=torch.float32)
    x14 = (10.0 * torch.sin(2 * math.pi * t14 / 24.0)).unsqueeze(0).unsqueeze(0) + torch.randn(1, 1, 300, generator=g14).cumsum(-1) * 0.1
    x14 = x14.abs() + 1.0
    cat_known14 = torch.randint(0, 4, (1, 1, 300 + 70), generator=g14).float()
    cat_known14[0, 0, 50:60] = float("nan")
    cat_known14[0, 0, 300 + 10: 300 + 20] = 99.0 # unseen category in future

    out14 = fc_128.forecast(
        x14,
        prediction_length=70,
        categorical_known_covariates=cat_known14,
        symmetric_averaging=True,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "max_context_truncation_sliding_categorical",
        "description": "B=1, V=1, T=300, H=70, max_context=128: tests initial truncation (300->128), AR sliding window, and target-encoding window",
        "batch": 1,
        "variates": 1,
        "context_length": 300,
        "prediction_length": 70,
        "max_context": 128,
        "context": x14.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": {
            "num_covariates": 1,
            "codes": [None if math.isnan(v) else v for v in cat_known14.reshape(-1).tolist()],
        },
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out14.quantiles.shape),
        "expected_quantiles": out14.quantiles.reshape(-1).tolist(),
    })

    # Scenario 15: Explicit observation_mask with masked steps (B=1, V=1, T=96, H=24)
    print("Generating scenario 15: explicit_observation_mask...")
    t15 = torch.arange(96, dtype=torch.float32)
    x15 = (10.0 * torch.sin(2 * math.pi * t15 / 24.0)).unsqueeze(0).unsqueeze(0)
    obs15 = torch.ones(1, 1, 96, dtype=torch.bool)
    obs15[0, 0, 15:25] = False
    obs15[0, 0, 70:75] = False

    out15 = forecaster.forecast(
        x15,
        prediction_length=24,
        observation_mask=obs15,
        symmetric_averaging=True,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "explicit_observation_mask",
        "description": "B=1, V=1, T=96, H=24, explicit observation_mask with unobserved regions at 15..24 and 70..74",
        "batch": 1,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": x15.reshape(-1).tolist(),
        "observation_mask": obs15.reshape(-1).tolist(),
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out15.quantiles.shape),
        "expected_quantiles": out15.quantiles.reshape(-1).tolist(),
    })

    scenarios.append(numeric_covariates_with_nan_gaps(forecaster))

    # Scenario 17: Leading NaN block, interior NaN gap and an entirely NaN target row (B=3, V=1, T=96, H=24)
    print("Generating scenario 17: leading_and_all_nan_targets...")
    t17 = torch.arange(96, dtype=torch.float32)
    x17 = torch.zeros(3, 1, 96)
    x17[0, 0, :] = 10.0 * torch.sin(2 * math.pi * t17 / 24.0)
    x17[0, 0, 0:35] = float("nan") # leading NaN block spanning over full patch (0..34)
    x17[1, 0, :] = 5.0 * torch.cos(2 * math.pi * t17 / 16.0)
    x17[1, 0, 10:20] = float("nan")
    x17[2, 0, :] = float("nan") # entirely NaN target row

    out17 = forecaster.forecast(
        x17,
        prediction_length=24,
        symmetric_averaging=True,
        clip_nonnegative=False,
    )
    scenarios.append({
        "name": "leading_and_all_nan_targets",
        "description": "B=3, V=1, T=96, H=24: row 0 has leading NaNs (0..34), row 1 interior NaNs (10..19), row 2 entirely NaN",
        "batch": 3,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": [None if math.isnan(v) else v for v in x17.reshape(-1).tolist()],
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out17.quantiles.shape),
        "expected_quantiles": out17.quantiles.reshape(-1).tolist(),
    })

    return scenarios


def univariate_unaligned_padding(forecaster: TimestralForecaster) -> dict[str, Any]:
    print("Generating scenario 03: univariate_unaligned_padding...")
    x3 = make_series(1, 1, 37, seed=7)
    out3 = forecaster.forecast(x3, prediction_length=12, symmetric_averaging=True, clip_nonnegative=False)
    return {
        "name": "univariate_unaligned_padding",
        "description": "B=1, V=1, T=37, H=12, unaligned length padded to patch multiple (64)",
        "batch": 1,
        "variates": 1,
        "context_length": 37,
        "prediction_length": 12,
        "max_context": None,
        "context": x3.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": None,
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out3.quantiles.shape),
        "expected_quantiles": out3.quantiles.reshape(-1).tolist(),
    }


def multi_block_autoregressive(forecaster: TimestralForecaster) -> dict[str, Any]:
    print("Generating scenario 07: multi_block_autoregressive...")
    x7 = make_series(1, 1, 96, seed=44)
    known7 = make_series(1, 1, 96 + 70, seed=55, scale=5.0, period=16.0)
    out7 = forecaster.forecast(x7, prediction_length=70, known_covariates=known7, symmetric_averaging=True, clip_nonnegative=False)
    return {
        "name": "multi_block_autoregressive",
        "description": "B=1, V=1, T=96, H=70, 3-block autoregressive rollout with continuous known covariate",
        "batch": 1,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 70,
        "max_context": None,
        "context": x7.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": None,
        "known_covariates": {
            "num_covariates": 1,
            "values": known7.reshape(-1).tolist(),
        },
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out7.quantiles.shape),
        "expected_quantiles": out7.quantiles.reshape(-1).tolist(),
    }


def numeric_covariates_with_nan_gaps(forecaster: TimestralForecaster) -> dict[str, Any]:
    print("Generating scenario 16: numeric_covariates_with_nan_gaps...")
    g16 = torch.Generator().manual_seed(1616)
    x16 = make_series(1, 1, 96, seed=16)
    past16 = torch.randn(1, 1, 96, generator=g16) * 3.0
    past16[0, 0, 20:30] = float("nan")
    known16 = torch.randn(1, 1, 96 + 24, generator=g16) * 2.0
    known16[0, 0, 40:50] = float("nan")
    known16[0, 0, 96 + 5 : 96 + 10] = float("nan")

    out16 = forecaster.forecast(
        x16,
        prediction_length=24,
        past_covariates=past16,
        known_covariates=known16,
        symmetric_averaging=True,
        clip_nonnegative=False,
    )
    return {
        "name": "numeric_covariates_with_nan_gaps",
        "description": "B=1, V=1, T=96, H=24, C_past=1, C_known=1 numeric covariates containing NaN gaps",
        "batch": 1,
        "variates": 1,
        "context_length": 96,
        "prediction_length": 24,
        "max_context": None,
        "context": x16.reshape(-1).tolist(),
        "observation_mask": None,
        "past_covariates": {
            "num_covariates": 1,
            "values": [None if math.isnan(v) else v for v in past16.reshape(-1).tolist()],
        },
        "known_covariates": {
            "num_covariates": 1,
            "values": [None if math.isnan(v) else v for v in known16.reshape(-1).tolist()],
        },
        "categorical_past_covariates": None,
        "categorical_known_covariates": None,
        "symmetric_averaging": True,
        "clip_nonnegative": False,
        "expected_shape": list(out16.quantiles.shape),
        "expected_quantiles": out16.quantiles.reshape(-1).tolist(),
    }


def generate_fixtures(variant: str, ckpt_dir: Path, out_file: Path, timestral_checkout: str | None) -> None:
    provenance = get_timestral_provenance(timestral_checkout)
    if provenance["git_dirty"] is not False:
        sys.exit("Cannot verify a clean timestral source revision; pass --timestral-checkout.")

    print(f"Loading official Timestral-{variant} model from: {ckpt_dir}")
    model = load_model(str(ckpt_dir), device="cpu")
    model.eval()

    forecaster = TimestralForecaster(model, per_shot_horizon_patches=1)
    config = model.config
    knots = list(config.quantile_knots)
    metadata = {
        "generator": "tools/generate_parity_fixtures.py",
        "checkpoint": f"mistralai/Timestral-{variant}",
        "checkpoint_revision": CHECKPOINT_REVISIONS[variant],
        "model_safetensors_sha256": compute_sha256(ckpt_dir / "model.safetensors"),
        "config_json_sha256": compute_sha256(ckpt_dir / "config.json"),
        "timestral_git_rev": provenance["git_rev"],
        "timestral_git_dirty": provenance["git_dirty"],
        "timestral_source_url": provenance["source_url"],
        "torch_version": torch.__version__,
        "per_shot_horizon_patches": 1,
        "patch_size": config.patch_size,
        "quantile_knots": knots,
        "num_quantiles": len(knots),
        "median_index": knots.index(0.5),
    }

    if variant == "209M":
        scenarios = generate_full_scenarios(forecaster, model)
    else:
        scenarios = [
            univariate_unaligned_padding(forecaster),
            multi_block_autoregressive(forecaster),
            numeric_covariates_with_nan_gaps(forecaster),
        ]

    payload = {
        "metadata": metadata,
        "scenarios": scenarios,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    file_size_kb = out_file.stat().st_size / 1024.0
    print(f"Successfully generated {len(scenarios)} parity scenarios in {out_file} ({file_size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Timestral parity fixtures")
    parser.add_argument("--variant", choices=CHECKPOINT_REVISIONS, default="209M")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("TIMESTRAL_CHECKPOINT"),
        help="Path to the official checkpoint directory for the selected variant",
    )
    parser.add_argument(
        "--download-checkpoint",
        type=Path,
        help="Download the pinned Hub checkpoint into this directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output JSON fixture file (defaults to the selected variant's fixture file)",
    )
    parser.add_argument(
        "--timestral-checkout",
        default=os.environ.get("TIMESTRAL_CHECKOUT"),
        help="Local git checkout of mistralai/timestral used to measure the revision (git rev-parse HEAD)",
    )
    args = parser.parse_args()

    if args.download_checkpoint is not None:
        if args.checkpoint is not None:
            parser.error("--download-checkpoint cannot be combined with a local checkpoint")
        from huggingface_hub import snapshot_download

        checkpoint = args.download_checkpoint.expanduser().resolve()
        snapshot_download(
            repo_id=f"mistralai/Timestral-{args.variant}",
            revision=CHECKPOINT_REVISIONS[args.variant],
            allow_patterns=["config.json", "model.safetensors"],
            local_dir=checkpoint,
        )
    elif args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
    else:
        parser.error("--checkpoint, TIMESTRAL_CHECKPOINT or --download-checkpoint is required")
    if not checkpoint.is_dir():
        parser.error(f"Checkpoint directory not found: {checkpoint}")

    suffix = "" if args.variant == "209M" else f"_{args.variant.lower()}"
    output = args.out or (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / f"parity_scenarios{suffix}.json"
    )

    generate_fixtures(args.variant, checkpoint, output, args.timestral_checkout)


if __name__ == "__main__":
    main()
