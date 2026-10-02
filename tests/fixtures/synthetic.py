"""Synthetic DataFrames and arrays for use in tests.

All fixtures produce small, deterministic datasets (3–10 rows).
Never use production data in tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def make_scores_df(n: int = 5, seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame with a ``score`` column in [0, 1]."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"score": rng.uniform(0, 1, n)})


def make_binary_labels(
    n: int = 10, pos_ratio: float = 0.4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_score) binary classification arrays."""
    rng = np.random.default_rng(seed)
    y_true = (rng.uniform(size=n) < pos_ratio).astype(int)
    y_score = np.clip(y_true * 0.6 + rng.uniform(0, 0.5, n), 0, 1)
    return y_true, y_score


def make_rul_arrays(n: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) RUL arrays in hours."""
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(10, 200, n)
    noise = rng.normal(0, 10, n)
    y_pred = np.clip(y_true + noise, 0, None)
    return y_true, y_pred


def make_alert_metadata_kwargs() -> dict:
    """Return a minimal valid dict for constructing an AlertMetadata."""
    now = datetime.now(timezone.utc)
    return {
        "env": "dev",
        "model_name": "test-model",
        "model_version": "1",
        "registered_model_uri": "models:/test-model/1",
        "mlflow_run_id": "abc123",
        "git_sha": "deadbeef",
        "source_system": "test-system",
        "source_asset_id": "asset-001",
        "input_window_start": now - timedelta(hours=1),
        "input_window_end": now,
        "score": 0.85,
        "threshold": 0.5,
        "severity": "warn",
    }


def make_feature_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a small feature DataFrame with two numeric columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.uniform(0, 10, n),
        }
    )
