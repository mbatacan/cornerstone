"""Layered configuration via pydantic-settings.

Load order (each layer overrides the previous):
  1. configs/default.yaml
  2. configs/{CORNERSTONE_ENV}.yaml
  3. Environment variables prefixed with CORNERSTONE_
  4. Databricks secret scope (when running on Databricks)

Usage::

    from src.config.settings import get_settings
    cfg = get_settings()
    print(cfg.mlflow.experiment_name)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class MLflowSettings(BaseModel):
    tracking_uri: str = "mlruns"
    experiment_name: str = "/Shared/ds-template/dev/experiments"
    registered_model_name: str = "ds-template-model"


class PredictionsSettings(BaseModel):
    output_path: str = (
        "output/predictions"  # local parquet dir for dev; Delta path in prod
    )


class DataSettings(BaseModel):
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"


class TrainingSettings(BaseModel):
    random_seed: int = 42
    test_size: float = 0.2
    n_estimators: int = 100


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Project-wide settings.

    Reads from environment variables with the prefix ``CORNERSTONE_``.
    Nested models are addressed with double-underscore, e.g.:
    ``CORNERSTONE_MLFLOW__TRACKING_URI=databricks``.
    """

    model_config = SettingsConfigDict(
        env_prefix="CORNERSTONE_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    env: str = "dev"
    project_name: str = "ds-template"

    mlflow: MLflowSettings = MLflowSettings()
    predictions: PredictionsSettings = PredictionsSettings()
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()

    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _build_settings() -> Settings:
    """Build Settings by merging default.yaml, {env}.yaml, then env vars (highest priority)."""
    env_name = os.getenv("CORNERSTONE_ENV", "dev")

    defaults = _load_yaml(_CONFIGS_DIR / "default.yaml")
    overrides = _load_yaml(_CONFIGS_DIR / f"{env_name}.yaml")
    # Init kwargs outrank env vars in pydantic-settings, so merge env values in explicitly.
    env_values = EnvSettingsSource(Settings)()

    merged = _deep_merge(_deep_merge(defaults, overrides), env_values)
    merged["env"] = env_name
    return Settings(**merged)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance. Call ``get_settings.cache_clear()`` in tests."""
    return _build_settings()
