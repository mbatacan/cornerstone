"""Unit tests for layered settings."""

import pytest

from src.config.settings import get_settings


def test_env_var_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("CORNERSTONE_TRAINING__N_ESTIMATORS", "7")
    assert get_settings().training.n_estimators == 7


def test_env_override_keeps_sibling_yaml_values(monkeypatch) -> None:
    monkeypatch.setenv("CORNERSTONE_TRAINING__N_ESTIMATORS", "7")
    assert get_settings().training.test_size == 0.2


def test_env_specific_yaml_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("CORNERSTONE_ENV", "staging")
    assert get_settings().mlflow.tracking_uri == "databricks"


def test_default_env_is_local_sqlite() -> None:
    cfg = get_settings()
    assert cfg.env == "local"
    assert cfg.mlflow.tracking_uri.startswith("sqlite:")


def test_dev_env_is_databricks_uc(monkeypatch) -> None:
    monkeypatch.setenv("CORNERSTONE_ENV", "dev")
    cfg = get_settings()
    assert cfg.mlflow.tracking_uri == "databricks"
    assert cfg.mlflow.registry_uri == "databricks-uc"
    assert cfg.mlflow.registered_model_name.count(".") == 2


def test_unknown_env_fails_loudly(monkeypatch) -> None:
    monkeypatch.setenv("CORNERSTONE_ENV", "nope")
    with pytest.raises(FileNotFoundError):
        get_settings()
