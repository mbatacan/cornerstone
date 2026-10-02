"""Unit tests for layered settings."""

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
