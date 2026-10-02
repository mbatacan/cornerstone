"""Smoke test: end-to-end training pipeline with a local MLflow run."""

from __future__ import annotations


import mlflow
import pytest


@pytest.mark.slow
def test_train_smoke(tmp_path, monkeypatch):
    """Training completes without error and logs metrics to a local MLflow run."""
    monkeypatch.setenv("CORNERSTONE_ENV", "dev")
    mlflow_dir = str(tmp_path / "mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_dir)

    # Clear settings cache so it picks up env vars set above
    from src.config.settings import get_settings

    get_settings.cache_clear()

    mlflow.set_tracking_uri(mlflow_dir)

    from src.models.train import train

    model_uri = train()

    assert model_uri.startswith("runs:/")

    # Verify metrics were logged
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_dir)
    runs = client.search_runs(
        experiment_ids=["0"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    assert runs, "Expected at least one MLflow run"
    metrics = runs[0].data.metrics
    assert "val_accuracy" in metrics
    assert "val_false_alarm_rate" in metrics
    assert metrics["val_accuracy"] > 0.5
