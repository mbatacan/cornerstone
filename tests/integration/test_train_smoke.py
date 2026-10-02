"""Smoke test: end-to-end training pipeline with a local MLflow run."""

import mlflow
import pytest

from src.config.settings import get_settings


@pytest.mark.slow
def test_train_smoke(tmp_path, monkeypatch) -> None:
    """Training completes and logs metrics to a local MLflow run."""
    mlflow_dir = str(tmp_path / "mlruns")
    monkeypatch.setenv("CORNERSTONE_ENV", "dev")
    monkeypatch.setenv("CORNERSTONE_MLFLOW__TRACKING_URI", mlflow_dir)
    get_settings.cache_clear()
    cfg = get_settings()

    from src.models.train import train

    model_uri = train()
    assert model_uri.startswith("runs:/")

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_dir)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    assert experiment is not None, "Expected the configured experiment to exist"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    assert runs, "Expected at least one MLflow run"
    metrics = runs[0].data.metrics
    assert metrics["val_accuracy"] > 0.5
    assert "val_macro_f1" in metrics
    assert "val_log_loss" in metrics
