"""Reference scoring/prediction entrypoint.

Loads a registered MLflow model, scores input data, and emits validated
alerts via ``emit_alert``.

Run locally::

    CORNERSTONE_ENV=dev python -m src.models.predict

Run via CLI::

    ds-template predict
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import mlflow.sklearn
import pandas as pd

from src.alerts.emit import emit_alert
from src.alerts.metadata import AlertMetadata
from src.config.settings import get_settings
from src.data.load_data import load_data
from src.features.build_features import add_sepal_area
from src.logging.logger import get_logger
from src.models.registry import get_latest_model_version, get_model_uri

logger = get_logger(__name__)


def _get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def predict(
    source_system: str = "iris-demo",
    source_asset_id: str = "asset-001",
    mlflow_run_id: str = "unknown",
) -> tuple[int, str]:
    """Score data and emit alerts for rows exceeding the configured threshold.

    Args:
        source_system: Identifier for the upstream data system.
        source_asset_id: Asset being scored (tail number, serial number, etc.).
        mlflow_run_id: MLflow run ID associated with the model used for scoring.

    Returns:
        Tuple of (alerts_written, output_path).
    """
    cfg = get_settings()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    model_version = get_latest_model_version(
        cfg.mlflow.registered_model_name, stage="None"
    )
    if model_version is None:
        logger.warning(
            "No registered model found for '%s'. Run training first.",
            cfg.mlflow.registered_model_name,
        )
        return 0, ""

    model_uri = get_model_uri(cfg.mlflow.registered_model_name, model_version)
    logger.info("Loading model: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    df = load_data()
    df = add_sepal_area(df)
    X = df.drop(columns=["target"])

    scores = pd.DataFrame({"score": model.predict_proba(X)[:, 1]})

    now = datetime.now(timezone.utc)
    metadata = AlertMetadata(
        env=cfg.env,
        model_name=cfg.mlflow.registered_model_name,
        model_version=str(model_version),
        registered_model_uri=model_uri,
        mlflow_run_id=mlflow_run_id,
        git_sha=_get_git_sha(),
        source_system=source_system,
        source_asset_id=source_asset_id,
        input_window_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
        input_window_end=now,
        score=0.0,  # per-row score filled in by emit_alert
        threshold=cfg.alerts.default_threshold,
        severity=cfg.alerts.default_severity,
    )

    n_written, output_path = emit_alert(
        scores=scores,
        metadata_defaults=metadata,
        score_col="score",
        threshold=cfg.alerts.default_threshold,
    )

    logger.info("Scoring complete: %d alert(s) written to %s", n_written, output_path)
    return n_written, output_path


if __name__ == "__main__":
    predict()
