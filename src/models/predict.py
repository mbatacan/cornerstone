"""Reference scoring entrypoint.

Loads a registered MLflow model, scores the input data, and writes validated
prediction rows to ``settings.predictions.output_path``.

Run locally::

    CORNERSTONE_ENV=dev python -m src.models.predict

Run via CLI::

    ds-template predict
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config.settings import get_settings
from src.data.load_data import load_data
from src.features.build_features import add_sepal_area
from src.logging.logger import get_logger
from src.models.registry import get_latest_model_version, get_model_uri
from src.models.schemas import PredictionRecord

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


def predict() -> tuple[int, str]:
    """Score data with the latest registered model and write validated rows.

    Returns:
        Tuple of (rows_written, output_path).

    Raises:
        RuntimeError: If no registered model version exists.
        pydantic.ValidationError: If any row fails ``PredictionRecord`` validation.
    """
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    model_name = cfg.mlflow.registered_model_name
    model_version = get_latest_model_version(model_name, stage="None")
    if model_version is None:
        raise RuntimeError(
            f"No registered model found for '{model_name}'. Run training first."
        )

    model_uri = get_model_uri(model_name, model_version)
    logger.info("Loading model: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    run_id = MlflowClient().get_model_version(model_name, model_version).run_id

    df = add_sepal_area(load_data())
    X = df.drop(columns=["target"])
    proba = model.predict_proba(X)

    scored_at = datetime.now(UTC)
    git_sha = _get_git_sha()
    records = [
        PredictionRecord(
            row_id=int(row_id),
            prediction=int(model.classes_[row_proba.argmax()]),
            score=float(row_proba.max()),
            model_name=model_name,
            model_version=str(model_version),
            run_id=str(run_id),
            git_sha=git_sha,
            scored_at=scored_at,
        )
        for row_id, row_proba in zip(X.index, proba, strict=True)
    ]

    out_dir = Path(cfg.predictions.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{scored_at:%Y%m%dT%H%M%S}.parquet"
    pd.DataFrame([r.model_dump() for r in records]).to_parquet(out_path, index=False)

    logger.info("Scoring complete: %d row(s) written to %s", len(records), out_path)
    return len(records), str(out_path)


if __name__ == "__main__":
    predict()
