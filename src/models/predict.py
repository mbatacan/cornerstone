"""Scoring entrypoint.

Loads a registered MLflow model, scores the input data, and writes validated
prediction rows to ``settings.predictions.output_path``.

Run locally::

    CORNERSTONE_ENV=local python -m src.models.predict

Run via CLI::

    ds-template predict
"""

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config.settings import get_settings
from src.data.load_data import load_data
from src.features.build_features import build_features
from src.logging.logger import get_logger
from src.models.registry import get_model_uri, get_version_by_alias
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


@dataclass(frozen=True)
class LoadedModel:
    """A registered model resolved from its alias, plus the metadata scoring needs."""

    model: Any
    model_name: str
    version: str
    run_id: str
    git_sha: str


def load_champion() -> LoadedModel:
    """Resolve the configured model alias in the registry and load that model.

    Shared by batch scoring and the API so both serve the same model.

    Raises:
        mlflow.exceptions.MlflowException: If the model or alias does not exist.
    """
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if cfg.mlflow.registry_uri:
        mlflow.set_registry_uri(cfg.mlflow.registry_uri)

    model_name = cfg.mlflow.registered_model_name
    alias = cfg.mlflow.model_alias
    version = get_version_by_alias(model_name, alias)

    model_uri = get_model_uri(model_name, alias)
    logger.info("Loading model: %s (v%s)", model_uri, version)
    return LoadedModel(
        model=mlflow.sklearn.load_model(model_uri),
        model_name=model_name,
        version=str(version),
        run_id=str(MlflowClient().get_model_version(model_name, version).run_id),
        git_sha=_get_git_sha(),
    )


def score(loaded: LoadedModel, features: pd.DataFrame) -> list[PredictionRecord]:
    """Score model-ready features and return one validated record per row.

    ``features`` must already have gone through ``build_features`` and must not
    contain the target column.
    """
    model = loaded.model
    proba = model.predict_proba(features)
    scored_at = datetime.now(UTC)
    return [
        PredictionRecord(
            row_id=int(row_id),
            prediction=int(model.classes_[row_proba.argmax()]),
            score=float(row_proba.max()),
            model_name=loaded.model_name,
            model_version=loaded.version,
            run_id=loaded.run_id,
            git_sha=loaded.git_sha,
            scored_at=scored_at,
        )
        for row_id, row_proba in zip(features.index, proba, strict=True)
    ]


def predict() -> tuple[int, str]:
    """Batch-score the data with the champion model and write validated rows.

    Returns:
        Tuple of (rows_written, output_path).

    Raises:
        mlflow.exceptions.MlflowException: If the model alias does not exist.
        pydantic.ValidationError: If any row fails ``PredictionRecord`` validation.
    """
    cfg = get_settings()
    loaded = load_champion()

    df = build_features(load_data())
    records = score(loaded, df.drop(columns=["target"]))

    out_dir = Path(cfg.predictions.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{datetime.now(UTC):%Y%m%dT%H%M%S}.parquet"
    pd.DataFrame([r.model_dump() for r in records]).to_parquet(out_path, index=False)

    logger.info("Scoring complete: %d row(s) written to %s", len(records), out_path)
    return len(records), str(out_path)


if __name__ == "__main__":
    predict()
