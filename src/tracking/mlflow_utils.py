"""MLflow tracking utilities with standardized auto-tagging.

Provides a context manager that every training/scoring entrypoint should use
instead of calling ``mlflow.start_run()`` directly.  This ensures every run
carries the tags the team needs to answer: "which code, which env, which user
produced this run?"

Usage::

    from src.tracking.mlflow_utils import start_run, log_model_with_signature

    with start_run(name="train-rf", env="dev") as run:
        # ... train ...
        mlflow.log_metric("val_false_alarm_rate", far)
        log_model_with_signature(clf, X_train, artifact_path="model")

Standard metric names (use these consistently across projects):
    train_loss, val_loss
    val_accuracy, val_macro_f1, val_log_loss
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Generator, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature

from src.logging.logger import get_logger

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


def _get_package_version() -> str:
    try:
        return importlib.metadata.version("ds-template")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


@contextmanager
def start_run(
    name: str,
    env: str = "dev",
    tags: Optional[dict[str, str]] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that starts an MLflow run with standard auto-tags.

    Auto-sets the following tags on every run:
        - ``git_sha``: short SHA of HEAD
        - ``env``: deployment environment (dev/staging/prod)
        - ``user``: current OS user or Databricks user
        - ``python_version``: running Python version
        - ``package_version``: installed ds-template version
        - ``platform``: "databricks" or "local"

    Args:
        name: Human-readable run name.
        env: Deployment environment string (dev/staging/prod).
        tags: Additional tags to set.  Merged with auto-tags; caller tags win.
        tracking_uri: Overrides MLflow tracking URI.  If not set, uses
            ``MLFLOW_TRACKING_URI`` env var or falls back to ``"mlruns"``.
        experiment_name: MLflow experiment path.  If not set, uses
            ``MLFLOW_EXPERIMENT_NAME`` env var.

    Yields:
        The active ``mlflow.ActiveRun`` object.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif _is_databricks():
        mlflow.set_tracking_uri("databricks")

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    auto_tags = {
        "git_sha": _get_git_sha(),
        "env": env,
        "user": os.getenv("USER", os.getenv("USERNAME", "unknown")),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "package_version": _get_package_version(),
        "platform": "databricks" if _is_databricks() else "local",
    }
    if tags:
        auto_tags.update(tags)

    with mlflow.start_run(run_name=name, tags=auto_tags) as run:
        logger.info("MLflow run started: %s (id=%s)", name, run.info.run_id)
        yield run
        logger.info("MLflow run finished: %s (id=%s)", name, run.info.run_id)


def log_model_with_signature(
    model: Any,
    X_sample: pd.DataFrame,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
) -> str:
    """Log a scikit-learn compatible model with an inferred signature.

    Always logs with a signature so the deploy team knows the exact input
    schema without reading the code.

    Args:
        model: Fitted model object (sklearn-compatible).
        X_sample: A sample of training features used to infer the schema.
            Typically ``X_train.iloc[:5]``.
        artifact_path: MLflow artifact path for the model.
        registered_model_name: If set, registers the model in the MLflow
            registry under this name after logging.

    Returns:
        The MLflow model URI (``runs:/<run_id>/<artifact_path>``).
    """
    signature = infer_signature(X_sample, model.predict(X_sample))
    result = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        registered_model_name=registered_model_name,
    )
    logger.info("Model logged: %s", result.model_uri)
    return result.model_uri
