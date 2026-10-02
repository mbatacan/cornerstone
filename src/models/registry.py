"""MLflow Model Registry helpers.

Thin wrappers around the MLflow client for common registry operations used
across training and deployment workflows.
"""

from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.logging.logger import get_logger

logger = get_logger(__name__)


def get_latest_model_version(
    model_name: str,
    stage: str = "None",
) -> Optional[str]:
    """Return the latest version number for a registered model at a given stage.

    Args:
        model_name: Registered model name in the MLflow registry.
        stage: Model stage filter (``"None"``, ``"Staging"``, ``"Production"``).
            Use ``"None"`` for undeployed versions.

    Returns:
        Version string (e.g. ``"3"``), or ``None`` if no versions exist.
    """
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            return None
        return versions[0].version
    except mlflow.exceptions.MlflowException as exc:
        logger.warning("Could not fetch model versions for %s: %s", model_name, exc)
        return None


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    archive_existing: bool = True,
) -> None:
    """Transition a registered model version to a new stage.

    Args:
        model_name: Registered model name.
        version: Version string to transition.
        stage: Target stage (``"Staging"`` or ``"Production"``).
        archive_existing: If True, archive any existing versions at the
            target stage before transitioning.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info(
        "Model %s v%s transitioned to %s (archive_existing=%s)",
        model_name,
        version,
        stage,
        archive_existing,
    )


def get_model_uri(model_name: str, version: str) -> str:
    """Return the MLflow model URI for a specific registered version.

    Args:
        model_name: Registered model name.
        version: Version string.

    Returns:
        URI in the form ``models:/<model_name>/<version>``.
    """
    return f"models:/{model_name}/{version}"
