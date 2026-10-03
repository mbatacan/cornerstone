"""MLflow Model Registry helpers.

Thin wrappers around the MLflow client for registry operations. Versions are
promoted with aliases (Unity Catalog style); rollback is repointing the alias.
"""

from mlflow.tracking import MlflowClient

from src.logging.logger import get_logger

logger = get_logger(__name__)


def get_version_by_alias(model_name: str, alias: str) -> str:
    """Return the version number a registered model alias points to.

    Raises ``mlflow.exceptions.MlflowException`` if the model or alias is missing.
    """
    return MlflowClient().get_model_version_by_alias(model_name, alias).version


def set_model_alias(model_name: str, alias: str, version: str) -> None:
    """Point an alias (e.g. ``"champion"``) at a registered model version."""
    MlflowClient().set_registered_model_alias(model_name, alias, version)
    logger.info("Model %s alias '%s' -> v%s", model_name, alias, version)


def get_model_uri(model_name: str, alias: str) -> str:
    """Return the MLflow model URI ``models:/<model_name>@<alias>``."""
    return f"models:/{model_name}@{alias}"
