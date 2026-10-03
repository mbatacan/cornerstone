"""Training entrypoint skeleton.

The orchestration is in place; implement ``fit_model`` and ``evaluate``.

Pattern:
    config load → seed → MLflow run → load data → feature engineering
    → train/eval → log metrics → log model with signature → (optionally) register

Run locally::

    CORNERSTONE_ENV=local python -m src.models.train

Run via CLI::

    ds-template train
"""

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.data.load_data import load_data
from src.features.build_features import build_features
from src.logging.logger import get_logger
from src.tracking.mlflow_utils import log_model_with_signature, start_run
from src.utils.seed import set_all_seeds

logger = get_logger(__name__)


def fit_model(X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """Fit and return the model. Wrap preprocessing and estimator in a Pipeline."""
    raise NotImplementedError("Implement fit_model for this project")


def evaluate(
    model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Return metrics chosen from the business decision, keyed by metric name."""
    raise NotImplementedError("Implement evaluate for this project")


def train() -> str:
    """Train a classifier and log results to MLflow.

    Returns:
        MLflow model URI of the logged model artifact.
    """
    cfg = get_settings()
    set_all_seeds(cfg.training.random_seed)

    if cfg.mlflow.registry_uri:
        mlflow.set_registry_uri(cfg.mlflow.registry_uri)

    logger.info("Loading data")
    df = load_data()
    df = build_features(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_seed,
        stratify=y,
    )

    with start_run(
        name=f"train-{cfg.project_name}",
        env=cfg.env,
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
    ) as run:
        mlflow.log_params(
            {
                "test_size": cfg.training.test_size,
                "random_seed": cfg.training.random_seed,
            }
        )

        logger.info("Training model")
        model = fit_model(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        logger.info("Eval metrics: %s", metrics)

        model_uri = log_model_with_signature(
            model,
            X_train.iloc[:5],
            artifact_path="model",
            registered_model_name=cfg.mlflow.registered_model_name,
        )

        logger.info("Run complete: %s", run.info.run_id)

    return model_uri


if __name__ == "__main__":
    train()
