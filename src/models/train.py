"""Reference training entrypoint.

Demonstrates the full template pattern:
    config load → seed → MLflow run → load data → feature engineering
    → train/eval → log metrics → log model with signature → (optionally) register

Run locally::

    CORNERSTONE_ENV=dev python -m src.models.train

Run via CLI::

    ds-template train
"""

from __future__ import annotations

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.data.load_data import load_data
from src.features.build_features import add_sepal_area
from src.logging.logger import get_logger
from src.monitoring.metrics import summarise_alert_performance
from src.tracking.mlflow_utils import log_model_with_signature, start_run
from src.utils.seed import set_all_seeds

logger = get_logger(__name__)


def train() -> str:
    """Train a classifier and log results to MLflow.

    Returns:
        MLflow model URI of the logged model artifact.
    """
    cfg = get_settings()
    set_all_seeds(cfg.training.random_seed)

    logger.info("Loading data")
    df = load_data()
    df = add_sepal_area(df)

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
                "n_estimators": cfg.training.n_estimators,
                "test_size": cfg.training.test_size,
                "random_seed": cfg.training.random_seed,
            }
        )

        logger.info("Training model")
        clf = RandomForestClassifier(
            n_estimators=cfg.training.n_estimators,
            random_state=cfg.training.random_seed,
        )
        clf.fit(X_train, y_train)

        # Binarise for alert-style metrics (class 1 = positive)
        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)

        # Binarise multi-class target for alert metric demo (class 2 vs rest)

        y_test_binary = (y_test == 2).astype(int).to_numpy()
        y_score_binary = clf.predict_proba(X_test)[:, 2]
        alert_metrics = summarise_alert_performance(
            y_test_binary,
            y_score_binary,
            threshold=cfg.alerts.default_threshold,
        )

        mlflow.log_metrics(
            {
                "val_accuracy": accuracy,
                "val_precision": alert_metrics["precision"],
                "val_recall": alert_metrics["recall"],
                "val_false_alarm_rate": alert_metrics["false_alarm_rate"],
                "val_f1": alert_metrics["f1"],
            }
        )

        logger.info(
            "Eval — accuracy: %.3f  FAR: %.3f  recall: %.3f",
            accuracy,
            alert_metrics["false_alarm_rate"],
            alert_metrics["recall"],
        )

        register_name = cfg.mlflow.registered_model_name if cfg.env != "dev" else None
        model_uri = log_model_with_signature(
            clf,
            X_train.iloc[:5],
            artifact_path="model",
            registered_model_name=register_name,
        )

        logger.info("Run complete: %s", run.info.run_id)

    return model_uri


if __name__ == "__main__":
    train()
