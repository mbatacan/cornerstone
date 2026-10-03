"""Local loop with no Databricks: register -> promote -> load champion -> score."""

import pytest
from sklearn.dummy import DummyClassifier

from src.config.settings import get_settings
from src.models.predict import load_champion, score
from src.models.registry import set_model_alias
from src.tracking.mlflow_utils import log_model_with_signature, start_run
from tests.fixtures.synthetic import make_feature_df


@pytest.mark.slow
def test_register_promote_load_score(tmp_path, monkeypatch) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("CORNERSTONE_MLFLOW__TRACKING_URI", uri)
    cfg = get_settings()

    X = make_feature_df()
    model = DummyClassifier(strategy="prior").fit(X, (X["feature_a"] > 0).astype(int))
    with start_run(
        "local-loop", env="local", tracking_uri=uri, experiment_name="local-loop"
    ):
        log_model_with_signature(
            model, X.iloc[:5], registered_model_name=cfg.mlflow.registered_model_name
        )
    set_model_alias(cfg.mlflow.registered_model_name, cfg.mlflow.model_alias, "1")

    loaded = load_champion()
    records = score(loaded, make_feature_df(n=4))
    assert len(records) == 4
    assert loaded.version == "1"
