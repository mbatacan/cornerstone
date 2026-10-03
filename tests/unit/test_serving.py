"""Unit tests for the API and the shared scoring function."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from src.models.predict import LoadedModel, score
from src.models.schemas import PredictionRecord
from src.serving import app as serving
from tests.fixtures.synthetic import make_feature_df


@pytest.fixture
def loaded() -> LoadedModel:
    """A tiny fitted model wrapped the way ``load_champion`` returns it."""
    X = make_feature_df()
    y = (X["feature_a"] > 0).astype(int)
    model = DummyClassifier(strategy="prior").fit(X, y)
    return LoadedModel(
        model=model, model_name="m", version="3", run_id="r", git_sha="abc"
    )


@pytest.fixture
def client(monkeypatch, loaded: LoadedModel):
    monkeypatch.setattr(serving, "load_champion", lambda: loaded)
    monkeypatch.setattr(serving, "build_features", lambda df: df)
    with TestClient(serving.app) as c:
        yield c


def test_score_returns_one_valid_record_per_row(loaded: LoadedModel) -> None:
    records = score(loaded, make_feature_df(n=5))
    assert len(records) == 5
    assert all(isinstance(r, PredictionRecord) for r in records)
    assert {r.model_version for r in records} == {"3"}


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_model_version(client: TestClient) -> None:
    assert client.get("/ready").json() == {"model_name": "m", "model_version": "3"}


def test_predict_returns_record_per_row(client: TestClient) -> None:
    rows = make_feature_df(n=3).to_dict(orient="records")
    resp = client.post("/predict", json={"rows": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    PredictionRecord.model_validate(body[0])


def test_predict_rejects_malformed_body(client: TestClient) -> None:
    assert client.post("/predict", json={"rows": "nope"}).status_code == 422


def test_startup_fails_loudly_without_model(monkeypatch) -> None:
    def boom() -> LoadedModel:
        raise RuntimeError("no champion")

    monkeypatch.setattr(serving, "load_champion", boom)
    with pytest.raises(RuntimeError, match="no champion"), TestClient(serving.app):
        pass


def test_score_probabilities_in_range(loaded: LoadedModel) -> None:
    scores = np.array([r.score for r in score(loaded, make_feature_df(n=10))])
    assert ((scores >= 0) & (scores <= 1)).all()
