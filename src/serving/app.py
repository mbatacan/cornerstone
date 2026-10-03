"""FastAPI app serving the champion model.

Batch scoring and this API share ``load_champion``, ``build_features`` and
``score``, so feature logic and the output schema cannot drift between them.

Run locally::

    uv run uvicorn src.serving.app:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.features.build_features import build_features
from src.logging.logger import get_logger
from src.models.predict import load_champion, score
from src.models.schemas import PredictionRecord

logger = get_logger(__name__)


class PredictRequest(BaseModel):
    """Request body. Replace the generic ``rows`` with typed feature fields.

    Typed fields give the OpenAPI spec (and the generated TypeScript client)
    real input types and make FastAPI reject bad input with a 422.
    """

    rows: list[dict[str, float | int | str | None]]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the champion model once at startup. A failed load crashes startup."""
    app.state.loaded = load_champion()
    yield


app = FastAPI(title="ds-template", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    """Readiness probe: reports which model version is loaded."""
    loaded = request.app.state.loaded
    return {"model_name": loaded.model_name, "model_version": loaded.version}


@app.post("/predict")
def predict(body: PredictRequest, request: Request) -> list[PredictionRecord]:
    """Score the rows and return one validated prediction record per row."""
    features = build_features(pd.DataFrame(body.rows))
    return score(request.app.state.loaded, features)
