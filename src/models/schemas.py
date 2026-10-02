"""Schema for rows written by the scoring pipeline."""

from datetime import datetime

from pydantic import BaseModel


class PredictionRecord(BaseModel):
    """One scored row. Every row is validated against this before it is written."""

    row_id: int
    prediction: int
    score: float
    model_name: str
    model_version: str
    run_id: str
    git_sha: str
    scored_at: datetime
