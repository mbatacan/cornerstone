"""AlertMetadata — the schema every emitted alert must carry.

This pydantic model defines the contract between the DS/MLE team and the
deploy team.  Every alert written by ``emit_alert`` is validated against
this model before being persisted, so the downstream consumers (Databricks
Delta table, dashboards, ops team) can always query:

    "Which model version fired this alert, on what input window,
     at what threshold, for which asset?"

Add project-specific fields by subclassing AlertMetadata and passing your
subclass to ``emit_alert``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AlertMetadata(BaseModel):
    """Metadata attached to every alert emitted by this pipeline.

    Args:
        alert_id: Unique identifier for this alert (auto-generated UUID4).
        emitted_at: UTC timestamp when the alert was generated (auto-filled).
        env: Deployment environment. Prevents dev noise from reaching prod dashboards.
        model_name: Registered MLflow model name.
        model_version: Registered MLflow model version string.
        registered_model_uri: Full MLflow URI, e.g. ``models:/my-model/3``.
        mlflow_run_id: MLflow run that produced the scoring artifact.
        git_sha: Git commit SHA of the code that emitted this alert.
        source_system: Upstream telemetry system (e.g. ``"aircraft-health-telemetry"``).
        source_asset_id: Asset being monitored (tail number, serial number, equipment ID).
        input_window_start: Start of the data window that was scored.
        input_window_end: End of the data window that was scored.
        feature_snapshot_uri: Optional pointer to the exact feature frame used
            (Delta table path or MLflow artifact URI).
        score: Raw model output score.
        threshold: Decision threshold that was crossed to fire this alert.
        severity: Routing hint for downstream consumers.
        prognostic_horizon_hours: For RUL-style models — predicted hours until failure.
        explanation_uri: Optional path to a SHAP/attribution artifact.
        upstream_pipeline_run_id: Databricks job run ID of the scoring job.
    """

    # Identity
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    emitted_at: datetime = Field(default_factory=_utcnow)
    env: Literal["dev", "staging", "prod"] = "dev"

    # Model provenance
    model_name: str
    model_version: str
    registered_model_uri: str
    mlflow_run_id: str
    git_sha: str

    # Asset / data source
    source_system: str
    source_asset_id: str
    input_window_start: datetime
    input_window_end: datetime
    feature_snapshot_uri: Optional[str] = None

    # Alert decision
    score: float
    threshold: float
    severity: Literal["info", "warn", "critical"] = "warn"

    # Prognostics extension
    prognostic_horizon_hours: Optional[float] = None

    # Traceability
    explanation_uri: Optional[str] = None
    upstream_pipeline_run_id: Optional[str] = None

    @model_validator(mode="after")
    def _window_order(self) -> "AlertMetadata":
        if self.input_window_start >= self.input_window_end:
            raise ValueError("input_window_start must be before input_window_end")
        return self

    def to_dict(self) -> dict:
        """Return a flat dict suitable for a DataFrame row or Delta write."""
        d = self.model_dump()
        # Ensure datetime fields are UTC-aware strings for Delta compatibility
        for key in ("emitted_at", "input_window_start", "input_window_end"):
            val = d[key]
            if isinstance(val, datetime):
                d[key] = val.isoformat()
        return d
