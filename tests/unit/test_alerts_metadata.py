"""Unit tests for AlertMetadata validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.alerts.metadata import AlertMetadata
from tests.fixtures.synthetic import make_alert_metadata_kwargs


def test_valid_metadata_round_trips():
    kwargs = make_alert_metadata_kwargs()
    meta = AlertMetadata(**kwargs)
    assert meta.model_name == "test-model"
    assert meta.env == "dev"
    assert meta.severity == "warn"


def test_alert_id_auto_generated():
    meta = AlertMetadata(**make_alert_metadata_kwargs())
    assert len(meta.alert_id) == 36  # UUID4 string


def test_emitted_at_auto_filled():
    meta = AlertMetadata(**make_alert_metadata_kwargs())
    assert meta.emitted_at.tzinfo is not None


def test_to_dict_returns_strings_for_datetimes():
    meta = AlertMetadata(**make_alert_metadata_kwargs())
    d = meta.to_dict()
    assert isinstance(d["emitted_at"], str)
    assert isinstance(d["input_window_start"], str)
    assert isinstance(d["input_window_end"], str)


def test_invalid_window_order_raises():
    kwargs = make_alert_metadata_kwargs()
    now = datetime.now(timezone.utc)
    kwargs["input_window_start"] = now
    kwargs["input_window_end"] = now - timedelta(hours=1)
    with pytest.raises(ValidationError, match="input_window_start must be before"):
        AlertMetadata(**kwargs)


def test_invalid_env_raises():
    kwargs = make_alert_metadata_kwargs()
    kwargs["env"] = "production"  # not in Literal["dev","staging","prod"]
    with pytest.raises(ValidationError):
        AlertMetadata(**kwargs)


def test_invalid_severity_raises():
    kwargs = make_alert_metadata_kwargs()
    kwargs["severity"] = "high"  # not in Literal
    with pytest.raises(ValidationError):
        AlertMetadata(**kwargs)


def test_optional_fields_default_to_none():
    meta = AlertMetadata(**make_alert_metadata_kwargs())
    assert meta.prognostic_horizon_hours is None
    assert meta.explanation_uri is None
    assert meta.feature_snapshot_uri is None
    assert meta.upstream_pipeline_run_id is None


def test_missing_required_field_raises():
    kwargs = make_alert_metadata_kwargs()
    del kwargs["model_name"]
    with pytest.raises(ValidationError):
        AlertMetadata(**kwargs)
