"""Unit tests for alerting performance metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.metrics import (
    false_alarm_rate,
    precision_at_k,
    summarise_alert_performance,
)


def test_precision_at_k_perfect():
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert precision_at_k(y_true, y_score, k=2) == 1.0


def test_precision_at_k_zero():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert precision_at_k(y_true, y_score, k=2) == 0.0


def test_precision_at_k_clamps_to_n():
    y_true = np.array([1, 0, 1])
    y_score = np.array([0.9, 0.5, 0.1])
    # k=10 should not error; clamps to len=3
    result = precision_at_k(y_true, y_score, k=10)
    assert 0.0 <= result <= 1.0


def test_precision_at_k_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(np.array([1, 0]), np.array([0.9, 0.1]), k=0)


def test_false_alarm_rate_no_false_alarms():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert false_alarm_rate(y_true, y_pred) == 0.0


def test_false_alarm_rate_all_false_alarms():
    y_true = np.array([0, 0])
    y_pred = np.array([1, 1])
    assert false_alarm_rate(y_true, y_pred) == 1.0


def test_false_alarm_rate_no_negatives():
    y_true = np.array([1, 1])
    y_pred = np.array([1, 1])
    assert false_alarm_rate(y_true, y_pred) == 0.0


def test_summarise_alert_performance_keys():
    y_true = np.array([1, 0, 1, 0, 1])
    y_score = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
    result = summarise_alert_performance(y_true, y_score, threshold=0.5)
    assert set(result.keys()) == {"precision", "recall", "false_alarm_rate", "f1"}


def test_summarise_alert_performance_with_k():
    y_true = np.array([1, 0, 1, 0, 1])
    y_score = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
    result = summarise_alert_performance(y_true, y_score, threshold=0.5, k=3)
    assert "precision_at_k" in result
    assert 0.0 <= result["precision_at_k"] <= 1.0
