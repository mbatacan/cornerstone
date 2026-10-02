"""Unit tests for prognostics metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.prognostics.metrics import (
    alpha_lambda_accuracy,
    rul_mae,
    rul_rmse,
    rul_score,
)


def test_rul_mae_perfect():
    y = np.array([100.0, 50.0, 25.0])
    assert rul_mae(y, y) == pytest.approx(0.0)


def test_rul_mae_known_value():
    y_true = np.array([100.0, 50.0])
    y_pred = np.array([90.0, 60.0])
    assert rul_mae(y_true, y_pred) == pytest.approx(10.0)


def test_rul_rmse_perfect():
    y = np.array([10.0, 20.0, 30.0])
    assert rul_rmse(y, y) == pytest.approx(0.0)


def test_rul_rmse_known_value():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([10.0, 0.0])
    # errors = [10, 10], rmse = 10
    assert rul_rmse(y_true, y_pred) == pytest.approx(10.0)


def test_alpha_lambda_perfect():
    y_true = np.array([100.0, 50.0, 25.0])
    assert alpha_lambda_accuracy(y_true, y_true, alpha=0.2) == pytest.approx(1.0)


def test_alpha_lambda_none_within_band():
    y_true = np.array([100.0])
    y_pred = np.array([200.0])  # 100% error, way outside 20% band
    assert alpha_lambda_accuracy(y_true, y_pred, alpha=0.2) == pytest.approx(0.0)


def test_alpha_lambda_excludes_zero_true():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([0.0, 95.0])
    # y_true[0]==0 excluded; y_pred[1] within 20% of 100
    assert alpha_lambda_accuracy(y_true, y_pred, alpha=0.2) == pytest.approx(1.0)


def test_alpha_lambda_all_zeros_returns_nan():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    result = alpha_lambda_accuracy(y_true, y_pred)
    assert math.isnan(result)


def test_rul_score_perfect_is_zero():
    y_true = np.array([50.0, 100.0])
    assert rul_score(y_true, y_true) == pytest.approx(0.0)


def test_rul_score_late_prediction_penalised_more():
    y_true = np.array([100.0])
    early = rul_score(y_true, np.array([110.0]))  # over-estimate (early alert)
    late = rul_score(y_true, np.array([90.0]))  # under-estimate (late alert)
    # NASA scoring: late predictions are penalised more
    assert late > early
