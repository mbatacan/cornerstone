"""Prognostics-specific evaluation metrics.

Standard metrics for Remaining Useful Life (RUL) prediction tasks.
These supplement the alerting metrics in ``src.monitoring.metrics`` for
models that produce a continuous RUL estimate rather than a binary alert.

References:
    Saxena et al. (2008) "Metrics for Evaluating Performance of Prognostic
    Algorithms."  IEEE Aerospace Conference.
"""

from __future__ import annotations

import numpy as np


def rul_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error of RUL predictions (in the same unit as y_true).

    Args:
        y_true: True RUL values (e.g. hours until failure).
        y_pred: Predicted RUL values.

    Returns:
        MAE (float).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rul_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error of RUL predictions.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        RMSE (float).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def alpha_lambda_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """Fraction of predictions within ±alpha of the true RUL.

    The alpha-lambda metric (Saxena et al. 2008) measures what proportion of
    RUL predictions fall within a relative cone around the true value.  A
    prediction at time t is considered accurate if::

        |y_pred - y_true| / y_true <= alpha   (when y_true > 0)

    Assets where ``y_true == 0`` are excluded from the calculation.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        alpha: Relative tolerance band (default 0.2 = ±20%).

    Returns:
        Fraction of predictions within the alpha band, in [0, 1].
    """
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    relative_error = np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]
    return float(np.mean(relative_error <= alpha))


def prognostic_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """Average prognostic horizon: mean RUL at which predictions enter the alpha band.

    For each sample, the prognostic horizon is the true RUL at the earliest
    time the prediction falls within ±alpha of the true value and stays there.
    This is a simplified scalar version — in practice this is computed over a
    time series per asset.

    Here, interpreted as the mean ``y_true`` value of samples where the
    prediction is already within the alpha band.

    Args:
        y_true: True RUL values (sorted descending by time, highest RUL first).
        y_pred: Predicted RUL values in the same order.
        alpha: Relative tolerance (default 0.2).

    Returns:
        Mean y_true value across samples within the alpha band (float).
        Returns ``nan`` if no samples are within the band.
    """
    mask = y_true > 0
    within = mask & (np.abs(y_pred - y_true) / np.where(mask, y_true, 1) <= alpha)
    if within.sum() == 0:
        return float("nan")
    return float(np.mean(y_true[within]))


def rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA CMAPSS asymmetric scoring function for RUL.

    Penalises late predictions (under-estimated RUL) more heavily than
    early predictions.  Commonly used with the CMAPSS turbofan dataset.

    Score = sum(exp(-d/13) - 1  if d < 0 else exp(d/10) - 1)
    where d = y_pred - y_true (positive = early prediction).

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Aggregate score (lower is better, 0 is perfect).
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))
