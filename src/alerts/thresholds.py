"""Threshold calibration utilities for alert models.

Helpers for selecting and evaluating decision thresholds on held-out
validation data.  These are intentionally separate from training logic
so the threshold can be re-tuned after model training without retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def find_threshold_at_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float = 0.90,
) -> float:
    """Return the lowest threshold that achieves at least ``target_precision``.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Model scores in [0, 1].
        target_precision: Minimum required precision (default 0.90).

    Returns:
        Threshold value.  Returns 1.0 if no threshold meets the target
        (i.e. the model cannot achieve the required precision).
    """
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve appends a trailing precision=1 with no threshold
    for prec, thresh in zip(precision[:-1], thresholds):
        if prec >= target_precision:
            return float(thresh)
    return 1.0


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate precision, recall, and false-alarm rate across a range of thresholds.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Model scores in [0, 1].
        thresholds: Array of thresholds to evaluate.  Defaults to
            ``np.linspace(0.1, 0.9, 17)``.

    Returns:
        DataFrame with columns: threshold, precision, recall, false_alarm_rate, f1.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    rows = []
    n_neg = int((y_true == 0).sum())
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / n_neg if n_neg > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "false_alarm_rate": far,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)
