"""Alerting performance metrics for post-deploy monitoring.

These metrics evaluate how well the alert model is performing in production
against labeled outcomes (e.g. confirmed failures, maintenance events).

Definitions:
    - precision@k: Of the top-k scored assets, what fraction had true events?
    - false_alarm_rate (FAR): FP / (FP + TN) — fraction of non-events that triggered alerts.
    - recall: TP / (TP + FN) — fraction of true events that were caught.
    - lead_time: For true-positive alerts, how many hours before the event did the alert fire?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    """Fraction of true positives in the top-k highest-scored predictions.

    Args:
        y_true: Binary ground-truth labels (0/1), shape (n,).
        y_score: Model scores, shape (n,).
        k: Number of top predictions to consider.

    Returns:
        Precision@k in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_k_idx]))


def false_alarm_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """False alarm rate: FP / (FP + TN).

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_pred: Binary predicted labels (0/1).

    Returns:
        FAR in [0, 1].  Returns 0.0 if there are no true negatives.
    """
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def alert_lead_time_stats(
    events_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    asset_col: str = "source_asset_id",
    event_time_col: str = "event_time",
    alert_time_col: str = "emitted_at",
) -> pd.DataFrame:
    """Compute per-asset lead time between the first alert and the confirmed event.

    For each asset in ``events_df``, finds the earliest alert in ``alerts_df``
    that fired before the event and computes the lead time in hours.

    Args:
        events_df: DataFrame of confirmed failure/maintenance events with columns
            ``[asset_col, event_time_col]``.
        alerts_df: DataFrame of emitted alerts with columns
            ``[asset_col, alert_time_col]``.
        asset_col: Column identifying the asset in both DataFrames.
        event_time_col: Column with the event timestamp in ``events_df``.
        alert_time_col: Column with the alert timestamp in ``alerts_df``.

    Returns:
        DataFrame with columns: ``[asset_col, event_time, first_alert_time,
        lead_time_hours, caught]``.
        ``caught`` is True when at least one alert fired before the event.
    """
    events_df = events_df.copy()
    alerts_df = alerts_df.copy()

    events_df[event_time_col] = pd.to_datetime(events_df[event_time_col], utc=True)
    alerts_df[alert_time_col] = pd.to_datetime(alerts_df[alert_time_col], utc=True)

    rows = []
    for _, event_row in events_df.iterrows():
        asset = event_row[asset_col]
        event_time = event_row[event_time_col]

        prior_alerts = alerts_df[
            (alerts_df[asset_col] == asset) & (alerts_df[alert_time_col] < event_time)
        ]

        if prior_alerts.empty:
            rows.append(
                {
                    asset_col: asset,
                    "event_time": event_time,
                    "first_alert_time": pd.NaT,
                    "lead_time_hours": float("nan"),
                    "caught": False,
                }
            )
        else:
            first_alert = prior_alerts[alert_time_col].min()
            lead_hours = (event_time - first_alert).total_seconds() / 3600
            rows.append(
                {
                    asset_col: asset,
                    "event_time": event_time,
                    "first_alert_time": first_alert,
                    "lead_time_hours": lead_hours,
                    "caught": True,
                }
            )

    return pd.DataFrame(rows)


def summarise_alert_performance(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    k: int | None = None,
) -> dict[str, float]:
    """Return a dict of standard alert performance metrics at a given threshold.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Model scores in [0, 1].
        threshold: Decision threshold.
        k: If provided, also computes precision@k.

    Returns:
        Dict with keys: precision, recall, false_alarm_rate, f1,
        and optionally precision_at_k.
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    result = {
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "f1": f1,
    }
    if k is not None:
        result["precision_at_k"] = precision_at_k(y_true, y_score, k)
    return result
