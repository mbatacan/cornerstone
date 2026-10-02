"""Population drift detection utilities.

Use these to compare the distribution of features or scores between
a reference dataset (e.g. training data) and a production window.
A significant drift signal means the model may be operating out of
its training distribution and should be flagged for retraining review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI < 0.1  → no significant change
    PSI 0.1–0.2 → moderate change, monitor
    PSI > 0.2  → significant change, investigate / retrain

    Args:
        reference: 1-D array of reference (baseline) values.
        current: 1-D array of current production values.
        n_bins: Number of equal-width bins derived from the reference distribution.
        epsilon: Small constant added to bin counts to avoid log(0).

    Returns:
        PSI value (float ≥ 0).
    """
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * n_bins)
    cur_pct = (cur_counts + epsilon) / (len(current) + epsilon * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Two-sample Kolmogorov-Smirnov test for distribution shift.

    Args:
        reference: 1-D reference distribution array.
        current: 1-D current distribution array.
        alpha: Significance level for the hypothesis test.

    Returns:
        Dict with keys: ``ks_statistic``, ``p_value``, ``drift_detected``
        (True when p_value < alpha).
    """
    result = stats.ks_2samp(reference, current)
    return {
        "ks_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drift_detected": bool(result.pvalue < alpha),
    }


def feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> pd.DataFrame:
    """Run PSI and KS drift tests across all numeric feature columns.

    Args:
        reference_df: Reference (training/baseline) feature DataFrame.
        current_df: Current production feature DataFrame.
        numeric_cols: List of columns to test.  Defaults to all shared numeric
            columns between the two DataFrames.
        psi_threshold: PSI value above which a feature is flagged as drifted.
        ks_alpha: KS test significance level.

    Returns:
        DataFrame with one row per feature and columns:
        feature, psi, ks_statistic, ks_p_value, psi_drift, ks_drift, any_drift.
    """
    if numeric_cols is None:
        shared = set(reference_df.columns) & set(current_df.columns)
        numeric_cols = [
            c for c in shared if pd.api.types.is_numeric_dtype(reference_df[c])
        ]

    rows = []
    for col in numeric_cols:
        ref_vals = reference_df[col].dropna().to_numpy()
        cur_vals = current_df[col].dropna().to_numpy()

        psi_val = population_stability_index(ref_vals, cur_vals)
        ks_result = ks_drift_test(ref_vals, cur_vals, alpha=ks_alpha)

        rows.append(
            {
                "feature": col,
                "psi": psi_val,
                "ks_statistic": ks_result["ks_statistic"],
                "ks_p_value": ks_result["p_value"],
                "psi_drift": psi_val > psi_threshold,
                "ks_drift": ks_result["drift_detected"],
                "any_drift": psi_val > psi_threshold or ks_result["drift_detected"],
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
