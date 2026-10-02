"""Unit tests for drift metrics."""

import numpy as np

from src.monitoring.drift import (
    feature_drift_report,
    ks_drift_test,
    population_stability_index,
)
from tests.fixtures.synthetic import make_feature_df


def test_psi_identical_distributions_is_near_zero() -> None:
    x = np.random.default_rng(0).normal(size=1000)
    assert population_stability_index(x, x) < 0.01


def test_psi_flags_a_large_shift() -> None:
    rng = np.random.default_rng(0)
    ref, cur = rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)
    assert population_stability_index(ref, cur) > 0.2


def test_ks_detects_shift_and_not_same_sample() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 500)
    assert ks_drift_test(ref, rng.normal(3, 1, 500))["drift_detected"]
    assert not ks_drift_test(ref, ref)["drift_detected"]


def test_feature_drift_report_ranks_shifted_feature_first() -> None:
    report = feature_drift_report(make_feature_df(), make_feature_df(shift=3.0, seed=1))
    assert report.iloc[0]["feature"] == "feature_a"
