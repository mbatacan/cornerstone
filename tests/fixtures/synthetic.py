"""Synthetic DataFrames for use in tests.

All fixtures produce small, deterministic datasets. Never use production data in tests.
"""

import numpy as np
import pandas as pd


def make_feature_df(n: int = 200, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Return a feature DataFrame; ``shift`` moves ``feature_a``'s mean."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(shift, 1, n),
            "feature_b": rng.uniform(0, 10, n),
        }
    )
