import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with engineered features added.

    Shared by training and scoring so features are never reimplemented at inference.
    """
    raise NotImplementedError("Implement build_features for this project")
