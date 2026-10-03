"""Seed utilities for reproducible experiments."""

import random

import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and optionally PyTorch.

    Call this at the start of every training script before any data loading
    or model initialisation.

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # ty: ignore[unresolved-import]  # optional dependency

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
