"""Global reproducible random state management."""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds for random, numpy. Call once at process start."""
    random.seed(seed)
    np.random.seed(seed)
    # scikit-learn uses numpy's global seed; no separate call needed.


def get_rng(seed: int = 42) -> np.random.Generator:
    """Return a seeded Generator for isolated use in functions."""
    return np.random.default_rng(seed)
