"""Shared pytest fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.schema import OPEN, HIGH, LOW, CLOSE, VOLUME, ADJ_CLOSE, SYMBOL, SOURCE, TIMESTAMP


@pytest.fixture
def synthetic_ohlcv():
    """
    200-bar synthetic OHLCV DataFrame with a known random-walk price process.
    GBM with mu=0.0003, sigma=0.01.
    """
    np.random.seed(42)
    n = 400
    dates = pd.bdate_range("2015-01-02", periods=n, freq="B")
    log_ret = np.random.normal(0.0003, 0.01, n)
    close = 10000 * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    df = pd.DataFrame({
        OPEN: open_,
        HIGH: high,
        LOW: low,
        CLOSE: close,
        ADJ_CLOSE: close,
        VOLUME: np.random.randint(1_000_000, 10_000_000, n).astype(float),
        SYMBOL: "SYNTHETIC",
        SOURCE: "test",
    }, index=dates)
    df.index.name = TIMESTAMP
    return df


@pytest.fixture
def synthetic_ohlcv_with_leakage(synthetic_ohlcv):
    """Synthetic data where a 'feature' is calculated using future information."""
    df = synthetic_ohlcv.copy()
    # Future-centred rolling mean — WRONG (should NOT be used as a feature)
    df["leaky_feature"] = df[CLOSE].rolling(5, center=True).mean()
    return df
