"""Causal volatility features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import OPEN, HIGH, LOW, CLOSE


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = df[CLOSE].shift(1)
    tr = pd.concat(
        [
            df[HIGH] - df[LOW],
            (df[HIGH] - prev_close).abs(),
            (df[LOW] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rename("true_range")


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / window, adjust=False).mean().rename(f"atr_{window}")


def realised_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualised realised volatility of log returns."""
    lr = np.log(df[CLOSE] / df[CLOSE].shift(1))
    return (lr.rolling(window).std() * np.sqrt(252)).rename(f"rvol_{window}")


def vol_percentile(df: pd.DataFrame, vol_window: int = 20, pct_window: int = 252) -> pd.Series:
    """Expanding or rolling percentile rank of realised vol."""
    rv = realised_vol(df, vol_window)
    return rv.rolling(pct_window).rank(pct=True).rename(f"vol_pct_{vol_window}_{pct_window}")


def vol_of_vol(df: pd.DataFrame, vol_window: int = 10, vov_window: int = 20) -> pd.Series:
    """Std dev of rolling realised vol — measures stability of vol."""
    rv = realised_vol(df, vol_window)
    return rv.rolling(vov_window).std().rename(f"vov_{vol_window}_{vov_window}")


def range_compression(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """ATR ratio: short ATR / long ATR. < 1 implies compression."""
    return (atr(df, short) / atr(df, long).replace(0, np.nan)).rename(
        f"range_compress_{short}_{long}"
    )


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["true_range"] = true_range(df)
    for w in [7, 14, 21]:
        out[f"atr_{w}"] = atr(df, w)
    for w in [10, 20, 60]:
        out[f"rvol_{w}"] = realised_vol(df, w)
    out["vol_pct_20_252"] = vol_percentile(df, 20, 252)
    out["vov_10_20"] = vol_of_vol(df, 10, 20)
    out["range_compress_5_20"] = range_compression(df, 5, 20)
    out["bb_width_20"] = _bb_width(df, 20)
    return out


def _bb_width(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mu = df[CLOSE].rolling(window).mean()
    sigma = df[CLOSE].rolling(window).std()
    return ((4 * n_std * sigma) / mu.replace(0, np.nan)).rename(f"bb_width_{window}")
