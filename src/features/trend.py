"""Causal trend features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CLOSE


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def price_minus_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Distance of close from SMA, as fraction of SMA."""
    ma = sma(df[CLOSE], window)
    return ((df[CLOSE] - ma) / ma).rename(f"sma_dist_{window}")


def price_minus_ema(df: pd.DataFrame, span: int) -> pd.Series:
    ma = ema(df[CLOSE], span)
    return ((df[CLOSE] - ma) / ma).rename(f"ema_dist_{span}")


def ma_cross_signal(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """
    +1 if fast SMA > slow SMA, -1 otherwise (causal).
    Signal is available at close of bar T.
    """
    fast_ma = sma(df[CLOSE], fast)
    slow_ma = sma(df[CLOSE], slow)
    return np.sign(fast_ma - slow_ma).rename(f"ma_cross_{fast}_{slow}")


def sma_slope(df: pd.DataFrame, window: int, slope_window: int = 5) -> pd.Series:
    """
    Fractional slope of SMA: (sma_t - sma_{t-k}) / sma_{t-k}.
    Uses only data available at close T.
    """
    ma = sma(df[CLOSE], window)
    return ((ma - ma.shift(slope_window)) / ma.shift(slope_window)).rename(
        f"sma_slope_{window}_{slope_window}"
    )


def pct_closes_above_sma(df: pd.DataFrame, price_window: int, sma_window: int) -> pd.Series:
    """Fraction of the last `price_window` closes that were above their SMA."""
    ma = sma(df[CLOSE], sma_window)
    above = (df[CLOSE] > ma).astype(float)
    return above.rolling(price_window).mean().rename(f"pct_above_sma_{sma_window}_{price_window}")


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in [20, 50, 100, 200]:
        out[f"sma_dist_{w}"] = price_minus_sma(df, w)
    for s in [20, 50, 200]:
        out[f"ema_dist_{s}"] = price_minus_ema(df, s)
    out["ma_cross_20_100"] = ma_cross_signal(df, 20, 100)
    out["ma_cross_50_200"] = ma_cross_signal(df, 50, 200)
    for w, sw in [(200, 20), (200, 63)]:
        out[f"sma_slope_{w}_{sw}"] = sma_slope(df, w, sw)
    out["pct_above_sma200_50d"] = pct_closes_above_sma(df, 50, 200)
    return out
