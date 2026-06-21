"""Causal breakout features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CLOSE, HIGH, LOW
from src.features.volatility import atr


def high_breakout(df: pd.DataFrame, window: int) -> pd.Series:
    """
    True if current close exceeds the highest close of the prior `window` sessions.
    Uses shift(1) so today's bar is not in the rolling window.
    """
    prior_high = df[CLOSE].shift(1).rolling(window).max()
    return (df[CLOSE] > prior_high).rename(f"high_breakout_{window}")


def low_breakdown(df: pd.DataFrame, window: int) -> pd.Series:
    """True if current close breaks below the lowest close of the prior `window` sessions."""
    prior_low = df[CLOSE].shift(1).rolling(window).min()
    return (df[CLOSE] < prior_low).rename(f"low_breakdown_{window}")


def breakout_strength(df: pd.DataFrame, window: int) -> pd.Series:
    """How far beyond the prior high the close is, normalised by ATR."""
    prior_high = df[CLOSE].shift(1).rolling(window).max()
    atr_14 = atr(df, 14)
    return ((df[CLOSE] - prior_high) / atr_14.replace(0, np.nan)).rename(
        f"brkout_strength_{window}"
    )


def failed_breakout(df: pd.DataFrame, window: int, lookback: int = 3) -> pd.Series:
    """
    True if a high breakout occurred within `lookback` sessions but price has since
    returned below the breakout level (failed breakout reversal signal).
    """
    brkout = high_breakout(df, window).astype(float)
    had_breakout = brkout.shift(1).rolling(lookback).max() > 0
    prior_high = df[CLOSE].shift(1 + lookback).rolling(window).max()
    return (had_breakout & (df[CLOSE] < prior_high)).rename(
        f"failed_brkout_{window}_{lookback}"
    )


def build_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in [20, 50, 100]:
        out[f"high_breakout_{w}"] = high_breakout(df, w)
        out[f"low_breakdown_{w}"] = low_breakdown(df, w)
        out[f"brkout_strength_{w}"] = breakout_strength(df, w)
    for w in [20, 50]:
        out[f"failed_brkout_{w}_3"] = failed_breakout(df, w, 3)
    return out
