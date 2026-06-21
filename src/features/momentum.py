"""Causal momentum features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CLOSE, HIGH, LOW


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder RSI using EWM — fully causal."""
    delta = df[CLOSE].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{window}")


def rate_of_change(df: pd.DataFrame, window: int) -> pd.Series:
    """Rate of change: (close_t / close_{t-window} - 1) * 100."""
    return ((df[CLOSE] / df[CLOSE].shift(window) - 1) * 100).rename(f"roc_{window}")


def consecutive_up(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive positive close-to-close sessions."""
    ret = df[CLOSE].pct_change()
    up = (ret > 0).astype(int)
    streak = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    return streak.rename("consecutive_up")


def consecutive_down(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive negative close-to-close sessions."""
    ret = df[CLOSE].pct_change()
    down = (ret < 0).astype(int)
    streak = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)
    return streak.rename("consecutive_down")


def dist_from_high(df: pd.DataFrame, window: int) -> pd.Series:
    """Distance of current close from rolling N-day high."""
    rolling_high = df[HIGH].rolling(window).max()
    return ((df[CLOSE] - rolling_high) / rolling_high).rename(f"dist_high_{window}")


def dist_from_low(df: pd.DataFrame, window: int) -> pd.Series:
    """Distance of current close from rolling N-day low."""
    rolling_low = df[LOW].rolling(window).min()
    return ((df[CLOSE] - rolling_low) / rolling_low).rename(f"dist_low_{window}")


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in [7, 14]:
        out[f"rsi_{w}"] = rsi(df, w)
    for w in [5, 10, 20]:
        out[f"roc_{w}"] = rate_of_change(df, w)
    out["consecutive_up"] = consecutive_up(df)
    out["consecutive_down"] = consecutive_down(df)
    for w in [20, 52]:
        out[f"dist_high_{w}"] = dist_from_high(df, w)
        out[f"dist_low_{w}"] = dist_from_low(df, w)
    return out
