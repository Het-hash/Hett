"""Causal mean-reversion features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CLOSE, HIGH, LOW


def rolling_zscore(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling z-score of close price (causal)."""
    mu = df[CLOSE].rolling(window).mean()
    sigma = df[CLOSE].rolling(window).std()
    return ((df[CLOSE] - mu) / sigma.replace(0, np.nan)).rename(f"zscore_{window}")


def bollinger_position(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """
    Position within Bollinger Bands: 0 = lower band, 1 = upper band.
    Values outside [0,1] indicate band breach.
    """
    mu = df[CLOSE].rolling(window).mean()
    sigma = df[CLOSE].rolling(window).std()
    upper = mu + n_std * sigma
    lower = mu - n_std * sigma
    return ((df[CLOSE] - lower) / (upper - lower).replace(0, np.nan)).rename(
        f"bb_pos_{window}"
    )


def bollinger_width(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger Band width normalised by midband: (upper - lower) / mid."""
    mu = df[CLOSE].rolling(window).mean()
    sigma = df[CLOSE].rolling(window).std()
    upper = mu + n_std * sigma
    lower = mu - n_std * sigma
    return ((upper - lower) / mu.replace(0, np.nan)).rename(f"bb_width_{window}")


def short_term_abnormal_return(df: pd.DataFrame, short: int = 3, long: int = 20) -> pd.Series:
    """
    Short-term return minus recent baseline:
    ret_short - mean(daily_ret, long).
    """
    daily = df[CLOSE].pct_change()
    recent_mean = daily.rolling(long).mean()
    short_ret = df[CLOSE].pct_change(short)
    return (short_ret - recent_mean * short).rename(f"abnormal_ret_{short}_{long}")


def build_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in [10, 20, 60]:
        out[f"zscore_{w}"] = rolling_zscore(df, w)
    for w in [20]:
        out[f"bb_pos_{w}"] = bollinger_position(df, w)
        out[f"bb_width_{w}"] = bollinger_width(df, w)
    out["abnormal_ret_3_20"] = short_term_abnormal_return(df, 3, 20)
    return out
