"""Causal return features — all computed using only past/current bar data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import OPEN, HIGH, LOW, CLOSE


def close_to_close(df: pd.DataFrame) -> pd.Series:
    """Simple percentage return from previous close to current close."""
    return df[CLOSE].pct_change().rename("ret_cc")


def log_return(df: pd.DataFrame) -> pd.Series:
    """Log return: ln(close_t / close_{t-1})."""
    return np.log(df[CLOSE] / df[CLOSE].shift(1)).rename("ret_log")


def open_to_close(df: pd.DataFrame) -> pd.Series:
    """Intraday return: (close - open) / open."""
    return ((df[CLOSE] - df[OPEN]) / df[OPEN]).rename("ret_oc")


def overnight_gap(df: pd.DataFrame) -> pd.Series:
    """Gap from prior close to today's open: (open_t - close_{t-1}) / close_{t-1}."""
    return ((df[OPEN] - df[CLOSE].shift(1)) / df[CLOSE].shift(1)).rename("gap_pct")


def rolling_return(df: pd.DataFrame, window: int) -> pd.Series:
    """Cumulative close-to-close return over the past `window` sessions (causal)."""
    return df[CLOSE].pct_change(window).rename(f"ret_{window}d")


def rolling_log_return(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling sum of log returns over `window` sessions."""
    lr = log_return(df)
    return lr.rolling(window).sum().rename(f"ret_log_{window}d")


def momentum_acceleration(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """Return acceleration: short-term return minus long-term return."""
    return (rolling_return(df, short) - rolling_return(df, long)).rename(
        f"mom_accel_{short}_{long}"
    )


def build_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach all return features to a copy of df."""
    out = pd.DataFrame(index=df.index)
    out["ret_cc"] = close_to_close(df)
    out["ret_log"] = log_return(df)
    out["ret_oc"] = open_to_close(df)
    out["gap_pct"] = overnight_gap(df)
    for w in [3, 5, 10, 20, 60]:
        out[f"ret_{w}d"] = rolling_return(df, w)
    for w in [5, 20]:
        out[f"ret_log_{w}d"] = rolling_log_return(df, w)
    out["mom_accel_5_20"] = momentum_acceleration(df, 5, 20)
    return out
