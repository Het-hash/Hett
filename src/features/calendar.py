"""Calendar / seasonal features (causal — based on today's date)."""
from __future__ import annotations

import pandas as pd


def day_of_week(df: pd.DataFrame) -> pd.Series:
    """0=Monday … 4=Friday."""
    return df.index.dayofweek.rename("day_of_week") if hasattr(df.index, "dayofweek") else pd.Series(df.index.map(lambda x: x.dayofweek), index=df.index, name="day_of_week")


def month(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.month, index=df.index, name="month")


def is_month_start(df: pd.DataFrame, n_sessions: int = 3) -> pd.Series:
    """True for the first `n_sessions` trading sessions of each month."""
    month_s = pd.Series(df.index.month, index=df.index)
    new_month = month_s != month_s.shift(1)
    result = pd.Series(False, index=df.index)
    for i in range(n_sessions):
        result |= new_month.shift(-i).fillna(False)
    return result.rename("is_month_start")


def is_month_end(df: pd.DataFrame, n_sessions: int = 3) -> pd.Series:
    """True for the last `n_sessions` trading sessions of each month."""
    month_s = pd.Series(df.index.month, index=df.index)
    end_of_month = month_s != month_s.shift(-1)
    result = pd.Series(False, index=df.index)
    for i in range(n_sessions):
        result |= end_of_month.shift(i).fillna(False)
    return result.rename("is_month_end")


def is_quarter_end(df: pd.DataFrame, n_sessions: int = 5) -> pd.Series:
    """True for the last `n_sessions` sessions of each calendar quarter."""
    quarter = pd.Series(df.index.quarter, index=df.index)
    end_of_quarter = quarter != quarter.shift(-1)
    result = pd.Series(False, index=df.index)
    for i in range(n_sessions):
        result |= end_of_quarter.shift(i).fillna(False)
    return result.rename("is_quarter_end")


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["day_of_week"] = pd.Series(df.index.dayofweek, index=df.index)
    out["month"] = pd.Series(df.index.month, index=df.index)
    out["quarter"] = pd.Series(df.index.quarter, index=df.index)
    out["is_month_start"] = is_month_start(df, 3)
    out["is_month_end"] = is_month_end(df, 3)
    out["is_quarter_end"] = is_quarter_end(df, 5)
    return out
