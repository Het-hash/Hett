"""Causal gap features."""
from __future__ import annotations

import pandas as pd

from src.data.schema import OPEN, CLOSE
from src.features.volatility import atr
from src.features.trend import sma


def gap_pct(df: pd.DataFrame) -> pd.Series:
    """Overnight gap: (open_t - close_{t-1}) / close_{t-1}."""
    return ((df[OPEN] - df[CLOSE].shift(1)) / df[CLOSE].shift(1)).rename("gap_pct")


def gap_atr_ratio(df: pd.DataFrame, atr_window: int = 14) -> pd.Series:
    """Gap expressed in ATR units."""
    g = (df[OPEN] - df[CLOSE].shift(1)).abs()
    a = atr(df, atr_window)
    return (g / a.replace(0, float("nan"))).rename(f"gap_atr_{atr_window}")


def gap_direction_vs_trend(df: pd.DataFrame, trend_window: int = 50) -> pd.Series:
    """
    +1 if gap direction matches trend direction, -1 if counter-trend, 0 if no trend.
    Trend direction: sign of (close - SMA(trend_window)).
    """
    trend = (df[CLOSE].shift(1) - sma(df, trend_window).shift(1)).apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    g = gap_pct(df).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (trend * g).rename(f"gap_vs_trend_{trend_window}")


def gap_fill_ratio(df: pd.DataFrame) -> pd.Series:
    """
    How much of the gap was filled intraday.
    For an up gap: (open - low) / gap; 1.0 = fully filled.
    For a down gap: (high - open) / |gap|.
    Returns 0 if no gap.
    """
    from src.data.schema import HIGH, LOW
    g = df[OPEN] - df[CLOSE].shift(1)
    fill = pd.Series(0.0, index=df.index)
    up_gap = g > 0
    dn_gap = g < 0
    fill[up_gap] = (df[OPEN][up_gap] - df[LOW][up_gap]) / g[up_gap]
    fill[dn_gap] = (df[HIGH][dn_gap] - df[OPEN][dn_gap]) / (-g[dn_gap])
    return fill.clip(0, 1).rename("gap_fill_ratio")


def build_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["gap_pct"] = gap_pct(df)
    out["gap_atr_14"] = gap_atr_ratio(df, 14)
    out["gap_vs_trend_50"] = gap_direction_vs_trend(df, 50)
    out["gap_fill_ratio"] = gap_fill_ratio(df)
    return out


# Allow sma() to be called on DataFrame or Series
def sma(df_or_series, window: int) -> pd.Series:
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series[CLOSE].rolling(window).mean()
    return df_or_series.rolling(window).mean()
