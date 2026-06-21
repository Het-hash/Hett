"""Signal condition builders — translate experiment configs into boolean masks."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.schema import CLOSE
from src.features.returns import rolling_return
from src.features.momentum import rsi, consecutive_down
from src.features.mean_reversion import rolling_zscore
from src.features.breakouts import high_breakout, low_breakdown
from src.features.trend import ma_cross_signal, sma
from src.features.gaps import gap_pct
from src.features.calendar import is_month_end
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Regime column names
TREND_REGIME_COL = "trend_regime"
VOL_REGIME_COL = "vol_regime"


def momentum_signal(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.0,
    regimes: Optional[pd.DataFrame] = None,
    regime_filter: Optional[str] = None,
) -> pd.Series:
    """Positive momentum signal: rolling return > threshold."""
    sig = rolling_return(df, lookback) > threshold
    sig = sig.rename(f"sig_mom_{lookback}")
    if regimes is not None and regime_filter:
        sig = _apply_regime_filter(sig, regimes, regime_filter)
    return sig


def mean_reversion_zscore_signal(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = -2.0,
    direction: str = "long",
    regimes: Optional[pd.DataFrame] = None,
    regime_filter: Optional[str] = None,
) -> pd.Series:
    """Mean reversion: zscore below threshold (oversold) for long."""
    z = rolling_zscore(df, window)
    if direction == "long":
        sig = z < threshold
    else:
        sig = z > (-threshold)
    sig = sig.rename(f"sig_mr_zscore_{window}_{threshold}")
    if regimes is not None and regime_filter:
        sig = _apply_regime_filter(sig, regimes, regime_filter)
    return sig


def mean_reversion_rsi_signal(
    df: pd.DataFrame,
    window: int = 14,
    threshold: float = 30.0,
    direction: str = "long",
    regimes: Optional[pd.DataFrame] = None,
    regime_filter: Optional[str] = None,
) -> pd.Series:
    """RSI-based mean reversion signal."""
    r = rsi(df, window)
    if direction == "long":
        sig = r < threshold
    else:
        sig = r > (100 - threshold)
    sig = sig.rename(f"sig_rsi_{window}_{threshold}")
    if regimes is not None and regime_filter:
        sig = _apply_regime_filter(sig, regimes, regime_filter)
    return sig


def consecutive_down_signal(
    df: pd.DataFrame,
    streak: int = 3,
    regimes: Optional[pd.DataFrame] = None,
    regime_filter: Optional[str] = None,
) -> pd.Series:
    """Mean reversion after consecutive down sessions."""
    sig = (consecutive_down(df) >= streak).rename(f"sig_consec_down_{streak}")
    if regimes is not None and regime_filter:
        sig = _apply_regime_filter(sig, regimes, regime_filter)
    return sig


def breakout_signal(
    df: pd.DataFrame,
    window: int = 20,
    direction: str = "long",
    vol_condition: Optional[str] = None,
    regimes: Optional[pd.DataFrame] = None,
    vol_col: Optional[str] = None,
) -> pd.Series:
    """Donchian-style breakout signal."""
    if direction == "long":
        sig = high_breakout(df, window)
    else:
        sig = low_breakdown(df, window)
    sig = sig.rename(f"sig_brkout_{window}_{direction}")

    if vol_condition == "low" and regimes is not None and VOL_REGIME_COL in regimes:
        sig = sig & (regimes[VOL_REGIME_COL] == "low")

    return sig


def ma_cross_trend_signal(
    df: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    direction: str = "long",
) -> pd.Series:
    """Trend-following signal based on MA crossover."""
    cross = ma_cross_signal(df, fast, slow)
    if direction == "long":
        return (cross > 0).rename(f"sig_ma_cross_{fast}_{slow}_long")
    else:
        return (cross < 0).rename(f"sig_ma_cross_{fast}_{slow}_short")


def pullback_in_trend_signal(
    df: pd.DataFrame,
    trend_window: int = 200,
    pullback_window: int = 5,
    pullback_threshold: float = -0.02,
) -> pd.Series:
    """Pullback within an uptrend: price > long SMA but short-term ret < threshold."""
    ma = sma(df[CLOSE], trend_window)
    in_uptrend = df[CLOSE] > ma
    short_ret = rolling_return(df, pullback_window)
    sig = in_uptrend & (short_ret < pullback_threshold)
    return sig.rename(f"sig_pullback_{trend_window}_{pullback_window}")


def gap_signal(
    df: pd.DataFrame,
    threshold: float = 0.01,
    direction: str = "long",
) -> pd.Series:
    """Gap continuation or reversal signal."""
    g = gap_pct(df)
    if direction == "long" and threshold > 0:
        sig = g > threshold  # gap-up continuation
    elif direction == "long" and threshold < 0:
        sig = g < threshold  # gap-down reversal (buy the dip)
    else:
        sig = g.abs() > abs(threshold)
    return sig.rename(f"sig_gap_{threshold}_{direction}")


def month_end_signal(df: pd.DataFrame, n_sessions: int = 3) -> pd.Series:
    """Calendar effect: last N sessions of the month."""
    return is_month_end(df, n_sessions).rename("sig_month_end")


def day_of_week_signal(df: pd.DataFrame, day: int) -> pd.Series:
    """Day-of-week signal (0=Mon, 4=Fri)."""
    dow = pd.Series(df.index.dayofweek, index=df.index)
    return (dow == day).rename(f"sig_dow_{day}")


def _apply_regime_filter(
    sig: pd.Series,
    regimes: pd.DataFrame,
    regime_filter: str,
) -> pd.Series:
    """Apply a regime filter to a signal mask."""
    if regime_filter == "trend_bull" and TREND_REGIME_COL in regimes:
        return sig & (regimes[TREND_REGIME_COL] == "bull")
    if regime_filter == "not_bear" and TREND_REGIME_COL in regimes:
        return sig & (regimes[TREND_REGIME_COL] != "bear")
    if regime_filter == "vol_low" and VOL_REGIME_COL in regimes:
        return sig & (regimes[VOL_REGIME_COL] == "low")
    logger.warning(f"Unknown regime filter: {regime_filter}")
    return sig
