"""Daily context features for Phase 1C contextual edge refinement.

All features use only information available at the START of a trading session
(i.e., data from previous sessions). No within-session future information is used.

Design notes
------------
Context features are computed at the DAILY level (one row per session) and
then broadcast onto the intraday TF DataFrame so that filters can be applied
at the exact bar where the parent signal fires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


# ---------------------------------------------------------------------------
# Daily context computation
# ---------------------------------------------------------------------------

def compute_daily_context(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Build a per-session context DataFrame from 1-min OHLC + VIX.

    Every column is known at the OPEN of the session (derived from the
    previous session or trailing historical data).  Index is the session date.
    """
    dates = df_1min.index.normalize()

    daily = pd.DataFrame({
        "open":      df_1min["open"].groupby(dates).first(),
        "high":      df_1min["high"].groupby(dates).max(),
        "low":       df_1min["low"].groupby(dates).min(),
        "close":     df_1min["close"].groupby(dates).last(),
        "vix_open":  df_1min["vix_open"].groupby(dates).first(),
        "vix_close": df_1min["vix_close"].groupby(dates).last(),
    })
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    daily = daily.sort_index()

    prev = daily.shift(1)
    prev2 = daily.shift(2)

    ctx = pd.DataFrame(index=daily.index)

    # ── Previous-session OHLC (known at today's open) ──────────────────────
    ctx["prev_close"]   = prev["close"]
    ctx["prev_open"]    = prev["open"]
    ctx["prev_high"]    = prev["high"]
    ctx["prev_low"]     = prev["low"]
    ctx["prev_range"]   = prev["high"] - prev["low"]
    ctx["prev_vix_close"] = prev["vix_close"]

    # Previous session return (open→close)
    ctx["prev_return"] = (prev["close"] - prev["open"]) / (prev["open"] + EPS)

    # Where did the previous close sit within the previous session range?
    ctx["prev_close_location"] = (prev["close"] - prev["low"]) / (ctx["prev_range"] + EPS)

    # ── Previous-range percentile (252-session rolling) ────────────────────
    prev_range_rel = ctx["prev_range"] / (prev["close"] + EPS)
    ctx["prev_range_pct_rank"] = prev_range_rel.rolling(252, min_periods=60).rank(pct=True)

    # ── VIX context ────────────────────────────────────────────────────────
    # Trailing VIX percentile at previous session close
    ctx["vix_pct_rank"] = prev["vix_close"].rolling(252, min_periods=60).rank(pct=True)

    # Overnight VIX change: today's vix_open vs yesterday's vix_close
    ctx["vix_overnight_chg"] = (daily["vix_open"] - prev["vix_close"]) / (prev["vix_close"] + EPS)

    # ── Daily trend (based on trailing SMA, fully causal) ─────────────────
    sma20  = prev["close"].rolling(20, min_periods=10).mean()
    sma50  = prev["close"].rolling(50, min_periods=25).mean()
    ctx["daily_sma20_slope"] = sma20 - sma20.shift(5)   # 5-day slope using past-closes
    ctx["daily_bull_20"]  = ctx["daily_sma20_slope"] > 0
    ctx["daily_above_20"] = prev["close"] > sma20
    ctx["daily_above_50"] = prev["close"] > sma50

    # Multi-timeframe trend agreement (both 20 & 50 agree)
    ctx["trend_agree_bull"] = ctx["daily_bull_20"] & ctx["daily_above_50"]
    ctx["trend_agree_bear"] = ~ctx["daily_bull_20"] & ~ctx["daily_above_50"]

    # ── Daily ATR (using previous sessions) ───────────────────────────────
    prev_prev_close = prev2["close"]
    tr = pd.concat([
        (prev["high"] - prev["low"]).abs(),
        (prev["high"] - prev_prev_close).abs(),
        (prev["low"]  - prev_prev_close).abs(),
    ], axis=1).max(axis=1)
    ctx["daily_atr_14"] = tr.rolling(14, min_periods=7).mean()

    # ── Gap relative to ATR (today's gap using yesterday's ATR) ───────────
    today_gap = daily["open"] - prev["close"]
    ctx["gap_pts"] = today_gap
    ctx["gap_rel_atr"] = today_gap.abs() / (ctx["daily_atr_14"] + EPS)

    # ── Today's open position relative to previous session ─────────────────
    ctx["open_above_prev_high"] = daily["open"] > prev["high"]
    ctx["open_below_prev_low"]  = daily["open"] < prev["low"]
    ctx["open_inside_prev"]     = ~ctx["open_above_prev_high"] & ~ctx["open_below_prev_low"]

    # ── Calendar ──────────────────────────────────────────────────────────
    ctx["day_of_week"] = daily.index.dayofweek   # 0=Mon … 4=Fri

    return ctx


def merge_context_to_tf(df_tf: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Broadcast daily context features onto a TF-aggregated bar DataFrame."""
    dates = df_tf.index.normalize()
    merged = df_tf.copy()
    for col in ctx.columns:
        merged[col] = ctx[col].reindex(dates).values
    return merged


# ---------------------------------------------------------------------------
# Filter catalogue
# ---------------------------------------------------------------------------

def _safe_bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


SINGLE_FILTERS: dict = {
    # --- Previous-day context ---
    "prev_bullish":         lambda df: _safe_bool(df["prev_return"] > 0),
    "prev_bearish":         lambda df: _safe_bool(df["prev_return"] < 0),
    "prev_strong_bull":     lambda df: _safe_bool(df["prev_return"] > 0.005),
    "prev_strong_bear":     lambda df: _safe_bool(df["prev_return"] < -0.005),
    "high_prev_range":      lambda df: _safe_bool(df["prev_range_pct_rank"] > 0.60),
    "low_prev_range":       lambda df: _safe_bool(df["prev_range_pct_rank"] < 0.40),
    "close_near_prev_high": lambda df: _safe_bool(df["prev_close_location"] > 0.65),
    "close_near_prev_low":  lambda df: _safe_bool(df["prev_close_location"] < 0.35),
    "close_mid_prev":       lambda df: _safe_bool(
                                (df["prev_close_location"] >= 0.35) &
                                (df["prev_close_location"] <= 0.65)),

    # --- VIX context ---
    "vix_high":             lambda df: _safe_bool(df["vix_pct_rank"] > 0.60),
    "vix_low":              lambda df: _safe_bool(df["vix_pct_rank"] < 0.40),
    "vix_normal":           lambda df: _safe_bool(
                                (df["vix_pct_rank"] >= 0.40) &
                                (df["vix_pct_rank"] <= 0.60)),
    "vix_overnight_up":     lambda df: _safe_bool(df["vix_overnight_chg"] > 0),
    "vix_overnight_down":   lambda df: _safe_bool(df["vix_overnight_chg"] < 0),
    "vix_large_overnight":  lambda df: _safe_bool(df["vix_overnight_chg"].abs() > 0.03),

    # --- Opening context ---
    "gap_up":               lambda df: _safe_bool(df["gap_pts"] > 0),
    "gap_down":             lambda df: _safe_bool(df["gap_pts"] < 0),
    "small_gap":            lambda df: _safe_bool(df["gap_rel_atr"] < 0.25),
    "large_gap":            lambda df: _safe_bool(df["gap_rel_atr"] > 0.50),
    "open_above_prev_high": lambda df: _safe_bool(df["open_above_prev_high"]),
    "open_below_prev_low":  lambda df: _safe_bool(df["open_below_prev_low"]),
    "open_inside_prev":     lambda df: _safe_bool(df["open_inside_prev"]),

    # --- Trend context ---
    "daily_bull_20":        lambda df: _safe_bool(df["daily_bull_20"]),
    "daily_bear_20":        lambda df: _safe_bool(~_safe_bool(df["daily_bull_20"])),
    "above_sma20":          lambda df: _safe_bool(df["daily_above_20"]),
    "below_sma20":          lambda df: _safe_bool(~_safe_bool(df["daily_above_20"])),
    "above_sma50":          lambda df: _safe_bool(df["daily_above_50"]),
    "trend_agree_bull":     lambda df: _safe_bool(df["trend_agree_bull"]),
    "trend_agree_bear":     lambda df: _safe_bool(df["trend_agree_bear"]),

    # --- Calendar ---
    "monday":               lambda df: pd.Series(df.index.dayofweek == 0, index=df.index),
    "tuesday":              lambda df: pd.Series(df.index.dayofweek == 1, index=df.index),
    "wednesday":            lambda df: pd.Series(df.index.dayofweek == 2, index=df.index),
    "thursday":             lambda df: pd.Series(df.index.dayofweek == 3, index=df.index),
    "friday":               lambda df: pd.Series(df.index.dayofweek == 4, index=df.index),
    "not_friday":           lambda df: pd.Series(df.index.dayofweek != 4, index=df.index),
    "not_monday":           lambda df: pd.Series(df.index.dayofweek != 0, index=df.index),
    "mid_week":             lambda df: pd.Series(df.index.dayofweek.isin([1, 2, 3]), index=df.index),
}

# Pre-specified two-filter combinations (economically motivated; not exhaustive)
TWO_FILTER_COMBOS: list = [
    ("vix_high",         "daily_bull_20"),
    ("vix_high",         "daily_bear_20"),
    ("vix_low",          "daily_bull_20"),
    ("vix_high",         "prev_bearish"),
    ("vix_high",         "open_inside_prev"),
    ("vix_low",          "small_gap"),
    ("prev_bullish",     "daily_bull_20"),
    ("prev_bearish",     "daily_bear_20"),
    ("prev_bullish",     "above_sma20"),
    ("high_prev_range",  "vix_high"),
    ("low_prev_range",   "vix_low"),
    ("vix_overnight_up", "prev_bearish"),
    ("small_gap",        "open_inside_prev"),
    ("large_gap",        "vix_high"),
    ("close_near_prev_high", "daily_bull_20"),
    ("close_near_prev_low",  "daily_bear_20"),
    ("trend_agree_bull", "prev_bullish"),
    ("trend_agree_bear", "prev_bearish"),
    ("not_friday",       "above_sma20"),
    ("mid_week",         "vix_high"),
]


def apply_filter(signal: pd.Series, mask: pd.Series) -> pd.Series:
    """Gate a signal with a boolean mask. Non-True mask entries → 0."""
    return signal.where(mask.reindex(signal.index).fillna(False), 0).astype(int)


def filter_trades(trades: list, mask: pd.Series) -> list:
    """Keep only trades whose entry_time falls in the True region of mask."""
    true_times = set(mask.index[mask.astype(bool)])
    return [t for t in trades if t["entry_time"] in true_times]
