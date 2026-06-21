"""Constrained intraday hypothesis library — one-shot, edge-triggered signals.

Each hypothesis fires AT MOST ONCE per session per setup event.  The
underlying condition must transition from False→True to create a signal.
Persistent true conditions do not fire repeatedly.

All signals are aligned to bar T close → execute at T+1 open (enforced by
the engine).  OR windows are specified in BARS at whatever timeframe the
feature DataFrame was computed on.

Family descriptions
-------------------
orb           Opening-range breakout — first clean break beyond OR high/low.
orb_failure   OR breakout followed by confirmed re-entry inside the range.
continuation  Single decision after the first-N-bar period closes.
mean_reversion z-score threshold cross with reset before re-entry.
prev_day      First crossing of previous-day high or low.
gap           One-shot signal based on gap direction, fired at bar 0.
vix_filter    VIX-confirmation modifier; may be used standalone or as filter.
time_block    Time-of-day momentum restricted to a window.
nifty_vix     Explicit NIFTY-up / VIX-down (or inverse) confirmation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.intraday.signals import (
    oneshot_edge,
    oneshot_level,
    mr_with_reset_signal,
    orb_failure_signal,
    signal_funnel,
)

HYPOTHESIS_CONFIGS: List[Dict] = [
    # A. Opening range breakout
    {"id": "ORB_15_LONG",  "family": "orb", "direction": "long",  "or_min": 15,
     "description": "15-min OR breakout long — first bar closing above OR high"},
    {"id": "ORB_30_LONG",  "family": "orb", "direction": "long",  "or_min": 30,
     "description": "30-min OR breakout long"},
    {"id": "ORB_60_LONG",  "family": "orb", "direction": "long",  "or_min": 60,
     "description": "60-min OR breakout long"},
    {"id": "ORB_15_SHORT", "family": "orb", "direction": "short", "or_min": 15,
     "description": "15-min OR breakdown short"},
    {"id": "ORB_30_SHORT", "family": "orb", "direction": "short", "or_min": 30,
     "description": "30-min OR breakdown short"},

    # B. Opening range failure
    {"id": "ORF_UP_SHORT",   "family": "orb_failure", "direction": "short", "or_min": 15,
     "description": "Fade upside OR break that re-enters range"},
    {"id": "ORF_DOWN_LONG",  "family": "orb_failure", "direction": "long",  "or_min": 15,
     "description": "Fade downside OR break that re-enters range"},

    # C. First-hour continuation — single decision at first-hour bar close
    {"id": "FH_CONT_LONG",  "family": "continuation", "direction": "long",  "or_min": 60,
     "threshold": 0.003,
     "description": "Long if up >0.3% over first hour"},
    {"id": "FH_CONT_SHORT", "family": "continuation", "direction": "short", "or_min": 60,
     "threshold": 0.003,
     "description": "Short if down >0.3% over first hour"},

    # D. Mean reversion on z-score (20-bar rolling, within-session)
    {"id": "MR_ZSCORE_LONG",  "family": "mean_reversion", "direction": "long",
     "entry_zscore": -2.0, "reset_zscore": -0.5,
     "description": "Long on z-score < -2; re-entry allowed after reset to -0.5"},
    {"id": "MR_ZSCORE_SHORT", "family": "mean_reversion", "direction": "short",
     "entry_zscore": 2.0,  "reset_zscore": 0.5,
     "description": "Short on z-score > 2; re-entry allowed after reset to 0.5"},

    # E. Previous-day levels — first crossing only
    {"id": "PDH_BREAK_LONG",  "family": "prev_day", "direction": "long",
     "description": "First close above previous-day high"},
    {"id": "PDL_BREAK_SHORT", "family": "prev_day", "direction": "short",
     "description": "First close below previous-day low"},

    # F. Gap — one-shot at first available bar after session opens
    {"id": "GAP_CONT_LONG",  "family": "gap", "direction": "long",
     "gap_threshold": 0.003,
     "description": "Gap-up (>0.3%) continuation — long at first bar"},
    {"id": "GAP_REV_LONG",   "family": "gap", "direction": "long",
     "gap_threshold": -0.003,
     "description": "Gap-down (<-0.3%) reversal long at first bar"},

    # G. VIX confirmation — long when price rising AND VIX falling (confirming)
    {"id": "NV_CONFIRM_LONG", "family": "nifty_vix", "direction": "long",
     "description": "Long when open_to_now_ret>0.1% AND vix_change_from_open<0 (one-shot)"},

    # H. First-hour momentum with VIX filter
    {"id": "FH_VIX_LONG", "family": "continuation_vix", "direction": "long",
     "or_min": 60, "threshold": 0.002,
     "description": "First-hour continuation long filtered by VIX confirmation"},
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_or_col(df: pd.DataFrame, or_min: int, side: str, tf_minutes: int) -> pd.Series:
    """Fetch the correct OR column given tf_minutes.  Falls back to smallest available."""
    n_bars = max(1, or_min // tf_minutes)
    col = f"or{n_bars}_high" if side == "high" else f"or{n_bars}_low"
    if col in df.columns:
        return df[col]
    # Fallback: find available column with closest bar count
    candidates = [c for c in df.columns if c.startswith(f"or") and c.endswith(f"_{side}")]
    if not candidates:
        return pd.Series(np.nan, index=df.index)
    # pick numerically closest
    def _n(c):
        return int(c.replace("or", "").replace("_high", "").replace("_low", ""))
    best = min(candidates, key=lambda c: abs(_n(c) - n_bars))
    return df[best]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_signal(
    df: pd.DataFrame,
    config: dict,
    tf_minutes: int = 5,
    return_funnel: bool = False,
) -> "pd.Series | tuple[pd.Series, dict]":
    """Generate a one-shot +1/-1/0 signal Series for ``config``.

    ``df`` must contain causal feature columns from
    :func:`src.intraday.features.compute_intraday_features`.

    ``tf_minutes`` is used to map minute-based OR windows to bar counts.

    If ``return_funnel`` is True, return (signal, funnel_dict) where
    funnel_dict contains diagnostic counts.
    """
    fam = config["family"]
    direction = config.get("direction", "long")
    d = 1 if direction == "long" else -1
    idx = df.index
    sig = pd.Series(0, index=idx, dtype=int)
    raw_cond = pd.Series(False, index=idx)

    if fam == "orb":
        or_min = config.get("or_min", 15)
        or_h = _get_or_col(df, or_min, "high", tf_minutes)
        or_l = _get_or_col(df, or_min, "low", tf_minutes)
        if direction == "long":
            raw_cond = (df["close"] > or_h) & or_h.notna()
            sig = oneshot_edge(raw_cond).astype(int)
        else:
            raw_cond = (df["close"] < or_l) & or_l.notna()
            sig = -oneshot_edge(raw_cond).astype(int)

    elif fam == "orb_failure":
        or_min = config.get("or_min", 15)
        or_h = _get_or_col(df, or_min, "high", tf_minutes)
        or_l = _get_or_col(df, or_min, "low", tf_minutes)
        sig = orb_failure_signal(df["close"], or_h, or_l, direction)
        # raw condition for ORF: within-OR after a break
        if direction == "short":
            raw_cond = or_h.notna() & (df["close"] <= or_h)
        else:
            raw_cond = or_l.notna() & (df["close"] >= or_l)

    elif fam == "continuation":
        or_min = config.get("or_min", 60)
        thr = config.get("threshold", 0.003)
        n_bars = max(1, or_min // tf_minutes)
        # Decision fires exactly at bar_num == n_bars (the first bar AFTER the period)
        exactly_at = (df["bar_num"] == n_bars)
        if direction == "long":
            raw_cond = exactly_at & (df["open_to_now_ret"] > thr)
            sig = oneshot_level(raw_cond).astype(int)
        else:
            raw_cond = exactly_at & (df["open_to_now_ret"] < -thr)
            sig = -oneshot_level(raw_cond).astype(int)

    elif fam == "mean_reversion":
        entry_z = config.get("entry_zscore", -2.0)
        reset_z = config.get("reset_zscore", -0.5)
        zscore = df["zscore_20"]
        raw_cond = (zscore < entry_z) if direction == "long" else (zscore > entry_z)
        raw_cond = raw_cond & zscore.notna()
        sig = mr_with_reset_signal(zscore, entry_z, reset_z, direction)

    elif fam == "prev_day":
        if config["id"] == "PDH_BREAK_LONG":
            raw_cond = (df["close"] > df["prev_day_high"]) & df["prev_day_high"].notna()
            sig = oneshot_edge(raw_cond).astype(int)
        else:
            raw_cond = (df["close"] < df["prev_day_low"]) & df["prev_day_low"].notna()
            sig = -oneshot_edge(raw_cond).astype(int)

    elif fam == "gap":
        gap_thr = config.get("gap_threshold", 0.003)
        # Gap determined at bar 0.  Signal fires at bar 0 (executed at bar 1 open).
        if gap_thr >= 0:
            raw_cond = (df["gap_pct"] > gap_thr) & (df["bar_num"] == 0)
            sig = oneshot_level(raw_cond).astype(int)
        else:
            raw_cond = (df["gap_pct"] < gap_thr) & (df["bar_num"] == 0)
            sig = oneshot_level(raw_cond).astype(int)  # long on gap-down reversal

    elif fam == "nifty_vix":
        ret_thr = 0.001  # >0.1% from open
        vix_cond = df["vix_change_from_open"] < 0
        price_cond = df["open_to_now_ret"] > ret_thr
        raw_cond = price_cond & vix_cond
        sig = oneshot_edge(raw_cond).astype(int)

    elif fam == "continuation_vix":
        or_min = config.get("or_min", 60)
        thr = config.get("threshold", 0.002)
        n_bars = max(1, or_min // tf_minutes)
        exactly_at = (df["bar_num"] == n_bars)
        vix_ok = df["vix_change_from_open"] < 0  # VIX falling = confirming
        raw_cond = exactly_at & (df["open_to_now_ret"] > thr) & vix_ok
        sig = oneshot_level(raw_cond).astype(int)

    else:
        raise ValueError(f"Unknown hypothesis family: {fam!r}")

    if return_funnel:
        funnel = signal_funnel(raw_cond.astype(bool), sig)
        return sig, funnel
    return sig
