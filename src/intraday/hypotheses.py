"""Constrained intraday hypothesis library.

Each hypothesis maps a feature DataFrame to a +1/-1/0 signal Series at bar T
close (the engine executes at T+1 open). Signals are deliberately simple,
economically motivated, and parameterised only by a small fixed config to
limit the multiple-testing surface.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

HYPOTHESIS_CONFIGS: List[Dict] = [
    # A. Opening range breakout
    {"id": "ORB_15_LONG", "family": "orb", "direction": "long", "or_bars": 15,
     "description": "15-bar opening-range breakout long"},
    {"id": "ORB_30_LONG", "family": "orb", "direction": "long", "or_bars": 30,
     "description": "30-bar opening-range breakout long"},
    {"id": "ORB_60_LONG", "family": "orb", "direction": "long", "or_bars": 60,
     "description": "60-bar opening-range breakout long"},
    {"id": "ORB_15_SHORT", "family": "orb", "direction": "short", "or_bars": 15,
     "description": "15-bar opening-range breakdown short"},
    {"id": "ORB_30_SHORT", "family": "orb", "direction": "short", "or_bars": 30,
     "description": "30-bar opening-range breakdown short"},

    # B. Opening range failure (fade the break)
    {"id": "ORF_UP_SHORT", "family": "orb_failure", "direction": "short", "or_bars": 15,
     "description": "Fade upside OR break that fails back inside"},
    {"id": "ORF_DOWN_LONG", "family": "orb_failure", "direction": "long", "or_bars": 15,
     "description": "Fade downside OR break that fails back inside"},

    # C. First-hour continuation
    {"id": "FH_CONT_LONG", "family": "continuation", "direction": "long", "or_bars": 60,
     "description": "Long if up strongly over first hour"},
    {"id": "FH_CONT_SHORT", "family": "continuation", "direction": "short", "or_bars": 60,
     "description": "Short if down strongly over first hour"},

    # D. Mean reversion on z-score
    {"id": "MR_ZSCORE_LONG", "family": "mean_reversion", "direction": "long",
     "zscore_threshold": -2.0, "description": "Long when z-score < -2"},
    {"id": "MR_ZSCORE_SHORT", "family": "mean_reversion", "direction": "short",
     "zscore_threshold": 2.0, "description": "Short when z-score > 2"},

    # E. Previous-day levels
    {"id": "PDH_BREAK_LONG", "family": "prev_day", "direction": "long",
     "description": "Long on break above previous-day high"},
    {"id": "PDL_BREAK_SHORT", "family": "prev_day", "direction": "short",
     "description": "Short on break below previous-day low"},

    # F. Gap
    {"id": "GAP_CONT_LONG", "family": "gap", "direction": "long",
     "gap_threshold": 0.003, "description": "Gap-up continuation long"},
    {"id": "GAP_REV_LONG", "family": "gap", "direction": "long",
     "gap_threshold": -0.003, "description": "Gap-down reversion long"},

    # G. VIX divergence filter (long when price up but VIX not confirming fall)
    {"id": "VIX_DIV_FILTER", "family": "vix", "direction": "long",
     "description": "Long momentum filtered by VIX confirmation"},

    # H. Time of day — avoid midday chop, trade morning momentum
    {"id": "TOD_AVOID_MIDDAY", "family": "time", "direction": "long",
     "description": "Morning-only momentum, avoid 11:30-14:00"},

    # I. NIFTY/VIX confirmation long
    {"id": "NV_CONFIRM_LONG", "family": "nifty_vix", "direction": "long",
     "description": "Long when price up and VIX falling (confirming)"},
]


def _dir_val(direction: str) -> int:
    return 1 if direction == "long" else -1


def generate_signal(df: pd.DataFrame, config: dict) -> pd.Series:
    """Return a +1/-1/0 signal Series for ``config`` over ``df``.

    ``df`` must contain causal feature columns from
    :func:`src.intraday.features.compute_intraday_features`.
    """
    fam = config["family"]
    d = config.get("direction", "long")
    dv = _dir_val(d)
    idx = df.index
    sig = pd.Series(0, index=idx, dtype=int)
    bar = df["bar_num"]

    # Common eligibility: OR-based families require OR known; avoid the very
    # last bars (engine squares off anyway, but keep signals clean).
    if fam == "orb":
        n = config["or_bars"]
        hi = df[f"or{n}_high"]
        lo = df[f"or{n}_low"]
        if d == "long":
            sig[(df["close"] > hi) & hi.notna()] = 1
        else:
            sig[(df["close"] < lo) & lo.notna()] = -1

    elif fam == "orb_failure":
        n = config["or_bars"]
        hi = df[f"or{n}_high"]
        lo = df[f"or{n}_low"]
        prev_close = df["close"].groupby(idx.normalize()).shift(1)
        if config["id"] == "ORF_UP_SHORT":
            # was above OR high, now back below → fade short
            failed = (prev_close > hi) & (df["close"] < hi) & hi.notna()
            sig[failed] = -1
        else:  # ORF_DOWN_LONG
            failed = (prev_close < lo) & (df["close"] > lo) & lo.notna()
            sig[failed] = 1

    elif fam == "continuation":
        n = config["or_bars"]
        # use open_to_now_ret as first-hour strength once bar_num >= n
        thr = 0.003
        eligible = bar >= n
        if d == "long":
            sig[eligible & (df["open_to_now_ret"] > thr)] = 1
        else:
            sig[eligible & (df["open_to_now_ret"] < -thr)] = -1

    elif fam == "mean_reversion":
        z = df["zscore_20"]
        thr = config["zscore_threshold"]
        if d == "long":
            sig[z < thr] = 1
        else:
            sig[z > thr] = -1

    elif fam == "prev_day":
        if config["id"] == "PDH_BREAK_LONG":
            sig[(df["close"] > df["prev_day_high"]) & df["prev_day_high"].notna()] = 1
        else:
            sig[(df["close"] < df["prev_day_low"]) & df["prev_day_low"].notna()] = -1

    elif fam == "gap":
        gt = config["gap_threshold"]
        if gt >= 0:  # gap-up continuation
            cond = (df["gap_pct"] > gt) & (df["open_to_now_ret"] > 0)
        else:  # gap-down reversion (long)
            cond = (df["gap_pct"] < gt) & (df["open_to_now_ret"] > 0)
        sig[cond] = dv

    elif fam == "vix":
        # Long when price rising from open AND VIX not diverging (confirming).
        confirming = ~df["nifty_vix_divergence"].astype(bool)
        cond = (df["open_to_now_ret"] > 0.001) & confirming
        sig[cond] = 1

    elif fam == "time":
        t = idx.time
        from datetime import time as _t
        morning = np.array([(x >= _t(9, 30)) and (x < _t(11, 30)) for x in t])
        cond = pd.Series(morning, index=idx) & (df["open_to_now_ret"] > 0.002)
        sig[cond] = 1

    elif fam == "nifty_vix":
        confirming = ~df["nifty_vix_divergence"].astype(bool)
        cond = (df["open_to_now_ret"] > 0.001) & (df["vix_change_from_open"] < 0) & confirming
        sig[cond] = 1

    else:
        raise ValueError(f"Unknown hypothesis family: {fam}")

    return sig
