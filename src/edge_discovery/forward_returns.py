"""
Forward return calculation anchored to EXECUTABLE prices.

Convention:
- Signal is generated at T (using close price and information available at T).
- Earliest execution is T+1 open.
- Forward returns are measured from T+1 open to future prices.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data.schema import OPEN, CLOSE


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: List[int] = [1, 3, 5, 10, 20],
) -> pd.DataFrame:
    """
    For each horizon h, compute:
      - fwd_open_{h}: (close_{T+h} - open_{T+1}) / open_{T+1}  [long perspective]
      - fwd_open_short_{h}: -(fwd_open_{h})                     [short perspective]

    These returns are aligned to day T's index row, so they can be joined
    with signals generated at T's close.

    CRITICAL: these are shifted forward — they must NEVER be used as inputs
    to signal generation. They are outputs only.
    """
    entry_open = df[OPEN].shift(-1)  # T+1 open — not available at T
    out = pd.DataFrame(index=df.index)

    for h in horizons:
        exit_close = df[CLOSE].shift(-h)  # close at T+h
        ret = (exit_close - entry_open) / entry_open.replace(0, np.nan)
        out[f"fwd_open_{h}d"] = ret
        out[f"fwd_open_short_{h}d"] = -ret

        # Next-open-to-next-open (1-day special case)
        if h == 1:
            exit_open = df[OPEN].shift(-2)
            ret_oo = (exit_open - entry_open) / entry_open.replace(0, np.nan)
            out["fwd_open_to_open_1d"] = ret_oo

    return out


def forward_return_stats(
    signal_mask: pd.Series,
    fwd_returns: pd.DataFrame,
    horizon_col: str,
) -> dict:
    """
    Compute descriptive statistics for forward returns conditional on signal.

    signal_mask: boolean Series, True where signal fires at T
    fwd_returns: DataFrame with forward return columns
    horizon_col: which column in fwd_returns to analyse
    """
    if horizon_col not in fwd_returns.columns:
        raise KeyError(f"Column '{horizon_col}' not in fwd_returns")

    rets = fwd_returns.loc[signal_mask, horizon_col].dropna()
    n = len(rets)
    if n == 0:
        return {"n": 0}

    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    stats = {
        "n": n,
        "signal_frequency_pct": round(signal_mask.sum() / len(signal_mask) * 100, 2),
        "mean": float(rets.mean()),
        "median": float(rets.median()),
        "std": float(rets.std()),
        "hit_rate": float((rets > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
        "win_loss_ratio": (
            abs(float(wins.mean()) / float(losses.mean()))
            if len(losses) > 0 and losses.mean() != 0
            else float("inf")
        ),
        "expected_value": float(rets.mean()),
        "profit_factor": (
            float(wins.sum() / (-losses.sum()))
            if losses.sum() != 0
            else float("inf")
        ),
        "skewness": float(rets.skew()),
        "downside_dev": float(rets[rets < 0].std()) if len(rets[rets < 0]) > 1 else 0.0,
        "worst": float(rets.min()),
        "best": float(rets.max()),
        "p5": float(rets.quantile(0.05)),
        "p95": float(rets.quantile(0.95)),
    }

    # T-stat (descriptive, not a trading signal)
    from scipy import stats as scipy_stats
    if n > 2:
        t, p = scipy_stats.ttest_1samp(rets, 0.0)
        stats["t_stat"] = float(t)
        stats["p_value"] = float(p)
        # 95% CI
        se = float(rets.sem())
        stats["ci_lower_95"] = stats["mean"] - 1.96 * se
        stats["ci_upper_95"] = stats["mean"] + 1.96 * se
        stats["standard_error"] = se
    else:
        stats["t_stat"] = float("nan")
        stats["p_value"] = float("nan")

    return stats
