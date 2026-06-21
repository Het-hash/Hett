"""Placebo (null-model) testing for intraday strategies.

Generates frequency-, time-of-day- and session-matched RANDOM signals and runs
them through the same execution engine. If the real strategy's net PnL is not
clearly better than the placebo distribution, the apparent edge is likely
noise or an execution artifact.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.intraday.engine import run_intraday_backtest


def run_placebo_simulations(
    df_1min: pd.DataFrame,
    real_trades: List[dict],
    n_sims: int = 500,
    seed: int = 42,
    cost_bps: float = 10.0,
    squareoff_time: str = "15:20",
    no_entry_after: str = "15:00",
) -> dict:
    """Run time-matched random-signal simulations.

    The placebo draws the same NUMBER of entries per session as the real
    strategy, at random bars within the eligible window (before no_entry_after),
    with random direction. Returns rank/p-value of the real result.
    """
    real_net = float(np.sum([t["net_pnl"] for t in real_trades])) if real_trades else 0.0

    idx = df_1min.sort_index().index
    dates = idx.normalize()
    times = idx.time
    from datetime import time as _t
    ne_h, ne_m = no_entry_after.split(":")
    ne = _t(int(ne_h), int(ne_m))

    # Per-session entry counts in the real strategy.
    real_entry_dates = pd.Series(
        [pd.Timestamp(t["entry_time"]).normalize() for t in real_trades]
    )
    counts_by_date = real_entry_dates.value_counts().to_dict()

    # Pre-index eligible bar positions per session.
    df_idx = pd.DataFrame({"date": dates, "time": times}, index=np.arange(len(idx)))
    eligible_mask = np.array([t < ne for t in times])
    session_to_positions = {}
    for dt, grp in df_idx.groupby("date"):
        pos = grp.index.values[eligible_mask[grp.index.values]]
        session_to_positions[dt] = pos

    rng = np.random.default_rng(seed)
    placebo_nets = np.empty(n_sims)

    for s in range(n_sims):
        sig = np.zeros(len(idx), dtype=int)
        for dt, k in counts_by_date.items():
            positions = session_to_positions.get(dt, np.array([]))
            if len(positions) == 0 or k == 0:
                continue
            chosen = rng.choice(positions, size=min(k, len(positions)), replace=False)
            dirs = rng.choice([-1, 1], size=len(chosen))
            sig[chosen] = dirs
        sig_series = pd.Series(sig, index=idx)
        res = run_intraday_backtest(
            df_1min,
            sig_series,
            cost_bps=cost_bps,
            squareoff_time=squareoff_time,
            no_entry_after=no_entry_after,
        )
        placebo_nets[s] = res["metrics"]["total_net_pnl"]

    better = float((placebo_nets >= real_net).mean())
    pct_rank = float((placebo_nets < real_net).mean())
    return {
        "real_net_pnl": real_net,
        "placebo_mean": float(placebo_nets.mean()),
        "placebo_std": float(placebo_nets.std()),
        "pct_rank": pct_rank,
        "better_than_pct": pct_rank,
        "p_value": better,
        "n_sims": int(n_sims),
    }
