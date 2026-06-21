"""Edge statistics for intraday trade lists."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_trade_stats(trades: List[dict], cost_bps: float = 10.0) -> dict:
    """Return a full statistics dict for a list of trade dicts."""
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "hit_rate": 0.0,
            "avg_net_pnl": 0.0,
            "median_net_pnl": 0.0,
            "total_net_pnl": 0.0,
            "std_net_pnl": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "t_stat": 0.0,
            "avg_holding_bars": 0.0,
            "total_cost": 0.0,
            "cost_bps": cost_bps,
        }
    nets = np.array([t["net_pnl"] for t in trades], dtype=float)
    costs = np.array([t["cost"] for t in trades], dtype=float)
    holding = np.array([t["holding_bars"] for t in trades], dtype=float)
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    std = float(nets.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(nets.mean() / (std / np.sqrt(n))) if std > 0 else 0.0
    pf = float(wins.sum() / (-losses.sum())) if losses.sum() < 0 else float("inf")
    return {
        "n_trades": int(n),
        "hit_rate": float((nets > 0).mean()),
        "avg_net_pnl": float(nets.mean()),
        "median_net_pnl": float(np.median(nets)),
        "total_net_pnl": float(nets.sum()),
        "std_net_pnl": std,
        "profit_factor": pf,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(nets.mean()),
        "t_stat": t_stat,
        "avg_holding_bars": float(holding.mean()),
        "total_cost": float(costs.sum()),
        "cost_bps": cost_bps,
    }


def block_bootstrap_ci(
    trades: List[dict],
    n_boot: int = 500,
    block_size: int = 20,
    seed: int = 42,
) -> Tuple[float, float]:
    """95% CI on the mean net PnL via a moving-block bootstrap.

    Preserves short-range autocorrelation by resampling contiguous blocks of
    trades (ordered by exit time).
    """
    nets = np.array([t["net_pnl"] for t in trades], dtype=float)
    n = len(nets)
    if n == 0:
        return (0.0, 0.0)
    if n <= block_size:
        block_size = max(1, n // 2)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([nets[s : s + block_size] for s in starts])[:n]
        means[b] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))
