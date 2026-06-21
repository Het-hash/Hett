"""Parameter stability and robustness analysis."""
from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.edge_discovery.forward_returns import forward_return_stats
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parameter_sweep(
    df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    signal_fn: Callable,
    param_grid: Dict[str, List],
    horizon: int = 5,
    min_obs: int = 30,
) -> pd.DataFrame:
    """
    Sweep over a parameter grid and compute forward return stats for each combination.
    Returns a DataFrame with one row per parameter combination.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    fwd_col = f"fwd_open_{horizon}d"
    if fwd_col not in fwd_returns.columns:
        raise KeyError(f"{fwd_col} not in fwd_returns")

    rows = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            sig = signal_fn(df, **params)
            sig = sig.fillna(False).astype(bool)
            if sig.sum() < min_obs:
                continue
            stats = forward_return_stats(sig, fwd_returns, fwd_col)
            row = {**params, **{k: v for k, v in stats.items() if isinstance(v, (int, float))}}
            rows.append(row)
        except Exception as exc:
            logger.debug(f"Param combo {params} failed: {exc}")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result


def subperiod_stability(
    df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 5,
    n_periods: int = 4,
) -> pd.DataFrame:
    """
    Split in-sample data into `n_periods` equal chronological subperiods
    and compute forward return stats in each.
    """
    fwd_col = f"fwd_open_{horizon}d"
    n = len(df)
    period_size = n // n_periods
    rows = []

    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        idx = df.index[start:end]
        sub_sig = signal.loc[idx]
        sub_fwd = fwd_returns.loc[idx]
        if sub_sig.sum() < 5:
            continue
        stats = forward_return_stats(sub_sig, sub_fwd, fwd_col)
        rows.append({
            "period": i + 1,
            "start": str(idx[0].date()),
            "end": str(idx[-1].date()),
            **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
        })

    return pd.DataFrame(rows).set_index("period") if rows else pd.DataFrame()


def regime_stability(
    df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    signal: pd.Series,
    regimes: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Compute signal performance broken down by each regime dimension."""
    fwd_col = f"fwd_open_{horizon}d"
    rows = []

    for col in regimes.columns:
        for regime_val in regimes[col].dropna().unique():
            mask = signal & (regimes[col] == regime_val)
            if mask.sum() < 10:
                continue
            stats = forward_return_stats(mask, fwd_returns, fwd_col)
            rows.append({
                "regime_dim": col,
                "regime_val": regime_val,
                "n": stats.get("n", 0),
                "mean": stats.get("mean", float("nan")),
                "hit_rate": stats.get("hit_rate", float("nan")),
                "ev": stats.get("expected_value", float("nan")),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
