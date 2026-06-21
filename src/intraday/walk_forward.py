"""Expanding-window walk-forward validation over trading sessions.

Folds are sized by SESSION COUNT (the natural unit for intraday research), not
row count. The training window expands each step; an embargo of whole sessions
separates train and test to avoid leakage across the boundary.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def intraday_walk_forward(
    df_1min: pd.DataFrame,
    sessions: List,
    signal_fn: Callable,
    backtest_fn: Callable,
    min_train_sessions: int = 250,
    step_sessions: int = 63,
    embargo_sessions: int = 5,
) -> dict:
    """Run expanding-window walk-forward.

    Parameters
    ----------
    signal_fn : fn(train_sessions, test_sessions, df_1min) -> pd.Series
        Produces a signal series indexed over the test sessions' bars.
    backtest_fn : fn(df_1min_test, signals) -> dict
        Runs the engine and returns a result dict with 'trades' and 'metrics'.
    """
    sessions = sorted(sessions)
    n = len(sessions)
    dates = df_1min.index.normalize()

    folds = []
    all_trades: List[dict] = []
    oos_equity_pieces = []

    start_test = min_train_sessions + embargo_sessions
    fold_id = 0
    test_start = start_test
    while test_start < n:
        test_end = min(test_start + step_sessions, n)
        train_sessions = sessions[: test_start - embargo_sessions]
        test_sessions = sessions[test_start:test_end]
        if len(test_sessions) == 0:
            break

        test_dates = pd.to_datetime(pd.Index(test_sessions))
        mask = dates.isin(test_dates)
        df_test = df_1min[mask]

        signals = signal_fn(train_sessions, test_sessions, df_1min)
        signals = signals.reindex(df_test.index).fillna(0)
        res = backtest_fn(df_test, signals)

        trades = res.get("trades", [])
        all_trades.extend(trades)
        oos_equity_pieces.append(res["equity_curve"])
        folds.append(
            {
                "fold": fold_id,
                "train_sessions": len(train_sessions),
                "test_sessions": len(test_sessions),
                "test_start": str(test_sessions[0]),
                "test_end": str(test_sessions[-1]),
                "n_trades": len(trades),
                "net_pnl": res["metrics"]["total_net_pnl"],
                "sharpe": res["metrics"]["sharpe"],
                "hit_rate": res["metrics"]["hit_rate"],
            }
        )
        logger.info(
            "WF fold %d: train=%d test=%d trades=%d net=%.0f",
            fold_id,
            len(train_sessions),
            len(test_sessions),
            len(trades),
            res["metrics"]["total_net_pnl"],
        )
        fold_id += 1
        test_start = test_end

    from src.intraday.stats import compute_trade_stats

    agg_stats = compute_trade_stats(all_trades)
    fold_nets = [f["net_pnl"] for f in folds]
    return {
        "folds": folds,
        "n_folds": len(folds),
        "oos_trades": all_trades,
        "oos_stats": agg_stats,
        "fold_net_pnls": fold_nets,
        "pct_positive_folds": float(np.mean([x > 0 for x in fold_nets])) if folds else 0.0,
    }
