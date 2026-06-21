"""Robustness and stress tests."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.backtesting.engine import run_backtest
from src.backtesting.metrics import compute_metrics
from src.edge_discovery.statistics import block_bootstrap_ci
from src.utils.logging import get_logger

logger = get_logger(__name__)


def subperiod_analysis(
    df: pd.DataFrame,
    signal: pd.Series,
    direction: str = "long",
    cost_profile=None,
    initial_capital: float = 1_000_000.0,
    n_periods: int = 4,
) -> pd.DataFrame:
    """Evaluate performance in equal-length chronological subperiods."""
    from src.backtesting.costs import ETF_CONSERVATIVE
    cost_profile = cost_profile or ETF_CONSERVATIVE
    n = len(df)
    period_size = n // n_periods
    rows = []
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        sub_df = df.iloc[start:end]
        sub_sig = signal.loc[sub_df.index]
        result = run_backtest(sub_df, sub_sig, direction=direction,
                              cost_profile=cost_profile, initial_capital=initial_capital)
        m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
        rows.append({
            "period": i + 1,
            "start": str(sub_df.index[0].date()),
            "end": str(sub_df.index[-1].date()),
            **m,
        })
    return pd.DataFrame(rows).set_index("period")


def crisis_exclusion_test(
    df: pd.DataFrame,
    signal: pd.Series,
    crisis_periods: Optional[List[tuple]] = None,
    direction: str = "long",
    cost_profile=None,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Exclude known crisis/rally periods and re-test.
    Default: GFC (2008-01 to 2009-03), COVID (2020-01 to 2020-06).
    """
    from src.backtesting.costs import ETF_CONSERVATIVE
    cost_profile = cost_profile or ETF_CONSERVATIVE

    if crisis_periods is None:
        crisis_periods = [
            ("2008-01-01", "2009-06-30"),
            ("2020-01-01", "2020-09-30"),
        ]

    exclude_mask = pd.Series(False, index=df.index)
    for start, end in crisis_periods:
        exclude_mask |= (df.index >= start) & (df.index <= end)

    non_crisis_df = df[~exclude_mask]
    non_crisis_sig = signal[~exclude_mask]

    if len(non_crisis_df) < 50:
        return {"error": "Too few non-crisis observations"}

    result = run_backtest(non_crisis_df, non_crisis_sig, direction=direction,
                          cost_profile=cost_profile, initial_capital=initial_capital)
    return compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)


def best_trade_removal(
    df: pd.DataFrame,
    signal: pd.Series,
    direction: str = "long",
    cost_profile=None,
    initial_capital: float = 1_000_000.0,
    n_remove: int = 5,
) -> dict:
    """Remove the N best trades and re-evaluate. High sensitivity = fragile."""
    from src.backtesting.costs import ETF_CONSERVATIVE
    cost_profile = cost_profile or ETF_CONSERVATIVE

    result = run_backtest(df, signal, direction=direction,
                          cost_profile=cost_profile, initial_capital=initial_capital)
    trades = result.trades
    if trades.empty or "net_return" not in trades.columns:
        return {"error": "No trades"}

    top_dates = trades.nlargest(n_remove, "net_return")["entry_date"].tolist()
    reduced_signal = signal.copy()
    for d in top_dates:
        if d in reduced_signal.index.astype(str):
            mask = reduced_signal.index.astype(str) == d
            reduced_signal[mask] = False

    result2 = run_backtest(df, reduced_signal, direction=direction,
                           cost_profile=cost_profile, initial_capital=initial_capital)
    m = compute_metrics(result2.returns, result2.equity_curve, result2.positions, result2.trades)
    m["n_trades_removed"] = n_remove
    return m


def bootstrap_metrics(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> dict:
    """Block bootstrap confidence intervals for Sharpe ratio and mean return."""
    sharpe_lo, sharpe_hi = block_bootstrap_ci(returns, "sharpe", block_size, n_bootstrap, seed=seed)
    mean_lo, mean_hi = block_bootstrap_ci(returns, "mean", block_size, n_bootstrap, seed=seed)
    return {
        "sharpe_ci_lo_95": sharpe_lo,
        "sharpe_ci_hi_95": sharpe_hi,
        "mean_ci_lo_95": mean_lo,
        "mean_ci_hi_95": mean_hi,
    }
