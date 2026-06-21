"""Parameter and cost sensitivity analysis."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.backtesting.costs import flat_cost_profile
from src.backtesting.engine import run_backtest
from src.backtesting.metrics import compute_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

SENSITIVITY_BPS = [0, 5, 10, 15, 25, 50]


def cost_sensitivity(
    df: pd.DataFrame,
    signal: pd.Series,
    direction: str = "long",
    initial_capital: float = 1_000_000.0,
    bps_levels: List[int] = SENSITIVITY_BPS,
) -> pd.DataFrame:
    """
    Run backtest at multiple cost levels and return metrics for each.
    """
    rows = []
    for bps in bps_levels:
        profile = flat_cost_profile(bps)
        result = run_backtest(df, signal, direction=direction,
                              cost_profile=profile, initial_capital=initial_capital)
        m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
        row = {"cost_bps_roundtrip": bps, **m}
        rows.append(row)
        logger.debug(f"Cost sensitivity: {bps}bps → CAGR={m.get('cagr', 'N/A'):.2%}")

    return pd.DataFrame(rows).set_index("cost_bps_roundtrip")


def execution_delay_sensitivity(
    df: pd.DataFrame,
    signal: pd.Series,
    direction: str = "long",
    cost_profile=None,
    initial_capital: float = 1_000_000.0,
    delays: List[int] = [1, 2, 3],
) -> pd.DataFrame:
    """
    Test performance with additional execution delay (shift signal by extra bars).
    delay=1 is the standard (T+1 open). delay=2 means T+2 open.
    """
    from src.backtesting.costs import ETF_CONSERVATIVE
    cost_profile = cost_profile or ETF_CONSERVATIVE
    rows = []
    for delay in delays:
        delayed_signal = signal.shift(delay - 1).fillna(False).astype(bool)
        result = run_backtest(df, delayed_signal, direction=direction,
                              cost_profile=cost_profile, initial_capital=initial_capital)
        m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
        rows.append({"extra_delay_bars": delay, **m})

    return pd.DataFrame(rows).set_index("extra_delay_bars")


def missed_trades_sensitivity(
    df: pd.DataFrame,
    signal: pd.Series,
    direction: str = "long",
    cost_profile=None,
    initial_capital: float = 1_000_000.0,
    drop_fracs: List[float] = [0.1, 0.2, 0.3],
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly drop a fraction of trades and re-run."""
    from src.backtesting.costs import ETF_CONSERVATIVE
    from src.utils.random_state import get_rng
    cost_profile = cost_profile or ETF_CONSERVATIVE
    rng = get_rng(seed)
    rows = []
    for frac in drop_fracs:
        mask = rng.random(len(signal)) > frac
        reduced_signal = signal & pd.Series(mask, index=signal.index)
        result = run_backtest(df, reduced_signal, direction=direction,
                              cost_profile=cost_profile, initial_capital=initial_capital)
        m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
        rows.append({"drop_fraction": frac, **m})

    return pd.DataFrame(rows).set_index("drop_fraction")
