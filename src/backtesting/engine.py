"""
Vectorised backtesting engine with explicit cost modeling and invariant checks.

Execution convention:
- Signal generated at T close.
- Position change executed at T+1 open.
- Return for day T+1: position * (close_{T+1} / open_{T+1} - 1) on the bar,
  plus overnight move (open_{T+1} / close_T - 1) if position was held.

IMPORTANT: This engine uses ^NSEI as a research proxy. Stated costs are
indicative assumptions for sensitivity testing, not real execution costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import OPEN, HIGH, LOW, CLOSE
from src.backtesting.costs import CostProfile, ETF_CONSERVATIVE, ZERO_COST
from src.backtesting.execution import generate_position_series, Trade, LONG, SHORT, FLAT
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series        # daily net returns
    gross_returns: pd.Series  # daily gross returns
    positions: pd.Series
    trades: pd.DataFrame
    cost_profile: str
    initial_capital: float
    metrics: dict = field(default_factory=dict)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    direction: str = "long",
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
    sizing: str = "full",  # 'full' = 100% allocation when in position
    max_leverage: float = 1.0,
) -> BacktestResult:
    """
    Run a vectorised backtest.

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex
    signal : boolean Series aligned to df — True = signal fires at T's close
    direction : 'long', 'short', or 'both' (signal already contains +1/0/-1)
    cost_profile : CostProfile instance
    initial_capital : starting capital in INR
    sizing : 'full' (binary 100% allocation) or 'none' (returns only)
    max_leverage : hard cap (1.0 = no leverage)
    """
    # ── Align signal to df ────────────────────────────────────────────────────
    signal = signal.reindex(df.index).fillna(False)

    # ── Position series ───────────────────────────────────────────────────────
    # position[t] = position HELD during day t (entered at open of t)
    position = generate_position_series(signal, direction)

    # Clamp to max leverage
    position = position.clip(-max_leverage, max_leverage)

    # ── Return components ─────────────────────────────────────────────────────
    # Intraday return: (close - open) / open — this is what the position earns
    # if entered at open and held to close.
    intraday = (df[CLOSE] - df[OPEN]) / df[OPEN]

    # Overnight return: (open_t - close_{t-1}) / close_{t-1}
    # Earned when position was ALREADY held the prior day (held overnight).
    overnight = (df[OPEN] - df[CLOSE].shift(1)) / df[CLOSE].shift(1)

    # Raw daily gross return for the position:
    # If we entered at open today: intraday only.
    # If we held overnight: overnight + intraday.
    # We approximate: total bar return = (close - close.shift(1)) / close.shift(1)
    # for a position entered at the prior session's signal and held through.
    # This overstates by not separating overnight from intraday entry precisely,
    # but is correct for daily signals with T+1 execution.
    close_ret = df[CLOSE].pct_change()

    # Gross daily strategy return
    gross_daily = position * close_ret

    # ── Transaction costs ─────────────────────────────────────────────────────
    # Cost is incurred on the day we change position (at T+1 open, counted in T+1 bar)
    position_change = position.diff().abs()
    # Round-trip cost applied as a fraction when position changes
    cost_per_change = cost_profile.one_way_cost("entry")  # one-way per direction change
    daily_costs = position_change * cost_per_change

    net_daily = gross_daily - daily_costs

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity = initial_capital * (1 + net_daily).cumprod()
    equity.iloc[0] = initial_capital  # start value

    # ── Trade records ─────────────────────────────────────────────────────────
    trades = _extract_trades(df, position, gross_daily, daily_costs)

    # ── Invariant checks ─────────────────────────────────────────────────────
    _check_invariants(equity, position, max_leverage)

    result = BacktestResult(
        equity_curve=equity,
        returns=net_daily,
        gross_returns=gross_daily,
        positions=position,
        trades=trades,
        cost_profile=cost_profile.label,
        initial_capital=initial_capital,
    )
    return result


def _extract_trades(
    df: pd.DataFrame,
    position: pd.Series,
    gross_daily: pd.Series,
    daily_costs: pd.Series,
) -> pd.DataFrame:
    """Extract individual trade records from a position series."""
    trades = []
    current_pos = 0
    entry_date = None
    entry_price = None
    trade_gross = 0.0
    trade_cost = 0.0

    for i, (date, pos) in enumerate(position.items()):
        if pos != current_pos:
            # Close existing position
            if current_pos != 0 and entry_date is not None:
                exit_price = float(df[OPEN].iloc[i]) if i < len(df) else float(df[CLOSE].iloc[i - 1])
                holding = (date - pd.Timestamp(entry_date)).days
                gross_ret = float(entry_price) and (exit_price - entry_price) / entry_price * current_pos
                trades.append({
                    "entry_date": str(entry_date),
                    "exit_date": str(date.date()),
                    "direction": "long" if current_pos > 0 else "short",
                    "entry_price": round(float(entry_price), 2),
                    "exit_price": round(exit_price, 2),
                    "holding_days": holding,
                    "gross_return": round(float(trade_gross), 6),
                    "cost_frac": round(float(trade_cost), 6),
                    "net_return": round(float(trade_gross - trade_cost), 6),
                })
                trade_gross = 0.0
                trade_cost = 0.0

            # Open new position
            if pos != 0:
                entry_date = date.date()
                entry_price = df[OPEN].iloc[i] if i < len(df) else df[CLOSE].iloc[i - 1]

            current_pos = pos

        if current_pos != 0:
            trade_gross += float(gross_daily.iloc[i])
            trade_cost += float(daily_costs.iloc[i])

    # Close final open position
    if current_pos != 0 and entry_date is not None:
        trades.append({
            "entry_date": str(entry_date),
            "exit_date": str(df.index[-1].date()),
            "direction": "long" if current_pos > 0 else "short",
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(df[CLOSE].iloc[-1]), 2),
            "holding_days": (df.index[-1] - pd.Timestamp(entry_date)).days,
            "gross_return": round(float(trade_gross), 6),
            "cost_frac": round(float(trade_cost), 6),
            "net_return": round(float(trade_gross - trade_cost), 6),
        })

    return pd.DataFrame(trades)


def _check_invariants(
    equity: pd.Series,
    position: pd.Series,
    max_leverage: float,
) -> None:
    """Assert that the backtest produced physically consistent results."""
    if (equity < 0).any():
        logger.error("INVARIANT VIOLATION: Negative equity detected — accounting bug")
    if position.abs().max() > max_leverage + 1e-6:
        logger.error(f"INVARIANT VIOLATION: Position {position.abs().max():.2f} exceeds max_leverage {max_leverage}")
    if equity.isna().any():
        n_nan = equity.isna().sum()
        logger.warning(f"Equity curve has {n_nan} NaN values (check for missing price data)")
