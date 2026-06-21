"""Baseline strategies for comparison."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.data.schema import CLOSE, OPEN
from src.backtesting.engine import run_backtest, BacktestResult
from src.backtesting.costs import CostProfile, ETF_CONSERVATIVE, ZERO_COST
from src.utils.random_state import get_rng
from src.utils.logging import get_logger

logger = get_logger(__name__)


def buy_and_hold(
    df: pd.DataFrame,
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """Always long from first bar."""
    signal = pd.Series(True, index=df.index)
    return run_backtest(df, signal, direction="long", cost_profile=cost_profile,
                        initial_capital=initial_capital)


def cash_only(
    df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """Always flat — equity stays constant at initial capital (0% return)."""
    signal = pd.Series(False, index=df.index)
    return run_backtest(df, signal, direction="long", cost_profile=ZERO_COST,
                        initial_capital=initial_capital)


def sma_trend(
    df: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """Long when fast SMA > slow SMA."""
    fast_ma = df[CLOSE].rolling(fast).mean()
    slow_ma = df[CLOSE].rolling(slow).mean()
    signal = fast_ma > slow_ma
    return run_backtest(df, signal, direction="long", cost_profile=cost_profile,
                        initial_capital=initial_capital)


def time_series_momentum(
    df: pd.DataFrame,
    lookback: int = 252,
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """Long when 12-month return is positive."""
    signal = df[CLOSE].pct_change(lookback) > 0
    return run_backtest(df, signal, direction="long", cost_profile=cost_profile,
                        initial_capital=initial_capital)


def mean_reversion_baseline(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = -1.5,
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """Long when zscore below threshold, hold for 5 days."""
    from src.features.mean_reversion import rolling_zscore
    z = rolling_zscore(df, window)
    raw_signal = z < threshold
    # Hold for 5 days after signal
    signal = raw_signal.rolling(5).max().fillna(False).astype(bool)
    return run_backtest(df, signal, direction="long", cost_profile=cost_profile,
                        initial_capital=initial_capital)


def donchian_breakout(
    df: pd.DataFrame,
    window: int = 20,
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """Long on N-day high breakout, exit on N-day low breakdown."""
    from src.features.breakouts import high_breakout, low_breakdown
    in_pos = False
    signals = pd.Series(False, index=df.index)
    for i, date in enumerate(df.index):
        if not in_pos:
            if i > 0 and high_breakout(df.iloc[:i+1], window).iloc[-1]:
                in_pos = True
        else:
            if low_breakdown(df.iloc[:i+1], window).iloc[-1]:
                in_pos = False
        signals.iloc[i] = in_pos

    return run_backtest(df, signals, direction="long", cost_profile=cost_profile,
                        initial_capital=initial_capital)


def random_entry(
    df: pd.DataFrame,
    signal_frequency: float = 0.3,
    seed: int = 42,
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """Random entry with fixed frequency — control for luck."""
    rng = get_rng(seed)
    signal = pd.Series(
        rng.random(len(df)) < signal_frequency,
        index=df.index,
        dtype=bool,
    )
    return run_backtest(df, signal, direction="long", cost_profile=cost_profile,
                        initial_capital=initial_capital)


def run_all_baselines(
    df: pd.DataFrame,
    cost_profile: CostProfile = ETF_CONSERVATIVE,
    initial_capital: float = 1_000_000.0,
    candidate_signal_frequency: float = 0.3,
) -> Dict[str, BacktestResult]:
    """Run all baseline strategies and return as a dict."""
    logger.info("Running baseline strategies…")
    baselines = {
        "buy_and_hold": buy_and_hold(df, cost_profile, initial_capital),
        "cash_only": cash_only(df, initial_capital),
        "sma_50_200": sma_trend(df, 50, 200, cost_profile, initial_capital),
        "momentum_252d": time_series_momentum(df, 252, cost_profile, initial_capital),
        "mean_reversion": mean_reversion_baseline(df, 20, -1.5, cost_profile, initial_capital),
        "donchian_20": donchian_breakout(df, 20, cost_profile, initial_capital),
        "random_entry": random_entry(df, candidate_signal_frequency, 42, cost_profile, initial_capital),
    }
    logger.info(f"Completed {len(baselines)} baseline strategies")
    return baselines
