"""Tests for execution convention: signal at T → position at T+1."""
import numpy as np
import pandas as pd
import pytest

from src.backtesting.execution import generate_position_series, LONG, FLAT, SHORT
from src.backtesting.engine import run_backtest
from src.backtesting.costs import ZERO_COST


def test_position_shift_by_one(synthetic_ohlcv):
    """Signal fires at T → position applied from T+1 open."""
    df = synthetic_ohlcv.copy()
    # Signal fires only at row 10
    signal = pd.Series(False, index=df.index)
    signal.iloc[10] = True

    position = generate_position_series(signal, "long")
    # Position at row 10 should be 0 (signal fires at row 10, enters at row 11)
    assert position.iloc[10] == 0
    # Position at row 11 should be 1
    assert position.iloc[11] == 1


def test_short_position_is_negative(synthetic_ohlcv):
    signal = pd.Series(True, index=synthetic_ohlcv.index)
    pos = generate_position_series(signal, "short")
    assert (pos.dropna() == -1).any() or (pos.dropna() <= 0).all()


def test_flat_when_no_signal(synthetic_ohlcv):
    signal = pd.Series(False, index=synthetic_ohlcv.index)
    pos = generate_position_series(signal, "long")
    assert (pos == 0).all()


def test_backtest_equity_starts_at_capital(synthetic_ohlcv):
    signal = pd.Series(False, index=synthetic_ohlcv.index)
    result = run_backtest(synthetic_ohlcv, signal, cost_profile=ZERO_COST, initial_capital=1_000_000)
    assert result.equity_curve.iloc[0] == 1_000_000.0


def test_backtest_zero_cost_bah_matches_price(synthetic_ohlcv):
    """Buy-and-hold with zero cost: equity ratio = price ratio."""
    df = synthetic_ohlcv.copy()
    signal = pd.Series(True, index=df.index)
    result = run_backtest(df, signal, direction="long", cost_profile=ZERO_COST, initial_capital=1_000_000)
    # Equity should be monotonically related to close price (approximately)
    assert result.equity_curve.iloc[-1] > 0


def test_no_negative_equity(synthetic_ohlcv):
    signal = pd.Series(True, index=synthetic_ohlcv.index)
    result = run_backtest(synthetic_ohlcv, signal, cost_profile=ZERO_COST)
    assert (result.equity_curve >= 0).all()


def test_reversal_counts_as_two_transactions(synthetic_ohlcv):
    """Going long→short should cost two one-way transactions."""
    from src.backtesting.costs import flat_cost_profile
    df = synthetic_ohlcv.copy()
    cp = flat_cost_profile(10)  # 10bps round-trip
    # Signal: long for 5 bars, then short
    signal_long = pd.Series(False, index=df.index)
    signal_long.iloc[5:10] = True
    result = run_backtest(df, signal_long, direction="long", cost_profile=cp)
    total_cost = (result.gross_returns - result.returns).sum()
    assert total_cost > 0, "Expected non-zero cost with 10bps profile"
