"""Tests for performance metrics."""
import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import compute_metrics, monthly_returns_table, annual_returns_table
from src.backtesting.engine import run_backtest
from src.backtesting.costs import ZERO_COST


def _make_flat_returns(n=500, ret=0.001):
    idx = pd.bdate_range("2015-01-02", periods=n)
    returns = pd.Series(ret, index=idx)
    equity = (1 + returns).cumprod() * 1_000_000
    positions = pd.Series(1.0, index=idx)
    return returns, equity, positions


def test_positive_cagr_for_positive_returns():
    ret, eq, pos = _make_flat_returns(ret=0.001)
    m = compute_metrics(ret, eq, pos)
    assert m["cagr"] > 0


def test_zero_returns_zero_cagr():
    ret, eq, pos = _make_flat_returns(ret=0.0)
    eq = pd.Series(1_000_000.0, index=ret.index)
    m = compute_metrics(ret, eq, pos)
    assert abs(m["cagr"]) < 1e-4


def test_max_drawdown_negative(synthetic_ohlcv):
    from src.backtesting.baselines import buy_and_hold
    result = buy_and_hold(synthetic_ohlcv, ZERO_COST)
    m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
    assert m["max_drawdown"] <= 0


def test_sharpe_ratio_reasonable(synthetic_ohlcv):
    result = run_backtest(synthetic_ohlcv, pd.Series(True, index=synthetic_ohlcv.index),
                          cost_profile=ZERO_COST)
    m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
    assert -10 < m["sharpe_ratio"] < 10


def test_monthly_returns_table_shape(synthetic_ohlcv):
    result = run_backtest(synthetic_ohlcv, pd.Series(True, index=synthetic_ohlcv.index),
                          cost_profile=ZERO_COST)
    tbl = monthly_returns_table(result.equity_curve)
    # Should have at most 12 columns
    assert tbl.shape[1] <= 12


def test_exposure_is_fraction(synthetic_ohlcv):
    signal = pd.Series(True, index=synthetic_ohlcv.index)
    result = run_backtest(synthetic_ohlcv, signal, cost_profile=ZERO_COST)
    m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
    assert 0 <= m["exposure_avg"] <= 1.0


def test_flat_signal_gives_zero_return(synthetic_ohlcv):
    signal = pd.Series(False, index=synthetic_ohlcv.index)
    result = run_backtest(synthetic_ohlcv, signal, cost_profile=ZERO_COST)
    total_ret = result.returns.sum()
    assert abs(total_ret) < 1e-10
