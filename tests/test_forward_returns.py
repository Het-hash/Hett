"""Tests for forward return calculation — no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.edge_discovery.forward_returns import compute_forward_returns, forward_return_stats
from src.data.schema import OPEN, CLOSE


def test_forward_return_alignment(synthetic_ohlcv):
    """
    fwd_open_5d at row T = (close[T+5] - open[T+1]) / open[T+1].
    Verify this manually for a few rows.
    """
    df = synthetic_ohlcv.copy()
    fwd = compute_forward_returns(df, horizons=[5])

    for i in range(10, 20):
        entry_open = float(df[OPEN].iloc[i + 1])
        exit_close = float(df[CLOSE].iloc[i + 5])
        expected = (exit_close - entry_open) / entry_open
        actual = float(fwd["fwd_open_5d"].iloc[i])
        assert abs(expected - actual) < 1e-8, f"Mismatch at row {i}"


def test_fwd_returns_nan_at_end(synthetic_ohlcv):
    """Last h rows must be NaN (no exit price available)."""
    df = synthetic_ohlcv.copy()
    fwd = compute_forward_returns(df, horizons=[5])
    assert fwd["fwd_open_5d"].iloc[-5:].isna().all()


def test_forward_return_not_usable_at_signal_time(synthetic_ohlcv):
    """
    fwd_open_1d at row T = (close[T+1] - open[T+1]) / open[T+1].
    Modifying open[T+1] must change fwd_open_1d at row T.
    """
    df = synthetic_ohlcv.copy()
    fwd = compute_forward_returns(df, horizons=[1])
    # Modify open at row 50 — this should change fwd_open_1d at row 49
    df_mod = df.copy()
    open_col = df_mod.columns.get_loc(OPEN)
    df_mod.iloc[50, open_col] = df_mod.iloc[50, open_col] * 100  # extreme change
    fwd_mod = compute_forward_returns(df_mod, horizons=[1])
    assert fwd["fwd_open_1d"].iloc[49] != fwd_mod["fwd_open_1d"].iloc[49], \
        "fwd_open_1d at T=49 should depend on open[50] (future data)"


def test_forward_return_stats_basic(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    fwd = compute_forward_returns(df, horizons=[5])
    signal = pd.Series(False, index=df.index)
    signal.iloc[10:50] = True  # 40 signals
    stats = forward_return_stats(signal, fwd, "fwd_open_5d")
    assert stats["n"] == 40
    assert 0 <= stats["hit_rate"] <= 1
    assert "mean" in stats
    assert "t_stat" in stats


def test_long_short_return_opposite(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    fwd = compute_forward_returns(df, horizons=[5])
    signal = pd.Series(True, index=df.index)
    long_mean = fwd["fwd_open_5d"].mean()
    short_mean = fwd["fwd_open_short_5d"].mean()
    assert abs(long_mean + short_mean) < 1e-10, "Long and short returns must sum to zero"
