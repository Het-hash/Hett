"""
Tests that verify no look-ahead leakage in feature calculations.

Critical property: feature value at index T must only depend on
data at indices ≤ T (i.e., at[T] and prior rows).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.schema import CLOSE, OPEN
from src.features.returns import rolling_return, overnight_gap
from src.features.trend import sma, price_minus_sma
from src.features.momentum import rsi, consecutive_down
from src.features.mean_reversion import rolling_zscore, bollinger_position
from src.features.volatility import atr, realised_vol
from src.features.breakouts import high_breakout, low_breakdown


def test_rolling_return_no_leakage(synthetic_ohlcv):
    """Modifying future data must not change past rolling return values."""
    df = synthetic_ohlcv.copy()
    ret_before = rolling_return(df, 5)

    df_modified = df.copy()
    df_modified.iloc[100:, df_modified.columns.get_loc(CLOSE)] *= 10

    ret_after = rolling_return(df_modified, 5)
    # Values before row 100 should be identical
    pd.testing.assert_series_equal(ret_before.iloc[:95], ret_after.iloc[:95])


def test_rsi_no_leakage(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    rsi_before = rsi(df, 14)
    df_modified = df.copy()
    df_modified.iloc[200:, df_modified.columns.get_loc(CLOSE)] *= 5
    rsi_after = rsi(df_modified, 14)
    pd.testing.assert_series_equal(rsi_before.iloc[:190], rsi_after.iloc[:190], check_exact=False, rtol=1e-5)


def test_atr_no_leakage(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    atr_before = atr(df, 14)
    df_modified = df.copy()
    df_modified.iloc[300:, df_modified.columns.get_loc(CLOSE)] *= 100
    atr_after = atr(df_modified, 14)
    pd.testing.assert_series_equal(atr_before.iloc[:290], atr_after.iloc[:290], check_exact=False, rtol=1e-4)


def test_zscore_no_leakage(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    z_before = rolling_zscore(df, 20)
    df_modified = df.copy()
    df_modified.iloc[200:, df_modified.columns.get_loc(CLOSE)] = 9999
    z_after = rolling_zscore(df_modified, 20)
    pd.testing.assert_series_equal(z_before.iloc[:180], z_after.iloc[:180], check_exact=False, rtol=1e-5)


def test_breakout_uses_prior_window(synthetic_ohlcv):
    """high_breakout uses shift(1) so today is not in the lookback window."""
    df = synthetic_ohlcv.copy()
    sig = high_breakout(df, 20)
    # Manually check: if close[T] > max(close[T-20:T-1]), it should be True
    for i in range(25, 30):
        prior_max = df[CLOSE].iloc[i - 20:i].max()
        expected = df[CLOSE].iloc[i] > prior_max
        assert sig.iloc[i] == expected, f"Breakout signal wrong at row {i}"


def test_overnight_gap_uses_prior_close(synthetic_ohlcv):
    """Gap at T = (open_T - close_{T-1}) / close_{T-1} — no future data."""
    df = synthetic_ohlcv.copy()
    gap = overnight_gap(df)
    # Modify future opens — should not change gaps in the past
    df_mod = df.copy()
    df_mod.iloc[100:, df_mod.columns.get_loc(OPEN)] *= 2
    gap_mod = overnight_gap(df_mod)
    pd.testing.assert_series_equal(gap.iloc[:99], gap_mod.iloc[:99])


def test_centred_rolling_is_detected_as_leaky(synthetic_ohlcv_with_leakage):
    """
    A feature computed with center=True WILL change when future values change.
    This test confirms the detection method works.
    """
    df = synthetic_ohlcv_with_leakage.copy()
    leaky_before = df["leaky_feature"].copy()
    df_modified = df.copy()
    df_modified.iloc[100:, df_modified.columns.get_loc(CLOSE)] *= 10
    df_modified["leaky_feature"] = df_modified[CLOSE].rolling(5, center=True).mean()
    # At rows 97-99, the centred window includes rows 100+, so values change
    changed = (leaky_before.iloc[95:100] != df_modified["leaky_feature"].iloc[95:100]).any()
    assert changed, "Expected centred rolling to leak future data"
