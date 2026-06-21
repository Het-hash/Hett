"""Tests for causal multi-timeframe aggregation and next-bar execution."""
import numpy as np
import pandas as pd
import pytest

from src.intraday import aggregator
from src.intraday.engine import run_intraday_backtest


def _minute_two_sessions():
    frames = []
    for day in ["2020-01-06", "2020-01-07"]:
        ts = pd.date_range(f"{day} 09:15", f"{day} 15:29", freq="1min")
        n = len(ts)
        op = np.arange(n, dtype=float) + 100
        cl = op + 0.5
        hi = op + 1.0
        lo = op - 1.0
        vix = np.full(n, 15.0)
        frames.append(pd.DataFrame({
            "open": op, "high": hi, "low": lo, "close": cl,
            "vix_open": vix, "vix_high": vix, "vix_low": vix, "vix_close": vix,
            "vix_envelope_repaired": False,
        }, index=ts))
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def minute_df():
    return _minute_two_sessions()


def test_bar_boundaries_5min(minute_df):
    tf = aggregator.resample_to_tf(minute_df, 5)
    day1 = tf[tf.index.normalize() == pd.Timestamp("2020-01-06")]
    times = day1.index.strftime("%H:%M").tolist()
    assert times[0] == "09:15"
    assert times[1] == "09:20"
    assert times[2] == "09:25"
    # 375 minutes / 5 = 75 bars exactly
    assert len(day1) == 75


def test_no_cross_session(minute_df):
    tf = aggregator.resample_to_tf(minute_df, 30)
    # No aggregated bar should mix two dates: check each bar's date is one date.
    dates = tf.index.normalize().unique()
    assert len(dates) == 2
    # Total bars = per-session bars summed, sessions independent.
    counts = tf.groupby(tf.index.normalize()).size()
    # 375/30 = 12.5 -> 13 bars (last partial) per session
    assert (counts == 13).all()


def test_ohlc_aggregation(minute_df):
    tf = aggregator.resample_to_tf(minute_df, 5)
    first_bar = tf.iloc[0]
    # first 5 1-min bars of session 1
    first5 = minute_df.iloc[:5]
    assert first_bar["open"] == first5["open"].iloc[0]
    assert first_bar["close"] == first5["close"].iloc[-1]
    assert first_bar["high"] == first5["high"].max()
    assert first_bar["low"] == first5["low"].min()


def test_final_partial_bar(minute_df):
    tf = aggregator.resample_to_tf(minute_df, 30)
    day1 = tf[tf.index.normalize() == pd.Timestamp("2020-01-06")]
    # Last bar should aggregate only the trailing 15 1-min bars (375 = 12*30+15)
    last_open_time = day1.index[-1].strftime("%H:%M")
    # 12 full 30-min buckets cover 09:15..15:14; partial starts 15:15
    assert last_open_time == "15:15"


def test_next_bar_execution():
    # Signal at bar T close must only execute at T+1 open.
    ts = pd.date_range("2020-01-06 09:15", "2020-01-06 15:29", freq="1min")
    n = len(ts)
    price = np.arange(n, dtype=float) + 100
    df = pd.DataFrame({
        "open": price, "high": price + 0.1, "low": price - 0.1, "close": price,
        "vix_open": 15.0, "vix_high": 15.0, "vix_low": 15.0, "vix_close": 15.0,
    }, index=ts)
    df.index.name = "timestamp"
    sig = pd.Series(0, index=ts)
    sig.iloc[10] = 1  # signal at bar 10 close
    res = run_intraday_backtest(df, sig, cost_bps=0.0)
    assert len(res["trades"]) == 1
    tr = res["trades"][0]
    # entry must be at bar 11 open, not bar 10
    assert tr["entry_time"] == ts[11]
    assert tr["entry_price"] == df["open"].iloc[11]
