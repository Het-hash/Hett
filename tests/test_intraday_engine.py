"""Tests for the intraday execution engine constraints."""
import numpy as np
import pandas as pd
import pytest

from src.intraday.engine import run_intraday_backtest


def _session(day="2020-01-06", trend=1.0):
    ts = pd.date_range(f"{day} 09:15", f"{day} 15:29", freq="1min")
    n = len(ts)
    price = 100 + trend * np.arange(n, dtype=float)
    df = pd.DataFrame({
        "open": price, "high": price + 0.1, "low": price - 0.1, "close": price,
        "vix_open": 15.0, "vix_high": 15.0, "vix_low": 15.0, "vix_close": 15.0,
    }, index=ts)
    df.index.name = "timestamp"
    return df


def test_no_overnight_position():
    df = pd.concat([_session("2020-01-06"), _session("2020-01-07")])
    sig = pd.Series(1, index=df.index)  # always long
    res = run_intraday_backtest(df, sig, cost_bps=0.0)
    pos = res["position"]
    # position at last bar of each session must be 0
    for _, day in pos.groupby(pos.index.normalize()):
        assert day.iloc[-1] == 0


def test_forced_squareoff():
    df = _session()
    sig = pd.Series(1, index=df.index)
    res = run_intraday_backtest(df, sig, cost_bps=0.0, squareoff_time="15:20")
    pos = res["position"]
    after = pos[pos.index.time >= pd.Timestamp("15:20").time()]
    assert (after == 0).all()
    # there must be a trade that exits at or before 15:20
    exits = [pd.Timestamp(t["exit_time"]).time() for t in res["trades"]]
    assert all(e <= pd.Timestamp("15:20").time() for e in exits)


def test_no_entry_after_cutoff():
    df = _session()
    sig = pd.Series(0, index=df.index)
    # signal fires at 14:59 close -> entry 15:00 blocked (>= cutoff)
    sig.loc[f"2020-01-06 15:30":] = 0
    sig.loc["2020-01-06 15:05"] = 1  # late signal
    res = run_intraday_backtest(df, sig, cost_bps=0.0, no_entry_after="15:00")
    for t in res["trades"]:
        assert pd.Timestamp(t["entry_time"]).time() < pd.Timestamp("15:00").time()


def test_reversal_cost():
    # A direct long->short reversal charges two legs (= round trip),
    # while a single entry+exit also charges two legs. Verify reversal
    # produces two trades and total cost equals two round trips.
    df = _session()
    sig = pd.Series(0, index=df.index)
    sig.iloc[10:20] = 1   # hold long; enter at bar 11 open
    sig.iloc[20:30] = -1  # reverse to short at bar 21 open
    res = run_intraday_backtest(df, sig, cost_bps=10.0)
    assert len(res["trades"]) == 2
    long_tr, short_tr = res["trades"]
    assert long_tr["direction"] == "long"
    assert short_tr["direction"] == "short"
    leg = 10.0 / 2 / 10000
    # long trade cost = (entry+exit)*leg
    exp_long = (long_tr["entry_price"] + long_tr["exit_price"]) * leg
    assert long_tr["cost"] == pytest.approx(exp_long, rel=1e-9)
    # the reversal exit price of long == entry price of short (same bar open)
    assert long_tr["exit_price"] == short_tr["entry_price"]
