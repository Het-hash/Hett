"""Tests for Phase 1C context filters and edge decay utilities."""
import numpy as np
import pandas as pd
import pytest

from src.intraday.context_filters import (
    compute_daily_context,
    merge_context_to_tf,
    SINGLE_FILTERS,
    TWO_FILTER_COMBOS,
    apply_filter,
    filter_trades,
)
from src.intraday.edge_decay import (
    per_year_stats,
    rolling_expectancy,
    pre_post_2020_split,
    be_cost_through_time,
    classify_decay,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_1min_df(n_sessions: int = 10, bars_per_session: int = 375) -> pd.DataFrame:
    """Create a minimal 1-min OHLC + VIX DataFrame."""
    dates = pd.date_range("2020-01-02", periods=n_sessions, freq="B")
    rows = []
    rng = np.random.default_rng(42)
    base = 10000.0
    for d in dates:
        price = base + rng.normal(0, 100)
        for m in range(bars_per_session):
            t = pd.Timestamp(f"{d.date()} 09:15:00") + pd.Timedelta(minutes=m)
            rows.append({
                "open": price + rng.normal(0, 5),
                "high": price + abs(rng.normal(10, 5)),
                "low":  price - abs(rng.normal(10, 5)),
                "close": price + rng.normal(0, 5),
                "vix_open": 15 + rng.normal(0, 1),
                "vix_close": 15 + rng.normal(0, 1),
            })
            price += rng.normal(0, 2)
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([r for r in
                                  (pd.Timestamp(f"{d.date()} 09:15:00") + pd.Timedelta(minutes=m)
                                   for d in dates for m in range(bars_per_session))])
    return df


# ── compute_daily_context ────────────────────────────────────────────────────

def test_compute_daily_context_shape():
    df = _make_1min_df(n_sessions=20)
    ctx = compute_daily_context(df)
    assert len(ctx) == 20
    required = ["prev_close", "prev_return", "prev_range", "vix_pct_rank",
                "gap_pts", "daily_bull_20", "day_of_week"]
    for col in required:
        assert col in ctx.columns, f"Missing column: {col}"


def test_compute_daily_context_causal():
    """prev_close on day 1 should be NaN (no prior session)."""
    df = _make_1min_df(n_sessions=30)
    ctx = compute_daily_context(df)
    assert pd.isna(ctx["prev_close"].iloc[0]), "First row prev_close must be NaN"
    # Day 2 should not be NaN
    assert not pd.isna(ctx["prev_close"].iloc[1])


def test_daily_context_vix_pct_rank_range():
    """vix_pct_rank must be in [0, 1] where not NaN."""
    df = _make_1min_df(n_sessions=100)
    ctx = compute_daily_context(df)
    valid = ctx["vix_pct_rank"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_daily_context_prev_range_non_negative():
    df = _make_1min_df(n_sessions=30)
    ctx = compute_daily_context(df)
    valid = ctx["prev_range"].dropna()
    assert (valid >= 0).all()


# ── merge_context_to_tf ───────────────────────────────────────────────────────

def test_merge_context_to_tf():
    df = _make_1min_df(n_sessions=20)
    ctx = compute_daily_context(df)
    # Simulate a 5-min TF aggregated df (just use 1-min as proxy)
    merged = merge_context_to_tf(df, ctx)
    assert "prev_return" in merged.columns
    assert len(merged) == len(df)


# ── Single filters ────────────────────────────────────────────────────────────

def test_all_single_filters_return_bool():
    df = _make_1min_df(n_sessions=100)
    ctx = compute_daily_context(df)
    merged = merge_context_to_tf(df, ctx)
    for fname, ffunc in SINGLE_FILTERS.items():
        result = ffunc(merged)
        assert result.dtype == bool or result.dtype == np.bool_, \
            f"Filter {fname} returned non-bool dtype: {result.dtype}"


def test_filter_prev_bullish_bearish_exclusive():
    df = _make_1min_df(n_sessions=100)
    ctx = compute_daily_context(df)
    merged = merge_context_to_tf(df, ctx)
    bull = SINGLE_FILTERS["prev_bullish"](merged)
    bear = SINGLE_FILTERS["prev_bearish"](merged)
    # They should not both be True on the same bar (ignoring NaN rows)
    overlap = bull & bear
    assert not overlap.any(), "prev_bullish and prev_bearish cannot both be True"


def test_calendar_filters_partition():
    df = _make_1min_df(n_sessions=100)
    ctx = compute_daily_context(df)
    merged = merge_context_to_tf(df, ctx)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    counts = sum(SINGLE_FILTERS[d](merged).astype(int) for d in days)
    # Each bar belongs to exactly one day
    assert (counts == 1).all()


# ── apply_filter / filter_trades ──────────────────────────────────────────────

def test_apply_filter_zeros_out_false_bars():
    idx = pd.date_range("2020-01-02 09:15", periods=10, freq="min")
    signal = pd.Series([1, 0, -1, 1, 0, 1, 0, -1, 0, 1], index=idx)
    mask = pd.Series([True, True, False, False, True, True, False, False, True, True], index=idx)
    result = apply_filter(signal, mask)
    assert result.iloc[2] == 0  # was -1, masked
    assert result.iloc[3] == 0  # was 1, masked
    assert result.iloc[0] == 1  # unchanged


def test_filter_trades_by_mask():
    idx = pd.date_range("2020-01-02 09:15", periods=5, freq="min")
    mask = pd.Series([True, False, True, False, True], index=idx)
    trades = [
        {"entry_time": idx[0], "net_pnl": 10},
        {"entry_time": idx[1], "net_pnl": -5},
        {"entry_time": idx[2], "net_pnl": 20},
    ]
    result = filter_trades(trades, mask)
    assert len(result) == 2
    assert result[0]["entry_time"] == idx[0]
    assert result[1]["entry_time"] == idx[2]


# ── edge_decay ────────────────────────────────────────────────────────────────

def _make_trades(years_and_pnls: dict) -> list:
    trades = []
    for year, pnls in years_and_pnls.items():
        for i, pnl in enumerate(pnls):
            trades.append({
                "entry_time": pd.Timestamp(f"{year}-01-{(i % 28) + 1:02d} 09:30"),
                "gross_pnl": pnl,
                "net_pnl": pnl - 20,
            })
    return trades


def test_per_year_stats_correct_years():
    trades = _make_trades({2018: [10]*15, 2019: [20]*15, 2020: [-5]*15})
    df = per_year_stats(trades, min_trades=10)
    assert set(df.index) == {2018, 2019, 2020}
    assert abs(df.loc[2019, "avg_gross_pnl"] - 20) < 1e-6


def test_classify_decay_declining():
    trades = _make_trades({
        2015: [50]*20, 2016: [30]*20, 2017: [10]*20, 2018: [-5]*20, 2019: [-15]*20
    })
    yearly = per_year_stats(trades, min_trades=10)
    result = classify_decay(yearly)
    assert result == "DECLINING"


def test_classify_decay_stable():
    trades = _make_trades({
        2015: [20]*20, 2016: [22]*20, 2017: [18]*20, 2018: [21]*20, 2019: [19]*20
    })
    yearly = per_year_stats(trades, min_trades=10)
    result = classify_decay(yearly)
    assert result == "STABLE"


def test_pre_post_2020():
    trades = _make_trades({2018: [50]*20, 2019: [40]*20, 2021: [10]*20, 2022: [5]*20})
    result = pre_post_2020_split(trades)
    assert "pre_2020" in result
    assert "post_2020" in result
    assert result["decay"] > 0  # pre > post


def test_be_cost_through_time():
    trades = _make_trades({2018: [50]*15, 2019: [30]*15})
    df = be_cost_through_time(trades, avg_price=20000.0)
    assert 2018 in df.index and 2019 in df.index
    assert df.loc[2018, "be_bps"] > df.loc[2019, "be_bps"]
