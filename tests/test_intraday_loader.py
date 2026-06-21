"""Tests for the intraday loader/validator using synthetic minute data."""
import numpy as np
import pandas as pd
import pytest

from src.intraday import loader


def _synthetic_minute(n_sessions=3, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    base_date = pd.Timestamp("2020-01-06")
    d = 0
    made = 0
    while made < n_sessions:
        day = base_date + pd.Timedelta(days=d)
        d += 1
        if day.weekday() >= 5:
            continue
        ts = pd.date_range(f"{day.date()} 09:15", f"{day.date()} 15:29", freq="1min")
        assert len(ts) == 375
        price = 10000 + np.cumsum(rng.normal(0, 1, 375))
        op = price
        cl = price + rng.normal(0, 0.5, 375)
        hi = np.maximum(op, cl) + np.abs(rng.normal(0, 0.5, 375))
        lo = np.minimum(op, cl) - np.abs(rng.normal(0, 0.5, 375))
        vix = 15 + np.cumsum(rng.normal(0, 0.05, 375))
        frames.append(pd.DataFrame({
            "open": op, "high": hi, "low": lo, "close": cl,
            "vix_open": vix, "vix_high": vix + 0.1, "vix_low": vix - 0.1,
            "vix_close": vix, "vix_envelope_repaired": False,
        }, index=ts))
        made += 1
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def minute_df():
    return _synthetic_minute(3)


def test_schema_columns(minute_df):
    for c in loader.DATA_COLS:
        assert c in minute_df.columns


def test_375_bars_per_session(minute_df):
    rep = loader.validate_minute_data(minute_df)
    assert rep["sessions_with_375_bars"] == rep["n_sessions"]
    assert rep["n_sessions"] == 3


def test_session_boundaries(minute_df):
    rep = loader.validate_minute_data(minute_df)
    assert rep["all_sessions_correct_boundaries"]
    assert rep["sessions_start_0915"] == 3
    assert rep["sessions_end_1529"] == 3


def test_no_duplicates(minute_df):
    rep = loader.validate_minute_data(minute_df)
    assert rep["duplicate_timestamps"] == 0
    dup = pd.concat([minute_df, minute_df.iloc[[0]]])
    rep2 = loader.validate_minute_data(dup)
    assert rep2["duplicate_timestamps"] == 1


def test_nifty_vix_alignment(minute_df):
    rep = loader.validate_minute_data(minute_df)
    assert rep["vix_aligned"]
    assert rep["vix_nans"] == 0
