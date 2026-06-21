"""Tests for data schema and validation."""
import numpy as np
import pandas as pd
import pytest

from src.data.schema import OPEN, HIGH, LOW, CLOSE, VOLUME, TIMESTAMP, enforce_schema
from src.data.validator import validate_ohlcv
from src.data.cleaner import clean_ohlcv


def test_enforce_schema_renames_columns():
    df = pd.DataFrame({
        "Open": [100.0], "High": [105.0], "Low": [99.0],
        "Close": [103.0], "Volume": [1_000_000.0],
    }, index=pd.DatetimeIndex(["2020-01-02"], name="Date"))
    df = enforce_schema(df, "TEST", "yfinance")
    assert OPEN in df.columns
    assert HIGH in df.columns
    assert LOW in df.columns
    assert CLOSE in df.columns
    assert VOLUME in df.columns
    assert df.index.name == TIMESTAMP


def test_validate_detects_duplicate_dates(synthetic_ohlcv):
    df = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[:5]])
    df = df.sort_index()
    report = validate_ohlcv(df, "TEST", "test", "now")
    assert report.duplicate_dates > 0


def test_validate_detects_nonpositive_price(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    df.iloc[5, df.columns.get_loc(CLOSE)] = -100
    report = validate_ohlcv(df, "TEST", "test", "now")
    assert any("non-positive" in e.lower() for e in report.confirmed_errors)


def test_validate_detects_high_lt_low(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    df.iloc[10, df.columns.get_loc(HIGH)] = df.iloc[10][LOW] - 1
    report = validate_ohlcv(df, "TEST", "test", "now")
    assert any("high" in e.lower() and "low" in e.lower() for e in report.confirmed_errors)


def test_validate_detects_extreme_returns(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    # Insert a 50% spike
    df.iloc[50, df.columns.get_loc(CLOSE)] *= 1.5
    report = validate_ohlcv(df, "TEST", "test", "now")
    assert len(report.extreme_returns) > 0


def test_clean_removes_all_nan_rows(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    # Insert an all-NaN row
    nan_date = pd.Timestamp("2099-01-01")
    df.loc[nan_date] = np.nan
    df = df.sort_index()
    cleaned, decisions = clean_ohlcv(df)
    assert nan_date not in cleaned.index
    assert any(d.action == "removed" and "NaN" in d.reason for d in decisions)


def test_clean_removes_duplicates(synthetic_ohlcv):
    df = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[:3]]).sort_index()
    cleaned, decisions = clean_ohlcv(df)
    assert cleaned.index.duplicated().sum() == 0


def test_quality_report_is_serialisable(synthetic_ohlcv):
    report = validate_ohlcv(synthetic_ohlcv, "TEST", "test", "2024-01-01")
    d = report.to_dict()
    import json
    # Must be JSON-serialisable
    json.dumps(d)
