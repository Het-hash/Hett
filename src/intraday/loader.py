"""Load and validate real 1-minute NIFTY 50 + India VIX data.

The strict minute dataset is an inner-join of NIFTY and VIX minute bars,
retaining only dates with exactly 375 aligned bars covering 09:15-15:29 IST.
Volume is intentionally absent (it is unreliable for this data and must not
be used).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_PATH = "/tmp/niftydata/nifty-real-data/nifty_vix_1min_strict.csv"

PRICE_COLS = ["open", "high", "low", "close"]
VIX_COLS = ["vix_open", "vix_high", "vix_low", "vix_close"]
DATA_COLS = PRICE_COLS + VIX_COLS + ["vix_envelope_repaired"]

SESSION_START = "09:15"
SESSION_END = "15:29"
BARS_PER_SESSION = 375


def load_minute_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the strict 1-minute dataset.

    Returns a DataFrame with a sorted DatetimeIndex (IST naive timestamps) and
    columns: open, high, low, close, vix_open, vix_high, vix_low, vix_close,
    vix_envelope_repaired.
    """
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the minute CSV.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index.name = "timestamp"

    missing = [c for c in DATA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Minute data missing required columns: {missing}")

    for c in PRICE_COLS + VIX_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    if df["vix_envelope_repaired"].dtype != bool:
        df["vix_envelope_repaired"] = (
            df["vix_envelope_repaired"].astype(str).str.lower().isin(["true", "1"])
        )

    df = df[DATA_COLS]
    logger.info(
        "Loaded %d 1-min bars (%s → %s)",
        len(df),
        df.index[0],
        df.index[-1],
    )
    return df


def validate_minute_data(df: pd.DataFrame) -> dict:
    """Validate structural and consistency properties of the minute data.

    Returns a quality report dict. ``passed`` is True only if there are no
    fatal issues (OHLC inconsistency, duplicates, NaNs in price columns).
    """
    report: dict = {}
    report["n_rows"] = int(len(df))
    report["first_timestamp"] = str(df.index[0])
    report["last_timestamp"] = str(df.index[-1])

    # Duplicates
    n_dupes = int(df.index.duplicated().sum())
    report["duplicate_timestamps"] = n_dupes

    # Monotonic index
    report["index_monotonic"] = bool(df.index.is_monotonic_increasing)

    # NaNs
    price_nans = int(df[PRICE_COLS].isna().sum().sum())
    vix_nans = int(df[VIX_COLS].isna().sum().sum())
    report["price_nans"] = price_nans
    report["vix_nans"] = vix_nans

    # OHLC consistency: high >= max(open, close, low); low <= min(open, close, high)
    hi = df["high"]
    lo = df["low"]
    op = df["open"]
    cl = df["close"]
    ohlc_bad = (
        (hi < lo)
        | (hi < op)
        | (hi < cl)
        | (lo > op)
        | (lo > cl)
    )
    report["ohlc_violations"] = int(ohlc_bad.sum())

    vhi = df["vix_high"]
    vlo = df["vix_low"]
    vix_ohlc_bad = (vhi < vlo) | (vhi < df["vix_open"]) | (vlo > df["vix_close"])
    # VIX envelope may be repaired; count but do not treat as fatal.
    report["vix_ohlc_violations"] = int(vix_ohlc_bad.sum())
    report["vix_envelope_repaired_count"] = int(df["vix_envelope_repaired"].sum())

    # Session structure
    dates = df.index.normalize()
    bars_per_day = df.groupby(dates).size()
    report["n_sessions"] = int(bars_per_day.shape[0])
    report["sessions_with_375_bars"] = int((bars_per_day == BARS_PER_SESSION).sum())
    report["sessions_not_375"] = int((bars_per_day != BARS_PER_SESSION).sum())
    report["min_bars_in_session"] = int(bars_per_day.min())
    report["max_bars_in_session"] = int(bars_per_day.max())

    # Boundary check: every session starts 09:15 and ends 15:29
    times = df.index.strftime("%H:%M")
    first_times = pd.Series(times, index=dates.values).groupby(level=0).first()
    last_times = pd.Series(times, index=dates.values).groupby(level=0).last()
    report["sessions_start_0915"] = int((first_times == SESSION_START).sum())
    report["sessions_end_1529"] = int((last_times == SESSION_END).sum())
    report["all_sessions_correct_boundaries"] = bool(
        (first_times == SESSION_START).all() and (last_times == SESSION_END).all()
    )

    # NIFTY/VIX alignment is implicit (joined dataset) — confirm no all-NaN VIX rows.
    report["vix_aligned"] = bool(vix_nans == 0)

    report["passed"] = bool(
        n_dupes == 0
        and price_nans == 0
        and report["ohlc_violations"] == 0
        and report["index_monotonic"]
    )

    logger.info(
        "Validation: sessions=%d, 375-bar=%d, ohlc_violations=%d, passed=%s",
        report["n_sessions"],
        report["sessions_with_375_bars"],
        report["ohlc_violations"],
        report["passed"],
    )
    return report
