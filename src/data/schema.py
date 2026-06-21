"""Canonical data schema and column name constants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


# ── Column names ────────────────────────────────────────────────────────────
TIMESTAMP = "timestamp"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
ADJ_CLOSE = "adj_close"
VOLUME = "volume"
SOURCE = "source"
SYMBOL = "symbol"

REQUIRED_OHLC = [OPEN, HIGH, LOW, CLOSE]
REQUIRED_COLS = REQUIRED_OHLC + [VOLUME]
ALL_COLS = REQUIRED_COLS + [ADJ_CLOSE, SOURCE, SYMBOL]


@dataclass
class DataQualityReport:
    symbol: str
    source: str
    downloaded_at: str
    first_date: str
    last_date: str
    total_rows: int
    missing_values: dict
    duplicate_dates: int
    suspected_gaps: List[str]
    extreme_returns: List[dict]
    zero_volume_dates: List[str]
    validation_warnings: List[str]
    confirmed_errors: List[str]
    rows_removed: int
    rows_flagged: int
    notes: List[str]

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


def enforce_schema(df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    """Rename columns to canonical names and add metadata columns."""
    rename_map = {
        "Open": OPEN, "High": HIGH, "Low": LOW, "Close": CLOSE,
        "Adj Close": ADJ_CLOSE, "Volume": VOLUME,
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if df.index.name in ("Date", "Datetime", "date", "datetime", "timestamp"):
        df.index.name = TIMESTAMP
    else:
        df.index.name = TIMESTAMP

    df[SYMBOL] = symbol
    df[SOURCE] = source

    # Ensure float dtypes for price columns
    for col in REQUIRED_OHLC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if VOLUME in df.columns:
        df[VOLUME] = pd.to_numeric(df[VOLUME], errors="coerce")

    return df
