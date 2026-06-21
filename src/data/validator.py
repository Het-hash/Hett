"""Data-quality validation producing structured reports."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.schema import (
    OPEN, HIGH, LOW, CLOSE, VOLUME, TIMESTAMP, DataQualityReport
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# NSE approximate trading calendar — weekdays only (simplified)
# We flag gaps > 5 calendar days as suspected gaps (allows for holidays).
MAX_CALENDAR_GAP_DAYS = 10


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    downloaded_at: str,
    known_holiday_dates: Optional[List[date]] = None,
) -> DataQualityReport:
    """
    Run all validation checks and return a structured DataQualityReport.
    Does not modify df.
    """
    warnings: List[str] = []
    errors: List[str] = []
    notes: List[str] = []

    # ── Basic counts ─────────────────────────────────────────────────────────
    total_rows = len(df)
    first_date = str(df.index[0].date()) if total_rows > 0 else "N/A"
    last_date = str(df.index[-1].date()) if total_rows > 0 else "N/A"

    # ── Missing values ────────────────────────────────────────────────────────
    missing_values = {col: int(df[col].isna().sum()) for col in [OPEN, HIGH, LOW, CLOSE, VOLUME] if col in df.columns}
    for col, count in missing_values.items():
        if count > 0:
            warnings.append(f"Column '{col}' has {count} missing values")

    # ── Duplicate dates ───────────────────────────────────────────────────────
    duplicate_dates = int(df.index.duplicated().sum())
    if duplicate_dates > 0:
        errors.append(f"{duplicate_dates} duplicate timestamps found")

    # ── Index sorted ─────────────────────────────────────────────────────────
    if not df.index.is_monotonic_increasing:
        errors.append("Index is not chronologically sorted")

    # ── Non-positive prices ───────────────────────────────────────────────────
    for col in [OPEN, HIGH, LOW, CLOSE]:
        if col not in df.columns:
            continue
        nonpos = (df[col] <= 0).sum()
        if nonpos > 0:
            errors.append(f"Column '{col}' has {nonpos} non-positive values")

    # ── High < Low ────────────────────────────────────────────────────────────
    if HIGH in df.columns and LOW in df.columns:
        hl_bad = (df[HIGH] < df[LOW]).sum()
        if hl_bad > 0:
            errors.append(f"{hl_bad} rows where High < Low")

    # ── Open/Close outside High-Low ───────────────────────────────────────────
    tol = 1e-4
    for col in [OPEN, CLOSE]:
        if col in df.columns and HIGH in df.columns and LOW in df.columns:
            bad = ((df[col] > df[HIGH] + tol) | (df[col] < df[LOW] - tol)).sum()
            if bad > 0:
                warnings.append(f"{bad} rows where {col} is outside [Low, High] (±tol)")

    # ── Weekend observations ──────────────────────────────────────────────────
    if hasattr(df.index, "dayofweek"):
        weekends = (df.index.dayofweek >= 5).sum()
        if weekends > 0:
            warnings.append(f"{weekends} weekend observations found")

    # ── Date gaps ─────────────────────────────────────────────────────────────
    suspected_gaps: List[str] = []
    if total_rows > 1:
        diffs = pd.Series(df.index).diff().dt.days.iloc[1:]
        gap_mask = diffs > MAX_CALENDAR_GAP_DAYS
        gap_starts = df.index[1:][gap_mask.values]
        gap_ends = df.index[:-1][gap_mask.values]
        for start, end in zip(gap_ends, gap_starts):
            days = (start.date() - end.date()).days
            entry = f"{end.date()} → {start.date()} ({days} calendar days)"
            suspected_gaps.append(entry)
            if days > 30:
                warnings.append(f"Large gap detected: {entry}")

    # ── Zero volume ───────────────────────────────────────────────────────────
    zero_volume_dates: List[str] = []
    if VOLUME in df.columns:
        zero_vol = df[VOLUME] == 0
        if zero_vol.any():
            zero_volume_dates = [str(d.date()) for d in df.index[zero_vol]]
            notes.append(
                f"{len(zero_volume_dates)} sessions with zero volume "
                "(expected for ^NSEI index — volume field not meaningful)"
            )

    # ── Extreme returns ───────────────────────────────────────────────────────
    extreme_returns: List[dict] = []
    if CLOSE in df.columns and total_rows > 1:
        ret = df[CLOSE].pct_change()
        extreme_mask = ret.abs() > 0.12
        for idx in df.index[extreme_mask]:
            extreme_returns.append({
                "date": str(idx.date()),
                "return_pct": round(float(ret.loc[idx]) * 100, 2),
                "close": float(df.loc[idx, CLOSE]),
            })
        if extreme_returns:
            warnings.append(
                f"{len(extreme_returns)} sessions with |return| > 12% — review these dates"
            )

    # ── Timezone consistency ──────────────────────────────────────────────────
    if hasattr(df.index, "tz") and df.index.tz is not None:
        notes.append(f"Index timezone: {df.index.tz} (expected tz-naive for daily data)")

    # ── Volume reliability note ───────────────────────────────────────────────
    notes.append(
        "^NSEI is an index — Yahoo Finance volume field is unreliable. "
        "Volume-based signals are disabled by default."
    )

    notes.append(
        "^NSEI cannot be traded directly. This dataset is a research proxy only. "
        "Reported costs do not represent real execution costs."
    )

    rows_removed = 0  # validator does not modify
    rows_flagged = len(extreme_returns) + len(zero_volume_dates)

    report = DataQualityReport(
        symbol=symbol,
        source=source,
        downloaded_at=downloaded_at,
        first_date=first_date,
        last_date=last_date,
        total_rows=total_rows,
        missing_values=missing_values,
        duplicate_dates=duplicate_dates,
        suspected_gaps=suspected_gaps,
        extreme_returns=extreme_returns,
        zero_volume_dates=zero_volume_dates,
        validation_warnings=warnings,
        confirmed_errors=errors,
        rows_removed=rows_removed,
        rows_flagged=rows_flagged,
        notes=notes,
    )

    _log_report(report)
    return report


def _log_report(r: DataQualityReport) -> None:
    logger.info(f"[bold]Data Quality Report[/bold] — {r.symbol}")
    logger.info(f"  Date range  : {r.first_date} → {r.last_date}")
    logger.info(f"  Rows        : {r.total_rows}")
    logger.info(f"  Duplicates  : {r.duplicate_dates}")
    logger.info(f"  Gaps > {MAX_CALENDAR_GAP_DAYS}d : {len(r.suspected_gaps)}")
    logger.info(f"  Extreme ret : {len(r.extreme_returns)}")
    if r.confirmed_errors:
        for e in r.confirmed_errors:
            logger.error(f"  ERROR: {e}")
    if r.validation_warnings:
        for w in r.validation_warnings:
            logger.warning(f"  WARN: {w}")
    for n in r.notes:
        logger.info(f"  NOTE: {n}")
