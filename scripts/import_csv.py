#!/usr/bin/env python3
"""
Import real NIFTY 50 daily OHLC data from a local CSV file.

Supported CSV formats:
  - NSE historical data (Date,Open,High,Low,Close,Shares Traded,Turnover)
  - Yahoo Finance export (Date,Open,High,Low,Close,Adj Close,Volume)
  - Generic (Date,Open,High,Low,Close[,Volume])

Usage:
    python scripts/import_csv.py --file NIFTY_50_DAILY.csv --symbol "^NSEI"

The script:
  1. Auto-detects column layout.
  2. Validates OHLC consistency and date range.
  3. Flags volume as unreliable for index data.
  4. Saves to data/raw/ and data/processed/ exactly as the downloader would.
  5. Writes a quality report to data/metadata/.
  6. NEVER falls back to synthetic data — raises an explicit error if the
     file is missing or invalid.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.data.schema import (
    OPEN, HIGH, LOW, CLOSE, VOLUME, ADJ_CLOSE, SYMBOL, SOURCE, TIMESTAMP,
    enforce_schema,
)
from src.data.cleaner import clean_ohlcv
from src.data.validator import validate_ohlcv
from src.data.cache import cache_processed
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_dirs, data_raw_dir, data_metadata_dir
from src.utils.serialization import save_parquet, save_json

logger = get_logger(__name__)

# ── Column-name aliases from known providers ──────────────────────────────────
_ALIASES = {
    # Date variants
    "date": TIMESTAMP, "Date": TIMESTAMP, "DATE": TIMESTAMP,
    "datetime": TIMESTAMP, "Datetime": TIMESTAMP,
    # Open
    "open": OPEN, "Open": OPEN, "OPEN": OPEN,
    # High
    "high": HIGH, "High": HIGH, "HIGH": HIGH,
    # Low
    "low": LOW, "Low": LOW, "LOW": LOW,
    # Close
    "close": CLOSE, "Close": CLOSE, "CLOSE": CLOSE,
    # Adj Close (Yahoo Finance)
    "adj close": ADJ_CLOSE, "Adj Close": ADJ_CLOSE, "adj_close": ADJ_CLOSE,
    # Volume — NSE calls it "Shares Traded" or "Volume"
    "volume": VOLUME, "Volume": VOLUME, "VOLUME": VOLUME,
    "shares traded": VOLUME, "Shares Traded": VOLUME,
    "turnover": "turnover",  # NSE turnover (crores) — not volume, kept as info
}


def _detect_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using alias map."""
    rename = {}
    for col in df.columns:
        canon = _ALIASES.get(col) or _ALIASES.get(col.strip())
        if canon and canon != col:
            rename[col] = canon
    df = df.rename(columns=rename)
    return df


def _parse_date_column(series: pd.Series) -> pd.DatetimeIndex:
    """
    Try multiple date formats. Raise on failure.
    NSE uses DD-MMM-YYYY (e.g. 01-Jan-2024).
    Yahoo uses YYYY-MM-DD.
    """
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            parsed = pd.to_datetime(series, format=fmt)
            logger.info(f"Date column parsed with format '{fmt}'")
            return pd.DatetimeIndex(parsed).normalize()
        except Exception:
            continue
    # Last resort — let pandas infer
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True)
        logger.warning("Date format inferred by pandas — verify dates are correct")
        return pd.DatetimeIndex(parsed).normalize()
    except Exception as e:
        raise ValueError(f"Cannot parse date column: {e}") from e


def load_csv_file(path: Path, symbol: str) -> pd.DataFrame:
    """
    Load, clean-up and validate a NIFTY CSV file.
    Raises ValueError with a clear message on any structural problem.
    Does NOT fall back to synthetic data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\n\nCSV file not found: {path}\n\n"
            "Please provide real NIFTY 50 daily OHLC data. Sources:\n"
            "  • NSE India: https://www.nseindia.com/market-data/equity-stockIndices-data\n"
            "    → Select 'NIFTY 50', set date range 1995–today, click 'Download'\n"
            "  • Yahoo Finance: https://finance.yahoo.com/quote/%5ENSEI/history/\n"
            "    → Download CSV\n"
            "  • Stooq: https://stooq.com/q/d/l/?s=%5Ensei&i=d\n\n"
            "Then run:\n"
            f"    python scripts/import_csv.py --file <path_to_csv> --symbol {symbol}\n"
        )

    logger.info(f"Reading CSV: {path} ({path.stat().st_size / 1024:.1f} KB)")

    # Try reading with and without thousands separator (NSE uses commas in numbers)
    for thousands in (",", None):
        try:
            df = pd.read_csv(path, thousands=thousands)
            break
        except Exception as e:
            if thousands is None:
                raise ValueError(f"Cannot read CSV file: {e}") from e

    logger.info(f"Raw CSV shape: {df.shape}, columns: {list(df.columns)}")

    # ── Rename columns ────────────────────────────────────────────────────────
    df = _detect_and_rename(df)

    # ── Find and parse date column ────────────────────────────────────────────
    if TIMESTAMP in df.columns:
        date_series = df[TIMESTAMP]
    else:
        # Fallback: first column is often the date
        date_series = df.iloc[:, 0]
        logger.warning(f"No date column found by name — using first column: '{df.columns[0]}'")

    dates = _parse_date_column(date_series)
    df.index = dates
    df.index.name = TIMESTAMP

    # ── Validate required columns ─────────────────────────────────────────────
    for col in [OPEN, HIGH, LOW, CLOSE]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in CSV.\n"
                f"Detected columns: {list(df.columns)}\n"
                "Expected: Date, Open, High, Low, Close (and optionally Volume)."
            )

    # ── Numeric conversion ────────────────────────────────────────────────────
    for col in [OPEN, HIGH, LOW, CLOSE]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Volume handling ───────────────────────────────────────────────────────
    if VOLUME not in df.columns:
        logger.warning("No Volume column found — adding zeros (index volume is unreliable)")
        df[VOLUME] = 0.0
    else:
        df[VOLUME] = pd.to_numeric(df[VOLUME], errors="coerce").fillna(0)

    # ── Drop non-OHLCV columns ────────────────────────────────────────────────
    keep = [c for c in [OPEN, HIGH, LOW, CLOSE, VOLUME, ADJ_CLOSE] if c in df.columns]
    df = df[keep]

    # ── Add metadata columns ──────────────────────────────────────────────────
    df[SYMBOL] = symbol
    df[SOURCE] = f"csv:{path.name}"

    # ── Sort chronologically ──────────────────────────────────────────────────
    df = df.sort_index()

    # ── Drop weekends ─────────────────────────────────────────────────────────
    n_before = len(df)
    df = df[df.index.dayofweek < 5]
    if len(df) < n_before:
        logger.warning(f"Dropped {n_before - len(df)} weekend rows")

    logger.info(
        f"Parsed {len(df)} trading sessions: "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} rows after parsing — this does not look like multi-year OHLC data.\n"
            "Please check that the file contains daily NIFTY 50 data from at least 1995."
        )

    return df


@click.command()
@click.option("--file", "csv_path", required=True,
              type=click.Path(exists=False), help="Path to NIFTY 50 daily CSV file")
@click.option("--symbol", default="^NSEI", help="Canonical symbol to use (default: ^NSEI)")
@click.option("--start", default=None, help="Trim to on/after this date (YYYY-MM-DD)")
@click.option("--end", default=None, help="Trim to on/before this date (YYYY-MM-DD)")
def main(csv_path, symbol, start, end):
    """
    Import real NIFTY 50 daily OHLC data from a CSV file.

    This is the ONLY supported path when online downloads are blocked.
    Synthetic GBM data is NOT accepted by this script.
    """
    configure_logging("INFO")
    ensure_dirs()

    path = Path(csv_path)
    df = load_csv_file(path, symbol)

    # ── Optional date trim ────────────────────────────────────────────────────
    if start:
        df = df[df.index >= pd.Timestamp(start)]
        logger.info(f"Trimmed to start={start}: {len(df)} rows remaining")
    if end:
        df = df[df.index <= pd.Timestamp(end)]
        logger.info(f"Trimmed to end={end}: {len(df)} rows remaining")

    if len(df) == 0:
        raise ValueError("No rows remain after date filtering.")

    # ── Clean ─────────────────────────────────────────────────────────────────
    cleaned, decisions = clean_ohlcv(df)
    logger.info(f"Cleaning: {len(decisions)} decisions/flags")

    # ── Validate ──────────────────────────────────────────────────────────────
    downloaded_at = datetime.now(timezone.utc).isoformat()
    report = validate_ohlcv(cleaned, symbol, f"csv:{path.name}", downloaded_at)

    if report.confirmed_errors:
        print("\n⚠ CONFIRMED DATA ERRORS:")
        for e in report.confirmed_errors:
            print(f"  [ERROR] {e}")
        print("\nFix the errors in the CSV before proceeding.")
        sys.exit(1)

    # ── Persist ───────────────────────────────────────────────────────────────
    safe_sym = symbol.replace("^", "").replace("/", "_")

    # Raw copy
    raw_path = data_raw_dir() / f"{safe_sym}_1d_raw.parquet"
    save_parquet(cleaned, raw_path)
    logger.info(f"Raw data saved: {raw_path}")

    # Metadata
    meta = {
        "symbol": symbol,
        "source": f"csv:{path.name}",
        "source_file": str(path.resolve()),
        "interval": "1d",
        "downloaded_at": downloaded_at,
        "first_date": report.first_date,
        "last_date": report.last_date,
        "rows": report.total_rows,
        "note": "Imported from local CSV — real market data",
    }
    save_json(meta, data_metadata_dir() / f"{safe_sym}_1d_meta.json")

    # Quality report
    save_json(report.to_dict(), data_metadata_dir() / f"{safe_sym}_1d_quality_report.json")
    if decisions:
        save_json(
            [d.to_dict() for d in decisions],
            data_metadata_dir() / f"{safe_sym}_1d_cleaning_decisions.json",
        )

    # Processed cache (this is what the pipeline reads)
    cache_processed(cleaned, symbol, "1d")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CSV IMPORT COMPLETE — REAL MARKET DATA")
    print("=" * 60)
    print(f"  Symbol     : {symbol}")
    print(f"  Source     : {path.name}")
    print(f"  Date range : {report.first_date} → {report.last_date}")
    print(f"  Rows       : {report.total_rows:,}")
    print(f"  Errors     : {len(report.confirmed_errors)}")
    print(f"  Warnings   : {len(report.validation_warnings)}")
    if report.validation_warnings:
        for w in report.validation_warnings[:5]:
            print(f"             ⚠ {w}")
    print(f"\n  NOTE: Index volume is unreliable for ^NSEI.")
    print(f"        Volume-based signals are disabled by default.")
    print(f"\n  Next step:")
    print(f"    python scripts/run_phase1.py --config config/base.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
