"""Data pipeline: download, cache, clean, validate."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.data.downloader import download_yfinance
from src.data.cleaner import clean_ohlcv
from src.data.validator import validate_ohlcv
from src.data.cache import cache_processed, load_processed
from src.data.schema import DataQualityReport
from src.utils.logging import get_logger
from src.utils.serialization import save_json
from src.utils.paths import data_metadata_dir

logger = get_logger(__name__)


def load_data(
    symbol: str = "^NSEI",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
    retry_attempts: int = 5,
    retry_backoff: float = 2.0,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Full pipeline: download → clean → validate → cache → return.

    Returns (processed_df, quality_report).
    On subsequent calls without force_refresh, returns cached processed data.
    """
    safe_sym = symbol.replace("^", "").replace("/", "_")

    if not force_refresh:
        cached = load_processed(symbol, interval)
        if cached is not None:
            meta_path = data_metadata_dir() / f"{safe_sym}_{interval}_meta.json"
            downloaded_at = "cached"
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                downloaded_at = meta.get("downloaded_at", "cached")
            report = validate_ohlcv(cached, symbol, "yfinance_cached", downloaded_at)
            return cached, report

    raw = download_yfinance(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        force_refresh=force_refresh,
    )

    downloaded_at = datetime.now(timezone.utc).isoformat()
    cleaned, decisions = clean_ohlcv(raw)
    report = validate_ohlcv(cleaned, symbol, "yfinance", downloaded_at)

    report_path = data_metadata_dir() / f"{safe_sym}_{interval}_quality_report.json"
    save_json(report.to_dict(), report_path)

    if decisions:
        decisions_path = data_metadata_dir() / f"{safe_sym}_{interval}_cleaning_decisions.json"
        save_json([d.to_dict() for d in decisions], decisions_path)

    cache_processed(cleaned, symbol, interval)

    logger.info(
        f"Pipeline complete: {report.first_date} → {report.last_date} "
        f"({report.total_rows} rows, {len(report.confirmed_errors)} errors, "
        f"{len(report.validation_warnings)} warnings)"
    )

    return cleaned, report
