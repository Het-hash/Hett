"""Yahoo Finance downloader with retry logic and metadata recording."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.data.schema import enforce_schema, TIMESTAMP
from src.utils.logging import get_logger
from src.utils.paths import data_raw_dir, data_metadata_dir
from src.utils.serialization import save_parquet, save_json

logger = get_logger(__name__)


class DownloadError(RuntimeError):
    pass


def download_yfinance(
    symbol: str = "^NSEI",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    retry_attempts: int = 5,
    retry_backoff: float = 2.0,
    force_refresh: bool = False,
    raw_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance with retry logic.

    Returns a DataFrame with canonical column names and DatetimeIndex named 'timestamp'.
    Raw data is persisted to data/raw/ before any cleaning.
    """
    raw_dir = raw_dir or data_raw_dir()
    metadata_dir = metadata_dir or data_metadata_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    safe_sym = symbol.replace("^", "").replace("/", "_")
    raw_path = raw_dir / f"{safe_sym}_{interval}_raw.parquet"
    meta_path = metadata_dir / f"{safe_sym}_{interval}_meta.json"

    if raw_path.exists() and not force_refresh:
        logger.info(f"Loading cached raw data from {raw_path}")
        df = pd.read_parquet(raw_path)
        logger.info(f"Cached data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} rows)")
        return df

    logger.info(f"Downloading {symbol} from Yahoo Finance (interval={interval})")
    # Note: 'progress' kwarg removed — not supported in yfinance >= 1.0
    kwargs: dict = {"interval": interval, "auto_adjust": False}
    if start_date:
        kwargs["start"] = start_date
    else:
        kwargs["period"] = "max"
    if end_date:
        kwargs["end"] = end_date

    df = _download_with_retry(symbol, kwargs, retry_attempts, retry_backoff)

    if df is None or df.empty:
        raise DownloadError(
            f"Download returned empty data for {symbol}. "
            "Check your internet connection and that the symbol is valid."
        )

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Enforce timezone-naive UTC dates for daily data
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = TIMESTAMP

    df = enforce_schema(df, symbol=symbol, source="yfinance")

    # Persist raw (schema-renamed) data untouched
    save_parquet(df, raw_path)
    logger.info(f"Raw data saved: {raw_path}")

    downloaded_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "symbol": symbol,
        "source": "yfinance",
        "interval": interval,
        "downloaded_at": downloaded_at,
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "rows": len(df),
        "start_requested": start_date,
        "end_requested": end_date,
    }
    save_json(meta, meta_path)
    logger.info(
        f"Downloaded {len(df)} rows: "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


def _download_with_retry(
    symbol: str,
    kwargs: dict,
    attempts: int,
    backoff: float,
) -> Optional[pd.DataFrame]:
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(**kwargs)
            if df is not None and not df.empty:
                return df
            logger.warning(f"Attempt {attempt}/{attempts}: empty result, retrying…")
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Attempt {attempt}/{attempts} failed: {exc}")
        if attempt < attempts:
            sleep_time = backoff * (2 ** (attempt - 1))
            logger.info(f"Retrying in {sleep_time:.0f}s…")
            time.sleep(sleep_time)
    if last_exc:
        raise DownloadError(f"All {attempts} download attempts failed: {last_exc}") from last_exc
    return None
