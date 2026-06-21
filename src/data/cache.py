"""Processed-data caching layer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import data_processed_dir
from src.utils.serialization import save_parquet, load_parquet

logger = get_logger(__name__)

_PROCESSED_FILENAME = "{safe_sym}_{interval}_processed.parquet"


def _processed_path(symbol: str, interval: str, base: Optional[Path] = None) -> Path:
    base = base or data_processed_dir()
    safe_sym = symbol.replace("^", "").replace("/", "_")
    return base / _PROCESSED_FILENAME.format(safe_sym=safe_sym, interval=interval)


def cache_processed(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "1d",
    base: Optional[Path] = None,
) -> Path:
    path = _processed_path(symbol, interval, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(df, path)
    logger.info(f"Processed data cached to {path}")
    return path


def load_processed(
    symbol: str,
    interval: str = "1d",
    base: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    path = _processed_path(symbol, interval, base)
    if not path.exists():
        return None
    df = load_parquet(path)
    logger.info(f"Loaded processed data from {path} ({len(df)} rows)")
    return df


def processed_exists(symbol: str, interval: str = "1d", base: Optional[Path] = None) -> bool:
    return _processed_path(symbol, interval, base).exists()
