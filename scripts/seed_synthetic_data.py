#!/usr/bin/env python3
"""
Seed the cache with synthetic NIFTY-like data for offline/CI testing.

This is NOT real market data. It generates a GBM price series calibrated
roughly to NIFTY 50 historical parameters (mu≈12% pa, sigma≈18% pa).
Do NOT use for any real-money decision. For development and CI only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import timezone

from src.data.schema import enforce_schema, TIMESTAMP, OPEN, HIGH, LOW, CLOSE, VOLUME, ADJ_CLOSE
from src.data.cache import cache_processed
from src.data.validator import validate_ohlcv
from src.utils.paths import ensure_dirs, data_raw_dir, data_metadata_dir
from src.utils.serialization import save_parquet, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_synthetic_nifty(
    start: str = "1990-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range(start, end, freq="B")
    n = len(dates)

    # GBM: NIFTY 50 approximate params
    mu_daily = 0.00047   # ~12% annual
    sigma_daily = 0.0113  # ~18% annual

    log_returns = np.random.normal(mu_daily - 0.5 * sigma_daily**2, sigma_daily, n)
    close = 1000 * np.exp(np.cumsum(log_returns))

    # OHLCV construction
    intraday_range = np.abs(np.random.normal(0, 0.008, n))
    high = close * (1 + intraday_range)
    low = close * (1 - intraday_range)
    open_ = close * np.exp(np.random.normal(0, 0.004, n))
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame({
        OPEN: open_,
        HIGH: high,
        LOW: low,
        CLOSE: close,
        ADJ_CLOSE: close,
        VOLUME: np.zeros(n),  # index volume is unreliable — set to 0
    }, index=dates)
    df.index.name = TIMESTAMP
    df["symbol"] = "^NSEI"
    df["source"] = "synthetic_gbm"
    return df


if __name__ == "__main__":
    ensure_dirs()

    # Guard: refuse to overwrite real data with synthetic data
    from src.utils.paths import data_metadata_dir
    from src.utils.serialization import load_json
    meta_path = data_metadata_dir() / "NSEI_1d_meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        src = meta.get("source", "")
        if "synthetic" not in src.lower() and "gbm" not in src.lower():
            print(
                "\n[REFUSED] Real NIFTY data already exists in cache.\n"
                "synthetic seeder will not overwrite it.\n"
                "Source in cache: " + src
            )
            sys.exit(0)

    print("\n[WARNING] Generating SYNTHETIC GBM data for unit-test fixtures ONLY.")
    print("         This data must NEVER be used for strategy research.")
    print("         It will be stored with source='synthetic_gbm' to trigger")
    print("         the pipeline guard in run_phase1.py.\n")
    logger.info("Generating synthetic NIFTY data (GBM, NOT real data)…")
    df = generate_synthetic_nifty()
    logger.info(f"Generated {len(df)} rows: {df.index[0].date()} → {df.index[-1].date()}")

    symbol = "^NSEI"
    safe_sym = "NSEI"

    # Save as raw
    raw_path = data_raw_dir() / f"{safe_sym}_1d_raw.parquet"
    save_parquet(df, raw_path)

    # Save metadata
    from datetime import datetime
    meta = {
        "symbol": symbol,
        "source": "synthetic_gbm",
        "interval": "1d",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "rows": len(df),
        "note": "SYNTHETIC GBM DATA — NOT REAL MARKET DATA",
    }
    save_json(meta, data_metadata_dir() / f"{safe_sym}_1d_meta.json")

    # Cache as processed
    cache_processed(df, symbol, "1d")

    # Validate
    report = validate_ohlcv(df, symbol, "synthetic_gbm", meta["downloaded_at"])
    save_json(report.to_dict(), data_metadata_dir() / f"{safe_sym}_1d_quality_report.json")

    print(f"\n✓ Synthetic data seeded: {report.first_date} → {report.last_date} ({report.total_rows} rows)")
    print("  NOTE: This is synthetic GBM data, not real NIFTY 50 data.")
    print("  Pipeline results on this data are for system validation only.")
