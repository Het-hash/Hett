#!/usr/bin/env python3
"""Download and cache NIFTY 50 data from Yahoo Finance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import yaml

from src.data import load_data
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_dirs

logger = get_logger(__name__)


@click.command()
@click.option("--config", default="config/base.yaml", help="Config file path")
@click.option("--force-refresh", is_flag=True, help="Force re-download")
@click.option("--symbol", default=None, help="Override symbol")
@click.option("--start", default=None, help="Override start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="Override end date (YYYY-MM-DD)")
def main(config, force_refresh, symbol, start, end):
    ensure_dirs()
    with open(config) as f:
        cfg = yaml.safe_load(f)

    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    sym = symbol or cfg["data"]["symbol"]
    start_date = start or cfg["data"].get("start_date")
    end_date = end or cfg["data"].get("end_date")

    logger.info(f"Downloading {sym}…")
    df, report = load_data(
        symbol=sym,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh or cfg["data"].get("force_refresh", False),
        retry_attempts=cfg["data"].get("retry_attempts", 5),
        retry_backoff=cfg["data"].get("retry_backoff_seconds", 2.0),
    )

    print(f"\n✓ Downloaded {len(df)} rows: {report.first_date} → {report.last_date}")
    if report.confirmed_errors:
        print(f"⚠ {len(report.confirmed_errors)} confirmed data errors — see quality report")
    if report.validation_warnings:
        print(f"ℹ {len(report.validation_warnings)} warnings")


if __name__ == "__main__":
    main()
