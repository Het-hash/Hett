#!/usr/bin/env python3
"""Validate cached data and print quality report."""
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
@click.option("--config", default="config/base.yaml")
def main(config):
    ensure_dirs()
    with open(config) as f:
        cfg = yaml.safe_load(f)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    df, report = load_data(symbol=cfg["data"]["symbol"])

    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    print(f"Symbol      : {report.symbol}")
    print(f"Source      : {report.source}")
    print(f"Date range  : {report.first_date} → {report.last_date}")
    print(f"Total rows  : {report.total_rows}")
    print(f"Duplicates  : {report.duplicate_dates}")
    print(f"Gaps > 10d  : {len(report.suspected_gaps)}")
    print(f"Extreme ret : {len(report.extreme_returns)}")
    print(f"Zero volume : {len(report.zero_volume_dates)}")

    if report.confirmed_errors:
        print("\nCONFIRMED ERRORS:")
        for e in report.confirmed_errors:
            print(f"  [ERROR] {e}")

    if report.validation_warnings:
        print("\nWARNINGS:")
        for w in report.validation_warnings:
            print(f"  [WARN] {w}")

    print("\nNOTES:")
    for n in report.notes:
        print(f"  [NOTE] {n}")

    if report.suspected_gaps:
        print("\nSUSPECTED GAPS:")
        for g in report.suspected_gaps[:10]:
            print(f"  {g}")

    if report.extreme_returns:
        print("\nEXTREME RETURNS:")
        for r in report.extreme_returns[:10]:
            print(f"  {r['date']}: {r['return_pct']:.1f}% (close={r['close']:.1f})")


if __name__ == "__main__":
    main()
