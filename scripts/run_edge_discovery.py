#!/usr/bin/env python3
"""Run edge discovery: compute features, regimes, and forward return analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import yaml
import pandas as pd

from src.data import load_data
from src.features.registry import build_all_features
from src.regimes.combined_regime import RegimeEngine
from src.edge_discovery.forward_returns import compute_forward_returns, forward_return_stats
from src.edge_discovery.statistics import unconditional_stats
from src.experiments.registry import ExperimentRegistry, _data_hash
from src.experiments.runner import run_candidate_experiments
from src.validation.splits import make_splits
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_dirs
from src.utils.serialization import save_json

logger = get_logger(__name__)


@click.command()
@click.option("--config", default="config/base.yaml")
@click.option("--experiments", default="config/experiments.yaml")
def main(config, experiments):
    ensure_dirs()
    with open(config) as f:
        cfg = yaml.safe_load(f)
    with open(experiments) as f:
        exp_cfg = yaml.safe_load(f)

    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    # Load data
    df, quality_report = load_data(symbol=cfg["data"]["symbol"])
    logger.info(f"Data: {quality_report.first_date} → {quality_report.last_date} ({quality_report.total_rows} rows)")

    # Splits
    sp = cfg["splits"]
    splits = make_splits(df, sp["development"], sp["validation"], sp["test"])

    # Features on dev only for regime fitting
    volume_reliable = cfg["data"].get("nsei_volume_reliable", False)

    # Fit regimes on dev, classify all
    regime_engine = RegimeEngine(cfg.get("regimes", {}))
    regime_engine.fit(splits.development)
    regimes_dev = regime_engine.classify(splits.development)
    regimes_val = regime_engine.classify(splits.validation)
    regime_report = regime_engine.report(regimes_dev)
    logger.info(f"Regime distribution (dev): {regime_report}")

    # Collect all experiment configs
    all_experiments = []
    for family in ["momentum", "mean_reversion", "breakout", "trend", "gap", "calendar"]:
        all_experiments.extend(exp_cfg.get(family, []))

    logger.info(f"Total hypothesis configurations: {len(all_experiments)}")

    # Run experiments
    registry = ExperimentRegistry()
    results = run_candidate_experiments(
        dev_df=splits.development,
        val_df=splits.validation,
        regimes_dev=regimes_dev,
        regimes_val=regimes_val,
        experiment_configs=all_experiments,
        registry=registry,
        seed=cfg["project"]["random_seed"],
    )

    # Summarise
    accepted = [r for r in results if r.get("status") == "accepted"]
    rejected = [r for r in results if r.get("status") == "rejected"]
    failed = [r for r in results if r.get("status") == "failed"]

    print(f"\n{'='*60}")
    print(f"EDGE DISCOVERY RESULTS")
    print(f"{'='*60}")
    print(f"Total experiments : {len(results)}")
    print(f"Accepted (val OOS): {len(accepted)}")
    print(f"Rejected          : {len(rejected)}")
    print(f"Failed (error)    : {len(failed)}")

    if accepted:
        print("\nACCEPTED CANDIDATES (validation period):")
        for r in accepted:
            sv = r.get("stats_val", {})
            print(f"  {r['experiment_id']}: EV={sv.get('mean',0)*100:.3f}% "
                  f"HR={sv.get('hit_rate',0):.1%} N={sv.get('n',0)}")
    else:
        print("\nNO candidates passed acceptance criteria on validation period.")
        print("This is a valid Phase 1 outcome — no evidence of a robust edge was found.")

    if rejected:
        print("\nREJECTED (first 10):")
        for r in rejected[:10]:
            print(f"  {r['experiment_id']}: {r.get('rejection_reason','')}")

    save_json(
        {"accepted": [r["experiment_id"] for r in accepted],
         "rejected": len(rejected), "failed": len(failed)},
        Path("outputs/experiments/edge_discovery_summary.json"),
    )


if __name__ == "__main__":
    main()
