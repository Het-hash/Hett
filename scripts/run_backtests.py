#!/usr/bin/env python3
"""Run baseline backtests and print performance metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import yaml

from src.data import load_data
from src.backtesting.baselines import run_all_baselines
from src.backtesting.metrics import compute_metrics
from src.backtesting.costs import ETF_CONSERVATIVE
from src.reports.tables import metrics_table, print_table
from src.reports.charts import plot_equity_curves, plot_drawdown, plot_monthly_heatmap
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_dirs
from src.utils.serialization import save_json

logger = get_logger(__name__)


@click.command()
@click.option("--config", default="config/base.yaml")
def main(config):
    ensure_dirs()
    with open(config) as f:
        cfg = yaml.safe_load(f)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    df, _ = load_data(symbol=cfg["data"]["symbol"])
    initial_capital = cfg["capital"]["initial"]

    baselines = run_all_baselines(df, ETF_CONSERVATIVE, initial_capital)

    metrics_all = {}
    equity_curves = {}
    for name, result in baselines.items():
        m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
        metrics_all[name] = m
        equity_curves[name] = result.equity_curve

    tbl = metrics_table(metrics_all)
    print_table(tbl, "BASELINE STRATEGY PERFORMANCE")

    save_json(metrics_all, Path("outputs/tables/baseline_metrics.json"))

    # Charts
    plot_equity_curves(equity_curves)
    plot_drawdown(equity_curves["buy_and_hold"], "buy_and_hold")
    plot_monthly_heatmap(equity_curves["buy_and_hold"], "Buy & Hold Monthly Returns")

    print("\nCharts saved to outputs/figures/")
    print("Metrics saved to outputs/tables/baseline_metrics.json")


if __name__ == "__main__":
    main()
