#!/usr/bin/env python3
"""Generate the Phase 1 HTML report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import yaml

from src.data import load_data
from src.backtesting.baselines import run_all_baselines
from src.backtesting.metrics import compute_metrics
from src.backtesting.costs import ETF_CONSERVATIVE
from src.reports.report_builder import ReportBuilder
from src.reports.charts import (
    plot_equity_curves, plot_drawdown, plot_monthly_heatmap,
    plot_return_distribution, plot_rolling_sharpe,
)
from src.reports.tables import metrics_table
from src.experiments.registry import ExperimentRegistry
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_dirs
from src.utils.serialization import load_json

logger = get_logger(__name__)


@click.command()
@click.option("--config", default="config/base.yaml")
def main(config):
    ensure_dirs()
    with open(config) as f:
        cfg = yaml.safe_load(f)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    df, quality_report = load_data(symbol=cfg["data"]["symbol"])
    initial_capital = cfg["capital"]["initial"]

    # Run baselines
    baselines = run_all_baselines(df, ETF_CONSERVATIVE, initial_capital)
    metrics_all = {}
    equity_curves = {}
    for name, result in baselines.items():
        m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
        metrics_all[name] = m
        equity_curves[name] = result.equity_curve

    # Generate charts
    eq_path = plot_equity_curves(equity_curves, "Baseline Equity Curves")
    bah_dd = plot_drawdown(equity_curves["buy_and_hold"], "buy_and_hold")
    month_path = plot_monthly_heatmap(equity_curves["buy_and_hold"])
    ret_dist = plot_return_distribution(baselines["buy_and_hold"].returns, "buy_and_hold")
    rs_path = plot_rolling_sharpe(baselines["buy_and_hold"].returns, label="buy_and_hold")

    # Load experiment results from registry
    registry = ExperimentRegistry()
    exp_records = registry.all_records()
    all_exp_results = registry._records if registry._records else []

    # Build report
    builder = ReportBuilder()

    # Executive summary
    bah_m = metrics_all.get("buy_and_hold", {})
    builder.add_section("Executive Summary", f"""
    <p>Phase 1 edge discovery for NIFTY 50 (research proxy: ^NSEI) is complete.</p>
    <p><strong>Data coverage:</strong> {quality_report.first_date} → {quality_report.last_date}
    ({quality_report.total_rows} trading sessions)</p>
    <p><strong>Buy &amp; hold CAGR:</strong> {bah_m.get('cagr',0)*100:.2f}% |
       <strong>Sharpe:</strong> {bah_m.get('sharpe_ratio',0):.2f} |
       <strong>Max DD:</strong> {bah_m.get('max_drawdown',0)*100:.2f}%</p>
    <div class='info'>
    <strong>IMPORTANT:</strong> ^NSEI is a research proxy. These are not executable returns.
    All cost assumptions are indicative only.
    </div>
    """)

    builder.add_data_quality(quality_report.to_dict())
    builder.add_metrics_table(metrics_all, "Baseline Strategy Performance")

    # Charts
    charts_html = ""
    for path, caption in [
        (eq_path, "Baseline equity curves (normalised)"),
        (bah_dd, "Buy & Hold drawdown"),
        (month_path, "Buy & Hold monthly return heatmap"),
        (ret_dist, "Buy & Hold return distribution"),
        (rs_path, "Buy & Hold rolling Sharpe (252d)"),
    ]:
        charts_html += builder.add_image(path, caption)
    builder.add_section("Charts", charts_html)

    # Experiment results
    builder.add_candidate_results(all_exp_results)

    # Limitations
    builder.add_limitations()

    # Go/No-Go decision
    accepted = [r for r in all_exp_results if r.get("status") == "accepted"]
    if accepted:
        decision = "CONDITIONAL GO"
        rationale = (
            f"{len(accepted)} candidate(s) passed validation criteria. "
            "They must pass the untouched test period and walk-forward evaluation before proceeding. "
            "Confirmation required before any real-money application."
        )
    else:
        decision = "NO-GO"
        rationale = (
            "No candidate passed all validation criteria on the held-out validation period. "
            "Finding no robust edge is a valid and honest Phase 1 outcome. "
            "Recommended next steps: expand hypothesis space, obtain better data sources, "
            "or review whether the proxy (^NSEI) adequately represents a tradable instrument."
        )

    builder.add_go_no_go(decision, rationale)

    # Build HTML
    data_coverage = f"{quality_report.first_date} → {quality_report.last_date} ({quality_report.total_rows} rows)"
    report_path = builder.build(data_coverage=data_coverage)

    print(f"\n✓ Report written to: {report_path}")
    print(f"  Decision: {decision}")
    print(f"  Rationale: {rationale}")


if __name__ == "__main__":
    main()
