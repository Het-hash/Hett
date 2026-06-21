#!/usr/bin/env python3
"""
Master Phase 1 pipeline.

Runs end-to-end:
  1. Download + validate data
  2. Feature engineering
  3. Regime classification
  4. Baseline backtests
  5. Candidate edge discovery
  6. Walk-forward validation (on accepted candidates)
  7. Stress tests
  8. Report generation

Usage:
    python scripts/run_phase1.py --config config/base.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import click
import yaml
import pandas as pd

from src.data import load_data
from src.features.registry import build_all_features
from src.regimes.combined_regime import RegimeEngine
from src.edge_discovery.forward_returns import compute_forward_returns
from src.edge_discovery.statistics import unconditional_stats
from src.backtesting.baselines import run_all_baselines
from src.backtesting.metrics import compute_metrics
from src.backtesting.costs import ETF_CONSERVATIVE
from src.validation.splits import make_splits
from src.validation.walk_forward import walk_forward
from src.validation.stress_tests import run_stress_suite
from src.experiments.registry import ExperimentRegistry
from src.experiments.runner import run_candidate_experiments, _build_signal
from src.reports.report_builder import ReportBuilder
from src.reports.charts import (
    plot_equity_curves, plot_drawdown, plot_monthly_heatmap,
    plot_return_distribution, plot_rolling_sharpe, plot_cost_sensitivity,
)
from src.reports.tables import metrics_table, print_table, candidate_ranking_table
from src.edge_discovery.ranking import rank_candidates
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_dirs
from src.utils.random_state import seed_everything
from src.utils.serialization import save_json

logger = get_logger(__name__)


@click.command()
@click.option("--config", default="config/base.yaml", help="Base configuration file")
@click.option("--experiments", default="config/experiments.yaml", help="Experiments config")
@click.option("--force-refresh", is_flag=True, help="Force data re-download")
@click.option("--skip-wf", is_flag=True, help="Skip walk-forward (faster run)")
def main(config, experiments, force_refresh, skip_wf):
    start_time = time.time()
    ensure_dirs()

    with open(config) as f:
        cfg = yaml.safe_load(f)
    with open(experiments) as f:
        exp_cfg = yaml.safe_load(f)

    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = cfg["project"]["random_seed"]
    seed_everything(seed)

    print("\n" + "="*70)
    print("  HETT PHASE 1 PIPELINE — NIFTY 50 EDGE DISCOVERY")
    print("="*70)

    # ── STEP 1: Data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading and validating data…")
    df, quality_report = load_data(
        symbol=cfg["data"]["symbol"],
        start_date=cfg["data"].get("start_date"),
        end_date=cfg["data"].get("end_date"),
        force_refresh=force_refresh or cfg["data"].get("force_refresh", False),
    )
    print(f"      ✓ {quality_report.total_rows} rows: {quality_report.first_date} → {quality_report.last_date}")
    if quality_report.confirmed_errors:
        print(f"      ⚠ {len(quality_report.confirmed_errors)} confirmed data errors")

    # ── STEP 2: Splits ────────────────────────────────────────────────────────
    print("\n[2/7] Creating chronological splits…")
    sp = cfg["splits"]
    splits = make_splits(df, sp["development"], sp["validation"], sp["test"])
    print(f"      Dev : {splits.development.index[0].date()} → {splits.development.index[-1].date()} ({len(splits.development)} rows)")
    print(f"      Val : {splits.validation.index[0].date()} → {splits.validation.index[-1].date()} ({len(splits.validation)} rows)")
    print(f"      Test: {splits.test.index[0].date()} → {splits.test.index[-1].date()} ({len(splits.test)} rows)")
    print("      *** TEST PERIOD IS LOCKED AND WILL NOT BE USED FOR SELECTION ***")

    # ── STEP 3: Regimes ───────────────────────────────────────────────────────
    print("\n[3/7] Fitting regime classifiers on development data…")
    regime_engine = RegimeEngine(cfg.get("regimes", {}))
    regime_engine.fit(splits.development)
    regimes_dev = regime_engine.classify(splits.development)
    regimes_val = regime_engine.classify(splits.validation)
    regime_report = regime_engine.report(regimes_dev)
    print(f"      Trend (dev):      {regime_report.get('trend', {})}")
    print(f"      Volatility (dev): {regime_report.get('volatility', {})}")

    # ── STEP 4: Baselines ─────────────────────────────────────────────────────
    print("\n[4/7] Running baseline strategies…")
    initial_capital = cfg["capital"]["initial"]
    baselines = run_all_baselines(df, ETF_CONSERVATIVE, initial_capital)
    baseline_metrics = {}
    equity_curves = {}
    for name, result in baselines.items():
        m = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)
        baseline_metrics[name] = m
        equity_curves[name] = result.equity_curve

    tbl = metrics_table(baseline_metrics)
    print_table(tbl, "BASELINE PERFORMANCE (full history)")
    save_json(baseline_metrics, Path("outputs/tables/baseline_metrics.json"))

    # ── STEP 5: Edge Discovery ────────────────────────────────────────────────
    print("\n[5/7] Running edge discovery experiments…")
    all_experiments = []
    for family in ["momentum", "mean_reversion", "breakout", "trend", "gap", "calendar"]:
        all_experiments.extend(exp_cfg.get(family, []))
    print(f"      Total hypothesis configurations: {len(all_experiments)}")

    registry = ExperimentRegistry()
    results = run_candidate_experiments(
        dev_df=splits.development,
        val_df=splits.validation,
        regimes_dev=regimes_dev,
        regimes_val=regimes_val,
        experiment_configs=all_experiments,
        registry=registry,
        seed=seed,
    )

    accepted = [r for r in results if r.get("status") == "accepted"]
    rejected = [r for r in results if r.get("status") in ("rejected", "failed")]
    print(f"      Accepted on validation: {len(accepted)}")
    print(f"      Rejected/failed: {len(rejected)}")

    if accepted:
        print("\n      ACCEPTED CANDIDATES:")
        for r in accepted:
            sv = r.get("stats_val", {})
            print(f"        {r['experiment_id']}: EV={sv.get('mean',0)*100:.3f}% "
                  f"HR={sv.get('hit_rate',0):.1%} N={sv.get('n',0)}")

    # ── STEP 6: Walk-forward (on accepted candidates only) ────────────────────
    wf_results = {}
    if accepted and not skip_wf:
        print("\n[6/7] Walk-forward validation on accepted candidates…")
        for r in accepted:
            exp_id = r["experiment_id"]
            cfg_entry = next((e for e in all_experiments if e.get("id") == exp_id), None)
            if not cfg_entry:
                continue

            def make_signal_fn(cfg_entry, regime_engine):
                def signal_fn(full_df, test_df):
                    # Refit regime engine on data up to test period
                    train_df = full_df.iloc[:-len(test_df)]
                    if len(train_df) > 100:
                        re = RegimeEngine(cfg.get("regimes", {}))
                        re.fit(train_df)
                        regimes = re.classify(full_df)
                    else:
                        regimes = None
                    return _build_signal(cfg_entry, full_df, regimes)
                return signal_fn

            def make_backtest_fn(cfg_entry):
                def backtest_fn(test_df, signal):
                    return run_backtest(test_df, signal.loc[test_df.index],
                                       direction=cfg_entry.get("direction", "long"),
                                       cost_profile=ETF_CONSERVATIVE,
                                       initial_capital=initial_capital)
                return backtest_fn

            from src.backtesting.engine import run_backtest
            wf_result = walk_forward(
                df=splits.development,
                signal_fn=make_signal_fn(cfg_entry, regime_engine),
                backtest_fn=make_backtest_fn(cfg_entry),
                metrics_fn=lambda res: compute_metrics(res.returns, res.equity_curve, res.positions, res.trades),
                min_train_bars=cfg["walk_forward"]["min_train_bars"],
                step_bars=cfg["walk_forward"]["step_bars"],
                expanding=cfg["walk_forward"]["expanding"],
                embargo_bars=sp.get("embargo_bars", 20),
            )
            wf_results[exp_id] = wf_result
            oos_m = wf_result.combined_oos_metrics
            print(f"      {exp_id} WF OOS: CAGR={oos_m.get('cagr',0)*100:.2f}% Sharpe={oos_m.get('sharpe_ratio',0):.2f}")
    else:
        if skip_wf:
            print("\n[6/7] Walk-forward SKIPPED (--skip-wf flag)")
        else:
            print("\n[6/7] Walk-forward SKIPPED (no accepted candidates)")

    # ── STEP 7: Report ────────────────────────────────────────────────────────
    print("\n[7/7] Generating charts and report…")
    eq_path = plot_equity_curves(equity_curves, "Baseline Equity Curves")
    bah_dd = plot_drawdown(equity_curves["buy_and_hold"], "buy_and_hold")
    month_path = plot_monthly_heatmap(equity_curves["buy_and_hold"])
    ret_dist = plot_return_distribution(baselines["buy_and_hold"].returns, "buy_and_hold")
    rs_path = plot_rolling_sharpe(baselines["buy_and_hold"].returns, label="buy_and_hold")

    builder = ReportBuilder()

    # Executive summary
    bah_m = baseline_metrics.get("buy_and_hold", {})
    builder.add_section("Executive Summary", f"""
    <p>Phase 1 edge discovery complete for NIFTY 50 (research proxy: ^NSEI).</p>
    <p><strong>Data:</strong> {quality_report.first_date} → {quality_report.last_date}
    ({quality_report.total_rows} sessions) | <strong>Split:</strong>
    Dev {sp['development']*100:.0f}% / Val {sp['validation']*100:.0f}% / Test {sp['test']*100:.0f}%</p>
    <p><strong>Buy &amp; Hold:</strong> CAGR={bah_m.get('cagr',0)*100:.2f}% |
    Sharpe={bah_m.get('sharpe_ratio',0):.2f} | MaxDD={bah_m.get('max_drawdown',0)*100:.2f}%</p>
    <div class='info'>^NSEI is a research proxy only. Costs are indicative. No real-money decisions should be based on these results.</div>
    """)

    builder.add_data_quality(quality_report.to_dict())
    builder.add_metrics_table(baseline_metrics, "Baseline Strategy Performance")

    charts_html = ""
    for path, caption in [
        (eq_path, "Baseline equity curves (normalised, start=1)"),
        (bah_dd, "Buy & Hold drawdown"),
        (month_path, "Buy & Hold monthly return heatmap"),
        (ret_dist, "Buy & Hold daily return distribution"),
        (rs_path, "Buy & Hold 252-day rolling Sharpe"),
    ]:
        charts_html += builder.add_image(path, caption)
    builder.add_section("Baseline Charts", charts_html)

    builder.add_candidate_results(results)
    builder.add_limitations()

    # Go/No-Go
    if accepted:
        # Only "GO" if candidates also pass walk-forward (if run)
        wf_pass = all(
            wf_results[r["experiment_id"]].combined_oos_metrics.get("cagr", 0) > 0
            for r in accepted
            if r["experiment_id"] in wf_results
        ) if wf_results else None

        if wf_pass is None:
            decision = "CONDITIONAL GO (walk-forward not run)"
            rationale = f"{len(accepted)} candidate(s) passed validation. Walk-forward not run — rerun without --skip-wf for full evaluation."
        elif wf_pass:
            decision = "CONDITIONAL GO"
            rationale = f"{len(accepted)} candidate(s) passed validation AND walk-forward OOS. Test period evaluation recommended before Phase 2."
        else:
            decision = "NO-GO"
            rationale = "Candidates passed validation but failed walk-forward OOS. Results are likely overfitted to validation period."
    else:
        decision = "NO-GO"
        rationale = (
            "No candidate passed all validation acceptance criteria. "
            "This is a valid and honest Phase 1 outcome. "
            "Findings: the tested hypotheses (momentum, mean reversion, breakouts, trend, gaps, calendar effects) "
            "did not show a statistically and economically sufficient edge on the NIFTY 50 daily series "
            "in the validation period under the configured cost assumptions. "
            "Recommended: expand data sources, refine hypotheses, or consider weekly timeframe."
        )

    builder.add_go_no_go(decision, rationale)

    data_coverage = f"{quality_report.first_date} → {quality_report.last_date} ({quality_report.total_rows} rows)"
    report_path = builder.build(data_coverage=data_coverage)

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("  PHASE 1 COMPLETE")
    print("="*70)
    print(f"  Time elapsed : {elapsed:.1f}s")
    print(f"  Data         : {quality_report.first_date} → {quality_report.last_date}")
    print(f"  Experiments  : {len(results)} run, {len(accepted)} accepted")
    print(f"  Decision     : {decision}")
    print(f"  Report       : {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()
