"""Phase 1B master pipeline: real intraday edge discovery for NIFTY 50.

Stages
------
1. Load + validate 1-min data (cached to data/processed parquet).
2. Session-count splits: 55% dev / 20% val / 25% test.
3. Compute causal intraday features (on a 5-min research timeframe).
4. Screen all hypotheses on the development period.
5. Evaluate survivors on validation (+ placebo).
6. Walk-forward (expanding window) on validation survivors.
7. Evaluate the UNTOUCHED test set exactly once, for confirmed candidates only.
8. Build report + print the final decision.

Run: ``python scripts/run_phase1b.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.logging import get_logger, configure_logging
from src.utils.paths import ensure_dirs, data_processed_dir
from src.utils.serialization import save_json, save_parquet, load_parquet
from src.intraday import loader, aggregator, features, hypotheses, stats, placebo
from src.intraday.engine import run_intraday_backtest
from src.intraday.walk_forward import intraday_walk_forward
from src.intraday.report import build_phase1b_report

logger = get_logger(__name__)

COST_BPS = 10.0
TF_MINUTES = 5  # research timeframe (5-min bars anchored to 09:15)
# OR windows in BARS on the research timeframe: 15/30/60 min => 3/6/12 bars.
DEV_FRAC, VAL_FRAC, TEST_FRAC = 0.55, 0.20, 0.25

# Screening thresholds (deliberately strict to control multiple testing).
MIN_TRADES = 30
DEV_T_THRESHOLD = 1.5
VAL_T_THRESHOLD = 1.5
PLACEBO_P_THRESHOLD = 0.05
WF_POS_FOLD_THRESHOLD = 0.55
TEST_T_THRESHOLD = 1.5


def _backtest(df, signals):
    return run_intraday_backtest(df, signals, cost_bps=COST_BPS)


# OR windows on the research timeframe (minute windows / TF).
OR_BAR_WINDOWS = sorted({max(1, w // TF_MINUTES) for w in (15, 30, 60)})


def _build_features(df_1min: pd.DataFrame) -> pd.DataFrame:
    df_tf = aggregator.resample_to_tf(df_1min, TF_MINUTES)
    feat = features.compute_intraday_features(df_tf, or_bar_windows=OR_BAR_WINDOWS)
    return feat


def _adapt_or_bars(config: dict) -> dict:
    """Convert minute OR windows to research-timeframe bar windows."""
    c = dict(config)
    if "or_bars" in c:
        c["or_bars"] = max(1, c["or_bars"] // TF_MINUTES)
    return c


def main() -> None:
    configure_logging("INFO")
    ensure_dirs()

    # ---- Stage 1: load + validate (with parquet cache) ----
    cache = data_processed_dir() / "NSEI_1min_processed.parquet"
    if cache.exists():
        logger.info("Loading cached processed minute data: %s", cache)
        df_1min = load_parquet(cache)
    else:
        df_1min = loader.load_minute_data()
        save_parquet(df_1min, cache)
        logger.info("Cached processed minute data to %s", cache)

    quality = loader.validate_minute_data(df_1min)
    save_json(quality, data_processed_dir() / "phase1b_data_quality.json")
    if not quality["passed"]:
        logger.warning("Data validation flagged issues: %s", quality)

    # ---- Stage 2: session-count splits ----
    sessions = aggregator.get_sessions(df_1min)
    n = len(sessions)
    n_dev = int(n * DEV_FRAC)
    n_val = int(n * VAL_FRAC)
    dev_sessions = sessions[:n_dev]
    val_sessions = sessions[n_dev : n_dev + n_val]
    test_sessions = sessions[n_dev + n_val :]
    logger.info(
        "Sessions: total=%d dev=%d val=%d test=%d",
        n, len(dev_sessions), len(val_sessions), len(test_sessions),
    )

    dates = df_1min.index.normalize()

    def slice_sessions(sess):
        mask = dates.isin(pd.to_datetime(pd.Index(sess)))
        return df_1min[mask]

    df_dev = slice_sessions(dev_sessions)
    df_val = slice_sessions(val_sessions)
    df_test = slice_sessions(test_sessions)

    # ---- Stage 3: features ----
    logger.info("Computing causal features (tf=%dmin)...", TF_MINUTES)
    feat_dev = _build_features(df_dev)
    feat_val = _build_features(df_val)

    # The feature frame is on the research timeframe; the engine runs on the
    # SAME frame so that 'next bar open' executes at the research-bar open.
    df_dev_tf = aggregator.resample_to_tf(df_dev, TF_MINUTES)
    df_val_tf = aggregator.resample_to_tf(df_val, TF_MINUTES)

    # ---- Stage 4: development screening ----
    logger.info("Screening %d hypotheses on development...", len(hypotheses.HYPOTHESIS_CONFIGS))
    dev_results = []
    survivors = []
    for cfg in hypotheses.HYPOTHESIS_CONFIGS:
        acfg = _adapt_or_bars(cfg)
        try:
            sig = hypotheses.generate_signal(feat_dev, acfg)
        except Exception as e:  # noqa
            logger.warning("Hypothesis %s failed: %s", cfg["id"], e)
            continue
        res = _backtest(df_dev_tf, sig)
        st = stats.compute_trade_stats(res["trades"], COST_BPS)
        survived = (
            st["n_trades"] >= MIN_TRADES
            and st["t_stat"] >= DEV_T_THRESHOLD
            and st["avg_net_pnl"] > 0
        )
        row = {"id": cfg["id"], "survived": survived, **st}
        dev_results.append(row)
        if survived:
            survivors.append(cfg)
    logger.info("Dev survivors: %s", [c["id"] for c in survivors])

    # ---- Stage 5: validation + placebo ----
    val_results = []
    confirmed = []
    for cfg in survivors:
        acfg = _adapt_or_bars(cfg)
        sig = hypotheses.generate_signal(feat_val, acfg)
        res = _backtest(df_val_tf, sig)
        st = stats.compute_trade_stats(res["trades"], COST_BPS)
        row = {"id": cfg["id"], **st}
        is_conf = st["n_trades"] >= MIN_TRADES and st["t_stat"] >= VAL_T_THRESHOLD and st["avg_net_pnl"] > 0
        if is_conf and res["trades"]:
            pl = placebo.run_placebo_simulations(
                df_val_tf, res["trades"], n_sims=200, cost_bps=COST_BPS
            )
            row["placebo_p"] = pl["p_value"]
            is_conf = is_conf and pl["p_value"] <= PLACEBO_P_THRESHOLD
        row["confirmed"] = bool(is_conf)
        val_results.append(row)
        if is_conf:
            confirmed.append(cfg)
    logger.info("Validation-confirmed: %s", [c["id"] for c in confirmed])

    # ---- Stage 6: walk-forward on confirmed (over dev+val sessions) ----
    devval_sessions = dev_sessions + val_sessions
    df_devval = slice_sessions(devval_sessions)
    df_devval_tf = aggregator.resample_to_tf(df_devval, TF_MINUTES)
    feat_devval = features.compute_intraday_features(
        df_devval_tf, or_bar_windows=OR_BAR_WINDOWS
    )

    wf_results = []
    wf_pass = []
    for cfg in confirmed:
        acfg = _adapt_or_bars(cfg)

        def signal_fn(train_sess, test_sess, _df, _acfg=acfg):
            test_dates = pd.to_datetime(pd.Index(test_sess))
            m = df_devval_tf.index.normalize().isin(test_dates)
            sub = feat_devval[m]
            return hypotheses.generate_signal(sub, _acfg)

        def backtest_fn(df_test, signals):
            return _backtest(df_test, signals)

        wf = intraday_walk_forward(
            df_devval_tf,
            devval_sessions,
            signal_fn,
            backtest_fn,
            min_train_sessions=max(250, len(dev_sessions) // 2),
            step_sessions=63,
            embargo_sessions=5,
        )
        oos = wf["oos_stats"]
        passed = (
            wf["pct_positive_folds"] >= WF_POS_FOLD_THRESHOLD
            and oos["n_trades"] >= MIN_TRADES
            and oos["t_stat"] >= 0.0
            and oos["avg_net_pnl"] > 0
        )
        wf_results.append({
            "id": cfg["id"],
            "n_folds": wf["n_folds"],
            "pct_positive_folds": wf["pct_positive_folds"],
            "oos_n_trades": oos["n_trades"],
            "oos_t_stat": oos["t_stat"],
            "oos_net_pnl": oos["total_net_pnl"],
        })
        if passed:
            wf_pass.append(cfg)
    logger.info("Walk-forward passed: %s", [c["id"] for c in wf_pass])

    # ---- Stage 7: untouched TEST, exactly once, for wf-passing candidates ----
    test_results = []
    if wf_pass:
        feat_test = _build_features(df_test)
        df_test_tf = aggregator.resample_to_tf(df_test, TF_MINUTES)
        for cfg in wf_pass:
            acfg = _adapt_or_bars(cfg)
            sig = hypotheses.generate_signal(feat_test, acfg)
            res = _backtest(df_test_tf, sig)
            st = stats.compute_trade_stats(res["trades"], COST_BPS)
            test_results.append({
                "id": cfg["id"],
                "sharpe": res["metrics"]["sharpe"],
                **st,
            })

    # ---- Stage 8: decision ----
    decision, rationale = _decide(dev_results, val_results, wf_results, test_results)

    results = {
        "decision": decision,
        "decision_rationale": rationale,
        "meta": {
            "n_dev_sessions": len(dev_sessions),
            "n_val_sessions": len(val_sessions),
            "n_test_sessions": len(test_sessions),
            "cost_bps": COST_BPS,
            "tf_minutes": TF_MINUTES,
        },
        "data_quality": quality,
        "dev_results": dev_results,
        "val_results": val_results,
        "wf_results": wf_results,
        "test_results": test_results,
    }
    save_json(results, data_processed_dir().parent / "metadata" / "phase1b_results.json")
    out = build_phase1b_report(results)
    logger.info("Report written: %s", out)

    print("\n" + "=" * 60)
    print("PHASE 1B FINAL DECISION:", decision)
    print(rationale)
    print("=" * 60)


def _decide(dev_results, val_results, wf_results, test_results):
    """Map evidence to one of the canonical decision labels."""
    n_confirmed = sum(1 for r in val_results if r.get("confirmed"))
    n_wf = len(wf_results)
    test_pass = [r for r in test_results if r.get("t_stat", 0) >= TEST_T_THRESHOLD and r.get("avg_net_pnl", 0) > 0]

    if test_pass:
        best = max(test_pass, key=lambda r: r.get("t_stat", 0))
        # Distinguish PAPER-TRADE READY vs RL ELIGIBLE by strength + breadth.
        if best.get("sharpe", 0) >= 1.0 and len(test_pass) >= 2:
            return (
                "RL ELIGIBLE",
                f"{len(test_pass)} hypotheses survived dev→val→placebo→walk-forward and "
                f"held up on the untouched test (best t={best['t_stat']:.2f}, "
                f"sharpe={best.get('sharpe',0):.2f}). Edge is robust enough to justify "
                f"an RL/learning layer.",
            )
        return (
            "PAPER-TRADE READY",
            f"Candidate '{best['id']}' survived full validation and the untouched test "
            f"(t={best['test_t_stat'] if 'test_t_stat' in best else best['t_stat']:.2f}). "
            f"Promote to forward paper trading.",
        )

    if n_wf > 0:
        return (
            "RESEARCH CANDIDATE",
            f"{n_wf} hypotheses passed validation+placebo and walk-forward but did NOT "
            f"clear the untouched-test bar. Worth further research, not deployment.",
        )
    if n_confirmed > 0:
        return (
            "WEAK EVIDENCE",
            f"{n_confirmed} hypotheses confirmed on validation but failed walk-forward "
            f"robustness. Evidence is weak and likely period-specific.",
        )
    return (
        "NO EDGE",
        "No hypothesis cleared the development+validation screening with statistically "
        "meaningful, cost-aware net edge. No intraday edge found.",
    )


if __name__ == "__main__":
    main()
