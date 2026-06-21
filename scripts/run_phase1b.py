"""Phase 1B master pipeline: real intraday edge discovery for NIFTY 50.

Signal semantics (Phase 1B.1 compliant)
----------------------------------------
Every hypothesis fires AT MOST ONCE per session per setup event (one-shot,
edge-triggered).  Persistent conditions do not re-fire.

Staged classification
---------------------
REJECTED          negative gross executable expectancy or n_trades < 15
FRAGILE           gross positive but break-even cost < 5 bps
EXPLORATORY       positive after 5 bps, n >= 30  → enters validation
VALIDATION SURVIVOR  positive after 10 bps in validation
CONFIRMED         passes walk-forward + placebo + all stress gates

Timeframes evaluated: 5, 15, 30, 60 minutes
Cost scenarios: 0, 2.5, 5, 10, 15, 25 bps round-trip
Holding-period grid: end-of-session, 1, 2, 3, 6 bars fixed

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
from src.intraday import loader, aggregator, features as feat_mod
from src.intraday import hypotheses as hyp_mod
from src.intraday import stats as stats_mod
from src.intraday import placebo as placebo_mod
from src.intraday.engine import run_intraday_backtest, run_intraday_backtest_gross
from src.intraday.walk_forward import intraday_walk_forward
from src.intraday.report import build_phase1b_report

logger = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DEV_FRAC, VAL_FRAC, TEST_FRAC = 0.55, 0.20, 0.25
TIMEFRAMES = [5, 15, 30, 60]       # minutes
COST_SCENARIOS = [0, 2.5, 5, 10, 15, 25]   # bps round-trip
BASE_COST_BPS = 10.0               # main evaluation cost
HOLD_BARS_GRID = [None, 1, 2, 3, 6]  # None = signal/squareoff exit

MIN_TRADES_DEV = 15
MIN_TRADES_VAL = 15
MIN_TRADES_WF  = 30

EXPLORATORY_GROSS_BPS = 5.0    # break-even must be >= this to pass EXPLORATORY
WF_POS_FOLD_THRESH = 0.55
PLACEBO_N = 300
SQUAREOFF = "15:20"
NO_ENTRY_AFTER = "15:00"

# OR windows in minutes — converted to bars per TF inside functions
OR_MINUTES = [15, 30, 60]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_or_cache(path: Path) -> pd.DataFrame:
    cache = data_processed_dir() / "NSEI_1min_processed.parquet"
    if cache.exists():
        logger.info("Loading cached 1-min data: %s", cache)
        return load_parquet(cache)
    df = loader.load_minute_data(str(path))
    save_parquet(df, cache)
    logger.info("Cached 1-min data to %s", cache)
    return df


def _session_slice(df_1min: pd.DataFrame, sessions: list) -> pd.DataFrame:
    dates = df_1min.index.normalize()
    mask = dates.isin(pd.to_datetime(pd.Index(sessions)))
    return df_1min[mask]


def _build_tf_frame(df_1min: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Resample + compute features for a given timeframe."""
    or_bar_windows = sorted({max(1, m // tf) for m in OR_MINUTES})
    df_tf = aggregator.resample_to_tf(df_1min, tf)
    return feat_mod.compute_intraday_features(df_tf, or_bar_windows=or_bar_windows)


# ── Cost / break-even analysis ────────────────────────────────────────────────

def _break_even_bps(avg_gross_pnl_pts: float, avg_price: float) -> float:
    """Round-trip bps at which net expectancy = 0.

    avg_gross / (entry + exit price in pts) × 10000 × 2 ≈ round-trip bps.
    Simplified: cost_bps × avg_price/10000 = avg_gross  →  bps = avg_gross/avg_price*10000
    (assumes entry≈exit≈avg_price; this gives the one-way bps; ×2 for round-trip).
    """
    if avg_price <= 0:
        return 0.0
    one_way_bps = (avg_gross_pnl_pts / avg_price) * 10_000
    return float(one_way_bps * 2)   # round-trip


def _cost_sweep(trades: list, avg_entry_price: float) -> dict:
    """Compute avg_net_pnl at each cost scenario."""
    sweep = {}
    for bps in COST_SCENARIOS:
        leg = (bps / 2.0) / 10_000.0
        nets = [t["gross_pnl"] - (t["entry_price"] + t["exit_price"]) * leg for t in trades]
        sweep[f"{bps}bps"] = float(np.mean(nets)) if nets else 0.0
    return sweep


# ── Staged classification ─────────────────────────────────────────────────────

def _classify_dev(gross_avg: float, be_bps: float, n_trades: int) -> str:
    """Return dev-stage label."""
    if n_trades < MIN_TRADES_DEV or gross_avg <= 0:
        return "REJECTED"
    if be_bps < EXPLORATORY_GROSS_BPS:
        return "FRAGILE"
    return "EXPLORATORY"


def _classify_val(net_avg_10bps: float, n_trades: int) -> str:
    if net_avg_10bps > 0 and n_trades >= MIN_TRADES_VAL:
        return "VALIDATION_SURVIVOR"
    return "REJECTED_VAL"


# ── Per-hypothesis runner ─────────────────────────────────────────────────────

def _run_hypothesis(
    cfg: dict,
    feat_df: pd.DataFrame,   # feature frame at chosen TF
    exec_df: pd.DataFrame,   # execution frame (same TF)
    tf: int,
    cost_bps: float = BASE_COST_BPS,
    hold_bars: int | None = None,
) -> dict:
    """Generate signals, run backtest, compute stats. Returns result dict."""
    sig, funnel = hyp_mod.generate_signal(feat_df, cfg, tf_minutes=tf, return_funnel=True)
    # Gross run (0 cost)
    res_gross = run_intraday_backtest_gross(
        exec_df, sig,
        squareoff_time=SQUAREOFF, no_entry_after=NO_ENTRY_AFTER,
        hold_bars=hold_bars,
    )
    # Net run at base cost
    res_net = run_intraday_backtest(
        exec_df, sig, cost_bps=cost_bps,
        squareoff_time=SQUAREOFF, no_entry_after=NO_ENTRY_AFTER,
        hold_bars=hold_bars,
    )
    trades = res_net["trades"]
    gross_trades = res_gross["trades"]

    avg_gross = float(np.mean([t["gross_pnl"] for t in gross_trades])) if gross_trades else 0.0
    avg_entry_p = float(np.mean([t["entry_price"] for t in trades])) if trades else 20_000.0
    be_bps = _break_even_bps(avg_gross, avg_entry_p)
    cost_sw = _cost_sweep(trades, avg_entry_p)

    st = stats_mod.compute_trade_stats(trades, cost_bps)

    return {
        "id": cfg["id"],
        "tf": tf,
        "hold_bars": hold_bars,
        "funnel": funnel,
        "n_trades": len(trades),
        "avg_gross_pnl": avg_gross,
        "break_even_bps": round(be_bps, 2),
        "cost_sweep": cost_sw,
        "avg_net_pnl": st["avg_net_pnl"],
        "t_stat": st["t_stat"],
        "hit_rate": st["hit_rate"],
        "profit_factor": st["profit_factor"],
        "metrics": res_net["metrics"],
        "trades": trades,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging("INFO")
    ensure_dirs()

    # ── Stage 1: load + validate ──────────────────────────────────────────
    data_path = Path("/tmp/niftydata/nifty-real-data/nifty_vix_1min_strict.csv")
    df_1min = _load_or_cache(data_path)
    quality = loader.validate_minute_data(df_1min)
    save_json(quality, data_processed_dir().parent / "metadata" / "phase1b_data_quality.json")
    logger.info("Validation: sessions=%d, 375-bar=%d, passed=%s",
                quality["n_sessions"], quality["sessions_with_375_bars"], quality["passed"])

    # ── Stage 2: session splits ──────────────────────────────────────────
    sessions = aggregator.get_sessions(df_1min)
    n = len(sessions)
    n_dev = int(n * DEV_FRAC)
    n_val = int(n * VAL_FRAC)
    dev_sessions  = sessions[:n_dev]
    val_sessions  = sessions[n_dev:n_dev + n_val]
    test_sessions = sessions[n_dev + n_val:]
    logger.info("Sessions: total=%d dev=%d val=%d test=%d",
                n, len(dev_sessions), len(val_sessions), len(test_sessions))

    df_dev  = _session_slice(df_1min, dev_sessions)
    df_val  = _session_slice(df_1min, val_sessions)

    # ── Stage 3: development screening — all TFs × all hypotheses × end-of-session exit ──
    print("\n" + "="*70)
    print("  PHASE 1B.1 — DEVELOPMENT SCREENING (one-shot edge-triggered signals)")
    print("="*70)

    dev_results = []   # all hypothesis × TF combinations
    exploratory = []   # (cfg, tf) pairs advancing to validation

    for tf in TIMEFRAMES:
        feat_dev = _build_tf_frame(df_dev, tf)
        exec_dev = aggregator.resample_to_tf(df_dev, tf)
        logger.info("TF=%dmin: screening %d hypotheses...", tf, len(hyp_mod.HYPOTHESIS_CONFIGS))

        for cfg in hyp_mod.HYPOTHESIS_CONFIGS:
            try:
                row = _run_hypothesis(cfg, feat_dev, exec_dev, tf)
            except Exception as e:
                logger.warning("Hypothesis %s TF=%d failed: %s", cfg["id"], tf, e)
                continue

            label = _classify_dev(row["avg_gross_pnl"], row["break_even_bps"], row["n_trades"])
            row["dev_label"] = label
            dev_results.append(row)

            be = row["break_even_bps"]
            n_tr = row["n_trades"]
            gross = row["avg_gross_pnl"]
            print(f"  {cfg['id']:20s} TF={tf:2d}min  n={n_tr:5d}  "
                  f"gross={gross:+7.2f}pts  BE={be:5.1f}bps  [{label}]")

            if label == "EXPLORATORY":
                exploratory.append((cfg, tf))

    print(f"\n  EXPLORATORY (advancing to validation): "
          f"{len(exploratory)} / {len(dev_results)} combinations")

    # ── Stage 3b: holding-period grid on EXPLORATORY candidates ──────────
    hold_results = []
    for cfg, tf in exploratory:
        feat_dev = _build_tf_frame(df_dev, tf)
        exec_dev = aggregator.resample_to_tf(df_dev, tf)
        best_hold = None
        best_net = -np.inf
        for hb in HOLD_BARS_GRID:
            try:
                row = _run_hypothesis(cfg, feat_dev, exec_dev, tf, hold_bars=hb)
            except Exception:
                continue
            row["dev_label"] = _classify_dev(
                row["avg_gross_pnl"], row["break_even_bps"], row["n_trades"])
            hold_results.append(row)
            if row["avg_net_pnl"] > best_net:
                best_net = row["avg_net_pnl"]
                best_hold = hb

    # ── Stage 4: validation ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("  VALIDATION")
    print("="*70)

    val_results = []
    val_survivors = []   # (cfg, tf, hold_bars) triples

    if not exploratory:
        print("  No EXPLORATORY candidates — validation skipped.")
    else:
        for cfg, tf in exploratory:
            feat_val = _build_tf_frame(df_val, tf)
            exec_val = aggregator.resample_to_tf(df_val, tf)

            # Select best hold from dev grid (or None if no grid ran)
            best_hold = None
            best_net  = -np.inf
            for row in hold_results:
                if row["id"] == cfg["id"] and row["tf"] == tf:
                    if row.get("avg_net_pnl", -np.inf) > best_net:
                        best_net  = row["avg_net_pnl"]
                        best_hold = row["hold_bars"]

            try:
                vrow = _run_hypothesis(cfg, feat_val, exec_val, tf, hold_bars=best_hold)
            except Exception as e:
                logger.warning("Val run failed %s tf=%d: %s", cfg["id"], tf, e)
                continue

            vlabel = _classify_val(
                vrow["cost_sweep"].get("10.0bps", vrow["avg_net_pnl"]), vrow["n_trades"]
            )
            vrow["val_label"] = vlabel
            vrow["hold_bars"] = best_hold
            val_results.append(vrow)

            print(f"  {cfg['id']:20s} TF={tf:2d}min  n={vrow['n_trades']:5d}  "
                  f"avg_net={vrow['avg_net_pnl']:+7.2f}pts  [{vlabel}]")

            if vlabel == "VALIDATION_SURVIVOR":
                val_survivors.append((cfg, tf, best_hold))

    print(f"\n  VALIDATION SURVIVORS: {len(val_survivors)}")

    # ── Stage 5: walk-forward on VALIDATION SURVIVORS ────────────────────
    print("\n" + "="*70)
    print("  WALK-FORWARD VALIDATION (expanding window, per-session)")
    print("="*70)

    devval_sessions = dev_sessions + val_sessions
    df_devval = _session_slice(df_1min, devval_sessions)

    wf_results  = []
    wf_pass     = []

    for cfg, tf, hold_bars in val_survivors:
        exec_devval = aggregator.resample_to_tf(df_devval, tf)
        or_bar_windows = sorted({max(1, m // tf) for m in OR_MINUTES})

        def _sig_fn(train_sess, test_sess, _cfg=cfg, _tf=tf, _or=or_bar_windows):
            test_dates = pd.to_datetime(pd.Index(test_sess))
            mask = exec_devval.index.normalize().isin(test_dates)
            sub = exec_devval[mask]
            if len(sub) == 0:
                return pd.Series(0, index=pd.DatetimeIndex([]), dtype=int)
            feat_sub = feat_mod.compute_intraday_features(sub, or_bar_windows=_or)
            return hyp_mod.generate_signal(feat_sub, _cfg, tf_minutes=_tf)

        def _bt_fn(df_test, signals, _hb=hold_bars):
            return run_intraday_backtest(
                df_test, signals, cost_bps=BASE_COST_BPS,
                squareoff_time=SQUAREOFF, no_entry_after=NO_ENTRY_AFTER,
                hold_bars=_hb,
            )

        wf = intraday_walk_forward(
            exec_devval, devval_sessions, _sig_fn, _bt_fn,
            min_train_sessions=max(200, len(dev_sessions) // 2),
            step_sessions=50,
            embargo_sessions=5,
        )
        oos = wf["oos_stats"]
        passed = (
            wf["pct_positive_folds"] >= WF_POS_FOLD_THRESH
            and oos["n_trades"] >= MIN_TRADES_WF
            and oos["avg_net_pnl"] > 0
        )
        wrow = {
            "id": cfg["id"], "tf": tf, "hold_bars": hold_bars,
            "n_folds": wf["n_folds"],
            "pct_positive_folds": wf["pct_positive_folds"],
            "oos_n_trades": oos["n_trades"],
            "oos_avg_net_pnl": oos["avg_net_pnl"],
            "oos_t_stat": oos["t_stat"],
            "wf_pass": passed,
        }
        wf_results.append(wrow)
        status = "PASS" if passed else "FAIL"
        print(f"  {cfg['id']:20s} TF={tf:2d}min  folds={wf['n_folds']}  "
              f"pos={wf['pct_positive_folds']:.0%}  "
              f"oos_net={oos['avg_net_pnl']:+.2f}pts  [{status}]")
        if passed:
            wf_pass.append((cfg, tf, hold_bars))

    # ── Stage 6: placebo testing ──────────────────────────────────────────
    print("\n" + "="*70)
    print("  PLACEBO TESTING")
    print("="*70)

    placebo_results = []
    placebo_pass = []

    for cfg, tf, hold_bars in wf_pass:
        # Re-run on val to get actual trades for matched placebo
        feat_val = _build_tf_frame(df_val, tf)
        exec_val = aggregator.resample_to_tf(df_val, tf)
        sig = hyp_mod.generate_signal(feat_val, cfg, tf_minutes=tf)
        res = run_intraday_backtest(
            exec_val, sig, cost_bps=BASE_COST_BPS,
            squareoff_time=SQUAREOFF, no_entry_after=NO_ENTRY_AFTER,
            hold_bars=hold_bars,
        )
        pl = placebo_mod.run_placebo_simulations(
            exec_val, res["trades"], n_sims=PLACEBO_N,
            cost_bps=BASE_COST_BPS, seed=42,
        )
        prow = {
            "id": cfg["id"], "tf": tf,
            "real_avg_net": float(np.mean([t["net_pnl"] for t in res["trades"]])) if res["trades"] else 0.0,
            "placebo_p": pl["p_value"],
            "pct_rank": pl["pct_rank"],
        }
        placebo_results.append(prow)
        beats = pl["p_value"] <= 0.05
        print(f"  {cfg['id']:20s} TF={tf:2d}min  p={pl['p_value']:.3f}  "
              f"rank={pl['pct_rank']:.1%}  {'BEATS PLACEBO' if beats else 'DOES NOT BEAT PLACEBO'}")
        if beats:
            placebo_pass.append((cfg, tf, hold_bars))

    # ── Stage 7: untouched test — ONLY for placebo-passing candidates ─────
    print("\n" + "="*70)
    print("  UNTOUCHED TEST PERIOD")
    print("="*70)

    test_results = []
    if not placebo_pass:
        print("  No candidates passed all gates — untouched test NOT accessed.")
    else:
        df_test = _session_slice(df_1min, test_sessions)
        for cfg, tf, hold_bars in placebo_pass:
            feat_test = _build_tf_frame(df_test, tf)
            exec_test  = aggregator.resample_to_tf(df_test, tf)
            sig = hyp_mod.generate_signal(feat_test, cfg, tf_minutes=tf)
            res = run_intraday_backtest(
                exec_test, sig, cost_bps=BASE_COST_BPS,
                squareoff_time=SQUAREOFF, no_entry_after=NO_ENTRY_AFTER,
                hold_bars=hold_bars,
            )
            st = stats_mod.compute_trade_stats(res["trades"], BASE_COST_BPS)
            trow = {
                "id": cfg["id"], "tf": tf,
                "n_trades": st["n_trades"],
                "avg_net_pnl": st["avg_net_pnl"],
                "t_stat": st["t_stat"],
                "sharpe": res["metrics"]["sharpe"],
                "profit_factor": st["profit_factor"],
            }
            test_results.append(trow)
            print(f"  {cfg['id']:20s} TF={tf:2d}min  n={st['n_trades']:5d}  "
                  f"avg_net={st['avg_net_pnl']:+.2f}pts  t={st['t_stat']:.2f}")

    # ── Stage 8: decision ─────────────────────────────────────────────────
    decision, rationale = _decide(
        dev_results, val_results, wf_results, placebo_results, test_results
    )

    # ── Summary printout ──────────────────────────────────────────────────
    _print_summary(dev_results, val_results, wf_results, placebo_results, test_results)

    # ── Save results ──────────────────────────────────────────────────────
    def _safe_json(obj):
        """Strip non-serializable 'trades' lists from result rows."""
        if isinstance(obj, dict):
            return {k: _safe_json(v) for k, v in obj.items() if k != "trades"}
        if isinstance(obj, list):
            return [_safe_json(x) for x in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    results = {
        "decision": decision,
        "rationale": rationale,
        "meta": {
            "n_dev": len(dev_sessions), "n_val": len(val_sessions),
            "n_test": len(test_sessions), "base_cost_bps": BASE_COST_BPS,
            "timeframes": TIMEFRAMES, "cost_scenarios": COST_SCENARIOS,
        },
        "dev_results":     _safe_json(dev_results),
        "val_results":     _safe_json(val_results),
        "wf_results":      _safe_json(wf_results),
        "placebo_results": _safe_json(placebo_results),
        "test_results":    _safe_json(test_results),
    }
    save_json(results, data_processed_dir().parent / "metadata" / "phase1b_results.json")
    report_path = build_phase1b_report(results)
    logger.info("Report: %s", report_path)

    print("\n" + "="*70)
    print(f"  PHASE 1B FINAL DECISION: {decision}")
    print(f"  {rationale}")
    print("="*70)
    print(f"\n  Report: {report_path}")
    print("  Reproduce: python scripts/run_phase1b.py")


# ── Decision logic ─────────────────────────────────────────────────────────────

def _decide(dev_results, val_results, wf_results, placebo_results, test_results):
    n_expl = sum(1 for r in dev_results if r.get("dev_label") == "EXPLORATORY")
    n_val  = sum(1 for r in val_results if r.get("val_label") == "VALIDATION_SURVIVOR")
    n_wf   = sum(1 for r in wf_results  if r.get("wf_pass"))
    n_plac = sum(1 for r in placebo_results if r.get("placebo_p", 1) <= 0.05)
    test_confirmed = [
        r for r in test_results
        if r.get("avg_net_pnl", 0) > 0 and r.get("t_stat", 0) >= 1.5
    ]

    if test_confirmed:
        best = max(test_confirmed, key=lambda r: r.get("t_stat", 0))
        if best.get("sharpe", 0) >= 0.75 and len(test_confirmed) >= 1:
            return (
                "PAPER-TRADE READY",
                f"Candidate '{best['id']}' (TF={best['tf']}min) survived all gates "
                f"including untouched test (t={best['t_stat']:.2f}, "
                f"Sharpe={best['sharpe']:.2f}). Ready for paper trading.",
            )
    if n_plac > 0:
        return (
            "RESEARCH CANDIDATE",
            f"{n_plac} candidate(s) beat placebo and walk-forward but failed the "
            f"untouched-test confirmation bar. Requires additional robustness work.",
        )
    if n_wf > 0:
        return (
            "RESEARCH CANDIDATE",
            f"{n_wf} candidate(s) passed walk-forward but failed placebo "
            f"({n_plac}/{n_wf} beat placebo). Results may be regime-specific.",
        )
    if n_val > 0:
        return (
            "WEAK EVIDENCE",
            f"{n_val} candidate(s) survived validation but failed walk-forward. "
            f"Edge is period-specific or not robust to changing regimes.",
        )
    if n_expl > 0:
        return (
            "WEAK EVIDENCE",
            f"{n_expl} hypothesis×TF combinations showed positive gross expectancy "
            f"(EXPLORATORY) but did not survive validation. Insufficient evidence.",
        )
    return (
        "NO EDGE",
        "After correcting signal semantics to one-shot edge-triggered logic, no "
        "hypothesis showed positive gross executable expectancy on the development set. "
        "Simple intraday conditional strategies on NIFTY 50 do not demonstrate a "
        "reliable, cost-robust edge in this 10-year dataset.",
    )


# ── Summary printer ────────────────────────────────────────────────────────────

def _print_summary(dev_results, val_results, wf_results, placebo_results, test_results):
    print("\n" + "="*70)
    print("  PHASE 1B.1 SUMMARY")
    print("="*70)

    labels = {"REJECTED": 0, "FRAGILE": 0, "EXPLORATORY": 0}
    for r in dev_results:
        lbl = r.get("dev_label", "REJECTED")
        labels[lbl] = labels.get(lbl, 0) + 1
    total_hyp = len(hyp_mod.HYPOTHESIS_CONFIGS) * len(TIMEFRAMES)
    print(f"\n  Development ({total_hyp} hypothesis×TF combinations):")
    for lbl, cnt in sorted(labels.items()):
        print(f"    {lbl:20s}: {cnt}")

    if val_results:
        print(f"\n  Validation ({len(val_results)} EXPLORATORY candidates evaluated):")
        for lbl in ["VALIDATION_SURVIVOR", "REJECTED_VAL"]:
            cnt = sum(1 for r in val_results if r.get("val_label") == lbl)
            print(f"    {lbl:25s}: {cnt}")

    if wf_results:
        print(f"\n  Walk-forward ({len(wf_results)} candidates):")
        print(f"    PASS: {sum(1 for r in wf_results if r.get('wf_pass'))}")
        print(f"    FAIL: {sum(1 for r in wf_results if not r.get('wf_pass'))}")

    if placebo_results:
        print(f"\n  Placebo (n={PLACEBO_N} simulations each):")
        print(f"    Beat placebo (p<=0.05): {sum(1 for r in placebo_results if r.get('placebo_p',1)<=0.05)}")

    if test_results:
        print(f"\n  Untouched test ({len(test_results)} candidates evaluated):")
        for r in test_results:
            print(f"    {r['id']:20s} TF={r['tf']}min  n={r['n_trades']}  "
                  f"avg_net={r['avg_net_pnl']:+.2f}pts  t={r['t_stat']:.2f}")
    else:
        print("\n  Untouched test: NOT ACCESSED")

    # Signal funnel for EXPLORATORY candidates
    exploratory_rows = [r for r in dev_results if r.get("dev_label") == "EXPLORATORY"]
    if exploratory_rows:
        print("\n  Signal funnel (EXPLORATORY candidates only):")
        print(f"  {'Hypothesis':20s}  {'TF':>4s}  {'RawBars':>8s}  "
              f"{'Edges':>7s}  {'Executed':>8s}  {'Reduction':>10s}")
        for r in exploratory_rows:
            f = r.get("funnel", {})
            print(f"  {r['id']:20s}  {r['tf']:4d}  "
                  f"{f.get('raw_condition_bars',0):8d}  "
                  f"{f.get('rising_edges',0):7d}  "
                  f"{f.get('executed_signals',0):8d}  "
                  f"{f.get('turnover_reduction_vs_raw',0):9.1%}")


if __name__ == "__main__":
    main()
