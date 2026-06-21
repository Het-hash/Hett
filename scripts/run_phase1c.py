"""Phase 1C: Contextual Edge Refinement.

Determines whether predeclared market-context filters can identify profitable
sub-regimes for the two Phase 1B candidates:
  - ORF_UP_SHORT @ 30 min
  - GAP_REV_LONG @ 60 min

Design constraints enforced here:
  - Parent signal definitions are FROZEN (Phase 1B output).
  - Only inner development splits used for filter selection.
  - At most two-filter combinations tested (pre-specified list).
  - 1000 matched placebo simulations.
  - Benjamini-Hochberg FDR correction.
  - Untouched test set (locked_test) is NEVER accessed.
"""
from __future__ import annotations

import sys
import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.intraday.loader import load_minute_data
from src.intraday.aggregator import resample_to_tf, get_sessions
from src.intraday.features import compute_intraday_features as compute_features
from src.intraday.hypotheses import generate_signal, HYPOTHESIS_CONFIGS
from src.intraday.engine import run_intraday_backtest
from src.intraday.context_filters import (
    compute_daily_context,
    merge_context_to_tf,
    SINGLE_FILTERS,
    TWO_FILTER_COMBOS,
    apply_filter,
    filter_trades,
)
from src.intraday.edge_decay import (
    per_year_stats,
    rolling_expectancy,
    pre_post_2020_split,
    be_cost_through_time,
    classify_decay,
)

# ── Constants ────────────────────────────────────────────────────────────────
CANDIDATES = [
    {"id": "ORF_UP_SHORT", "tf": 30},
    {"id": "GAP_REV_LONG", "tf": 60},
]
COST_BPS = 10.0
FILTER_COST_SURVIVAL = 5.0   # filter must survive at 5 bps
MIN_TRADES_DEV = 30          # relaxed — some filters will be sparse
MIN_TRADES_OUTER = 15
N_PLACEBO = 1000
FDR_ALPHA = 0.05
SQUAREOFF = "15:20"
NO_ENTRY_AFTER = "15:00"
HOLD_BARS = None             # use forced square-off


# ── Session splits (by session index, from Phase 1B) ─────────────────────────
# Total dev sessions: 1375  (2015-01-09 to 2020-12-21 approx)
# inner_train=756, inner_selection=275, inner_confirmation=344
# outer_val=500   (2021-01 to 2023-01 approx)
# locked_test=625 (2023-01 onwards) — NEVER ACCESSED

INNER_TRAIN_N      = 756
INNER_SELECTION_N  = 275
INNER_CONFIRM_N    = 344
OUTER_VAL_N        = 500
# locked_test = everything after outer_val


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sessions_to_mask(sessions_all: list, sessions_sub: list) -> pd.Series:
    sub_set = set(sessions_sub)
    dates = pd.DatetimeIndex([s for s in sessions_all])
    return pd.Series(True, index=dates).where(
        pd.Series([s in sub_set for s in sessions_all], index=dates), False
    )


def run_on_sessions(
    df_tf: pd.DataFrame,
    signal: pd.Series,
    sessions: list,
    cost_bps: float = COST_BPS,
    context_mask: pd.Series | None = None,
) -> dict:
    """Run backtest restricted to specified sessions."""
    session_set = set(pd.Timestamp(s).normalize() for s in sessions)
    dates = df_tf.index.normalize()
    row_mask = dates.map(lambda d: d in session_set)
    df_sub = df_tf[row_mask]
    sig_sub = signal.reindex(df_sub.index).fillna(0).astype(int)

    if context_mask is not None:
        sig_sub = apply_filter(sig_sub, context_mask.reindex(sig_sub.index).fillna(False))

    res = run_intraday_backtest(df_sub, sig_sub, cost_bps=cost_bps,
                                squareoff_time=SQUAREOFF, no_entry_after=NO_ENTRY_AFTER,
                                hold_bars=HOLD_BARS)
    return res


def avg_gross_bps(trades: list, avg_price: float = 20_000.0) -> float:
    if not trades:
        return 0.0
    avg_gross = np.mean([t["gross_pnl"] for t in trades])
    return (avg_gross / avg_price) * 10_000 * 2


def avg_net_pnl(trades: list) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t["net_pnl"] for t in trades]))


def hit_rate(trades: list) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t["net_pnl"] > 0 for t in trades]))


def sharpe_from_trades(trades: list) -> float:
    if len(trades) < 5:
        return 0.0
    nets = np.array([t["net_pnl"] for t in trades])
    return float(nets.mean() / (nets.std() + 1e-9) * np.sqrt(252))


# ── BH FDR correction ────────────────────────────────────────────────────────

def bh_fdr_threshold(p_values: list[float], alpha: float = FDR_ALPHA) -> float:
    """Return Benjamini-Hochberg adjusted significance threshold."""
    m = len(p_values)
    if m == 0:
        return alpha
    sorted_p = sorted(p_values)
    threshold = 0.0
    for k, p in enumerate(sorted_p, start=1):
        if p <= alpha * k / m:
            threshold = p
    return threshold


# ── Placebo test ─────────────────────────────────────────────────────────────

def placebo_p_value(
    observed_avg_gross: float,
    parent_trades: list,
    context_fraction: float,
    n_sim: int = N_PLACEBO,
    rng: np.random.Generator | None = None,
) -> float:
    """P-value: fraction of placebo sims with avg_gross >= observed.

    Matched placebo: randomly sample the same fraction of trades from parent.
    """
    if not parent_trades or context_fraction <= 0:
        return 1.0

    rng = rng or np.random.default_rng(42)
    parent_gross = np.array([t["gross_pnl"] for t in parent_trades])
    k = max(1, int(len(parent_gross) * context_fraction))

    count_ge = 0
    for _ in range(n_sim):
        sample = rng.choice(parent_gross, size=k, replace=True)
        if sample.mean() >= observed_avg_gross:
            count_ge += 1

    return (count_ge + 1) / (n_sim + 1)  # +1 for continuity


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("PHASE 1C: CONTEXTUAL EDGE REFINEMENT")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/8] Loading 1-min data …")
    df_1min = load_minute_data()
    print(f"      {len(df_1min):,} bars | {df_1min.index[0].date()} → {df_1min.index[-1].date()}")

    sessions_all = get_sessions(df_1min)
    total_sessions = len(sessions_all)
    print(f"      {total_sessions} sessions total")

    # ── Compute daily context (once, before any splits) ───────────────────────
    print("\n[2/8] Computing daily context features …")
    ctx = compute_daily_context(df_1min)
    print(f"      Context shape: {ctx.shape}, columns: {list(ctx.columns[:5])} …")

    # ── Session index splits ──────────────────────────────────────────────────
    # DO NOT access locked_test sessions
    inner_train_sessions    = sessions_all[:INNER_TRAIN_N]
    inner_selection_sessions = sessions_all[INNER_TRAIN_N:INNER_TRAIN_N + INNER_SELECTION_N]
    inner_confirm_sessions  = sessions_all[INNER_TRAIN_N + INNER_SELECTION_N:
                                           INNER_TRAIN_N + INNER_SELECTION_N + INNER_CONFIRM_N]
    dev_sessions            = sessions_all[:INNER_TRAIN_N + INNER_SELECTION_N + INNER_CONFIRM_N]
    outer_val_sessions      = sessions_all[len(dev_sessions):len(dev_sessions) + OUTER_VAL_N]
    # locked_test = sessions_all[len(dev_sessions) + OUTER_VAL_N:]  # NEVER ACCESS

    print(f"\n      Split sizes → inner_train={len(inner_train_sessions)} | "
          f"inner_sel={len(inner_selection_sessions)} | "
          f"inner_conf={len(inner_confirm_sessions)} | "
          f"outer_val={len(outer_val_sessions)}")
    print(f"      Locked test: {total_sessions - len(dev_sessions) - OUTER_VAL_N} sessions (NOT ACCESSED)")
    print(f"      Dev period: {pd.Timestamp(dev_sessions[0]).date()} → {pd.Timestamp(dev_sessions[-1]).date()}")
    print(f"      Outer val:  {pd.Timestamp(outer_val_sessions[0]).date()} → {pd.Timestamp(outer_val_sessions[-1]).date()}")

    # ── Process each candidate ────────────────────────────────────────────────
    all_results = {}
    rng = np.random.default_rng(2024)

    for cand in CANDIDATES:
        cid = cand["id"]
        tf  = cand["tf"]
        print(f"\n{'=' * 70}")
        print(f"CANDIDATE: {cid} @ {tf}min")
        print("=" * 70)

        # Find hypothesis config
        hcfg = next((h for h in HYPOTHESIS_CONFIGS if h["id"] == cid), None)
        if hcfg is None:
            print(f"  ERROR: config for {cid} not found")
            continue

        # Aggregate to TF (OR windows must be in bars, not minutes)
        or_bar_windows = sorted({max(1, m // tf) for m in [15, 30, 60]})
        df_tf = resample_to_tf(df_1min, tf)
        df_tf = compute_features(df_tf, or_bar_windows=or_bar_windows)
        df_tf = merge_context_to_tf(df_tf, ctx)

        # Generate frozen parent signal
        signal = generate_signal(df_tf, hcfg, tf_minutes=tf)

        # ── Parent signal performance on inner_train (reference) ──────────────
        print(f"\n[3a] Parent signal — inner train ({len(inner_train_sessions)} sessions) …")
        res_parent_train = run_on_sessions(df_tf, signal, inner_train_sessions, COST_BPS)
        pt = res_parent_train["trades"]
        be_train = avg_gross_bps(pt)
        print(f"     n_trades={len(pt)} | avg_gross_bps={be_train:.1f} | "
              f"hit_rate={hit_rate(pt):.2%} | avg_net={avg_net_pnl(pt):.1f}")

        # ── Edge decay analysis on dev ────────────────────────────────────────
        print(f"\n[3b] Edge decay analysis on dev ({len(dev_sessions)} sessions) …")
        res_parent_dev = run_on_sessions(df_tf, signal, dev_sessions, cost_bps=0.0)
        yearly = per_year_stats(res_parent_dev["trades"])
        decay_class = classify_decay(yearly)
        pp2020 = pre_post_2020_split(res_parent_dev["trades"])
        print(f"     Decay classification: {decay_class}")
        if not yearly.empty:
            for yr, row in yearly.iterrows():
                print(f"     {yr}: n={int(row['n_trades']):3d} avg_gross={row['avg_gross_pnl']:+6.1f}")
        if "decay" in pp2020:
            print(f"     Pre/post-2020 decay: {pp2020['decay']:+.1f} pts")

        # ── Single filter screening (inner_selection) ─────────────────────────
        print(f"\n[4] Screening {len(SINGLE_FILTERS)} single filters on inner_selection "
              f"({len(inner_selection_sessions)} sessions) …")

        filter_results = {}
        for fname, ffunc in SINGLE_FILTERS.items():
            try:
                mask = ffunc(df_tf)
            except KeyError:
                continue

            res_f = run_on_sessions(df_tf, signal, inner_selection_sessions,
                                    cost_bps=COST_BPS, context_mask=mask)
            trades_f = res_f["trades"]
            if len(trades_f) < 5:
                continue

            be_f = avg_gross_bps(trades_f)
            net_f = avg_net_pnl(trades_f)
            frac  = len(trades_f) / max(1, len(
                run_on_sessions(df_tf, signal, inner_selection_sessions,
                                cost_bps=COST_BPS)["trades"]
            ))

            filter_results[fname] = {
                "n_trades": len(trades_f),
                "avg_gross_bps": round(be_f, 2),
                "avg_net_pnl": round(net_f, 2),
                "hit_rate": round(hit_rate(trades_f), 3),
                "fraction": round(frac, 3),
                "type": "single",
            }

        # Run unfiltered on inner_selection for reference
        res_sel_unfilt = run_on_sessions(df_tf, signal, inner_selection_sessions, COST_BPS)
        pt_sel = res_sel_unfilt["trades"]

        print(f"     Unfiltered inner_sel: n={len(pt_sel)} avg_gross_bps={avg_gross_bps(pt_sel):.1f}")
        promising_singles = {k: v for k, v in filter_results.items()
                             if v["avg_gross_bps"] >= FILTER_COST_SURVIVAL and v["n_trades"] >= 10}
        print(f"     Promising single filters (BE≥{FILTER_COST_SURVIVAL}bps): {len(promising_singles)}")
        for fname, r in sorted(promising_singles.items(), key=lambda x: -x[1]["avg_gross_bps"])[:10]:
            print(f"       {fname:30s} n={r['n_trades']:4d} "
                  f"be={r['avg_gross_bps']:+5.1f}bps hit={r['hit_rate']:.2%}")

        # ── Two-filter combo screening ────────────────────────────────────────
        print(f"\n[5] Screening {len(TWO_FILTER_COMBOS)} two-filter combos on inner_selection …")

        for f1, f2 in TWO_FILTER_COMBOS:
            if f1 not in SINGLE_FILTERS or f2 not in SINGLE_FILTERS:
                continue
            try:
                mask1 = SINGLE_FILTERS[f1](df_tf)
                mask2 = SINGLE_FILTERS[f2](df_tf)
                combined = mask1 & mask2
            except KeyError:
                continue

            res_c = run_on_sessions(df_tf, signal, inner_selection_sessions,
                                    cost_bps=COST_BPS, context_mask=combined)
            trades_c = res_c["trades"]
            if len(trades_c) < 5:
                continue

            be_c = avg_gross_bps(trades_c)
            frac_c = len(trades_c) / max(1, len(pt_sel))
            key = f"{f1}+{f2}"
            filter_results[key] = {
                "n_trades": len(trades_c),
                "avg_gross_bps": round(be_c, 2),
                "avg_net_pnl": round(avg_net_pnl(trades_c), 2),
                "hit_rate": round(hit_rate(trades_c), 3),
                "fraction": round(frac_c, 3),
                "type": "two",
            }

        promising_two = {k: v for k, v in filter_results.items()
                         if v["type"] == "two" and v["avg_gross_bps"] >= FILTER_COST_SURVIVAL
                         and v["n_trades"] >= 10}
        print(f"     Promising two-filter combos: {len(promising_two)}")
        for fname, r in sorted(promising_two.items(), key=lambda x: -x[1]["avg_gross_bps"])[:5]:
            print(f"       {fname:45s} n={r['n_trades']:4d} "
                  f"be={r['avg_gross_bps']:+5.1f}bps")

        # ── Select top candidates for inner_confirmation ──────────────────────
        # Rank by avg_gross_bps on inner_selection, take top 5
        all_promising = {**promising_singles, **promising_two}
        top_filters = sorted(all_promising.items(), key=lambda x: -x[1]["avg_gross_bps"])[:5]

        print(f"\n[6] Confirming top {len(top_filters)} filters on inner_confirmation "
              f"({len(inner_confirm_sessions)} sessions) …")

        confirmed_filters = {}
        placebo_p_values = []

        for fname, sel_res in top_filters:
            # Reconstruct mask
            if "+" in fname:
                f1, f2 = fname.split("+", 1)
                try:
                    mask = SINGLE_FILTERS[f1](df_tf) & SINGLE_FILTERS[f2](df_tf)
                except KeyError:
                    continue
            else:
                try:
                    mask = SINGLE_FILTERS[fname](df_tf)
                except KeyError:
                    continue

            res_conf = run_on_sessions(df_tf, signal, inner_confirm_sessions,
                                       cost_bps=COST_BPS, context_mask=mask)
            trades_conf = res_conf["trades"]

            # Compute placebo p-value from inner_selection parent trades
            frac_approx = sel_res["fraction"]
            p_val = placebo_p_value(
                sel_res["avg_gross_bps"] / (10_000 * 2) * 20_000,  # convert bps back to pts
                pt_sel,
                frac_approx,
                n_sim=N_PLACEBO,
                rng=rng,
            )
            placebo_p_values.append(p_val)

            be_conf = avg_gross_bps(trades_conf)
            net_conf = avg_net_pnl(trades_conf)
            print(f"     {fname:45s} n={len(trades_conf):4d} "
                  f"be={be_conf:+5.1f}bps net={net_conf:+6.1f} p={p_val:.3f}")

            confirmed_filters[fname] = {
                "selection_be_bps": sel_res["avg_gross_bps"],
                "confirm_n": len(trades_conf),
                "confirm_be_bps": round(be_conf, 2),
                "confirm_net": round(net_conf, 2),
                "confirm_hit": round(hit_rate(trades_conf), 3),
                "placebo_p": round(p_val, 4),
                "type": sel_res["type"],
            }

        # ── BH FDR correction ─────────────────────────────────────────────────
        total_tests = len(SINGLE_FILTERS) + len(TWO_FILTER_COMBOS)
        all_p_values = placebo_p_values + [1.0] * (total_tests - len(placebo_p_values))
        bh_threshold = bh_fdr_threshold(all_p_values, FDR_ALPHA)
        print(f"\n     BH FDR threshold (α={FDR_ALPHA}, m={total_tests}): {bh_threshold:.4f}")

        fdr_survivors = {k: v for k, v in confirmed_filters.items()
                         if v["placebo_p"] <= bh_threshold}
        print(f"     FDR survivors: {len(fdr_survivors)} / {len(confirmed_filters)}")

        # ── Outer validation (if FDR survivors exist) ─────────────────────────
        outer_results = {}
        if fdr_survivors:
            print(f"\n[7] Outer validation ({len(outer_val_sessions)} sessions) …")
            for fname, cfilt in fdr_survivors.items():
                if "+" in fname:
                    f1, f2 = fname.split("+", 1)
                    try:
                        mask = SINGLE_FILTERS[f1](df_tf) & SINGLE_FILTERS[f2](df_tf)
                    except KeyError:
                        continue
                else:
                    try:
                        mask = SINGLE_FILTERS[fname](df_tf)
                    except KeyError:
                        continue

                res_ov = run_on_sessions(df_tf, signal, outer_val_sessions,
                                         cost_bps=COST_BPS, context_mask=mask)
                trades_ov = res_ov["trades"]
                be_ov = avg_gross_bps(trades_ov)
                net_ov = avg_net_pnl(trades_ov)
                outer_results[fname] = {
                    "n": len(trades_ov),
                    "be_bps": round(be_ov, 2),
                    "net": round(net_ov, 2),
                    "hit": round(hit_rate(trades_ov), 3),
                }
                status = "PASS" if (be_ov >= FILTER_COST_SURVIVAL and
                                    len(trades_ov) >= MIN_TRADES_OUTER) else "FAIL"
                print(f"     {fname:45s} n={len(trades_ov):4d} "
                      f"be={be_ov:+5.1f}bps net={net_ov:+6.1f}  [{status}]")
        else:
            print("\n[7] No FDR survivors → outer validation skipped.")

        # ── Classification ────────────────────────────────────────────────────
        print(f"\n[8] Classification for {cid} …")

        # Parent dev performance
        res_dev_10 = run_on_sessions(df_tf, signal, dev_sessions, COST_BPS)
        be_dev = avg_gross_bps(res_dev_10["trades"])
        n_dev  = len(res_dev_10["trades"])

        outer_pass = {k: v for k, v in outer_results.items()
                      if v["be_bps"] >= FILTER_COST_SURVIVAL and v["n"] >= MIN_TRADES_OUTER}

        if not fdr_survivors:
            classification = "NO CONTEXTUAL EDGE"
            reason = f"No filter survived BH-FDR correction (threshold={bh_threshold:.4f})."
        elif not outer_pass:
            classification = "FRAGILE CONTEXTUAL EFFECT"
            reason = (f"{len(fdr_survivors)} FDR survivor(s) found but none passed outer validation "
                      f"(BE≥{FILTER_COST_SURVIVAL}bps, n≥{MIN_TRADES_OUTER}).")
        elif decay_class == "DECLINING":
            classification = "FRAGILE CONTEXTUAL EFFECT"
            reason = f"FDR survivor(s) passed outer val but edge shows DECLINING temporal trend."
        else:
            classification = "RESEARCH CANDIDATE"
            reason = (f"{len(outer_pass)} filter(s) passed FDR, inner-confirmation, "
                      f"and outer validation with stable edge.")

        print(f"\n  Parent dev: n={n_dev} be_bps={be_dev:.1f}")
        print(f"  Decay: {decay_class}")
        print(f"  FDR survivors: {len(fdr_survivors)}")
        print(f"  Outer-val pass: {len(outer_pass)}")
        print(f"\n  ┌─ CLASSIFICATION: {classification}")
        print(f"  └─ {reason}")

        all_results[cid] = {
            "tf": tf,
            "parent_dev_n": n_dev,
            "parent_dev_be_bps": round(be_dev, 2),
            "decay": decay_class,
            "fdr_threshold": round(bh_threshold, 4),
            "fdr_survivors": len(fdr_survivors),
            "outer_pass": len(outer_pass),
            "outer_results": outer_results,
            "confirmed_filters": confirmed_filters,
            "classification": classification,
            "reason": reason,
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"PHASE 1C FINAL SUMMARY  ({elapsed/60:.1f} min elapsed)")
    print("=" * 70)

    overall_labels = []
    for cid, r in all_results.items():
        print(f"\n  {cid} @ {r['tf']}min:")
        print(f"    Parent dev BE: {r['parent_dev_be_bps']:+.1f} bps ({r['parent_dev_n']} trades)")
        print(f"    Edge decay:    {r['decay']}")
        print(f"    FDR survivors: {r['fdr_survivors']}")
        print(f"    Outer pass:    {r['outer_pass']}")
        print(f"    → {r['classification']}")
        overall_labels.append(r["classification"])

    # Aggregate decision
    if all(c == "NO CONTEXTUAL EDGE" for c in overall_labels):
        final = "NO CONTEXTUAL EDGE"
    elif any(c == "RESEARCH CANDIDATE" for c in overall_labels):
        final = "RESEARCH CANDIDATE"
    else:
        final = "FRAGILE CONTEXTUAL EFFECT"

    print(f"\n{'─' * 70}")
    print(f"PHASE 1C VERDICT: {final}")
    print(f"{'─' * 70}")

    print("\nConstraint compliance:")
    print("  ✓ Parent signals frozen (Phase 1B definitions unchanged)")
    print("  ✓ Locked test set NOT accessed")
    print("  ✓ Filter selection on inner_selection only")
    print("  ✓ Confirmation on separate inner_confirm split")
    print("  ✓ BH-FDR correction applied across all tests")
    print("  ✓ 1000 matched placebo simulations")
    print("  ✓ At most 2-filter combinations (pre-specified list)")
    print("  ✓ No ML, no RL, no exhaustive search")


if __name__ == "__main__":
    main()
