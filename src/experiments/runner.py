"""Experiment runner: iterates hypothesis library, evaluates, and records."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.edge_discovery.conditions import (
    momentum_signal, mean_reversion_zscore_signal, mean_reversion_rsi_signal,
    consecutive_down_signal, breakout_signal, ma_cross_trend_signal,
    pullback_in_trend_signal, gap_signal, month_end_signal, day_of_week_signal,
)
from src.edge_discovery.forward_returns import compute_forward_returns, forward_return_stats
from src.edge_discovery.excursions import compute_excursions, excursion_stats
from src.edge_discovery.statistics import (
    unconditional_stats, effect_size, block_bootstrap_ci, by_year_stats, by_regime_stats
)
from src.edge_discovery.multiple_testing import benjamini_hochberg
from src.edge_discovery.stability import parameter_sweep, subperiod_stability
from src.backtesting.engine import run_backtest
from src.backtesting.metrics import compute_metrics
from src.backtesting.costs import ETF_CONSERVATIVE, flat_cost_profile
from src.validation.sensitivity import cost_sensitivity
from src.experiments.registry import ExperimentRegistry, _data_hash
from src.utils.logging import get_logger
from src.utils.random_state import seed_everything

logger = get_logger(__name__)

MIN_OBS = 30
HORIZONS = [1, 3, 5, 10, 20]
PRIMARY_HORIZON = 5
COST_SWEEP_BPS = [0, 10, 25, 50]


def _acceptance_criteria(stats_oos: dict, n_min: int = MIN_OBS) -> tuple:
    """
    Return (passed: bool, reasons: list).
    A candidate passes only if ALL criteria are met.
    """
    reasons = []
    passed = True

    n = stats_oos.get("n", 0)
    if n < n_min:
        reasons.append(f"Insufficient OOS observations: {n} < {n_min}")
        passed = False

    ev = stats_oos.get("mean", 0)
    if ev <= 0:
        reasons.append(f"Non-positive OOS expected value: {ev:.4f}")
        passed = False

    hr = stats_oos.get("hit_rate", 0)
    if hr < 0.45:
        reasons.append(f"Hit rate below 45%: {hr:.2%}")
        passed = False

    return passed, reasons


def run_candidate_experiments(
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    regimes_dev: Optional[pd.DataFrame],
    regimes_val: Optional[pd.DataFrame],
    experiment_configs: list,
    registry: ExperimentRegistry,
    seed: int = 42,
) -> List[dict]:
    """
    Run all candidate experiments on dev (in-sample) and val (out-of-sample).
    Returns list of result dicts for ranking.
    """
    seed_everything(seed)
    all_results = []

    fwd_dev = compute_forward_returns(dev_df, HORIZONS)
    fwd_val = compute_forward_returns(val_df, HORIZONS)
    uncond = unconditional_stats(dev_df["close"].pct_change().dropna())

    data_hash = _data_hash(dev_df)
    split_boundaries = {
        "dev_start": str(dev_df.index[0].date()),
        "dev_end": str(dev_df.index[-1].date()),
        "val_start": str(val_df.index[0].date()),
        "val_end": str(val_df.index[-1].date()),
    }

    for cfg in experiment_configs:
        exp_id = cfg.get("id", "UNKNOWN")
        desc = cfg.get("description", "")
        logger.info(f"Running experiment: {exp_id} — {desc}")

        try:
            result = _run_single_experiment(
                exp_id=exp_id,
                desc=desc,
                cfg=cfg,
                dev_df=dev_df,
                val_df=val_df,
                fwd_dev=fwd_dev,
                fwd_val=fwd_val,
                regimes_dev=regimes_dev,
                regimes_val=regimes_val,
                uncond=uncond,
                split_boundaries=split_boundaries,
                data_hash=data_hash,
                registry=registry,
                seed=seed,
            )
            all_results.append(result)
        except Exception as exc:
            logger.warning(f"Experiment {exp_id} failed: {exc}")
            registry.log(
                experiment_id=exp_id,
                description=desc,
                feature_set=[],
                parameters=cfg,
                split_boundaries=split_boundaries,
                cost_profile="etf_conservative",
                seed=seed,
                metrics={},
                data_hash=data_hash,
                status="failed",
                rejection_reason=str(exc),
            )

    logger.info(f"Completed {len(all_results)}/{len(experiment_configs)} experiments")
    return all_results


def _build_signal(cfg: dict, df: pd.DataFrame, regimes: Optional[pd.DataFrame]) -> pd.Series:
    """Build a signal Series from an experiment config dict."""
    family = cfg.get("family", "")
    regime_filter = cfg.get("regime_filter")

    if family == "momentum":
        lookback = cfg.get("lookbacks", [20])[0] if "lookbacks" in cfg else cfg.get("lookback", 20)
        threshold = cfg.get("threshold", 0.0)
        return momentum_signal(df, lookback, threshold, regimes, regime_filter)

    if family == "mean_reversion":
        feature = cfg.get("feature", "zscore")
        if feature == "zscore":
            window = cfg.get("zscore_window", [20])[0] if isinstance(cfg.get("zscore_window"), list) else cfg.get("zscore_window", 20)
            threshold = cfg.get("thresholds", [-2.0])[0] if isinstance(cfg.get("thresholds"), list) else cfg.get("thresholds", -2.0)
            return mean_reversion_zscore_signal(df, window, threshold, "long", regimes, regime_filter)
        if feature == "rsi":
            window = cfg.get("windows", [14])[0] if isinstance(cfg.get("windows"), list) else cfg.get("windows", 14)
            threshold = cfg.get("thresholds", [30])[0] if isinstance(cfg.get("thresholds"), list) else cfg.get("thresholds", 30)
            return mean_reversion_rsi_signal(df, window, threshold, "long", regimes, regime_filter)
        if feature == "consecutive_down":
            streak = cfg.get("streaks", [3])[0] if isinstance(cfg.get("streaks"), list) else cfg.get("streaks", 3)
            return consecutive_down_signal(df, streak, regimes, regime_filter)

    if family == "breakout":
        lookback = cfg.get("lookbacks", [20])[0] if isinstance(cfg.get("lookbacks"), list) else cfg.get("lookbacks", 20)
        direction = cfg.get("direction", "long")
        return breakout_signal(df, lookback, direction, cfg.get("vol_condition"), regimes)

    if family == "trend":
        feature = cfg.get("feature", "ma_cross")
        if feature == "ma_cross":
            fast = cfg.get("fast_windows", [50])[0]
            slow = cfg.get("slow_windows", [200])[0]
            return ma_cross_trend_signal(df, fast, slow, cfg.get("direction", "long"))
        if feature == "pullback_in_trend":
            return pullback_in_trend_signal(df, cfg.get("trend_window", 200),
                                            cfg.get("pullback_window", [5])[0] if isinstance(cfg.get("pullback_window"), list) else cfg.get("pullback_window", 5),
                                            cfg.get("pullback_threshold", -0.02))

    if family == "gap":
        threshold = cfg.get("thresholds", [0.01])[0] if isinstance(cfg.get("thresholds"), list) else cfg.get("thresholds", 0.01)
        return gap_signal(df, threshold, cfg.get("direction", "long"))

    if family == "calendar":
        feature = cfg.get("feature", "day_of_week")
        if feature == "month_end":
            return month_end_signal(df)
        if feature == "day_of_week":
            day = cfg.get("days", [0])[0]
            return day_of_week_signal(df, day)

    raise ValueError(f"Cannot build signal for family='{family}', feature='{cfg.get('feature')}'")


def _run_single_experiment(
    exp_id: str,
    desc: str,
    cfg: dict,
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fwd_dev: pd.DataFrame,
    fwd_val: pd.DataFrame,
    regimes_dev: Optional[pd.DataFrame],
    regimes_val: Optional[pd.DataFrame],
    uncond: dict,
    split_boundaries: dict,
    data_hash: str,
    registry: ExperimentRegistry,
    seed: int,
) -> dict:
    direction = cfg.get("direction", "long")
    fwd_col = f"fwd_open_{PRIMARY_HORIZON}d"

    # Build signals
    sig_dev = _build_signal(cfg, dev_df, regimes_dev).fillna(False).astype(bool)
    sig_val = _build_signal(cfg, val_df, regimes_val).fillna(False).astype(bool)

    if sig_dev.sum() < MIN_OBS:
        msg = f"Only {sig_dev.sum()} dev signals < {MIN_OBS} minimum"
        registry.log(exp_id, desc, [], cfg, split_boundaries, "etf_conservative",
                     seed, {}, data_hash, "rejected", msg)
        return {"experiment_id": exp_id, "status": "rejected", "reason": msg}

    # In-sample stats
    stats_dev = forward_return_stats(sig_dev, fwd_dev, fwd_col)
    stats_dev["effect_size"] = effect_size(
        fwd_dev.loc[sig_dev, fwd_col].dropna(),
        fwd_dev[fwd_col].dropna()
    )
    stats_dev["by_year"] = by_year_stats(sig_dev, fwd_dev[fwd_col]).to_dict()

    # OOS stats
    stats_val = forward_return_stats(sig_val, fwd_val, fwd_col)

    # Acceptance check on val
    passed, reasons = _acceptance_criteria(stats_val)

    # Backtest on dev + val combined for equity curve
    full_df = pd.concat([dev_df, val_df])
    full_sig = pd.concat([sig_dev, sig_val])
    result = run_backtest(full_df, full_sig, direction=direction,
                          cost_profile=ETF_CONSERVATIVE, initial_capital=1_000_000)
    metrics = compute_metrics(result.returns, result.equity_curve, result.positions, result.trades)

    # Cost sensitivity on val only
    cs = cost_sensitivity(val_df, sig_val, direction)
    ev_at_25bps = float(
        forward_return_stats(sig_val, compute_forward_returns(val_df, [PRIMARY_HORIZON]), fwd_col).get("mean", 0)
    )

    status = "accepted" if passed else "rejected"
    rejection_reason = "; ".join(reasons) if not passed else None

    run_id = registry.log(
        experiment_id=exp_id,
        description=desc,
        feature_set=[cfg.get("feature", "")],
        parameters={k: v for k, v in cfg.items() if k not in ("id", "description")},
        split_boundaries=split_boundaries,
        cost_profile="etf_conservative",
        seed=seed,
        metrics=metrics,
        data_hash=data_hash,
        status=status,
        rejection_reason=rejection_reason,
    )

    return {
        "experiment_id": exp_id,
        "description": desc,
        "status": status,
        "rejection_reason": rejection_reason,
        "stats_dev": stats_dev,
        "stats_val": stats_val,
        "metrics": metrics,
        "cost_sensitivity": cs,
        "ev_at_25bps": ev_at_25bps,
        "run_id": run_id,
    }
