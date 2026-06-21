"""Statistical tests for edge discovery."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def unconditional_stats(returns: pd.Series) -> dict:
    """Statistics on unconditional (market) returns for comparison."""
    ret = returns.dropna()
    return {
        "n": len(ret),
        "mean": float(ret.mean()),
        "median": float(ret.median()),
        "std": float(ret.std()),
        "hit_rate": float((ret > 0).mean()),
        "skewness": float(ret.skew()),
        "worst": float(ret.min()),
        "best": float(ret.max()),
    }


def effect_size(signal_returns: pd.Series, all_returns: pd.Series) -> float:
    """Cohen's d: effect size of signal returns vs unconditional returns."""
    s = signal_returns.dropna()
    a = all_returns.dropna()
    if len(s) < 2 or len(a) < 2:
        return float("nan")
    pooled_std = np.sqrt((s.std() ** 2 + a.std() ** 2) / 2)
    if pooled_std == 0:
        return float("nan")
    return float((s.mean() - a.mean()) / pooled_std)


def block_bootstrap_ci(
    returns: pd.Series,
    statistic: str = "mean",
    block_size: int = 20,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """
    Block bootstrap confidence interval for a statistic on time-series returns.
    Uses overlapping blocks of size `block_size`.
    """
    rng = np.random.default_rng(seed)
    ret = returns.dropna().values
    n = len(ret)
    if n < block_size * 2:
        return (float("nan"), float("nan"))

    stat_vals = []
    for _ in range(n_bootstrap):
        # Draw block starts with replacement
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        sample = np.concatenate([ret[s : s + block_size] for s in starts])[:n]
        if statistic == "mean":
            stat_vals.append(sample.mean())
        elif statistic == "sharpe":
            s = sample.std()
            stat_vals.append(sample.mean() / s * np.sqrt(252) if s > 0 else 0.0)

    alpha = 1 - ci
    lo = float(np.percentile(stat_vals, alpha / 2 * 100))
    hi = float(np.percentile(stat_vals, (1 - alpha / 2) * 100))
    return lo, hi


def mann_whitney_test(signal_returns: pd.Series, all_returns: pd.Series) -> dict:
    """Non-parametric test: are signal returns drawn from a different distribution?"""
    s = signal_returns.dropna()
    a = all_returns.dropna()
    if len(s) < 5 or len(a) < 5:
        return {"u_stat": float("nan"), "p_value": float("nan")}
    u, p = scipy_stats.mannwhitneyu(s, a, alternative="two-sided")
    return {"u_stat": float(u), "p_value": float(p)}


def by_year_stats(
    signal_mask: pd.Series,
    fwd_returns: pd.Series,
) -> pd.DataFrame:
    """Per-year forward return statistics for a signal."""
    combined = pd.concat([signal_mask.rename("signal"), fwd_returns.rename("ret")], axis=1)
    combined = combined[combined["signal"]].dropna(subset=["ret"])
    combined["year"] = combined.index.year
    rows = []
    for yr, grp in combined.groupby("year"):
        r = grp["ret"]
        rows.append({
            "year": yr,
            "n": len(r),
            "mean": float(r.mean()),
            "hit_rate": float((r > 0).mean()),
            "total_return": float(r.sum()),
        })
    return pd.DataFrame(rows).set_index("year")


def by_regime_stats(
    signal_mask: pd.Series,
    fwd_returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Per-regime forward return statistics."""
    combined = pd.concat(
        [signal_mask.rename("signal"), fwd_returns.rename("ret"), regimes.rename("regime")],
        axis=1,
    )
    combined = combined[combined["signal"]].dropna(subset=["ret", "regime"])
    rows = []
    for regime_val, grp in combined.groupby("regime"):
        r = grp["ret"]
        rows.append({
            "regime": regime_val,
            "n": len(r),
            "mean": float(r.mean()),
            "hit_rate": float((r > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("regime")
