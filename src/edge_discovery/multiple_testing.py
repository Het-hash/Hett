"""Multiple testing correction and false discovery rate control."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def benjamini_hochberg(p_values: List[float], alpha: float = 0.10) -> List[bool]:
    """
    Benjamini-Hochberg FDR procedure.
    Returns a boolean list: True if hypothesis is rejected (significant after correction).
    """
    m = len(p_values)
    if m == 0:
        return []
    p = np.array(p_values)
    order = np.argsort(p)
    p_sorted = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    reject_sorted = np.zeros(m, dtype=bool)
    below = p_sorted <= thresholds
    if below.any():
        last_reject = np.where(below)[0].max()
        reject_sorted[: last_reject + 1] = True
    result = np.zeros(m, dtype=bool)
    result[order] = reject_sorted
    return result.tolist()


def bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Bonferroni correction."""
    m = len(p_values)
    threshold = alpha / m if m > 0 else alpha
    return [p <= threshold for p in p_values]


def deflated_sharpe_ratio(
    sr_hat: float,
    n_trials: int,
    sr_benchmark: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Adjusts for the expected max Sharpe from multiple trials.
    Returns the probability that the true SR > 0.
    """
    if n_trials <= 0 or n_obs <= 1:
        return float("nan")

    # Expected max SR under IID normal
    gamma = 0.5772156649  # Euler-Mascheroni constant
    sr_max = (1 - gamma) * scipy_stats.norm.ppf(1 - 1 / n_trials) + gamma * scipy_stats.norm.ppf(
        1 - 1 / (n_trials * np.e)
    )

    # Adjust for non-normality
    sr_adj = (
        (sr_hat - sr_max)
        * np.sqrt(n_obs - 1)
        / np.sqrt(1 - skewness * sr_hat + (kurtosis - 1) / 4 * sr_hat ** 2)
    )
    return float(scipy_stats.norm.cdf(sr_adj))


def count_effective_tests(experiment_configs: List[dict]) -> int:
    """Count total parameter combinations across all experiments."""
    total = 0
    for cfg in experiment_configs:
        n = 1
        for v in cfg.values():
            if isinstance(v, list):
                n *= len(v)
        total += n
    return total


def placebo_test(
    returns: pd.Series,
    signal_frequency: float,
    n_simulations: int = 1000,
    horizon: int = 5,
    seed: int = 42,
) -> dict:
    """
    Test whether a signal's performance exceeds random signals with the same frequency.
    Returns distribution of random mean returns and p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_signals = max(1, int(n * signal_frequency))
    results = []
    ret_arr = returns.dropna().values

    for _ in range(n_simulations):
        idx = rng.choice(len(ret_arr) - horizon, size=n_signals, replace=False)
        sim_rets = np.array([ret_arr[i : i + horizon].mean() for i in idx])
        results.append(sim_rets.mean())

    results = np.array(results)
    return {
        "random_mean": float(results.mean()),
        "random_std": float(results.std()),
        "random_p95": float(np.percentile(results, 95)),
        "n_simulations": n_simulations,
    }
