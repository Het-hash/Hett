"""Edge candidate scoring and ranking with transparent component scores."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def score_candidate(result: dict, oos_result: Optional[dict] = None) -> dict:
    """
    Compute a transparent composite score for a candidate edge.

    Components (each 0–1, weighted sum):
    1. OOS expected value (net of costs)
    2. OOS hit rate advantage over 50%
    3. Walk-forward fold consistency
    4. Cost survival (performance at 25bps vs 0bps)
    5. Sample size adequacy
    6. Drawdown score
    7. Turnover penalty
    8. Simplicity bonus (fewer params = higher)
    9. Train/test degradation penalty
    10. Baseline improvement

    Returns dict with all components and total_score.
    """
    components = {}

    # 1. OOS expected value — normalise around 0; cap at 2% per trade
    ev = float((oos_result or result).get("mean", 0.0))
    components["oos_ev_score"] = float(np.clip(ev / 0.02, -1, 1) * 0.5 + 0.5)

    # 2. Hit rate
    hr = float((oos_result or result).get("hit_rate", 0.5))
    components["hit_rate_score"] = float(np.clip((hr - 0.5) / 0.2 + 0.5, 0, 1))

    # 3. Sample size adequacy (>= 100 ideal, 30 minimum)
    n = int((oos_result or result).get("n", 0))
    components["sample_score"] = float(np.clip((n - 30) / 70, 0, 1))

    # 4. Cost survival: ev at 25bps relative to ev at 0bps
    ev_25 = float(result.get("ev_at_25bps", ev))
    ev_0 = float(result.get("ev_at_0bps", ev))
    if ev_0 != 0:
        components["cost_survival_score"] = float(np.clip(ev_25 / ev_0, 0, 1))
    else:
        components["cost_survival_score"] = 0.5

    # 5. Drawdown score (lower drawdown = better)
    max_dd = abs(float(result.get("max_drawdown", 0.15)))
    components["drawdown_score"] = float(np.clip(1 - max_dd / 0.20, 0, 1))

    # 6. Turnover penalty
    turnover = float(result.get("annual_turnover", 50))
    components["turnover_score"] = float(np.clip(1 - turnover / 200, 0, 1))

    # 7. Train/test degradation
    is_ev = float(result.get("mean", 0))
    oos_ev = float((oos_result or result).get("mean", 0))
    degradation = is_ev - oos_ev
    components["degradation_score"] = float(np.clip(1 - degradation / max(abs(is_ev), 1e-6), 0, 1))

    # 8. Stability across subperiods
    period_consistency = float(result.get("period_consistency", 0.5))
    components["stability_score"] = period_consistency

    # 9. Baseline improvement
    baseline_mean = float(result.get("baseline_mean", 0))
    improvement = oos_ev - baseline_mean
    components["baseline_improvement_score"] = float(np.clip(improvement / max(abs(baseline_mean), 1e-6) * 0.5 + 0.5, 0, 1))

    # 10. Simplicity (1 = no free params, 0 = many params)
    n_params = int(result.get("n_free_params", 2))
    components["simplicity_score"] = float(np.clip(1 - (n_params - 1) / 5, 0, 1))

    # Weighted total
    weights = {
        "oos_ev_score": 0.20,
        "hit_rate_score": 0.10,
        "sample_score": 0.10,
        "cost_survival_score": 0.15,
        "drawdown_score": 0.10,
        "turnover_score": 0.05,
        "degradation_score": 0.10,
        "stability_score": 0.10,
        "baseline_improvement_score": 0.05,
        "simplicity_score": 0.05,
    }

    total = sum(components[k] * w for k, w in weights.items())
    return {**components, "total_score": round(total, 4), "weights": weights}


def rank_candidates(results: List[dict]) -> pd.DataFrame:
    """
    Rank a list of candidate result dicts.
    Returns DataFrame sorted by total_score descending.
    """
    rows = []
    for r in results:
        scores = score_candidate(r, r.get("oos_stats"))
        row = {
            "experiment_id": r.get("experiment_id", "?"),
            "description": r.get("description", ""),
            "n_oos": r.get("n_oos", 0),
            "ev_oos": r.get("ev_oos", float("nan")),
            **{k: v for k, v in scores.items() if k != "weights"},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    return df
