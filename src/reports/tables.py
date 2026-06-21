"""Table generation for Phase 1 report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tabulate import tabulate

from src.utils.paths import tables_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _save_csv(df: pd.DataFrame, name: str, base: Optional[Path] = None) -> Path:
    out_dir = base or tables_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path)
    return path


def metrics_table(metrics_dict: Dict[str, dict], base: Optional[Path] = None) -> pd.DataFrame:
    """Build a comparison table from a dict of {strategy_name: metrics_dict}."""
    rows = []
    key_metrics = [
        "cagr", "ann_volatility", "max_drawdown", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "win_rate", "expectancy",
        "trade_count", "exposure_avg", "annual_turnover",
    ]
    for strategy, m in metrics_dict.items():
        row = {"strategy": strategy}
        for k in key_metrics:
            v = m.get(k, float("nan"))
            if k in ("cagr", "ann_volatility", "max_drawdown", "win_rate", "expectancy"):
                row[k] = f"{v*100:.2f}%" if isinstance(v, float) else v
            else:
                row[k] = f"{v:.2f}" if isinstance(v, float) else v
        rows.append(row)

    df = pd.DataFrame(rows).set_index("strategy")
    if base:
        _save_csv(df, "metrics_comparison", base)
    return df


def forward_return_table(stats: Dict[str, dict], base: Optional[Path] = None) -> pd.DataFrame:
    """Build a table showing forward return stats for multiple experiments."""
    rows = []
    for exp_id, s in stats.items():
        row = {
            "experiment_id": exp_id,
            "n": s.get("n", 0),
            "signal_freq_%": s.get("signal_frequency_pct", float("nan")),
            "mean_%": round(s.get("mean", 0) * 100, 3),
            "median_%": round(s.get("median", 0) * 100, 3),
            "hit_rate": round(s.get("hit_rate", 0), 3),
            "profit_factor": round(s.get("profit_factor", 0), 2),
            "t_stat": round(s.get("t_stat", float("nan")), 2),
            "p_value": round(s.get("p_value", float("nan")), 3),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("experiment_id")
    if base:
        _save_csv(df, "forward_return_stats", base)
    return df


def annual_returns_table(
    equity_curves: Dict[str, pd.Series],
    base: Optional[Path] = None,
) -> pd.DataFrame:
    """Annual return table for multiple strategies."""
    from src.backtesting.metrics import annual_returns_table as _art
    dfs = {}
    for name, eq in equity_curves.items():
        annual = _art(eq)
        dfs[name] = annual["annual_return"]
    df = pd.DataFrame(dfs) * 100
    df.index.name = "year"
    if base:
        _save_csv(df, "annual_returns", base)
    return df.round(2)


def candidate_ranking_table(results: List[dict], base: Optional[Path] = None) -> pd.DataFrame:
    from src.edge_discovery.ranking import rank_candidates
    df = rank_candidates(results)
    if base and not df.empty:
        _save_csv(df, "candidate_ranking", base)
    return df


def print_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
