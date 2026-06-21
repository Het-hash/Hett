"""Performance metrics calculation."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    positions: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.065,  # ~Indian repo rate
    periods_per_year: int = 252,
) -> dict:
    """
    Comprehensive performance metrics.
    All metrics derived from the net daily return series.
    """
    ret = returns.dropna()
    eq = equity.dropna()

    if len(ret) < 2:
        return {"error": "Insufficient data"}

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return = float((eq.iloc[-1] / eq.iloc[0]) - 1)
    n_years = len(ret) / periods_per_year
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 1e-6)) - 1)
    ann_vol = float(ret.std() * np.sqrt(periods_per_year))

    # ── Drawdown ──────────────────────────────────────────────────────────────
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    # Drawdown duration
    in_dd = drawdown < 0
    dd_streaks = (in_dd != in_dd.shift()).cumsum()
    dd_durations = in_dd.groupby(dd_streaks).sum()
    max_dd_duration = int(dd_durations.max()) if in_dd.any() else 0

    # ── Risk-adjusted ratios ──────────────────────────────────────────────────
    daily_rf = risk_free_rate / periods_per_year
    excess = ret - daily_rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(periods_per_year)) if excess.std() > 0 else 0.0

    downside = ret[ret < daily_rf]
    sortino_denom = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 else 1e-6
    sortino = float((cagr - risk_free_rate) / sortino_denom) if sortino_denom > 0 else 0.0

    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # ── Trade metrics ─────────────────────────────────────────────────────────
    trade_metrics = {}
    if trades is not None and not trades.empty and "net_return" in trades.columns:
        tr = trades["net_return"]
        wins = tr[tr > 0]
        losses = tr[tr <= 0]
        trade_metrics = {
            "trade_count": len(tr),
            "win_rate": float((tr > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
            "expectancy": float(tr.mean()),
            "profit_factor": float(wins.sum() / (-losses.sum())) if losses.sum() != 0 else float("inf"),
            "avg_holding_days": float(trades.get("holding_days", pd.Series([0])).mean()),
            "best_trade": float(tr.max()),
            "worst_trade": float(tr.min()),
        }
        # Losing streak
        losing_streak = _max_streak(tr <= 0)
        trade_metrics["longest_losing_streak"] = losing_streak

    # ── Exposure / Turnover ───────────────────────────────────────────────────
    exposure = float(positions.abs().mean())
    pos_changes = positions.diff().abs().sum()
    annual_turnover = float(pos_changes / max(n_years, 1e-6))

    # ── Tail risk ─────────────────────────────────────────────────────────────
    var_95 = float(ret.quantile(0.05))
    es_95 = float(ret[ret <= var_95].mean()) if (ret <= var_95).any() else float("nan")

    metrics = {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "ann_volatility": round(ann_vol, 4),
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_duration_days": max_dd_duration,
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "exposure_avg": round(exposure, 3),
        "annual_turnover": round(annual_turnover, 1),
        "var_95": round(var_95, 4),
        "expected_shortfall_95": round(es_95, 4),
        "n_days": len(ret),
        "n_years": round(n_years, 2),
        **trade_metrics,
    }
    return metrics


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    """Build a year × month table of monthly returns."""
    monthly = equity.resample("ME").last().pct_change().dropna()
    monthly.index = pd.to_datetime(monthly.index)
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    table.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ][:len(table.columns)]
    return table.round(4)


def annual_returns_table(equity: pd.Series) -> pd.DataFrame:
    """Annual return summary."""
    yearly = equity.resample("YE").last().pct_change().dropna()
    yearly.index = yearly.index.year
    yearly.name = "annual_return"
    return yearly.to_frame()


def _max_streak(condition: pd.Series) -> int:
    """Length of the longest consecutive True streak."""
    streak = condition.astype(int)
    groups = (streak != streak.shift()).cumsum()
    return int(streak.groupby(groups).sum().max())
