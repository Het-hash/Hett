"""Chart generation for Phase 1 report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.paths import figures_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (12, 5)


def _save(fig: plt.Figure, name: str, base: Optional[Path] = None) -> Path:
    out_dir = base or figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved chart: {path}")
    return path


def plot_equity_curves(
    curves: Dict[str, pd.Series],
    title: str = "Equity Curves",
    base: Optional[Path] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, eq in curves.items():
        ax.plot(eq.index, eq / eq.iloc[0], label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Normalised Equity (start=1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, "equity_curves", base)


def plot_drawdown(equity: pd.Series, label: str = "", base: Optional[Path] = None) -> Path:
    from src.risk.drawdown import drawdown_series
    dd = drawdown_series(equity) * 100
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(dd.index, dd, 0, alpha=0.5, color="red", label=label)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, f"drawdown_{label.replace(' ', '_')}", base)


def plot_monthly_heatmap(equity: pd.Series, title: str = "Monthly Returns", base: Optional[Path] = None) -> Path:
    from src.backtesting.metrics import monthly_returns_table
    table = monthly_returns_table(equity) * 100
    fig, ax = plt.subplots(figsize=(14, max(4, len(table) * 0.5)))
    sns.heatmap(
        table, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Return (%)"}
    )
    ax.set_title(title)
    return _save(fig, "monthly_heatmap", base)


def plot_return_distribution(returns: pd.Series, label: str = "", base: Optional[Path] = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(returns.dropna() * 100, bins=60, edgecolor="white", color="steelblue", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(float(returns.mean() * 100), color="red", linestyle="-", linewidth=1.2, label=f"Mean={returns.mean()*100:.2f}%")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Return Distribution — {label}")
    ax.legend()
    return _save(fig, f"return_dist_{label.replace(' ', '_')}", base)


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    label: str = "",
    base: Optional[Path] = None,
) -> Path:
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sr = (rolling_mean / rolling_std * np.sqrt(252)).dropna()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(rolling_sr.index, rolling_sr, label=f"{window}d Rolling Sharpe", linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.axhline(1, color="green", linestyle=":", linewidth=0.7, label="Sharpe=1")
    ax.set_title(f"Rolling Sharpe Ratio — {label}")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, f"rolling_sharpe_{label.replace(' ', '_')}", base)


def plot_parameter_heatmap(
    sweep_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "mean",
    title: str = "Parameter Heatmap",
    base: Optional[Path] = None,
) -> Path:
    if sweep_df.empty or param_x not in sweep_df or param_y not in sweep_df:
        logger.warning("Cannot plot parameter heatmap — insufficient data")
        return None

    pivot = sweep_df.pivot_table(values=metric, index=param_y, columns=param_x)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot * 100, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title(f"{title} — {metric} (%)")
    return _save(fig, f"param_heatmap_{title.replace(' ', '_')}", base)


def plot_cost_sensitivity(
    cost_df: pd.DataFrame,
    metric: str = "cagr",
    label: str = "",
    base: Optional[Path] = None,
) -> Path:
    if cost_df.empty or metric not in cost_df:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_df.index, cost_df[metric] * 100, marker="o", label=metric)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Round-trip Cost (bps)")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"Cost Sensitivity — {label}")
    ax.legend()
    return _save(fig, f"cost_sensitivity_{label.replace(' ', '_')}", base)


def plot_regime_comparison(
    regime_stats: pd.DataFrame,
    metric: str = "mean",
    title: str = "Performance by Regime",
    base: Optional[Path] = None,
) -> Path:
    if regime_stats.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(regime_stats))
    bars = ax.bar(x, regime_stats[metric] * 100 if metric in regime_stats else 0,
                  color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_stats.index.astype(str), rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(title)
    return _save(fig, f"regime_comparison_{title.replace(' ', '_')}", base)
