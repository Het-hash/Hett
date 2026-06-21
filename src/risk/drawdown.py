"""Drawdown analysis utilities."""
from __future__ import annotations

import pandas as pd
import numpy as np


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return percentage drawdown from running peak."""
    peak = equity.cummax()
    return (equity - peak) / peak


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def drawdown_table(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Find top N drawdown episodes with start, trough, end, and depth."""
    dd = drawdown_series(equity)
    in_dd = dd < 0
    rows = []
    started = None
    for date, val in dd.items():
        if val < 0 and started is None:
            started = date
        elif val == 0 and started is not None:
            episode = dd[started:date]
            rows.append({
                "start": started,
                "trough": episode.idxmin(),
                "end": date,
                "depth": float(episode.min()),
                "duration_days": (date - started).days,
            })
            started = None
    if started is not None:
        episode = dd[started:]
        rows.append({
            "start": started,
            "trough": episode.idxmin(),
            "end": dd.index[-1],
            "depth": float(episode.min()),
            "duration_days": (dd.index[-1] - started).days,
        })
    result = pd.DataFrame(rows).sort_values("depth").head(top_n)
    return result
