"""Maximum Favourable Excursion (MFE) and Maximum Adverse Excursion (MAE)."""
from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np

from src.data.schema import HIGH, LOW, OPEN, CLOSE


def compute_excursions(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    max_horizon: int = 20,
    direction: str = "long",
) -> pd.DataFrame:
    """
    For each signal date T, compute the MFE and MAE experienced from T+1 open
    over the subsequent `max_horizon` sessions.

    direction: 'long' or 'short'
    Returns a DataFrame indexed by signal dates with columns:
        mfe, mae (both positive fractions)
    """
    signal_dates = df.index[signal_mask]
    results = []

    for date in signal_dates:
        loc = df.index.get_loc(date)
        entry_loc = loc + 1
        if entry_loc >= len(df):
            continue

        entry_price = df[OPEN].iloc[entry_loc]
        end_loc = min(entry_loc + max_horizon, len(df))

        window = df.iloc[entry_loc:end_loc]
        if window.empty:
            continue

        if direction == "long":
            excursion_high = (window[HIGH].max() - entry_price) / entry_price
            excursion_low = (window[LOW].min() - entry_price) / entry_price
            mfe = max(excursion_high, 0)
            mae = max(-excursion_low, 0)
        else:  # short
            excursion_low = (entry_price - window[LOW].min()) / entry_price
            excursion_high = (entry_price - window[HIGH].max()) / entry_price
            mfe = max(excursion_low, 0)
            mae = max(-excursion_high, 0)

        results.append({"date": date, "mfe": mfe, "mae": mae, "entry_price": entry_price})

    if not results:
        return pd.DataFrame(columns=["mfe", "mae", "entry_price"])

    out = pd.DataFrame(results).set_index("date")
    return out


def excursion_stats(exc: pd.DataFrame) -> dict:
    if exc.empty:
        return {}
    return {
        "mfe_mean": float(exc["mfe"].mean()),
        "mfe_median": float(exc["mfe"].median()),
        "mfe_p75": float(exc["mfe"].quantile(0.75)),
        "mae_mean": float(exc["mae"].mean()),
        "mae_median": float(exc["mae"].median()),
        "mae_p75": float(exc["mae"].quantile(0.75)),
        "mfe_gt_mae_pct": float((exc["mfe"] > exc["mae"]).mean()),
    }
