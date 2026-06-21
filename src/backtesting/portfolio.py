"""Portfolio accounting helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_sizing(
    signal: pd.Series,
    df: pd.DataFrame,
    method: str = "full",
    vol_window: int = 20,
    target_vol: float = 0.15,
    atr_window: int = 14,
    risk_per_trade: float = 0.01,
    capital: float = 1_000_000.0,
    max_position: float = 1.0,
) -> pd.Series:
    """
    Convert a boolean signal into a position size fraction [0, max_position].

    Methods:
    - 'full': binary 0 or 1 allocation
    - 'fixed_frac': fixed fraction (0.5)
    - 'inv_vol': inverse-volatility sizing targeting target_vol
    - 'atr': ATR-based risk sizing
    """
    if method == "full":
        return signal.astype(float).clip(0, max_position)

    if method == "fixed_frac":
        return (signal.astype(float) * 0.5).clip(0, max_position)

    if method == "inv_vol":
        from src.features.volatility import realised_vol
        from src.data.schema import CLOSE

        # Compute position size from data available at T (use shift to avoid look-ahead)
        rv = realised_vol(df, vol_window)
        rv = rv.shift(1)  # use prior session's vol estimate
        size = (target_vol / rv.replace(0, np.nan)).clip(0, max_position)
        return (signal.astype(float) * size).clip(0, max_position)

    if method == "atr":
        from src.features.volatility import atr as compute_atr
        from src.data.schema import CLOSE

        atr_vals = compute_atr(df, atr_window).shift(1)
        price = df[CLOSE].shift(1)
        # Units: risk_per_trade capital / ATR, expressed as fraction of portfolio
        frac = (capital * risk_per_trade) / (atr_vals * price * capital)
        size = frac.clip(0, max_position)
        return (signal.astype(float) * size).clip(0, max_position)

    raise ValueError(f"Unknown sizing method: {method}")
