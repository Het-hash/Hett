"""Market structure regime: Trending / Sideways / Transitional using ADX."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import HIGH, LOW, CLOSE
from src.utils.logging import get_logger

logger = get_logger(__name__)

STRUCT_TRENDING = "trending"
STRUCT_SIDEWAYS = "sideways"
STRUCT_TRANSITIONAL = "transitional"


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder ADX — causal."""
    high = df[HIGH]
    low = df[LOW]
    close = df[CLOSE]

    up_move = high.diff()
    dn_move = -low.diff()

    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_w = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_w.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx_series.rename(f"adx_{window}")


def directional_efficiency(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Efficiency ratio: net displacement / path length over `window` bars.
    Close to 1 = strong trend; close to 0 = random/sideways.
    """
    net = (df[CLOSE] - df[CLOSE].shift(window)).abs()
    path = df[CLOSE].diff().abs().rolling(window).sum()
    return (net / path.replace(0, np.nan)).rename(f"efficiency_{window}")


class StructureRegimeClassifier:
    def __init__(self, adx_window: int = 14, adx_threshold: float = 25.0, eff_window: int = 20):
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold
        self.eff_window = eff_window

    def classify(self, df: pd.DataFrame) -> pd.Series:
        adx_vals = adx(df, self.adx_window)
        eff = directional_efficiency(df, self.eff_window)

        regime = pd.Series(STRUCT_TRANSITIONAL, index=df.index, name="structure_regime")
        regime[adx_vals >= self.adx_threshold] = STRUCT_TRENDING
        regime[(adx_vals < self.adx_threshold) & (eff < 0.3)] = STRUCT_SIDEWAYS
        return regime

    def report(self, regime: pd.Series) -> dict:
        counts = regime.value_counts()
        total = len(regime.dropna())
        result = {}
        for label in [STRUCT_TRENDING, STRUCT_SIDEWAYS, STRUCT_TRANSITIONAL]:
            n = int(counts.get(label, 0))
            result[label] = {"count": n, "pct": round(n / total * 100, 1) if total > 0 else 0}
        return result
