"""Trend regime classifier: Bull / Bear / Neutral."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import CLOSE
from src.utils.logging import get_logger

logger = get_logger(__name__)

TREND_BULL = "bull"
TREND_BEAR = "bear"
TREND_NEUTRAL = "neutral"


class TrendRegimeClassifier:
    """
    Classifies each day into Bull/Bear/Neutral using price relative to
    a long SMA and the slope of that SMA.

    Thresholds are fitted on training data (slope percentiles) and frozen
    for validation/test to avoid look-ahead.
    """

    def __init__(
        self,
        sma_window: int = 200,
        slope_window: int = 63,
        bear_slope_pct: float = 33.0,
        bull_slope_pct: float = 67.0,
    ):
        self.sma_window = sma_window
        self.slope_window = slope_window
        self.bear_slope_pct = bear_slope_pct
        self.bull_slope_pct = bull_slope_pct

        # Fitted thresholds (set by fit())
        self._bear_threshold: Optional[float] = None
        self._bull_threshold: Optional[float] = None

    def _compute_slope(self, series: pd.Series) -> pd.Series:
        ma = series.rolling(self.sma_window).mean()
        slope = (ma - ma.shift(self.slope_window)) / ma.shift(self.slope_window)
        return slope

    def fit(self, df: pd.DataFrame) -> "TrendRegimeClassifier":
        """Fit slope percentile thresholds on training data."""
        slope = self._compute_slope(df[CLOSE])
        valid_slope = slope.dropna()
        self._bear_threshold = float(np.percentile(valid_slope, self.bear_slope_pct))
        self._bull_threshold = float(np.percentile(valid_slope, self.bull_slope_pct))
        logger.info(
            f"TrendRegime fitted: bear_slope≤{self._bear_threshold:.4f}, "
            f"bull_slope≥{self._bull_threshold:.4f}"
        )
        return self

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a Series with values TREND_BULL / TREND_BEAR / TREND_NEUTRAL.
        Uses only data available at each bar's close.
        Requires fit() to have been called first.
        """
        if self._bear_threshold is None:
            raise RuntimeError("Call fit() before classify()")

        slope = self._compute_slope(df[CLOSE])
        ma = df[CLOSE].rolling(self.sma_window).mean()
        above_ma = df[CLOSE] > ma

        regime = pd.Series(TREND_NEUTRAL, index=df.index, name="trend_regime")
        regime[above_ma & (slope >= self._bull_threshold)] = TREND_BULL
        regime[~above_ma & (slope <= self._bear_threshold)] = TREND_BEAR

        return regime

    def fit_classify(self, df: pd.DataFrame) -> pd.Series:
        """Convenience: fit on full df, then classify. USE ONLY FOR EXPLORATION."""
        return self.fit(df).classify(df)

    def report(self, regime: pd.Series) -> dict:
        counts = regime.value_counts()
        total = len(regime.dropna())
        result = {}
        for label in [TREND_BULL, TREND_BEAR, TREND_NEUTRAL]:
            n = int(counts.get(label, 0))
            result[label] = {"count": n, "pct": round(n / total * 100, 1) if total > 0 else 0}
        return result
