"""Volatility regime classifier: Low / Normal / High."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import CLOSE
from src.features.volatility import realised_vol
from src.utils.logging import get_logger

logger = get_logger(__name__)

VOL_LOW = "low"
VOL_NORMAL = "normal"
VOL_HIGH = "high"


class VolatilityRegimeClassifier:
    """
    Classifies each day into Low/Normal/High volatility using rolling
    realised volatility percentile rank fitted on training data.
    """

    def __init__(
        self,
        vol_window: int = 20,
        pct_window: int = 252,
        low_pct: float = 33.0,
        high_pct: float = 67.0,
    ):
        self.vol_window = vol_window
        self.pct_window = pct_window
        self.low_pct = low_pct
        self.high_pct = high_pct

        self._low_threshold: Optional[float] = None
        self._high_threshold: Optional[float] = None

    def _compute_vol(self, df: pd.DataFrame) -> pd.Series:
        return realised_vol(df, self.vol_window)

    def fit(self, df: pd.DataFrame) -> "VolatilityRegimeClassifier":
        """Fit absolute vol thresholds from training data percentiles."""
        vol = self._compute_vol(df).dropna()
        self._low_threshold = float(np.percentile(vol, self.low_pct))
        self._high_threshold = float(np.percentile(vol, self.high_pct))
        logger.info(
            f"VolRegime fitted: low≤{self._low_threshold:.4f}, "
            f"high≥{self._high_threshold:.4f} (ann. vol)"
        )
        return self

    def classify(self, df: pd.DataFrame) -> pd.Series:
        if self._low_threshold is None:
            raise RuntimeError("Call fit() before classify()")
        vol = self._compute_vol(df)
        regime = pd.Series(VOL_NORMAL, index=df.index, name="vol_regime")
        regime[vol <= self._low_threshold] = VOL_LOW
        regime[vol >= self._high_threshold] = VOL_HIGH
        return regime

    def fit_classify(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).classify(df)

    def report(self, regime: pd.Series) -> dict:
        counts = regime.value_counts()
        total = len(regime.dropna())
        result = {}
        for label in [VOL_LOW, VOL_NORMAL, VOL_HIGH]:
            n = int(counts.get(label, 0))
            result[label] = {"count": n, "pct": round(n / total * 100, 1) if total > 0 else 0}
        return result
