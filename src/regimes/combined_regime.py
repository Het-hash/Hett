"""Combined regime: builds all three regime dimensions into one DataFrame."""
from __future__ import annotations

import pandas as pd

from src.regimes.trend_regime import TrendRegimeClassifier
from src.regimes.volatility_regime import VolatilityRegimeClassifier
from src.regimes.structure_regime import StructureRegimeClassifier
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeEngine:
    """
    Fits regime classifiers on training data, then classifies any period.

    Usage:
        engine = RegimeEngine()
        engine.fit(train_df)
        regimes = engine.classify(full_df)
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        trend_cfg = cfg.get("trend", {})
        vol_cfg = cfg.get("volatility", {})
        struct_cfg = cfg.get("structure", {})

        self.trend_clf = TrendRegimeClassifier(
            sma_window=trend_cfg.get("sma_window", 200),
            slope_window=trend_cfg.get("slope_window", 63),
            bear_slope_pct=trend_cfg.get("bear_slope_pct", 33),
            bull_slope_pct=trend_cfg.get("bull_slope_pct", 67),
        )
        self.vol_clf = VolatilityRegimeClassifier(
            vol_window=vol_cfg.get("atr_window", 20),
            pct_window=vol_cfg.get("percentile_window", 252),
            low_pct=vol_cfg.get("low_pct", 33),
            high_pct=vol_cfg.get("high_pct", 67),
        )
        self.struct_clf = StructureRegimeClassifier(
            adx_window=struct_cfg.get("adx_window", 14),
            adx_threshold=struct_cfg.get("adx_trending_threshold", 25),
            eff_window=struct_cfg.get("efficiency_window", 20),
        )

    def fit(self, train_df: pd.DataFrame) -> "RegimeEngine":
        """Fit all classifiers on training data only."""
        self.trend_clf.fit(train_df)
        self.vol_clf.fit(train_df)
        logger.info("RegimeEngine fitted on training data")
        return self

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns: trend_regime, vol_regime, structure_regime."""
        regimes = pd.DataFrame(index=df.index)
        regimes["trend_regime"] = self.trend_clf.classify(df)
        regimes["vol_regime"] = self.vol_clf.classify(df)
        regimes["structure_regime"] = self.struct_clf.classify(df)
        return regimes

    def fit_classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and classify on the same df. Use for exploration only."""
        return self.fit(df).classify(df)

    def report(self, regimes: pd.DataFrame) -> dict:
        out = {}
        if "trend_regime" in regimes:
            out["trend"] = self.trend_clf.report(regimes["trend_regime"])
        if "vol_regime" in regimes:
            out["volatility"] = self.vol_clf.report(regimes["vol_regime"])
        if "structure_regime" in regimes:
            out["structure"] = self.struct_clf.report(regimes["structure_regime"])
        return out
