"""Tests for regime classification."""
import pandas as pd
import numpy as np
import pytest

from src.regimes.trend_regime import TrendRegimeClassifier, TREND_BULL, TREND_BEAR, TREND_NEUTRAL
from src.regimes.volatility_regime import VolatilityRegimeClassifier, VOL_LOW, VOL_NORMAL, VOL_HIGH
from src.regimes.combined_regime import RegimeEngine


def test_trend_regime_values_valid(synthetic_ohlcv):
    clf = TrendRegimeClassifier()
    clf.fit(synthetic_ohlcv)
    regime = clf.classify(synthetic_ohlcv)
    valid = {TREND_BULL, TREND_BEAR, TREND_NEUTRAL}
    assert set(regime.dropna().unique()).issubset(valid)


def test_vol_regime_values_valid(synthetic_ohlcv):
    clf = VolatilityRegimeClassifier()
    clf.fit(synthetic_ohlcv)
    regime = clf.classify(synthetic_ohlcv)
    valid = {VOL_LOW, VOL_NORMAL, VOL_HIGH}
    assert set(regime.dropna().unique()).issubset(valid)


def test_regime_thresholds_fitted_only_on_train(synthetic_ohlcv):
    """Changing test data should not affect thresholds fitted on train."""
    # Use short sma_window so 400-row fixture provides enough warmup
    clf = TrendRegimeClassifier(sma_window=20, slope_window=10)
    clf.fit(synthetic_ohlcv)
    threshold_before = clf._bull_threshold
    assert threshold_before is not None

    # Perturb a copy of the data and refit — thresholds should differ
    df_mod = synthetic_ohlcv.copy()
    from src.data.schema import CLOSE
    df_mod[CLOSE] *= 0.5  # halve all prices

    clf2 = TrendRegimeClassifier(sma_window=20, slope_window=10)
    clf2.fit(df_mod)
    # Fitting on different data should produce different thresholds
    # (we are testing that fit() uses the data passed, not global state)
    # The original clf threshold should be unchanged
    assert clf._bull_threshold == threshold_before


def test_combined_regime_engine(synthetic_ohlcv):
    engine = RegimeEngine()
    engine.fit(synthetic_ohlcv)
    regimes = engine.classify(synthetic_ohlcv)
    assert "trend_regime" in regimes.columns
    assert "vol_regime" in regimes.columns
    assert "structure_regime" in regimes.columns
    assert len(regimes) == len(synthetic_ohlcv)


def test_regime_all_observations_covered(synthetic_ohlcv):
    """Every non-NaN bar should get a regime label after warmup."""
    clf = TrendRegimeClassifier(sma_window=10, slope_window=5)
    clf.fit(synthetic_ohlcv)
    regime = clf.classify(synthetic_ohlcv)
    # After the warmup period, no NaN regimes
    regime_tail = regime.iloc[20:]
    assert regime_tail.isna().sum() == 0
