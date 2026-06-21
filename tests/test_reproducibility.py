"""Tests that repeated runs with the same seed produce identical results."""
import numpy as np
import pandas as pd
import pytest

from src.utils.random_state import seed_everything, get_rng
from src.backtesting.baselines import random_entry
from src.backtesting.costs import ZERO_COST


def test_random_entry_reproducible(synthetic_ohlcv):
    seed_everything(42)
    result1 = random_entry(synthetic_ohlcv, signal_frequency=0.3, seed=42, cost_profile=ZERO_COST)
    seed_everything(42)
    result2 = random_entry(synthetic_ohlcv, signal_frequency=0.3, seed=42, cost_profile=ZERO_COST)
    pd.testing.assert_series_equal(result1.returns, result2.returns)
    pd.testing.assert_series_equal(result1.equity_curve, result2.equity_curve)


def test_different_seeds_give_different_results(synthetic_ohlcv):
    result1 = random_entry(synthetic_ohlcv, signal_frequency=0.3, seed=42, cost_profile=ZERO_COST)
    result2 = random_entry(synthetic_ohlcv, signal_frequency=0.3, seed=99, cost_profile=ZERO_COST)
    assert not result1.returns.equals(result2.returns)


def test_rng_reproducible():
    rng1 = get_rng(42)
    rng2 = get_rng(42)
    arr1 = rng1.random(100)
    arr2 = rng2.random(100)
    np.testing.assert_array_equal(arr1, arr2)


def test_feature_calculations_deterministic(synthetic_ohlcv):
    """Feature computation should always produce the same result."""
    from src.features.registry import build_all_features
    f1 = build_all_features(synthetic_ohlcv)
    f2 = build_all_features(synthetic_ohlcv)
    pd.testing.assert_frame_equal(f1, f2)
