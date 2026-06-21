"""Tests for cost model."""
import pytest
from src.backtesting.costs import (
    CostProfile, ZERO_COST, ETF_CONSERVATIVE, flat_cost_profile, get_profile
)


def test_zero_cost_is_zero():
    assert ZERO_COST.round_trip_cost(1_000_000) == 0.0


def test_etf_conservative_positive():
    cost = ETF_CONSERVATIVE.round_trip_cost(1_000_000)
    assert cost > 0


def test_flat_cost_profile_bps():
    profile = flat_cost_profile(20)
    # flat_cost_profile(20) sets half=10bps per side for brokerage and spread.
    # round_trip_cost adds 2 * (brokerage + spread) = 2 * (0.001 + 0.001) = 0.004
    cost = profile.round_trip_cost(100)
    assert abs(cost - 0.004) < 1e-6


def test_round_trip_greater_than_one_way():
    cost_rt = ETF_CONSERVATIVE.round_trip_cost(1_000_000)
    cost_ow = ETF_CONSERVATIVE.one_way_cost("entry")
    assert cost_rt > cost_ow


def test_get_profile():
    p = get_profile("zero")
    assert p.label == "zero"
    with pytest.raises(ValueError):
        get_profile("nonexistent_profile")
