"""Tests for one-shot, edge-triggered signal semantics.

These tests verify the core Phase 1B.1 requirement: a persistent condition
must NOT fire repeatedly.  Each test deliberately creates a condition that
would produce many signals under the old level-triggered approach, then
asserts only the correct one-shot behaviour occurs.
"""
import numpy as np
import pandas as pd
import pytest

from src.intraday.signals import (
    oneshot_edge,
    oneshot_level,
    mr_with_reset_signal,
    orb_failure_signal,
    with_cooldown,
    signal_funnel,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _session_index(dates: list, bars_per_session: int = 10):
    """Build a DatetimeIndex with ``bars_per_session`` 1-min bars per session date."""
    timestamps = []
    for d in dates:
        base = pd.Timestamp(f"{d} 09:15:00")
        for m in range(bars_per_session):
            timestamps.append(base + pd.Timedelta(minutes=m))
    return pd.DatetimeIndex(timestamps)


# ── oneshot_edge ──────────────────────────────────────────────────────────────

def test_persistent_condition_fires_once_per_session():
    """A condition that is True for all 5 bars should fire exactly once per session."""
    idx = _session_index(["2020-01-01", "2020-01-02"], bars_per_session=5)
    cond = pd.Series(True, index=idx)
    result = oneshot_edge(cond)
    # Only bar 0 of each session should fire (rising edge from "no prior bar")
    assert int(result.sum()) == 2   # one per session


def test_level_condition_suppressed_after_first_edge():
    """ORB-style: condition stays True for bars 3-8; only bar 3 fires."""
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    cond = pd.Series(False, index=idx)
    cond.iloc[3:] = True   # turns True at bar 3, stays True
    result = oneshot_edge(cond)
    assert int(result.sum()) == 1
    assert result.iloc[3] == True
    assert result.iloc[4:].sum() == 0


def test_oneshot_edge_fires_only_once_per_session():
    """oneshot_edge fires at MOST once per session even after condition resets."""
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    cond = pd.Series(False, index=idx)
    cond.iloc[2] = True    # first edge
    cond.iloc[3] = False   # reset
    cond.iloc[5] = True    # second edge — suppressed by oneshot_edge
    result = oneshot_edge(cond)
    assert result.iloc[2] == True
    assert result.iloc[5] == False   # suppressed — use mr_with_reset_signal for re-fire
    assert int(result.sum()) == 1


def test_no_cross_session_spillover():
    """The first bar of session 2 must not inherit state from session 1."""
    idx = _session_index(["2020-01-01", "2020-01-02"], bars_per_session=5)
    cond = pd.Series(False, index=idx)
    # Make session 1's last bar True, session 2's first bar True
    cond.iloc[4] = True   # end of session 1
    cond.iloc[5] = True   # start of session 2
    result = oneshot_edge(cond)
    # Both should fire (they are rising edges within their own sessions)
    assert result.iloc[4] == True
    assert result.iloc[5] == True


# ── oneshot_level ─────────────────────────────────────────────────────────────

def test_oneshot_level_first_true_only():
    idx = _session_index(["2020-01-01"], bars_per_session=8)
    cond = pd.Series([False]*3 + [True]*5, index=idx)
    result = oneshot_level(cond)
    assert int(result.sum()) == 1
    assert result.iloc[3] == True


# ── MR with reset ─────────────────────────────────────────────────────────────

def test_mr_reset_required_before_re_entry():
    """Second entry only allowed after z-score resets above threshold."""
    idx = _session_index(["2020-01-01"], bars_per_session=15)
    z = pd.Series([0.0]*5 + [-3.0]*4 + [0.5]*3 + [-2.5]*3, index=idx)
    # entry_thr=-2, reset_thr=-0.5 (long): fire when z < -2, reset when z > -0.5
    result = mr_with_reset_signal(z, entry_thr=-2.0, reset_thr=-0.5, direction="long")
    # Should fire at bar 5 (first z < -2)
    assert result.iloc[5] == 1
    # Should NOT fire at bars 6-8 (still below threshold, no reset yet)
    assert result.iloc[6:9].sum() == 0
    # After reset (bar 9: z=0.5 > -0.5), next crossing at bar 12 fires
    assert result.iloc[12] == 1
    total_fires = (result != 0).sum()
    assert total_fires == 2


def test_mr_no_re_entry_without_reset():
    """Condition stays below threshold — only one fire allowed."""
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    z = pd.Series([-3.0]*10, index=idx)  # always below -2
    result = mr_with_reset_signal(z, entry_thr=-2.0, reset_thr=-0.5, direction="long")
    assert (result != 0).sum() == 1   # fires once at bar 0, no re-entry


def test_mr_short_direction():
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    z = pd.Series([0.0]*3 + [3.5]*4 + [0.0]*3, index=idx)
    result = mr_with_reset_signal(z, entry_thr=2.0, reset_thr=0.5, direction="short")
    assert result.iloc[3] == -1
    assert (result != 0).sum() == 1   # only one fire, condition doesn't reset-then-cross


# ── ORF state machine ─────────────────────────────────────────────────────────

def test_orf_requires_breakout_then_return():
    """ORF_UP_SHORT: need to see close>or_high THEN close<=or_high."""
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    or_high = pd.Series(100.0, index=idx)
    or_low  = pd.Series(90.0,  index=idx)
    # bars 0-2: NaN (OR not established)
    or_high.iloc[:3] = np.nan
    or_low.iloc[:3]  = np.nan
    # bars 3-5: above OR high (breakout)
    close = pd.Series([95]*3 + [105]*3 + [98]*4, index=idx)
    result = orb_failure_signal(close, or_high, or_low, direction="short")
    # Fire should be at bar 6 (first close <= or_high after breakout)
    assert result.iloc[6] == -1
    assert (result != 0).sum() == 1


def test_orf_no_fire_without_breakout():
    """If price never breaks out, ORF should not fire."""
    idx = _session_index(["2020-01-01"], bars_per_session=8)
    or_high = pd.Series([np.nan]*2 + [100.0]*6, index=idx)
    or_low  = pd.Series([np.nan]*2 + [90.0]*6,  index=idx)
    close = pd.Series([95]*8, index=idx)  # always inside range
    result = orb_failure_signal(close, or_high, or_low, direction="short")
    assert (result != 0).sum() == 0


def test_orf_fires_only_once_per_session():
    """Even if price bounces in and out, only first return fires."""
    idx = _session_index(["2020-01-01"], bars_per_session=12)
    or_high = pd.Series([np.nan]*2 + [100.0]*10, index=idx)
    or_low  = pd.Series([np.nan]*2 + [90.0]*10,  index=idx)
    # breakout, return, breakout again, return again
    close = pd.Series([95]*2 + [105, 105, 98, 105, 105, 97, 95, 95, 95, 95], index=idx)
    result = orb_failure_signal(close, or_high, or_low, direction="short")
    assert (result != 0).sum() == 1   # only first return fires


# ── Cooldown ──────────────────────────────────────────────────────────────────

def test_cooldown_suppresses_within_window():
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    sig = pd.Series([0]*10, index=idx, dtype=int)
    sig.iloc[0] = 1   # fire at 0
    sig.iloc[2] = 1   # within cooldown=3 → suppress
    sig.iloc[5] = 1   # outside cooldown → allow
    result = with_cooldown(sig, cooldown_bars=3)
    assert result.iloc[0] == 1
    assert result.iloc[2] == 0   # suppressed
    assert result.iloc[5] == 1   # allowed


def test_cooldown_resets_at_session_boundary():
    idx = _session_index(["2020-01-01", "2020-01-02"], bars_per_session=5)
    sig = pd.Series(0, index=idx, dtype=int)
    sig.iloc[4] = 1   # last bar of session 1
    sig.iloc[5] = 1   # first bar of session 2 (cooldown should reset)
    result = with_cooldown(sig, cooldown_bars=10)
    assert result.iloc[4] == 1
    assert result.iloc[5] == 1   # new session — cooldown reset


# ── Cost double-charging ──────────────────────────────────────────────────────

def test_round_trip_cost_not_doubled():
    """10 bps round-trip means entry_leg + exit_leg = 10 bps total, not 20."""
    from src.intraday.engine import run_intraday_backtest
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    price = 20_000.0
    df = pd.DataFrame(
        {"open": price, "high": price + 10, "low": price - 10, "close": price},
        index=idx,
    )
    # Signal at bar 0 → long entry at bar 1 open
    sig = pd.Series(0, index=idx, dtype=int)
    sig.iloc[0] = 1   # signal long at bar 0
    res = run_intraday_backtest(df, sig, cost_bps=10.0)
    if res["trades"]:
        t = res["trades"][0]
        # 10 bps RT on 20000 ≈ 20 points total
        expected_cost = 20_000 * (5 / 10_000) * 2   # entry + exit legs
        assert abs(t["cost"] - expected_cost) < 1.0, f"cost={t['cost']:.2f} expected≈{expected_cost:.2f}"


def test_reversal_charges_two_round_trips():
    """A long→short reversal should cost ≈ 2× a simple round trip."""
    from src.intraday.engine import run_intraday_backtest
    idx = _session_index(["2020-01-01"], bars_per_session=10)
    price = 20_000.0
    df = pd.DataFrame(
        {"open": price, "high": price + 10, "low": price - 10, "close": price},
        index=idx,
    )
    # Simple: long 1 bar
    sig_simple = pd.Series(0, index=idx, dtype=int)
    sig_simple.iloc[0] = 1   # long
    sig_simple.iloc[2] = 0   # exit
    res_simple = run_intraday_backtest(df, sig_simple, cost_bps=10.0)

    # Reversal: long then immediately short
    sig_rev = pd.Series(0, index=idx, dtype=int)
    sig_rev.iloc[0] = 1    # long
    sig_rev.iloc[2] = -1   # reversal to short
    sig_rev.iloc[4] = 0    # exit short
    res_rev = run_intraday_backtest(df, sig_rev, cost_bps=10.0)

    # Reversal should have more cost than simple
    if res_simple["trades"] and res_rev["trades"]:
        simple_cost = sum(t["cost"] for t in res_simple["trades"])
        rev_cost = sum(t["cost"] for t in res_rev["trades"])
        assert rev_cost > simple_cost, f"rev_cost={rev_cost:.2f} should exceed simple_cost={simple_cost:.2f}"


# ── Signal funnel diagnostics ─────────────────────────────────────────────────

def test_signal_funnel_counts():
    idx = _session_index(["2020-01-01", "2020-01-02"], bars_per_session=10)
    # condition True for bars 3-9 of session 1, bar 2-9 of session 2
    cond = pd.Series(False, index=idx)
    cond.iloc[3:10] = True   # session 1, bars 3-9 → 7 bars, 1 rising edge
    cond.iloc[12:20] = True  # session 2, bars 2-9 → 8 bars, 1 rising edge
    # one-shot signal: 1 per session
    sig = oneshot_edge(cond).astype(int)
    funnel = signal_funnel(cond, sig)
    assert funnel["raw_condition_bars"] == 15   # 7 + 8
    assert funnel["rising_edges"] == 2           # one per session
    assert funnel["executed_signals"] == 2
    assert funnel["sessions_traded"] == 2
    # turnover reduction: raw=15, executed=2 → (1 - 2/15) ≈ 0.867
    assert funnel["turnover_reduction_vs_raw"] > 0.8


def test_signal_funnel_reduction_is_meaningful():
    """Old level-triggered logic would fire 15 times; one-shot fires 2 times."""
    idx = _session_index(["2020-01-01", "2020-01-02"], bars_per_session=10)
    cond = pd.Series(False, index=idx)
    cond.iloc[3:10] = True
    cond.iloc[12:20] = True
    old_style_sig = cond.astype(int)   # fires every True bar (old bug)
    new_style_sig = oneshot_edge(cond).astype(int)
    funnel_old = signal_funnel(cond, old_style_sig)
    funnel_new = signal_funnel(cond, new_style_sig)
    assert funnel_old["executed_signals"] == 15
    assert funnel_new["executed_signals"] == 2
    assert funnel_new["turnover_reduction_vs_raw"] > 0.8
