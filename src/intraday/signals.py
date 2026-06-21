"""Signal semantics utilities for one-shot, edge-triggered intraday signals.

All functions here preserve the fundamental rule: a condition that is
persistently true must NOT fire repeatedly. A new signal fires only on a
rising edge (False→True transition), and subsequent same-direction signals are
suppressed until the condition resets.

Every function works on a pandas Series with a DatetimeIndex. "Per session"
grouping uses index.normalize() (the date component).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dates(s: pd.Series) -> pd.Series:
    """Return the date (day) component of the index as a Series."""
    return s.index.normalize()


# ---------------------------------------------------------------------------
# Core edge / one-shot utilities
# ---------------------------------------------------------------------------

def oneshot_edge(condition: pd.Series) -> pd.Series:
    """Keep only the FIRST rising edge per session.

    A rising edge is a False→True transition in ``condition``.  If the
    condition is already True at bar 0 of a session it still counts as a
    rising edge (there is no prior bar to compare to within this session).

    Returns a boolean Series with the same index.
    """
    dates = _dates(condition)
    cond_bool = condition.astype(bool)
    # shift within session; bar 0 has no prior bar → shift produces NaN → fillna→cast bool
    prev = cond_bool.groupby(dates).shift(1).fillna(False).astype(bool)
    rising = cond_bool & ~prev  # True only on the transition
    # Keep at most the first per session
    cumsum = rising.groupby(dates).cumsum()
    return (cumsum == 1) & rising


def oneshot_level(condition: pd.Series) -> pd.Series:
    """Keep only the FIRST True bar per session (level-based, not edge).

    Use for signals that are determined at a fixed point in the session
    (e.g., gap determined at bar 0).  The first True bar fires; all
    subsequent True bars in the same session are suppressed.
    """
    cond_bool = condition.astype(bool)
    dates = _dates(condition)
    cumsum = cond_bool.groupby(dates).cumsum()
    return (cumsum == 1) & cond_bool


def with_cooldown(signal: pd.Series, cooldown_bars: int) -> pd.Series:
    """Suppress re-entries within ``cooldown_bars`` bars of the last signal.

    The cooldown is cross-bar but does NOT cross session boundaries — each
    session resets the cooldown state.
    """
    if cooldown_bars <= 0:
        return signal
    sig = signal.copy()
    dates = _dates(signal)
    for date, grp in signal.groupby(dates):
        idx = list(grp.index)
        vals = grp.values.copy()
        last_fire = -cooldown_bars - 1
        for i, v in enumerate(vals):
            if v != 0:
                if i - last_fire <= cooldown_bars:
                    vals[i] = 0
                else:
                    last_fire = i
        for i, ix in enumerate(idx):
            sig.at[ix] = vals[i]
    return sig


# ---------------------------------------------------------------------------
# Family-specific state machines
# ---------------------------------------------------------------------------

def mr_with_reset_signal(
    zscore: pd.Series,
    entry_thr: float,
    reset_thr: float,
    direction: str,
) -> pd.Series:
    """Mean-reversion one-shot with mandatory reset before re-entry.

    ``direction='long'``: fire when zscore < entry_thr; reset when zscore > reset_thr.
    ``direction='short'``: fire when zscore > entry_thr; reset when zscore < reset_thr.

    Per session: once fired, suppress until reset condition is met.
    After reset, the signal may fire again (allowing multiple trades per session
    IF the z-score crosses the threshold multiple times, each after a reset).
    """
    d = 1 if direction == "long" else -1
    sig = pd.Series(0, index=zscore.index, dtype=int)
    dates = _dates(zscore)

    for date, grp in zscore.groupby(dates):
        idx_list = list(grp.index)
        vals = grp.values
        waiting_reset = False
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            if not waiting_reset:
                crossed = (d == 1 and v < entry_thr) or (d == -1 and v > entry_thr)
                if crossed:
                    sig.at[idx_list[i]] = d
                    waiting_reset = True
            else:
                reset = (d == 1 and v > reset_thr) or (d == -1 and v < reset_thr)
                if reset:
                    waiting_reset = False
    return sig


def orb_failure_signal(
    close: pd.Series,
    or_high: pd.Series,
    or_low: pd.Series,
    direction: str,
) -> pd.Series:
    """Opening-range failure: breakout then confirmed return inside range.

    State machine per session:
      State 0: waiting for a breakout.
      State 1: breakout seen, waiting for price to return inside the OR.
      Fired: once returned inside range → fire opposite signal; suppress further.

    ``direction='short'`` → fade an upside breakout (ORF_UP_SHORT).
    ``direction='long'``  → fade a downside breakdown (ORF_DOWN_LONG).
    """
    d = -1 if direction == "short" else 1
    sig = pd.Series(0, index=close.index, dtype=int)
    dates = _dates(close)

    for date, grp in close.groupby(dates):
        idx_list = list(grp.index)
        cl = grp.values
        oh = or_high.loc[idx_list].values
        ol = or_low.loc[idx_list].values

        state = 0  # 0 = looking for break, 1 = break seen
        fired = False
        for i in range(len(idx_list)):
            if np.isnan(oh[i]) or np.isnan(ol[i]):
                continue
            if fired:
                break
            if state == 0:
                if direction == "short" and cl[i] > oh[i]:
                    state = 1
                elif direction == "long" and cl[i] < ol[i]:
                    state = 1
            elif state == 1:
                if direction == "short" and cl[i] <= oh[i]:
                    sig.at[idx_list[i]] = -1
                    fired = True
                elif direction == "long" and cl[i] >= ol[i]:
                    sig.at[idx_list[i]] = 1
                    fired = True
    return sig


# ---------------------------------------------------------------------------
# Signal funnel diagnostics
# ---------------------------------------------------------------------------

def signal_funnel(
    condition: pd.Series,
    signal: pd.Series,
) -> dict:
    """Count signal pipeline stages for diagnostic reporting.

    Parameters
    ----------
    condition : bool Series — raw condition (level True/False)
    signal    : int Series — +1/-1/0 after all filters

    Returns
    -------
    dict with keys: raw_condition_bars, rising_edges, executed_signals,
    sessions_traded, avg_trades_per_session, turnover_vs_raw
    """
    cond = condition.astype(bool)
    dates = _dates(cond)

    raw_bars = int(cond.sum())

    prev = cond.groupby(dates).shift(1).fillna(False).astype(bool)
    rising = cond & ~prev
    n_edges = int(rising.sum())

    executed = (signal != 0).astype(int)
    n_exec = int(executed.sum())
    sessions_with_trade = int(executed.groupby(dates).any().sum())
    n_sessions = int(dates.nunique())
    avg_per_session = n_exec / max(n_sessions, 1)

    raw_rate = raw_bars / max(n_sessions, 1)
    reduction = 1.0 - (n_exec / max(raw_bars, 1))

    return {
        "raw_condition_bars": raw_bars,
        "rising_edges": n_edges,
        "executed_signals": n_exec,
        "sessions_traded": sessions_with_trade,
        "total_sessions": n_sessions,
        "avg_trades_per_session": round(avg_per_session, 3),
        "turnover_reduction_vs_raw": round(reduction, 3),
    }
