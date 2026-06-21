"""Intraday execution engine.

Executes a signal series under strict intraday-only constraints:
  - A signal at bar T close is acted on at the NEXT bar's open (T+1).
  - No new entries after ``no_entry_after``.
  - Any open position is force-squared-off at ``squareoff_time``.
  - No overnight positions: every session ends flat.

Cost accounting
---------------
``cost_bps`` is the ROUND-TRIP cost in basis points (entry leg + exit leg).
One leg = cost_bps / 2 bps.

For a flat→long→flat sequence:
  cost = entry_price * (cost_bps/2/10000) + exit_price * (cost_bps/2/10000)

For a reversal (long→short in one step):
  cost = two legs (closing long) + two legs (opening short) = 2 round trips.

This is deliberately conservative: a reversal should not be cheaper than
two separate flat→position transitions.
"""
from __future__ import annotations

from datetime import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def run_intraday_backtest(
    df_1min: pd.DataFrame,
    signals: pd.Series,
    cost_bps: float = 10.0,
    squareoff_time: str = "15:20",
    no_entry_after: str = "15:00",
    initial_capital: float = 1_000_000,
    hold_bars: Optional[int] = None,
) -> dict:
    """Run an intraday backtest.

    Parameters
    ----------
    df_1min : DataFrame with open/high/low/close columns (any timeframe).
    signals : +1 (long) / -1 (short) / 0 (flat) at bar T close.
              The engine executes at bar T+1 open.
    cost_bps : round-trip cost in basis points. One leg = cost_bps/2 bps.
    squareoff_time : force-close all positions at this time.
    no_entry_after : do not open new positions at or after this time.
    hold_bars : if not None, force-exit N bars after entry (before squareoff).
    """
    df = df_1min.sort_index()
    sig = signals.reindex(df.index).fillna(0).astype(int)

    so_t = _parse_time(squareoff_time)
    ne_t = _parse_time(no_entry_after)

    # ONE LEG cost as fraction of trade value
    leg_frac = (cost_bps / 2.0) / 10_000.0

    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    idx = df.index
    times = idx.time
    dates = idx.normalize()
    sig_vals = sig.values

    n = len(df)
    position = np.zeros(n, dtype=int)

    trades: List[Dict] = []
    equity = np.empty(n, dtype=float)
    cash = initial_capital

    cur_pos = 0
    entry_price = np.nan
    entry_i = -1
    mfe = 0.0
    mae = 0.0
    gross_cum = 0.0  # running gross pnl (before costs)

    def _cost(price: float) -> float:
        """Cost of ONE leg at this price."""
        return price * leg_frac

    def close_trade(exit_i, exit_price, reason):
        nonlocal cur_pos, entry_price, entry_i, mfe, mae, cash, gross_cum
        direction = cur_pos
        gross = direction * (exit_price - entry_price)
        # Entry leg + exit leg
        cost = _cost(entry_price) + _cost(exit_price)
        net = gross - cost
        gross_cum += gross
        cash += net
        trades.append(
            {
                "entry_time": idx[entry_i],
                "exit_time": idx[exit_i],
                "direction": "long" if direction > 0 else "short",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_pnl": float(gross),
                "cost": float(cost),
                "net_pnl": float(net),
                "mfe": float(mfe),
                "mae": float(mae),
                "holding_bars": int(exit_i - entry_i),
                "exit_reason": reason,
            }
        )
        cur_pos = 0
        entry_price = np.nan
        entry_i = -1
        mfe = 0.0
        mae = 0.0

    for i in range(n):
        t = times[i]
        new_session = i == 0 or dates[i] != dates[i - 1]

        # ── Square-off or session boundary ─────────────────────────────────
        if cur_pos != 0:
            force_close = new_session or t >= so_t
            if not force_close and hold_bars is not None:
                force_close = (i - entry_i) >= hold_bars
            if force_close:
                reason = "squareoff" if t >= so_t else ("session_end" if new_session else "hold_bars")
                close_trade(i, opens[i], reason)

        # ── Determine desired position from prior bar's signal ──────────────
        desired = sig_vals[i - 1] if i > 0 and not new_session else 0
        can_entry = (t < ne_t) and (t < so_t) and not new_session

        if desired != cur_pos:
            if cur_pos != 0:
                # Reversal: closing existing + opening new = 2 separate legs EACH direction
                # We close first, then immediately re-open.
                close_reason = "reversal" if desired != 0 else "signal_exit"
                close_trade(i, opens[i], close_reason)
                if desired != 0 and can_entry:
                    # Extra cost for opening the new side: treated as new entry
                    cur_pos = desired
                    entry_price = opens[i]
                    entry_i = i
                    mfe = 0.0
                    mae = 0.0
            elif desired != 0 and can_entry:
                cur_pos = desired
                entry_price = opens[i]
                entry_i = i
                mfe = 0.0
                mae = 0.0

        # ── Update MFE / MAE ────────────────────────────────────────────────
        if cur_pos != 0:
            hi_rel = cur_pos * (highs[i] - entry_price)
            lo_rel = cur_pos * (lows[i] - entry_price)
            mfe = max(mfe, hi_rel, lo_rel)
            mae = min(mae, hi_rel, lo_rel)

        position[i] = cur_pos
        unreal = cur_pos * (closes[i] - entry_price) if cur_pos != 0 else 0.0
        equity[i] = cash + unreal

    # Safety
    if cur_pos != 0:
        close_trade(n - 1, closes[n - 1], "session_end")
        equity[n - 1] = cash

    equity_curve = pd.Series(equity, index=idx, name="equity")
    metrics = _compute_metrics(trades, equity_curve, initial_capital)
    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "position": pd.Series(position, index=idx, name="position"),
        "metrics": metrics,
        "gross_pnl_total": gross_cum,
    }


def run_intraday_backtest_gross(
    df: pd.DataFrame,
    signals: pd.Series,
    **kwargs,
) -> dict:
    """Run at 0 bps cost to isolate gross executable expectancy."""
    kw = {**kwargs, "cost_bps": 0.0}
    return run_intraday_backtest(df, signals, **kw)


def _compute_metrics(trades, equity_curve, initial_capital) -> dict:
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0,
            "n_trades": 0, "hit_rate": 0.0, "profit_factor": 0.0,
            "avg_net_pnl": 0.0, "total_net_pnl": 0.0,
            "avg_gross_pnl": 0.0, "total_gross_pnl": 0.0,
            "total_cost": 0.0, "n_days": 0,
        }
    nets = np.array([t["net_pnl"] for t in trades])
    gross = np.array([t["gross_pnl"] for t in trades])
    costs = np.array([t["cost"] for t in trades])
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    hit_rate = float((nets > 0).mean())
    pf = float(wins.sum() / (-losses.sum())) if losses.sum() < 0 else float("inf")

    eq = equity_curve
    daily_eq = eq.groupby(eq.index.normalize()).last()
    daily_ret = daily_eq.pct_change().dropna()
    n_days = max(len(daily_eq), 1)
    years = n_days / 252.0
    total_return = daily_eq.iloc[-1] / initial_capital
    cagr = float(total_return ** (1 / years) - 1) if years > 0 and total_return > 0 else 0.0
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if len(daily_ret) > 1 and daily_ret.std() > 0
        else 0.0
    )
    running_max = daily_eq.cummax()
    dd = (daily_eq - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": int(n_trades),
        "hit_rate": hit_rate,
        "profit_factor": pf,
        "avg_net_pnl": float(nets.mean()),
        "total_net_pnl": float(nets.sum()),
        "avg_gross_pnl": float(gross.mean()),
        "total_gross_pnl": float(gross.sum()),
        "total_cost": float(costs.sum()),
        "n_days": int(n_days),
    }
