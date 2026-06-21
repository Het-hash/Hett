"""Intraday execution engine.

Executes a signal series under strict intraday-only constraints:
  - A signal at bar T close is acted on at the NEXT bar's open (T+1).
  - No new entries after ``no_entry_after``.
  - Any open position is force-squared-off at ``squareoff_time``.
  - No overnight positions: every session ends flat.

Costs are charged per position change. A flat→long, long→flat, etc. each cost
one one-way leg; a direct reversal (long→short) costs two legs (= round trip).
``cost_bps`` is the round-trip cost in basis points, so one leg = cost_bps/2.
"""
from __future__ import annotations

from datetime import time
from typing import Dict, List

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
) -> dict:
    """Run an intraday backtest. See module docstring for semantics."""
    df = df_1min.sort_index()
    sig = signals.reindex(df.index).fillna(0).astype(int)

    so_t = _parse_time(squareoff_time)
    ne_t = _parse_time(no_entry_after)

    leg_cost_frac = (cost_bps / 2.0) / 10000.0

    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    idx = df.index
    times = idx.time
    dates = idx.normalize()
    sig_vals = sig.values

    n = len(df)
    position = np.zeros(n, dtype=int)  # position held DURING bar i (entered at its open)

    trades: List[Dict] = []
    equity = np.empty(n, dtype=float)
    cash = initial_capital

    # Desired position to enter at bar i open = signal at bar i-1 close,
    # subject to no-entry-after and square-off rules and session boundaries.
    cur_pos = 0
    entry_price = np.nan
    entry_i = -1
    mfe = 0.0
    mae = 0.0

    def close_trade(exit_i, exit_price, reason):
        nonlocal cur_pos, entry_price, entry_i, mfe, mae, cash
        direction = cur_pos
        gross = direction * (exit_price - entry_price)
        # two legs (entry + exit) charged at exit time on combined notional
        cost = (entry_price + exit_price) * leg_cost_frac
        net = gross - cost
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

        # Square-off: at/after squareoff time, force flat at this bar's open.
        if cur_pos != 0:
            if new_session or t >= so_t:
                close_trade(i, opens[i], "squareoff" if t >= so_t else "session_end")

        # Determine entry/adjustment at this bar's open based on prior bar signal.
        desired = sig_vals[i - 1] if i > 0 and not new_session else 0
        # Block entries after cutoff and at/after square-off.
        can_trade = (t < ne_t) and (t < so_t)

        if can_trade:
            if desired != cur_pos:
                # close existing if any (reversal or exit)
                if cur_pos != 0 and desired != cur_pos:
                    close_trade(i, opens[i], "reversal" if desired != 0 else "signal_exit")
                # open new if desired non-zero
                if desired != 0 and cur_pos == 0:
                    cur_pos = desired
                    entry_price = opens[i]
                    entry_i = i
                    mfe = 0.0
                    mae = 0.0
        else:
            # no new entries, but allow signal-driven exits (flatten) before squareoff
            if cur_pos != 0 and desired == 0 and t < so_t:
                close_trade(i, opens[i], "signal_exit")

        # Update MFE/MAE on the bar we are holding through.
        if cur_pos != 0:
            fav = cur_pos * (highs[i] - entry_price)
            adv = cur_pos * (lows[i] - entry_price)
            # for long: high is favorable, low adverse; for short reversed via sign
            best = max(cur_pos * (highs[i] - entry_price), cur_pos * (lows[i] - entry_price))
            worst = min(cur_pos * (highs[i] - entry_price), cur_pos * (lows[i] - entry_price))
            mfe = max(mfe, best)
            mae = min(mae, worst)

        position[i] = cur_pos
        # mark-to-market equity (unrealized using current close)
        unreal = cur_pos * (closes[i] - entry_price) if cur_pos != 0 else 0.0
        equity[i] = cash + unreal

    # Safety: ensure flat at end
    if cur_pos != 0:
        close_trade(n - 1, closes[n - 1], "session_end")
        equity[n - 1] = cash

    equity_curve = pd.Series(equity, index=idx, name="equity")
    metrics = _compute_metrics(trades, equity_curve, dates, initial_capital)
    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "position": pd.Series(position, index=idx, name="position"),
        "metrics": metrics,
    }


def _compute_metrics(trades, equity_curve, dates, initial_capital) -> dict:
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "n_trades": 0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "avg_net_pnl": 0.0,
            "total_net_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_cost": 0.0,
        }
    nets = np.array([t["net_pnl"] for t in trades])
    gross = np.array([t["gross_pnl"] for t in trades])
    costs = np.array([t["cost"] for t in trades])
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    hit_rate = float((nets > 0).mean())
    profit_factor = float(wins.sum() / (-losses.sum())) if losses.sum() < 0 else float("inf")

    # Daily returns from equity for sharpe / cagr / dd.
    eq = equity_curve
    daily_eq = eq.groupby(eq.index.normalize()).last()
    daily_ret = daily_eq.pct_change().dropna()
    n_days = max(len(daily_eq), 1)
    years = n_days / 252.0
    total_return = daily_eq.iloc[-1] / initial_capital
    cagr = float(total_return ** (1 / years) - 1) if years > 0 and total_return > 0 else 0.0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    running_max = daily_eq.cummax()
    dd = (daily_eq - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": int(n_trades),
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "avg_net_pnl": float(nets.mean()),
        "total_net_pnl": float(nets.sum()),
        "gross_pnl": float(gross.sum()),
        "total_cost": float(costs.sum()),
        "n_days": int(n_days),
    }
