"""Causal intraday features.

Every feature at bar T is computed using only information available at or
before bar T (within-session cumulative quantities, trailing rolling windows,
and previous-day daily levels known at the open). No ``.shift(-n)`` is used.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

EPS = 1e-9


def _daily_from_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Build a daily OHLC + VIX-close frame indexed by date from minute data."""
    dates = df.index.normalize()
    daily = pd.DataFrame(
        {
            "day_open": df["open"].groupby(dates).first(),
            "day_high": df["high"].groupby(dates).max(),
            "day_low": df["low"].groupby(dates).min(),
            "day_close": df["close"].groupby(dates).last(),
            "day_vix_close": df["vix_close"].groupby(dates).last(),
        }
    )
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.sort_index()


def compute_intraday_features(
    df: pd.DataFrame,
    session_dates: Optional[List] = None,
    or_bar_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Return ``df`` augmented with causal intraday feature columns.

    Works on any timeframe (1-min, 5-min, ...). Opening-range windows are
    expressed in *bars* via ``or_bar_windows``, so callers should pass bar
    counts appropriate to the timeframe. Defaults assume a 1-min frame
    (15/30/60 bars). ``distance_from_or_*`` always uses the smallest window.
    """
    if or_bar_windows is None:
        or_bar_windows = [15, 30, 60]
    or_bar_windows = sorted(set(int(w) for w in or_bar_windows if w >= 1))
    or_ref = or_bar_windows[0]
    out = df.copy()
    dates = out.index.normalize()

    # Previous-day daily levels (known at session open).
    daily = _daily_from_minute(out)
    prev_daily = daily.shift(1)  # shift forward in time → previous session
    prev_map = prev_daily.reindex(dates)
    out["prev_day_close"] = prev_map["day_close"].values
    out["prev_day_high"] = prev_map["day_high"].values
    out["prev_day_low"] = prev_map["day_low"].values
    prev_vix_close = prev_map["day_vix_close"].values

    g = out.groupby(dates, group_keys=False)

    # bar_num within session
    out["bar_num"] = g.cumcount()

    # session open (causal: constant = first bar open, known from bar 0)
    out["session_open"] = g["open"].transform("first")
    session_vix_open = g["vix_open"].transform("first")

    # running session high / low up to T
    out["session_high_so_far"] = g["high"].cummax()
    out["session_low_so_far"] = g["low"].cummin()

    # Gap and returns
    out["gap_pct"] = (out["session_open"] - out["prev_day_close"]) / (
        out["prev_day_close"] + EPS
    )
    out["open_to_now_ret"] = (out["close"] - out["session_open"]) / (
        out["session_open"] + EPS
    )
    out["prev_close_to_now_ret"] = (out["close"] - out["prev_day_close"]) / (
        out["prev_day_close"] + EPS
    )

    # Recent returns (within session, no cross-session leakage)
    close = out["close"]
    out["ret_1bar"] = g["close"].pct_change(1)
    out["ret_3bar"] = g["close"].pct_change(3)
    out["ret_5bar"] = g["close"].pct_change(5)

    # Opening ranges. OR_N high/low become known at bar N-1 close; available
    # from bar_num >= N. Computed causally: rolling cummax/cummin then frozen
    # at the OR window boundary.
    for n in or_bar_windows:
        hi_cum = g["high"].cummax()
        lo_cum = g["low"].cummin()
        bar = out["bar_num"]
        # value of cummax/cummin at bar n-1 within each session
        or_hi = hi_cum.where(bar == n - 1)
        or_lo = lo_cum.where(bar == n - 1)
        or_hi = or_hi.groupby(dates).transform(lambda s: s.ffill())
        or_lo = or_lo.groupby(dates).transform(lambda s: s.ffill())
        # mask: only known once bar_num >= n
        or_hi = or_hi.where(bar >= n)
        or_lo = or_lo.where(bar >= n)
        out[f"or{n}_high"] = or_hi
        out[f"or{n}_low"] = or_lo

    # VIX features
    out["vix_level"] = out["vix_close"]
    out["vix_change_from_open"] = (out["vix_close"] - session_vix_open) / (
        session_vix_open + EPS
    )
    out["vix_change_from_prev_close"] = (out["vix_close"] - prev_vix_close) / (
        prev_vix_close + EPS
    )

    # Rolling stats (trailing, within-session)
    out["rolling_mean_20"] = g["close"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    out["rolling_std_20"] = g["close"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    out["zscore_20"] = (out["close"] - out["rolling_mean_20"]) / (
        out["rolling_std_20"] + EPS
    )

    # ATR(14): trailing true-range mean within session
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.groupby(dates).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )

    # Distance from reference OR (smallest window) levels
    out["distance_from_or_high"] = (out["close"] - out[f"or{or_ref}_high"]) / (
        out[f"or{or_ref}_high"] + EPS
    )
    out["distance_from_or_low"] = (out["close"] - out[f"or{or_ref}_low"]) / (
        out[f"or{or_ref}_low"] + EPS
    )

    # Candle body
    out["candle_body"] = (out["close"] - out["open"]) / (
        (out["high"] - out["low"]) + EPS
    )

    # NIFTY/VIX divergence: True when sign of price-return-from-open does NOT
    # match the "expected" inverse-VIX direction. Normally price up ↔ vix down,
    # so a confirming move has sign(open_to_now_ret) == sign(-vix_change_from_open).
    price_sign = np.sign(out["open_to_now_ret"].fillna(0.0))
    expected_sign = np.sign(-out["vix_change_from_open"].fillna(0.0))
    out["nifty_vix_divergence"] = (price_sign != expected_sign) & (price_sign != 0)

    return out
