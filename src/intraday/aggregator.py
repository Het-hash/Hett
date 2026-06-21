"""Causal multi-timeframe aggregation of 1-minute bars.

Bars are anchored to the session open (09:15) and never span two trading
days. The final bar of a session may be partial (e.g. 375 min / 30 = 12.5
buckets) and is retained as a smaller bar.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def get_sessions(df_1min: pd.DataFrame) -> List:
    """Return the sorted list of unique trading dates (datetime.date)."""
    return sorted(pd.Index(df_1min.index.normalize()).unique().date.tolist())


def resample_to_tf(df_1min: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Aggregate 1-min bars to ``tf_minutes`` bars, anchored to 09:15.

    OHLC aggregation: open=first, high=max, low=min, close=last. VIX columns use
    the same rules. ``vix_envelope_repaired`` (if present) becomes True if any
    constituent bar was repaired. No bar spans two sessions; the trailing
    partial bucket of a session is kept.

    The resulting index is the timestamp of the FIRST 1-min bar in each bucket
    (i.e. the bucket open time), so the value is known at the bucket close.
    """
    if tf_minutes < 1:
        raise ValueError("tf_minutes must be >= 1")
    if tf_minutes == 1:
        return df_1min.copy()

    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vix_open": "first",
        "vix_high": "max",
        "vix_low": "min",
        "vix_close": "last",
    }
    has_repaired = "vix_envelope_repaired" in df_1min.columns
    if has_repaired:
        agg_map["vix_envelope_repaired"] = "max"

    cols = [c for c in agg_map if c in df_1min.columns]

    out_frames = []
    dates = df_1min.index.normalize()
    for _, day_df in df_1min.groupby(dates):
        day_df = day_df.sort_index()
        # bar position within session (0 = 09:15)
        n = len(day_df)
        bar_pos = np.arange(n)
        bucket = bar_pos // tf_minutes
        grouped = day_df[cols].groupby(bucket)
        agg = grouped.agg({c: agg_map[c] for c in cols})
        # index = open timestamp of each bucket
        open_times = day_df.index.to_series().groupby(bucket).first().values
        agg.index = pd.DatetimeIndex(open_times)
        out_frames.append(agg)

    result = pd.concat(out_frames).sort_index()
    result.index.name = df_1min.index.name or "timestamp"
    if has_repaired:
        result["vix_envelope_repaired"] = result["vix_envelope_repaired"].astype(bool)
    return result
