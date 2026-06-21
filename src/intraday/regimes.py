"""Intraday and session-level regime classification.

Thresholds are FIT on development daily data only, then frozen. Session-level
classification uses only previous-day information; intraday classification
uses only causal within-session features.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class IntradayRegimeEngine:
    """Fit-once, frozen-threshold regime engine."""

    def __init__(self) -> None:
        self.fitted = False
        self.vix_low_thr = None
        self.vix_high_thr = None
        self.trend_thr = None
        self.vol_low_thr = None
        self.vol_high_thr = None
        self.gap_thr = None

    def fit(self, dev_daily_df: pd.DataFrame) -> "IntradayRegimeEngine":
        """Fit thresholds on daily development data.

        Expects columns: day_open, day_high, day_low, day_close, day_vix_close.
        """
        vix = dev_daily_df["day_vix_close"].dropna()
        self.vix_low_thr = float(vix.quantile(0.33))
        self.vix_high_thr = float(vix.quantile(0.67))

        daily_ret = dev_daily_df["day_close"].pct_change().dropna()
        self.trend_thr = float(daily_ret.abs().quantile(0.5))

        daily_range = (
            (dev_daily_df["day_high"] - dev_daily_df["day_low"])
            / dev_daily_df["day_close"]
        ).dropna()
        self.vol_low_thr = float(daily_range.quantile(0.33))
        self.vol_high_thr = float(daily_range.quantile(0.67))

        gap = (
            dev_daily_df["day_open"] / dev_daily_df["day_close"].shift(1) - 1
        ).dropna()
        self.gap_thr = float(gap.abs().quantile(0.8))

        self.fitted = True
        return self

    def classify_session(self, session_date, prev_day_info: Dict) -> Dict:
        """Classify a session using only previous-day info.

        ``prev_day_info`` keys: prev_close, prev_high, prev_low, prev_vix_close,
        prev_prev_close (for prev-day trend), session_open (today's open, known
        at 09:15), prev_day_range_pct.
        """
        if not self.fitted:
            raise RuntimeError("Engine not fitted.")

        vix = prev_day_info.get("prev_vix_close", np.nan)
        if np.isnan(vix):
            vix_regime = "normal"
        elif vix >= self.vix_high_thr:
            vix_regime = "high"
        elif vix <= self.vix_low_thr:
            vix_regime = "low"
        else:
            vix_regime = "normal"

        prev_close = prev_day_info.get("prev_close", np.nan)
        prev_prev = prev_day_info.get("prev_prev_close", np.nan)
        if np.isnan(prev_close) or np.isnan(prev_prev) or prev_prev == 0:
            prev_trend = "neutral"
        else:
            r = prev_close / prev_prev - 1
            if r > self.trend_thr:
                prev_trend = "bull"
            elif r < -self.trend_thr:
                prev_trend = "bear"
            else:
                prev_trend = "neutral"

        rng = prev_day_info.get("prev_day_range_pct", np.nan)
        if np.isnan(rng):
            prev_vol = "normal"
        elif rng >= self.vol_high_thr:
            prev_vol = "high"
        elif rng <= self.vol_low_thr:
            prev_vol = "low"
        else:
            prev_vol = "normal"

        s_open = prev_day_info.get("session_open", np.nan)
        if np.isnan(s_open) or np.isnan(prev_close) or prev_close == 0:
            gap_regime = "normal"
        else:
            gap = s_open / prev_close - 1
            if gap >= self.gap_thr:
                gap_regime = "large_up"
            elif gap <= -self.gap_thr:
                gap_regime = "large_down"
            else:
                gap_regime = "normal"

        return {
            "vix_regime": vix_regime,
            "prev_trend": prev_trend,
            "prev_vol": prev_vol,
            "gap_regime": gap_regime,
        }

    def classify_intraday(self, df_session: pd.DataFrame) -> pd.Series:
        """Classify each bar of a session causally.

        Uses directional efficiency (net move / path length) over a trailing
        window plus NIFTY/VIX confirmation. Requires causal feature columns
        (``open_to_now_ret``, ``vix_change_from_open``, ``close``).
        """
        close = df_session["close"]
        window = 20
        net_move = close.diff(window).abs()
        path = close.diff().abs().rolling(window, min_periods=window).sum()
        efficiency = net_move / (path + 1e-9)

        labels = pd.Series("choppy", index=df_session.index, dtype=object)
        labels[efficiency >= 0.5] = "trending"
        labels[(efficiency < 0.5) & (efficiency >= 0.2)] = "mean_reverting"
        # bars before window is full stay "choppy" (insufficient info)
        labels[efficiency.isna()] = "choppy"

        if "nifty_vix_divergence" in df_session.columns:
            div = df_session["nifty_vix_divergence"].astype(bool)
            vix_label = pd.Series(
                np.where(div, "vix_diverging", "vix_confirming"),
                index=df_session.index,
            )
            labels = labels.str.cat(vix_label, sep="|")
        return labels
