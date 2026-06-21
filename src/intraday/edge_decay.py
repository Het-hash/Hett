"""Edge decay analysis for Phase 1C contextual edge refinement.

Analyses whether a signal's gross expectancy has decayed over time.
All analysis uses only the development period (inner splits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_year_stats(trades: list, min_trades: int = 10) -> pd.DataFrame:
    """Return per-calendar-year trade statistics."""
    if not trades:
        return pd.DataFrame()

    rows = []
    df = pd.DataFrame(trades)
    df["year"] = pd.DatetimeIndex(df["entry_time"]).year

    for yr, grp in df.groupby("year"):
        n = len(grp)
        if n < min_trades:
            continue
        gross = grp["gross_pnl"].values
        rows.append({
            "year": int(yr),
            "n_trades": n,
            "avg_gross_pnl": float(gross.mean()),
            "hit_rate": float((gross > 0).mean()),
            "total_gross_pnl": float(gross.sum()),
            "std_gross_pnl": float(gross.std()),
        })

    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def rolling_expectancy(trades: list, window_sessions: int = 200) -> pd.DataFrame:
    """Rolling average gross pnl over a session window."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades).sort_values("entry_time")
    df["session_date"] = pd.DatetimeIndex(df["entry_time"]).normalize()

    sessions = sorted(df["session_date"].unique())
    rows = []
    for i, sess in enumerate(sessions):
        window_start = sessions[max(0, i - window_sessions + 1)]
        mask = (df["session_date"] >= window_start) & (df["session_date"] <= sess)
        sub = df[mask]
        if len(sub) < 10:
            continue
        rows.append({
            "end_session": sess,
            "n_trades": len(sub),
            "avg_gross_pnl": float(sub["gross_pnl"].mean()),
            "hit_rate": float((sub["gross_pnl"] > 0).mean()),
        })

    return pd.DataFrame(rows).set_index("end_session") if rows else pd.DataFrame()


def pre_post_2020_split(trades: list) -> dict:
    """Compare pre-2020 vs 2020+ gross expectancy."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    df["year"] = pd.DatetimeIndex(df["entry_time"]).year

    pre = df[df["year"] < 2020]["gross_pnl"]
    post = df[df["year"] >= 2020]["gross_pnl"]

    result = {}
    for label, s in [("pre_2020", pre), ("post_2020", post)]:
        if len(s) >= 5:
            result[label] = {
                "n_trades": len(s),
                "avg_gross_pnl": float(s.mean()),
                "hit_rate": float((s > 0).mean()),
                "std": float(s.std()),
            }

    if "pre_2020" in result and "post_2020" in result:
        result["decay"] = result["pre_2020"]["avg_gross_pnl"] - result["post_2020"]["avg_gross_pnl"]

    return result


def be_cost_through_time(trades: list, avg_price: float = 20_000.0) -> pd.DataFrame:
    """Break-even cost (bps) by year — shows if edge is eroding."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["year"] = pd.DatetimeIndex(df["entry_time"]).year

    rows = []
    for yr, grp in df.groupby("year"):
        avg_gross = grp["gross_pnl"].mean()
        if avg_gross > 0 and avg_price > 0:
            be_bps = (avg_gross / avg_price) * 10_000 * 2
        else:
            be_bps = 0.0
        rows.append({"year": int(yr), "n_trades": len(grp), "be_bps": round(be_bps, 2)})

    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def classify_decay(yearly: pd.DataFrame) -> str:
    """Classify temporal stability of the edge.

    Returns: STABLE / DECLINING / RECENT_ONLY / INSUFFICIENT_DATA
    """
    if yearly.empty or len(yearly) < 3:
        return "INSUFFICIENT_DATA"

    years = yearly.index.values
    avgs = yearly["avg_gross_pnl"].values

    # Fit linear trend
    coeffs = np.polyfit(years, avgs, 1)
    slope = coeffs[0]

    recent_2 = avgs[-2:].mean() if len(avgs) >= 2 else avgs[-1]
    early_2 = avgs[:2].mean() if len(avgs) >= 2 else avgs[0]

    if slope < -5 and recent_2 < 0:
        return "DECLINING"
    if slope < -5 and early_2 > 10 and recent_2 < early_2 * 0.5:
        return "DECLINING"
    if early_2 <= 0 and recent_2 > 5:
        return "RECENT_ONLY"
    return "STABLE"
