"""Phase 1B: Real Intraday Edge Discovery for NIFTY 50.

A causal, session-based intraday research toolkit operating on real
1-minute NIFTY 50 + India VIX data. All feature computation obeys a strict
causal constraint (only data at or before bar T is used), execution enforces
intraday-only positions with mandatory square-off, and validation uses
session-count-based expanding-window walk-forward plus placebo testing.
"""
from __future__ import annotations

__all__ = [
    "loader",
    "aggregator",
    "features",
    "regimes",
    "engine",
    "hypotheses",
    "stats",
    "placebo",
    "walk_forward",
    "report",
]
