"""Clean raw OHLCV data without silently modifying suspicious observations."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data.schema import OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME, TIMESTAMP
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CleaningDecision:
    """Record of a cleaning action taken on a specific row."""

    def __init__(self, date: str, field: str, action: str, reason: str, old_value=None, new_value=None):
        self.date = date
        self.field = field
        self.action = action
        self.reason = reason
        self.old_value = old_value
        self.new_value = new_value

    def to_dict(self) -> dict:
        return vars(self)


def clean_ohlcv(
    df: pd.DataFrame,
    extreme_return_threshold: float = 0.20,
) -> Tuple[pd.DataFrame, List[CleaningDecision]]:
    """
    Apply necessary structural fixes and return (cleaned_df, decisions).

    Policy:
    - Remove rows where ALL price columns are NaN (provider artifact).
    - Remove exact duplicate index entries.
    - Do NOT fill or interpolate suspicious prices — flag them instead.
    - Do NOT silently cap extreme returns.
    """
    decisions: List[CleaningDecision] = []
    original_len = len(df)

    # Sort chronologically
    df = df.sort_index()

    # Remove rows where all OHLC are NaN (provider sometimes appends empty rows)
    all_nan_mask = df[[OPEN, HIGH, LOW, CLOSE]].isna().all(axis=1)
    if all_nan_mask.any():
        for idx in df.index[all_nan_mask]:
            decisions.append(CleaningDecision(
                str(idx.date()), "OHLC", "removed",
                "All price columns are NaN — provider artifact row"
            ))
        df = df[~all_nan_mask]

    # Remove duplicate index entries (keep first occurrence, flag the rest)
    dup_mask = df.index.duplicated(keep="first")
    if dup_mask.any():
        for idx in df.index[dup_mask]:
            decisions.append(CleaningDecision(
                str(idx.date()), "index", "removed",
                "Duplicate timestamp — keeping first occurrence"
            ))
        df = df[~dup_mask]

    # Flag (but do NOT remove) extreme single-session returns
    if len(df) > 1:
        ret = df[CLOSE].pct_change()
        extreme_mask = ret.abs() > extreme_return_threshold
        for idx in df.index[extreme_mask]:
            r = ret.loc[idx]
            decisions.append(CleaningDecision(
                str(idx.date()), CLOSE, "flagged",
                f"Extreme return {r:.1%} — review manually",
                old_value=float(df.loc[idx, CLOSE]),
            ))

    rows_removed = original_len - len(df)
    if rows_removed:
        logger.info(f"Cleaning removed {rows_removed} rows")
    if decisions:
        logger.info(f"Cleaning produced {len(decisions)} decisions/flags")

    return df, decisions
