"""Purging and embargo to prevent look-ahead leakage in overlapping labels."""
from __future__ import annotations

import pandas as pd


def purge_overlap(
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    max_horizon: int = 20,
) -> pd.DatetimeIndex:
    """
    Remove from train_index any bars whose forward labels overlap with test_index.

    For horizon h, a signal at T uses prices up to T+h.
    If T+h >= test_start, then T = test_start - h must be removed from training.
    """
    test_start = test_index[0]
    purge_start = test_start - pd.Timedelta(days=max_horizon * 2)  # conservative calendar buffer
    return train_index[train_index < purge_start]


def embargo_end(
    test_end: pd.Timestamp,
    embargo_bars: int = 20,
    index: pd.DatetimeIndex = None,
) -> pd.Timestamp:
    """
    Return the timestamp after which data can safely be used in the next training fold,
    given an embargo of `embargo_bars` after the test period ends.
    """
    if index is not None:
        test_end_loc = index.get_loc(test_end)
        embargo_loc = min(test_end_loc + embargo_bars, len(index) - 1)
        return index[embargo_loc]
    return test_end + pd.Timedelta(days=embargo_bars * 2)
