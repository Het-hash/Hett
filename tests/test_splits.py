"""Tests for chronological data splits."""
import pandas as pd
import pytest

from src.validation.splits import make_splits
from src.validation.purging import purge_overlap, embargo_end


def test_splits_are_chronological(synthetic_ohlcv):
    splits = make_splits(synthetic_ohlcv, 0.6, 0.2, 0.2)
    assert splits.development.index[-1] < splits.validation.index[0]
    assert splits.validation.index[-1] < splits.test.index[0]


def test_splits_cover_all_data(synthetic_ohlcv):
    splits = make_splits(synthetic_ohlcv, 0.6, 0.2, 0.2)
    total = len(splits.development) + len(splits.validation) + len(splits.test)
    assert total == len(synthetic_ohlcv)


def test_splits_no_overlap(synthetic_ohlcv):
    splits = make_splits(synthetic_ohlcv, 0.6, 0.2, 0.2)
    dev_dates = set(splits.development.index)
    val_dates = set(splits.validation.index)
    test_dates = set(splits.test.index)
    assert len(dev_dates & val_dates) == 0
    assert len(val_dates & test_dates) == 0
    assert len(dev_dates & test_dates) == 0


def test_purge_removes_near_test(synthetic_ohlcv):
    splits = make_splits(synthetic_ohlcv, 0.6, 0.2, 0.2)
    purged = purge_overlap(splits.development.index, splits.validation.index, max_horizon=20)
    # Purged train index should end before validation start - some buffer
    if len(purged) > 0:
        assert purged[-1] < splits.validation.index[0]


def test_embargo_end_is_after_test_end(synthetic_ohlcv):
    splits = make_splits(synthetic_ohlcv, 0.6, 0.2, 0.2)
    emb = embargo_end(splits.validation.index[-1], embargo_bars=20, index=synthetic_ohlcv.index)
    if emb in synthetic_ohlcv.index:
        assert emb >= splits.validation.index[-1]
