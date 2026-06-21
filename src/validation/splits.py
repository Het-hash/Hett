"""Chronological train/validation/test splits for time-series data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplits:
    development: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    dev_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"DataSplits(\n"
            f"  development: {self.development.index[0].date()} → {self.development.index[-1].date()}"
            f" ({len(self.development)} rows)\n"
            f"  validation:  {self.validation.index[0].date()} → {self.validation.index[-1].date()}"
            f" ({len(self.validation)} rows)\n"
            f"  test:        {self.test.index[0].date()} → {self.test.index[-1].date()}"
            f" ({len(self.test)} rows)\n"
            f")"
        )


def make_splits(
    df: pd.DataFrame,
    dev_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> DataSplits:
    """
    Create chronological splits. Fractions must sum to 1.0.

    CRITICAL: Test period is never touched until final evaluation.
    Regime thresholds and parameters are fitted on development only.
    """
    assert abs(dev_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    n = len(df)
    n_dev = int(n * dev_frac)
    n_val = int(n * val_frac)

    dev = df.iloc[:n_dev]
    val = df.iloc[n_dev : n_dev + n_val]
    test = df.iloc[n_dev + n_val :]

    splits = DataSplits(
        development=dev,
        validation=val,
        test=test,
        dev_end=dev.index[-1],
        val_end=val.index[-1],
        test_end=test.index[-1],
    )
    logger.info(repr(splits))
    return splits
