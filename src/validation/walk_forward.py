"""Walk-forward validation with expanding training window."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.validation.purging import purge_overlap, embargo_end
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_n: int
    test_n: int
    oos_metrics: dict = field(default_factory=dict)
    oos_returns: Optional[pd.Series] = None


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold]
    combined_oos_returns: Optional[pd.Series]
    combined_oos_metrics: dict

    def fold_metrics_df(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            row = {"fold": f.fold_id, "train_start": f.train_start, "train_end": f.train_end,
                   "test_start": f.test_start, "test_end": f.test_end,
                   "train_n": f.train_n, "test_n": f.test_n}
            row.update(f.oos_metrics)
            rows.append(row)
        return pd.DataFrame(rows)


def walk_forward(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    backtest_fn: Callable[[pd.DataFrame, pd.Series], object],
    metrics_fn: Callable[[object], dict],
    min_train_bars: int = 252,
    step_bars: int = 63,
    expanding: bool = True,
    embargo_bars: int = 20,
    max_horizon: int = 20,
) -> WalkForwardResult:
    """
    Expanding (or rolling) walk-forward validation.

    For each fold:
    1. Train/fit on [train_start, train_end].
    2. Generate signals on test period.
    3. Backtest and compute metrics on test period only.
    4. Advance by step_bars (with embargo).

    signal_fn: takes a DataFrame (train data for fitting) and returns a signal
               function OR takes full df and is called per fold.
    backtest_fn: takes (df_test, signal_test) and returns a BacktestResult.
    metrics_fn: takes BacktestResult and returns metrics dict.
    """
    folds: List[WalkForwardFold] = []
    all_oos_returns: List[pd.Series] = []

    n = len(df)
    train_start_loc = 0
    test_start_loc = min_train_bars

    fold_id = 0
    while test_start_loc + step_bars <= n:
        fold_id += 1
        test_end_loc = min(test_start_loc + step_bars, n)

        train_df = df.iloc[train_start_loc:test_start_loc]
        test_df = df.iloc[test_start_loc:test_end_loc]

        if len(train_df) < min_train_bars or len(test_df) < 2:
            test_start_loc = test_end_loc
            continue

        logger.info(
            f"WF Fold {fold_id}: train {train_df.index[0].date()}→{train_df.index[-1].date()}"
            f" | test {test_df.index[0].date()}→{test_df.index[-1].date()}"
        )

        try:
            # Generate signal on test period (signal_fn may internally use only train data)
            signal_test = signal_fn(df.iloc[: test_end_loc], test_df)

            # Run backtest on test period only
            result = backtest_fn(test_df, signal_test.loc[test_df.index])
            oos_metrics = metrics_fn(result)
            oos_returns = result.returns if hasattr(result, "returns") else None

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                train_n=len(train_df),
                test_n=len(test_df),
                oos_metrics=oos_metrics,
                oos_returns=oos_returns,
            )
            folds.append(fold)

            if oos_returns is not None:
                all_oos_returns.append(oos_returns)

        except Exception as exc:
            logger.warning(f"Fold {fold_id} failed: {exc}")

        # Advance
        if expanding:
            # Training window expands; only test start moves
            test_start_loc = test_end_loc + embargo_bars
        else:
            # Rolling window
            train_start_loc = test_start_loc
            test_start_loc = test_end_loc + embargo_bars

    # Combine all OOS returns
    combined_oos = None
    combined_metrics = {}
    if all_oos_returns:
        combined_oos = pd.concat(all_oos_returns).sort_index()
        from src.backtesting.metrics import compute_metrics
        equity = (1 + combined_oos).cumprod() * 1_000_000
        positions_flat = pd.Series(1.0, index=combined_oos.index)  # approximation
        combined_metrics = compute_metrics(combined_oos, equity, positions_flat)

    return WalkForwardResult(
        folds=folds,
        combined_oos_returns=combined_oos,
        combined_oos_metrics=combined_metrics,
    )
