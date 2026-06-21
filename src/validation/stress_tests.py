"""Stress test suite aggregator."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from src.validation.sensitivity import cost_sensitivity, execution_delay_sensitivity, missed_trades_sensitivity
from src.validation.robustness import subperiod_analysis, crisis_exclusion_test, best_trade_removal, bootstrap_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_stress_suite(
    df: pd.DataFrame,
    signal: pd.Series,
    returns: pd.Series,
    direction: str = "long",
    cost_profile=None,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, object]:
    """
    Run the full stress test suite and return results as a dict.
    """
    from src.backtesting.costs import ETF_CONSERVATIVE
    cp = cost_profile or ETF_CONSERVATIVE

    logger.info("Running stress test suite…")
    results = {}

    try:
        results["cost_sensitivity"] = cost_sensitivity(df, signal, direction, initial_capital)
        logger.info("✓ Cost sensitivity")
    except Exception as e:
        logger.warning(f"Cost sensitivity failed: {e}")

    try:
        results["delay_sensitivity"] = execution_delay_sensitivity(df, signal, direction, cp, initial_capital)
        logger.info("✓ Delay sensitivity")
    except Exception as e:
        logger.warning(f"Delay sensitivity failed: {e}")

    try:
        results["missed_trades"] = missed_trades_sensitivity(df, signal, direction, cp, initial_capital)
        logger.info("✓ Missed trades")
    except Exception as e:
        logger.warning(f"Missed trades failed: {e}")

    try:
        results["subperiod"] = subperiod_analysis(df, signal, direction, cp, initial_capital)
        logger.info("✓ Subperiod analysis")
    except Exception as e:
        logger.warning(f"Subperiod analysis failed: {e}")

    try:
        results["crisis_exclusion"] = crisis_exclusion_test(df, signal, direction=direction, cost_profile=cp, initial_capital=initial_capital)
        logger.info("✓ Crisis exclusion")
    except Exception as e:
        logger.warning(f"Crisis exclusion failed: {e}")

    try:
        results["best_trade_removal"] = best_trade_removal(df, signal, direction, cp, initial_capital, n_remove=5)
        logger.info("✓ Best trade removal")
    except Exception as e:
        logger.warning(f"Best trade removal failed: {e}")

    try:
        results["bootstrap"] = bootstrap_metrics(returns)
        logger.info("✓ Bootstrap CI")
    except Exception as e:
        logger.warning(f"Bootstrap failed: {e}")

    logger.info(f"Stress suite complete: {len(results)} tests run")
    return results
