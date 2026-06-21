"""Risk limit checks."""
from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def check_drawdown_limit(equity: pd.Series, limit: float = 0.15) -> bool:
    """Return True if max drawdown exceeds limit."""
    from src.risk.drawdown import max_drawdown
    dd = abs(max_drawdown(equity))
    if dd > limit:
        logger.warning(f"Drawdown limit breached: {dd:.2%} > {limit:.2%}")
        return True
    return False


def check_exposure_limit(positions: pd.Series, max_exposure: float = 1.0) -> bool:
    """Return True if any position exceeds max exposure."""
    peak = positions.abs().max()
    if peak > max_exposure + 1e-6:
        logger.warning(f"Exposure limit breached: {peak:.2f} > {max_exposure:.2f}")
        return True
    return False
