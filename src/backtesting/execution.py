"""
Execution model: translates signals to trade records.

Convention:
- Signal at T (using close and all info available at T close).
- Execution at T+1 open (earliest possible).
- Position changes create a transaction at the T+1 open price.
- Reversals (long→short) count as TWO transactions: close + open.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.schema import OPEN, CLOSE
from src.backtesting.costs import CostProfile, ETF_CONSERVATIVE
from src.utils.logging import get_logger

logger = get_logger(__name__)

LONG = 1
FLAT = 0
SHORT = -1


@dataclass
class Trade:
    entry_date: str
    exit_date: Optional[str]
    direction: int  # LONG=1, SHORT=-1
    entry_price: float
    exit_price: Optional[float]
    gross_return: float = 0.0
    cost_frac: float = 0.0
    net_return: float = 0.0
    holding_days: int = 0
    closed: bool = False

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "direction": "long" if self.direction == LONG else "short",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "gross_return": round(self.gross_return, 6),
            "cost_frac": round(self.cost_frac, 6),
            "net_return": round(self.net_return, 6),
            "holding_days": self.holding_days,
        }


def generate_position_series(
    signal: pd.Series,
    direction: str = "long",
) -> pd.Series:
    """
    Convert a boolean entry signal into a position series.

    For 'long': position = 1 when signal is True, else 0.
    For 'short': position = -1 when signal is True, else 0.
    For 'both': position follows signal value directly (expects +1/0/-1).

    Signal at T → Position applied from T+1 open.
    Position series is indexed the same as signal but represents the position
    held DURING each bar (i.e., already shifted by one bar).
    """
    if direction == "long":
        raw_pos = signal.astype(int)
    elif direction == "short":
        raw_pos = -signal.astype(int)
    else:
        raw_pos = signal.astype(int)

    # Shift: signal fires at T, position entered at T+1
    return raw_pos.shift(1).fillna(0).astype(int).rename("position")
