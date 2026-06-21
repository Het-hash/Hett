"""Position sizing methods."""
from __future__ import annotations
# Re-exports apply_sizing from portfolio.py with a risk-aware interface.

from src.backtesting.portfolio import apply_sizing

__all__ = ["apply_sizing"]
