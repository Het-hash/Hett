"""Cost model for realistic transaction cost simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CostProfile:
    """All costs are fractions of trade value unless noted."""
    label: str = "custom"
    # Brokerage as fraction (e.g., 0.0002 = 2bps)
    brokerage_frac: float = 0.0002
    # STT fraction (buy+sell or sell only depending on product)
    stt_frac: float = 0.0001
    # Exchange transaction charge
    exchange_txn_frac: float = 0.0000345
    # SEBI charge
    sebi_frac: float = 0.000001
    # GST on (brokerage + exchange charges)
    gst_rate: float = 0.18
    # Stamp duty on buy side only
    stamp_frac: float = 0.00015
    # Bid-ask spread (half-spread on each side)
    spread_frac: float = 0.0002
    # Slippage fraction
    slippage_frac: float = 0.0002
    # Whether costs apply symmetrically for short (simplified)
    short_same_as_long: bool = True

    def round_trip_cost(self, trade_value: float, direction: str = "long") -> float:
        """
        Compute total round-trip cost (entry + exit) for a given trade value.
        Returns cost as a fraction of trade value (not absolute INR).
        """
        # Per side: brokerage + spread + slippage
        per_side_direct = self.brokerage_frac + self.spread_frac + self.slippage_frac
        # Regulatory (simplified: apply on both sides)
        regulatory = self.stt_frac + self.exchange_txn_frac + self.sebi_frac
        gst_component = (self.brokerage_frac + self.exchange_txn_frac) * self.gst_rate
        # Stamp on entry only (buy side)
        stamp = self.stamp_frac if direction == "long" else 0.0

        # Round trip = 2 * per-side direct + regulatory round trip + stamp
        total_frac = 2 * per_side_direct + 2 * regulatory + 2 * gst_component + stamp
        return total_frac

    def one_way_cost(self, direction: str = "entry") -> float:
        """One-way cost fraction (for modelling entry and exit separately)."""
        c = self.brokerage_frac + self.spread_frac + self.slippage_frac
        c += self.stt_frac + self.exchange_txn_frac + self.sebi_frac
        c += (self.brokerage_frac + self.exchange_txn_frac) * self.gst_rate
        if direction == "entry":
            c += self.stamp_frac
        return c


# Pre-built profiles
ZERO_COST = CostProfile(
    label="zero",
    brokerage_frac=0,
    stt_frac=0,
    exchange_txn_frac=0,
    sebi_frac=0,
    gst_rate=0,
    stamp_frac=0,
    spread_frac=0,
    slippage_frac=0,
)

ETF_CONSERVATIVE = CostProfile(
    label="etf_conservative",
    brokerage_frac=0.0002,
    stt_frac=0.0001,
    exchange_txn_frac=0.0000345,
    sebi_frac=0.000001,
    gst_rate=0.18,
    stamp_frac=0.00015,
    spread_frac=0.0002,
    slippage_frac=0.0002,
)

ETF_AGGRESSIVE = CostProfile(
    label="etf_aggressive",
    brokerage_frac=0.0003,
    stt_frac=0.0001,
    exchange_txn_frac=0.0000345,
    sebi_frac=0.000001,
    gst_rate=0.18,
    stamp_frac=0.00015,
    spread_frac=0.0004,
    slippage_frac=0.0005,
)


def flat_cost_profile(bps: float) -> CostProfile:
    """Create a simple flat round-trip cost profile for sensitivity sweeps."""
    half = bps / 2 / 10000  # bps per side → fraction
    return CostProfile(
        label=f"flat_{bps}bps",
        brokerage_frac=half,
        stt_frac=0,
        exchange_txn_frac=0,
        sebi_frac=0,
        gst_rate=0,
        stamp_frac=0,
        spread_frac=half,
        slippage_frac=0,
    )


PROFILE_MAP = {
    "zero": ZERO_COST,
    "etf_conservative": ETF_CONSERVATIVE,
    "etf_aggressive": ETF_AGGRESSIVE,
}


def get_profile(name: str) -> CostProfile:
    if name not in PROFILE_MAP:
        raise ValueError(f"Unknown cost profile '{name}'. Available: {list(PROFILE_MAP)}")
    return PROFILE_MAP[name]
