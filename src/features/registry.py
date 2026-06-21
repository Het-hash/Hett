"""Feature registry: build all feature families into one DataFrame."""
from __future__ import annotations

import pandas as pd

from src.features.returns import build_return_features
from src.features.trend import build_trend_features
from src.features.momentum import build_momentum_features
from src.features.mean_reversion import build_mean_reversion_features
from src.features.volatility import build_volatility_features
from src.features.breakouts import build_breakout_features
from src.features.gaps import build_gap_features
from src.features.calendar import build_calendar_features
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_all_features(df: pd.DataFrame, volume_reliable: bool = False) -> pd.DataFrame:
    """
    Compute all causal features and return as a single DataFrame aligned to df's index.

    volume_reliable=False (default for ^NSEI): volume-dependent signals are excluded.
    """
    families = [
        ("returns", build_return_features),
        ("trend", build_trend_features),
        ("momentum", build_momentum_features),
        ("mean_reversion", build_mean_reversion_features),
        ("volatility", build_volatility_features),
        ("breakouts", build_breakout_features),
        ("gaps", build_gap_features),
        ("calendar", build_calendar_features),
    ]

    parts = []
    for name, builder in families:
        try:
            feat = builder(df)
            parts.append(feat)
            logger.debug(f"Built {len(feat.columns)} {name} features")
        except Exception as exc:
            logger.warning(f"Feature family '{name}' failed: {exc}")

    features = pd.concat(parts, axis=1)

    # Deduplicate columns that might appear in multiple families
    features = features.loc[:, ~features.columns.duplicated()]

    if not volume_reliable:
        vol_cols = [c for c in features.columns if "volume" in c.lower() or "vwap" in c.lower()]
        if vol_cols:
            features = features.drop(columns=vol_cols)
            logger.info(f"Dropped {len(vol_cols)} volume-dependent features (^NSEI volume unreliable)")

    logger.info(f"Total features built: {len(features.columns)}")
    return features
