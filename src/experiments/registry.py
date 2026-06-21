"""Experiment registry — records all experiments run with full lineage."""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.paths import experiments_dir
from src.utils.serialization import save_json, load_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _data_hash(df: pd.DataFrame) -> str:
    h = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:12]
    return h


class ExperimentRegistry:
    """
    Persists experiment records to JSON files in outputs/experiments/.
    Each run gets a unique ID and timestamp.
    Repeated runs with identical inputs + seed produce identical IDs if desired.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or experiments_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[dict] = []
        self._load_existing()

    def _load_existing(self) -> None:
        index_path = self.base_dir / "experiment_index.json"
        if index_path.exists():
            try:
                self._records = load_json(index_path)
            except Exception:
                self._records = []

    def _save_index(self) -> None:
        save_json(self._records, self.base_dir / "experiment_index.json")

    def log(
        self,
        experiment_id: str,
        description: str,
        feature_set: List[str],
        parameters: dict,
        split_boundaries: dict,
        cost_profile: str,
        seed: int,
        metrics: dict,
        data_hash: str = "",
        status: str = "completed",
        rejection_reason: Optional[str] = None,
        output_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]
        record = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "data_hash": data_hash,
            "feature_set": feature_set,
            "parameters": parameters,
            "split_boundaries": split_boundaries,
            "cost_profile": cost_profile,
            "seed": seed,
            "metrics": metrics,
            "status": status,
            "rejection_reason": rejection_reason,
            "output_paths": output_paths or {},
        }
        self._records.append(record)
        self._save_index()

        # Also save individual record
        run_path = self.base_dir / f"{experiment_id}_{run_id}.json"
        save_json(record, run_path)
        logger.info(f"Experiment logged: {experiment_id} [{run_id}] status={status}")
        return run_id

    def all_records(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records)

    def get_by_id(self, experiment_id: str) -> List[dict]:
        return [r for r in self._records if r["experiment_id"] == experiment_id]

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all experiments."""
        df = self.all_records()
        if df.empty:
            return df
        cols = ["experiment_id", "description", "timestamp", "status", "rejection_reason"]
        metric_cols = []
        if "metrics" in df.columns:
            metrics_df = pd.json_normalize(df["metrics"])
            for c in ["cagr", "sharpe_ratio", "max_drawdown", "trade_count"]:
                if c in metrics_df.columns:
                    df[c] = metrics_df[c].values
                    metric_cols.append(c)
        return df[[c for c in cols + metric_cols if c in df.columns]]
