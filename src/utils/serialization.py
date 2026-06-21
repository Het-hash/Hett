"""Serialization helpers for DataFrames, dicts, and experiment records."""
from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pandas as pd


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return super().default(obj)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_Encoder)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_csv(path: Path, parse_dates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=parse_dates)
    return df
