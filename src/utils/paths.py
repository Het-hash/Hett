"""Centralised path resolution for the project."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root regardless of where code is called from."""
    return Path(__file__).resolve().parent.parent.parent


def data_raw_dir() -> Path:
    return project_root() / "data" / "raw"


def data_processed_dir() -> Path:
    return project_root() / "data" / "processed"


def data_metadata_dir() -> Path:
    return project_root() / "data" / "metadata"


def config_dir() -> Path:
    return project_root() / "config"


def outputs_dir() -> Path:
    return project_root() / "outputs"


def figures_dir() -> Path:
    return outputs_dir() / "figures"


def tables_dir() -> Path:
    return outputs_dir() / "tables"


def experiments_dir() -> Path:
    return outputs_dir() / "experiments"


def reports_dir() -> Path:
    return outputs_dir() / "reports"


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in [
        data_raw_dir(),
        data_processed_dir(),
        data_metadata_dir(),
        figures_dir(),
        tables_dir(),
        experiments_dir(),
        reports_dir(),
    ]:
        d.mkdir(parents=True, exist_ok=True)
