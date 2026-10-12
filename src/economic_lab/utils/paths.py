"""Project paths.

All code and notebooks resolve data locations through these helpers, so that
nothing depends on the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root.

    Resolution order:
    1. ``ECONOMIC_LAB_ROOT`` environment variable (useful in CI or unusual layouts).
    2. The first parent of this file that contains ``pyproject.toml``.
    """
    env = os.environ.get("ECONOMIC_LAB_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate project root; set ECONOMIC_LAB_ROOT.")


def raw_data_dir() -> Path:
    """Raw downloads, exactly as published by the source (never edited in place)."""
    return project_root() / "data" / "econ_data"


def processed_data_dir() -> Path:
    """Standardised datasets built from raw files (``macro_*.parquet``)."""
    return project_root() / "data" / "processed"


def manifest_path() -> Path:
    """JSON provenance manifest for raw downloads (URL, timestamp, SHA-256)."""
    return raw_data_dir() / "manifest.json"
