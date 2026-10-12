"""Data provenance: checksums and a JSON manifest of every raw file.

Official macro data are revised. To make results reproducible we record, for each
raw file, where it came from, when it was obtained, and a SHA-256 checksum. If the
checksum changes after a refresh, the source has published a new vintage.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest: Path) -> dict[str, Any]:
    """Load the manifest, returning an empty one if the file does not exist yet."""
    if not manifest.exists():
        return {"files": {}}
    with open(manifest, encoding="utf-8") as f:
        return json.load(f)


def record_file(
    manifest: Path,
    name: str,
    file_path: Path,
    *,
    url: str | None,
    method: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add or update the manifest entry for ``name`` and return it.

    ``method`` is ``"automatic"`` for scripted downloads and ``"manual"`` for files
    placed by hand (e.g. RBNZ spreadsheets, which may not be fetched automatically).
    Previous entries are kept under ``history`` so earlier vintages stay documented.
    """
    data = load_manifest(manifest)
    root = manifest.parent
    try:
        rel = str(file_path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(file_path)
    entry: dict[str, Any] = {
        "path": rel,
        "url": url,
        "method": method,
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }
    if extra:
        entry.update(extra)

    old = data["files"].get(name)
    if old is not None:
        history = old.pop("history", [])
        if old.get("sha256") != entry["sha256"]:
            history.append(old)
        entry["history"] = history

    data["files"][name] = entry
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return entry


def verify_file(manifest: Path, name: str, file_path: Path) -> bool:
    """True if ``file_path`` still matches the checksum recorded for ``name``."""
    entry = load_manifest(manifest)["files"].get(name)
    return entry is not None and entry["sha256"] == sha256_file(file_path)
