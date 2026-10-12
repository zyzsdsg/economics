"""Download raw files safely and record their provenance."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import requests

from economic_lab.data.provenance import record_file

USER_AGENT = "economic-lab/0.1 (academic research; https://github.com/zyzsdsg/economics)"

# Magic bytes let us reject an HTML error page saved under a spreadsheet name.
_SIGNATURES: dict[str, bytes] = {
    ".xlsx": b"PK\x03\x04",  # zip container
    ".xls": b"\xd0\xcf\x11\xe0",  # OLE2 compound document
}


def check_file_format(file_path: Path, suffix: str | None = None) -> None:
    """Raise ``ValueError`` if the file content does not match the expected type.

    Parameters
    ----------
    file_path : Path
        File to inspect.
    suffix : str, optional
        Expected type (``".xlsx"``, ``".xls"``, ``".csv"``). Defaults to the file's
        own suffix; pass it explicitly when checking a temporary ``.part`` file.
    """
    suffix = (suffix or file_path.suffix).lower()
    head = file_path.read_bytes()[:512]
    if suffix in _SIGNATURES and not head.startswith(_SIGNATURES[suffix]):
        raise ValueError(f"{file_path.name}: content is not a valid {suffix} file")
    if suffix == ".csv" and head.lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise ValueError(f"{file_path.name}: received an HTML page instead of CSV")


def discover_url(landing_page: str, pattern: str, timeout: float = 60) -> str:
    """Find the first link on ``landing_page`` that matches the regex ``pattern``."""
    resp = requests.get(landing_page, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    match = re.search(pattern, resp.text)
    if match is None:
        raise LookupError(f"No link matching {pattern!r} on {landing_page}")
    return match.group(0)


def download_data(
    url: str,
    file_path: Path,
    *,
    overwrite: bool = False,
    timeout: float = 120,
    manifest: Path | None = None,
    source_name: str | None = None,
) -> Path:
    """Download ``url`` to ``file_path`` and return the path.

    Parameters
    ----------
    url : str
        File URL.
    file_path : Path
        Destination. Parent directories are created.
    overwrite : bool, default False
        If False and the file exists, nothing is downloaded (original behaviour).
        If True, the existing file is first moved to ``_vintages/`` with a
        timestamp, so earlier data vintages are never destroyed.
    timeout : float
        Seconds to wait for the server.
    manifest : Path, optional
        If given, the download is recorded there (URL, time, SHA-256, HTTP headers).
    source_name : str, optional
        Key used in the manifest; defaults to the file stem.

    Notes
    -----
    The body is streamed to a ``.part`` file and renamed only after the HTTP status
    and file format have been checked, so an interrupted or failed download never
    leaves a corrupt file that would later be mistaken for real data.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        print(f"Data already exists at {file_path}")
        return file_path

    tmp = file_path.with_name(file_path.name + ".part")
    try:
        with requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True
        ) as response:
            response.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            headers = {
                k: response.headers[k]
                for k in ("Last-Modified", "ETag", "Content-Type")
                if k in response.headers
            }
        check_file_format(tmp, suffix=file_path.suffix)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if file_path.exists():
        _archive_vintage(file_path)
    tmp.replace(file_path)
    print(f"Downloaded data to {file_path}")

    if manifest is not None:
        record_file(
            manifest,
            source_name or file_path.stem,
            file_path,
            url=url,
            method="automatic",
            extra={"http_headers": headers},
        )
    return file_path


def _archive_vintage(file_path: Path) -> Path:
    """Move an existing raw file to ``_vintages/`` with a UTC timestamp suffix."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    vintage_dir = file_path.parent / "_vintages"
    vintage_dir.mkdir(exist_ok=True)
    target = vintage_dir / f"{file_path.stem}_{stamp}{file_path.suffix}"
    file_path.replace(target)
    return target
