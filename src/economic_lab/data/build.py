"""Build the standardised datasets ``macro_{daily,monthly,quarterly}.parquet``.

Pipeline: raw file -> reader -> cleaning (documented below) -> validation ->
frequency alignment -> parquet + JSON sidecar describing provenance.

Cleaning rules (each one is logged in the sidecar's ``notes``)
---------------------------------------------------------------
1. OECD/FRED NZ CPI contains ``0.0`` placeholders before 1926: values <= 0 are set
   to missing and only the longest gap-free quarterly run up to the latest
   observation is kept.
2. Dallas Fed IGREA dates are normalised to month starts (see reader).
3. Daily NZD oil prices are computed only on dates where both Brent and the
   exchange rate are observed. No forward-filling.
4. Monthly/quarterly averages of daily data require >= 80% business-day coverage,
   which drops the incomplete latest period.
5. EIA's *daily* Brent series is sparse in 1993-2005 (e.g. 26 of 65 business days in
   1995Q1), so averaging it would leave dozens of missing quarters. The headline
   monthly/quarterly ``brent_usd`` therefore uses EIA's published *monthly* averages,
   and ``oil_nzd = brent_usd_m / nzd_usd_m`` is formed monthly before averaging to
   quarters. ``oil_nzd_from_daily`` (mean of the daily ratio, where coverage allows)
   is kept as a cross-check; the two differ only by the small covariance between
   intra-month oil and exchange-rate movements.
6. RBNZ series (``rbnz_series`` in ``configs/variables.yaml``) are re-dated from
   period end to period start. Daily B1/B2 series are averaged like rule 4, so
   quarterly OCR is the average rate over the quarter, not the end-of-quarter rate.
   Where a published monthly table has the same definition and a longer history
   (B1 TWI: monthly averages from 1999), it replaces the aggregated daily series at
   monthly/quarterly frequency after ``splice_check`` confirms they agree. Note that
   in the B2 *monthly* table the OCR is a month-end value while bank bills and bonds
   are monthly averages, so that table is not spliced.
7. Headline CPI comes from RBNZ M1. The OECD/FRED CPI is used only if M1 is missing;
   otherwise it serves as a cross-check of q/q inflation (recorded in the notes).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from economic_lab.data import readers, synthetic, transforms
from economic_lab.data.provenance import load_manifest
from economic_lab.data.sources import SOURCES
from economic_lab.data.validation import validate_series
from economic_lab.utils.paths import manifest_path, processed_data_dir, project_root, raw_data_dir

Frequency = Literal["daily", "monthly", "quarterly"]


@dataclass
class BuildResult:
    daily: pd.DataFrame
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    data_kind: str
    notes: list[str] = field(default_factory=list)
    reports: list = field(default_factory=list)


def _raw(name: str) -> Path:
    return raw_data_dir() / SOURCES[name]["filename"]


def _load_variable_config() -> dict:
    with open(project_root() / "configs" / "variables.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rbnz(notes: list[str], reports: list) -> dict[str, pd.DataFrame]:
    """RBNZ series listed under ``rbnz_series`` in ``configs/variables.yaml``.

    Returns frames keyed by ``"Q"``, ``"M"`` and ``"B"`` (business days). Quarterly
    (M1, M5, M9, M14) and monthly (B1 monthly) tables are re-dated to period start;
    daily tables (B1, B2) are returned as published. An entry may name a
    ``long_history`` monthly series with the same definition as its daily series
    (e.g. TWI monthly averages back to 1999); it is returned under the same name in
    the ``"M"`` frame. An entry may also name a ``history`` series: the same series
    id in an earlier file at the same frequency (e.g. B2 daily 1985-2017); the two
    are joined with :func:`join_vintages`. Sources whose file is missing are skipped
    with a note.
    """
    by_source: dict[str, dict[str, str]] = {}
    for var, spec in _load_variable_config()["rbnz_series"].items():
        by_source.setdefault(spec["source"], {})[spec["series_id"]] = var
        for key in ("long_history", "history"):
            if key in spec:
                h = spec[key]
                by_source.setdefault(h["source"], {})[h["series_id"]] = var

    frames: dict[str, list[pd.DataFrame]] = {"Q": [], "M": [], "B": []}
    for source, ids in by_source.items():
        path = _raw(source)
        if not path.exists():
            notes.append(f"RBNZ {source} not used: {path.name} missing.")
            continue
        data, _ = readers.read_rbnz_xlsx(path)
        missing = sorted(set(ids) - set(data.columns))
        if missing:
            raise KeyError(f"{path.name}: series ids not found: {missing}")
        out = data[list(ids)].rename(columns=ids).dropna(how="all")
        freq = {"quarterly": "Q", "monthly": "M", "daily": "B"}[SOURCES[source]["frequency"]]
        if freq != "B":
            out.index = transforms.to_period_start(out.index, freq)
        frames[freq].append(out)
        for col in out.columns:
            reports.append(validate_series(out[col].dropna(), expected_freq=freq, name=col))
        notes.append(
            f"RBNZ {path.name}: {', '.join(out.columns)} "
            f"({out.index.min():%Y-%m} to {out.index.max():%Y-%m})."
        )
    out: dict[str, pd.DataFrame] = {}
    for freq, parts in frames.items():
        merged = pd.DataFrame()
        for part in parts:
            for col in part.columns:
                if col in merged.columns:
                    joined, jump = join_vintages(merged[col], part[col])
                    merged = merged.reindex(merged.index.union(joined.index))
                    merged[col] = joined
                    notes.append(f"{col}: joined two RBNZ files; change across the seam {jump:+.3f}.")
                else:
                    merged = merged.join(part[col], how="outer") if not merged.empty else part[[col]]
        out[freq] = merged
    return out


def join_vintages(a: pd.Series, b: pd.Series, tol: float = 1e-9) -> tuple[pd.Series, float]:
    """Join two files holding the same series for different date ranges.

    Overlapping dates must agree (otherwise the files are not the same series).
    Returns the joined series and the change between the last observation of the
    earlier file and the first of the later one, so a level break at the seam can be
    spotted in the build notes.
    """
    a, b = a.dropna(), b.dropna()
    common = a.index.intersection(b.index)
    if len(common) and (a[common] - b[common]).abs().max() > tol:
        raise ValueError(f"{a.name}: files disagree on {len(common)} overlapping dates")
    first, second = (a, b) if a.index.min() <= b.index.min() else (b, a)
    seam = second.index[second.index > first.index.max()]
    jump = float(second[seam[0]] - first.iloc[-1]) if len(seam) else 0.0
    return first.combine_first(second).rename(a.name), jump


def splice_check(long: pd.Series, short: pd.Series, rel_tol: float = 1e-3) -> float:
    """Max relative gap between two versions of a series over their overlap.

    Used before replacing an aggregated daily series by a longer published monthly
    one: both must measure the same thing (e.g. a monthly *average*, not a month-end
    value). Raises ``ValueError`` if the gap exceeds ``rel_tol`` or there is no overlap.
    """
    a, b = long.align(short, join="inner")
    ok = a.notna() & b.notna()
    if not ok.any():
        raise ValueError(f"{long.name}: no overlap to check the splice")
    gap = float(((a[ok] - b[ok]).abs() / b[ok].abs()).max())
    if gap > rel_tol:
        raise ValueError(f"{long.name}: monthly and daily-average versions differ by {gap:.2%}")
    return gap


def build_from_raw() -> BuildResult:
    """Build datasets from files in ``data/econ_data``. Raises if core files are missing."""
    notes: list[str] = []
    reports = []

    brent = readers.read_eia_xls(_raw("brent_daily"))
    fx = readers.read_fred_csv(_raw("nzd_usd_daily"), "nzd_usd", units="USD per NZD")
    reports += [
        validate_series(brent, expected_freq="B", min_value=0.0, max_value=500.0),
        validate_series(fx, expected_freq="B", min_value=0.2, max_value=1.6),
    ]
    oil_nzd = transforms.convert_currency(brent, fx).rename("oil_nzd")
    daily = pd.concat([brent, fx, oil_nzd], axis=1)
    daily.index.name = "date"

    brent_m = readers.read_eia_xls(_raw("brent_monthly"))
    brent_m.index = transforms.to_period_start(brent_m.index, "M")
    reports.append(validate_series(brent_m, expected_freq="M", min_value=0.0, max_value=500.0))
    fx_m = transforms.aggregate(fx, "M", "mean")
    monthly = pd.DataFrame({"brent_usd": brent_m, "nzd_usd": fx_m})
    monthly["oil_nzd"] = monthly["brent_usd"] / monthly["nzd_usd"]
    monthly["oil_nzd_from_daily"] = transforms.aggregate(oil_nzd, "M", "mean")

    def to_quarter(s: pd.Series) -> pd.Series:
        """Quarterly mean, only for quarters with all three months observed."""
        g = s.dropna().groupby(s.dropna().index.to_period("Q")).agg(["mean", "size"])
        out = g.loc[g["size"] == 3, "mean"]
        out.index = out.index.to_timestamp()
        return out.rename(s.name)

    quarterly = pd.DataFrame({c: to_quarter(monthly[c]) for c in ("brent_usd", "nzd_usd", "oil_nzd")})
    quarterly["oil_nzd_from_daily"] = transforms.aggregate(oil_nzd, "Q", "mean")
    notes.append(
        "brent_usd monthly/quarterly from EIA monthly averages (daily series sparse in 1993-2005); "
        "oil_nzd = monthly brent_usd / monthly mean nzd_usd, quarterly = mean of 3 months."
    )

    if _raw("dallasfed_igrea").exists():
        igrea = readers.read_dallasfed_igrea(_raw("dallasfed_igrea"))
        reports.append(validate_series(igrea, expected_freq="M"))
        monthly = monthly.join(igrea, how="outer")
        quarterly = quarterly.join(to_quarter(igrea), how="outer")
        notes.append("IGREA dates normalised to month start; quarterly = mean of 3 months.")

    rbnz = _rbnz(notes, reports)
    rbnz_q, rbnz_m, rbnz_d = rbnz["Q"], rbnz["M"], rbnz["B"]
    if not rbnz_d.empty:
        daily = daily.join(rbnz_d, how="outer")
        for col in rbnz_d.columns:
            monthly[col] = transforms.aggregate(rbnz_d[col], "M", "mean")
            quarterly[col] = transforms.aggregate(rbnz_d[col], "Q", "mean")
        notes.append(
            "RBNZ daily series (B1/B2) averaged over business days to months/quarters "
            "(>= 80% coverage); OCR and yields are period averages, not end-of-period."
        )
    for col in rbnz_m.columns:
        long = rbnz_m[col].dropna()
        if col in monthly.columns:
            gap = splice_check(long, monthly[col])
            notes.append(
                f"{col}: monthly/quarterly values from the RBNZ monthly table "
                f"({long.index[0]:%Y-%m}-); matches the daily average within {gap:.3%}."
            )
        monthly = monthly.drop(columns=col, errors="ignore").join(long, how="outer")
        quarterly = quarterly.drop(columns=col, errors="ignore").join(to_quarter(long), how="outer")

    if _raw("nz_cpi_oecd").exists():
        oecd = readers.read_fred_csv(_raw("nz_cpi_oecd"), "cpi", units="index 2015=100")
        n_zero = int((oecd <= 0).sum())
        oecd = transforms.contiguous_tail(oecd.where(oecd > 0), "Q")
        oecd.index = transforms.to_period_start(oecd.index, "Q")
        reports.append(validate_series(oecd, expected_freq="Q", min_value=0.01))
        if "cpi" in rbnz_q.columns:
            # Cross-check: both are Stats NZ CPI, so q/q inflation should agree up to rounding.
            diff = (transforms.pct_change(rbnz_q["cpi"].dropna(), 1, "Q")
                    - transforms.pct_change(oecd, 1, "Q")).dropna()
            notes.append(
                f"Headline CPI from RBNZ M1. Cross-check vs OECD/FRED CPI q/q inflation over "
                f"{len(diff)} common quarters: max abs difference {diff.abs().max():.2f} pp."
            )
        else:
            rbnz_q["cpi"] = oecd
            notes.append(
                f"Headline CPI from OECD MEI via FRED (fallback, RBNZ M1 missing). Removed {n_zero} "
                f"zero placeholders; kept contiguous run {oecd.index[0]:%Y-%m} to {oecd.index[-1]:%Y-%m}."
            )
    if not rbnz_q.empty:
        quarterly = quarterly.join(rbnz_q, how="outer")

    for rep in reports:
        rep.raise_if_failed()

    quarterly.index.name = monthly.index.name = "date"
    for df in (daily, monthly, quarterly):
        df.attrs["data_kind"] = "real"
    return BuildResult(daily, monthly, quarterly, "real", notes, reports)


def build_synthetic(seed: int = 42) -> BuildResult:
    """Build the offline synthetic datasets (same column names as the real ones)."""
    params = synthetic.SyntheticParams()
    daily = synthetic.make_synthetic_daily(params, seed)
    monthly = pd.DataFrame({c: transforms.aggregate(daily[c], "M") for c in daily.columns})
    quarterly = synthetic.make_synthetic_quarterly(params, seed)
    monthly.attrs["data_kind"] = "synthetic"
    return BuildResult(daily, monthly, quarterly, "synthetic", ["Synthetic DGP, see economic_lab.data.synthetic."])


def write_datasets(result: BuildResult, out_dir: Path | None = None) -> dict[str, Path]:
    """Write parquet files plus a ``.meta.json`` sidecar for each frequency."""
    out_dir = out_dir or processed_data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path())["files"]
    paths = {}
    for freq in ("daily", "monthly", "quarterly"):
        df: pd.DataFrame = getattr(result, freq)
        path = out_dir / f"macro_{freq}.parquet"
        df.to_parquet(path)
        meta = {
            "data_kind": result.data_kind,
            "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_rows": len(df),
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "columns": list(df.columns),
            "notes": result.notes,
            "raw_inputs": {k: v["sha256"] for k, v in manifest.items()}
            if result.data_kind == "real"
            else {},
        }
        if result.data_kind == "synthetic":
            meta["true_params"] = result.quarterly.attrs.get("true_params", {})
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths[freq] = path
    return paths


def load_dataset(
    frequency: Frequency = "quarterly", prefer: Literal["auto", "real", "synthetic"] = "auto"
) -> pd.DataFrame:
    """Load a standardised dataset.

    ``prefer="auto"`` uses ``data/processed/macro_{frequency}.parquet`` if it exists,
    otherwise falls back to synthetic data. The returned frame always has
    ``attrs["data_kind"]`` set to ``"real"`` or ``"synthetic"``; notebooks print it
    so synthetic results are never mistaken for empirical findings.

    Setting the environment variable ``ECONOMIC_LAB_FORCE_SYNTHETIC=1`` makes
    ``"auto"`` behave like ``"synthetic"`` (used by the notebook smoke tests).
    """
    if prefer == "auto" and os.environ.get("ECONOMIC_LAB_FORCE_SYNTHETIC") == "1":
        prefer = "synthetic"
    path = processed_data_dir() / f"macro_{frequency}.parquet"
    meta_path = path.with_suffix(".meta.json")
    if prefer != "synthetic" and path.exists():
        df = pd.read_parquet(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        df.attrs["data_kind"] = meta.get("data_kind", "unknown")
        df.attrs["notes"] = meta.get("notes", [])
        return df
    if prefer == "real":
        raise FileNotFoundError(f"{path} not found; run scripts/build_datasets.py")
    result = build_synthetic()
    df = getattr(result, frequency)
    df.attrs["data_kind"] = "synthetic"
    return df
