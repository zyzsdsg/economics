"""Parsers that turn raw source files into tidy pandas objects.

Every reader returns data indexed by a ``DatetimeIndex`` named ``"date"`` and stores
``units`` and ``source`` in ``Series.attrs``. Readers do *not* clean economic
content (e.g. zero placeholders); they only parse the file layout faithfully.
Cleaning is a separate, documented step in :mod:`economic_lab.data.build`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _finalise(s: pd.Series, name: str, units: str, source: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna().astype("float64")
    s.index = pd.DatetimeIndex(s.index, name="date")
    s = s.sort_index()
    s.name = name
    s.attrs.update(units=units, source=source)
    return s


def read_eia_xls(path: Path, name: str = "brent_usd") -> pd.Series:
    """EIA ``hist_xls`` workbook: sheet ``Data 1``, two preamble rows, then Date | value."""
    df = pd.read_excel(path, sheet_name="Data 1", skiprows=2)
    date_col, value_col = df.columns[:2]
    s = df.set_index(pd.to_datetime(df[date_col]))[value_col]
    return _finalise(s, name, units="USD per barrel", source=f"EIA ({Path(path).name})")


def read_fred_csv(path: Path, name: str | None = None, units: str = "") -> pd.Series:
    """FRED ``fredgraph.csv``: a date column then one value column.

    Missing values appear as blanks (or ``"."`` in older files) and are dropped.
    """
    df = pd.read_csv(path, na_values=[".", ""])
    date_col, value_col = df.columns[:2]
    s = df.set_index(pd.to_datetime(df[date_col]))[value_col]
    return _finalise(s, name or value_col, units=units, source=f"FRED {value_col}")


def read_dallasfed_igrea(path: Path) -> pd.Series:
    """Dallas Fed Index of Global Real Economic Activity (monthly).

    Recent rows in the published file carry mid-month dates (e.g. ``2026-07-26``)
    instead of month starts, so dates are normalised to the first of the month.
    Duplicated months after normalisation raise an error rather than being merged.
    """
    df = pd.read_excel(path, header=None, skiprows=1).iloc[:, :2]
    df.columns = ["date", "igrea"]
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    if df["date"].duplicated().any():
        raise ValueError("igrea: more than one observation per month after normalisation")
    s = df.set_index("date")["igrea"]
    return _finalise(s, "igrea", units="percent deviation from trend", source="Dallas Fed IGREA")


def read_worldbank_pinksheet(path: Path, sheet: str = "Monthly Prices") -> pd.DataFrame:
    """World Bank Pink Sheet monthly prices.

    Layout: four title rows, a row of commodity names, a row of units, then rows
    whose first cell is ``YYYYMmm``. ``"…"`` marks missing values. Returns a wide
    DataFrame; units are stored in ``df.attrs["units"]`` keyed by column.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    first = raw.iloc[:, 0].astype(str)
    data_rows = first.str.fullmatch(r"\d{4}M\d{2}")
    start = int(np.argmax(data_rows.to_numpy()))
    names = raw.iloc[start - 2, 1:].astype(str).str.strip()
    units = raw.iloc[start - 1, 1:].astype(str).str.strip()
    body = raw.loc[data_rows, :]
    index = pd.to_datetime(body.iloc[:, 0].str.replace("M", "-"), format="%Y-%m")
    df = body.iloc[:, 1:].replace("…", np.nan).apply(pd.to_numeric, errors="coerce")
    df.columns = names.to_list()
    df.index = pd.DatetimeIndex(index, name="date")
    df.attrs["units"] = dict(zip(names, units, strict=True))
    df.attrs["source"] = "World Bank Pink Sheet"
    return df


def read_rbnz_xlsx(path: Path, sheet: str | int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse an RBNZ statistics workbook into ``(data, metadata)``.

    RBNZ workbooks have several descriptive header rows above the data, the last of
    which is labelled ``Series Id`` in the first column. This parser locates that row
    instead of hard-coding row numbers, because layouts have changed over time.

    Returns
    -------
    data : DataFrame
        Columns are RBNZ series ids; index is the observation date.
    metadata : DataFrame
        One row per series id; columns are the header-row labels found above the
        ``Series Id`` row (e.g. series description, unit, note).

    Notes
    -----
    Layout checked against M1, M5, M9, M14, B1 and B2 workbooks published in 2026:
    sheet ``Data``; two unlabelled rows (group, series name), then ``Notes``,
    ``Unit`` and ``Series Id`` rows; quarterly data are dated at quarter end. The
    unlabelled rows appear in ``metadata`` as ``header_0`` and ``header_1``.
    """
    book = pd.ExcelFile(path)
    if sheet is None:
        sheet = "Data" if "Data" in book.sheet_names else book.sheet_names[0]
    raw = pd.read_excel(book, sheet_name=sheet, header=None)

    labels = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    hits = np.flatnonzero(labels.to_numpy() == "series id")
    if len(hits) == 0:
        raise ValueError(f"{Path(path).name}: no 'Series Id' header row found")
    id_row = int(hits[0])

    ids = raw.iloc[id_row, 1:].astype(str).str.strip()
    meta = raw.iloc[:id_row, 1:].T
    meta.columns = [
        str(v).strip() if pd.notna(v) else f"header_{i}" for i, v in enumerate(raw.iloc[:id_row, 0])
    ]
    meta.index = pd.Index(ids, name="series_id")

    body = raw.iloc[id_row + 1 :, :]
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    body = body.loc[dates.notna()]
    data = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    data.columns = ids.to_list()
    data.index = pd.DatetimeIndex(dates.dropna(), name="date")
    return data, meta
