"""Readers, validation, provenance and download safety, all offline."""

import json

import numpy as np
import pandas as pd
import pytest
import requests

from economic_lab.data import build, downloader, provenance, readers
from economic_lab.data.build import build_synthetic, write_datasets
from economic_lab.data.validation import missing_report, seasonality_test, validate_series


def test_fred_reader_handles_missing_markers(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("observation_date,DEXUSNZ\n2024-01-02,0.63\n2024-01-03,\n2024-01-04,.\n2024-01-05,0.64\n")
    s = readers.read_fred_csv(f, "nzd_usd")
    assert len(s) == 2 and s.index.name == "date" and s.dtype == "float64"


def _rbnz_workbook(path, ids, dates, values):
    """Workbook in the layout RBNZ uses (checked against hm1/hm5/hb2 published 2026):
    two unlabelled header rows (group, series), then Notes, Unit, Series Id, then data."""
    rows = [
        [None] + ["Consumers price index (CPI)"] * len(ids),
        [None] + [f"series {i}" for i in ids],
        ["Notes"] + [None] * len(ids),
        ["Unit"] + ["Index"] * len(ids),
        ["Series Id", *ids],
    ] + [[d, *v] for d, v in zip(dates, values, strict=True)]
    pd.DataFrame(rows).to_excel(path, sheet_name="Data", header=False, index=False)


def test_rbnz_reader_finds_series_id_row(tmp_path):
    f = tmp_path / "hm1.xlsx"
    dates = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")]
    _rbnz_workbook(f, ["CPI.A", "CPI.B"], dates, [[1000.0, 1000.0], [1010.0, 1005.0]])
    data, meta = readers.read_rbnz_xlsx(f)
    assert list(data.columns) == ["CPI.A", "CPI.B"]
    assert meta.loc["CPI.B", "Unit"] == "Index" and meta.loc["CPI.B", "header_1"] == "series CPI.B"
    assert data.loc["2024-06-30", "CPI.A"] == 1010.0


def test_rbnz_loader_redates_quarters_and_checks_ids(tmp_path, monkeypatch):
    dates = pd.date_range("2023-03-31", periods=6, freq="QE")
    _rbnz_workbook(tmp_path / "hm1.xlsx", ["CPI.Q.C.ia"], dates, [[1000.0 + i] for i in range(6)])
    monkeypatch.setattr(build, "_raw", lambda name: tmp_path / "hm1.xlsx")
    config = {"rbnz_series": {"cpi": {"source": "rbnz_cpi", "series_id": "CPI.Q.C.ia"}}}
    monkeypatch.setattr(build, "_load_variable_config", lambda: config)
    notes, reports = [], []
    frames = build._rbnz(notes, reports)
    q = frames["Q"]
    assert frames["B"].empty and frames["M"].empty and list(q.columns) == ["cpi"]
    assert q.index[0] == pd.Timestamp("2023-01-01")  # 2023Q1 dated at quarter start
    assert all(r.passed for r in reports)
    config["rbnz_series"]["cpi"]["series_id"] = "NOT.AN.ID"
    with pytest.raises(KeyError):
        build._rbnz([], [])


def test_rbnz_loader_joins_daily_history_files(tmp_path, monkeypatch):
    old_d, new_d = pd.bdate_range("2017-12-20", "2017-12-29"), pd.bdate_range("2018-01-03", "2018-01-12")
    _rbnz_workbook(tmp_path / "old.xlsx", ["INM.DP1.N"], old_d, [[1.75]] * len(old_d))
    _rbnz_workbook(tmp_path / "new.xlsx", ["INM.DP1.N"], new_d, [[1.75]] * len(new_d))
    files = {"rbnz_interest_rates": "new.xlsx", "rbnz_interest_rates_hist": "old.xlsx"}
    monkeypatch.setattr(build, "_raw", lambda name: tmp_path / files[name])
    spec = {"source": "rbnz_interest_rates", "series_id": "INM.DP1.N",
            "history": {"source": "rbnz_interest_rates_hist", "series_id": "INM.DP1.N"}}
    monkeypatch.setattr(build, "_load_variable_config", lambda: {"rbnz_series": {"ocr": spec}})
    daily = build._rbnz([], [])["B"]
    assert daily["ocr"].first_valid_index() == old_d[0]  # history kept, not cut at 2018
    assert daily["ocr"].notna().sum() == len(old_d) + len(new_d)


def test_splice_check_accepts_same_definition_rejects_month_end():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    rng = np.random.default_rng(1)
    daily = pd.Series(70 + np.cumsum(0.1 * rng.standard_normal(len(idx))), index=idx, name="twi")
    avg = daily.groupby(daily.index.to_period("M")).mean()
    avg.index = avg.index.to_timestamp()
    last = daily.groupby(daily.index.to_period("M")).last()
    last.index = last.index.to_timestamp()
    longer = pd.concat([pd.Series([69.0], index=[pd.Timestamp("2019-12-01")]), avg.round(2)])
    assert build.splice_check(longer.rename("twi"), avg) < 1e-3  # published rounding only
    with pytest.raises(ValueError):
        build.splice_check(last.rename("twi"), avg)  # month-end is a different definition


def test_join_vintages_reports_seam_and_rejects_conflicts():
    old = pd.Series([1.75, 1.75, 1.75], index=pd.bdate_range("2017-12-27", periods=3), name="ocr")
    new = pd.Series([1.75, 2.00], index=pd.bdate_range("2018-01-03", periods=2), name="ocr")
    joined, jump = build.join_vintages(new, old)  # order of arguments does not matter
    assert len(joined) == 5 and joined.index.is_monotonic_increasing and jump == 0.0
    clash = pd.Series([9.0], index=[old.index[-1]], name="ocr")
    with pytest.raises(ValueError):
        build.join_vintages(old, clash)


def test_validation_flags_duplicates_range_and_alignment():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-07-15"])
    rep = validate_series(pd.Series([1.0, -1.0, 2.0], index=idx), expected_freq="Q", min_value=0)
    assert not rep.passed
    text = " ".join(rep.errors)
    assert "duplicate" in text and "below" in text and "aligned" in text


def test_validation_passes_clean_quarterly():
    s = pd.Series(np.arange(1, 9.0), index=pd.date_range("2020-01-01", periods=8, freq="QS"))
    assert validate_series(s, expected_freq="Q", min_value=0).passed


def test_missing_report_counts():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, 2, 3]})
    rep = missing_report(df)
    assert rep.loc["a", "n_missing"] == 1 and rep.loc["b", "n_missing"] == 0


def test_seasonality_test_detects_quarterly_pattern():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=80, freq="QS")
    seasonal = pd.Series(np.tile([0.0, 1.0, 0.0, -1.0], 20) + 0.2 * rng.standard_normal(80), index=idx)
    noise = pd.Series(rng.standard_normal(80), index=idx)
    assert seasonality_test(seasonal, "Q")["p_value"] < 0.01
    assert seasonality_test(noise, "Q")["p_value"] > 0.01


def test_manifest_records_and_keeps_history(tmp_path):
    f = tmp_path / "raw" / "a.csv"
    f.parent.mkdir()
    f.write_text("v1")
    m = tmp_path / "raw" / "manifest.json"
    e1 = provenance.record_file(m, "a", f, url="http://x", method="manual")
    assert e1["sha256"] == provenance.sha256_file(f) and e1["path"] == "a.csv"
    f.write_text("v2")
    provenance.record_file(m, "a", f, url="http://x", method="manual")
    entry = json.loads(m.read_text())["files"]["a"]
    assert len(entry["history"]) == 1 and entry["history"][0]["sha256"] == e1["sha256"]
    assert provenance.verify_file(m, "a", f)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body, self.status_code, self.headers = body, status, {"ETag": "abc"}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_rejects_http_error_and_html(tmp_path, monkeypatch):
    target = tmp_path / "hm1.xlsx"
    monkeypatch.setattr(downloader.requests, "get", lambda *a, **k: _FakeResponse(b"<html>", 403))
    with pytest.raises(requests.HTTPError):
        downloader.download_data("http://x", target)
    monkeypatch.setattr(downloader.requests, "get", lambda *a, **k: _FakeResponse(b"<!DOCTYPE html>"))
    with pytest.raises(ValueError):
        downloader.download_data("http://x", target)
    assert not target.exists() and not list(tmp_path.glob("*.part"))


def test_download_archives_previous_vintage(tmp_path, monkeypatch):
    target = tmp_path / "x.csv"
    target.write_text("old")
    monkeypatch.setattr(downloader.requests, "get", lambda *a, **k: _FakeResponse(b"date,v\n"))
    downloader.download_data("http://x", target)  # exists, no overwrite -> untouched
    assert target.read_text() == "old"
    downloader.download_data("http://x", target, overwrite=True, manifest=tmp_path / "m.json")
    assert target.read_text() == "date,v\n"
    assert [p.read_text() for p in (tmp_path / "_vintages").iterdir()] == ["old"]


def test_synthetic_build_roundtrip(tmp_path):
    res = build_synthetic()
    paths = write_datasets(res, tmp_path)
    q = pd.read_parquet(paths["quarterly"])
    meta = json.loads(paths["quarterly"].with_suffix(".meta.json").read_text())
    assert meta["data_kind"] == "synthetic" and "b0" in meta["true_params"]
    assert {"cpi", "cpi_tradable", "cpi_nontradable", "oil_nzd"} <= set(q.columns)
    assert validate_series(q["cpi"], expected_freq="Q", min_value=0).passed
