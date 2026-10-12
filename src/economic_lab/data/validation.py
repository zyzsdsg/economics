"""Data-quality checks for time series.

Checks return a :class:`ValidationReport` instead of raising, so that a notebook
can display every problem at once. Call ``report.raise_if_failed()`` in pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

ExpectedFreq = Literal["B", "W", "M", "Q", "A"]

# Typical spacing in days between consecutive observations and a tolerance band.
_SPACING_DAYS = {"B": (1, 5), "W": (7, 7), "M": (28, 31), "Q": (90, 92), "A": (365, 366)}


@dataclass
class ValidationReport:
    name: str
    n_obs: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError(f"{self.name}: " + "; ".join(self.errors))

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.name}: {self.n_obs} obs, {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"
                 if self.n_obs else f"[{status}] {self.name}: empty"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  WARNING {w}" for w in self.warnings]
        return "\n".join(lines)


def check_frequency(index: pd.DatetimeIndex, expected: ExpectedFreq) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` about date spacing and alignment.

    For business-day data, gaps longer than a week are only warnings: official daily
    series (e.g. EIA Brent in 1994-2006) do contain such holes, and aggregation with a
    coverage threshold handles them. For monthly/quarterly data any irregular spacing
    or misaligned date is an error, because it breaks lag and growth-rate calculations.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if len(index) < 2:
        return errors, warnings
    gaps = np.diff(index.values).astype("timedelta64[D]").astype(int)
    lo, hi = _SPACING_DAYS[expected]
    if expected == "B":
        if (index.dayofweek >= 5).any():
            errors.append(f"{int((index.dayofweek >= 5).sum())} weekend dates in business-day data")
        long_gaps = gaps > 7
        if long_gaps.any():
            warnings.append(
                f"{int(long_gaps.sum())} gaps longer than one week (max {int(gaps.max())} days)"
            )
        # Many short gaps can be as harmful as one long gap: check yearly coverage.
        years = pd.Series(1, index=index).groupby(index.year).size()
        full = years.index[(years.index > index[0].year) & (years.index < index[-1].year)]
        expected_days = pd.Series(
            {y: len(pd.bdate_range(f"{y}-01-01", f"{y}-12-31")) for y in full}, dtype=float
        )
        coverage = years.reindex(full) / expected_days
        sparse = coverage[coverage < 0.8]
        if len(sparse):
            warnings.append(
                f"{len(sparse)} years with <80% business-day coverage "
                f"({sparse.index.min()}-{sparse.index.max()}, min {sparse.min():.0%})"
            )
    else:
        bad = int(((gaps < lo) | (gaps > hi)).sum())
        if bad:
            errors.append(f"{bad} irregular spacings for {expected} data (expected {lo}-{hi} days)")
        if expected in ("M", "Q"):
            starts = index.to_period(expected).to_timestamp()
            if not (starts == index).all():
                errors.append(f"dates not aligned to {expected} period start")
    return errors, warnings


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column counts of missing values, first/last valid date and coverage share."""
    rows = {}
    for c in df.columns:
        s = df[c]
        rows[c] = {
            "n_missing": int(s.isna().sum()),
            "share_missing": float(s.isna().mean()),
            "first_valid": s.first_valid_index(),
            "last_valid": s.last_valid_index(),
        }
    return pd.DataFrame(rows).T


def validate_series(
    s: pd.Series,
    *,
    expected_freq: ExpectedFreq,
    min_value: float | None = None,
    max_value: float | None = None,
    min_start: str | None = None,
    name: str | None = None,
) -> ValidationReport:
    """Validate one series: index type, ordering, duplicates, frequency, range, coverage."""
    name = name or str(s.name)
    rep = ValidationReport(
        name, int(s.notna().sum()), s.first_valid_index(), s.last_valid_index()
    )
    if not isinstance(s.index, pd.DatetimeIndex):
        rep.errors.append("index is not a DatetimeIndex")
        return rep
    if s.index.has_duplicates:
        rep.errors.append(f"{int(s.index.duplicated().sum())} duplicate dates")
    if not s.index.is_monotonic_increasing:
        rep.errors.append("dates not sorted")
    if not pd.api.types.is_numeric_dtype(s):
        rep.errors.append(f"non-numeric dtype {s.dtype}")
        return rep
    valid = s.dropna()
    freq_errors, freq_warnings = check_frequency(valid.index, expected_freq)
    rep.errors += freq_errors
    rep.warnings += freq_warnings
    if min_value is not None and (valid < min_value).any():
        rep.errors.append(f"{int((valid < min_value).sum())} values below {min_value}")
    if max_value is not None and (valid > max_value).any():
        rep.errors.append(f"{int((valid > max_value).sum())} values above {max_value}")
    n_inner_missing = int(s.loc[s.first_valid_index() : s.last_valid_index()].isna().sum()) if len(valid) else 0
    if n_inner_missing:
        rep.warnings.append(f"{n_inner_missing} missing values inside the sample")
    if min_start is not None and len(valid) and valid.index[0] > pd.Timestamp(min_start):
        rep.warnings.append(f"sample starts {valid.index[0]:%Y-%m-%d}, later than required {min_start}")
    return rep


def seasonality_test(x: pd.Series, freq: Literal["M", "Q"]) -> dict[str, float]:
    """F-test that period dummies jointly explain ``x`` (e.g. q/q inflation).

    Regress x on a constant and (m-1) season dummies; under H0 (no seasonality) all
    dummy coefficients are zero. A small p-value in a series that is supposed to be
    seasonally adjusted is a red flag. NZ CPI is *not* seasonally adjusted by Stats NZ,
    so q/q CPI inflation is expected to show some seasonality.
    """
    from scipy import stats

    y = x.dropna()
    season = y.index.month if freq == "M" else y.index.quarter
    dummies = pd.get_dummies(season, drop_first=True, dtype=float).to_numpy()
    n, m = len(y), dummies.shape[1]
    X = np.column_stack([np.ones(n), dummies])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    rss_u = float(np.sum((y.to_numpy() - X @ beta) ** 2))
    rss_r = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    f = ((rss_r - rss_u) / m) / (rss_u / (n - m - 1))
    return {"F": f, "df1": m, "df2": n - m - 1, "p_value": float(stats.f.sf(f, m, n - m - 1))}
