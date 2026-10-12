"""Unit-explicit transformations for macro time series.

Conventions used everywhere in the project
-------------------------------------------
- Monthly and quarterly series are indexed by the **first day of the period**
  (``2024-10-01`` is 2024Q4), whatever convention the source uses.
- ``pct_change``   : 100 * (x_t / x_{t-k} - 1)            -> percent
- ``log_diff``     : 100 * (ln x_t - ln x_{t-k})          -> approx. percent (log points)
- ``pp_change``    : r_t - r_{t-k} for a series that is already a rate -> percentage points
- ``annualise``    : (1 + g/100)^m - 1, compounding a per-period growth rate
- q/q inflation is ``pct_change(cpi, 1)`` on quarterly data; y/y is ``pct_change(cpi, 4)``.

Functions that use ``shift`` require a *regular* index (no missing periods),
because shifting a gappy series silently compares the wrong dates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Freq = Literal["M", "Q"]
_PERIODS_PER_YEAR = {"M": 12, "Q": 4}


def to_period_start(index: pd.DatetimeIndex, freq: Freq) -> pd.DatetimeIndex:
    """Map any date within a period to the first day of that period."""
    return pd.DatetimeIndex(index.to_period(freq).to_timestamp(), name=index.name)


def require_regular(index: pd.DatetimeIndex, freq: Freq) -> None:
    """Raise if ``index`` is not a gap-free, duplicate-free sequence of periods."""
    periods = index.to_period(freq)
    if periods.has_duplicates:
        raise ValueError("index has duplicate periods")
    if len(periods) > 1:
        expected = pd.period_range(periods.min(), periods.max(), freq=freq)
        if len(expected) != len(periods) or not (np.sort(periods.asi8) == expected.asi8).all():
            raise ValueError(f"index has missing {freq} periods; reindex before shifting")


def convert_currency(price_usd: pd.Series, usd_per_nzd: pd.Series) -> pd.Series:
    """Convert a USD price to NZD: ``price_nzd = price_usd / usd_per_nzd``.

    ``usd_per_nzd`` is the NZD/USD quote (USD received for one NZD), e.g. 0.60.
    If 1 NZD buys 0.60 USD then 1 USD costs 1/0.60 NZD, so dividing converts USD to
    NZD. Only dates present in both series are kept (inner join); no forward-fill,
    because filling exchange rates across holidays is a modelling choice that
    should be made explicitly.
    """
    if (usd_per_nzd <= 0).any():
        raise ValueError("exchange rate must be strictly positive")
    price, fx = price_usd.align(usd_per_nzd, join="inner")
    out = price / fx
    out.name = f"{price_usd.name or 'price'}_nzd"
    return out


def aggregate(
    series: pd.Series,
    freq: Freq,
    how: Literal["mean", "last", "first", "sum", "max", "min"] = "mean",
    min_coverage: float = 0.8,
) -> pd.Series:
    """Aggregate daily (business-day) data to monthly or quarterly.

    A period is kept only if the number of observations is at least
    ``min_coverage`` times the number of business days in that period. This drops
    the incomplete current quarter and periods with large data gaps, instead of
    reporting an average of a few days as if it were a full-quarter average.
    """
    s = series.dropna()
    periods = s.index.to_period(freq)
    grouped = s.groupby(periods)
    out = grouped.agg(how)
    counts = grouped.size()
    expected = pd.Series(
        [len(pd.bdate_range(p.start_time, p.end_time)) for p in counts.index], index=counts.index
    )
    out = out[counts >= min_coverage * expected]
    out.index = pd.DatetimeIndex(out.index.to_timestamp(), name="date")
    out.name = series.name
    return out


def pct_change(x: pd.Series, periods: int = 1, freq: Freq | None = None) -> pd.Series:
    """Percent change ``100 * (x_t / x_{t-k} - 1)``. Pass ``freq`` to check regularity."""
    if freq is not None:
        require_regular(x.index, freq)
    return 100.0 * (x / x.shift(periods) - 1.0)


def log_diff(x: pd.Series, periods: int = 1, scale: float = 100.0, freq: Freq | None = None) -> pd.Series:
    """Log difference ``scale * (ln x_t - ln x_{t-k})`` (log points when scale=100)."""
    if freq is not None:
        require_regular(x.index, freq)
    if (x <= 0).any():
        raise ValueError("log_diff requires strictly positive values")
    return scale * np.log(x).diff(periods)


def pp_change(rate: pd.Series, periods: int = 1, freq: Freq | None = None) -> pd.Series:
    """Percentage-point change of a series that is already a rate (e.g. OCR, inflation)."""
    if freq is not None:
        require_regular(rate.index, freq)
    return rate - rate.shift(periods)


def annualise(rate_pct: pd.Series | float, periods_per_year: int) -> pd.Series | float:
    """Compound a per-period percent growth rate to an annual rate."""
    return 100.0 * ((1.0 + rate_pct / 100.0) ** periods_per_year - 1.0)


def yoy(x: pd.Series, freq: Freq) -> pd.Series:
    """Year-on-year percent change of an index level."""
    return pct_change(x, _PERIODS_PER_YEAR[freq], freq=freq)


def make_lags(df: pd.DataFrame | pd.Series, lags: int | list[int], prefix_sep: str = "_L") -> pd.DataFrame:
    """Return lagged copies named ``{col}_L{k}``.

    ``lags=2`` gives lags 1 and 2; a list gives exactly those lags (0 = contemporaneous).
    The first rows are NaN: the caller decides whether to drop them, so that every
    model compared on the same data uses the same effective sample.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    ks = list(range(1, lags + 1)) if isinstance(lags, int) else list(lags)
    cols = {
        f"{c}{prefix_sep}{k}" if k else c: df[c].shift(k) for c in df.columns for k in ks
    }
    return pd.DataFrame(cols, index=df.index)


def contiguous_tail(x: pd.Series, freq: Freq) -> pd.Series:
    """Keep the longest gap-free stretch that ends at the last valid observation."""
    s = x.dropna()
    p = s.index.to_period(freq)
    gaps = np.flatnonzero(np.diff(p.asi8) != 1)
    start = gaps[-1] + 1 if len(gaps) else 0
    return s.iloc[start:]
