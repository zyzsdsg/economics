r"""Time-series foundations: autocorrelation, white-noise tests, unit-root tests.

ADF regression (regression="c"):
    Δy_t = a + gamma * y_{t-1} + sum_{i=1}^{p} phi_i Δy_{t-i} + e_t
H0: gamma = 0 (unit root) vs H1: gamma < 0 (stationary).
Under H0 the t-ratio of gamma does NOT follow a t distribution: y_{t-1} is a random
walk, so its sample moments converge to functionals of Brownian motion. Critical
values come from simulation (MacKinnon 1994, 2010), which we take from statsmodels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from economic_lab.econometrics.ols import ols

Trend = Literal["n", "c", "ct"]


def simulate_ar1(n: int, phi: float, sigma: float = 1.0, c: float = 0.0, seed: int | None = None) -> np.ndarray:
    """y_t = c + phi y_{t-1} + sigma e_t, started at the stationary mean (or 0 if phi = 1)."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = c / (1 - phi) if abs(phi) < 1 else 0.0
    e = sigma * rng.standard_normal(n)
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + e[t]
    return y


def acf(x: np.ndarray | pd.Series, nlags: int) -> np.ndarray:
    r"""Sample autocorrelations rho_0..rho_nlags.

    rho_k = gamma_k / gamma_0, gamma_k = (1/n) sum_{t=k+1}^{n} (x_t - xbar)(x_{t-k} - xbar).
    Dividing by n (not n-k) keeps the implied autocovariance matrix positive definite.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)] - np.nanmean(x)
    n = len(x)
    gamma0 = x @ x / n
    return np.array([1.0] + [(x[k:] @ x[:-k]) / n / gamma0 for k in range(1, nlags + 1)])


def ljung_box(resid: np.ndarray | pd.Series, lags: int, model_df: int = 0) -> dict[str, float]:
    r"""Ljung-Box Q = n(n+2) sum_{k=1}^{h} rho_k^2 / (n-k) ~ chi2(h - model_df) under white noise."""
    r = acf(resid, lags)[1:]
    n = int(np.sum(~np.isnan(np.asarray(resid, dtype=float))))
    q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1)))
    df = lags - model_df
    return {"Q": float(q), "df": df, "p_value": float(stats.chi2.sf(q, df))}


def _adf_design(y: np.ndarray, lags: int, trend: Trend, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows t = start..n-1 of the ADF regression (start >= lags + 1)."""
    dy = np.diff(y)  # dy[t-1] = y_t - y_{t-1}
    t_idx = np.arange(start, len(y))
    cols = [y[t_idx - 1]]
    cols += [dy[t_idx - 1 - i] for i in range(1, lags + 1)]
    if trend in ("c", "ct"):
        cols.append(np.ones(len(t_idx)))
    if trend == "ct":
        cols.append(t_idx.astype(float))
    return dy[t_idx - 1], np.column_stack(cols)


@dataclass
class ADFResult:
    stat: float
    p_value: float
    lags: int
    nobs: int
    critical_values: dict[str, float]
    trend: str

    def __str__(self) -> str:
        cv = ", ".join(f"{k}: {v:.2f}" for k, v in self.critical_values.items())
        return (f"ADF({self.trend}, lags={self.lags}) stat={self.stat:.3f}, p={self.p_value:.3f}, "
                f"n={self.nobs}; critical values {cv}")


def adf_test(
    y: np.ndarray | pd.Series,
    lags: int | None = None,
    trend: Trend = "c",
    maxlag: int | None = None,
    autolag: Literal["aic", "bic"] = "aic",
) -> ADFResult:
    """Augmented Dickey-Fuller test.

    If ``lags`` is None the lag length is chosen by AIC/BIC over 0..maxlag, with all
    candidate regressions estimated on the *same* sample (the one available for
    maxlag), so that information criteria are comparable. The chosen model is then
    re-estimated on the largest sample available for its lag length. This mirrors
    ``statsmodels.tsa.stattools.adfuller``.
    """
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    n = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
        maxlag = min(n // 2 - (3 if trend == "ct" else 2 if trend == "c" else 1) - 1, maxlag)
    if lags is None:
        best = None
        for p in range(maxlag + 1):
            dep, X = _adf_design(y, p, trend, start=maxlag + 1)
            res = ols(dep, X)
            ic = res.aic if autolag == "aic" else res.bic
            if best is None or ic < best[0]:
                best = (ic, p)
        lags = best[1]
    dep, X = _adf_design(y, lags, trend, start=lags + 1)
    res = ols(dep, X)
    stat = float(res.tvalues.iloc[0])
    crit = mackinnoncrit(N=1, regression=trend, nobs=len(dep))
    return ADFResult(
        stat=stat,
        p_value=float(mackinnonp(stat, regression=trend, N=1)),
        lags=lags,
        nobs=len(dep),
        critical_values={"1%": crit[0], "5%": crit[1], "10%": crit[2]},
        trend=trend,
    )
