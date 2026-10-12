r"""Autoregressive distributed lag (ARDL) models and their error-correction form.

ARDL(p, q) with one regressor:
    y_t = c + sum_{i=1}^{p} phi_i y_{t-i} + sum_{j=0}^{q} beta_j x_{t-j} + u_t

Long-run multiplier (effect of a permanent unit change in x on y once dynamics settle):
    theta = sum_j beta_j / (1 - sum_i phi_i),    requires sum_i phi_i < 1.

Unrestricted error-correction model (UECM), an exact reparametrisation of the same
regression (identical residuals, see :func:`fit_uecm`):
    Δy_t = c + pi_y y_{t-1} + pi_x x_{t-1}
               + sum_{i=1}^{p-1} psi_i Δy_{t-i} + sum_{j=0}^{q-1} omega_j Δx_{t-j} + u_t
with pi_y = -(1 - sum phi_i)   (speed of adjustment, alpha)
     pi_x = sum beta_j          so theta = -pi_x / pi_y.

When to use which:
- y and x both stationary (e.g. q/q inflation and q/q oil-price change): estimate the
  ARDL in rates; theta is the cumulative pass-through. No cointegration needed.
- y and x both I(1) (e.g. log price levels): an ECM is meaningful only if they are
  cointegrated; the Pesaran-Shin-Smith bounds test checks H0: pi_y = pi_x = 0 and
  has non-standard critical values.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from economic_lab.econometrics.ols import CovType, OLSResult, delta_method, ols, wald_test


def ardl_design(
    y: pd.Series,
    x: pd.DataFrame | pd.Series,
    p: int,
    q: int | dict[str, int],
    trend: bool = True,
    start: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build the ARDL regression. Columns: const, {y}_L1..p, {x}_L0..q.

    ``start`` drops the first ``start`` rows so models with different (p, q) can be
    compared on a common sample; it defaults to max(p, q).
    """
    if isinstance(x, pd.Series):
        x = x.to_frame()
    qs = q if isinstance(q, dict) else {c: q for c in x.columns}
    cols: dict[str, pd.Series] = {}
    if trend:
        cols["const"] = pd.Series(1.0, index=y.index)
    for i in range(1, p + 1):
        cols[f"{y.name}_L{i}"] = y.shift(i)
    for c in x.columns:
        for j in range(qs[c] + 1):
            cols[f"{c}_L{j}"] = x[c].shift(j)
    X = pd.DataFrame(cols, index=y.index)
    start = max([p, *qs.values()]) if start is None else start
    X, yy = X.iloc[start:], y.iloc[start:]
    keep = X.notna().all(axis=1) & yy.notna()
    return yy[keep], X[keep]


@dataclass
class ARDLResult:
    ols: OLSResult
    p: int
    q: dict[str, int]
    y_name: str

    @property
    def ar_sum(self) -> float:
        return float(sum(self.ols.params[f"{self.y_name}_L{i}"] for i in range(1, self.p + 1)))

    def long_run(self) -> pd.DataFrame:
        """Long-run multipliers theta_x with delta-method standard errors."""
        names = list(self.ols.params.index)
        ar_idx = [names.index(f"{self.y_name}_L{i}") for i in range(1, self.p + 1)]
        rows = {}
        for c, qc in self.q.items():
            x_idx = [names.index(f"{c}_L{j}") for j in range(qc + 1)]

            def theta(b, x_idx=x_idx):
                return b[x_idx].sum() / (1.0 - b[ar_idx].sum())

            value, se = delta_method(self.ols, theta)
            rows[c] = {"long_run": value, "std_err": se, "z": value / se}
        return pd.DataFrame(rows).T

    def cumulative_response(
        self, x_name: str, horizons: int = 12, params: pd.Series | None = None
    ) -> pd.Series:
        """Dynamic multipliers: response of y at h = 0..H to a permanent unit step in x.

        Equivalently, the partial sums of the impulse multipliers psi_0..psi_h. When y and
        x are growth rates, this is the cumulative response of the *level* of y to a
        one-off permanent rise in the *level* of x. Computed by iterating the ARDL with
        all shocks zero; it converges to theta.

        ``params`` overrides the estimated coefficients (same index), which is how
        parameter-simulation confidence bands are produced.
        """
        b = self.ols.params if params is None else params
        phi = [b[f"{self.y_name}_L{i}"] for i in range(1, self.p + 1)]
        beta = [b[f"{x_name}_L{j}"] for j in range(self.q[x_name] + 1)]
        resp = np.zeros(horizons + 1)
        for h in range(horizons + 1):
            ar = sum(phi[i] * resp[h - 1 - i] for i in range(self.p) if h - 1 - i >= 0)
            dl = sum(beta[j] for j in range(len(beta)) if h - j >= 0)  # step: x = 1 for t >= 0
            resp[h] = ar + dl
        return pd.Series(resp, index=pd.RangeIndex(horizons + 1, name="h"), name=f"step_{x_name}")


def fit_ardl(
    y: pd.Series,
    x: pd.DataFrame | pd.Series,
    p: int,
    q: int | dict[str, int],
    *,
    trend: bool = True,
    cov_type: CovType = "nonrobust",
    hac_lags: int | None = None,
    start: int | None = None,
) -> ARDLResult:
    """Estimate an ARDL(p, q) by OLS."""
    if isinstance(x, pd.Series):
        x = x.to_frame()
    qs = q if isinstance(q, dict) else {c: q for c in x.columns}
    yy, X = ardl_design(y, x, p, qs, trend=trend, start=start)
    res = ols(yy, X, cov_type=cov_type, hac_lags=hac_lags)
    return ARDLResult(ols=res, p=p, q=qs, y_name=str(y.name))


def select_ardl_order(
    y: pd.Series,
    x: pd.DataFrame | pd.Series,
    max_p: int,
    max_q: int,
    ic: str = "bic",
    min_p: int = 1,
) -> pd.DataFrame:
    """Grid-search (p, q) on a common estimation sample; returns a table sorted by ``ic``.

    Using the same sample for every candidate is essential: otherwise models with
    fewer lags get extra observations and their likelihoods are not comparable.
    """
    if isinstance(x, pd.Series):
        x = x.to_frame()
    start = max(max_p, max_q)
    rows = []
    for p, q in product(range(min_p, max_p + 1), range(max_q + 1)):
        res = fit_ardl(y, x, p, q, start=start).ols
        rows.append({"p": p, "q": q, "aic": res.aic, "bic": res.bic, "nobs": res.nobs})
    return pd.DataFrame(rows).sort_values(ic).reset_index(drop=True)


def uecm_design(y: pd.Series, x: pd.Series, p: int, q: int) -> tuple[pd.Series, pd.DataFrame]:
    """Regression for the unrestricted ECM of an ARDL(p, q) with one regressor (p, q >= 1)."""
    if p < 1 or q < 1:
        raise ValueError("UECM requires p >= 1 and q >= 1")
    dy, dx = y.diff(), x.diff()
    cols = {"const": pd.Series(1.0, index=y.index), f"{y.name}_L1": y.shift(1), f"{x.name}_L1": x.shift(1)}
    for i in range(1, p):
        cols[f"d_{y.name}_L{i}"] = dy.shift(i)
    for j in range(q):
        cols[f"d_{x.name}_L{j}"] = dx.shift(j)
    X = pd.DataFrame(cols)
    start = max(p, q)
    X, dep = X.iloc[start:], dy.iloc[start:].rename(f"d_{y.name}")
    keep = X.notna().all(axis=1) & dep.notna()
    return dep[keep], X[keep]


@dataclass
class UECMResult:
    ols: OLSResult
    y_name: str
    x_name: str

    @property
    def speed_of_adjustment(self) -> float:
        """alpha = pi_y: share of last period's disequilibrium closed each period (negative)."""
        return float(self.ols.params[f"{self.y_name}_L1"])

    @property
    def long_run(self) -> float:
        return float(-self.ols.params[f"{self.x_name}_L1"] / self.ols.params[f"{self.y_name}_L1"])

    def bounds_f_stat(self) -> float:
        """F statistic of H0: pi_y = pi_x = 0 (no level relationship). Compare with PSS bounds."""
        names = list(self.ols.params.index)
        R = np.zeros((2, len(names)))
        R[0, names.index(f"{self.y_name}_L1")] = 1
        R[1, names.index(f"{self.x_name}_L1")] = 1
        return wald_test(self.ols, R)["F"]


def fit_uecm(y: pd.Series, x: pd.Series, p: int, q: int) -> UECMResult:
    dep, X = uecm_design(y, x, p, q)
    return UECMResult(ols(dep, X), str(y.name), str(x.name))
