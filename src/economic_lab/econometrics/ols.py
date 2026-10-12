r"""Ordinary least squares from first principles.

Model: y = X beta + u, with y (n,), X (n, k), beta (k,), u (n,).

Objective: minimise S(beta) = (y - X beta)'(y - X beta).
First-order condition: dS/dbeta = -2 X'(y - X beta) = 0  =>  X'X beta = X'y  (normal equations).
If X has full column rank, beta_hat = (X'X)^{-1} X'y.

Numerically we never form (X'X)^{-1}: forming X'X squares the condition number
(kappa(X'X) = kappa(X)^2), so near-collinear regressors lose twice as many digits.
QR (X = QR, then R beta = Q'y) works with kappa(X) directly; SVD additionally
handles rank deficiency by truncating tiny singular values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import linalg, stats

Solver = Literal["inverse", "normal", "qr", "svd", "lstsq"]
CovType = Literal["nonrobust", "HC0", "HAC"]


def solve_inverse(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Textbook formula (X'X)^{-1} X'y. For teaching only: least stable option."""
    return np.linalg.inv(X.T @ X) @ X.T @ y


def solve_normal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the normal equations X'X b = X'y with a Cholesky factorisation."""
    c, low = linalg.cho_factor(X.T @ X)
    return linalg.cho_solve((c, low), X.T @ y)


def solve_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reduced QR: X = Q R with Q (n,k) orthonormal, R (k,k) upper triangular.

    ||y - X b||^2 = ||Q'y - R b||^2 + const, minimised by back-substitution R b = Q'y.
    """
    Q, R = np.linalg.qr(X, mode="reduced")
    return linalg.solve_triangular(R, Q.T @ y, lower=False)


def solve_svd(X: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """SVD: X = U diag(s) V'. b = V diag(1/s) U'y, dropping s_i < rcond * s_max.

    With a rank-deficient X this returns the minimum-norm least-squares solution.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > rcond * s.max()
    return Vt[keep].T @ ((U[:, keep].T @ y) / s[keep])


def solve(X: np.ndarray, y: np.ndarray, method: Solver = "qr") -> np.ndarray:
    if method == "inverse":
        return solve_inverse(X, y)
    if method == "normal":
        return solve_normal(X, y)
    if method == "qr":
        return solve_qr(X, y)
    if method == "svd":
        return solve_svd(X, y)
    if method == "lstsq":
        return np.linalg.lstsq(X, y, rcond=None)[0]
    raise ValueError(f"unknown method {method!r}")


def newey_west_lags(n: int) -> int:
    """Common rule of thumb L = floor(4 (n/100)^(2/9)) (Newey and West, 1994)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def hac_meat(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    r"""Newey-West long-run covariance of the scores g_t = x_t u_t, shape (k, k).

    S = Gamma_0 + sum_{l=1}^{L} w_l (Gamma_l + Gamma_l'),  Gamma_l = sum_t g_t g_{t-l}',
    with Bartlett weights w_l = 1 - l/(L+1), which guarantee S is positive semi-definite.
    """
    g = X * u[:, None]  # (n, k): row t is x_t' u_t
    S = g.T @ g
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = g[lag:].T @ g[:-lag]
        S += w * (gamma + gamma.T)
    return S


@dataclass
class OLSResult:
    """Container mirroring the parts of a statsmodels result we use."""

    params: pd.Series
    cov_params: pd.DataFrame
    resid: pd.Series
    fittedvalues: pd.Series
    nobs: int
    df_resid: int
    ssr: float
    rsquared: float
    cov_type: str
    use_t: bool
    extra: dict = field(default_factory=dict)

    @property
    def bse(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.cov_params.to_numpy())), index=self.params.index)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        t = np.abs(self.tvalues.to_numpy())
        p = 2 * (stats.t.sf(t, self.df_resid) if self.use_t else stats.norm.sf(t))
        return pd.Series(p, index=self.params.index)

    @property
    def sigma2(self) -> float:
        return self.ssr / self.df_resid

    @property
    def llf(self) -> float:
        """Gaussian log-likelihood at the MLE of sigma^2 (= ssr / n)."""
        n = self.nobs
        return -0.5 * n * (np.log(2 * np.pi) + np.log(self.ssr / n) + 1.0)

    @property
    def aic(self) -> float:
        return -2 * self.llf + 2 * len(self.params)

    @property
    def bic(self) -> float:
        return -2 * self.llf + np.log(self.nobs) * len(self.params)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coef": self.params, "std_err": self.bse, "t": self.tvalues, "p_value": self.pvalues}
        )


def ols(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    method: Solver = "qr",
    cov_type: CovType = "nonrobust",
    hac_lags: int | None = None,
    hac_small_sample: bool = False,
) -> OLSResult:
    """Estimate y = X beta + u by least squares.

    Parameters
    ----------
    y : (n,) array or Series
    X : (n, k) array or DataFrame. Include a constant column yourself if wanted.
    method : solver used for beta (all give the same answer when X is well conditioned).
    cov_type :
        ``"nonrobust"``: sigma^2 (X'X)^{-1}, sigma^2 = SSR/(n-k). Valid under homoskedastic,
        serially uncorrelated errors. Inference uses the t distribution.
        ``"HC0"``: White heteroskedasticity-robust (X'X)^{-1} (sum u_t^2 x_t x_t') (X'X)^{-1}.
        ``"HAC"``: Newey-West heteroskedasticity- and autocorrelation-consistent sandwich.
        Robust variants use normal critical values (asymptotic justification), matching
        statsmodels' defaults.
    hac_lags : truncation lag L for HAC; default ``newey_west_lags(n)``.
    hac_small_sample : multiply the HAC covariance by n/(n-k). Off by default, matching
        statsmodels' ``cov_type="HAC"`` default.
    """
    names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(np.shape(X)[1])]
    index = X.index if isinstance(X, pd.DataFrame) else (y.index if isinstance(y, pd.Series) else None)
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    if Xa.ndim != 2 or ya.ndim != 1 or Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"shape mismatch: X {Xa.shape}, y {ya.shape}")
    if not (np.isfinite(Xa).all() and np.isfinite(ya).all()):
        raise ValueError("X and y must not contain NaN/inf; drop missing rows first")
    n, k = Xa.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than regressors ({k})")

    rank = np.linalg.matrix_rank(Xa)
    if rank < k:
        constant = [nm for nm, col in zip(names, Xa.T, strict=True) if np.ptp(col) == 0]
        raise ValueError(
            f"X has rank {rank} < {k} columns (perfect collinearity). "
            f"Constant columns: {constant}. Drop redundant regressors, e.g. a dummy "
            "that is zero throughout the sample."
        )

    beta = solve(Xa, ya, method)
    fitted = Xa @ beta
    u = ya - fitted
    ssr = float(u @ u)
    tss = float(((ya - ya.mean()) ** 2).sum())

    # (X'X)^{-1} via QR: X'X = R'R, so (X'X)^{-1} = R^{-1} R^{-T}.
    _, R = np.linalg.qr(Xa, mode="reduced")
    Rinv = linalg.solve_triangular(R, np.eye(k), lower=False)
    bread = Rinv @ Rinv.T

    extra: dict = {}
    if cov_type == "nonrobust":
        cov = (ssr / (n - k)) * bread
        use_t = True
    elif cov_type in ("HC0", "HAC"):
        L = 0 if cov_type == "HC0" else (newey_west_lags(n) if hac_lags is None else hac_lags)
        cov = bread @ hac_meat(Xa, u, L) @ bread
        if cov_type == "HAC" and hac_small_sample:
            cov *= n / (n - k)
        extra["hac_lags"] = L
        use_t = False
    else:
        raise ValueError(f"unknown cov_type {cov_type!r}")

    return OLSResult(
        params=pd.Series(beta, index=names),
        cov_params=pd.DataFrame(cov, index=names, columns=names),
        resid=pd.Series(u, index=index),
        fittedvalues=pd.Series(fitted, index=index),
        nobs=n,
        df_resid=n - k,
        ssr=ssr,
        rsquared=1.0 - ssr / tss if tss > 0 else np.nan,
        cov_type=cov_type,
        use_t=use_t,
        extra=extra,
    )


def wald_test(result: OLSResult, R: np.ndarray, r: np.ndarray | None = None) -> dict[str, float]:
    """Wald test of H0: R beta = r using the result's covariance matrix.

    W = (R b - r)' [R V R']^{-1} (R b - r) ~ chi2(q) asymptotically; F = W/q.
    """
    R = np.atleast_2d(R)
    r = np.zeros(R.shape[0]) if r is None else np.asarray(r, dtype=float)
    diff = R @ result.params.to_numpy() - r
    V = R @ result.cov_params.to_numpy() @ R.T
    W = float(diff @ np.linalg.solve(V, diff))
    q = R.shape[0]
    return {
        "wald": W,
        "F": W / q,
        "df": q,
        "p_chi2": float(stats.chi2.sf(W, q)),
        "p_F": float(stats.f.sf(W / q, q, result.df_resid)),
    }


def delta_method(
    result: OLSResult, func, eps: float = 1e-6
) -> tuple[float, float]:
    """Value and standard error of a scalar g(beta) via the delta method.

    Var(g(b)) ~= grad' V grad, with the gradient computed by central differences.
    """
    b = result.params.to_numpy()
    value = float(func(b))
    grad = np.empty_like(b)
    for i in range(len(b)):
        step = np.zeros_like(b)
        step[i] = eps * max(1.0, abs(b[i]))
        grad[i] = (func(b + step) - func(b - step)) / (2 * step[i])
    se = float(np.sqrt(grad @ result.cov_params.to_numpy() @ grad))
    return value, se


def condition_number(X: np.ndarray) -> float:
    """kappa(X) = s_max / s_min. Rule of thumb: > 1e3 signals collinearity worth checking."""
    s = np.linalg.svd(np.asarray(X, dtype=float), compute_uv=False)
    return float(s[0] / s[-1])
