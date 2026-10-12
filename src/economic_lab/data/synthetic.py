"""Synthetic NZ-style macro data with a *known* data-generating process (DGP).

Used for offline notebooks, CI and estimator tests. Nothing produced here is an
empirical finding about New Zealand; every DataFrame carries
``attrs["data_kind"] = "synthetic"``.

DGP (quarterly, all rates in percent per quarter)
-------------------------------------------------
Daily layer (business days):
    ln oil_usd_d   = ln oil_usd_{d-1} + e_oil_d                 random walk
    ln nzd_usd_d   = mu_fx + phi_fx (ln nzd_usd_{d-1} - mu_fx) + e_fx_d
    corr(e_oil, e_fx) = rho_oil_fx  (NZD is a commodity currency)
    oil_nzd_d      = oil_usd_d / nzd_usd_d

Quarterly layer (q = quarterly average of the daily series):
    doil_t   = 100 * Δ ln oil_nzd_t
    gap_t    = a_gap * gap_{t-1} + u_gap_t
    pi_T_t   = c_T + rho_T pi_T_{t-1} + b0 doil_t + b1 doil_{t-1} + u_T_t     (tradable)
    pi_N_t   = c_N + rho_N pi_N_{t-1} + g_N gap_{t-1} + u_N_t                 (non-tradable)
    pi_t     = w_T pi_T_t + (1 - w_T) pi_N_t                                    (headline)
    cpi_t    = cpi_{t-1} * (1 + pi_t / 100)          and similarly for the sub-indices
    ocr_t    = r_s ocr_{t-1} + (1-r_s) [r_star + 1.5 (4 pi_{t-1} - 2) + 0.5 gap_{t-1}] + u_r

Implied long-run effect of a permanent 1% rise in the NZD oil price on tradable
prices: (b0 + b1) / (1 - rho_T).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from economic_lab.data.transforms import aggregate, convert_currency


@dataclass(frozen=True)
class SyntheticParams:
    start: str = "1995-01-01"
    end: str = "2024-12-31"
    oil_usd_0: float = 20.0
    sigma_oil_d: float = 0.02
    nzd_usd_0: float = 0.60
    phi_fx: float = 0.999
    sigma_fx_d: float = 0.006
    rho_oil_fx: float = 0.3
    a_gap: float = 0.8
    sigma_gap: float = 0.6
    c_T: float = 0.15
    rho_T: float = 0.3
    b0: float = 0.04
    b1: float = 0.02
    sigma_T: float = 0.5
    c_N: float = 0.45
    rho_N: float = 0.5
    g_N: float = 0.15
    sigma_N: float = 0.25
    w_T: float = 0.43
    r_s: float = 0.8
    r_star: float = 4.0
    sigma_r: float = 0.25

    @property
    def long_run_tradable_passthrough(self) -> float:
        return (self.b0 + self.b1) / (1.0 - self.rho_T)


def make_synthetic_daily(params: SyntheticParams | None = None, seed: int = 42) -> pd.DataFrame:
    """Business-day oil price (USD), NZD/USD rate and NZD oil price."""
    params = params or SyntheticParams()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(params.start, params.end, name="date")
    n = len(dates)
    cov = np.array(
        [
            [params.sigma_oil_d**2, params.rho_oil_fx * params.sigma_oil_d * params.sigma_fx_d],
            [params.rho_oil_fx * params.sigma_oil_d * params.sigma_fx_d, params.sigma_fx_d**2],
        ]
    )
    shocks = rng.multivariate_normal(np.zeros(2), cov, size=n)
    log_oil = np.log(params.oil_usd_0) + np.cumsum(shocks[:, 0])
    mu = np.log(params.nzd_usd_0)
    log_fx = np.empty(n)
    log_fx[0] = mu
    for d in range(1, n):
        log_fx[d] = mu + params.phi_fx * (log_fx[d - 1] - mu) + shocks[d, 1]
    df = pd.DataFrame({"brent_usd": np.exp(log_oil), "nzd_usd": np.exp(log_fx)}, index=dates)
    df["oil_nzd"] = convert_currency(df["brent_usd"], df["nzd_usd"])
    df.attrs["data_kind"] = "synthetic"
    return df


def make_synthetic_quarterly(
    params: SyntheticParams | None = None, seed: int = 42
) -> pd.DataFrame:
    """Quarterly dataset following the module-level DGP.

    Columns: brent_usd, nzd_usd, oil_nzd (quarterly means), cpi, cpi_tradable,
    cpi_nontradable (index levels, first quarter = 1000), output_gap (% of potential),
    ocr (% p.a.). Inflation rates are *not* stored; compute them with
    :mod:`economic_lab.data.transforms` exactly as for real data.
    """
    params = params or SyntheticParams()
    daily = make_synthetic_daily(params, seed)
    q = pd.DataFrame({c: aggregate(daily[c], "Q", "mean") for c in daily.columns})
    rng = np.random.default_rng(seed + 1)
    p = params
    T = len(q)
    doil = 100 * np.log(q["oil_nzd"]).diff().fillna(0.0).to_numpy()

    gap, pi_T, pi_N, pi, ocr = (np.zeros(T) for _ in range(5))
    pi_T[0] = p.c_T / (1 - p.rho_T)
    pi_N[0] = p.c_N / (1 - p.rho_N)
    pi[0] = p.w_T * pi_T[0] + (1 - p.w_T) * pi_N[0]
    ocr[0] = p.r_star
    for t in range(1, T):
        gap[t] = p.a_gap * gap[t - 1] + p.sigma_gap * rng.standard_normal()
        pi_T[t] = (
            p.c_T + p.rho_T * pi_T[t - 1] + p.b0 * doil[t] + p.b1 * doil[t - 1]
            + p.sigma_T * rng.standard_normal()
        )
        pi_N[t] = p.c_N + p.rho_N * pi_N[t - 1] + p.g_N * gap[t - 1] + p.sigma_N * rng.standard_normal()
        pi[t] = p.w_T * pi_T[t] + (1 - p.w_T) * pi_N[t]
        target = p.r_star + 1.5 * (4 * pi[t - 1] - 2.0) + 0.5 * gap[t - 1]
        ocr[t] = max(0.25, p.r_s * ocr[t - 1] + (1 - p.r_s) * target + p.sigma_r * rng.standard_normal())

    def level(rates: np.ndarray) -> np.ndarray:
        growth = 1.0 + rates / 100.0
        growth[0] = 1.0
        return 1000.0 * np.cumprod(growth)

    q["cpi"] = level(pi)
    q["cpi_tradable"] = level(pi_T)
    q["cpi_nontradable"] = level(pi_N)
    q["output_gap"] = gap
    q["ocr"] = ocr
    q.attrs["data_kind"] = "synthetic"
    q.attrs["true_params"] = asdict(p)
    return q
