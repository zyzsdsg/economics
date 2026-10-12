"""Estimators: recover known parameters and match statsmodels."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL, UECM
from statsmodels.tsa.stattools import adfuller

from economic_lab.data.synthetic import SyntheticParams, make_synthetic_quarterly
from economic_lab.data.transforms import log_diff, pct_change
from economic_lab.econometrics.ardl import fit_ardl, fit_uecm, select_ardl_order
from economic_lab.econometrics.ols import condition_number, ols, wald_test
from economic_lab.econometrics.timeseries import acf, adf_test, ljung_box, simulate_ar1


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(123)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    beta = np.array([1.0, 2.0, -0.5])
    y = X @ beta + rng.normal(scale=0.5, size=n)
    return X, y, beta


@pytest.mark.parametrize("method", ["inverse", "normal", "qr", "svd", "lstsq"])
def test_ols_recovers_true_coefficients(regression_data, method):
    X, y, beta = regression_data
    res = ols(y, X, method=method)
    # Within 4 standard errors of the truth (fails with prob. < 1e-4 per coefficient).
    assert np.all(np.abs(res.params.to_numpy() - beta) < 4 * res.bse.to_numpy())


def test_ols_matches_statsmodels(regression_data):
    X, y, _ = regression_data
    ours, ref = ols(y, X), sm.OLS(y, X).fit()
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(ours.bse, ref.bse, rtol=1e-10)
    np.testing.assert_allclose(ours.pvalues, ref.pvalues, rtol=1e-8, atol=1e-300)
    assert ours.aic == pytest.approx(ref.aic) and ours.rsquared == pytest.approx(ref.rsquared)


def test_hac_matches_statsmodels():
    rng = np.random.default_rng(1)
    n = 300
    X = np.column_stack([np.ones(n), simulate_ar1(n, 0.7, seed=2)])
    y = X @ [0.5, 1.0] + simulate_ar1(n, 0.6, seed=3)
    for L in (0, 4, 8):
        ours = ols(y, X, cov_type="HAC", hac_lags=L)
        ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
        np.testing.assert_allclose(ours.bse, ref.bse, rtol=1e-10)
    assert rng is not None


def test_hac_larger_than_nonrobust_with_positive_autocorrelation():
    n = 400
    x = simulate_ar1(n, 0.8, seed=10)
    X = np.column_stack([np.ones(n), x])
    y = X @ [0.0, 1.0] + simulate_ar1(n, 0.8, seed=11)
    assert ols(y, X, cov_type="HAC", hac_lags=8).bse.iloc[1] > ols(y, X).bse.iloc[1]


def test_shape_and_nan_validation():
    with pytest.raises(ValueError):
        ols(np.ones(5), np.ones((4, 2)))
    with pytest.raises(ValueError):
        ols(np.array([1.0, np.nan, 2, 3]), np.ones((4, 1)))


def test_qr_more_accurate_than_inverse_when_ill_conditioned():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    X = np.vander(t, 10, increasing=True)  # polynomial basis: kappa ~ 1e7
    beta = np.ones(10)
    y = X @ beta
    assert condition_number(X) > 1e6
    err_qr = np.abs(ols(y, X, method="qr").params.to_numpy() - beta).max()
    err_inv = np.abs(ols(y, X, method="inverse").params.to_numpy() - beta).max()
    assert err_qr < err_inv
    assert rng is not None


def test_wald_test_matches_statsmodels(regression_data):
    X, y, _ = regression_data
    R = np.array([[0, 1, 0], [0, 0, 1]])
    ref = sm.OLS(y, X).fit().f_test(R)
    assert wald_test(ols(y, X), R)["F"] == pytest.approx(float(ref.fvalue))


def test_acf_and_ljung_box():
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf as sm_acf

    x = simulate_ar1(400, 0.5, seed=4)
    np.testing.assert_allclose(acf(x, 5), sm_acf(x, nlags=5), rtol=1e-10)
    assert ljung_box(x, 6)["Q"] == pytest.approx(float(acorr_ljungbox(x, [6])["lb_stat"].iloc[0]))
    assert acf(x, 1)[1] == pytest.approx(0.5, abs=0.1)


@pytest.mark.parametrize("trend", ["n", "c", "ct"])
def test_adf_matches_statsmodels(trend):
    x = np.cumsum(np.random.default_rng(5).normal(size=250))
    ours = adf_test(x, trend=trend)
    ref = adfuller(x, regression=trend, autolag="AIC", result_object=False)
    assert ours.stat == pytest.approx(ref[0]) and ours.lags == ref[2]
    assert ours.p_value == pytest.approx(ref[1])


def test_adf_power_and_size():
    assert adf_test(simulate_ar1(400, 0.5, seed=6)).p_value < 0.01  # stationary: reject
    assert adf_test(simulate_ar1(400, 1.0, seed=7)).p_value > 0.05  # random walk: do not reject


@pytest.fixture(scope="module")
def synthetic_ardl_data():
    q = make_synthetic_quarterly(SyntheticParams(start="1950-01-01"))
    pi_T = pct_change(q["cpi_tradable"], 1, freq="Q").rename("pi_T")
    doil = log_diff(q["oil_nzd"], 1, freq="Q").rename("doil")
    return pd.concat([pi_T, doil], axis=1).dropna()


def test_ardl_matches_statsmodels(synthetic_ardl_data):
    d = synthetic_ardl_data
    ours = fit_ardl(d["pi_T"], d["doil"], 2, 2)
    ref = ARDL(d["pi_T"], 2, d[["doil"]], 2, trend="c").fit()
    np.testing.assert_allclose(ours.ols.params, ref.params, rtol=1e-8)
    np.testing.assert_allclose(ours.ols.bse, ref.bse, rtol=1e-8)


def test_ardl_recovers_long_run_passthrough(synthetic_ardl_data):
    d = synthetic_ardl_data
    true = SyntheticParams().long_run_tradable_passthrough
    lr = fit_ardl(d["pi_T"], d["doil"], 1, 1).long_run().loc["doil"]
    assert abs(lr["long_run"] - true) < 3 * lr["std_err"]


def test_ardl_step_response_converges_to_long_run(synthetic_ardl_data):
    d = synthetic_ardl_data
    res = fit_ardl(d["pi_T"], d["doil"], 1, 1)
    assert res.cumulative_response("doil", 60).iloc[-1] == pytest.approx(
        res.long_run().loc["doil", "long_run"], rel=1e-6
    )


def test_order_selection_uses_common_sample(synthetic_ardl_data):
    d = synthetic_ardl_data
    table = select_ardl_order(d["pi_T"], d["doil"], max_p=4, max_q=4)
    assert table["nobs"].nunique() == 1
    assert table.iloc[0]["q"] >= 1  # the DGP has oil effects at lags 0 and 1


def test_uecm_is_reparametrisation_of_ardl(synthetic_ardl_data):
    q = make_synthetic_quarterly()
    y, x = np.log(q["cpi"]).rename("lcpi"), np.log(q["oil_nzd"]).rename("loil")
    ardl = fit_ardl(y, x, 2, 2)
    uecm = fit_uecm(y, x, 2, 2)
    np.testing.assert_allclose(ardl.ols.resid.to_numpy(), uecm.ols.resid.to_numpy(), atol=1e-10)
    assert uecm.long_run == pytest.approx(ardl.long_run().loc["loil", "long_run"], rel=1e-6)
    ref = UECM(y, 2, x.to_frame(), 2, trend="c").fit()
    np.testing.assert_allclose(uecm.ols.params.to_numpy(), ref.params.to_numpy(), rtol=1e-8)
    assert uecm.bounds_f_stat() == pytest.approx(ref.bounds_test(case=3).statistic, rel=1e-8)


def test_rank_deficient_design_gives_clear_error():
    X = np.column_stack([np.ones(10), np.arange(10.0), np.zeros(10)])
    with pytest.raises(ValueError, match="rank 2 < 3"):
        ols(np.arange(10.0), X)
