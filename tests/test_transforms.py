import numpy as np
import pandas as pd
import pytest

from economic_lab.data import transforms as tf


def quarterly(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="QS"), dtype=float)


def test_currency_conversion_divides_by_usd_per_nzd():
    idx = pd.bdate_range("2024-01-01", periods=3)
    oil = pd.Series([80.0, 90.0, 60.0], index=idx, name="brent_usd")
    fx = pd.Series([0.60, 0.50, 0.60], index=idx)
    out = tf.convert_currency(oil, fx)
    np.testing.assert_allclose(out, [80 / 0.6, 180.0, 100.0])
    assert out.name == "brent_usd_nzd"


def test_currency_conversion_inner_join_and_positive_rate():
    oil = pd.Series([80.0, 90.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    fx = pd.Series([0.6], index=pd.to_datetime(["2024-01-02"]))
    assert len(tf.convert_currency(oil, fx)) == 1
    with pytest.raises(ValueError):
        tf.convert_currency(oil, pd.Series([0.0], index=oil.index[:1]))


def test_qoq_yoy_and_annualised_inflation():
    cpi = quarterly([100.0, 101.0, 102.01, 103.0301, 104.060401])  # 1% per quarter
    qoq = tf.pct_change(cpi, 1, freq="Q")
    np.testing.assert_allclose(qoq.dropna(), 1.0)
    np.testing.assert_allclose(tf.yoy(cpi, "Q").dropna(), 100 * (1.01**4 - 1))
    assert tf.annualise(1.0, 4) == pytest.approx(4.0604, abs=1e-4)


def test_log_diff_approximates_but_differs_from_pct_change():
    x = quarterly([100.0, 110.0])
    assert tf.pct_change(x).iloc[-1] == pytest.approx(10.0)
    assert tf.log_diff(x).iloc[-1] == pytest.approx(100 * np.log(1.1))  # 9.531 log points


def test_pp_change_is_difference_not_percent():
    rate = quarterly([2.0, 3.0])
    assert tf.pp_change(rate).iloc[-1] == pytest.approx(1.0)  # +1 pp, i.e. +50 percent


def test_shift_on_gappy_index_is_refused():
    idx = pd.to_datetime(["2020-01-01", "2020-04-01", "2020-10-01"])  # Q3 missing
    with pytest.raises(ValueError, match="missing"):
        tf.pct_change(pd.Series([1.0, 2.0, 3.0], index=idx), freq="Q")


def test_make_lags_names_and_values():
    s = quarterly([1, 2, 3]).rename("x")
    lags = tf.make_lags(s, [0, 1, 2])
    assert list(lags.columns) == ["x", "x_L1", "x_L2"]
    assert lags["x_L2"].iloc[2] == 1
    assert lags["x_L1"].isna().sum() == 1


def test_aggregate_drops_incomplete_period():
    idx = pd.bdate_range("2024-01-01", "2024-04-05")  # Q1 complete, Q2 only 5 days
    s = pd.Series(1.0, index=idx, name="x")
    out = tf.aggregate(s, "Q", "mean")
    assert list(out.index) == [pd.Timestamp("2024-01-01")]


def test_aggregate_mean_vs_last():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    assert tf.aggregate(s, "Q", "last").iloc[0] == len(idx) - 1
    assert tf.aggregate(s, "Q", "mean").iloc[0] == pytest.approx((len(idx) - 1) / 2)


def test_contiguous_tail_and_period_start():
    idx = pd.to_datetime(["1920-01-01", "1921-01-01", "1990-03-31", "1990-06-30", "1990-09-30"])
    s = pd.Series([1, 2, 3, 4, 5.0], index=idx)
    s.index = tf.to_period_start(s.index, "Q")
    tail = tf.contiguous_tail(s, "Q")
    assert list(tail.values) == [3, 4, 5]
    assert tail.index[0] == pd.Timestamp("1990-01-01")
