"""Checks of the hypothesis tests: correctness, false-positive rate, power."""

import numpy as np
import pytest

from knownpred import stats_tests as T


def _ar1(n, phi, rng):
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_diebold_mariano_iid_matches_paired_t():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200) + 0.1
    dm = T.diebold_mariano(a, b, lag=0)
    from scipy import stats

    t = stats.ttest_rel(a, b)
    # with lag 0 and h = 1 the HLN-corrected statistic equals the paired t exactly
    assert dm["statistic"] == pytest.approx(t.statistic, rel=1e-10)
    assert dm["p_two_sided"] == pytest.approx(t.pvalue, rel=1e-8)


def test_diebold_mariano_size_under_autocorrelation():
    """Rejection rate under H0 stays near 5 % with strongly autocorrelated
    loss differentials (a naive t-test would reject far more often)."""
    rng = np.random.default_rng(1)
    rej_dm = rej_naive = 0
    reps = 600
    for _ in range(reps):
        d = _ar1(300, 0.6, rng)
        if T.diebold_mariano(d, np.zeros_like(d))["p_two_sided"] < 0.05:
            rej_dm += 1
        if T.diebold_mariano(d, np.zeros_like(d), lag=0)["p_two_sided"] < 0.05:
            rej_naive += 1
    # Measured false-positive rates of the HAC test (600 reps): 5-6 % for iid
    # differentials, 7.5-10 % at phi = 0.6, up to ~21 % at phi = 0.9 with n = 130
    # (documented in docs/methods.md). Here: phi = 0.6, n = 300.
    assert rej_dm / reps < 0.12
    assert rej_naive / reps > 0.2


def test_diebold_mariano_detects_real_difference_and_respects_groups():
    rng = np.random.default_rng(2)
    d = rng.normal(loc=-0.3, size=(5, 100))
    groups = np.repeat(np.arange(5), 100)
    res = T.diebold_mariano(d.ravel(), np.zeros(500), groups=groups)
    assert res["p_a_better"] < 1e-6
    assert res["mean_diff"] < 0


def test_circular_shift_test_size_and_power():
    rng = np.random.default_rng(3)
    stat = T.confident_half_error_change
    # H0: spread unrelated to errors, both autocorrelated
    p_null = []
    for _ in range(200):
        s = np.exp(0.3 * _ar1(130, 0.7, rng))
        e = np.abs(_ar1(130, 0.7, rng))
        p_null.append(T.circular_shift_test(stat, s, e, min_shift=5)["p_value"])
    assert np.mean(np.array(p_null) < 0.05) < 0.10
    # H1: errors scale with spread
    s = np.exp(0.5 * _ar1(130, 0.5, rng))
    e = np.abs(s * rng.standard_normal(130))
    assert T.circular_shift_test(stat, s, e, min_shift=5)["p_value"] < 0.05


def test_circular_shift_test_groups_require_equal_length():
    with pytest.raises(ValueError):
        T.circular_shift_test(
            T.confident_half_error_change, np.ones(5), np.ones(5), groups=[0, 0, 1, 1, 1]
        )


def test_holm_and_bh_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(T.holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(T.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert np.all(T.holm(p) >= T.benjamini_hochberg(p))
