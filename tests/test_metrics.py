import numpy as np
import pytest

from knownpred import metrics as M
from knownpred.distributions import SHASH, Gaussian


def test_crps_gaussian_matches_large_ensemble_and_quantile_integral():
    rng = np.random.default_rng(0)
    mu, sigma = np.array([0.0, 1.0, -2.0]), np.array([1.0, 0.5, 3.0])
    y = np.array([0.3, 2.0, -1.0])
    exact = M.crps_gaussian(y, mu, sigma)
    ens = mu[:, None] + sigma[:, None] * rng.standard_normal((3, 200_000))
    np.testing.assert_allclose(M.crps_ensemble(y, ens), exact, rtol=5e-3)
    d = Gaussian(mu, sigma)
    np.testing.assert_allclose(M.crps_from_ppf(y, d.ppf, n_levels=4000), exact, rtol=2e-3)


def test_crps_ensemble_sorted_identity_equals_bruteforce():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 37))
    y = rng.normal(size=5)
    brute = np.abs(x - y[:, None]).mean(1) - 0.5 * np.abs(x[:, :, None] - x[:, None, :]).mean(
        (1, 2)
    )
    np.testing.assert_allclose(M.crps_ensemble(y, x), brute, atol=1e-12)
    brute_fair = np.abs(x - y[:, None]).mean(1) - 0.5 * np.abs(x[:, :, None] - x[:, None, :]).sum(
        (1, 2)
    ) / (37 * 36)
    np.testing.assert_allclose(M.crps_ensemble(y, x, fair=True), brute_fair, atol=1e-12)


def test_crps_shash_quantile_integral_matches_ensemble():
    rng = np.random.default_rng(3)
    d = SHASH(
        np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([0.7, -0.5]), np.array([0.8, 1.2])
    )
    y = np.array([0.5, -1.0])
    ens = d.sample(200_000, rng)
    np.testing.assert_allclose(M.crps_from_ppf(y, d.ppf, 2000), M.crps_ensemble(y, ens), rtol=5e-3)


def test_pit_of_calibrated_forecast_is_uniform():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=20_000)
    sigma = np.exp(rng.normal(scale=0.3, size=20_000))
    y = mu + sigma * rng.standard_normal(20_000)
    pit = M.pit_values(y, Gaussian(mu, sigma).cdf)
    assert M.pit_d(pit) < 3 * M.pit_d_expected(len(pit))
    # overconfident forecast is detected
    pit_bad = M.pit_values(y, Gaussian(mu, 0.5 * sigma).cdf)
    assert M.pit_d(pit_bad) > 10 * M.pit_d_expected(len(pit))


def test_spread_skill_reliable_forecast_ratio_near_one():
    rng = np.random.default_rng(5)
    sigma = np.exp(rng.normal(scale=0.5, size=50_000))
    y = sigma * rng.standard_normal(50_000)
    ss = M.spread_skill(y, np.zeros_like(y), sigma, n_bins=5)
    np.testing.assert_allclose(ss["ratio"], 1.0, atol=0.05)
    assert M.spread_error_ratio(y, 0.0, sigma) == pytest.approx(1.0, abs=0.03)


def test_discard_curve_monotone_when_sigma_informative():
    rng = np.random.default_rng(6)
    sigma = np.exp(rng.normal(size=10_000))
    err = np.abs(sigma * rng.standard_normal(10_000))
    f, v = M.discard_curve(err, sigma)
    assert np.all(np.diff(v) > 0)
    assert v[-1] == pytest.approx(err.mean())


def test_confident_set_overlap_limits():
    s = np.arange(100.0)
    assert M.confident_set_overlap(s, s, 0.2) == 1.0
    assert M.confident_set_overlap(-s, s, 0.2) == 0.0


def test_circular_block_bootstrap_indices_cover_uniformly():
    rng = np.random.default_rng(7)
    idx = M.circular_block_bootstrap_indices(n=50, block_length=7, n_boot=4000, rng=rng)
    assert idx.shape == (4000, 50)
    counts = np.bincount(idx.ravel(), minlength=50) / idx.size
    np.testing.assert_allclose(counts, 1 / 50, rtol=0.05)


def test_block_bootstrap_ci_wider_for_autocorrelated_series():
    rng = np.random.default_rng(8)
    n = 600
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    iid = M.block_bootstrap_ci(lambda i: x[i].mean(), n, block_length=1, n_boot=1000)
    blk = M.block_bootstrap_ci(lambda i: x[i].mean(), n, block_length=30, n_boot=1000)
    assert blk["boot_std"] > 2 * iid["boot_std"]
    assert blk["lower"] < blk["estimate"] < blk["upper"]
