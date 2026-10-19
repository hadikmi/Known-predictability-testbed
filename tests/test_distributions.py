import numpy as np
import pytest
from scipy import integrate, stats

from knownpred.distributions import SHASH, Gaussian


def test_shash_reduces_to_normal():
    d = SHASH(loc=1.5, scale=2.0, eps=0.0, delta=1.0)
    y = np.linspace(-6, 9, 41)
    np.testing.assert_allclose(d.logpdf(y), stats.norm.logpdf(y, 1.5, 2.0), atol=1e-12)
    np.testing.assert_allclose(d.cdf(y), stats.norm.cdf(y, 1.5, 2.0), atol=1e-12)
    np.testing.assert_allclose(d.mean(), 1.5, atol=1e-12)
    np.testing.assert_allclose(d.std(), 2.0, atol=1e-10)


@pytest.mark.parametrize("eps,delta", [(0.5, 1.0), (-0.8, 0.7), (0.3, 1.6), (1.2, 0.9)])
def test_shash_pdf_integrates_to_one_and_moments_match_quadrature(eps, delta):
    d = SHASH(loc=0.3, scale=1.7, eps=eps, delta=delta)
    total, _ = integrate.quad(d.pdf, -np.inf, np.inf, limit=200)
    m1, _ = integrate.quad(lambda y: y * d.pdf(y), -np.inf, np.inf, limit=200)
    m2, _ = integrate.quad(lambda y: (y - m1) ** 2 * d.pdf(y), -np.inf, np.inf, limit=200)
    assert total == pytest.approx(1.0, abs=1e-7)
    assert float(d.mean()) == pytest.approx(m1, rel=1e-6, abs=1e-8)
    assert float(d.std()) == pytest.approx(np.sqrt(m2), rel=1e-6)


@pytest.mark.parametrize("eps,delta", [(0.5, 1.0), (-0.8, 0.7), (0.3, 1.6)])
def test_shash_cdf_ppf_inverse_and_cdf_is_integral_of_pdf(eps, delta):
    d = SHASH(loc=-1.0, scale=0.5, eps=eps, delta=delta)
    p = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(p)), p, atol=1e-10)
    y0 = float(d.ppf(0.3))
    val, _ = integrate.quad(d.pdf, -np.inf, y0, limit=200)
    assert val == pytest.approx(0.3, abs=1e-7)
    assert float(d.median()) == pytest.approx(float(d.ppf(0.5)), abs=1e-12)


def test_shash_sampling_matches_moments():
    rng = np.random.default_rng(1)
    d = SHASH(
        loc=np.array([0.0, 2.0]),
        scale=np.array([1.0, 0.5]),
        eps=np.array([0.6, -0.4]),
        delta=np.array([0.9, 1.3]),
    )
    s = d.sample(400_000, rng)
    np.testing.assert_allclose(s.mean(axis=-1), d.mean(), atol=0.01)
    np.testing.assert_allclose(s.std(axis=-1), d.std(), rtol=0.01)


def test_shash_positive_eps_gives_right_skew():
    rng = np.random.default_rng(2)
    s = SHASH(0.0, 1.0, 0.8, 1.0).sample(200_000, rng)
    assert stats.skew(s.ravel()) > 0.5


def test_gaussian_basic():
    d = Gaussian(mu=np.array([0.0, 1.0]), sigma=np.array([1.0, 2.0]))
    np.testing.assert_allclose(d.cdf(d.ppf(0.8)), 0.8)
    with pytest.raises(ValueError):
        Gaussian(0.0, -1.0)
