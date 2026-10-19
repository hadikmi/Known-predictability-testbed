"""Tests for the PyTorch models. Skipped when PyTorch is not installed."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from knownpred.attribution import integrated_gradients  # noqa: E402
from knownpred.distributions import SHASH, Gaussian  # noqa: E402
from knownpred.models.losses import gaussian_nll, shash_logpdf  # noqa: E402
from knownpred.models.networks import ProbabilisticMLP  # noqa: E402
from knownpred.models.train import DeepEnsemble, NetForecaster, TrainConfig  # noqa: E402


def test_torch_losses_match_numpy_reference():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    loc, scale = rng.normal(size=50), np.exp(rng.normal(size=50) * 0.3)
    eps, delta = rng.normal(size=50) * 0.5, np.exp(rng.normal(size=50) * 0.3)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    np.testing.assert_allclose(
        shash_logpdf(t(y), t(loc), t(scale), t(eps), t(delta)).numpy(),
        SHASH(loc, scale, eps, delta).logpdf(y),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        gaussian_nll(t(y), t(loc), t(scale)).item(),
        -Gaussian(loc, scale).logpdf(y).mean(),
        atol=1e-10,
    )


def test_network_initialised_near_standard_normal():
    net = ProbabilisticMLP(7, "shash")
    p = net(torch.zeros(4, 7))
    np.testing.assert_allclose(p["scale"].detach().numpy(), 1.0, atol=0.02)
    np.testing.assert_allclose(p["delta"].detach().numpy(), 1.0, atol=0.02)
    np.testing.assert_allclose(p["eps"].detach().numpy(), 0.0, atol=0.02)


def _heteroscedastic_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 2))
    sigma = 0.2 + 0.8 * (x[:, 0] > 0)  # spread depends on x0 only
    y = x[:, 1] + sigma * rng.standard_normal(n)
    return pd.DataFrame(x, columns=["a", "b"]), pd.Series(y), sigma


@pytest.mark.slow
@pytest.mark.parametrize("family", ["gaussian", "shash"])
def test_network_learns_heteroscedastic_spread(family):
    X, y, _ = _heteroscedastic_data(4000, 1)
    Xv, yv, _ = _heteroscedastic_data(1000, 2)
    Xt, _, sig_t = _heteroscedastic_data(1000, 3)
    m = NetForecaster(
        TrainConfig(family=family, hidden=(32, 32), max_epochs=200, patience=20, seed=0)
    ).fit(X, y, Xv, yv)
    fc = m.predict(Xt)
    lo, hi = sig_t == 0.2, sig_t == 1.0
    assert fc.std[lo].mean() == pytest.approx(0.2, rel=0.3)
    assert fc.std[hi].mean() == pytest.approx(1.0, rel=0.2)


@pytest.mark.slow
def test_training_is_reproducible_and_ensemble_runs():
    X, y, _ = _heteroscedastic_data(1000, 1)
    Xv, yv, _ = _heteroscedastic_data(300, 2)
    cfg = TrainConfig(hidden=(16,), max_epochs=20, patience=5, seed=3)
    a = NetForecaster(cfg).fit(X, y, Xv, yv).predict(Xv)
    b = NetForecaster(cfg).fit(X, y, Xv, yv).predict(Xv)
    np.testing.assert_allclose(a.mean, b.mean)
    ens = DeepEnsemble(cfg, n_members=2, samples_per_member=50).fit(X, y, Xv, yv).predict(Xv)
    assert ens.members.shape == (len(Xv), 100)
    assert np.all(ens.std >= ens.info["epistemic_std"])


def test_integrated_gradients_completeness():
    X, y, _ = _heteroscedastic_data(500, 1)
    m = NetForecaster(TrainConfig(hidden=(16,), max_epochs=5, patience=5)).fit(X, y, X, y)
    ig = integrated_gradients(m, X.iloc[:20], output="log_sigma", n_steps=200)
    with torch.no_grad():
        x = m.scaler_.tx(X.iloc[:20])
        f_x = torch.log(m.model_(x)["sigma"]).numpy()
        f_0 = torch.log(m.model_(torch.zeros_like(x))["sigma"]).numpy()
    np.testing.assert_allclose(ig.sum(axis=1).to_numpy(), f_x - f_0, atol=1e-3)
