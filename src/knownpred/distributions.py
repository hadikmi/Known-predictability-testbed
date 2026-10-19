"""Parametric predictive distributions (NumPy reference implementations).

Two families are used for the network outputs:

* **Gaussian** N(mu, sigma).
* **Sinh-arcsinh (SHASH)** distribution of Jones & Pewsey (2009, Biometrika
  96(4), 761-780), with a location-scale extension. A standardised SHASH
  variable X with skewness parameter ``eps`` and tail-weight parameter
  ``delta > 0`` is defined through

      Z = sinh(delta * asinh(X) - eps) ~ N(0, 1),

  so that X = sinh((asinh(Z) + eps) / delta). With eps = 0 and delta = 1 it
  reduces to the standard normal. The location-scale variable is
  Y = loc + scale * X. Note that ``loc`` and ``scale`` are not the mean and
  standard deviation of Y unless eps = 0 and delta = 1; ``mean()`` and
  ``std()`` return the actual moments (closed form from Jones & Pewsey 2009,
  Section 3).

These NumPy versions are the reference against which the PyTorch loss
functions in ``knownpred.models.losses`` are tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


@dataclass
class Gaussian:
    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if np.any(self.sigma <= 0):
            raise ValueError("sigma must be positive")

    def logpdf(self, y):
        z = (np.asarray(y) - self.mu) / self.sigma
        return -0.5 * z**2 - np.log(self.sigma) - _LOG_SQRT_2PI

    def pdf(self, y):
        return np.exp(self.logpdf(y))

    def cdf(self, y):
        return stats.norm.cdf(np.asarray(y), self.mu, self.sigma)

    def ppf(self, p):
        return stats.norm.ppf(p, self.mu, self.sigma)

    def mean(self):
        return self.mu

    def std(self):
        return self.sigma

    def median(self):
        return self.mu

    def sample(self, n: int, rng: np.random.Generator):
        return self.mu[..., None] + self.sigma[..., None] * rng.standard_normal(
            (*np.shape(self.mu), n)
        )


def _shash_P(q: float) -> float:
    """P_q from Jones & Pewsey (2009), eq. for the moments of the SHASH."""
    k = 0.25
    return (
        np.exp(k) / np.sqrt(8.0 * np.pi) * (special.kv((q + 1) / 2, k) + special.kv((q - 1) / 2, k))
    )


@dataclass
class SHASH:
    loc: np.ndarray
    scale: np.ndarray
    eps: np.ndarray  # skewness parameter (eps > 0: right skew)
    delta: np.ndarray  # tail weight (delta < 1: heavier tails than normal)

    def __post_init__(self):
        self.loc = np.asarray(self.loc, dtype=float)
        self.scale = np.asarray(self.scale, dtype=float)
        self.eps = np.asarray(self.eps, dtype=float)
        self.delta = np.asarray(self.delta, dtype=float)
        if np.any(self.scale <= 0) or np.any(self.delta <= 0):
            raise ValueError("scale and delta must be positive")

    # standardised helpers
    def _x(self, y):
        return (np.asarray(y, dtype=float) - self.loc) / self.scale

    def _S(self, x):
        return np.sinh(self.delta * np.arcsinh(x) - self.eps)

    def logpdf(self, y):
        x = self._x(y)
        S = self._S(x)
        C = np.sqrt(1.0 + S**2)
        return (
            np.log(self.delta)
            + np.log(C)
            - 0.5 * np.log1p(x**2)
            - _LOG_SQRT_2PI
            - 0.5 * S**2
            - np.log(self.scale)
        )

    def pdf(self, y):
        return np.exp(self.logpdf(y))

    def cdf(self, y):
        return special.ndtr(self._S(self._x(y)))

    def ppf(self, p):
        z = special.ndtri(np.asarray(p, dtype=float))
        x = np.sinh((np.arcsinh(z) + self.eps) / self.delta)
        return self.loc + self.scale * x

    def median(self):
        return self.loc + self.scale * np.sinh(self.eps / self.delta)

    def mean(self):
        P1 = np.vectorize(_shash_P)(1.0 / self.delta)
        mx = np.sinh(self.eps / self.delta) * P1
        return self.loc + self.scale * mx

    def std(self):
        P1 = np.vectorize(_shash_P)(1.0 / self.delta)
        P2 = np.vectorize(_shash_P)(2.0 / self.delta)
        mx = np.sinh(self.eps / self.delta) * P1
        ex2 = 0.5 * (np.cosh(2.0 * self.eps / self.delta) * P2 - 1.0)
        return self.scale * np.sqrt(np.maximum(ex2 - mx**2, 0.0))

    def sample(self, n: int, rng: np.random.Generator):
        z = rng.standard_normal((*np.shape(self.loc), n))
        x = np.sinh((np.arcsinh(z) + self.eps[..., None]) / self.delta[..., None])
        return self.loc[..., None] + self.scale[..., None] * x
