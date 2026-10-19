"""A common container for probabilistic forecasts from any method.

Every method (climatology, linear regression, gradient boosting, neural
networks, the Monte Carlo truth) returns a ``Forecast`` so that all of them
are scored by exactly the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .distributions import SHASH, Gaussian


@dataclass
class Forecast:
    name: str
    mean: np.ndarray
    std: np.ndarray
    dist: Gaussian | SHASH | None = None  # parametric forecasts
    members: np.ndarray | None = None  # ensemble / mixture forecasts, shape (N, M)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.dist is None) == (self.members is None):
            raise ValueError("provide exactly one of `dist` or `members`")
        self.mean = np.asarray(self.mean, float)
        self.std = np.asarray(self.std, float)

    # --- scores -------------------------------------------------------------
    def crps(self, y) -> np.ndarray:
        if self.members is not None:
            return metrics.crps_ensemble(y, self.members, fair=True)
        if isinstance(self.dist, Gaussian):
            return metrics.crps_gaussian(y, self.dist.mu, self.dist.sigma)
        return metrics.crps_from_ppf(y, self.dist.ppf, n_levels=400)

    def pit(self, y) -> np.ndarray:
        if self.members is not None:
            return metrics.pit_ensemble(y, self.members)
        return metrics.pit_values(y, self.dist.cdf)

    def interval(self, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        if self.members is not None:
            q = np.quantile(self.members, [lo, hi], axis=1)
            return q[0], q[1]
        return self.dist.ppf(lo), self.dist.ppf(hi)

    def logscore(self, y) -> np.ndarray | None:
        """Negative log predictive density; None for ensemble forecasts."""
        if self.dist is None:
            return None
        return -self.dist.logpdf(y)

    @classmethod
    def from_gaussian(cls, name, mu, sigma, **info) -> Forecast:
        return cls(name, mu, sigma, dist=Gaussian(mu, sigma), info=info)

    @classmethod
    def from_shash(cls, name, loc, scale, eps, delta, **info) -> Forecast:
        d = SHASH(loc, scale, eps, delta)
        return cls(name, d.mean(), d.std(), dist=d, info=info)

    @classmethod
    def from_members(cls, name, members, **info) -> Forecast:
        members = np.asarray(members, float)
        return cls(name, members.mean(1), members.std(1, ddof=1), members=members, info=info)
