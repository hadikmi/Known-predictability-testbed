"""Monte Carlo ground truth for the forecast distribution of the toy system.

For each test initial state the true conditional distribution of the target
is approximated by ``n_members`` independent stochastic continuations of the
same system. Because the predictors contain the full Markov state
(T, H, S, calendar month), this ensemble *is* the ideal probabilistic
forecast; the only approximation is Monte Carlo sampling error, which is
reported explicitly (standard error of the ensemble mean and standard
deviation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import crps_ensemble
from .simulate import SystemState, ToySystem


@dataclass
class GroundTruth:
    """Summary of the true forecast distribution for N initial states."""

    mean: np.ndarray  # (N,)
    std: np.ndarray  # (N,)
    quantiles: np.ndarray  # (N, n_q)
    quantile_levels: np.ndarray  # (n_q,)
    members: np.ndarray  # (N, M) ensemble members (kept for CRPS / PIT)
    se_mean: np.ndarray  # Monte Carlo standard error of the mean
    se_std: np.ndarray  # approximate MC standard error of the std

    def oracle_crps(self, y: np.ndarray) -> np.ndarray:
        """CRPS of the ideal (true-distribution) forecast for outcomes y."""
        return crps_ensemble(y, self.members)

    def to_frame(self, index=None) -> pd.DataFrame:
        cols = {"true_mean": self.mean, "true_std": self.std}
        for q, lev in zip(self.quantiles.T, self.quantile_levels, strict=True):
            cols[f"true_q{lev:.2f}"] = q
        return pd.DataFrame(cols, index=index)


def compute_ground_truth(
    system: ToySystem,
    df_states: pd.DataFrame,
    lead: int,
    n_members: int = 1000,
    seed: int = 0,
    quantile_levels=(0.05, 0.25, 0.5, 0.75, 0.95),
) -> GroundTruth:
    """Re-simulate ``n_members`` continuations from every row of ``df_states``.

    ``df_states`` must contain end-of-month columns T, H, S and month.
    """
    rng = np.random.default_rng(seed)
    init = SystemState(
        T=df_states["T"].to_numpy(dtype=float),
        H=df_states["H"].to_numpy(dtype=float),
        S=df_states["S"].to_numpy(dtype=float),
        month=df_states["month"].to_numpy(dtype=int),
    )
    members = system.forecast_ensemble(init, lead=lead, n_members=n_members, rng=rng)
    mean = members.mean(axis=1)
    std = members.std(axis=1, ddof=1)
    q_levels = np.asarray(quantile_levels, dtype=float)
    quantiles = np.quantile(members, q_levels, axis=1).T
    # Standard errors. For the std, use the large-sample formula that holds for
    # any distribution: Var(s^2) ~ (m4 - s^4) / M, then delta method.
    m4 = ((members - mean[:, None]) ** 4).mean(axis=1)
    se_var = np.sqrt(np.maximum(m4 - std**4, 0.0) / n_members)
    se_std = se_var / (2.0 * std)
    return GroundTruth(
        mean=mean,
        std=std,
        quantiles=quantiles,
        quantile_levels=q_levels,
        members=members,
        se_mean=std / np.sqrt(n_members),
        se_std=se_std,
    )
