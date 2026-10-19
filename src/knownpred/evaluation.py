"""Scoring a set of forecasts against outcomes and, when available, the truth.

Two kinds of metrics are reported and should be read differently:

* **Outcome-based** (CRPS, PIT, coverage, spread-error ratio): what can be
  computed in any real application. They are noisy, because each test case
  contributes one random outcome, and serially correlated.
* **Truth-based** (spread rank correlation, confident-set overlap, log-spread
  error, mean error against the true mean, excess CRPS over the ideal
  forecast): only available in the testbed. They compare the forecast
  distribution with the true distribution directly and are far less noisy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .forecast import Forecast
from .truth import GroundTruth


def evaluate(
    fc: Forecast,
    y: np.ndarray,
    months: np.ndarray,
    crps_ref: np.ndarray,
    truth: GroundTruth | None = None,
    crps_truth: np.ndarray | None = None,
    confident_fraction: float = 0.2,
    block: int = 24,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    y = np.asarray(y, float)
    crps = fc.crps(y)
    pit = fc.pit(y)
    lo, hi = fc.interval(0.9)
    n = len(y)
    res = {
        "model": fc.name,
        "n": n,
        "crps": float(crps.mean()),
        "crpss_vs_clim": M.skill_score(crps, crps_ref),
        "pit_d": M.pit_d(pit),
        "pit_d_expected_iid": M.pit_d_expected(n),
        "coverage90": M.central_interval_coverage(y, lo, hi),
        "spread_error_ratio": M.spread_error_ratio(y, fc.mean, fc.std),
    }
    ci = M.block_bootstrap_ci(
        lambda i: M.skill_score(crps[i], crps_ref[i]), n, block, n_boot=n_boot, seed=seed
    )
    res["crpss_lo"], res["crpss_hi"] = ci["lower"], ci["upper"]
    ci = M.block_bootstrap_ci(
        lambda i: M.spread_error_ratio(y[i], fc.mean[i], fc.std[i]),
        n,
        block,
        n_boot=n_boot,
        seed=seed,
    )
    res["ser_lo"], res["ser_hi"] = ci["lower"], ci["upper"]

    if truth is not None:
        sd_scale = float(np.mean(truth.std))
        res.update(
            {
                "mean_rmse_vs_truth": float(
                    np.sqrt(np.mean((fc.mean - truth.mean) ** 2)) / sd_scale
                ),
                "log_sigma_rmse": M.log_sigma_rmse(fc.std, truth.std),
                "sigma_spearman_all": M.sigma_rank_correlation(fc.std, truth.std),
                "sigma_spearman_within_month": M.within_group_spearman(fc.std, truth.std, months),
                "overlap_all": M.confident_set_overlap(fc.std, truth.std, confident_fraction),
                "overlap_within_month": M.within_group_overlap(
                    fc.std, truth.std, months, confident_fraction
                ),
            }
        )
        if crps_truth is not None:
            excess = crps - crps_truth
            res["excess_crps"] = float(excess.mean())
            ci = M.block_bootstrap_ci(
                lambda i: excess[i].mean(), n, block, n_boot=n_boot, seed=seed
            )
            res["excess_crps_lo"], res["excess_crps_hi"] = ci["lower"], ci["upper"]
    return res


def discard_table(
    forecasts: list[Forecast], y: np.ndarray, truth: GroundTruth | None = None, fractions=None
) -> pd.DataFrame:
    """Mean absolute error of the forecast mean for the most-confident fraction.

    If the truth is given, two references are added: the truth's own mean
    ranked by the true spread ('truth (oracle)'), and the truth's mean
    ranked randomly ('random ranking').
    """
    fractions = np.linspace(0.1, 1.0, 10) if fractions is None else fractions
    rows = {}
    for fc in forecasts:
        f, v = M.discard_curve(np.abs(y - fc.mean), fc.std, fractions)
        rows[fc.name] = v
    if truth is not None:
        _, rows["truth (oracle)"] = M.discard_curve(np.abs(y - truth.mean), truth.std, fractions)
        rng = np.random.default_rng(0)
        _, rows["random ranking"] = M.discard_curve(
            np.abs(y - truth.mean), rng.permutation(len(y)), fractions
        )
    return pd.DataFrame(rows, index=pd.Index(np.round(fractions, 3), name="fraction_retained"))
