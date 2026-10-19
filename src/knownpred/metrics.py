"""Verification metrics for probabilistic forecasts.

References
----------
Gneiting & Raftery (2007), JASA 102(477): proper scoring rules, CRPS.
Gneiting et al. (2005), MWR 133: closed-form CRPS of the Gaussian.
Ferro (2014), QJRMS 140: the 'fair' ensemble CRPS.
Hamill (2001), MWR 129: rank / PIT histograms.
Nipen & Stull (2011), Tellus A 63: PIT-histogram deviation statistic.
Gordon & Barnes (2022), Geophys. Res. Lett.: discard plots of error versus fraction of
most-confident forecasts retained, as used in the forecasts-of-opportunity
literature; implemented here independently.
Künsch (1989), Ann. Stat. 17; Politis & Romano (1992): block bootstrap for
dependent data (circular block bootstrap implemented here).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- CRPS


def crps_gaussian(y, mu, sigma) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma) for outcomes y (Gneiting et al. 2005)."""
    y, mu, sigma = np.broadcast_arrays(
        np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_ensemble(y, members, fair: bool = False) -> np.ndarray:
    """CRPS of an ensemble forecast. ``members`` has shape (N, M).

    Uses CRPS = E|X - y| - 0.5 E|X - X'| with the O(M log M) sorted-ensemble
    identity for the second term. ``fair=True`` gives the unbiased estimator
    of Ferro (2014), appropriate for small ensembles.
    """
    y = np.asarray(y, float)
    x = np.sort(np.asarray(members, float), axis=-1)
    M = x.shape[-1]
    term1 = np.abs(x - y[..., None]).mean(axis=-1)
    i = np.arange(1, M + 1)
    pair_sum = 2.0 * np.sum((2 * i - M - 1) * x, axis=-1)  # sum_{i,j} |x_i - x_j|
    denom = M * (M - 1) if fair else M * M
    return term1 - 0.5 * pair_sum / denom


def crps_from_ppf(y, ppf: Callable[[np.ndarray], np.ndarray], n_levels: int = 400) -> np.ndarray:
    """CRPS of any continuous forecast from its quantile function.

    CRPS(F, y) = 2 * int_0^1 QS_p(F^{-1}(p), y) dp, with the quantile score
    QS_p(q, y) = (1{y < q} - p)(q - y). Integrated with the midpoint rule on
    ``n_levels`` levels; error is O(1/n_levels^2) away from the tails.
    ``ppf`` must accept an array of shape (n_levels, 1) and broadcast it
    against the forecast parameters of shape (N,).
    """
    y = np.asarray(y, float)
    p = (np.arange(n_levels) + 0.5) / n_levels
    q = np.asarray(ppf(p[:, None]))  # (n_levels, N)
    qs = ((y[None, :] < q).astype(float) - p[:, None]) * (q - y[None, :])
    return 2.0 * qs.mean(axis=0)


# --------------------------------------------------------------------------- PIT / calibration


def pit_values(y, cdf: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Probability integral transform F(y) for continuous forecasts."""
    return np.asarray(cdf(np.asarray(y, float)))


def pit_ensemble(y, members, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomised PIT for ensembles (ties broken uniformly)."""
    rng = rng or np.random.default_rng(0)
    members = np.asarray(members, float)
    below = (members < np.asarray(y)[:, None]).sum(axis=1)
    equal = (members == np.asarray(y)[:, None]).sum(axis=1)
    M = members.shape[1]
    return (below + rng.uniform(size=below.shape) * (equal + 1)) / (M + 1)


def pit_histogram(pit: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return counts / counts.sum(), edges


def pit_d(pit: np.ndarray, n_bins: int = 10) -> float:
    """PIT deviation D = sqrt(mean_k (f_k - 1/B)^2) (Nipen & Stull 2011)."""
    freq, _ = pit_histogram(pit, n_bins)
    return float(np.sqrt(np.mean((freq - 1.0 / n_bins) ** 2)))


def pit_d_expected(n: int, n_bins: int = 10) -> float:
    """Expected D for a perfectly calibrated forecast with n samples."""
    return float(np.sqrt((1.0 - 1.0 / n_bins) / (n * n_bins)))


def central_interval_coverage(y, lower, upper) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lower) & (y <= upper)))


def spread_skill(y, mean, sigma, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Binned spread-skill relation.

    Samples are grouped into quantile bins of predicted sigma; for a reliable
    forecast the RMSE of the mean equals the RMS predicted sigma in every bin.
    """
    y, mean, sigma = (np.asarray(a, float) for a in (y, mean, sigma))
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, sigma, side="right") - 1, 0, n_bins - 1)
    rms_sigma = np.array([np.sqrt(np.mean(sigma[idx == b] ** 2)) for b in range(n_bins)])
    rmse = np.array([np.sqrt(np.mean((y[idx == b] - mean[idx == b]) ** 2)) for b in range(n_bins)])
    return {"rms_sigma": rms_sigma, "rmse": rmse, "ratio": rms_sigma / rmse}


def spread_error_ratio(y, mean, sigma) -> float:
    """RMS predicted sigma / RMSE of the mean (1 = reliable on average)."""
    return float(
        np.sqrt(np.mean(np.asarray(sigma) ** 2)) / np.sqrt(np.mean((np.asarray(y) - mean) ** 2))
    )


# --------------------------------------------------------------------------- confidence


def discard_curve(
    error: np.ndarray, confidence_key: np.ndarray, fractions: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean error of the most-confident fraction of forecasts.

    ``confidence_key``: smaller = more confident (e.g. predicted sigma).
    Returns (fractions retained, mean error of the retained samples).
    """
    if fractions is None:
        fractions = np.linspace(0.05, 1.0, 20)
    order = np.argsort(confidence_key, kind="stable")
    e = np.asarray(error)[order]
    n = len(e)
    vals = np.array([e[: max(1, round(f * n))].mean() for f in fractions])
    return np.asarray(fractions), vals


def confident_set_overlap(pred_sigma, true_sigma, fraction: float = 0.2) -> float:
    """Share of the truly most-predictable ``fraction`` of cases that the
    model also places in its most-confident ``fraction`` (1 = perfect, ``fraction`` = chance)."""
    n = len(pred_sigma)
    k = max(1, round(fraction * n))
    top_pred = set(np.argsort(pred_sigma, kind="stable")[:k])
    top_true = set(np.argsort(true_sigma, kind="stable")[:k])
    return len(top_pred & top_true) / k


def sigma_rank_correlation(pred_sigma, true_sigma) -> float:
    return float(stats.spearmanr(pred_sigma, true_sigma).statistic)


def within_group_spearman(pred, true, groups) -> float:
    """Mean Spearman correlation computed separately within each group
    (e.g. issue month) and averaged with equal weights.

    Removes the seasonal cycle of spread, which any method that knows the
    calendar month can reproduce, and isolates state-dependent skill. A
    forecast whose spread is constant within a group scores 0 for it.
    """
    pred, true, groups = (np.asarray(a) for a in (pred, true, groups))
    vals = []
    for g in np.unique(groups):
        m = groups == g
        if np.ptp(pred[m]) == 0:
            vals.append(0.0)
        else:
            vals.append(stats.spearmanr(pred[m], true[m]).statistic)
    return float(np.mean(vals))


def within_group_overlap(pred_sigma, true_sigma, groups, fraction: float = 0.2) -> float:
    """``confident_set_overlap`` computed within each group, then averaged."""
    pred_sigma, true_sigma, groups = (np.asarray(a) for a in (pred_sigma, true_sigma, groups))
    return float(
        np.mean(
            [
                confident_set_overlap(pred_sigma[groups == g], true_sigma[groups == g], fraction)
                for g in np.unique(groups)
            ]
        )
    )


def log_sigma_rmse(pred_sigma, true_sigma) -> float:
    """RMS of log(pred / true) spread: 0.1 ~ 10 % typical spread error."""
    return float(np.sqrt(np.mean(np.log(np.asarray(pred_sigma) / np.asarray(true_sigma)) ** 2)))


def skill_score(score, reference) -> float:
    """1 - score / reference for negatively oriented scores (e.g. CRPS)."""
    return float(1.0 - np.mean(score) / np.mean(reference))


# --------------------------------------------------------------------------- bootstrap


def circular_block_bootstrap_indices(
    n: int, block_length: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Index arrays (n_boot, n) for the circular block bootstrap.

    Blocks of ``block_length`` consecutive indices start at uniformly random
    positions and wrap around the end of the series, which keeps every
    observation equally likely to be drawn (Politis & Romano 1992).
    """
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be in [1, n]")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def block_bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    block_length: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile confidence interval of ``statistic(idx)`` under a circular
    block bootstrap of a time-ordered sample of length n.

    ``statistic`` receives an index array and returns a scalar, so paired
    statistics (e.g. a skill-score difference between two models evaluated
    on the same cases) resample both models consistently.
    """
    rng = np.random.default_rng(seed)
    idx = circular_block_bootstrap_indices(n, block_length, n_boot, rng)
    boot = np.array([statistic(i) for i in idx])
    est = statistic(np.arange(n))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(est),
        "lower": float(lo),
        "upper": float(hi),
        "boot_std": float(boot.std(ddof=1)),
    }
