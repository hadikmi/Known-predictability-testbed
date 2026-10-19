"""Formal hypothesis tests for forecast comparison, and multiple-testing control.

Paired score differences
------------------------
``diebold_mariano`` tests H0: E[L_a - L_b] = 0 for two forecasts scored on the
same cases (Diebold & Mariano, 1995, J. Bus. Econ. Stat. 13). The variance of
the mean loss differential is estimated with the Newey-West (1987, Econometrica
55) heteroscedasticity- and autocorrelation-consistent estimator (Bartlett
kernel), and the statistic carries the small-sample correction of Harvey,
Leybourne & Newbold (1997, Int. J. Forecasting 13), compared with a Student-t
distribution with n - 1 degrees of freedom. The truncation lag is chosen from
the data (Andrews, 1991, AR(1) plug-in) unless given. With a proper score such as the
CRPS, a significant difference means one forecast distribution is better on
average, not just its mean.

For data made of several independent series (e.g. ensemble members), the
Newey-West sums are computed within each series only (``groups``), so no lag
products are formed across the boundary between two members.

Confidence ranking
------------------
``circular_shift_test`` tests whether a forecast's own confidence (predicted
spread) is related to its realised errors, H0: no relation. The null
distribution is built by circularly shifting the spread series against the
error series. This keeps the autocorrelation of both series intact and only
breaks their alignment, so serial dependence does not inflate the false-positive
rate (a circular-shift, or "torus-translation", randomisation test). Its
false-positive rate is checked by simulation in ``tests/test_stats_tests.py``.
Shifts shorter than
``min_shift`` are excluded so that near-alignment does not leak the signal.

Multiple testing
----------------
``holm`` (Holm, 1979, Scand. J. Stat. 6) controls the family-wise error rate;
``benjamini_hochberg`` (Benjamini & Hochberg, 1995, JRSS-B 57) controls the
false discovery rate. Both return adjusted p-values.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------- HAC variance


def newey_west_lag(n: int) -> int:
    """Rule-of-thumb truncation lag floor(4 (n/100)^(2/9)) (Newey & West, 1994)."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def andrews_lag(d: np.ndarray, groups: np.ndarray | None = None) -> int:
    """Data-driven Bartlett bandwidth from an AR(1) plug-in (Andrews, 1991,
    Econometrica 59): S = 1.1447 (alpha n)^(1/3), alpha = 4 rho^2 / ((1-rho)^2 (1+rho)^2).

    The lag-1 autocorrelation is estimated within groups and capped at 0.97.
    Adapts to the persistence of the loss differential, unlike the fixed
    rule of thumb, which under-corrects strongly autocorrelated series.
    """
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    same = (
        np.ones(n - 1, bool)
        if groups is None
        else (np.asarray(groups)[1:] == np.asarray(groups)[:-1])
    )
    num = float(np.sum(d[1:][same] * d[:-1][same]))
    den = float(np.sum(d[:-1][same] ** 2)) or np.finfo(float).tiny
    rho = float(np.clip(num / den, -0.97, 0.97))
    alpha = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    S = 1.1447 * (alpha * n) ** (1.0 / 3.0)
    return int(min(np.floor(S), n - 1))


def _autocov_sum(d: np.ndarray, groups: np.ndarray | None, lag: int) -> float:
    """Bartlett-weighted long-run variance of d (mean already removed)."""
    if groups is None:
        groups = np.zeros(len(d), dtype=int)
    n = len(d)
    lrv = float(np.sum(d * d)) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        same = groups[k:] == groups[:-k]  # lag products only within a series
        gamma_k = float(np.sum(d[k:][same] * d[:-k][same])) / n
        lrv += 2.0 * w * gamma_k
    return lrv


def diebold_mariano(
    loss_a,
    loss_b,
    groups=None,
    lag: int | None = None,
    horizon: int = 1,
) -> dict:
    """Diebold-Mariano test with HLN correction and Newey-West variance.

    Returns the mean loss difference (a - b), the statistic, two-sided and
    one-sided (H1: a has lower loss) p-values, and the lag used.
    Cases must be ordered in time within each group.
    """
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if a.shape != b.shape:
        raise ValueError("loss arrays must have the same shape")
    d = a - b
    n = len(d)
    g = None if groups is None else np.asarray(groups)
    if lag is None:
        lag = max(andrews_lag(d, g), horizon - 1)
    dbar = float(d.mean())
    lrv = _autocov_sum(d - dbar, g, lag)
    if lrv <= 0:
        lrv = float(np.var(d)) or np.finfo(float).tiny
    dm = dbar / np.sqrt(lrv / n)
    h = horizon
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t = dm * hln
    p_two = 2.0 * stats.t.sf(abs(t), df=n - 1)
    p_less = stats.t.cdf(t, df=n - 1)  # H1: mean(a - b) < 0
    same = np.ones(n - 1, bool) if g is None else (g[1:] == g[:-1])
    dc = d - dbar
    rho1 = float(np.sum(dc[1:][same] * dc[:-1][same]) / max(np.sum(dc[:-1][same] ** 2), 1e-300))
    return {
        "mean_diff": dbar,
        "lag1_autocorr": rho1,
        "statistic": float(t),
        "p_two_sided": float(p_two),
        "p_a_better": float(p_less),
        "n": n,
        "lag": lag,
    }


# ----------------------------------------------------------------------------- confidence


def circular_shift_test(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    spread,
    error,
    groups=None,
    min_shift: int = 1,
    max_shifts: int | None = None,
    seed: int = 0,
    alternative: str = "less",
) -> dict:
    """Shift test of the association between predicted spread and errors.

    ``statistic(spread, error)`` returns a scalar; with ``alternative='less'``
    smaller values are evidence against H0 (e.g. the relative change in mean
    absolute error of the most-confident half, which should be negative when
    confidence is informative).

    With ``groups`` (e.g. ensemble members of equal length), the spread is
    shifted by the same amount within each group, so pairs never cross
    between groups. All admissible shifts are used unless ``max_shifts`` caps
    them (then a random subset is drawn).
    """
    s, e = np.asarray(spread, float), np.asarray(error, float)
    if groups is None:
        groups = np.zeros(len(s), dtype=int)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    idx = [np.flatnonzero(groups == lab) for lab in labels]
    lengths = {len(i) for i in idx}
    if len(lengths) != 1:
        raise ValueError("all groups must have the same length for a common shift")
    L = lengths.pop()
    shifts = np.arange(min_shift, L - min_shift + 1)
    if len(shifts) == 0:
        raise ValueError("series too short for the requested min_shift")
    if max_shifts is not None and len(shifts) > max_shifts:
        shifts = np.random.default_rng(seed).choice(shifts, size=max_shifts, replace=False)

    observed = float(statistic(s, e))
    null = np.empty(len(shifts))
    for j, k in enumerate(shifts):
        s_shift = s.copy()
        for i in idx:
            s_shift[i] = np.roll(s[i], k)
        null[j] = statistic(s_shift, e)
    if alternative == "less":
        p = (1 + np.sum(null <= observed)) / (1 + len(null))
    elif alternative == "greater":
        p = (1 + np.sum(null >= observed)) / (1 + len(null))
    else:
        c = np.median(null)
        p = (1 + np.sum(np.abs(null - c) >= abs(observed - c))) / (1 + len(null))
    return {
        "statistic": observed,
        "p_value": float(p),
        "n_shifts": int(len(null)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
    }


def confident_half_error_change(spread: np.ndarray, error: np.ndarray) -> float:
    """Relative change in mean |error| of the most-confident half vs all cases."""
    half = np.argsort(spread, kind="stable")[: len(spread) // 2]
    return float(np.abs(error[half]).mean() / np.abs(error).mean() - 1.0)


# ----------------------------------------------------------------------------- multiplicity


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error rate)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank_from_top, i in enumerate(order):
        rank = m - rank_from_top
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj
