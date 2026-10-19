"""Attribution of the predicted spread to input variables.

Two complementary tools:

* **Grouped permutation importance of the spread** (model-agnostic,
  NumPy). For a predictor group j, the inputs of group j are shuffled across
  samples and the importance is the mean absolute relative change of the
  predicted standard deviation,  mean |sigma(X_perm) - sigma(X)| / mean sigma(X).
  Calendar month (its sine and cosine) is shuffled as one group so that
  no impossible month encodings are created. Because the toy system's true
  forecast distribution can be evaluated at *any* state by Monte Carlo, the
  same quantity can be computed for the truth, which gives a reference
  attribution that real-world XAI studies do not have. Shuffled states can
  lie off the system's attractor; the importance therefore measures the
  sensitivity of the spread function, not a physical causal effect.

* **Integrated gradients** (Sundararajan et al., 2017, ICML) of the network's
  log predicted spread, for PyTorch models (see ``integrated_gradients``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

DEFAULT_GROUPS = {
    "T": ["T"],
    "H": ["H"],
    "S": ["S"],
    "month": ["month_sin", "month_cos"],
    "distractor_1": ["distractor_1"],
    "distractor_2": ["distractor_2"],
}


def permutation_spread_importance(
    sigma_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
    n_repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Grouped permutation importance of a spread function.

    Returns a DataFrame with the mean and standard deviation (over repeats)
    of the importance of every group.
    """
    groups = groups or DEFAULT_GROUPS
    rng = np.random.default_rng(seed)
    base = np.asarray(sigma_fn(X))
    rows = []
    for name, cols in groups.items():
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            vals.append(np.mean(np.abs(np.asarray(sigma_fn(Xp)) - base)) / np.mean(base))
        rows.append({"group": name, "importance": float(np.mean(vals)), "std": float(np.std(vals))})
    return pd.DataFrame(rows).set_index("group")


def truth_sigma_fn(
    system, lead: int, n_members: int = 400, seed: int = 0
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Spread function of the true forecast distribution (Monte Carlo).

    Maps predictor rows back to the Markov state (T, H, S, month) and returns
    the ensemble standard deviation of S at ``lead``. Distractor columns are
    ignored, as they are by the system itself.
    """
    from .baselines import issue_month
    from .simulate import SystemState

    def fn(X: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(seed)  # common random numbers across calls
        init = SystemState(
            T=X["T"].to_numpy(float),
            H=X["H"].to_numpy(float),
            S=X["S"].to_numpy(float),
            month=issue_month(X),
        )
        return system.forecast_ensemble(init, lead, n_members, rng).std(axis=1, ddof=1)

    return fn


def integrated_gradients(
    model, X: pd.DataFrame, output: str = "log_sigma", n_steps: int = 64
) -> pd.DataFrame:
    """Integrated gradients of a NetForecaster output w.r.t. its inputs.

    Baseline: the training-mean input (zero in standardised space). Output
    is in standardised input units; the completeness property
    sum_j IG_j = f(x) - f(baseline) is checked in the tests.
    ``output``: 'log_sigma' (log of the spread / scale parameter) or 'mu'
    (mean or location parameter).
    """
    import torch

    net = model.model_
    net.eval()
    x = model.scaler_.tx(X[model.columns_])
    baseline = torch.zeros_like(x)
    alphas = (torch.arange(n_steps, dtype=torch.float32) + 0.5) / n_steps  # midpoint rule
    total = torch.zeros_like(x)
    for a in alphas:
        xi = (baseline + a * (x - baseline)).requires_grad_(True)
        f = _scalar_output(net(xi), output)
        (g,) = torch.autograd.grad(f.sum(), xi)
        total += g
    ig = ((x - baseline) * total / n_steps).detach().numpy()
    return pd.DataFrame(ig, columns=model.columns_, index=X.index)


def _scalar_output(params, output: str):
    import torch

    if output == "log_sigma":
        return torch.log(params["sigma"] if "sigma" in params else params["scale"])
    if output == "mu":
        return params["mu"] if "mu" in params else params["loc"]
    raise ValueError(output)
