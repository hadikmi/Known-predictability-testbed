"""Small fully connected networks that output distribution parameters.

The architecture is intentionally simple (a few-layer MLP on seven tabular
predictors): the question is whether the *uncertainty output* is right, not
how far architecture search can push the mean.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ("gaussian", "shash")
N_PARAMS = {"gaussian": 2, "shash": 4}

# Bounds keep the SHASH parameters in a numerically safe range. delta in
# [0.3, 3] spans tails much heavier and much lighter than Gaussian; |eps| <= 3
# allows strong skewness. Both are wide for standardised targets.
_MIN_SIGMA = 1e-3
_EPS_MAX = 3.0
_DELTA_MIN, _DELTA_MAX = 0.3, 3.0


class ProbabilisticMLP(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        family: str = "gaussian",
        hidden: tuple[int, ...] = (64, 64),
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.family = family
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, N_PARAMS[family])
        # Start near a standard normal in standardised target space:
        # mu = 0, sigma ~ 1 (softplus^-1(1) ~ 0.5413), eps = 0, delta ~ 1.
        nn.init.normal_(self.head.weight, std=1e-3)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[1] = 0.5413
            if family == "shash":
                self.head.bias[3] = _delta_raw_for(1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.head(self.body(x))
        sigma = F.softplus(out[:, 1]) + _MIN_SIGMA
        if self.family == "gaussian":
            return {"mu": out[:, 0], "sigma": sigma}
        eps = _EPS_MAX * torch.tanh(out[:, 2] / _EPS_MAX)
        delta = _DELTA_MIN + (_DELTA_MAX - _DELTA_MIN) * torch.sigmoid(out[:, 3])
        return {"loc": out[:, 0], "scale": sigma, "eps": eps, "delta": delta}


def _delta_raw_for(delta: float) -> float:
    p = (delta - _DELTA_MIN) / (_DELTA_MAX - _DELTA_MIN)
    return float(torch.logit(torch.tensor(p)))
