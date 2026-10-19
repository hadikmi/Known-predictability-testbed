"""Negative log-likelihood losses (PyTorch).

They mirror ``knownpred.distributions`` term by term; ``tests/test_torch.py``
checks that both give identical values.
"""

from __future__ import annotations

import math

import torch

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of y under N(mu, sigma)."""
    z = (y - mu) / sigma
    return (0.5 * z**2 + torch.log(sigma) + _LOG_SQRT_2PI).mean()


def shash_logpdf(
    y: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, eps: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Log density of the location-scale SHASH (Jones & Pewsey 2009)."""
    x = (y - loc) / scale
    S = torch.sinh(delta * torch.asinh(x) - eps)
    return (
        torch.log(delta)
        + 0.5 * torch.log1p(S**2)  # log C, with C = cosh(.) = sqrt(1 + S^2)
        - 0.5 * torch.log1p(x**2)
        - _LOG_SQRT_2PI
        - 0.5 * S**2
        - torch.log(scale)
    )


def shash_nll(y, loc, scale, eps, delta) -> torch.Tensor:
    return -shash_logpdf(y, loc, scale, eps, delta).mean()
