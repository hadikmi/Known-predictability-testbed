"""Training, prediction and deep ensembles for the probabilistic networks."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
import torch

from ..forecast import Forecast
from .losses import gaussian_nll, shash_nll
from .networks import ProbabilisticMLP


@dataclass
class TrainConfig:
    family: str = "gaussian"
    hidden: tuple[int, ...] = (64, 64)
    dropout: float = 0.0
    activation: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 30
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        self.hidden = tuple(self.hidden)  # YAML gives lists

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds any third-party code using the legacy global RNG
    torch.manual_seed(seed)


@dataclass
class Standardizer:
    """Mean/std scaling fitted on the training set only."""

    x_mean: np.ndarray = field(default=None)
    x_std: np.ndarray = field(default=None)
    y_mean: float = 0.0
    y_std: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Standardizer:
        self.x_mean = X.to_numpy().mean(0)
        self.x_std = X.to_numpy().std(0)
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean, self.y_std = float(y.mean()), float(y.std())
        return self

    def tx(self, X: pd.DataFrame) -> torch.Tensor:
        return torch.as_tensor((X.to_numpy() - self.x_mean) / self.x_std, dtype=torch.float32)

    def ty(self, y: pd.Series) -> torch.Tensor:
        return torch.as_tensor((y.to_numpy() - self.y_mean) / self.y_std, dtype=torch.float32)


def _loss(family: str, params: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    if family == "gaussian":
        return gaussian_nll(y, params["mu"], params["sigma"])
    return shash_nll(y, params["loc"], params["scale"], params["eps"], params["delta"])


class NetForecaster:
    """Fit one network with early stopping on the validation likelihood."""

    def __init__(self, config: TrainConfig | None = None, name: str | None = None):
        self.config = config or TrainConfig()
        self.name = name or f"NN ({self.config.family})"

    def fit(
        self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> NetForecaster:
        c = self.config
        set_seed(c.seed)
        self.columns_ = list(X.columns)
        self.scaler_ = Standardizer().fit(X, y)
        Xt, yt = self.scaler_.tx(X).to(c.device), self.scaler_.ty(y).to(c.device)
        Xv, yv = self.scaler_.tx(X_val).to(c.device), self.scaler_.ty(y_val).to(c.device)
        self.model_ = ProbabilisticMLP(
            len(self.columns_), c.family, c.hidden, c.dropout, c.activation
        ).to(c.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        gen = torch.Generator(device="cpu").manual_seed(c.seed)

        best, best_state, wait = np.inf, None, 0
        self.history_ = {"train": [], "val": []}
        n = Xt.shape[0]
        for _epoch in range(c.max_epochs):
            self.model_.train()
            perm = torch.randperm(n, generator=gen)
            tot = 0.0
            for i in range(0, n, c.batch_size):
                idx = perm[i : i + c.batch_size].to(c.device)
                opt.zero_grad()
                loss = _loss(c.family, self.model_(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            self.model_.eval()
            with torch.no_grad():
                val = _loss(c.family, self.model_(Xv), yv).item()
            self.history_["train"].append(tot / n)
            self.history_["val"].append(val)
            if val < best - 1e-5:
                best, best_state, wait = val, copy.deepcopy(self.model_.state_dict()), 0
            else:
                wait += 1
                if wait >= c.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.best_val_nll_ = best
        self.n_epochs_ = len(self.history_["val"])
        return self

    @torch.no_grad()
    def predict_params(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Distribution parameters in physical units of the target."""
        self.model_.eval()
        p = {
            k: v.cpu().numpy().astype(float)
            for k, v in self.model_(
                self.scaler_.tx(X[self.columns_]).to(self.config.device)
            ).items()
        }
        ym, ys = self.scaler_.y_mean, self.scaler_.y_std
        if self.config.family == "gaussian":
            return {"mu": p["mu"] * ys + ym, "sigma": p["sigma"] * ys}
        return {
            "loc": p["loc"] * ys + ym,
            "scale": p["scale"] * ys,
            "eps": p["eps"],
            "delta": p["delta"],
        }

    def predict(self, X: pd.DataFrame) -> Forecast:
        p = self.predict_params(X)
        if self.config.family == "gaussian":
            return Forecast.from_gaussian(self.name, p["mu"], p["sigma"], epochs=self.n_epochs_)
        return Forecast.from_shash(
            self.name, p["loc"], p["scale"], p["eps"], p["delta"], epochs=self.n_epochs_
        )


class DeepEnsemble:
    """Equally weighted mixture of independently initialised networks
    (Lakshminarayanan et al., 2017, NeurIPS). The mixture is represented by
    samples so that it can be scored like any ensemble forecast; its mean and
    standard deviation are computed exactly from the components."""

    def __init__(
        self, config: TrainConfig | None = None, n_members: int = 5, samples_per_member: int = 200
    ):
        self.config = config or TrainConfig()
        self.n_members = n_members
        self.samples_per_member = samples_per_member
        self.name = f"deep ensemble ({n_members}x {self.config.family})"

    def fit(self, X, y, X_val, y_val) -> DeepEnsemble:
        self.members_ = []
        for k in range(self.n_members):
            cfg = replace(self.config, seed=self.config.seed + 1000 * (k + 1))
            self.members_.append(NetForecaster(cfg).fit(X, y, X_val, y_val))
        return self

    def predict(self, X: pd.DataFrame, seed: int = 0) -> Forecast:
        rng = np.random.default_rng(seed)
        comps = [m.predict(X) for m in self.members_]
        samples = np.concatenate(
            [c.dist.sample(self.samples_per_member, rng) for c in comps], axis=1
        )
        means = np.stack([c.mean for c in comps])
        stds = np.stack([c.std for c in comps])
        mix_mean = means.mean(0)
        mix_std = np.sqrt((stds**2 + means**2).mean(0) - mix_mean**2)
        f = Forecast.from_members(self.name, samples)
        f.mean, f.std = mix_mean, mix_std
        f.info["epistemic_std"] = means.std(0)
        return f
