"""Non-neural reference forecasts.

These set the bar a neural network has to clear. If a two-stage gradient
boosting model recovers the true predictability as well as a network does,
the network is not needed for this problem.

* ``Climatology``: for each issue month, the empirical distribution of the
  training targets issued in that month (no state information).
* ``SeasonalLinear``: ordinary least squares on the predictors and their
  interactions with the first annual harmonic, with a Gaussian error whose
  standard deviation depends on issue month only. Captures the mean signal
  but, by construction, no state-dependent spread.
* ``TwoStageBoosting``: a heteroscedastic Gaussian built from two
  histogram gradient-boosting regressors (scikit-learn): one for the mean,
  one for log squared out-of-fold residuals (the residual-variance approach
  of e.g. Davidian & Carroll, 1987, JASA 82). Its overall spread scale is
  then calibrated on the validation set with a single multiplicative factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .forecast import Forecast

# E[log chi^2_1] = digamma(1/2) + log 2; bias of log(r^2) as an estimator of log(sigma^2)
_E_LOG_CHI2_1 = float(special.digamma(0.5) + np.log(2.0))


def issue_month(X: pd.DataFrame) -> np.ndarray:
    """Recover calendar month (1-12) from the month_sin / month_cos columns.

    Tables without these columns (one issue month only, as in Part B) map
    every row to month 0, i.e. a single group.
    """
    if "month_sin" not in X.columns:
        return np.zeros(len(X), dtype=int)
    ang = np.arctan2(X["month_sin"].to_numpy(), X["month_cos"].to_numpy())
    m = np.rint(ang / (2 * np.pi) * 12).astype(int) % 12
    return np.where(m == 0, 12, m)


class Climatology:
    name = "climatology"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Climatology:
        m = issue_month(X)
        self.samples_ = {k: np.sort(y.to_numpy()[m == k]) for k in np.unique(m)}
        return self

    def predict(self, X: pd.DataFrame, n_members: int = 200) -> Forecast:
        m = issue_month(X)
        p = (np.arange(n_members) + 0.5) / n_members
        members = np.vstack([np.quantile(self.samples_[k], p) for k in m])
        return Forecast.from_members(self.name, members)


def _linear_design(X: pd.DataFrame) -> np.ndarray:
    if "month_sin" not in X.columns:
        return X.to_numpy()
    base = X.drop(columns=["month_sin", "month_cos"])
    cols = [base.to_numpy(), X[["month_sin", "month_cos"]].to_numpy()]
    for h in ("month_sin", "month_cos"):
        cols.append(base.to_numpy() * X[[h]].to_numpy())
    return np.hstack(cols)


def leave_one_out_climatology(y: np.ndarray, name: str = "climatology (leave-one-out)"):
    """Cross-validated climatology for a short record: the forecast for
    sample i is the empirical distribution of all other samples."""
    y = np.asarray(y, float)
    n = len(y)
    members = np.vstack([np.delete(y, i) for i in range(n)])
    return Forecast.from_members(name, members)


class SeasonalLinear:
    name = "linear (seasonal sigma)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalLinear:
        A = _linear_design(X)
        self.model_ = LinearRegression().fit(A, y.to_numpy())
        resid = y.to_numpy() - self.model_.predict(A)
        m = issue_month(X)
        self.sigma_ = {k: resid[m == k].std(ddof=1) for k in np.unique(m)}
        if "month_sin" not in X.columns:
            self.name = "linear (constant sigma)"
        return self

    def predict(self, X: pd.DataFrame) -> Forecast:
        mu = self.model_.predict(_linear_design(X))
        sigma = np.array([self.sigma_[k] for k in issue_month(X)])
        return Forecast.from_gaussian(self.name, mu, sigma)


class TwoStageBoosting:
    name = "gradient boosting (two-stage)"

    def __init__(
        self, seed: int = 0, n_folds: int = 5, max_iter: int = 400, learning_rate: float = 0.05
    ):
        self.seed, self.n_folds = seed, n_folds
        self.kw = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> TwoStageBoosting:
        Xa, ya = X.to_numpy(), y.to_numpy()
        # Out-of-fold residuals avoid fitting the variance model to in-sample
        # residuals, which are shrunk by overfitting. Contiguous folds
        # (no shuffling) limit leakage through autocorrelation.
        oof = np.empty_like(ya)
        for tr, te in KFold(self.n_folds, shuffle=False).split(Xa):
            oof[te] = HistGradientBoostingRegressor(**self.kw).fit(Xa[tr], ya[tr]).predict(Xa[te])
        self.mean_model_ = HistGradientBoostingRegressor(**self.kw).fit(Xa, ya)
        log_r2 = np.log((ya - oof) ** 2 + 1e-12)
        self.var_model_ = HistGradientBoostingRegressor(**self.kw).fit(Xa, log_r2)
        # one-parameter recalibration of the spread on the validation set
        mu_v, s_v = self._raw(X_val)
        self.scale_ = float(np.sqrt(np.mean(((y_val.to_numpy() - mu_v) / s_v) ** 2)))
        return self

    def _raw(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xa = X.to_numpy()
        mu = self.mean_model_.predict(Xa)
        sigma = np.sqrt(np.exp(self.var_model_.predict(Xa) - _E_LOG_CHI2_1))
        return mu, sigma

    def predict(self, X: pd.DataFrame) -> Forecast:
        mu, sigma = self._raw(X)
        return Forecast.from_gaussian(self.name, mu, sigma * self.scale_, spread_scale=self.scale_)
