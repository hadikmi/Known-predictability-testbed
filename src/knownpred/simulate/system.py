"""Stochastic toy climate system with exactly computable predictability.

The system has three coupled components, integrated with a monthly time unit:

1. **ENSO-like recharge oscillator** (after Jin, 1997, J. Atmos. Sci.;
   linear stochastic form as in Burgers et al., 2005, GRL), written in
   scaled units:

       dT = (-lambda_T(m) T + omega H) dt + sigma_T * g(T) dW_T
       dH = (-omega T - lambda_H H) dt + sigma_H dW_H

   T is an eastern-Pacific SST anomaly [K] and H a thermocline-depth
   anomaly expressed in K-equivalent units. The seasonally varying damping
   lambda_T(m) (strongest in boreal spring) produces seasonal phase locking.
   g(T) = max(1 + b*T, g_min) is state-dependent (multiplicative) noise, a
   simplified form of the state-dependent noise discussed for ENSO by e.g.
   Levine & Jin (2010, J. Atmos. Sci.). It makes forecast spread depend on
   the ENSO state.

2. **Teleconnected regional precipitation** [mm/day], monthly:

       P_t = beta(m) * Tbar_t + s(m, Tbar_t) * eps_t

   Tbar_t is the monthly-mean T, beta(m) is non-zero only in the cool
   season (Nov-Mar), and eps_t is zero-mean, unit-variance, positively
   skewed (centred Gamma) noise, so the forecast distribution is non-Gaussian.
   The noise amplitude s(m, T) has a seasonal cycle and, in the cool season
   only, grows with T (wetter and more variable during El Nino, drier and less
   variable during La Nina). This is a second, deliberately built-in source
   of state-dependent predictability.

3. **Linear-reservoir land storage** [mm], monthly:

       S_t = a * S_{t-1} + P_t * days_per_month,   a = exp(-1 / tau_S)

   A conceptual storage anomaly with e-folding memory tau_S. It is not a
   model of any observed storage product.

The full Markov state at the end of month t is (T, H, S, calendar month).
Predictors are snapshot (end-of-month) values of T, H and S plus two
irrelevant AR(1) "distractor" indices. Given that state, the forecast
distribution of S at any lead is obtained exactly (up to Monte Carlo error)
by re-simulating many stochastic continuations; see ``knownpred.truth``.

All parameter values are illustrative. They are chosen to give an ENSO-like
period of roughly four years and a plausible magnitude of variability, not
fitted to observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12.0

PREDICTORS = ("T", "H", "S", "month_sin", "month_cos", "distractor_1", "distractor_2")


@dataclass(frozen=True)
class SystemParams:
    """Parameters of the toy system. Units are given in the module docstring."""

    # Recharge oscillator
    omega: float = 2.0 * np.pi / 48.0  # rad / month (~4-year period)
    lambda_T0: float = 0.06  # 1 / month, annual-mean damping of T
    lambda_T1: float = 0.05  # 1 / month, amplitude of seasonal damping cycle
    damping_peak_month: int = 4  # month (1-12) of strongest damping (spring barrier)
    lambda_H: float = 0.02  # 1 / month
    sigma_T: float = 0.30  # K / sqrt(month)
    sigma_H: float = 0.15  # K / sqrt(month)
    noise_slope_b: float = 0.5  # 1 / K, multiplicative-noise slope
    noise_floor: float = 0.2  # minimum of g(T)
    # Teleconnection
    beta0: float = 0.8  # mm/day per K, peak cool-season teleconnection strength
    tele_peak_month: int = 1  # month of strongest teleconnection
    precip_noise0: float = 0.4  # mm/day, annual-mean precipitation noise amplitude
    precip_noise_seasonal: float = 0.5  # fractional seasonal modulation of noise
    precip_noise_enso_slope: float = 0.6  # 1 / K, ENSO modulation of cool-season noise
    precip_noise_enso_floor: float = 0.25  # minimum of the ENSO noise factor
    precip_gamma_shape: float = 2.0  # Gamma shape k; skewness = 2 / sqrt(k)
    # Storage
    tau_S: float = 6.0  # months, reservoir e-folding time
    # Distractors (irrelevant predictors)
    distractor_phi: float = 0.9  # AR(1) coefficient
    # Numerics
    substeps: int = 30  # Euler-Maruyama substeps per month

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemState:
    """End-of-month state. Arrays broadcast against each other."""

    T: np.ndarray
    H: np.ndarray
    S: np.ndarray
    month: np.ndarray  # calendar month (1-12) of the month that just ended

    def copy(self) -> SystemState:
        return SystemState(self.T.copy(), self.H.copy(), self.S.copy(), self.month.copy())


@dataclass
class ToySystem:
    """Integrator for the toy system. Vectorised over arbitrary array shapes."""

    params: SystemParams = field(default_factory=SystemParams)

    # ----- seasonal coefficient functions ---------------------------------
    def lambda_T(self, month: np.ndarray) -> np.ndarray:
        p = self.params
        return p.lambda_T0 + p.lambda_T1 * np.cos(2 * np.pi * (month - p.damping_peak_month) / 12)

    def beta(self, month: np.ndarray) -> np.ndarray:
        p = self.params
        return p.beta0 * np.maximum(0.0, np.cos(2 * np.pi * (month - p.tele_peak_month) / 12))

    def precip_noise_amp(self, month: np.ndarray, T_bar: np.ndarray) -> np.ndarray:
        """Noise amplitude of P: seasonal cycle times an ENSO factor that is
        active only in the teleconnection season (larger spread in El Nino)."""
        p = self.params
        seasonal = 1.0 + p.precip_noise_seasonal * np.cos(
            2 * np.pi * (month - p.tele_peak_month) / 12
        )
        active = self.beta(month) / p.beta0 if p.beta0 != 0 else 0.0 * month
        enso = np.maximum(
            1.0 + p.precip_noise_enso_slope * active * T_bar, p.precip_noise_enso_floor
        )
        return p.precip_noise0 * seasonal * enso

    def noise_multiplier(self, T: np.ndarray) -> np.ndarray:
        p = self.params
        return np.maximum(1.0 + p.noise_slope_b * T, p.noise_floor)

    # ----- one month --------------------------------------------------------
    def step_month(
        self, state: SystemState, rng: np.random.Generator
    ) -> tuple[SystemState, np.ndarray]:
        """Advance every element of ``state`` by one calendar month.

        Returns the new end-of-month state and the monthly precipitation
        anomaly P of the month that was simulated.
        """
        p = self.params
        month = state.month % 12 + 1  # month being simulated
        T, H = state.T.copy(), state.H.copy()
        dt = 1.0 / p.substeps
        sqdt = np.sqrt(dt)
        lam = self.lambda_T(month)
        T_acc = np.zeros(np.broadcast(T, H).shape)
        for _ in range(p.substeps):
            dWT = rng.standard_normal(T_acc.shape) * sqdt
            dWH = rng.standard_normal(T_acc.shape) * sqdt
            dT = (-lam * T + p.omega * H) * dt + p.sigma_T * self.noise_multiplier(T) * dWT
            dH = (-p.omega * T - p.lambda_H * H) * dt + p.sigma_H * dWH
            T = T + dT
            H = H + dH
            T_acc += T
        T_bar = T_acc / p.substeps

        k = p.precip_gamma_shape
        theta = 1.0 / np.sqrt(k)  # unit variance
        eps = rng.gamma(k, theta, size=T_acc.shape) - k * theta  # zero mean
        P = self.beta(month) * T_bar + self.precip_noise_amp(month, T_bar) * eps

        a = np.exp(-1.0 / p.tau_S)
        S = a * state.S + P * DAYS_PER_MONTH
        return SystemState(T=T, H=H, S=S, month=np.broadcast_to(month, T.shape).copy()), P

    # ----- long control run -------------------------------------------------
    def run(
        self,
        n_years: int,
        seed: int,
        spinup_years: int = 50,
        start_year: int = 1,
    ) -> pd.DataFrame:
        """Run a single long 'control' simulation and return monthly records.

        Each row holds the end-of-month state for that month, the monthly
        precipitation of that month, and the two distractor indices.
        """
        rng = np.random.default_rng(seed)
        p = self.params
        state = SystemState(T=np.zeros(1), H=np.zeros(1), S=np.zeros(1), month=np.array([12]))
        for _ in range(spinup_years * 12):
            state, _ = self.step_month(state, rng)

        n = n_years * 12
        out = {k: np.empty(n) for k in ("T", "H", "S", "P", "distractor_1", "distractor_2")}
        months = np.empty(n, dtype=int)
        d = rng.standard_normal(2)
        innov = np.sqrt(1.0 - p.distractor_phi**2)
        for i in range(n):
            state, P = self.step_month(state, rng)
            d = p.distractor_phi * d + innov * rng.standard_normal(2)
            out["T"][i], out["H"][i], out["S"][i], out["P"][i] = (
                state.T[0],
                state.H[0],
                state.S[0],
                P[0],
            )
            out["distractor_1"][i], out["distractor_2"][i] = d
            months[i] = state.month[0]

        index = pd.period_range(start=f"{start_year:04d}-01", periods=n, freq="M")
        df = pd.DataFrame(out, index=index)
        df.index.name = "time"
        df["month"] = months
        assert np.array_equal(df["month"].to_numpy(), index.month.to_numpy())
        return df

    # ----- ensembles from given initial states ---------------------------------
    def forecast_ensemble(
        self,
        init: SystemState,
        lead: int,
        n_members: int,
        rng: np.random.Generator,
        batch_size: int = 500,
    ) -> np.ndarray:
        """Monte Carlo forecast of S at ``lead`` months for each initial state.

        Parameters
        ----------
        init : end-of-month states, 1-D arrays of length N.
        lead : forecast lead in months (>= 1).
        n_members : number of stochastic continuations per initial state.

        Returns
        -------
        array of shape (N, n_members) with S(t0 + lead).
        """
        if lead < 1:
            raise ValueError("lead must be >= 1 month")
        N = init.T.shape[0]
        out = np.empty((N, n_members))
        for start in range(0, N, batch_size):
            sl = slice(start, min(start + batch_size, N))
            shape = (sl.stop - sl.start, n_members)
            st = SystemState(
                T=np.broadcast_to(init.T[sl, None], shape).copy(),
                H=np.broadcast_to(init.H[sl, None], shape).copy(),
                S=np.broadcast_to(init.S[sl, None], shape).copy(),
                month=np.broadcast_to(init.month[sl, None], shape).copy(),
            )
            for _ in range(lead):
                st, _ = self.step_month(st, rng)
            out[sl] = st.S
        return out

    def with_params(self, **changes) -> ToySystem:
        return ToySystem(replace(self.params, **changes))


def make_supervised(df: pd.DataFrame, lead: int) -> tuple[pd.DataFrame, pd.Series]:
    """Build (predictors at t0, target S at t0 + lead) pairs from a run."""
    X = pd.DataFrame(
        {
            "T": df["T"],
            "H": df["H"],
            "S": df["S"],
            "month_sin": np.sin(2 * np.pi * df["month"] / 12),
            "month_cos": np.cos(2 * np.pi * df["month"] / 12),
            "distractor_1": df["distractor_1"],
            "distractor_2": df["distractor_2"],
        },
        index=df.index,
    )
    y = df["S"].shift(-lead).rename(f"S_lead{lead}")
    valid = y.notna()
    return X[valid], y[valid]
