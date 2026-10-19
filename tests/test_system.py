import numpy as np
import pytest

from knownpred.simulate import SystemState, ToySystem, make_supervised
from knownpred.truth import compute_ground_truth


@pytest.fixture(scope="module")
def run():
    return ToySystem().run(n_years=120, seed=11)


def test_run_is_reproducible():
    a = ToySystem().run(n_years=5, seed=3, spinup_years=2)
    b = ToySystem().run(n_years=5, seed=3, spinup_years=2)
    c = ToySystem().run(n_years=5, seed=4, spinup_years=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_enso_like_statistics(run):
    T = run["T"].to_numpy()
    assert 0.6 < T.std() < 1.3
    # positive skewness from multiplicative noise
    assert ((T - T.mean()) ** 3).mean() / T.std() ** 3 > 0.3
    # dominant period between 3 and 6 years
    f = np.fft.rfftfreq(T.size)
    pw = np.abs(np.fft.rfft(T - T.mean())) ** 2
    period_years = 1 / f[1:][np.argmax(pw[1:])] / 12
    assert 3 < period_years < 6


def test_teleconnection_only_in_cool_season(run):
    corr = run.groupby("month").apply(
        lambda g: np.corrcoef(g["T"], g["P"])[0, 1], include_groups=False
    )
    assert corr.loc[[12, 1, 2]].min() > 0.3
    assert corr.loc[[5, 6, 7, 8]].abs().max() < 0.15


def test_storage_is_linear_reservoir(run):
    a = np.exp(-1 / ToySystem().params.tau_S)
    S, P = run["S"].to_numpy(), run["P"].to_numpy()
    np.testing.assert_allclose(S[1:], a * S[:-1] + P[1:] * 365.25 / 12, rtol=1e-10, atol=1e-8)


def test_make_supervised_alignment(run):
    X, y = make_supervised(run, lead=3)
    assert len(X) == len(run) - 3
    np.testing.assert_array_equal(y.to_numpy(), run["S"].to_numpy()[3:])
    np.testing.assert_array_equal(X["S"].to_numpy(), run["S"].to_numpy()[:-3])


def test_forecast_ensemble_starts_from_given_state():
    sys = ToySystem()
    init = SystemState(
        T=np.array([1.0, -1.0]),
        H=np.array([0.0, 0.5]),
        S=np.array([10.0, -20.0]),
        month=np.array([9, 9]),
    )
    rng = np.random.default_rng(0)
    ens = sys.forecast_ensemble(init, lead=1, n_members=4000, rng=rng)
    # storage memory: ensemble mean retains a fraction of the initial anomaly
    a = np.exp(-1 / sys.params.tau_S)
    # October: beta(10) = 0 so P has zero mean and E[S1] = a * S0
    se = ens.std(axis=1) / np.sqrt(4000)
    assert np.all(np.abs(ens.mean(axis=1) - a * init.S) < 4 * se)


def test_ground_truth_spread_is_state_dependent(run):
    sys = ToySystem()
    nov = run[run["month"] == 10].iloc[:80]  # issued end of Oct, lead 3 -> Jan
    gt = compute_ground_truth(sys, nov, lead=3, n_members=400, seed=1)
    assert np.corrcoef(gt.std, nov["T"])[0, 1] > 0.5
    assert gt.std.std() / gt.std.mean() > 3 * (gt.se_std / gt.std).mean()
