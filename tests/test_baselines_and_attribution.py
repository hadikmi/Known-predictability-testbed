import numpy as np
import pandas as pd
import pytest

from knownpred.attribution import permutation_spread_importance, truth_sigma_fn
from knownpred.baselines import Climatology, SeasonalLinear, TwoStageBoosting, issue_month
from knownpred.evaluation import evaluate
from knownpred.simulate import ToySystem, make_supervised
from knownpred.truth import compute_ground_truth


@pytest.fixture(scope="module")
def data():
    sys = ToySystem()
    tr, va, te = (sys.run(n, seed=s, spinup_years=10) for n, s in ((150, 1), (40, 2), (40, 3)))
    return sys, make_supervised(tr, 3), make_supervised(va, 3), make_supervised(te, 3), te


def test_issue_month_roundtrip():
    m = np.arange(1, 13)
    X = pd.DataFrame(
        {"month_sin": np.sin(2 * np.pi * m / 12), "month_cos": np.cos(2 * np.pi * m / 12)}
    )
    np.testing.assert_array_equal(issue_month(X), m)


def test_baselines_produce_valid_forecasts(data):
    _, (Xtr, ytr), (Xva, yva), (Xte, yte), _ = data
    fcs = [
        Climatology().fit(Xtr, ytr).predict(Xte),
        SeasonalLinear().fit(Xtr, ytr).predict(Xte),
        TwoStageBoosting(max_iter=100).fit(Xtr, ytr, Xva, yva).predict(Xte),
    ]
    clim_crps = fcs[0].crps(yte.to_numpy())
    for fc in fcs:
        assert fc.mean.shape == fc.std.shape == (len(Xte),)
        assert np.all(fc.std > 0)
        pit = fc.pit(yte.to_numpy())
        assert np.all((pit >= 0) & (pit <= 1))
    # state information must beat climatology clearly at 3-month lead
    assert fcs[1].crps(yte.to_numpy()).mean() < 0.8 * clim_crps.mean()


def test_linear_spread_is_constant_within_month(data):
    _, (Xtr, ytr), _, (Xte, _), _ = data
    fc = SeasonalLinear().fit(Xtr, ytr).predict(Xte)
    m = issue_month(Xte)
    assert all(np.ptp(fc.std[m == k]) == 0 for k in range(1, 13))


def test_evaluate_truth_against_itself(data):
    sys, _, _, (Xte, yte), te = data
    gt = compute_ground_truth(sys, te.loc[Xte.index], 3, n_members=300, seed=0)
    from knownpred.forecast import Forecast

    truth = Forecast.from_members("truth", gt.members)
    clim = truth.crps(yte.to_numpy()) * 2  # any positive reference
    res = evaluate(
        truth, yte.to_numpy(), issue_month(Xte), clim, gt, truth.crps(yte.to_numpy()), n_boot=50
    )
    assert res["sigma_spearman_within_month"] == pytest.approx(1.0)
    assert res["excess_crps"] == pytest.approx(0.0)
    assert res["log_sigma_rmse"] < 0.05  # ddof difference only


def test_truth_attribution_assigns_zero_to_storage_and_distractors(data):
    sys, _, _, (Xte, _), _ = data
    X = Xte.iloc[:150]
    imp = permutation_spread_importance(
        truth_sigma_fn(sys, 3, n_members=200, seed=1), X, n_repeats=1
    )
    # spread of S(t+3) does not depend on S(t) (linear reservoir) nor on distractors;
    # common random numbers make this exact up to floating point
    for g in ("S", "distractor_1", "distractor_2"):
        assert imp.loc[g, "importance"] < 1e-8
    assert imp.loc["T", "importance"] > 0.05
