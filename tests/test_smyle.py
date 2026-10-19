"""SMYLE ground-truth extraction on a synthetic file with the real layout."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xarray")
pytest.importorskip("cftime")

from knownpred.data import smyle  # noqa: E402


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("smyle")
    smyle.write_fake_smyle(d / smyle.SMYLE_FILE.format(var="PRECT"))
    return d


def test_target_leads_are_jan_to_mar_after_november_start(fake):
    ds = smyle.open_smyle("PRECT", fake)
    np.testing.assert_array_equal(smyle.target_leads(ds, [1, 2, 3]), [2, 3, 4])


def test_ground_truth_recovers_planted_spread(fake):
    cfg = {
        "regions": {"target_box": {"lat": [31, 37], "lon": [245, 257]}},
        "season": {"target_months": [1, 2, 3]},
        "observations": {"forced_response": "quadratic"},
    }
    gt = smyle.build(cfg, "PRECT", local_dir=fake)
    assert list(gt["target_year"])[:2] == [1971, 1972]
    assert (gt["members"] == 20).all()
    # standardised: all-members variance ~1
    m = gt.filter(regex=r"^m\d+$").to_numpy()
    assert np.nanstd(m) == pytest.approx(1.0, rel=1e-6)
    planted = pd.read_csv(fake / "CESM2-SMYLE_NOV.PRECT.1970-2020.enso.csv", index_col=0).iloc[:, 0]
    rho = np.corrcoef(planted.loc[gt["target_year"]].to_numpy(), np.log(gt["smyle_std"]))[0, 1]
    assert rho > 0.7
    assert (gt["smyle_std_se"] / gt["smyle_std"]).round(3).eq(round(1 / np.sqrt(38), 3)).all()


def test_smyle_significance_family(fake, tmp_path):
    from knownpred.experiments.significance import smyle_tests

    cfg = {
        "regions": {"target_box": {"lat": [31, 37], "lon": [245, 257]}},
        "season": {"target_months": [1, 2, 3]},
        "observations": {"forced_response": "quadratic"},
    }
    gt = smyle.build(cfg, "PRECT", local_dir=fake)
    gt.to_csv(tmp_path / "smyle_truth.csv", index=False)
    rng = np.random.default_rng(5)
    planted = pd.read_csv(fake / "CESM2-SMYLE_NOV.PRECT.1970-2020.enso.csv", index_col=0).iloc[:, 0]
    years = gt["target_year"].to_numpy()
    pred = pd.DataFrame({"year": years, "y": gt["m1"].to_numpy(), "climatology_loo_crps": 0.6})
    for k in ("boosting", "nn_gaussian", "nn_shash", "deep_ensemble"):
        pred[f"{k}_std"] = np.exp(
            0.3 * planted.loc[years].to_numpy() + 0.1 * rng.standard_normal(len(years))
        )
        pred[f"{k}_mean"] = 0.0
    pred["boosting_std"] = rng.permutation(
        pred["boosting_std"].to_numpy()
    )  # an uninformative model
    pred.to_csv(tmp_path / "predictions_observations.csv", index=False)
    rows = pd.DataFrame(smyle_tests(tmp_path))
    assert set(rows["model"]) == {"boosting", "nn_gaussian", "nn_shash", "deep_ensemble", "SMYLE"}
    rho = rows[rows["effect_definition"].str.startswith("Spearman")].set_index("model")
    assert rho.loc["nn_shash", "effect"] > 0.5 and rho.loc["nn_shash", "p_value"] < 0.05
    assert abs(rho.loc["boosting", "effect"]) < 0.4
