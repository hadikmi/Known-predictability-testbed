"""End-to-end check of the Part B pipeline on synthetic stand-in files."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xarray")

from knownpred.data.fake import write_fake_inputs  # noqa: E402
from knownpred.experiments.part_b import evaluate_stage, preprocess  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return {
        "seed": 1,
        "output_dir": str(tmp_path / "out"),
        "paths": {
            "raw": str(tmp_path / "raw"),
            "interim": str(tmp_path / "interim"),
            "processed": str(tmp_path / "proc"),
        },
        "regions": {
            "sst_domain": {"lat": [-30, 30], "lon": [120, 290]},
            "nino34": {"lat": [-5, 5], "lon": [190, 240]},
            "target_box": {"lat": [31, 37], "lon": [245, 257]},
        },
        "cesm2": {"forcing_variant": "cmip6"},
        "observations": {
            "years": [1895, 2000],
            "base_period": [1951, 2000],
            "forced_response": "quadratic",
        },
        "season": {
            "issue_month": 10,
            "target_months": [1, 2, 3],
            "antecedent_months": 3,
            "tendency_months": 3,
            "n_eofs": 4,
        },
        "split": {"fractions": [0.6, 0.2, 0.2]},
        "models": {"climatology": {}, "linear": {}, "boosting": {"max_iter": 50}},
        "evaluation": {
            "bootstrap_block_years": 5,
            "bootstrap_samples": 100,
            "confident_fraction": 0.2,
        },
    }


def test_part_b_pipeline_runs_and_recovers_planted_teleconnection(cfg, tmp_path):
    from pathlib import Path

    write_fake_inputs(
        Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]), n_members=10, years=(1890, 1990)
    )
    mod, obs = preprocess(cfg)
    assert set(mod["split"]) == {"train", "val", "test"}
    assert obs["year"].min() == 1896 and obs["year"].max() == 2000
    # each member's samples are complete years and the members do not overlap across splits
    assert mod.groupby("member_id")["split"].nunique().max() == 1
    # standardisation: training targets have zero mean, unit variance
    tr = mod[mod["split"] == "train"]
    assert abs(tr["target"].mean()) < 1e-8 and tr["target"].std() == pytest.approx(1.0)
    # the planted Oct -> JFM teleconnection is visible in both datasets
    assert np.corrcoef(mod["nino34"], mod["target"])[0, 1] > 0.3
    assert np.corrcoef(obs["nino34"], obs["target"])[0, 1] > 0.3

    res = evaluate_stage(cfg)
    obs_tab = res["observations"]
    assert obs_tab.loc["linear (constant sigma)", "crpss_vs_clim"] > 0.05
    out = Path(cfg["output_dir"])
    for f in (
        "metrics_observations.csv",
        "metrics_cesm2le_test.csv",
        "predictions_observations.csv",
    ):
        assert (out / f).exists()
    pred = pd.read_csv(out / "predictions_observations.csv")
    assert pred["year"].is_monotonic_increasing
