"""Part A experiments: can uncertainty-aware models recover the true
state-dependent predictability of a system where that predictability is known?

Questions (see README):
  Q1  Does the predicted spread track the true spread across states and leads?
  Q2  Are the model's most-confident cases the truly most-predictable ones?
  Q3  Does attribution of the predicted spread point to the true sources?
  Q4  Does confidence stay honest when the system changes after training?
  Q5  How many years of training data are needed for useful spread estimates?

Usage:
    python -m knownpred.experiments.part_a --config configs/part_a.yaml [--quick]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..attribution import permutation_spread_importance, truth_sigma_fn
from ..baselines import Climatology, SeasonalLinear, TwoStageBoosting, issue_month
from ..evaluation import discard_table, evaluate
from ..forecast import Forecast
from ..simulate import ToySystem, make_supervised
from ..truth import compute_ground_truth
from .common import (
    derive_seed,
    load_config,
    log,
    provenance,
    setup_logging,
    torch_available,
    write_json,
)

TORCH_MODELS = ("nn_gaussian", "nn_shash", "deep_ensemble")


# --------------------------------------------------------------------------- models


def make_model(key: str, params: dict, seed: int):
    """Instantiate a model by its config key. Returns None if unavailable."""
    params = dict(params or {})
    if key == "climatology":
        return Climatology()
    if key == "linear":
        return SeasonalLinear()
    if key == "boosting":
        return TwoStageBoosting(seed=seed, **params)
    if key in TORCH_MODELS:
        if not torch_available():
            return None
        from ..models.train import DeepEnsemble, NetForecaster, TrainConfig

        if key == "deep_ensemble":
            n_members = params.pop("n_members", 5)
            return DeepEnsemble(TrainConfig(seed=seed, **params), n_members=n_members)
        return NetForecaster(TrainConfig(seed=seed, **params))
    raise KeyError(key)


def fit_model(model, X, y, X_val, y_val):
    if isinstance(model, (Climatology, SeasonalLinear)):
        return model.fit(X, y)
    return model.fit(X, y, X_val, y_val)


def fit_all(cfg: dict, X, y, X_val, y_val, tag: str) -> dict:
    fitted = {}
    for key, params in cfg["models"].items():
        model = make_model(key, params, derive_seed(cfg["seed"], "model", key, tag))
        if model is None:
            log.warning("skipping %s: PyTorch is not installed (pip install knownpred[ml])", key)
            continue
        t0 = time.time()
        fitted[key] = fit_model(model, X, y, X_val, y_val)
        log.info("  fitted %-14s in %6.1f s", key, time.time() - t0)
    return fitted


# --------------------------------------------------------------------------- data


def simulate_splits(cfg: dict, system: ToySystem) -> dict[str, pd.DataFrame]:
    d = cfg["data"]
    out = {}
    for split in ("train", "val", "test"):
        seed = derive_seed(cfg["seed"], "run", split)
        out[split] = system.run(d[f"{split}_years"], seed=seed, spinup_years=d["spinup_years"])
        log.info("simulated %s: %d years (seed %d)", split, d[f"{split}_years"], seed)
    return out


def truth_for(cfg, system, df_test, X_test, lead, tag):
    t0 = time.time()
    gt = compute_ground_truth(
        system,
        df_test.loc[X_test.index],
        lead,
        cfg["truth"]["n_members"],
        seed=derive_seed(cfg["seed"], "truth", tag, lead),
    )
    log.info(
        "  ground truth (%d states x %d members) in %.1f s",
        len(X_test),
        cfg["truth"]["n_members"],
        time.time() - t0,
    )
    return gt


# --------------------------------------------------------------------------- Q1, Q2


def score_all(cfg, forecasts: dict[str, Forecast], y, months, gt, seed_tag) -> pd.DataFrame:
    ev = cfg["evaluation"]
    truth_fc = Forecast.from_members("truth", gt.members)
    crps_ref = forecasts["climatology"].crps(y)
    crps_truth = truth_fc.crps(y)
    rows = []
    for fc in [*forecasts.values(), truth_fc]:
        rows.append(
            evaluate(
                fc,
                y,
                months,
                crps_ref,
                gt,
                crps_truth,
                confident_fraction=ev["confident_fraction"],
                block=ev["bootstrap_block_months"],
                n_boot=ev["bootstrap_samples"],
                seed=derive_seed(cfg["seed"], "boot", seed_tag),
            )
        )
    return pd.DataFrame(rows).set_index("model")


def run_main(cfg: dict, out: Path) -> dict:
    system = ToySystem().with_params(**cfg.get("system", {}))
    runs = simulate_splits(cfg, system)
    keep = {}
    for lead in cfg["leads"]:
        log.info("lead %d months", lead)
        Xtr, ytr = make_supervised(runs["train"], lead)
        Xva, yva = make_supervised(runs["val"], lead)
        Xte, yte = make_supervised(runs["test"], lead)
        gt = truth_for(cfg, system, runs["test"], Xte, lead, "base")
        fitted = fit_all(cfg, Xtr, ytr, Xva, yva, tag=f"lead{lead}")
        forecasts = {k: m.predict(Xte) for k, m in fitted.items()}
        y, months = yte.to_numpy(), issue_month(Xte)

        scores = score_all(cfg, forecasts, y, months, gt, f"lead{lead}")
        scores.insert(0, "lead", lead)
        scores.to_csv(out / f"metrics_lead{lead}.csv")
        discard_table(list(forecasts.values()), y, gt).to_csv(out / f"discard_lead{lead}.csv")

        pred = pd.DataFrame(
            {
                "month": months,
                "y": y,
                "T": Xte["T"].to_numpy(),
                "H": Xte["H"].to_numpy(),
                "S": Xte["S"].to_numpy(),
                "true_mean": gt.mean,
                "true_std": gt.std,
                "true_se_std": gt.se_std,
            },
            index=Xte.index,
        )
        for k, fc in forecasts.items():
            pred[f"{k}_mean"], pred[f"{k}_std"] = fc.mean, fc.std
            pred[f"{k}_pit"] = fc.pit(y)
        pred["truth_pit"] = Forecast.from_members("truth", gt.members).pit(y)
        pred.to_csv(out / f"predictions_lead{lead}.csv.gz")
        log.info(
            "\n%s",
            scores[
                [
                    "crpss_vs_clim",
                    "excess_crps",
                    "sigma_spearman_within_month",
                    "overlap_within_month",
                    "log_sigma_rmse",
                    "pit_d",
                    "spread_error_ratio",
                ]
            ].round(3),
        )
        keep[lead] = {"fitted": fitted, "X_test": Xte, "gt": gt, "runs": runs, "system": system}
    return keep


# --------------------------------------------------------------------------- Q3


def run_attribution(cfg: dict, keep: dict, out: Path) -> pd.DataFrame:
    a = cfg["attribution"]
    k = keep[a["lead"]]
    X = k["X_test"]
    rng = np.random.default_rng(derive_seed(cfg["seed"], "attribution", "subsample"))
    Xs = X.iloc[np.sort(rng.choice(len(X), size=min(a["n_states"], len(X)), replace=False))]
    seed = derive_seed(cfg["seed"], "attribution", "perm")
    tables = {
        "truth": permutation_spread_importance(
            truth_sigma_fn(k["system"], a["lead"], a["truth_members"], seed=seed),
            Xs,
            n_repeats=a["n_repeats"],
            seed=seed,
        )
    }
    for key, model in k["fitted"].items():
        if key == "climatology":
            continue
        tables[key] = permutation_spread_importance(
            lambda Z, m=model: m.predict(Z).std, Xs, n_repeats=a["n_repeats"], seed=seed
        )
    df = pd.concat(tables, names=["model"]).reset_index()
    df.to_csv(out / "attribution.csv", index=False)
    log.info(
        "spread attribution (lead %d):\n%s",
        a["lead"],
        df.pivot(index="group", columns="model", values="importance").round(3),
    )
    return df


# --------------------------------------------------------------------------- Q4


def run_shift(cfg: dict, keep: dict, out: Path) -> pd.DataFrame:
    lead = cfg["shift_lead"]
    k = keep[lead]
    rows = []
    scenarios = {"no shift (reference)": {}, **cfg["shift_scenarios"]}
    for name, changes in scenarios.items():
        log.info("shift scenario: %s %s", name, changes)
        system = k["system"].with_params(**changes)
        if changes:
            run = system.run(
                cfg["data"]["test_years"],
                seed=derive_seed(cfg["seed"], "shift", name),
                spinup_years=cfg["data"]["spinup_years"],
            )
            X, yv = make_supervised(run, lead)
            gt = truth_for(cfg, system, run, X, lead, f"shift-{name}")
        else:
            X, gt = k["X_test"], k["gt"]
            yv = make_supervised(k["runs"]["test"], lead)[1]
        forecasts = {key: m.predict(X) for key, m in k["fitted"].items()}
        s = score_all(cfg, forecasts, yv.to_numpy(), issue_month(X), gt, f"shift-{name}")
        s.insert(0, "scenario", name)
        rows.append(s.reset_index())
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "shift.csv", index=False)
    return df


# --------------------------------------------------------------------------- Q5


def run_training_length(cfg: dict, keep: dict, out: Path) -> pd.DataFrame:
    tl = cfg["training_length"]
    lead = tl["lead"]
    k = keep[lead]
    system, X_te, gt = k["system"], k["X_test"], k["gt"]
    y_te = make_supervised(k["runs"]["test"], lead)[1].to_numpy()
    months = issue_month(X_te)
    rows = []
    for years in tl["years"]:
        for rep in range(tl["repeats"]):
            tag = f"len{years}-rep{rep}"
            log.info("training length %d years, repeat %d", years, rep)
            tr = system.run(
                years,
                seed=derive_seed(cfg["seed"], "tl-train", years, rep),
                spinup_years=cfg["data"]["spinup_years"],
            )
            va = system.run(
                max(15, years // 4),
                seed=derive_seed(cfg["seed"], "tl-val", years, rep),
                spinup_years=cfg["data"]["spinup_years"],
            )
            Xtr, ytr = make_supervised(tr, lead)
            Xva, yva = make_supervised(va, lead)
            fitted = fit_all(cfg, Xtr, ytr, Xva, yva, tag=tag)
            forecasts = {key: m.predict(X_te) for key, m in fitted.items()}
            s = score_all(cfg, forecasts, y_te, months, gt, tag).reset_index()
            s.insert(0, "repeat", rep)
            s.insert(0, "train_years", years)
            rows.append(s)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "training_length.csv", index=False)
    return df


# --------------------------------------------------------------------------- driver


def quick_config(cfg: dict) -> dict:
    """Small settings for a smoke test (minutes on a laptop CPU)."""
    cfg = {**cfg}
    cfg["leads"] = [3]
    cfg["data"] = {**cfg["data"], "train_years": 200, "val_years": 50, "test_years": 100}
    cfg["truth"] = {"n_members": 300}
    cfg["attribution"] = {
        **cfg["attribution"],
        "lead": 3,
        "n_states": 300,
        "truth_members": 200,
        "n_repeats": 1,
    }
    cfg["training_length"] = {"lead": 3, "years": [70, 200], "repeats": 1}
    cfg["evaluation"] = {**cfg["evaluation"], "bootstrap_samples": 200}
    cfg["shift_lead"] = 3
    for key in TORCH_MODELS:
        if key in cfg["models"]:
            cfg["models"][key] = {**cfg["models"][key], "max_epochs": 60, "patience": 10}
    cfg["output_dir"] = cfg["output_dir"] + "_quick"
    return cfg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/part_a.yaml")
    ap.add_argument("--quick", action="store_true", help="small smoke-test settings")
    ap.add_argument(
        "--skip", nargs="*", default=[], choices=["attribution", "shift", "training_length"]
    )
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    if args.quick:
        cfg = quick_config(cfg)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    meta = provenance(cfg)
    meta["torch_models_run"] = torch_available()
    write_json({"provenance": meta, "config": cfg}, out / "run_info.json")

    t0 = time.time()
    keep = run_main(cfg, out)
    if "attribution" not in args.skip:
        run_attribution(cfg, keep, out)
    if "shift" not in args.skip:
        run_shift(cfg, keep, out)
    if "training_length" not in args.skip:
        run_training_length(cfg, keep, out)
    log.info("done in %.1f min; results in %s", (time.time() - t0) / 60, out)


if __name__ == "__main__":
    logging.captureWarnings(True)
    main()
