"""Part B: the Part A pipeline applied to real data.

Question carried over from Part A (Q4): a network trained in a climate
model's world reports its own confidence. Does that confidence stay honest
(i) on unseen members of the same model and (ii) on observations?

Stages (each can be run separately):
    preprocess   CESM2-LE + ERSSTv5 + nClimGrid -> sample tables
    evaluate     fit models on CESM2-LE train members, score on CESM2-LE test
                 members and on observations

Usage:
    python -m knownpred.data.download --config configs/part_b.yaml
    python -m knownpred.experiments.part_b --config configs/part_b.yaml --stage all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines import leave_one_out_climatology
from ..evaluation import discard_table, evaluate
from .common import (
    derive_seed,
    load_config,
    log,
    provenance,
    setup_logging,
    torch_available,
    write_json,
)
from .part_a import fit_all

# ----------------------------------------------------------------------------- preprocess


def preprocess(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    import xarray as xr

    from ..data import preprocess as pp
    from ..data.dataset import build_samples

    P = {k: Path(v) for k, v in cfg["paths"].items()}
    R, S, Obs = cfg["regions"], cfg["season"], cfg["observations"]
    c = cfg["cesm2"]
    P["processed"].mkdir(parents=True, exist_ok=True)

    # --- CESM2-LE (already relabelled to month-start time in the extract step)
    ts = pp.temperature_to_degC(
        xr.open_dataarray(P["interim"] / f"cesm2le_{c['forcing_variant']}_ts.nc")
    )
    pr = pp.precip_to_mm_per_day(
        xr.open_dataarray(P["interim"] / f"cesm2le_{c['forcing_variant']}_precip.nc")
    )
    complete = pr.notnull().all([d for d in pr.dims if d != "member_id"]).values
    ts, pr = (
        ts.isel(member_id=complete),
        pr.isel(member_id=complete),
    )  # members covering the full period
    log.info(
        "CESM2-LE: %d members, %s to %s",
        ts.sizes["member_id"],
        str(ts.time.values[0])[:7],
        str(ts.time.values[-1])[:7],
    )

    # --- ERSSTv5 on the CESM grid, common land/ice mask
    ersst = xr.open_dataset(P["raw"] / "ersstv5_sst.mnmean.nc")["sst"]
    ersst = pp.to_month_start(pp.standardize_coords(ersst))
    ersst = pp.temperature_to_degC(ersst).sel(
        time=slice(f"{Obs['years'][0]}-01-01", f"{Obs['years'][1]}-12-31")
    )
    ersst_dom = pp.select_box(ersst, R["sst_domain"])
    ersst_rg = pp.regrid_bilinear(ersst_dom, ts)
    ocean = ersst_rg.notnull().all("time") & ts.notnull().all(("time", "member_id"))
    ts, ersst_rg = ts.where(ocean), ersst_rg.where(ocean)
    log.info("common ocean mask: %d of %d grid cells", int(ocean.sum()), ocean.size)

    # --- nClimGrid regional mean (large file: lazy, regional subset first)
    import importlib.util

    chunks = (
        {"time": 120} if importlib.util.find_spec("dask") else None
    )  # lazy read when dask exists
    ncg = xr.open_dataset(P["raw"] / "nclimgrid_prcp.nc", chunks=chunks)["prcp"]
    ncg = pp.standardize_coords(ncg)
    ncg = pp.select_box(ncg, R["target_box"])
    ncg = pp.to_month_start(ncg).sel(
        time=slice(f"{Obs['years'][0]}-01-01", f"{Obs['years'][1]}-12-31")
    )
    pr_obs = pp.region_mean(pp.precip_to_mm_per_day(ncg)).compute()

    # --- anomalies and forced-response removal
    # Model: ensemble mean = forced response (incl. climatology) -> internal variability.
    ts_anom = pp.remove_ensemble_mean(ts)
    pr_mod = pp.remove_ensemble_mean(pp.region_mean(pr))
    # Observations: base-period climatology, then a polynomial trend (one realisation only).
    deg = {"quadratic": 2, "linear": 1}[Obs["forced_response"]]
    sst_obs_anom = pp.remove_polynomial_trend(
        pp.monthly_anomalies(ersst_rg, tuple(Obs["base_period"])), deg
    )
    pr_obs_anom = pp.remove_polynomial_trend(
        pp.monthly_anomalies(pr_obs, tuple(Obs["base_period"])), deg
    )
    nino_obs = pp.remove_polynomial_trend(
        pp.monthly_anomalies(pp.region_mean(ersst, R["nino34"]), tuple(Obs["base_period"])), deg
    )
    nino_mod = pp.region_mean(ts_anom, R["nino34"])

    # --- EOFs of issue-month SST anomalies, fitted on training members only
    from ..data.dataset import split_members

    splits = split_members(
        ts_anom["member_id"].values,
        cfg["split"]["fractions"],
        seed=derive_seed(cfg["seed"], "members"),
    )
    write_json(
        {k: list(map(str, v)) for k, v in splits.items()}, P["processed"] / "member_split.json"
    )
    issue = ts_anom.sel(time=ts_anom["time"].dt.month == S["issue_month"])
    eof = pp.EOF(S["n_eofs"]).fit(issue.sel(member_id=splits["train"]), ("time", "member_id"))
    log.info("EOF explained variance: %s", np.round(eof.explained_variance_ratio_, 3))
    np.savez(
        P["processed"] / "eof.npz",
        patterns=eof.patterns(),
        evr=eof.explained_variance_ratio_,
        lat=ts["lat"].values,
        lon=ts["lon"].values,
    )

    def pcs_of(field, dims):
        values = eof.transform(field, dims)
        coords = {d: field[d] for d in dims}
        shape = [field.sizes[d] for d in dims]
        return xr.DataArray(
            values.reshape(*shape, -1), coords=coords, dims=(*dims, "mode")
        ).assign_coords(mode=np.arange(1, S["n_eofs"] + 1))

    # PCs are only needed in the issue month: project those fields only.
    pcs_mod = pcs_of(issue, ("time", "member_id"))
    pcs_obs = pcs_of(
        sst_obs_anom.sel(time=sst_obs_anom["time"].dt.month == S["issue_month"]), ("time",)
    )

    kw = dict(
        issue_month=S["issue_month"],
        target_months=S["target_months"],
        antecedent_months=S["antecedent_months"],
        tendency_months=S["tendency_months"],
    )
    mod = build_samples(pcs_mod, nino_mod, pr_mod, **kw).reset_index()
    obs = build_samples(pcs_obs, nino_obs, pr_obs_anom, **kw).reset_index()
    mod["split"] = np.select(
        [mod["member_id"].isin(splits[k]) for k in splits], list(splits), "none"
    )

    # --- standardisation: each dataset by its own reference statistics
    cols = [c for c in mod.columns if c not in ("member_id", "year", "split")]
    ref_m = mod[mod["split"] == "train"][cols]
    base = obs[(obs["year"] >= Obs["base_period"][0]) & (obs["year"] <= Obs["base_period"][1])][
        cols
    ]
    mod[cols] = (mod[cols] - ref_m.mean()) / ref_m.std()
    obs[cols] = (obs[cols] - base.mean()) / base.std()
    mod.to_csv(P["processed"] / "samples_cesm2le.csv", index=False)
    obs.to_csv(P["processed"] / "samples_obs.csv", index=False)
    log.info(
        "samples: CESM2-LE %d (train %d), observations %d (%d-%d)",
        len(mod),
        (mod["split"] == "train").sum(),
        len(obs),
        obs["year"].min(),
        obs["year"].max(),
    )
    return mod, obs


# ----------------------------------------------------------------------------- evaluate


def evaluate_stage(cfg: dict) -> dict[str, pd.DataFrame]:
    P = Path(cfg["paths"]["processed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    mod = pd.read_csv(P / "samples_cesm2le.csv")
    obs = pd.read_csv(P / "samples_obs.csv").sort_values("year")
    feats = [c for c in mod.columns if c not in ("member_id", "year", "split", "target")]
    tr, va, te = (mod[mod["split"] == s] for s in ("train", "val", "test"))
    fitted = fit_all(cfg, tr[feats], tr["target"], va[feats], va["target"], tag="partb")
    ev = cfg["evaluation"]

    results = {}
    for name, df, ref_fn in (
        ("cesm2le_test", te.sort_values(["member_id", "year"]), None),
        ("observations", obs, leave_one_out_climatology),
    ):
        y = df["target"].to_numpy()
        fcs = {k: m.predict(df[feats]) for k, m in fitted.items()}
        if ref_fn is not None:  # real-world reference: cross-validated observed climatology
            fcs = {"climatology_loo": ref_fn(y), **fcs}
        ref_key = "climatology_loo" if ref_fn is not None else "climatology"
        crps_ref = fcs[ref_key].crps(y)
        rows = [
            evaluate(
                fc,
                y,
                np.zeros(len(y)),
                crps_ref,
                block=ev["bootstrap_block_years"],
                n_boot=ev["bootstrap_samples"],
                seed=derive_seed(cfg["seed"], "boot", name),
            )
            for fc in fcs.values()
        ]
        table = pd.DataFrame(rows).set_index("model")
        table.to_csv(out / f"metrics_{name}.csv")
        discard_table(list(fcs.values()), y).to_csv(out / f"discard_{name}.csv")
        pred = df[["year"] + (["member_id"] if "member_id" in df else [])].copy()
        pred["y"] = y
        for k, fc in fcs.items():
            pred[f"{k}_mean"], pred[f"{k}_std"], pred[f"{k}_pit"] = fc.mean, fc.std, fc.pit(y)
        pred.to_csv(out / f"predictions_{name}.csv", index=False)
        log.info(
            "%s:\n%s",
            name,
            table[
                [
                    "crpss_vs_clim",
                    "crpss_lo",
                    "crpss_hi",
                    "pit_d",
                    "coverage90",
                    "spread_error_ratio",
                    "ser_lo",
                    "ser_hi",
                ]
            ].round(3),
        )
        results[name] = table
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/part_b.yaml")
    ap.add_argument("--stage", choices=["preprocess", "evaluate", "all"], default="all")
    ap.add_argument(
        "--fake-data",
        action="store_true",
        help="write small synthetic stand-in input files first (pipeline check, no download)",
    )
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    meta = provenance(cfg)
    meta["torch_models_run"] = torch_available()
    write_json({"provenance": meta, "config": cfg}, out / f"run_info_{args.stage}.json")
    if args.fake_data:
        from ..data.fake import write_fake_inputs

        log.warning("using SYNTHETIC stand-in inputs: results say nothing about the real climate")
        cfg["output_dir"] = cfg["output_dir"] + "_fake"
        for k in cfg["paths"]:
            cfg["paths"][k] = str(
                Path(cfg["paths"][k]).parent / "fake" / Path(cfg["paths"][k]).name
            )
        cfg["observations"]["years"] = [1895, 2000]
        write_fake_inputs(Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]))
    if args.stage in ("preprocess", "all"):
        preprocess(cfg)
    if args.stage in ("evaluate", "all"):
        evaluate_stage(cfg)


if __name__ == "__main__":
    main()
