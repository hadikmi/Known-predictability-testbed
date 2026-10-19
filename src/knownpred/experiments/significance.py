"""Formal significance tests on the saved per-case scores, with multiple-testing control.

Four families of hypotheses, fixed before looking at the p-values; Holm
(family-wise error) and Benjamini-Hochberg (false discovery rate) adjustments
are applied within each family:

  A-skill   Part A, each non-linear model vs the linear model, CRPS,
            leads 1/3/6. Diebold-Mariano, two-sided. 12 tests.
  B-skill   Part B, each model vs climatology, CRPS, CESM2-LE test members and
            observations. Diebold-Mariano, one-sided (H1: model better). 10 tests.
  B-vs-lin  Part B, each non-linear model vs the linear model, CRPS, both
            datasets. Diebold-Mariano, two-sided. 8 tests.
  B-conf    Part B, does the model's predicted spread order its errors?
            Circular-shift test of the error change in the most-confident
            half, one-sided (H1: smaller errors). 8 tests. The linear model has
            constant spread and is excluded.
  B-smyle   Part B, observed years 1971-2020 with CESM2 SMYLE hindcasts as a
            perfect-model truth: (i) does each model's predicted spread rank
            the years like the SMYLE ensemble spread does (Spearman, circular
            shift, one-sided); (ii) does the SMYLE spread itself order the
            observed errors of the SMYLE ensemble mean; (iii) is the SMYLE
            ensemble forecast better than climatology (Diebold-Mariano).
            Run only when runs/part_b/smyle_truth.csv exists.

Usage:  python -m knownpred.experiments.significance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..stats_tests import (
    benjamini_hochberg,
    circular_shift_test,
    confident_half_error_change,
    diebold_mariano,
    holm,
)

NONLINEAR = ["boosting", "nn_gaussian", "nn_shash", "deep_ensemble"]
ALL_MODELS = ["linear", *NONLINEAR]


def _dm_row(family, dataset, a, b, la, lb, groups=None, one_sided=False, **extra):
    r = diebold_mariano(la, lb, groups=groups)
    return {
        "family": family,
        "dataset": dataset,
        "model": a,
        "reference": b,
        "test": "Diebold-Mariano (HLN, Andrews HAC)",
        "alternative": "model better (one-sided)" if one_sided else "two-sided",
        "effect": r["mean_diff"],
        "effect_definition": "mean CRPS(model) - CRPS(reference)",
        "statistic": r["statistic"],
        "p_value": r["p_a_better"] if one_sided else r["p_two_sided"],
        "n": r["n"],
        "hac_lag": r["lag"],
        "lag1_autocorr": r["lag1_autocorr"],
        **extra,
    }


def part_a_tests(run_dir: Path, leads=(1, 3, 6)) -> list[dict]:
    rows = []
    for lead in leads:
        f = run_dir / f"predictions_lead{lead}.csv.gz"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        if "linear_crps" not in p:
            raise RuntimeError(f"{f} has no per-case CRPS; rerun Part A with the current code")
        for k in NONLINEAR:
            if f"{k}_crps" in p:
                rows.append(
                    _dm_row(
                        "A-skill",
                        f"toy system, lead {lead}",
                        k,
                        "linear",
                        p[f"{k}_crps"],
                        p["linear_crps"],
                    )
                )
    return rows


def _part_b_frames(run_dir: Path):
    out = {}
    f = run_dir / "predictions_cesm2le_test.csv"
    if f.exists():
        p = pd.read_csv(f).sort_values(["member_id", "year"]).reset_index(drop=True)
        out["CESM2-LE test members"] = (p, p["member_id"].to_numpy(), "climatology")
    f = run_dir / "predictions_observations.csv"
    if f.exists():
        p = pd.read_csv(f).sort_values("year").reset_index(drop=True)
        out["observations"] = (p, None, "climatology_loo")
    return out


def part_b_tests(run_dir: Path, min_shift: int = 5) -> list[dict]:
    rows = []
    for dataset, (p, groups, ref) in _part_b_frames(run_dir).items():
        if f"{ref}_crps" not in p:
            raise RuntimeError("Part B predictions have no per-case CRPS; rerun the evaluate stage")
        for k in ALL_MODELS:
            rows.append(
                _dm_row(
                    "B-skill",
                    dataset,
                    k,
                    ref,
                    p[f"{k}_crps"],
                    p[f"{ref}_crps"],
                    groups=groups,
                    one_sided=True,
                )
            )
        for k in NONLINEAR:
            rows.append(
                _dm_row(
                    "B-vs-lin",
                    dataset,
                    k,
                    "linear",
                    p[f"{k}_crps"],
                    p["linear_crps"],
                    groups=groups,
                )
            )
        for k in NONLINEAR:
            e = np.abs(p["y"].to_numpy() - p[f"{k}_mean"].to_numpy())
            r = circular_shift_test(
                confident_half_error_change,
                p[f"{k}_std"].to_numpy(),
                e,
                groups=groups,
                min_shift=min_shift,
            )
            rows.append(
                {
                    "family": "B-conf",
                    "dataset": dataset,
                    "model": k,
                    "reference": "shifted spread",
                    "test": "circular-shift randomisation",
                    "alternative": "smaller errors (one-sided)",
                    "effect": r["statistic"],
                    "effect_definition": "relative change in mean |error|, most-confident half vs all",
                    "statistic": (r["statistic"] - r["null_mean"]) / r["null_sd"],
                    "p_value": r["p_value"],
                    "n": len(p),
                    "n_shifts": r["n_shifts"],
                }
            )
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats

    return float(stats.spearmanr(a, b).statistic)


def smyle_tests(run_dir: Path, min_shift: int = 5) -> list[dict]:
    f_truth = run_dir / "smyle_truth.csv"
    f_obs = run_dir / "predictions_observations.csv"
    if not (f_truth.exists() and f_obs.exists()):
        return []
    from ..metrics import crps_ensemble

    gt = pd.read_csv(f_truth).set_index("target_year")
    p = pd.read_csv(f_obs).set_index("year")
    years = sorted(set(gt.index) & set(p.index))
    gt, p = gt.loc[years], p.loc[years]
    s_true = gt["smyle_std"].to_numpy()
    rows = []
    for k in NONLINEAR:
        r = circular_shift_test(
            _spearman, p[f"{k}_std"].to_numpy(), s_true, min_shift=min_shift, alternative="greater"
        )
        rows.append(
            {
                "family": "B-smyle",
                "dataset": f"observed years {years[0]}-{years[-1]}",
                "model": k,
                "reference": "SMYLE ensemble spread",
                "test": "circular-shift randomisation",
                "alternative": "positive rank correlation (one-sided)",
                "effect": r["statistic"],
                "effect_definition": "Spearman correlation, predicted spread vs SMYLE spread",
                "statistic": (r["statistic"] - r["null_mean"]) / r["null_sd"],
                "p_value": r["p_value"],
                "n": len(years),
                "n_shifts": r["n_shifts"],
            }
        )
    members = gt.filter(regex=r"^m\d+$").to_numpy()
    y = p["y"].to_numpy()
    err = np.abs(y - gt["smyle_mean"].to_numpy())
    r = circular_shift_test(confident_half_error_change, s_true, err, min_shift=min_shift)
    rows.append(
        {
            "family": "B-smyle",
            "dataset": f"observed years {years[0]}-{years[-1]}",
            "model": "SMYLE",
            "reference": "shifted spread",
            "test": "circular-shift randomisation",
            "alternative": "smaller errors (one-sided)",
            "effect": r["statistic"],
            "effect_definition": "relative change in mean |error|, most-confident half vs all",
            "statistic": (r["statistic"] - r["null_mean"]) / r["null_sd"],
            "p_value": r["p_value"],
            "n": len(years),
            "n_shifts": r["n_shifts"],
        }
    )
    crps_smyle = crps_ensemble(y, members, fair=True)
    rows.append(
        _dm_row(
            "B-smyle",
            f"observed years {years[0]}-{years[-1]}",
            "SMYLE",
            "climatology_loo",
            crps_smyle,
            p["climatology_loo_crps"].to_numpy(),
            one_sided=True,
        )
    )
    return rows


def adjust(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    df["p_bh"] = np.nan
    for _fam, idx in df.groupby("family").groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p_value"].to_numpy())
        df.loc[idx, "p_bh"] = benjamini_hochberg(df.loc[idx, "p_value"].to_numpy())
    df["significant_holm_5pct"] = df["p_holm"] < 0.05
    df["significant_bh_5pct"] = df["p_bh"] < 0.05
    return df


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--part-a", default="runs/part_a")
    ap.add_argument("--part-b", default="runs/part_b")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args(argv)
    rows = (
        part_a_tests(Path(args.part_a))
        + part_b_tests(Path(args.part_b))
        + smyle_tests(Path(args.part_b))
    )
    df = adjust(pd.DataFrame(rows))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "significance.csv", index=False)
    show = df[
        ["family", "dataset", "model", "reference", "effect", "p_value", "p_holm", "p_bh"]
    ].copy()
    show["effect"] = show["effect"].map(lambda v: f"{v:+.4f}")
    for c in ("p_value", "p_holm", "p_bh"):
        show[c] = show[c].map(lambda v: f"{v:.3g}")
    (out / "significance.md").write_text(show.to_markdown(index=False))
    print(show.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
