"""Figures for the README and docs, built from the saved run outputs.

Usage:  python -m knownpred.plotting [--part-a runs/part_a] [--part-b runs/part_b]

Colour follows the model, never its rank: every model has a fixed colour in
``MODEL_STYLE``. The categorical colours were checked for colour-vision-
deficiency separation; three of them have low contrast on white, so every
figure carries a legend or direct labels, and every number shown is also in
a CSV table under results/tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

# key in config -> (display name in tables, colour, line style)
MODEL_STYLE = {
    "truth": ("truth", INK, "--"),
    "climatology": ("climatology", MUTED, ":"),
    "linear": ("linear (seasonal sigma)", "#2a78d6", "-"),
    "boosting": ("gradient boosting (two-stage)", "#eb6834", "-"),
    "nn_gaussian": ("NN (gaussian)", "#1baf7a", "-"),
    "nn_shash": ("NN (shash)", "#eda100", "-"),
    "deep_ensemble": ("deep ensemble (5x gaussian)", "#e87ba4", "-"),
}
NAME_TO_KEY = {v[0]: k for k, v in MODEL_STYLE.items()}
SHORT = {
    "truth": "truth",
    "climatology": "climatology",
    "linear": "linear",
    "boosting": "boosting",
    "nn_gaussian": "NN Gaussian",
    "nn_shash": "NN SHASH",
    "deep_ensemble": "deep ensemble",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def _key(name: str) -> str:
    return NAME_TO_KEY.get(name, name)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _models_in(df_index) -> list[str]:
    keys = [_key(n) for n in df_index]
    return [k for k in MODEL_STYLE if k in keys]


# ----------------------------------------------------------------------------- Part A


def fig_system(run_dir: Path, out: Path):
    from .simulate import ToySystem

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    df = ToySystem().run(30, seed=7, spinup_years=20)
    t = np.arange(len(df)) / 12
    ax = axes[0]
    ax.plot(t, df["T"], color="#2a78d6", lw=1.4, label="T")
    ax.set_xlabel("years")
    ax.set_ylabel("T (K)")
    ax.set_title("a  Simulated ENSO-like index (30 years)", loc="left")
    pred = pd.read_csv(run_dir / "predictions_lead3.csv.gz")
    ax = axes[1]
    for lead, col in zip((1, 3, 6), ("#2a78d6", "#eb6834", "#1baf7a"), strict=True):
        p = pd.read_csv(run_dir / f"predictions_lead{lead}.csv.gz")
        g = p.groupby("month")["true_std"]
        ax.plot(
            g.mean().index, g.mean().values, color=col, marker="o", ms=4, label=f"lead {lead} mo"
        )
        ax.fill_between(
            g.mean().index, g.quantile(0.1), g.quantile(0.9), color=col, alpha=0.12, lw=0
        )
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_xlabel("issue month")
    ax.set_ylabel("true forecast std of S (mm)")
    ax.set_title("b  True spread: mean and 10-90 % range", loc="left")
    ax.legend(loc="upper left")
    ax = axes[2]
    oct_ = pred[pred["month"] == 10]
    ax.scatter(oct_["T"], oct_["true_std"], s=8, color="#2a78d6", alpha=0.5, lw=0)
    ax.set_xlabel("T at issue (K)")
    ax.set_ylabel("true forecast std of S (mm)")
    ax.set_title("c  Oct issue, 3-mo lead: spread vs ENSO", loc="left")
    fig.tight_layout()
    _save(fig, out / "fig1_system.png")


def fig_sigma_vs_truth(run_dir: Path, out: Path, lead: int = 3):
    p = pd.read_csv(run_dir / f"predictions_lead{lead}.csv.gz")
    m = pd.read_csv(run_dir / f"metrics_lead{lead}.csv", index_col=0)
    keys = [k for k in _models_in(m.index) if k not in ("truth", "climatology")]
    fig, axes = plt.subplots(
        1, len(keys), figsize=(3.1 * len(keys), 3.1), sharex=True, sharey=True, squeeze=False
    )
    lim = [p["true_std"].min() * 0.8, p["true_std"].max() * 1.1]
    for ax, k in zip(axes[0], keys, strict=True):
        name, col, _ = MODEL_STYLE[k]
        ax.scatter(p["true_std"], p[f"{k}_std"], s=4, color=col, alpha=0.35, lw=0)
        ax.plot(lim, lim, color=INK, lw=1, ls="--")
        rho = m.loc[name, "sigma_spearman_within_month"]
        err = m.loc[name, "log_sigma_rmse"]
        ax.set_title(SHORT[k], loc="left")
        ax.text(
            0.04,
            0.96,
            f"within-month rank corr. {rho:.2f}\nRMS log-spread error {err:.2f}",
            transform=ax.transAxes,
            va="top",
            color=INK2,
            fontsize=8,
        )
        ax.set_xlabel("true std (mm)")
    axes[0][0].set_ylabel("predicted std (mm)")
    for ax in axes[0]:
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    fig.suptitle(
        f"Predicted vs true forecast spread, test set, {lead}-month lead (dashed: 1:1)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out / f"fig2_sigma_vs_truth_lead{lead}.png")


def fig_discard(run_dir: Path, out: Path, lead: int = 3):
    d = pd.read_csv(run_dir / f"discard_lead{lead}.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for col in d.columns:
        k = _key(col)
        if col == "random ranking":
            color, ls, label = MUTED, ":", "random ranking"
        elif col.startswith("truth"):
            _, color, ls = MODEL_STYLE["truth"]
            label = "truth (oracle ranking)"
        else:
            _, color, ls = MODEL_STYLE.get(k, (col, INK2, "-"))
            label = SHORT.get(k, col)
        if k == "climatology":
            continue  # its mean error is far larger and off-scale
        ax.plot(
            d.index * 100,
            d[col],
            color=color,
            ls=ls,
            label=label,
            marker="o" if ls == "-" else None,
            ms=3,
        )
    ax.invert_xaxis()
    ax.set_xlabel("% of forecasts retained (most confident first)")
    ax.set_ylabel("mean absolute error of the mean (mm)")
    ax.set_title(f"Discard plot, {lead}-month lead", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, out / f"fig3_discard_lead{lead}.png")


def fig_by_lead(run_dir: Path, out: Path, leads=(1, 3, 6)):
    tabs = pd.concat([pd.read_csv(run_dir / f"metrics_lead{L}.csv", index_col=0) for L in leads])
    keys = [k for k in _models_in(tabs.index.unique()) if k not in ("truth", "climatology")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    panels = [
        (
            "sigma_spearman_within_month",
            "within-month rank corr.\n(predicted vs true spread)",
            "a  State-dependent spread skill",
        ),
        ("log_sigma_rmse", "RMS log(predicted / true std)", "b  Spread error (lower is better)"),
        ("excess_crps", "CRPS minus CRPS of truth (mm)", "c  Excess CRPS over the ideal forecast"),
    ]
    for ax, (col, ylab, title) in zip(axes, panels, strict=True):
        for k in keys:
            name, color, _ = MODEL_STYLE[k]
            v = tabs.loc[name].set_index("lead")[col]
            ax.plot(v.index, v.values, color=color, marker="o", ms=5, label=SHORT[k])
            if col == "excess_crps":
                lo = tabs.loc[name].set_index("lead")["excess_crps_lo"]
                hi = tabs.loc[name].set_index("lead")["excess_crps_hi"]
                ax.vlines(v.index, lo, hi, color=color, lw=1)
        ax.set_xticks(list(leads))
        ax.set_xlabel("lead (months)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out / "fig4_metrics_by_lead.png")


def fig_attribution(run_dir: Path, out: Path):
    a = pd.read_csv(run_dir / "attribution.csv")
    piv = a.pivot(index="group", columns="model", values="importance")
    err = a.pivot(index="group", columns="model", values="std")
    order = ["month", "T", "H", "S", "distractor_1", "distractor_2"]
    piv, err = piv.loc[order], err.loc[order]
    cols = [k for k in MODEL_STYLE if k in piv.columns]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    h = 0.8 / len(cols)
    y = np.arange(len(order))
    for i, k in enumerate(cols):
        _, color, _ = MODEL_STYLE[k]
        ax.barh(
            y + (i - (len(cols) - 1) / 2) * h,
            piv[k],
            height=h * 0.85,
            xerr=err[k],
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=SHORT[k],
            error_kw={"ecolor": INK2, "lw": 0.8},
        )
    ax.set_yticks(
        y,
        [
            "calendar month",
            "T (ENSO SST)",
            "H (thermocline)",
            "S (storage)",
            "distractor 1",
            "distractor 2",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "permutation importance of the predicted spread\n(mean |change in std| / mean std)"
    )
    ax.set_title("Which inputs drive the predicted spread? (3-month lead)", loc="left")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.text(
        1.02,
        0.02,
        "truth: S and\ndistractors = 0\nby construction",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=INK2,
    )
    fig.tight_layout()
    _save(fig, out / "fig5_spread_attribution.png")


def fig_shift(run_dir: Path, out: Path):
    s = pd.read_csv(run_dir / "shift.csv")
    s["key"] = s["model"].map(_key)
    scen = list(dict.fromkeys(s["scenario"]))
    keys = [k for k in MODEL_STYLE if k in set(s["key"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, (col, lo, hi, xlab, title, ref) in zip(
        axes,
        [
            (
                "spread_error_ratio",
                "ser_lo",
                "ser_hi",
                "spread / error ratio (1 = reliable)",
                "a  Reliability after a change",
                1.0,
            ),
            (
                "coverage90",
                None,
                None,
                "coverage of the 90 % interval",
                "b  Interval coverage",
                0.9,
            ),
        ],
        strict=True,
    ):
        for j, k in enumerate(keys):
            name, color, _ = MODEL_STYLE[k]
            sub = s[s["key"] == k].set_index("scenario").loc[scen]
            yy = np.arange(len(scen)) + (j - (len(keys) - 1) / 2) * 0.13
            ax.scatter(
                sub[col], yy, color=color, s=28, zorder=3, label=SHORT[k], edgecolor="white", lw=0.6
            )
            if lo:
                ax.hlines(yy, sub[lo], sub[hi], color=color, lw=1)
        ax.axvline(ref, color=INK, lw=1, ls="--")
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left")
    axes[0].set_yticks(range(len(scen)), [x.replace("_", " ") for x in scen])
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, out / "fig6_distribution_shift.png")


def fig_training_length(run_dir: Path, out: Path):
    t = pd.read_csv(run_dir / "training_length.csv")
    t["key"] = t["model"].map(_key)
    keys = [k for k in MODEL_STYLE if k in set(t["key"]) and k not in ("truth", "climatology")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (col, ylab, title) in zip(
        axes,
        [
            (
                "sigma_spearman_within_month",
                "within-month rank corr. (spread)",
                "a  State-dependent spread skill",
            ),
            (
                "excess_crps",
                "CRPS minus CRPS of truth (mm)",
                "b  Excess CRPS over the ideal forecast",
            ),
        ],
        strict=True,
    ):
        for k in keys:
            _, color, _ = MODEL_STYLE[k]
            g = t[t["key"] == k].groupby("train_years")[col]
            ax.plot(g.mean().index, g.mean().values, color=color, marker="o", ms=5, label=SHORT[k])
            ax.fill_between(g.mean().index, g.min(), g.max(), color=color, alpha=0.15, lw=0)
        ax.set_xscale("log")
        yrs = sorted(t["train_years"].unique())
        ax.set_xticks(yrs, [str(y) for y in yrs])
        ax.minorticks_off()
        ax.axvline(130, color=MUTED, lw=1, ls=":")
        ax.text(
            130,
            ax.get_ylim()[1],
            " ~ length of the\n observed record",
            fontsize=7.5,
            color=INK2,
            va="top",
        )
        ax.set_xlabel("years of training data")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=8, loc="center right")
    fig.tight_layout()
    _save(fig, out / "fig7_training_length.png")


def fig_pit(run_dir: Path, out: Path, lead: int = 3):
    p = pd.read_csv(run_dir / f"predictions_lead{lead}.csv.gz")
    keys = [k for k in MODEL_STYLE if f"{k}_pit" in p.columns or k == "truth"]
    fig, axes = plt.subplots(
        1, len(keys), figsize=(2.3 * len(keys), 2.4), sharey=True, squeeze=False
    )
    for ax, k in zip(axes[0], keys, strict=True):
        _, color, _ = MODEL_STYLE[k]
        v = p["truth_pit" if k == "truth" else f"{k}_pit"]
        freq, edges = np.histogram(v, bins=10, range=(0, 1))
        ax.bar(
            edges[:-1],
            freq / freq.sum(),
            width=0.1,
            align="edge",
            color=color if k != "truth" else INK2,
            edgecolor="white",
            linewidth=1,
        )
        ax.axhline(0.1, color=INK, lw=1, ls="--")
        ax.set_title(SHORT[k], loc="left")
        ax.set_xlabel("PIT")
    axes[0][0].set_ylabel("frequency")
    fig.tight_layout()
    _save(fig, out / f"fig8_pit_lead{lead}.png")


def tables_part_a(run_dir: Path, out: Path):
    cols = [
        "crpss_vs_clim",
        "crpss_lo",
        "crpss_hi",
        "excess_crps",
        "sigma_spearman_within_month",
        "overlap_within_month",
        "log_sigma_rmse",
        "mean_rmse_vs_truth",
        "pit_d",
        "coverage90",
        "spread_error_ratio",
    ]
    allt = []
    for f in sorted(run_dir.glob("metrics_lead*.csv")):
        m = pd.read_csv(f, index_col=0)
        allt.append(m[["lead", *cols]])
    tab = pd.concat(allt).round(3)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "part_a_metrics.csv")
    (out / "part_a_metrics.md").write_text(tab.to_markdown())
    for name in ("attribution", "shift", "training_length"):
        if (run_dir / f"{name}.csv").exists():
            pd.read_csv(run_dir / f"{name}.csv").round(4).to_csv(
                out / f"part_a_{name}.csv", index=False
            )


# ----------------------------------------------------------------------------- Part B


def figs_part_b(run_dir: Path, out: Path, tables: Path):
    for name in ("cesm2le_test", "observations"):
        f = run_dir / f"metrics_{name}.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f, index_col=0)
        m.round(3).to_csv(tables / f"part_b_metrics_{name}.csv")
        (tables / f"part_b_metrics_{name}.md").write_text(m.round(3).to_markdown())
    fo = run_dir / "metrics_observations.csv"
    fm = run_dir / "metrics_cesm2le_test.csv"
    if not (fo.exists() and fm.exists()):
        return
    mo, mm = pd.read_csv(fo, index_col=0), pd.read_csv(fm, index_col=0)
    keys = [
        k
        for k in MODEL_STYLE
        if (
            MODEL_STYLE[k][0] in mm.index
            or MODEL_STYLE[k][0].replace("seasonal", "constant") in mm.index
        )
        and k not in ("truth", "climatology")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, (col, lo, hi, xlab, ref) in zip(
        axes,
        [
            (
                "crpss_vs_clim",
                "crpss_lo",
                "crpss_hi",
                "CRPSS vs climatology (95 % block-bootstrap CI)",
                0.0,
            ),
            ("spread_error_ratio", "ser_lo", "ser_hi", "spread / error ratio (1 = reliable)", 1.0),
        ],
        strict=True,
    ):
        for j, k in enumerate(keys):
            name, color, _ = MODEL_STYLE[k]
            for i, (tab, _label) in enumerate(
                ((mm, "CESM2-LE test members"), (mo, "observations"))
            ):
                nm = name if name in tab.index else name.replace("seasonal", "constant")
                yv = i + (j - (len(keys) - 1) / 2) * 0.14
                ax.scatter(
                    tab.loc[nm, col],
                    yv,
                    color=color,
                    s=28,
                    zorder=3,
                    edgecolor="white",
                    lw=0.6,
                    label=SHORT[k] if i == 0 else None,
                )
                ax.hlines(yv, tab.loc[nm, lo], tab.loc[nm, hi], color=color, lw=1)
        ax.axvline(ref, color=INK, lw=1, ls="--")
        ax.set_xlabel(xlab)
    axes[0].set_yticks([0, 1], ["CESM2-LE\ntest members", "observations"])
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Part B: trained on CESM2-LE, evaluated in the model world and the real world",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out / "fig9_part_b_transfer.png")


def fig_part_b_confidence(run_dir: Path, out: Path, tables: Path, seed: int = 1):
    """Error of the most-confident half vs all forecasts, with block-bootstrap CIs.

    Also writes the numbers to results/tables/part_b_confidence.csv. A model
    with constant spread (linear) has no meaningful ranking and serves as a
    reference that should show no effect.
    """
    from .metrics import block_bootstrap_ci

    keys = ["linear", "boosting", "nn_gaussian", "nn_shash", "deep_ensemble"]
    rows = []
    for name, label, block, sort_cols in (
        ("cesm2le_test", "CESM2-LE test members", 5, ["member_id", "year"]),
        ("observations", "observations", 5, ["year"]),
    ):
        f = run_dir / f"predictions_{name}.csv"
        if not f.exists():
            return
        p = pd.read_csv(f).sort_values(sort_cols)
        y = p["y"].to_numpy()
        for k in keys:
            mu, sd = p[f"{k}_mean"].to_numpy(), p[f"{k}_std"].to_numpy()

            def stat(i, y=y, mu=mu, sd=sd):
                e = np.abs(y[i] - mu[i])
                half = np.argsort(sd[i], kind="stable")[: len(i) // 2]
                return e[half].mean() / e.mean() - 1.0

            ci = block_bootstrap_ci(stat, len(y), block, n_boot=2000, seed=seed)
            rows.append({"dataset": label, "model": k, **ci})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(tables / "part_b_confidence.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for j, k in enumerate(keys):
        _, color, _ = MODEL_STYLE[k]
        sub = df[df["model"] == k].reset_index(drop=True)
        yy = np.arange(len(sub)) + (j - (len(keys) - 1) / 2) * 0.13
        ax.scatter(
            sub["estimate"] * 100,
            yy,
            color=color,
            s=30,
            zorder=3,
            edgecolor="white",
            lw=0.6,
            label=SHORT[k],
        )
        ax.hlines(yy, sub["lower"] * 100, sub["upper"] * 100, color=color, lw=1)
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_yticks([0, 1], ["CESM2-LE\ntest members", "observations\n(130 years)"])
    ax.invert_yaxis()
    ax.set_xlabel(
        "change in mean absolute error, most-confident half vs all forecasts (%)\n(95 % block-bootstrap CI; linear has constant spread, so no real ranking)"
    )
    ax.set_title("Do confident forecasts have smaller errors?", loc="left")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, out / "fig10_part_b_confidence.png")


def fig_smyle(run_dir: Path, out: Path, tables: Path):
    """Part B with the CESM2 SMYLE perfect-model truth, observed years 1971-2021.

    Panel a: SMYLE 20-member ensemble per target year (box: interquartile,
    whiskers: range) with the observed value. Panel b: predicted spread of
    the deep ensemble vs the SMYLE ensemble spread. Also writes an
    exploratory table (not part of the pre-defined test families).
    """
    from .metrics import crps_ensemble, spread_error_ratio
    from .stats_tests import circular_shift_test, confident_half_error_change, diebold_mariano

    f_truth, f_obs = run_dir / "smyle_truth.csv", run_dir / "predictions_observations.csv"
    if not (f_truth.exists() and f_obs.exists()):
        return
    g = pd.read_csv(f_truth).set_index("target_year")
    p = pd.read_csv(f_obs).set_index("year")
    years = sorted(set(g.index) & set(p.index))
    g, p = g.loc[years], p.loc[years]
    mem = g.filter(regex=r"^m\d+$").to_numpy()
    y = p["y"].to_numpy()

    crps_s = crps_ensemble(y, mem, fair=True)
    rows = [
        {
            "model": "SMYLE (initialised, 20 members)",
            "crps": crps_s.mean(),
            "spread_error_ratio": spread_error_ratio(y, mem.mean(1), mem.std(1, ddof=1)),
        }
    ]
    for k in ["linear", "boosting", "nn_gaussian", "nn_shash", "deep_ensemble"]:
        r = diebold_mariano(crps_s, p[f"{k}_crps"].to_numpy())
        row = {
            "model": SHORT[k],
            "crps": p[f"{k}_crps"].mean(),
            "spread_error_ratio": spread_error_ratio(y, p[f"{k}_mean"], p[f"{k}_std"]),
            "smyle_minus_model_crps": r["mean_diff"],
            "p_two_sided_vs_smyle": r["p_two_sided"],
        }
        if k != "linear":
            e = np.abs(y - p[f"{k}_mean"].to_numpy())
            c = circular_shift_test(
                confident_half_error_change, p[f"{k}_std"].to_numpy(), e, min_shift=5
            )
            row.update(
                {"confident_half_error_change": c["statistic"], "p_confident_half": c["p_value"]}
            )
        rows.append(row)
    rows.append({"model": "climatology (leave-one-out)", "crps": p["climatology_loo_crps"].mean()})
    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(tables / "part_b_smyle_exploratory.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.boxplot(
        mem.T,
        positions=years,
        widths=0.6,
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": GRID, "edgecolor": MUTED, "linewidth": 0.8},
        medianprops={"color": INK2, "linewidth": 1},
        whiskerprops={"color": MUTED, "linewidth": 0.8},
        capprops={"color": MUTED, "linewidth": 0.8},
    )
    ax.scatter(years, y, color="#eb6834", s=16, zorder=3, label="observed (nClimGrid)")
    ax.set_xticks(years[::5], [str(v) for v in years[::5]])
    ax.set_xlabel("target year (January-March)")
    ax.set_ylabel("standardised precipitation anomaly")
    ax.set_title("a  CESM2 SMYLE: 20 members from the observed 1 November state", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1]
    k = "deep_ensemble"
    sd = p[f"{k}_std"] / p[f"{k}_std"].mean()
    ax.errorbar(
        g["smyle_std"],
        sd,
        xerr=g["smyle_std_se"],
        fmt="o",
        ms=4,
        color=MODEL_STYLE[k][1],
        ecolor=GRID,
        elinewidth=1,
    )
    from scipy import stats as _st

    rho = _st.spearmanr(sd, g["smyle_std"]).statistic
    ax.set_xlabel("SMYLE ensemble spread (standardised; bars: sampling s.e.)")
    ax.set_ylabel("deep-ensemble spread / its mean")
    ax.set_title(f"b  Spread per year (Spearman {rho:.2f})", loc="left")
    fig.tight_layout()
    _save(fig, out / "fig11_part_b_smyle.png")


def _nclimgrid_correlation(
    raw: Path, processed: Path, cache: Path, box=(25.0, 50.0, -125.0, -100.0), coarsen=6
):
    """Correlation of October Nino-3.4 (observed) with Jan-Mar nClimGrid
    precipitation on a coarsened (6 x 1/24 deg = 0.25 deg) western-US grid,
    1896-2025. Cached as NetCDF because the source file is ~5 GB."""
    import xarray as xr

    if cache.exists():
        return xr.open_dataarray(cache).load()
    f = raw / "nclimgrid_prcp.nc"
    if not f.exists():
        return None
    obs = pd.read_csv(processed / "samples_obs.csv").set_index("year")
    ds = xr.open_dataset(f, chunks={"time": 60})
    pr = ds["prcp"].sortby("lat").sel(lat=slice(box[0], box[1]), lon=slice(box[2], box[3]))
    pr = pr.sel(time=pr["time"].dt.month.isin([1, 2, 3]))
    pr = pr.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean()
    jfm = pr.groupby(pr["time"].dt.year.rename("year")).mean("time").compute()
    years = sorted(set(jfm["year"].values) & set(obs.index))
    x = xr.DataArray(obs.loc[years, "nino34"].to_numpy(), dims="year", coords={"year": years})
    y = jfm.sel(year=years)
    corr = xr.corr(x, y, dim="year")
    corr.name = "corr"
    corr.attrs = {
        "description": "corr(Oct Nino-3.4, Jan-Mar precipitation), nClimGrid, "
        f"{years[0]}-{years[-1]}"
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    corr.to_netcdf(cache)
    return corr


def fig_map(processed: Path, raw: Path, run_dir: Path, out: Path, regions: dict):
    """Fig. 0: where the Part B predictors and target are.

    a) EOF 1 of October tropical Pacific SST anomalies (CESM2-LE training
       members), with the Nino-3.4 box and the target box.
    b) Observed correlation of October Nino-3.4 with Jan-Mar precipitation
       over the western US (nClimGrid), with the target box.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    # Explicit 1:50m Natural Earth layers (no automatic scale switching).
    LAND = cfeature.NaturalEarthFeature("physical", "land", "50m")
    COAST = cfeature.NaturalEarthFeature("physical", "coastline", "50m", facecolor="none")
    STATES = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lakes", "50m", facecolor="none"
    )
    BORDERS = cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "50m", facecolor="none"
    )

    f = processed / "eof.npz"
    if not f.exists():
        return
    d = np.load(f)
    pat = d["patterns"][0] / np.nanmax(np.abs(d["patterns"][0]))
    lat, lon = d["lat"], d["lon"]
    corr = _nclimgrid_correlation(raw, processed, run_dir / "nclimgrid_nino34_corr.nc")

    def box(ax, b, color, label, lw=1.6):
        (s, n), (w, e) = b["lat"], b["lon"]
        ax.add_patch(
            Rectangle(
                (w, s),
                e - w,
                n - s,
                transform=ccrs.PlateCarree(),
                fill=False,
                edgecolor=color,
                linewidth=lw,
                zorder=5,
            )
        )
        ax.text(
            w,
            n + 1.0,
            label,
            transform=ccrs.PlateCarree(),
            color=color,
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1])
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=200))
    ax.set_extent([110, 300, -35, 50], crs=ccrs.PlateCarree())
    m = ax.pcolormesh(
        lon, lat, pat, cmap="RdBu_r", vmin=-1, vmax=1, transform=ccrs.PlateCarree(), shading="auto"
    )
    ax.add_feature(LAND, facecolor="#e4e3df", zorder=2)
    ax.add_feature(COAST, linewidth=0.5, edgecolor=MUTED, zorder=3)
    ax.add_feature(STATES, linewidth=0.3, edgecolor=MUTED, zorder=3)
    box(ax, regions["nino34"], INK, "Niño-3.4")
    box(ax, regions["sst_domain"], INK2, "SST predictor domain", lw=0.8)
    box(ax, regions["target_box"], "#eb6834", "target: U.S. Southwest")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID)
    gl.top_labels = gl.right_labels = False
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, shrink=0.6)
    cb.set_label("EOF 1 loading (normalised)")
    ax.set_title(
        f"a  Leading pattern of October SST anomalies, CESM2-LE ({d['evr'][0] * 100:.0f} % of variance)",
        loc="left",
    )

    ax = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    ax.set_extent([-125, -100, 25, 50], crs=ccrs.PlateCarree())
    if corr is not None:
        m2 = ax.pcolormesh(
            corr["lon"],
            corr["lat"],
            corr,
            cmap="BrBG",
            vmin=-0.6,
            vmax=0.6,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        cb2 = fig.colorbar(m2, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, shrink=0.8)
        cb2.set_label("correlation")
    ax.add_feature(COAST, linewidth=0.5, edgecolor=INK2)
    ax.add_feature(STATES, linewidth=0.4, edgecolor=INK2)
    ax.add_feature(BORDERS, linewidth=0.6, edgecolor=INK2)
    box(ax, regions["target_box"], "#eb6834", "target box", lw=2.0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID)
    gl.top_labels = gl.right_labels = False
    period = corr.attrs.get("description", "").split(", ")[-1] if corr is not None else ""
    ax.set_title(
        f"b  Oct Niño-3.4 vs Jan–Mar precipitation\n    (observed, nClimGrid {period})", loc="left"
    )
    fig.tight_layout()
    _save(fig, out / "fig0_part_b_map.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-a", default="runs/part_a")
    ap.add_argument("--part-b", default="runs/part_b")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--config-b", default="configs/part_b.yaml")
    args = ap.parse_args(argv)
    _style()
    a, figs, tabs = Path(args.part_a), Path(args.figures), Path(args.tables)
    tabs.mkdir(parents=True, exist_ok=True)
    if (a / "metrics_lead3.csv").exists():
        fig_system(a, figs)
        fig_sigma_vs_truth(a, figs)
        fig_discard(a, figs)
        if all((a / f"metrics_lead{L}.csv").exists() for L in (1, 3, 6)):
            fig_by_lead(a, figs)
        if (a / "attribution.csv").exists():
            fig_attribution(a, figs)
        if (a / "shift.csv").exists():
            fig_shift(a, figs)
        if (a / "training_length.csv").exists():
            fig_training_length(a, figs)
        fig_pit(a, figs)
        tables_part_a(a, tabs)
    if Path(args.part_b).exists():
        import yaml

        cfg_b = yaml.safe_load(Path(args.config_b).read_text())
        try:
            fig_map(
                Path(cfg_b["paths"]["processed"]),
                Path(cfg_b["paths"]["raw"]),
                Path(args.part_b),
                figs,
                cfg_b["regions"],
            )
        except ImportError:
            print("cartopy not installed: skipping the map figure (pip install cartopy)")
        figs_part_b(Path(args.part_b), figs, tabs)
        fig_part_b_confidence(Path(args.part_b), figs, tabs)
        fig_smyle(Path(args.part_b), figs, tabs)


if __name__ == "__main__":
    main()
