# knownpred: testing uncertainty-aware forecasts where the predictability is known

[![CI](https://github.com/OWNER/known-predictability-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/known-predictability-testbed/actions/workflows/ci.yml)

Neural networks that predict a full distribution instead of a single value are increasingly used to find **forecasts of opportunity**: initial states from which the future is more predictable than usual (e.g. Mayer & Barnes, 2021; Gordon & Barnes, 2022). In climate-model or observational data the true predictability of each state is unknown. One can therefore check whether a network is calibrated on average, but not whether its "confident" cases are the ones that are actually predictable.

This repository builds a small stochastic climate system in which the true forecast distribution can be computed for every state, and uses it to test uncertainty-aware models (Part A). It then applies the same pipeline to real data: training on the CESM2 Large Ensemble and evaluating on observations (Part B).

![The toy system](results/figures/fig1_system.png)
*(a) The ENSO-like index of the toy system. (b) True forecast spread of the storage anomaly by issue month and lead (line: mean; band: 10–90 % range across states). (c) For forecasts issued at the end of October, the true spread grows with the ENSO state: a built-in window of opportunity during La Niña.*

## Questions

| | Question | How it is measured |
|---|---|---|
| Q1 | Does the predicted spread follow the true spread across states and leads? | rank correlation and RMS log error against the true σ |
| Q2 | Are the most-confident forecasts the truly most-predictable ones? | overlap of confident sets; discard plots |
| Q3 | Does attribution of the predicted spread point to the true sources of predictability? | permutation importance compared with the exact attribution of the true σ |
| Q4 | Does the reported confidence stay reliable when the system changes after training? | spread–error ratio and interval coverage under three imposed changes; Part B: model → observations |
| Q5 | How many years of training data are needed before spread estimates are useful? | Q1/Q2 metrics for 70–1000 years of training data |

## The testbed (Part A)

Three coupled components, all monthly (full equations and parameters: [`docs/methods.md`](docs/methods.md)):

1. **ENSO-like recharge oscillator** (after Jin, 1997; Burgers et al., 2005), with seasonal damping and **state-dependent noise**.
2. **Regional precipitation**, linked to ENSO in November–March only. Its noise is skewed (Gamma) and, in the cool season, grows with the ENSO state.
3. **Land storage**: a linear reservoir with six-month memory. The forecast target is storage 1, 3 or 6 months ahead.

The forecast models observe the full state (ENSO index, thermocline, storage, calendar month) and two irrelevant "distractor" series. The true forecast distribution for any test state is therefore obtained by running 1000 stochastic continuations from it. This makes some exact statements possible. For example, the true spread does not depend on current storage (storage shifts only the mean) or on the distractors. Real-world attribution studies have no such reference.

## Results so far

> **Status.** The non-neural models below were run in full. The PyTorch models (Gaussian and SHASH networks, a deep ensemble) are implemented and tested but have **not been run yet**; their rows will be added after the first run on a GPU workstation. See [`docs/reproducibility.md`](docs/reproducibility.md). Part B has been verified only on synthetic stand-in files so far.

Test set: 500 independent simulated years. Training: 1000 years. Lead: 3 months. Brackets are 95 % block-bootstrap intervals.

| Model | CRPSS vs climatology | Excess CRPS over truth (mm) | Spread rank corr. within month | Confident-set overlap (chance 0.20) | RMS log spread error | Mean error vs true mean (σ units) | PIT D | 90 % coverage |
|---|---|---|---|---|---|---|---|---|
| climatology | 0 | 14.70 [12.95, 16.46] | 0.00 | 0.20 | 0.91 | 2.19 | 0.007 | 0.877 |
| linear, seasonal σ | 0.542 [0.512, 0.569] | 0.77 [0.64, 0.92] | 0.00 | 0.20 | 0.31 | 0.29 | 0.021 | 0.914 |
| gradient boosting, two-stage | 0.534 [0.507, 0.560] | 0.96 [0.71, 1.25] | 0.47 | 0.50 | 0.23 | 0.45 | 0.021 | 0.918 |
| truth (ideal forecast) | 0.572 [0.541, 0.600] | 0 | 1 | 1 | 0 | 0 | 0.005 | 0.902 |

All leads: [`results/tables/part_a_metrics.md`](results/tables/part_a_metrics.md).

**Observations from the baselines, and the questions they raise for the networks:**

- **Outcome-based scores did not reveal which model captured the state-dependent spread.**
  - The linear model has, by construction, no state-dependent spread. Yet it has slightly lower CRPS than gradient boosting, whose spread tracks the truth (within-month rank correlation 0.47 versus 0; RMS log spread error 0.23 versus 0.31).
  - The difference comes from the mean: boosting's mean is further from the true mean (0.45 versus 0.29 σ).
  - Both models have the same PIT deviation (0.021).
  - So aggregate CRPS and PIT measured mostly the mean and the average spread. Can a network do well on both mean and spread at once, and would anything other than the truth-based metrics show it?
- **Confidence was ranked better than chance, but far from the truth.** Boosting placed half of the truly most predictable 20 % of cases in its own most-confident 20 % (chance: 20 %). Its discard curve ([fig. 3](results/figures/fig3_discard_lead3.png)) lies below the linear model's for most retained fractions, and above the truth's.
- **The spread attribution contained spurious contributions** ([fig. 5](results/figures/fig5_spread_attribution.png)).
  - Boosting attributed 6–8 % relative spread changes each to storage and to both distractors, all exactly zero in the truth.
  - Its attribution to the ENSO index (0.26) and to the calendar month (0.41) was close to the truth (0.27, 0.46).
  - Would an attribution of a network's spread, with integrated gradients or permutation, contain fewer such contributions, or more?
- **After the system changed, every trained model became overconfident** ([fig. 6](results/figures/fig6_distribution_shift.png)).
  - With a 50 % stronger teleconnection, the spread–error ratio dropped to 0.79 for both models (truth: 1.04), and boosting's 90 % intervals covered 86 % of outcomes.
  - This is the situation Part B tests with real data.
- **Short records limited spread estimation more than mean estimation** ([fig. 7](results/figures/fig7_training_length.png)).
  - With 70 years of training data, the within-month spread rank correlation of boosting fell to 0.20, and its excess CRPS (4.1 mm) was three times that of the linear model (1.3 mm). The two converged only at about 1000 years.
  - Observational records are about 70–130 years long. Does a network's spread estimate degrade the same way, and does an ensemble help?

![Predicted vs true spread](results/figures/fig2_sigma_vs_truth_lead3.png)

## Part B: the same pipeline on real data

The question carried over from Part A: a model trained in a climate model's world reports its own confidence. Is that confidence reliable on unseen members of the same model, and on the real world?

| | |
|---|---|
| Training data | CESM2-LE, CMIP6 forcing variant, 50 members, historical + SSP3-7.0 (1850–2050); split by member 70/15/15 % |
| Observations | ERSSTv5 sea surface temperature; NOAA nClimGrid precipitation (1895–2025) |
| Predictors (October) | first 6 EOFs of tropical Pacific SST anomalies (fitted on training members; observations projected onto them), Niño-3.4, its Jul–Oct tendency, Aug–Oct regional precipitation |
| Target | January–March mean precipitation anomaly, U.S. Southwest (31–37°N, 115–103°W) |
| Forced response | model: ensemble mean removed; observations: quadratic trend removed |
| Reference forecast | leave-one-year-out observed climatology |

The processing steps are each unit-tested on small arrays (`tests/test_preprocess.py`): CESM end-of-interval time stamps relabelled by month, unit conversion with month lengths, area weighting, periodic regridding, common masks and EOFs fitted only on training members. Data provenance (URL, access date, SHA-256) is written to `data/raw/MANIFEST.json`, and the licences and citations are in [`docs/data.md`](docs/data.md).

## Repository layout

```
configs/                 experiment configurations (all settings and seeds)
src/knownpred/
  simulate/system.py     toy system: recharge oscillator, precipitation, storage
  truth.py               Monte Carlo ground truth
  distributions.py       Gaussian and SHASH distributions (NumPy reference)
  metrics.py             CRPS, PIT, spread-skill, discard curves, block bootstrap
  baselines.py           climatology, linear, two-stage gradient boosting
  models/                PyTorch networks, losses, training, deep ensemble
  attribution.py         permutation importance (models and truth), integrated gradients
  evaluation.py          one scoring function for every model
  data/                  Part B: sources registry, download, xarray preprocessing, samples
  experiments/           Part A and Part B drivers, provenance and seeding
  plotting.py            figures and tables in results/
tests/                   unit, integration and end-to-end tests
results/                 figures and tables (committed); large outputs stay in runs/
docs/                    methods, data sources, reproducibility
```

## Quick start

```bash
pip install -e ".[all]"                              # or: conda env create -f environment.yml
pytest -m "not slow"                                 # ~1.5 min
python -m knownpred.experiments.part_a --quick       # smoke test
python -m knownpred.experiments.part_a               # full Part A
python -m knownpred.data.download                    # Part B data
python -m knownpred.experiments.part_b               # Part B
python -m knownpred.plotting                         # figures and tables
```

`make all` runs the whole chain. `python -m knownpred.experiments.part_b --fake-data` runs the Part B pipeline on synthetic stand-in files without downloading anything; those results have no scientific meaning.

## Relation to prior work, and attribution

This is independent work. The questions build on published research on forecasts of opportunity and on explainable and uncertainty-aware neural networks in the geosciences, in particular:

- forecasts of opportunity identified by networks: Mayer & Barnes (2021), *Geophys. Res. Lett.*; Gordon & Barnes (2022), *Geophys. Res. Lett.*;
- adding uncertainty outputs (including SHASH) to regression networks: Barnes, Barnes & Gordon (2021), arXiv:2109.07250;
- synthetic benchmarks for explainable AI: Mamalakis, Barnes & Ebert-Uphoff (2022), *Artificial Intelligence for the Earth Systems*.

The testbed adds an exact reference for the predictability of each state and for the attribution of the spread. All code here was written from scratch from the published descriptions of the methods; no code was copied or adapted from other repositories. Methods are cited where they are implemented (module docstrings) and in [`docs/methods.md`](docs/methods.md). No affiliation with or endorsement by any of the cited authors or data providers is implied.

## Limitations

- The toy system is deliberately simple: 2 + 1 + 1 variables and linear dynamics apart from the noise. Results show how methods behave when the answer is known, not how they will behave in a high-dimensional climate model. They are also tied to the parameter choices in [`docs/methods.md`](docs/methods.md).
- Truth-based metrics carry Monte Carlo error: about 2–3 % of each true σ with 1000 members.
- Only one network architecture size is tested, and the hyperparameters were not tuned. A conclusion of the form "networks do or do not recover X" holds for this setup only.
- In Part B, removing a quadratic trend from a single observed realisation is a crude estimate of the forced response. Standardising each dataset by its own statistics corrects variance bias but uses two statistics from part of the observed record. The observational evaluation has about 130 samples, so its confidence intervals are wide.

## Citation and licence

Code: MIT ([`LICENSE`](LICENSE)). Please cite via [`CITATION.cff`](CITATION.cff). External datasets keep their own terms ([`docs/data.md`](docs/data.md)).
