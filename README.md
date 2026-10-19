# knownpred: testing uncertainty-aware forecasts where the predictability is known

[![CI](https://github.com/OWNER/known-predictability-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/known-predictability-testbed/actions/workflows/ci.yml)

Neural networks that predict a full distribution instead of a single value are increasingly used to find **forecasts of opportunity**: initial states from which the future is more predictable than usual (e.g. Mayer & Barnes, 2021; Gordon & Barnes, 2022). In climate-model or observational data the true predictability of each state is unknown. One can therefore check whether a network is calibrated on average, but not whether its "confident" cases are the ones that are actually predictable.

This repository builds a small stochastic climate system in which the true forecast distribution can be computed for every state, and uses it to test uncertainty-aware models (Part A). It then applies the same pipeline to real data: training on the CESM2 Large Ensemble and evaluating on observations (Part B).

![Predicted vs true spread](results/figures/fig2_sigma_vs_truth_lead3.png)
*The core comparison this testbed makes possible. Each point is one test forecast (3-month lead): the spread a model predicted against the true spread of the system from that state. Linear regression can only vary its spread with the season (horizontal bands). Gradient boosting follows the truth loosely. The probabilistic networks follow it closely, but still only partly within each season (rank correlation 0.53–0.56).*

**At a glance.**
- **Part A (synthetic, exact truth):** the networks recovered the true forecast spread best and beat linear regression at 1–3 month leads (formally significant). Every model still attributed some spread to irrelevant inputs, became overconfident when the system changed, and needed long records to learn the spread.
- **Part B (CESM2-LE → observations, U.S. Southwest winter precipitation):** skill over climatology is modest. With multiple-testing control, no model is significantly better than climatology or than linear regression in the 130-year observed record. A perfect-model truth from initialised CESM2 SMYLE hindcasts (1971–2021) is used to test the networks' year-by-year confidence.

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

![The toy system](results/figures/fig1_system.png)
*(a) The ENSO-like index of the toy system. (b) True forecast spread of the storage anomaly by issue month and lead (line: mean; band: 10–90 % range across states). (c) For forecasts issued at the end of October, the true spread grows with the ENSO state: a built-in window of opportunity during La Niña.*

## Part A results

Test set: 500 independent simulated years. Training: 1000 years. Lead: 3 months. Brackets are 95 % block-bootstrap intervals. All six models were run on CPU in 13 minutes ([`docs/reproducibility.md`](docs/reproducibility.md)).

| Model | CRPSS vs climatology | Excess CRPS over truth (mm) | Spread rank corr. within month | Confident-set overlap (chance 0.20) | RMS log spread error | Mean error vs true mean (σ units) | PIT D | 90 % coverage |
|---|---|---|---|---|---|---|---|---|
| climatology | 0 | 14.70 [12.95, 16.46] | 0.00 | 0.20 | 0.91 | 2.19 | 0.007 | 0.877 |
| linear, seasonal σ | 0.542 [0.512, 0.569] | 0.77 [0.64, 0.92] | 0.00 | 0.20 | 0.31 | 0.29 | 0.021 | 0.914 |
| gradient boosting, two-stage | 0.534 [0.507, 0.560] | 0.96 [0.71, 1.25] | 0.47 | 0.50 | 0.23 | 0.45 | 0.021 | 0.918 |
| NN, Gaussian | 0.558 [0.529, 0.585] | 0.37 [0.25, 0.50] | 0.53 | 0.56 | 0.11 | 0.28 | 0.020 | 0.899 |
| NN, SHASH | 0.559 [0.529, 0.585] | 0.34 [0.21, 0.48] | 0.55 | 0.59 | 0.12 | 0.28 | 0.011 | 0.897 |
| deep ensemble (5 × Gaussian NN) | 0.558 [0.529, 0.585] | 0.36 [0.24, 0.48] | 0.56 | 0.58 | 0.11 | 0.28 | 0.022 | 0.915 |
| truth (ideal forecast) | 0.572 [0.541, 0.600] | 0 | 1 | 1 | 0 | 0 | 0.005 | 0.902 |

All leads: [`results/tables/part_a_metrics.md`](results/tables/part_a_metrics.md).

**Q1. Does the predicted spread follow the true spread?**
- **The networks recovered the spread more closely than the other models.**
  - Their spread error (RMS log 0.11–0.12) was half that of gradient boosting (0.23).
  - Their gap to the ideal forecast's CRPS (0.34–0.37 mm) was half that of the linear model (0.77 mm), with non-overlapping intervals.
  - Paired Diebold–Mariano tests confirm that every network had lower CRPS than the linear model at the 1- and 3-month leads (Holm-adjusted p < 10⁻⁶ in all six tests).
- **Outcome-based scores alone barely separated the models.** The CRPSS intervals of all five trained models overlap, and the linear model, which has no state-dependent spread at all, ranks above gradient boosting on CRPSS. Only the comparison with the truth shows the differences in spread.
- **Only the SHASH network matched the shape of the distribution.** Its PIT deviation (0.011) was half that of every other model (≈0.021), which fits it being the only model that can represent the skewed precipitation noise.
- **The networks' advantage shrank at the 6-month lead.**
  - Their excess CRPS (0.50–0.60 mm) overlapped the linear model's (0.55 mm), although their spread rank correlation rose to 0.68–0.75. No network's CRPS differed significantly from the linear model's (Holm-adjusted p = 1), while gradient boosting was significantly worse (p < 10⁻⁴).
  - At that lead the error of the mean dominates the score ([fig. 4](results/figures/fig4_metrics_by_lead.png)).

**Q2. Are the most-confident forecasts the most predictable ones?**
- Partly. The networks placed 56–59 % of the truly most predictable 20 % of cases in their own most-confident 20 % (chance: 20 %; boosting: 50 %).
- The discard curves ([fig. 3](results/figures/fig3_discard_lead3.png)) lie between random ranking and the oracle ranking.

**Q3. Does attribution of the spread find the true sources?** ([fig. 5](results/figures/fig5_spread_attribution.png))
- **The large contributions were right.** Calendar month and the ENSO index were attributed at close to their true importance by every non-linear model (month 0.41–0.43 vs 0.46; ENSO 0.26–0.28 vs 0.27).
- **Every non-linear model also attributed spread to inputs with exactly zero true importance.**
  - Storage received 0.06–0.08, and the two distractors 0.03–0.07 each.
  - The deep ensemble had the smallest distractor attributions (0.03–0.04) but not a smaller storage attribution (0.06).
- Storage is correlated with the ENSO state in the training data, because it integrates ENSO-driven precipitation. Shuffling it therefore also creates combinations of inputs the models never saw. Whether the storage attribution reflects the models' use of storage as a stand-in for ENSO, or an artefact of permutation outside the data distribution, is a question for conditional attribution methods.

**Q4. Does confidence stay reliable when the system changes?** ([fig. 6](results/figures/fig6_distribution_shift.png))
- No. With a 50 % stronger teleconnection, every trained model became overconfident:
  - spread–error ratio 0.78–0.81 (truth 1.04);
  - networks' 90 % intervals covered 83–86 % of outcomes.
- The networks were no better calibrated after the change than the linear model, although they were closer to the truth before it.

**Q5. How much training data is needed?**

![Training length](results/figures/fig7_training_length.png)
- **At 70 years the networks were worse than the linear model.** Their excess CRPS was 1.5–1.8 mm against 1.3 mm, and their 90 % intervals covered only 83–85 % of outcomes.
- **The single networks only overtook the linear model at about 300 years.** The deep ensemble already did so at 150 years.
- The state-dependent spread skill of the networks still rose from 0.37–0.40 at 70 years to about 0.55 at 1000 years.
- Observational records are 70–130 years long, which falls in the range where the networks' confidence was least reliable here.

## Part B: the same pipeline on real data

The question carried over from Part A: a model trained in a climate model's world reports its own confidence. Is that confidence reliable on unseen members of the same model, and on the real world?

![Region](results/figures/fig0_part_b_map.png)
*(a) Predictors come from the tropical Pacific. Shown is the leading pattern of October SST anomalies in CESM2-LE (ENSO), with the Niño-3.4 box. (b) The target is January–March precipitation over the U.S. Southwest box (orange; roughly Arizona and New Mexico). There, observed winter precipitation correlates most strongly with the preceding October's Niño-3.4, the wet side of the classic El Niño north–south dipole.*

| | |
|---|---|
| Training data | CESM2-LE, CMIP6 forcing variant, 50 members, historical + SSP3-7.0 (1900–2050); split by member 70/15/15 % |
| Observations | ERSSTv5 sea surface temperature; NOAA nClimGrid precipitation (1895–2025) |
| Predictors (October) | first 6 EOFs of tropical Pacific SST anomalies (fitted on training members; observations projected onto them), Niño-3.4, its Jul–Oct tendency, Aug–Oct regional precipitation |
| Target | January–March mean precipitation anomaly, U.S. Southwest (31–37°N, 115–103°W) |
| Forced response | model: ensemble mean removed; observations: quadratic trend removed |
| Reference forecast | leave-one-year-out observed climatology |

The processing steps are each unit-tested on small arrays (`tests/test_preprocess.py`): CESM end-of-interval time stamps relabelled by month, unit conversion with month lengths, area weighting, periodic regridding, common masks and EOFs fitted only on training members. Data provenance (URL, access date, SHA-256) is written to `data/raw/MANIFEST.json`, and the licences and citations are in [`docs/data.md`](docs/data.md).

### Sanity checks on the processed data

- The leading SST EOF explains 63 % of the October variance in CESM2-LE and tracks Niño-3.4 at r = 0.98 in the model and 0.96 in observations.
- October Niño-3.4 correlates with January–March Southwest precipitation at r = 0.44 in CESM2-LE and r = 0.34 in observations (1896–2025), so the model's teleconnection is somewhat stronger than observed.
- Known events are reproduced: the 1982–83 and 1997–98 El Niños were followed by wet winters (target +1.2 and +1.1 σ), and the 2015–16 El Niño by the well-documented dry exception (−0.9 σ).

### Part B results

Samples: 1050 held-out CESM2-LE member-years and 130 observed years. Brackets are 95 % block-bootstrap intervals (5-year blocks). Full tables: [`results/tables/`](results/tables/).

| Model | CRPSS, CESM2-LE test | CRPSS, observations | Spread/error ratio, observations | Error change, most-confident half, CESM2-LE | Error change, most-confident half, observations |
|---|---|---|---|---|---|
| linear, constant σ | 0.120 [0.094, 0.145] | 0.058 [−0.038, 0.143] | 0.99 [0.86, 1.15] | +2 % [−5, +5] | −2 % [−14, +13] |
| gradient boosting | 0.113 [0.087, 0.137] | 0.085 [0.007, 0.156] | 1.09 [0.96, 1.25] | −5 % [−10, 0] | −4 % [−19, +12] |
| NN, Gaussian | 0.123 [0.097, 0.147] | 0.078 [−0.011, 0.155] | 1.00 [0.87, 1.15] | −6 % [−10, −1] | −18 % [−33, −3] |
| NN, SHASH | 0.121 [0.096, 0.144] | 0.073 [−0.015, 0.146] | 1.01 [0.87, 1.18] | −6 % [−10, 0] | −19 % [−29, −2] |
| deep ensemble | 0.122 [0.096, 0.146] | 0.072 [−0.018, 0.151] | 1.01 [0.88, 1.17] | −6 % [−11, −1] | −21 % [−33, −4] |

![Confident forecasts](results/figures/fig10_part_b_confidence.png)

<details><summary>Skill and reliability by dataset (figure)</summary>

![Part B transfer](results/figures/fig9_part_b_transfer.png)
</details>

Formal tests, Holm-corrected within each pre-defined family (full table: [`results/tables/significance.md`](results/tables/significance.md)):

| Question | CESM2-LE test members | Observations (130 years) |
|---|---|---|
| Does each model beat climatology? (Diebold–Mariano, one-sided) | yes, all five (adjusted p < 10⁻⁴) | no model after Holm (adjusted p ≥ 0.14); gradient boosting only under Benjamini–Hochberg (0.046) |
| Does any non-linear model beat the linear model? (Diebold–Mariano, two-sided) | no (adjusted p ≥ 0.58) | no (adjusted p ≥ 0.56) |
| Does the model's own spread order its errors? (circular-shift test, one-sided) | not after Holm (adjusted p = 0.099 for all four); networks pass Benjamini–Hochberg (0.026) | not after Holm (adjusted p = 0.066–0.30); networks pass Benjamini–Hochberg (0.026–0.044) |

- **Skill was modest.** In held-out CESM2-LE members all models gained about 12 % in CRPS over climatology, a clear result. In observations the point estimates were 6–9 %, and no model was significantly better than climatology once the five comparisons were corrected for multiple testing.
- **The models were indistinguishable from each other.** No non-linear model beat the linear model on either dataset, in contrast to Part A, where the networks clearly did.
- **Average reliability carried over to observations within wide limits.** Every model's spread/error ratio in observations was compatible with 1, but the intervals span roughly ±15 %. With 130 years, the overconfidence seen in Part A after a change of system (ratio ≈0.8) could not be detected here.
- **The networks learned a state-dependent spread, and the evidence that it orders real errors is suggestive but not conclusive.**
  - In all three networks the predicted spread rose with October Niño-3.4 (r = 0.68–0.85): wetter El Niño winters were forecast as more uncertain, drier La Niña winters as less.
  - In observations the networks' most-confident half of years had 18–21 % smaller errors than all years (unadjusted p = 0.008–0.033). After Holm correction this is not significant at 5 %. It does hold at a 5 % false discovery rate.
  - In CESM2-LE the same effect was smaller (−6 %) and showed the same pattern: it passes the false-discovery-rate correction but not Holm's.
- **Open question raised by the size difference** (−18 to −21 % in observations versus −6 % in the model): observed precipitation is more skewed (skewness 1.1 versus 0.7), and dry La Niña winters are bounded near zero rainfall. Does that make spread more predictable in the real world than in CESM2, or is the larger observed effect sampling noise in 130 years? The SMYLE comparison below is designed to help answer it.

### Perfect-model ground truth for real years (CESM2 SMYLE)

Part B has no exact truth for the predictability of each real year. The closest available reference is an *initialised* hindcast ensemble with the same model: CESM2 SMYLE (Yeager et al., 2022) starts 20 members from the observed state on 1 November of every year 1970–2019. For each real year, the spread of those 20 members for January–March Southwest precipitation is CESM2's own forecast uncertainty from that exact initial state. It plays the role the 1000-member truth plays in Part A, with three caveats: it is the truth *of the model*, not of the real climate; 20 members give a spread estimate with about 16 % sampling error; and it uses the full initial state, so it is an upper bound on what the statistical models could know.

The tests (family `B-smyle` in `significance.py`):
1. Does each model's predicted spread rank the 50 real years like the SMYLE spread does?
2. Does the SMYLE spread itself order the observed errors?
3. Is SMYLE better than climatology?

Data: the November-start SMYLE precipitation collection (NCAR GDEX dataset d010074, CC BY 4.0), 51 start years × 20 members, regional January–March means over the same box, drift-corrected, trend-removed and standardised like the observations. Observed target years 1971–2021 (n = 51).

![SMYLE](results/figures/fig11_part_b_smyle.png)

**Sanity checks.**
- The SMYLE ensemble mean correlates with the observed target at r = 0.45, and with October Niño-3.4 at r = 0.73.
- Its spread/error ratio against observations is 0.97, so the 20-member ensemble is about as wide as its errors.
- The SMYLE spread is itself larger in El Niño years (r = 0.37 with Niño-3.4), the same direction as the networks' spread.

**Pre-defined tests (family `B-smyle`, Holm-corrected).**

| Question | Result |
|---|---|
| Does each model's spread rank the years like SMYLE's spread? (Spearman, circular shift) | networks ρ = 0.33–0.35 (unadjusted p = 0.02–0.05), gradient boosting ρ = 0.19 (p = 0.09); none significant after Holm (adjusted p ≥ 0.12) |
| Does the SMYLE spread order the observed errors of the SMYLE mean? | no: +5 % error change, p = 0.65 |
| Is SMYLE better than climatology? (Diebold–Mariano) | yes: CRPS 0.51 vs 0.61, Holm-adjusted p = 0.022 |

For reference, the expected Spearman correlation between the true spread and a 20-member estimate of it is about 0.82. So the 0.33–0.35 found here sits well below what sampling noise in SMYLE alone would allow; the networks' spreads are not mainly a noisy copy of SMYLE's.

**Exploratory comparisons on the same 51 years** (not part of a corrected family; [`results/tables/part_b_smyle_exploratory.csv`](results/tables/part_b_smyle_exploratory.csv)):
- **Skill:** the statistical models trained only on the uninitialised CESM2-LE had the same CRPS as SMYLE: 0.48–0.49 against 0.51, with no paired difference near significance (p = 0.42–0.63).
- **Confidence:** over these years, the networks' own most-confident half again had 22–25 % smaller errors (p = 0.047: only one of the 42 shifted alignments gave an effect as large).

**Questions these results raise.**
- The networks' spread agrees only moderately with the perfect-model spread from the full initial state. Yet in the observed record it orders the real errors at least as well as the SMYLE spread does (the SMYLE spread shows no ordering at all). Is the networks' spread capturing something about the real precipitation distribution that SMYLE's 20 members do not, such as the bounded, skewed dry La Niña winters? Or is this sampling variability in 51 years?
- A longer hindcast set, or a second initialised model (the same NCAR collection holds E3SMv2.1 SMYLE), would help separate the two.

Run with:

```bash
python -m knownpred.experiments.part_b --stage smyle   # reads the needed chunks from thredds.rda.ucar.edu
python -m knownpred.experiments.significance
```

## Repository layout

```
configs/                 experiment configurations (all settings and seeds)
src/knownpred/
  simulate/system.py     toy system: recharge oscillator, precipitation, storage
  truth.py               Monte Carlo ground truth
  distributions.py       Gaussian and SHASH distributions (NumPy reference)
  metrics.py             CRPS, PIT, spread-skill, discard curves, block bootstrap
  stats_tests.py         Diebold-Mariano (HAC), circular-shift test, Holm / Benjamini-Hochberg
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
python -m knownpred.experiments.significance         # formal tests, multiple-testing control
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
- Each network configuration was trained once per experiment (one seed; the deep ensemble aside). Seed-to-seed spread is visible only in the training-length experiment (three repeats).
- In Part B, removing a quadratic trend from a single observed realisation is a crude estimate of the forced response. Standardising each dataset by its own statistics corrects variance bias but uses two statistics (mean, standard deviation) from 1951–2010, which lies inside the evaluation period. The observational evaluation has 130 samples, so its confidence intervals are wide, and the Part B results come from one region, one season and one lead.
- The SMYLE comparison covers 51 years and one model; SMYLE's 20-member spread carries ~16 % sampling error per year, and its truth is CESM2's, not the real climate system's.
- The observational test is not fully independent of model development: the region, season and predictors were chosen before seeing the results, but the well-known ENSO teleconnection to the Southwest is the reason they were chosen.

## Citation and licence

Code: MIT ([`LICENSE`](LICENSE)). Please cite via [`CITATION.cff`](CITATION.cff). External datasets keep their own terms ([`docs/data.md`](docs/data.md)).
