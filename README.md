# knownpred: testing uncertainty-aware forecasts where the predictability is known

[![CI](https://github.com/hadikmi/Known-predictability-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/hadikmi/Known-predictability-testbed/actions/workflows/ci.yml)

> **Status: initial release (v0.1), in active development.**

Probabilistic neural networks are used to find **forecasts of opportunity**: states from which the climate is more predictable than usual (e.g. Mayer & Barnes, 2021; Gordon & Barnes, 2022). In real data the true predictability of each state is unknown. One can check whether a model is calibrated on average, but not whether its "confident" forecasts are the truly predictable ones.

This repository approaches the problem in two parts:

- **Part A, a testbed with an exact truth.** A small stochastic climate system (ENSO-like oscillator → teleconnected precipitation → land storage), in which the true forecast distribution of every state is computed by Monte Carlo re-simulation.
- **Part B, the same pipeline on real data.** Training on the CESM2 Large Ensemble and testing on observations (ERSSTv5, nClimGrid). Initialised CESM2 SMYLE hindcasts serve as a perfect-model reference for 1971–2021.

![Region](results/figures/fig0_part_b_map.png)
*Part B. (a) October tropical Pacific SST anomalies (leading CESM2-LE pattern, ENSO, 63 % of variance) are the predictors. (b) The target is January–March precipitation over the U.S. Southwest (orange box), where the observed correlation with the preceding October's Niño-3.4 is strongest (nClimGrid, 1896–2025).*

![Predicted vs true spread](results/figures/fig2_sigma_vs_truth_lead3.png)
*Part A. Each dot is one test forecast (3-month lead): the spread a model predicted (y) against the true spread of the system from that state (x); dashed line = perfect agreement. The **linear** model is a deliberately simple baseline, "seasonally calibrated but state-blind". It assigns one spread per issue month, equal on average to that month's true spread, so each month appears as a horizontal band. The bands are horizontal by construction, not an error: the model cannot tell a quiet La Niña state (true spread ≈ 7 mm) from a volatile El Niño state (≈ 90 mm) issued in the same month. **Gradient boosting** follows the truth loosely. The **probabilistic networks** follow the diagonal closely, though only partly within each season (rank correlation 0.53–0.56).*

## Key findings

- **Spread recovery.** The probabilistic networks (Gaussian, SHASH, deep ensemble) recovered the true forecast spread with half the error of gradient boosting. They beat linear regression at 1–3 month leads (Diebold–Mariano, Holm-adjusted p < 10⁻⁶).
- **Confidence ranking.** Their most-confident forecasts were close to the truly most predictable ones: discard curves near the oracle down to about 30 % of forecasts retained. Only the skew-aware SHASH network matched the shape of the true distribution.
- **Measurable failure modes.**
  - Confidence became overconfident after a change in the system (spread/error ratio ≈ 0.8).
  - Spread estimation needed long training records (about 150–300 years). That is why Part B trains on a 50-member climate-model ensemble.
- **Real data.**
  - The pipeline reproduces the known El Niño teleconnection and events.
  - All models clearly beat climatology in held-out CESM2-LE members (p < 10⁻⁴) and stay reliable on average in observations.
  - The networks' spread grows with El Niño, the same direction as in the SMYLE hindcasts.

All figures, tables, formal tests and open questions: **[docs/results.md](docs/results.md)**.

## Methods in brief

| | |
|---|---|
| Models | climatology, linear regression, two-stage gradient boosting, Gaussian and SHASH neural networks, deep ensemble (PyTorch) |
| Scoring | CRPS, PIT, interval coverage, spread/error ratio, discard curves; truth-based spread metrics (Part A) |
| Statistics | circular block bootstrap; Diebold–Mariano tests with HAC variance; circular-shift randomisation tests; Holm and Benjamini–Hochberg corrections |
| Data (Part B) | CESM2-LE (AWS Zarr), ERSSTv5, nClimGrid, CESM2 SMYLE; download provenance and checksums recorded |

Details: [methods](docs/methods.md) · [data sources and licences](docs/data.md) · [reproducibility and repository layout](docs/reproducibility.md)

## Quick start

```bash
pip install -e ".[all]"                          # or: conda env create -f environment.yml
pytest -m "not slow"
python -m knownpred.experiments.part_a           # Part A (~13 min on CPU)
python -m knownpred.data.download                # Part B data
python -m knownpred.experiments.part_b           # Part B
python -m knownpred.experiments.significance     # formal tests
python -m knownpred.plotting                     # figures and tables
```

## Next steps

Questions arising from these results (details in [docs/results.md](docs/results.md#open-questions-and-future-work)):

- Can more complex models improve the within-season ranking of confidence? Candidates: convolutional networks on full SST fields, or fine-tuned climate foundation models.
- Can conditional attribution methods avoid the spurious spread attribution seen with permutation importance?
- Does the networks' confidence in observations hold across more regions, seasons and a second initialised model?

## Relation to prior work

Independent work building on published research on forecasts of opportunity and uncertainty-aware networks in the geosciences ([Mayer & Barnes, 2021](https://doi.org/10.1029/2020GL092092); [Gordon & Barnes, 2022](https://doi.org/10.1029/2022GL098635); [Barnes, Barnes & Gordillo, 2021](https://arxiv.org/abs/2109.07250); [Mamalakis et al., 2022](https://doi.org/10.1175/AIES-D-22-0012.1)). Full references: [docs/methods.md](docs/methods.md#references). All code was written from scratch from the published method descriptions. No affiliation with or endorsement by the cited authors or data providers is implied.

## Licence and citation

MIT ([LICENSE](LICENSE)); cite via [CITATION.cff](CITATION.cff). External datasets keep their own terms ([docs/data.md](docs/data.md)).
