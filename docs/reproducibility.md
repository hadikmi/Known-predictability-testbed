# Reproducibility

## Environment

```bash
conda env create -f environment.yml && conda activate knownpred
# or
pip install -e ".[all]"          # PyTorch: install the build for your CUDA version first
```

Every run writes `run_info.json` next to its outputs, with the configuration, its SHA-256 checksum, the git commit (and whether the working tree had uncommitted changes), the package versions and the platform.

## Seeds

Each random stream gets its own seed, derived from the master seed in the configuration file and a component name (`experiments.common.derive_seed`). Examples: the training simulation, the ground truth for lead 3, the gradient-boosting model for lead 6. Changing one component therefore leaves the data and seeds of all others unchanged. PyTorch models are seeded per member; on GPU, cuDNN kernels can still introduce run-to-run differences at the level of floating-point round-off.

## Commands and run times

| Step | Command | CPU time (2-core laptop-class VM) |
|---|---|---|
| Tests (fast) | `pytest -m "not slow"` | ~1.5 min |
| Part A smoke test | `python -m knownpred.experiments.part_a --quick` | ~0.5 min (baselines) |
| Part A full | `python -m knownpred.experiments.part_a` | 7 min for the non-neural models; the networks add one training per lead, scenario and training-length repeat (31 runs of each network type). Not yet timed: see the note below. |
| Part B data | `python -m knownpred.data.download` | dominated by download bandwidth |
| Part B | `python -m knownpred.experiments.part_b` | minutes |
| Part B pipeline check | `python -m knownpred.experiments.part_b --fake-data` | ~5 s |
| Figures and tables | `python -m knownpred.plotting` | ~10 s |

## What has been run, and where

| Component | Status |
|---|---|
| Toy system, ground truth, metrics, non-neural baselines, attribution | run and tested; results in `results/` |
| PyTorch models (Gaussian / SHASH networks, deep ensemble, integrated gradients) | written and covered by tests (`tests/test_torch.py`), but **not yet executed**: PyTorch could not be installed in the environment where the repository was built. The first run happens on the author's workstation. The neural-network rows of `results/tables/part_a_metrics.csv` appear after that run. |
| Part B real-data pipeline | every step verified end to end on synthetic stand-in files with the real structure (`tests/test_part_b_pipeline.py`, `--fake-data`). Not yet run on the real CESM2-LE / ERSSTv5 / nClimGrid files. |

This table is updated as each component is run.
