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

Measured on the cloud VM where the published results were produced (2 CPU cores, 8 GB RAM, no GPU; PyTorch 2.14 CPU, xarray 2026.9, Python 3.13).

| Step | Command | Time |
|---|---|---|
| Tests (all, incl. network tests) | `pytest` | ~20 s (more when other jobs share the CPU) |
| Part A smoke test | `python -m knownpred.experiments.part_a --quick` | ~0.5 min |
| Part A full, all six models | `python -m knownpred.experiments.part_a` | 13 min |
| Part B observations | `python -m knownpred.data.download --only obs` | ~1.5 min (1.96 GB) |
| Part B CESM2-LE extraction | `python -m knownpred.data.download --only cesm2` | 16 min (~600 Zarr chunks of 80–110 MB; ~50 GB transferred, 0.54 GB kept) |
| Part B preprocess + evaluate | `python -m knownpred.experiments.part_b` | 2 min |
| Part B pipeline check without download | `python -m knownpred.experiments.part_b --fake-data` | ~5 s |
| Figures and tables | `python -m knownpred.plotting` | ~15 s |

## What has been run

Everything in `results/` was produced by the commands above from the configurations in `configs/`. The `run_info*.json` files in `runs/` record the git commit, configuration checksum and package versions of each run.

## Data access notes

- CESM2-LE is read over HTTPS from `https://ncar-cesm2-lens.s3.us-west-2.amazonaws.com/` (no AWS credentials or `s3fs` needed). The intake-esm catalog lives at `https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json`.
- The Zarr chunks hold one member × 600 months × the full globe, so the transfer volume is set by the period and the number of members, not by the region or months kept. The configured period (1900–2050) is aligned to chunk boundaries.
- Behind a proxy, the HTTPS reader uses the standard `HTTPS_PROXY` variable (`trust_env=True`). Observation downloads resume after dropped connections (up to 10 attempts).
- Observation files change as NOAA appends months. The SHA-256 checksums in `data/raw/MANIFEST.json` identify the exact versions used; to reproduce the published numbers exactly, the observation period in `configs/part_b.yaml` ends in 2025.
