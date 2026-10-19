# data/

Nothing in this folder is committed except this file (see `.gitignore`).

Part A needs no data: it simulates everything.

Part B creates:

| Folder | Contents | Created by |
|---|---|---|
| `raw/` | `ersstv5_sst.mnmean.nc`, `nclimgrid_prcp.nc`, `MANIFEST.json` (URL, access date, size, SHA-256 of every file) | `python -m knownpred.data.download` |
| `interim/` | regional CESM2-LE extracts (`cesm2le_cmip6_ts.nc`, `cesm2le_cmip6_precip.nc`) | same, reading the public Zarr stores on AWS |
| `processed/` | sample tables, EOF patterns, the member split | `python -m knownpred.experiments.part_b --stage preprocess` |
| `fake/` | synthetic stand-in files for pipeline tests (no scientific meaning) | `--fake-data` |

Sources, licences and citations: [`docs/data.md`](../docs/data.md).
