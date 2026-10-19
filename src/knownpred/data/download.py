"""Acquisition of the Part B datasets.

* Observations (ERSSTv5, nClimGrid) are downloaded once over HTTPS with
  resumable transfers; a SHA-256 checksum and the access date are recorded
  in ``data/raw/MANIFEST.json`` so that the exact files used are documented.
* CESM2-LE fields are read lazily from the public Zarr stores on AWS and
  only the needed region and period are computed and saved as netCDF.

Run as:  python -m knownpred.data.download --config configs/part_b.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from pathlib import Path

from .sources import CESM2_LENS, ERSST_V5, NCLIMGRID, DataSource

log = logging.getLogger(__name__)

CESM2_CATALOG_JSON = (
    "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json"
)


# ----------------------------------------------------------------------------- HTTPS files


def _sha256(path: Path, chunk: int = 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial ``.part`` file if present."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        mode = "ab" if pos and r.status_code == 206 else "wb"
        total = int(r.headers.get("Content-Length", 0)) + (pos if mode == "ab" else 0)
        done = pos if mode == "ab" else 0
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=2**20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (200 * 2**20) < 2**20:
                    log.info("  %s: %.0f / %.0f MB", dest.name, done / 2**20, total / 2**20)
    part.rename(dest)
    return dest


def fetch_source(src: DataSource, raw_dir: Path, filename: str) -> Path:
    """Download the first reachable URL of a source and record it in the manifest."""
    dest = raw_dir / filename
    manifest_path = raw_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if dest.exists() and src.key in manifest:
        log.info("%s already present (%s)", dest.name, manifest[src.key]["sha256"][:12])
        return dest
    last_err = None
    for url in src.urls:
        try:
            log.info("downloading %s from %s", src.key, url)
            for attempt in range(1, 11):  # resume after dropped connections
                try:
                    download_file(url, dest)
                    break
                except Exception as err:
                    if attempt == 10:
                        raise
                    log.warning(
                        "  attempt %d interrupted (%s); resuming", attempt, type(err).__name__
                    )
            manifest[src.key] = {
                "file": filename,
                "url": url,
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
                "accessed": date.today().isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            return dest
        except Exception as err:  # try the next mirror
            log.warning("  failed: %s", err)
            last_err = err
    raise RuntimeError(f"could not download {src.key}") from last_err


# ----------------------------------------------------------------------------- CESM2-LE


def cesm2_catalog():
    """The intake-esm catalog table of the CESM2-LE Zarr stores (pandas)."""
    import pandas as pd
    import requests

    meta = requests.get(CESM2_CATALOG_JSON, timeout=60).json()
    csv_url = meta.get("catalog_file") or meta.get("catalog_file_url")
    if csv_url is None:
        raise RuntimeError("catalog JSON has no 'catalog_file' entry")
    if not csv_url.startswith("http"):
        csv_url = CESM2_CATALOG_JSON.rsplit("/", 1)[0] + "/" + csv_url
    return pd.read_csv(csv_url)


def find_store(
    catalog, variable: str, experiment: str, forcing_variant: str, frequency: str = "monthly"
) -> str:
    sel = catalog[
        (catalog["variable"] == variable)
        & (catalog["experiment"] == experiment)
        & (catalog["forcing_variant"] == forcing_variant)
        & (catalog["frequency"] == frequency)
    ]
    if len(sel) != 1:
        raise LookupError(
            f"{len(sel)} stores for {variable}/{experiment}/{forcing_variant}/{frequency}"
        )
    return sel["path"].iloc[0]


S3_HTTPS = "https://ncar-cesm2-lens.s3.us-west-2.amazonaws.com/"


def https_path(path: str) -> str:
    """Public S3 path -> HTTPS URL (works without AWS credentials or boto)."""
    return path.replace("s3://ncar-cesm2-lens/", S3_HTTPS)


def open_cesm2(variable: str, experiments, forcing_variant: str, catalog=None):
    """Lazily open one CESM2-LE variable, concatenating experiments in time.

    The stores are read over HTTPS with ``trust_env=True`` so that standard
    proxy settings (HTTPS_PROXY) are honoured.
    """
    import xarray as xr

    catalog = catalog if catalog is not None else cesm2_catalog()
    parts = []
    for exp in experiments:
        path = https_path(find_store(catalog, variable, exp, forcing_variant))
        log.info("opening %s", path)
        parts.append(
            xr.open_zarr(
                path, consolidated=True, storage_options={"client_kwargs": {"trust_env": True}}
            )
        )
    ds = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override")
    return ds


def months_needed(season: dict) -> dict[str, list[int]]:
    """Calendar months each CESM2 variable is needed for (see data.dataset.build_samples).

    SST: the issue month and the start of the Nino-3.4 tendency window.
    Precipitation: the antecedent window and the target season.
    Downloading only these months reduces the transfer roughly threefold.
    """
    issue = season["issue_month"]
    earlier = (issue - season["tendency_months"] - 1) % 12 + 1
    antecedent = [((issue - k - 1) % 12) + 1 for k in range(season["antecedent_months"])]
    return {
        "ts": sorted({issue, earlier}),
        "precip": sorted(set(antecedent) | set(season["target_months"])),
    }


def extract_cesm2(cfg: dict, out_dir: Path) -> dict[str, Path]:
    """Extract regional, month-subset CESM2-LE TS and precipitation to netCDF.

    The Zarr chunks hold one member x 600 months x the full globe, so every
    chunk in the requested period is downloaded in full; only the subset is
    kept. Members are processed one at a time and saved to ``parts/`` so an
    interrupted extraction resumes where it stopped and memory stays small.
    """
    import dask
    import xarray as xr

    from .preprocess import fix_cesm_time, select_box, standardize_coords

    c = cfg["cesm2"]
    months = months_needed(cfg["season"])
    catalog = cesm2_catalog()
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    paths = {}
    jobs = {
        "ts": (["TS"], cfg["regions"]["sst_domain"]),
        "precip": (c["precip_variables"], cfg["regions"]["target_box"]),
    }
    for name, (variables, box) in jobs.items():
        dest = out_dir / f"cesm2le_{c['forcing_variant']}_{name}.nc"
        if dest.exists():
            log.info("%s exists, skipping", dest.name)
            paths[name] = dest
            continue
        lazy = {}
        for v in variables:
            ds = fix_cesm_time(open_cesm2(v, c["experiments"], c["forcing_variant"], catalog))
            arr = standardize_coords(ds[v])
            arr = select_box(arr, box).sel(
                time=slice(f"{c['years'][0]}-01-01", f"{c['years'][1]}-12-31")
            )
            lazy[v] = arr.sel(time=arr["time"].dt.month.isin(months[name]))
        members = list(lazy[variables[0]]["member_id"].values)
        if c.get("members"):
            members = members[: c["members"]]
        files = []
        for i, m in enumerate(members):
            f = parts_dir / f"{name}_{m}.nc"
            if not f.exists():
                with dask.config.set(scheduler="threads", num_workers=c.get("workers", 4)):
                    sub = sum(lazy[v].sel(member_id=m) for v in variables).load()  # PRECC + PRECL
                sub.name = name
                sub.to_netcdf(f.with_suffix(".tmp"))
                f.with_suffix(".tmp").rename(f)
            files.append(f)
            log.info("  %s member %d/%d (%s) done", name, i + 1, len(members), m)
        da = xr.concat([xr.open_dataarray(f) for f in files], dim="member_id").load()
        da.name = name
        da.attrs["units"] = lazy[variables[0]].attrs.get("units", "")
        da.attrs["source"] = (
            f"CESM2-LE {c['forcing_variant']} {'+'.join(variables)}; " + CESM2_LENS.citation
        )
        da.to_netcdf(dest)
        log.info("wrote %s %s", dest.name, dict(da.sizes))
        paths[name] = dest
    return paths


def main(argv=None) -> None:
    from ..experiments.common import load_config, setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/part_b.yaml")
    ap.add_argument("--only", choices=["obs", "cesm2"], default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw"])
    if args.only in (None, "obs"):
        fetch_source(ERSST_V5, raw, "ersstv5_sst.mnmean.nc")
        fetch_source(NCLIMGRID, raw, "nclimgrid_prcp.nc")
    if args.only in (None, "cesm2"):
        extract_cesm2(cfg, Path(cfg["paths"]["interim"]))


if __name__ == "__main__":
    main()
