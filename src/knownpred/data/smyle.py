"""Initialised-hindcast ground truth for Part B (CESM2 SMYLE).

CESM2 SMYLE (Yeager et al., 2022, Geosci. Model Dev. 15, 6451-6493) is a set of
hindcasts with the same model as CESM2-LE, initialised from the observed state
on 1 February, May, August and November of each year 1970-2019, 20 members
each, run 24 months. The 20 members of one start date differ only by a tiny
atmospheric perturbation, so their spread is the model's own forecast
uncertainty for that particular real initial state: a perfect-model estimate
of state-dependent predictability for real years. It plays the role that the
1000-member Monte Carlo truth plays in Part A, with three differences:

* it is the truth *of the model* (CESM2), not of the real climate system;
* 20 members give a spread estimate with ~16 % sampling error (sd/sqrt(2(M-1)));
* it conditions on the full initial state, while the statistical models
  see only a handful of predictors, so it is an upper bound on what they
  could know.

Files used: the November-start collection CESM2-SMYLE_NOV.<VAR>.1970-2020.nc
of NCAR GDEX dataset d010074 (precipitation: variable PRE in mm/day), with dimensions (Y start year, L lead month 0-23,
M member 1-20, lat, lon) and a verification time(Y, L). Only the target box
and the leads that fall in the target season are read (OPeNDAP subsetting),
read through HTTP range requests (only the needed HDF5 chunks are transferred).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SMYLE_FILE = "CESM2-SMYLE_NOV.{var}.1970-2020.nc"  # precipitation: var = "PRE", mm/day
SMYLE_HOSTS = ("https://thredds.rda.ucar.edu", "https://tds.gdex.ucar.edu")
SMYLE_HTTP = "{host}/thredds/fileServer/files/d010074/data/" + SMYLE_FILE
SMYLE_CITATION = (
    "Yeager, S. G., et al. (2022): The Seasonal-to-Multiyear Large Ensemble (SMYLE) "
    "prediction system using the Community Earth System Model version 2. Geosci. Model "
    "Dev., 15, 6451-6493, doi:10.5194/gmd-15-6451-2022. Data: NCAR GDEX, CESM2 SMYLE, "
    "doi:10.26024/PWMA-RE41 (CC BY 4.0)."
)


def target_leads(ds, target_months: list[int]) -> np.ndarray:
    """Lead indices L whose verification month is in ``target_months`` of the
    calendar year after the start year (e.g. Jan-Mar after a 1 Nov start)."""
    t = ds["time"].isel(Y=0)
    months = (
        np.array([v.month for v in t.values])
        if t.dtype == object
        else pd.to_datetime(t.values).month
    )
    years = (
        np.array([v.year for v in t.values]) if t.dtype == object else pd.to_datetime(t.values).year
    )
    start = int(ds["Y"].values[0])
    sel = np.flatnonzero(np.isin(months, target_months) & (years == start + 1))
    if len(sel) != len(target_months):
        raise ValueError(f"expected {len(target_months)} target leads, found {sel.tolist()}")
    return sel


def _time_coder():
    import xarray as xr

    coders = getattr(xr, "coders", None)
    if coders is not None and hasattr(coders, "CFDatetimeCoder"):
        return {"decode_times": coders.CFDatetimeCoder(use_cftime=True)}
    return {"decode_times": True, "use_cftime": True}


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP Range requests.

    Each read is retried with back-off and rotated across mirror hosts, so a
    flaky server does not abort the extraction. Wrapped in a large
    ``io.BufferedReader``, it lets h5py/h5netcdf read only the HDF5 chunks a
    selection needs (here ~0.7 GB of a 5.4 GB file). OPeNDAP is not used
    because the server rejects these files' 64-bit integer coordinates.
    """

    def __init__(self, urls: list[str], retries: int = 8, timeout: int = 120):
        import requests

        self.urls, self.retries, self.timeout = list(urls), retries, timeout
        self.session = requests.Session()
        self.pos = 0
        self.size = None
        for u in self.urls:
            try:
                r = self.session.head(u, timeout=timeout, allow_redirects=True)
                r.raise_for_status()
                self.size = int(r.headers["Content-Length"])
                break
            except Exception as err:
                log.warning("HEAD %s failed: %s", u, err)
        if self.size is None:
            raise RuntimeError("no mirror answered")
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b):
        import time

        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        rng = f"bytes={self.pos}-{self.pos + n - 1}"
        for attempt in range(self.retries):
            url = self.urls[attempt % len(self.urls)]
            try:
                r = self.session.get(url, headers={"Range": rng}, timeout=self.timeout)
                if r.status_code != 206:
                    raise OSError(f"HTTP {r.status_code}")
                data = r.content
                if len(data) != n:
                    raise OSError(f"short read {len(data)} of {n}")
                b[:n] = data
                self.pos += n
                self.bytes_read += n
                return n
            except Exception as err:
                wait = min(60, 2**attempt)
                log.warning("  range read failed (%s); retry %d in %d s", err, attempt + 1, wait)
                time.sleep(wait)
        raise OSError(f"range read {rng} failed after {self.retries} attempts")


def download_parallel(urls: list[str], dest: Path, parts: int = 4, retries: int = 30) -> Path:
    """Download a large file in ``parts`` byte-range segments in parallel.

    Each segment is resumable and retried with back-off, rotating across
    mirror hosts; segments are concatenated and the SHA-256 is recorded in
    ``<dest>.sha256``. Used for SMYLE because the server drops connections
    and range-reading HDF5 metadata over a slow link is inefficient.
    """
    import hashlib
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = None
    for u in urls:
        try:
            r = requests.head(u, timeout=120, allow_redirects=True)
            r.raise_for_status()
            size = int(r.headers["Content-Length"])
            break
        except Exception as err:
            log.warning("HEAD %s failed: %s", u, err)
    if size is None:
        raise RuntimeError("no mirror answered")
    bounds = np.linspace(0, size, parts + 1).astype(int)

    def fetch(k: int) -> Path:
        lo, hi = int(bounds[k]), int(bounds[k + 1]) - 1
        part = dest.with_suffix(dest.suffix + f".part{k}")
        for attempt in range(retries):
            have = part.stat().st_size if part.exists() else 0
            if lo + have > hi:
                return part
            url = urls[attempt % len(urls)]
            try:
                with requests.get(
                    url, headers={"Range": f"bytes={lo + have}-{hi}"}, stream=True, timeout=120
                ) as r:
                    if r.status_code != 206:
                        raise OSError(f"HTTP {r.status_code}")
                    with open(part, "ab") as f:
                        for chunk in r.iter_content(2**20):
                            f.write(chunk)
            except Exception as err:
                wait = min(60, 2 ** min(attempt, 6))
                log.warning(
                    "  segment %d interrupted (%s); resuming in %d s", k, type(err).__name__, wait
                )
                time.sleep(wait)
        raise OSError(f"segment {k} failed after {retries} attempts")

    t0 = time.time()
    with ThreadPoolExecutor(parts) as ex:
        segs = list(ex.map(fetch, range(parts)))
    h = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as out:
        for sp in segs:
            with open(sp, "rb") as f:
                while b := f.read(2**24):
                    out.write(b)
                    h.update(b)
    if tmp.stat().st_size != size:
        raise OSError(f"size mismatch: {tmp.stat().st_size} != {size}")
    tmp.rename(dest)
    for sp in segs:
        sp.unlink()
    dest.with_suffix(dest.suffix + ".sha256").write_text(
        f"{h.hexdigest()}  {dest.name}  {urls[0]}\n"
    )
    log.info("downloaded %s (%.2f GB) in %.1f min", dest.name, size / 1e9, (time.time() - t0) / 60)
    return dest


def open_smyle(var: str, local_dir: Path | None = None):
    """Open a SMYLE November-start file: a local copy if present, otherwise
    the remote file through HTTP range requests (mirrors tried in turn)."""
    import xarray as xr

    if local_dir is not None:
        f = Path(local_dir) / SMYLE_FILE.format(var=var)
        if f.exists():
            return xr.open_dataset(f, **_time_coder())
    urls = [SMYLE_HTTP.format(host=h, var=var) for h in SMYLE_HOSTS]
    log.info("reading %s by HTTP range requests", urls[0])
    raw = RangeFile(urls)
    buf = io.BufferedReader(raw, buffer_size=8 * 2**20)
    ds = xr.open_dataset(buf, engine="h5netcdf", **_time_coder())
    ds.encoding["_range_file_obj"] = raw
    return ds


def smyle_target(ds, var: str, box: dict, target_months: list[int]):
    """Seasonal-mean, area-weighted target-box value per (start year, member).

    Returns an xarray DataArray (Y, M) in mm/day for precipitation.
    """
    from .preprocess import area_weights, select_box, standardize_coords

    leads = target_leads(ds, target_months)
    da = standardize_coords(ds[var].isel(L=leads))
    da = select_box(da, box)
    w = area_weights(da["lat"]).fillna(0)
    reg = da.weighted(w).mean(("lat", "lon")).mean("L")
    units = str(ds[var].attrs.get("units", "")).replace(" ", "").lower()
    if units in ("m/s", "ms-1") or (units == "" and float(np.abs(reg).max()) < 1e-3):
        reg = reg * 1000.0 * 86400.0
    elif units in ("mm/s", "kg/m2/s", "kgm-2s-1"):
        reg = reg * 86400.0
    reg.attrs["units"] = "mm/day"
    return reg.load()


def smyle_ground_truth(reg, forced_response: str = "quadratic") -> pd.DataFrame:
    """Per-start-year ensemble statistics of the SMYLE target.

    Anomalies are taken from the all-years, all-members hindcast climatology
    (the usual lead-dependent drift correction), the forced trend is removed
    with a polynomial of the ensemble mean over start years (as for
    observations), and the result is standardised by the climatological
    (all-years, all-members) standard deviation so that it is comparable
    with the standardised observed target.

    Columns: target_year, smyle_mean, smyle_std, smyle_std_se, members.
    """
    x = np.asarray(reg.transpose("Y", "M").values, float)  # (years, members)
    years = np.asarray(reg["Y"].values, int)
    x = x - np.nanmean(x)
    deg = {"quadratic": 2, "linear": 1, "none": 0}[forced_response]
    if deg:
        em = np.nanmean(x, axis=1)
        coef = np.polyfit(years - years.mean(), em, deg)
        x = x - np.polyval(coef, years - years.mean())[:, None]
    x = x / np.nanstd(x)
    M = np.sum(np.isfinite(x), axis=1)
    sd = np.nanstd(x, axis=1, ddof=1)
    return pd.DataFrame(
        {
            "target_year": years + 1,
            "smyle_mean": np.nanmean(x, axis=1),
            "smyle_std": sd,
            "smyle_std_se": sd / np.sqrt(2.0 * (M - 1)),
            "members": M,
        }
    ).assign(**{f"m{j + 1}": x[:, j] for j in range(x.shape[1])})


def fetch_smyle(var: str, local_dir: Path, parts: int = 4) -> Path:
    urls = [SMYLE_HTTP.format(host=h, var=var) for h in SMYLE_HOSTS]
    return download_parallel(urls, Path(local_dir) / SMYLE_FILE.format(var=var), parts=parts)


def build(cfg: dict, var: str = "PRE", local_dir: Path | None = None) -> pd.DataFrame:
    """SMYLE truth table; the regional target per (year, member) is cached in
    ``local_dir`` as ``smyle_<var>_target.nc`` so the download happens once."""
    import xarray as xr

    cache = None if local_dir is None else Path(local_dir) / f"smyle_{var}_target.nc"
    if cache is not None and cache.exists():
        reg = xr.open_dataarray(cache).load()
    else:
        if local_dir is not None and cfg.get("smyle", {}).get("download", True):
            fetch_smyle(var, local_dir)  # full file, resumable; no-op if present
        ds = open_smyle(var, local_dir)
        reg = smyle_target(ds, var, cfg["regions"]["target_box"], cfg["season"]["target_months"])
        raw = ds.encoding.get("_range_file_obj")
        if raw is not None:
            log.info("SMYLE: %.0f MB transferred", raw.bytes_read / 2**20)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            reg.attrs["source"] = SMYLE_HTTP.format(host=SMYLE_HOSTS[0], var=var)
            reg.attrs["citation"] = SMYLE_CITATION
            reg.attrs["accessed"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
            reg.to_netcdf(cache)
    return smyle_ground_truth(reg, cfg["observations"]["forced_response"])


def write_fake_smyle(path: Path, years=range(1970, 2021), members: int = 20, seed: int = 0) -> Path:
    """Synthetic file with the SMYLE layout and a planted ENSO-dependent spread
    (for tests only; no scientific meaning)."""
    import cftime
    import xarray as xr

    rng = np.random.default_rng(seed)
    years = np.asarray(list(years))
    lat = np.arange(30.0, 38.1, 1.0)
    lon = np.arange(244.0, 258.1, 1.25)
    L = np.arange(24)
    enso = rng.standard_normal(len(years))
    spread = np.exp(0.35 * enso)  # larger spread in "El Nino" years
    data = np.empty((len(years), 24, members, lat.size, lon.size), dtype="float32")
    for i in range(len(years)):
        member_anom = spread[i] * rng.standard_normal(members) + 0.5 * enso[i]
        base = 1.2 + member_anom[None, :, None, None]
        data[i] = (
            (base + 0.1 * rng.standard_normal((24, members, lat.size, lon.size))) / 86400.0 / 1000.0
        )
    time = np.empty((len(years), 24), dtype=object)
    for i, y in enumerate(years):
        for j in range(24):
            m = (11 + j - 1) % 12 + 1
            yy = y + (11 + j - 1) // 12
            time[i, j] = cftime.DatetimeNoLeap(yy, m, 15)
    ds = xr.Dataset(
        {"PRE": (("Y", "L", "M", "lat", "lon"), data, {"units": "m/s"})},
        coords={
            "Y": years,
            "L": L,
            "M": np.arange(1, members + 1),
            "lat": lat,
            "lon": lon,
            "time": (("Y", "L"), time),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    pd.Series(enso, index=years + 1, name="planted_enso").to_csv(path.with_suffix(".enso.csv"))
    return path
