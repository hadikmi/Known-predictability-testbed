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
of NCAR dataset d010074, with dimensions (Y start year, L lead month 0-23,
M member 1-20, lat, lon) and a verification time(Y, L). Only the target box
and the leads that fall in the target season are read (OPeNDAP subsetting),
or the full file is downloaded when OPeNDAP is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SMYLE_FILE = "CESM2-SMYLE_NOV.{var}.1970-2020.nc"
SMYLE_HOSTS = ("https://thredds.rda.ucar.edu", "https://tds.gdex.ucar.edu")
SMYLE_DAP = "{host}/thredds/dodsC/files/d010074/data/" + SMYLE_FILE
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


def open_smyle(var: str, local_dir: Path | None = None):
    """Open a SMYLE November-start file: local copy, else OPeNDAP on each host."""
    import xarray as xr

    if local_dir is not None:
        f = Path(local_dir) / SMYLE_FILE.format(var=var)
        if f.exists():
            return xr.open_dataset(f, **_time_coder())
    last = None
    for host in SMYLE_HOSTS:
        url = SMYLE_DAP.format(host=host, var=var)
        try:
            log.info("opening %s", url)
            return xr.open_dataset(url, **_time_coder())
        except Exception as err:  # try the next host
            log.warning("  failed: %s", err)
            last = err
    raise RuntimeError(f"could not open SMYLE {var}") from last


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


def build(cfg: dict, var: str = "PRECT", local_dir: Path | None = None) -> pd.DataFrame:
    ds = open_smyle(var, local_dir)
    reg = smyle_target(ds, var, cfg["regions"]["target_box"], cfg["season"]["target_months"])
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
        {"PRECT": (("Y", "L", "M", "lat", "lon"), data, {"units": "m/s"})},
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
