"""Small synthetic stand-ins for the Part B input files.

They have the same structure as the real files (names, dimensions,
coordinate conventions, units, CESM end-of-month time stamps already fixed
as in the extract step, land masked as NaN in SST) and a planted ENSO
teleconnection, so the whole Part B pipeline can be exercised in CI and on
a laptop without downloading anything. They are not data and no result
obtained from them means anything about the real climate system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _enso_index(n_months: int, rng: np.random.Generator) -> np.ndarray:
    """AR(2) index with a ~4-year quasi-period and unit variance."""
    r, period = 0.97, 48.0
    a1, a2 = 2 * r * np.cos(2 * np.pi / period), -(r**2)
    x = np.zeros(n_months + 240)
    for t in range(2, len(x)):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + rng.standard_normal()
    x = x[240:]
    return x / x.std()


def _pattern(lat, lon):
    return np.exp(-((lon[None, :] - 230.0) ** 2) / (2 * 30.0**2) - lat[:, None] ** 2 / (2 * 8.0**2))


def _jfm_response(t: pd.DatetimeIndex, enso: np.ndarray, rng, scale: float) -> np.ndarray:
    """Precipitation anomaly: JFM responds to the preceding October index."""
    out = rng.normal(scale=scale, size=len(t))
    oct_val = {y: enso[i] for i, (y, m) in enumerate(zip(t.year, t.month, strict=True)) if m == 10}
    for i, (y, m) in enumerate(zip(t.year, t.month, strict=True)):
        if m in (1, 2, 3) and (y - 1) in oct_val:
            out[i] += 0.6 * scale * oct_val[y - 1]
    return out


def write_fake_inputs(
    raw: Path,
    interim: Path,
    n_members: int = 12,
    years=(1880, 2000),
    obs_years=(1895, 2000),
    seed=0,
) -> None:
    import xarray as xr

    rng = np.random.default_rng(seed)
    raw.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    # --- CESM2-LE-like extracts on a ~4 deg grid
    t = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-01", freq="MS")
    lat, lon = np.arange(-30.0, 30.1, 4.0), np.arange(120.0, 290.1, 5.0)
    pat = _pattern(lat, lon)
    seasonal = 1.5 * np.cos(2 * np.pi * (t.month.to_numpy() - 3) / 12)
    forced = 0.008 * np.maximum(t.year.to_numpy() - 1950, 0)
    ts = np.empty((n_members, len(t), lat.size, lon.size), dtype="float32")
    pr = np.empty((n_members, len(t), 3, 4), dtype="float32")
    for m in range(n_members):
        e = _enso_index(len(t), rng)
        ts[m] = (
            300.0
            + (seasonal + forced)[:, None, None]
            + e[:, None, None] * pat
            + 0.3 * rng.standard_normal((len(t), lat.size, lon.size))
        )
        p_mm = (
            1.0
            + 0.3 * np.cos(2 * np.pi * (t.month.to_numpy() - 1) / 12)
            + _jfm_response(t, e, rng, 0.4)
        )
        pr[m] = np.maximum(p_mm, 0.0)[:, None, None] / 86400.0 / 1000.0  # mm/day -> m/s
    members = [f"r{m + 1}i1001p1f1" for m in range(n_members)]
    land = (lon[None, :] > 275) & (lat[:, None] > 8)  # a 'Central America' corner
    ts[:, :, land] = 290.0  # land surface temperature present in TS but masked via ERSST
    xr.DataArray(
        ts,
        dims=("member_id", "time", "lat", "lon"),
        coords={"member_id": members, "time": t, "lat": lat, "lon": lon},
        name="ts",
        attrs={"units": "K"},
    ).to_netcdf(interim / "cesm2le_cmip6_ts.nc")
    xr.DataArray(
        pr,
        dims=("member_id", "time", "lat", "lon"),
        coords={
            "member_id": members,
            "time": t,
            "lat": [32.0, 34.0, 36.0],
            "lon": [246.0, 249.0, 252.0, 255.0],
        },
        name="precip",
        attrs={"units": "m/s"},
    ).to_netcdf(interim / "cesm2le_cmip6_precip.nc")

    # --- ERSSTv5-like: 4 deg, 0-356 E, mid-month stamps, land NaN, degC
    to = pd.date_range(f"{obs_years[0]}-01-01", f"{obs_years[1]}-12-01", freq="MS") + pd.Timedelta(
        days=14
    )
    olat, olon = np.arange(-40.0, 40.1, 4.0), np.arange(0.0, 359.0, 4.0)
    e = _enso_index(len(to), rng)
    sst = (
        27.0
        + 1.5 * np.cos(2 * np.pi * (to.month.to_numpy() - 3) / 12)[:, None, None]
        + e[:, None, None] * _pattern(olat, olon)
    )
    sst = (
        sst
        + 0.3 * rng.standard_normal(sst.shape)
        + 0.006 * np.maximum(to.year.to_numpy() - 1950, 0)[:, None, None]
    )
    sst[:, (olon[None, :] > 275) & (olat[:, None] > 8)] = np.nan
    xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst.astype("float32"), {"units": "degC"})},
        coords={"time": to, "lat": olat[::-1], "lon": olon},
    ).isel(lat=slice(None, None, -1)).to_netcdf(raw / "ersstv5_sst.mnmean.nc")

    # --- nClimGrid-like: 0.5 deg over the Southwest, longitudes in -180..180, monthly totals in mm
    tn = pd.date_range(f"{obs_years[0]}-01-01", f"{obs_years[1]}-12-01", freq="MS")
    nlat, nlon = np.arange(30.0, 38.01, 0.5), np.arange(-117.0, -101.9, 0.5)
    e_m = e  # same index as the fake ERSST, aligned in time
    p = np.maximum(1.0 + _jfm_response(tn, e_m, rng, 0.4), 0.0) * tn.days_in_month.to_numpy()
    prcp = p[:, None, None] * (1 + 0.1 * rng.standard_normal((len(tn), nlat.size, nlon.size)))
    xr.Dataset(
        {
            "prcp": (
                ("time", "lat", "lon"),
                np.maximum(prcp, 0).astype("float32"),
                {"units": "millimeters"},
            )
        },
        coords={"time": tn, "lat": nlat, "lon": nlon},
    ).to_netcdf(raw / "nclimgrid_prcp.nc")
