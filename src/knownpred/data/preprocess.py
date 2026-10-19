"""Preprocessing of gridded climate data with xarray.

Each function does one documented step so that it can be unit-tested on
small synthetic arrays (tests/test_preprocess.py). Conventions after
preprocessing:

* coordinates are named ``lat`` (ascending) and ``lon`` (0-360 E);
* ``time`` is labelled by the first day of the month (datetime64), so that a
  value at 1990-01-01 is the January 1990 mean in every dataset;
* temperature is in degC and precipitation in mm/day.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- coordinates


def standardize_coords(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename lat/lon variants, wrap longitude to [0, 360) and sort both."""
    ren = {}
    for name in list(obj.dims) + list(obj.coords):
        low = str(name).lower()
        if low in ("latitude", "nav_lat", "y_lat") and "lat" not in obj.coords:
            ren[name] = "lat"
        if low in ("longitude", "nav_lon", "x_lon") and "lon" not in obj.coords:
            ren[name] = "lon"
    obj = obj.rename(ren)
    if "lon" in obj.coords:
        obj = obj.assign_coords(lon=(obj["lon"] % 360.0)).sortby("lon")
    if "lat" in obj.coords:
        obj = obj.sortby("lat")
    return obj


def select_box(da: xr.DataArray, box: dict) -> xr.DataArray:
    """Select a lat/lon box given as {lat: [south, north], lon: [west, east]}
    in degrees east (0-360). Boxes crossing the prime meridian
    (west > east) are handled."""
    s, n = box["lat"]
    w, e = (x % 360.0 for x in box["lon"])
    da = da.sel(lat=slice(s, n))
    if w <= e:
        return da.sel(lon=slice(w, e))
    return xr.concat([da.sel(lon=slice(w, 360.0)), da.sel(lon=slice(0.0, e))], dim="lon")


def area_weights(lat: xr.DataArray) -> xr.DataArray:
    """cos(latitude) weights, proportional to grid-cell area on a regular grid."""
    return np.cos(np.deg2rad(lat))


def region_mean(da: xr.DataArray, box: dict | None = None) -> xr.DataArray:
    """Area-weighted mean over a box; missing values (e.g. land in SST
    fields or ocean in land fields) are excluded from the weights."""
    if box is not None:
        da = select_box(da, box)
    return da.weighted(area_weights(da["lat"]).fillna(0)).mean(dim=("lat", "lon"))


# ----------------------------------------------------------------------------- time


def _month_start_index(years, months) -> pd.DatetimeIndex:
    return pd.to_datetime({"year": np.asarray(years), "month": np.asarray(months), "day": 1})


def fix_cesm_time(ds: xr.Dataset, stamp: str = "auto") -> xr.Dataset:
    """Relabel CESM monthly output by the month it averages.

    CESM history files stamp monthly means at the end of the averaging
    interval (e.g. the January 1850 mean carries 1850-02-01 00:00). With
    ``stamp='auto'`` the midpoint of ``time_bnds``/``time_bounds`` is used
    when present; otherwise, if every stamp is at 00:00 on day 1, the
    end-of-interval convention is assumed and one month is subtracted.
    ``stamp='end'`` forces the shift, ``stamp='mid'`` takes stamps as is.
    """
    bnds_name = next((b for b in ("time_bnds", "time_bounds") if b in ds), None)
    if stamp == "auto" and bnds_name is not None:
        b = ds[bnds_name]
        bdim = [d for d in b.dims if d != "time"][0]
        mid = b.isel({bdim: 0}) + (b.isel({bdim: 1}) - b.isel({bdim: 0})) / 2
        years, months = mid.dt.year.values, mid.dt.month.values
    else:
        t = ds["time"]
        years, months = t.dt.year.values.copy(), t.dt.month.values.copy()
        at_month_start = bool(np.all(t.dt.day.values == 1) and np.all(t.dt.hour.values == 0))
        if stamp == "end" or (stamp == "auto" and at_month_start):
            if stamp == "auto":
                log.warning("no time bounds found; assuming CESM end-of-interval time stamps")
            months = months - 1
            years = np.where(months == 0, years - 1, years)
            months = np.where(months == 0, 12, months)
    out = ds.assign_coords(time=_month_start_index(years, months))
    if bnds_name is not None:
        out = out.drop_vars(bnds_name)
    if not out.indexes["time"].is_unique:
        raise ValueError("duplicate months after time relabelling; check the time convention")
    return out


def to_month_start(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Relabel monthly data (stamped anywhere inside the month) by month start."""
    t = obj["time"]
    return obj.assign_coords(time=_month_start_index(t.dt.year.values, t.dt.month.values))


# ----------------------------------------------------------------------------- units


def precip_to_mm_per_day(da: xr.DataArray) -> xr.DataArray:
    """Convert precipitation to mm/day based on its ``units`` attribute."""
    u = str(da.attrs.get("units", "")).replace(" ", "").lower()
    if u in ("m/s", "ms-1", "ms**-1", "m.s-1"):
        out = da * 1000.0 * 86400.0
    elif u in ("kgm-2s-1", "kg/m2/s", "kg/m^2/s", "kgm**-2s**-1", "mm/s"):
        out = da * 86400.0
    elif u in ("mm/day", "mmday-1", "mm/d", "mmd-1"):
        out = da
    elif u in ("mm", "millimeters", "millimetres", "mm/month"):
        # monthly totals: divide by the number of days in each month
        out = da / da["time"].dt.days_in_month
    else:
        raise ValueError(f"unrecognised precipitation units {da.attrs.get('units')!r}")
    out.attrs = {**da.attrs, "units": "mm/day"}
    return out


def temperature_to_degC(da: xr.DataArray) -> xr.DataArray:
    u = str(da.attrs.get("units", "")).replace(" ", "").lower()
    if u in ("k", "kelvin"):
        out = da - 273.15
    elif u in ("degc", "deg_c", "celsius", "degreesc", "degrees_celsius", "°c"):
        out = da
    else:
        raise ValueError(f"unrecognised temperature units {da.attrs.get('units')!r}")
    out.attrs = {**da.attrs, "units": "degC"}
    return out


# ----------------------------------------------------------------------------- anomalies


def monthly_anomalies(da: xr.DataArray, base_period: tuple[int, int]) -> xr.DataArray:
    """Anomalies relative to the calendar-month climatology of ``base_period``
    (inclusive years). With an ensemble dimension, the climatology is pooled
    over members."""
    base = da.sel(time=slice(f"{base_period[0]}-01-01", f"{base_period[1]}-12-31"))
    reduce_dims = ["time"] + [d for d in da.dims if d == "member_id"]
    clim = base.groupby("time.month").mean(reduce_dims)
    return (da.groupby("time.month") - clim).drop_vars("month")


def remove_ensemble_mean(da: xr.DataArray, member_dim: str = "member_id") -> xr.DataArray:
    """Remove the forced response estimated by the ensemble mean.

    What remains in each member is (an estimate of) internal variability.
    With M members the estimate of the forced response contains internal
    noise of variance sigma^2 / M, which also slightly damps each member's
    residual variance by a factor (M - 1) / M.
    """
    return da - da.mean(member_dim)


def remove_polynomial_trend(da: xr.DataArray, degree: int = 2, dim: str = "time") -> xr.DataArray:
    """Remove a least-squares polynomial in time, separately for each grid
    point / series. Used as the forced-response estimate for observations,
    which have only one realisation."""
    fit = da.polyfit(dim, degree, skipna=True)  # datetime coordinates are handled by xarray
    trend = xr.polyval(da[dim], fit.polyfit_coefficients)
    return da - trend


# ----------------------------------------------------------------------------- regridding


def regrid_bilinear(src: xr.DataArray, target: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Bilinear interpolation of ``src`` onto the lat/lon of ``target``.

    The source is padded periodically in longitude so that targets near
    0/360 E are interpolated rather than set missing. Bilinear interpolation
    is adequate here because the source (2 deg ERSST) is coarser than the
    target (~1 deg CESM) and fields are smooth; it is not conservative.
    """
    pad_w = src.isel(lon=slice(-2, None)).assign_coords(lon=lambda d: d["lon"] - 360.0)
    pad_e = src.isel(lon=slice(0, 2)).assign_coords(lon=lambda d: d["lon"] + 360.0)
    padded = xr.concat([pad_w, src, pad_e], dim="lon")
    return padded.interp(lat=target["lat"], lon=target["lon"], method="linear")


# ----------------------------------------------------------------------------- EOFs


class EOF:
    """Empirical orthogonal functions via SVD, with sqrt(cos lat) weighting.

    Fitted on a 2-D sample x space matrix (samples can combine time and
    ensemble members). Grid points that are missing anywhere in the fitting
    data are excluded. ``transform`` projects new fields (e.g. observations
    on the same grid) onto the fitted patterns, which is how observations are
    expressed in the model's predictor space.
    """

    def __init__(self, n_modes: int = 6):
        self.n_modes = n_modes

    def fit(self, da: xr.DataArray, sample_dims: tuple[str, ...]) -> EOF:
        X, self.space_coords_ = self._matrix(da, sample_dims)
        self.valid_ = np.all(np.isfinite(X), axis=0)
        Xv = X[:, self.valid_]
        self.mean_ = Xv.mean(0)
        _, s, vt = np.linalg.svd(Xv - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_modes]
        var = s**2
        self.explained_variance_ratio_ = var[: self.n_modes] / var.sum()
        pcs = (Xv - self.mean_) @ self.components_.T
        self.pc_std_ = pcs.std(0)
        # sign convention: positive loading-weighted mean (reproducible signs)
        sign = np.sign(self.components_.sum(1))
        sign[sign == 0] = 1
        self.components_ *= sign[:, None]
        return self

    def transform(self, da: xr.DataArray, sample_dims: tuple[str, ...]) -> np.ndarray:
        X, _ = self._matrix(da, sample_dims)
        Xv = np.where(np.isfinite(X[:, self.valid_]), X[:, self.valid_], self.mean_)
        return ((Xv - self.mean_) @ self.components_.T) / self.pc_std_

    def patterns(self) -> np.ndarray:
        """EOF patterns on the grid (unweighted), shape (n_modes, lat, lon)."""
        lat, lon, w = self.space_coords_
        full = np.full((self.n_modes, w.size), np.nan)
        full[:, self.valid_] = self.components_ / w.ravel()[self.valid_]
        return full.reshape(self.n_modes, lat.size, lon.size)

    @staticmethod
    def _matrix(da: xr.DataArray, sample_dims):
        da = da.transpose(*sample_dims, "lat", "lon")
        w = np.sqrt(area_weights(da["lat"])).values[:, None] * np.ones((1, da.sizes["lon"]))
        X = (da.values * w).reshape(-1, w.size)
        return X, (da["lat"].values, da["lon"].values, w)
