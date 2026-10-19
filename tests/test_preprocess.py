"""Unit tests for the xarray preprocessing steps (skipped without xarray)."""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from knownpred.data import preprocess as pp  # noqa: E402
from knownpred.data.dataset import build_samples, seasonal_mean, split_members  # noqa: E402


def _field(
    nt=24, lats=(-10.0, 0.0, 10.0), lons=(0.0, 90.0, 180.0, 270.0), start="2000-01-01", seed=0
):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(nt, len(lats), len(lons))),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range(start, periods=nt, freq="MS"),
            "lat": list(lats),
            "lon": list(lons),
        },
    )


def test_standardize_coords_renames_wraps_and_sorts():
    da = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("latitude", "longitude"),
        coords={"latitude": [10.0, -10.0], "longitude": [-90.0, 0.0, 90.0]},
    )
    out = pp.standardize_coords(da)
    assert list(out.dims) == ["lat", "lon"]
    np.testing.assert_array_equal(out["lat"], [-10, 10])
    np.testing.assert_array_equal(out["lon"], [0, 90, 270])
    # value at (lat=10, lon=-90) must follow its coordinates
    assert float(out.sel(lat=10, lon=270)) == float(da.sel(latitude=10, longitude=-90))


def test_select_box_across_prime_meridian():
    da = _field()
    box = pp.select_box(da, {"lat": [-10, 10], "lon": [260, 10]})
    np.testing.assert_array_equal(box["lon"], [270, 0])


def test_region_mean_area_weighting():
    lats = (0.0, 60.0)
    da = xr.DataArray(
        np.array([[[1.0], [3.0]]]),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000", periods=1, freq="MS"),
            "lat": list(lats),
            "lon": [0.0],
        },
    )
    w = np.cos(np.deg2rad(np.array(lats)))
    assert pp.region_mean(da).item() == pytest.approx((1 * w[0] + 3 * w[1]) / w.sum())
    # missing values are excluded from the weights
    da2 = da.copy()
    da2[0, 1, 0] = np.nan
    assert pp.region_mean(da2).item() == pytest.approx(1.0)


def test_fix_cesm_time_end_stamps_and_bounds():
    stamps = pd.date_range("1850-02-01", periods=14, freq="MS")  # CESM: Jan mean stamped Feb 1
    ds = xr.Dataset({"x": ("time", np.arange(14.0))}, coords={"time": stamps})
    out = pp.fix_cesm_time(ds)
    assert out.indexes["time"][0] == pd.Timestamp("1850-01-01")
    assert out.indexes["time"][-1] == pd.Timestamp("1851-02-01")
    # with bounds the midpoint decides, whatever the stamp
    lo = pd.date_range("1850-01-01", periods=14, freq="MS")
    hi = pd.date_range("1850-02-01", periods=14, freq="MS")
    ds_b = ds.assign(time_bnds=(("time", "nbnd"), np.stack([lo.values, hi.values], axis=1)))
    out_b = pp.fix_cesm_time(ds_b)
    assert out_b.indexes["time"][0] == pd.Timestamp("1850-01-01")
    assert "time_bnds" not in out_b


def test_precip_unit_conversions():
    t = pd.date_range("2001-01-01", periods=3, freq="MS")  # Jan (31 d), Feb (28 d), Mar
    ms = xr.DataArray(np.full(3, 1e-8), dims="time", coords={"time": t}, attrs={"units": "m/s"})
    np.testing.assert_allclose(pp.precip_to_mm_per_day(ms), 1e-8 * 1000 * 86400)
    tot = xr.DataArray(
        np.array([31.0, 28.0, 62.0]),
        dims="time",
        coords={"time": t},
        attrs={"units": "millimeters"},
    )
    np.testing.assert_allclose(pp.precip_to_mm_per_day(tot), [1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        pp.precip_to_mm_per_day(tot.assign_attrs(units="furlongs"))


def test_temperature_conversion():
    k = xr.DataArray([273.15, 300.0], dims="x", attrs={"units": "K"})
    np.testing.assert_allclose(pp.temperature_to_degC(k), [0.0, 26.85])


def test_monthly_anomalies_remove_climatology():
    t = pd.date_range("2000-01-01", periods=48, freq="MS")
    seasonal = np.tile(np.arange(12.0), 4)
    da = xr.DataArray(seasonal + 5.0, dims="time", coords={"time": t})
    np.testing.assert_allclose(pp.monthly_anomalies(da, (2000, 2003)), 0.0, atol=1e-12)


def test_remove_ensemble_mean_and_polynomial_trend():
    t = pd.date_range("1950-01-01", periods=120, freq="MS")
    x = (t - t[0]).days.to_numpy() / 365.25  # years since start (uneven month lengths)
    trend = 0.1 * x + 0.02 * x**2
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(5, 120))
    da = xr.DataArray(
        trend + noise, dims=("member_id", "time"), coords={"member_id": list("abcde"), "time": t}
    )
    resid = pp.remove_ensemble_mean(da)
    np.testing.assert_allclose(resid.mean("member_id"), 0.0, atol=1e-12)
    one = xr.DataArray(trend, dims="time", coords={"time": t})
    np.testing.assert_allclose(pp.remove_polynomial_trend(one, 2), 0.0, atol=1e-6)


def test_regrid_bilinear_reproduces_linear_field_and_wraps_longitude():
    lats, lons = np.arange(-20.0, 21.0, 4.0), np.arange(0.0, 360.0, 4.0)
    src = xr.DataArray(
        np.add.outer(lats * 0.5, np.zeros_like(lons)),
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )
    tgt = xr.Dataset(
        coords={"lat": np.arange(-18.0, 19.0, 1.0), "lon": np.array([1.0, 179.5, 358.5])}
    )
    out = pp.regrid_bilinear(src, tgt)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, np.add.outer(tgt["lat"].values * 0.5, np.zeros(3)), atol=1e-10)


def test_eof_recovers_planted_pattern():
    rng = np.random.default_rng(1)
    lats, lons = np.linspace(-20, 20, 9), np.linspace(140, 260, 13)
    pattern = np.exp(-((lons[None, :] - 220) ** 2) / 800 - lats[:, None] ** 2 / 100)
    amp = rng.normal(size=400)
    data = amp[:, None, None] * pattern + 0.05 * rng.normal(size=(400, 9, 13))
    da = xr.DataArray(
        data, dims=("time", "lat", "lon"), coords={"time": np.arange(400), "lat": lats, "lon": lons}
    )
    eof = pp.EOF(3).fit(da, ("time",))
    assert eof.explained_variance_ratio_[0] > 0.9
    p1 = eof.patterns()[0]
    assert abs(np.corrcoef(p1.ravel(), pattern.ravel())[0, 1]) > 0.99
    pcs = eof.transform(da, ("time",))
    assert abs(np.corrcoef(pcs[:, 0], amp)[0, 1]) > 0.99
    np.testing.assert_allclose(pcs.std(0), 1.0, rtol=1e-6)


def test_seasonal_mean_labels_wrapping_season_by_last_month():
    t = pd.date_range("2000-01-01", periods=36, freq="MS")
    da = xr.DataArray(np.arange(36.0), dims="time", coords={"time": t})
    djf = seasonal_mean(da, [12, 1, 2])
    # DJF 2001 = Dec 2000 (11), Jan 2001 (12), Feb 2001 (13)
    assert float(djf.sel(year=2001)) == pytest.approx(12.0)
    assert np.isnan(float(djf.sel(year=2000)))  # incomplete (no Dec 1999)


def test_build_samples_aligns_issue_month_with_next_year_target():
    years = np.arange(1950, 1960)
    t = pd.date_range("1950-01-01", periods=120, freq="MS")
    nino = xr.DataArray(
        np.where(t.month == 10, t.year - 1949.0, 0.0), dims="time", coords={"time": t}
    )
    # target = JFM mean equals the previous October's Nino value
    prev_oct = np.where(t.month <= 3, t.year - 1950.0, 0.0)
    precip = xr.DataArray(prev_oct, dims="time", coords={"time": t})
    pcs = xr.DataArray(
        np.zeros((120, 2)), dims=("time", "mode"), coords={"time": t, "mode": [1, 2]}
    )
    df = build_samples(pcs, nino, precip, issue_month=10, target_months=[1, 2, 3])
    assert list(df.index) == list(years[1:])  # 1951..1959
    np.testing.assert_allclose(df["target"], df["nino34"])


def test_split_members_disjoint_and_complete():
    s = split_members(np.arange(50), seed=3)
    allm = np.concatenate(list(s.values()))
    assert sorted(allm) == list(range(50))
    assert len(s["train"]) == 35 and len(s["val"]) == 8 or len(s["val"]) == 7
