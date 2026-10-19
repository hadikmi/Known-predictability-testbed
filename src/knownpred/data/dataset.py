"""Build seasonal forecast samples from preprocessed monthly fields.

One sample per (member, target year): predictors describe the state at the
end of the issue month, the target is the mean precipitation anomaly over
the target months of the following year. Example with the default
configuration: predictors from October conditions, target = January-March
mean precipitation over the U.S. Southwest box, i.e. a lead of two months
between the issue month and the first target month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def seasonal_mean(da: xr.DataArray, months: list[int]) -> xr.DataArray:
    """Mean over consecutive calendar ``months`` (may wrap the year, e.g.
    [12, 1, 2]), labelled by the year of the LAST month."""
    months = list(months)
    sel = da.sel(time=da["time"].dt.month.isin(months))
    year = sel["time"].dt.year
    if months != sorted(months):  # wraps the year end: Dec belongs to next year's season
        year = xr.where(sel["time"].dt.month > months[-1], year + 1, year)
    out = sel.groupby(year.rename("year")).mean("time")
    counts = sel["time"].groupby(year.rename("year")).count()
    return out.where(counts == len(months))  # incomplete seasons -> NaN


def month_value(da: xr.DataArray, month: int, year_offset: int = 0) -> xr.DataArray:
    """Values of one calendar month, labelled by year + ``year_offset``."""
    sel = da.sel(time=da["time"].dt.month == month)
    return (
        sel.assign_coords(year=("time", sel["time"].dt.year.values + year_offset))
        .swap_dims({"time": "year"})
        .drop_vars("time")
    )


def window_mean(
    da: xr.DataArray, end_month: int, length: int, year_offset: int = 0
) -> xr.DataArray:
    """Mean of the ``length`` months ending with ``end_month`` (same year)."""
    months = [((end_month - k - 1) % 12) + 1 for k in range(length)][::-1]
    if months != sorted(months):
        raise ValueError("antecedent window must not cross the year boundary")
    sel = da.sel(time=da["time"].dt.month.isin(months))
    out = sel.groupby(sel["time"].dt.year.rename("year")).mean("time")
    return out.assign_coords(year=out["year"] + year_offset)


def build_samples(
    pcs: xr.DataArray,
    nino34: xr.DataArray,
    precip: xr.DataArray,
    issue_month: int,
    target_months: list[int],
    antecedent_months: int = 3,
    tendency_months: int = 3,
) -> pd.DataFrame:
    """Assemble predictors and target into one table.

    Parameters
    ----------
    pcs : (time, mode[, member_id]) standardised SST principal components.
    nino34 : (time[, member_id]) Nino-3.4 anomaly index.
    precip : (time[, member_id]) regional-mean precipitation anomaly.

    The issue month belongs to the year before the target season when the
    target months start earlier in the calendar than the issue month.
    """
    offset = 1 if target_months[0] <= issue_month else 0
    earlier = (issue_month - tendency_months - 1) % 12 + 1
    earlier_offset = offset + (1 if earlier > issue_month else 0)  # crossed into the previous year
    parts = {
        "nino34": month_value(nino34, issue_month, offset),
        "nino34_tendency": month_value(nino34, issue_month, offset)
        - month_value(nino34, earlier, earlier_offset),
        "precip_antecedent": window_mean(precip, issue_month, antecedent_months, offset),
    }
    pc_issue = month_value(pcs, issue_month, offset)
    for m in pc_issue["mode"].values:
        parts[f"sst_pc{int(m)}"] = pc_issue.sel(mode=m, drop=True)
    parts["target"] = seasonal_mean(precip, target_months)
    ds = xr.Dataset(parts)
    df = ds.to_dataframe().dropna()
    keep = [c for c in df.columns if c in parts]
    return df[keep]


def split_members(
    members: np.ndarray, fractions=(0.7, 0.15, 0.15), seed: int = 0
) -> dict[str, np.ndarray]:
    """Random split of ensemble members into train / val / test sets.

    Members share the forced response (removed beforehand) but have
    independent internal variability, so a member split leaves no shared
    trajectory between sets.
    """
    rng = np.random.default_rng(seed)
    m = rng.permutation(np.asarray(members))
    n_tr = int(round(fractions[0] * len(m)))
    n_va = int(round(fractions[1] * len(m)))
    return {
        "train": np.sort(m[:n_tr]),
        "val": np.sort(m[n_tr : n_tr + n_va]),
        "test": np.sort(m[n_tr + n_va :]),
    }
