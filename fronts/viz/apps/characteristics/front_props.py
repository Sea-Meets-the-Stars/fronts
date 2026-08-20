"""Front-property panels: geometry and colocated statistics per front.

Where the distribution panels in :mod:`.panels` describe *grid cells*, these
describe *fronts* -- one row per labelled front, from the geometry and
colocation parquet that ``build_v5`` steps 3 and 4 produce.

Six panels:

===  ==========================================
 a   PDF of front length
 b   PDF of front orientation
 c   JPDF latitude x length
 d   JPDF latitude x orientation
 e   JPDF {field statistic} x length
 f   JPDF {field statistic} x orientation
===  ==========================================

Panels (e) and (f) need a per-front value of the selected field, which is
what the **statistic** selector picks: ``{field}_{stat}`` in the colocation
table, e.g. ``gradb2_median``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fronts.viz.apps.common.render import MPL_LOCK, new_figure
from fronts.viz.apps.common.selection import BBox, wrap180

FIGSIZE = (4.1, 3.0)
DPI = 110

#: Length is heavy-tailed enough that a linear axis is unreadable.
LOG_LENGTH = True

#: Fields whose per-front statistic is best read in log10.
LOG_FIELDS = {"gradb2", "gradtheta2", "gradsalt2", "gradrho2", "gradeta2"}


def merged_table(provider, date: str) -> pd.DataFrame:
    """Geometry joined to colocation, one row per front.

    The join is ``label == flabel``, inner -- identical to
    ``fronts.properties.viz_loaders.merge_geometry_colocation``, but done
    here rather than imported.  Importing that module executes
    ``fronts.properties.__init__``, which reaches ``fronts.llc.io`` and
    from there ``dbof.cli.zarr_to_netcdf``, so a one-line merge would drag
    the entire preprocessing repo into the web server's import path.
    """
    geom = provider.geometry(date)
    if geom.empty:
        return pd.DataFrame()

    # Colocation is step 4 and lands long after step 3.  Every geometric
    # panel -- length, orientation, latitude -- comes from the geometry
    # table alone, so a missing colocation costs the two panels that plot
    # a colocated field and nothing else.
    try:
        coloc = provider.colocation(date)
    except Exception:                                       # noqa: BLE001
        return geom
    if coloc.empty:
        return geom
    return geom.merge(coloc, left_on="label", right_on="flabel", how="inner")


def in_region(df: pd.DataFrame, box: BBox) -> pd.DataFrame:
    """Fronts whose centroid falls inside the selected box.

    A front is a single row with one centroid, so this is a point-in-box
    test -- not the cell mask used for the grid statistics.  A long front
    straddling the boundary belongs to whichever side its centroid is on,
    which is the only choice that keeps each front counted once.
    """
    if df.empty or box.is_global:
        return df

    lat = df["centroid_lat"].to_numpy(dtype=float)
    lon = wrap180(df["centroid_lon"].to_numpy(dtype=float))

    in_lat = (lat >= box.lat0) & (lat <= box.lat1)
    if box.wraps():
        in_lon = (lon >= box.lon0) | (lon <= box.lon1)
    else:
        in_lon = (lon >= box.lon0) & (lon <= box.lon1)

    return df[in_lat & in_lon]


def stat_column(df: pd.DataFrame, field: str, stat: str) -> str | None:
    """``('gradb2', 'median')`` -> ``'gradb2_median'`` if it exists."""
    col = f"{field}_{stat}"
    if col in df.columns:
        return col
    # Fall back to any statistic of the same field rather than showing
    # nothing -- the panel title says which one it used.
    for alt in ("median", "mean"):
        if f"{field}_{alt}" in df.columns:
            return f"{field}_{alt}"
    return None


def available_fields(df: pd.DataFrame) -> list[str]:
    """Field names that have at least one statistic column."""
    from fronts.viz.apps import config

    suffixes = tuple(f"_{s}" for s in config.FRONT_STATS)
    names = {c.rsplit("_", 1)[0] for c in df.columns if c.endswith(suffixes)}
    return sorted(names)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def _blank(message: str):
    fig, ax = new_figure(FIGSIZE, DPI)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8,
            wrap=True, transform=ax.transAxes, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    fig.tight_layout()
    return fig


def _clean(values, log=False):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
        v = np.log10(v)
    return v


def figure_length_pdf(df: pd.DataFrame, *, title=""):
    """(a) PDF of front length."""
    with MPL_LOCK:
        if df.empty:
            return _blank("no geometry table for this region")
        if "length_km" not in df:
            # Naming the columns present turns "nothing drew" into a
            # diagnosis: the geometry parquet is there, it just does not
            # carry the column this panel needs.
            return _blank("geometry table has no 'length_km' column; "
                          f"columns: {', '.join(map(str, df.columns[:12]))}")
        v = _clean(df["length_km"], log=LOG_LENGTH)
        if v.size == 0:
            return _blank("no fronts in this region")

        fig, ax = new_figure(FIGSIZE, DPI)
        ax.hist(v, bins=40, density=True, histtype="stepfilled",
                alpha=0.78, color="steelblue", edgecolor="none")
        ax.set_xlabel("log10(front length [km])" if LOG_LENGTH
                      else "front length [km]", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"{title}  n={v.size:,}", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        return fig


def figure_orientation_pdf(df: pd.DataFrame, *, title=""):
    """(b) PDF of front orientation.

    ``geometry.calculate_front_orientation`` returns ``abs(degrees)``, so
    the range is 0-90: 0 is north-south, 90 is east-west.
    """
    with MPL_LOCK:
        if df.empty or "orientation" not in df:
            return _blank("no geometry table for this region")
        v = _clean(df["orientation"])
        if v.size == 0:
            return _blank("no fronts in this region")

        fig, ax = new_figure(FIGSIZE, DPI)
        ax.hist(v, bins=np.linspace(0, 90, 37), density=True,
                histtype="stepfilled", alpha=0.78, color="indianred",
                edgecolor="none")
        ax.set_xlabel("orientation [deg]  (0 = N–S, 90 = E–W)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_xlim(0, 90)
        ax.set_title(f"{title}  n={v.size:,}", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        return fig


def _jpdf(ax, x, y, *, bins, xlabel, ylabel, cmap="magma", log_counts=True):
    """Shared 2-D histogram panel."""
    from matplotlib.colors import LogNorm

    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size == 0:
        ax.text(0.5, 0.5, "no fronts", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#555")
        ax.set_xticks([])
        ax.set_yticks([])
        return None, 0

    h, xe, ye = np.histogram2d(x, y, bins=bins)
    h = h.T
    norm = LogNorm(vmin=1, vmax=max(h.max(), 2)) if log_counts else None
    mesh = ax.pcolormesh(xe, ye, np.where(h > 0, h, np.nan),
                         cmap=cmap, norm=norm, shading="auto")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return mesh, x.size


def _length_axis(df):
    v = df["length_km"].to_numpy(dtype=float)
    if LOG_LENGTH:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log10(np.where(v > 0, v, np.nan)), "log10(length [km])"
    return v, "length [km]"


def figure_lat_vs(df: pd.DataFrame, which: str, *, title=""):
    """(c)/(d) JPDF of latitude against length or orientation."""
    with MPL_LOCK:
        need = "length_km" if which == "length" else "orientation"
        if df.empty or need not in df or "centroid_lat" not in df:
            return _blank("no geometry table for this region")

        lat = df["centroid_lat"].to_numpy(dtype=float)
        if which == "length":
            v, vlabel = _length_axis(df)
            vbins = 40
        else:
            v, vlabel = df["orientation"].to_numpy(dtype=float), "orientation [deg]"
            vbins = np.linspace(0, 90, 37)

        fig, ax = new_figure(FIGSIZE, DPI)
        mesh, n = _jpdf(ax, v, lat, bins=[vbins, np.linspace(-80, 80, 41)],
                        xlabel=vlabel, ylabel="latitude [deg]")
        if mesh is not None:
            fig.colorbar(mesh, ax=ax, label="fronts", pad=0.02)
        ax.set_title(f"{title}  n={n:,}", fontsize=8)
        fig.tight_layout()
        return fig


def figure_field_vs(df: pd.DataFrame, field: str, stat: str, which: str, *,
                    title=""):
    """(e)/(f) JPDF of a per-front field statistic against length/orientation."""
    with MPL_LOCK:
        need = "length_km" if which == "length" else "orientation"
        if df.empty or need not in df:
            return _blank("no geometry table for this region")

        col = stat_column(df, field, stat)
        if col is None:
            return _blank(
                f"no colocated statistic for {field!r}\n"
                "(the colocation table has no matching column)"
            )

        c = df[col].to_numpy(dtype=float)
        log = field in LOG_FIELDS
        if log:
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.log10(np.where(c > 0, c, np.nan))
        clabel = f"log10({col})" if log else col

        if which == "length":
            v, vlabel = _length_axis(df)
            vbins = 40
        else:
            v, vlabel = df["orientation"].to_numpy(dtype=float), "orientation [deg]"
            vbins = np.linspace(0, 90, 37)

        fig, ax = new_figure(FIGSIZE, DPI)
        mesh, n = _jpdf(ax, v, c, bins=[vbins, 40],
                        xlabel=vlabel, ylabel=clabel, cmap="viridis")
        if mesh is not None:
            fig.colorbar(mesh, ax=ax, label="fronts", pad=0.02)
        ax.set_title(f"{title}  n={n:,}", fontsize=8)
        fig.tight_layout()
        return fig
