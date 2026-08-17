"""The display pyramid: a regular lat/lon view of an irregular grid.

The LLC rect grid is not a regular lat/lon grid, so a field cannot be
handed to ``hv.Image`` -- that would silently misplace data -- and
datashading a curvilinear ``hv.QuadMesh`` at 224 million points is far too
slow to drive an interactive map.

So the map is drawn from a **pyramid**: each field is binned once onto a
regular lat/lon raster at a few widths, and cached.  On that raster
``hv.Image`` is correct, datashader is fast, and Pacific-centring really
is just a column roll.

The pyramid is for looking at.  It never feeds a number -- every statistic
on page 1 is computed on the native grid (see
:mod:`~fronts.viz.apps.common.selection`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fronts.viz.apps import config


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------

def regrid(
    values: np.ndarray,
    XC: np.ndarray,
    YC: np.ndarray,
    width: int,
    *,
    lat_range: tuple[float, float] = config.PYRAMID_LAT_RANGE,
    reduce: str = "mean",
    fill_gaps: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin an irregular field onto a regular lat/lon raster.

    Parameters
    ----------
    values : numpy.ndarray
        2-D field on the native grid.  NaNs are ignored.
    XC, YC : numpy.ndarray
        2-D longitude / latitude, same shape as *values*.
    width : int
        Number of longitude columns.  Height is derived so cells are
        roughly square in degrees.
    lat_range : tuple of float
        Latitude extent of the raster.
    reduce : {'mean', 'max', 'any'}
        How to combine several native cells landing in one output cell.
        ``'max'`` suits integer labels; ``'any'`` suits binary masks.
    fill_gaps : bool
        Fill lone empty cells from their neighbours (``'mean'`` only).
        See :func:`_fill_isolated`.

    Returns
    -------
    lon, lat : numpy.ndarray
        1-D cell centres of the output raster.
    out : numpy.ndarray
        ``(height, width)`` raster.  Empty cells are NaN (or 0 for the
        integer reductions).
    """
    lat0, lat1 = lat_range
    height = max(int(round(width * (lat1 - lat0) / 360.0)), 2)

    lon_edges = np.linspace(-180.0, 180.0, width + 1)
    lat_edges = np.linspace(lat0, lat1, height + 1)

    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    x = ((np.asarray(XC).ravel() + 180.0) % 360.0) - 180.0
    y = np.asarray(YC).ravel()
    v = np.asarray(values).ravel()

    ix = np.clip(((x + 180.0) / 360.0 * width).astype(np.int64), 0, width - 1)
    iy = np.clip(
        ((y - lat0) / (lat1 - lat0) * height).astype(np.int64), 0, height - 1
    )

    good = np.isfinite(v) & (y >= lat0) & (y <= lat1)
    flat = iy[good] * width + ix[good]
    vg = v[good]

    if reduce == "any":
        out = np.zeros(height * width, dtype=np.uint8)
        np.maximum.at(out, flat, (vg > 0).astype(np.uint8))
        return lon, lat, out.reshape(height, width)

    if reduce == "max":
        out = np.zeros(height * width, dtype=np.int64)
        np.maximum.at(out, flat, vg.astype(np.int64))
        return lon, lat, out.reshape(height, width)

    sums = np.zeros(height * width, dtype=np.float64)
    counts = np.zeros(height * width, dtype=np.int64)
    np.add.at(sums, flat, vg)
    np.add.at(counts, flat, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, sums / counts, np.nan)
    out = out.reshape(height, width)

    if fill_gaps:
        out = _fill_isolated(out)
    return lon, lat, out.astype(np.float32)


def _fill_isolated(arr: np.ndarray) -> np.ndarray:
    """Fill lone empty cells from their neighbours.

    Even at a width the source can support, the mapping from an irregular
    grid to a regular one leaves the occasional bin with no sample -- most
    visibly as a one-pixel stripe where a row of the source happens to
    straddle a boundary.  A cell with at least three finite orthogonal
    neighbours is interior to the data, so filling it is interpolation
    rather than invention; a cell with fewer is a genuine edge (coastline,
    the polar cut-off) and is left alone.

    Display only.  Statistics never touch the pyramid.
    """
    out = arr.copy()
    empty = ~np.isfinite(out)
    if not empty.any():
        return out

    total = np.zeros_like(out)
    count = np.zeros(out.shape, dtype=np.int16)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        rolled = np.roll(out, shift, axis=axis)
        good = np.isfinite(rolled)
        total[good] += rolled[good]
        count += good

    fillable = empty & (count >= 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[fillable] = total[fillable] / count[fillable]
    return out


# --------------------------------------------------------------------------
# Pacific centring
# --------------------------------------------------------------------------

def to_pacific(lon: np.ndarray, arr: np.ndarray):
    """Roll a regular raster so longitude runs 0..360.

    Only valid on the *regridded* raster, where the grid genuinely is
    regular -- which is the reason the pyramid exists.
    """
    lon360 = lon % 360.0
    order = np.argsort(lon360)
    return lon360[order], arr[:, order]


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def usable_width(width: int, source_shape) -> int:
    """Clamp a requested pyramid width to what the source can actually fill.

    Binning is a reduction, never an interpolation: an output cell that no
    input point lands in stays NaN.  Ask for a raster finer than the source
    and most cells come back empty, so the map renders as a shredded
    stripe pattern.

    Datashader hides this -- it re-aggregates to screen resolution, so the
    holes never reach the browser -- which is exactly why the bug only
    showed up once datashader was unavailable.  Clamping fixes it at the
    source instead of relying on something downstream to paper over it.
    """
    return max(min(int(width), int(source_shape[1])), 2)


def _cache_path(key: str) -> Path:
    d = config.CACHE_DIR / "pyramid"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.npz"


def _key(date, name, width, reduce, shape) -> str:
    raw = f"{date}|{name}|{width}|{reduce}|{shape}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def level(
    provider,
    date: str,
    name: str,
    width: int,
    *,
    reduce: str = "mean",
    pacific: bool = True,
    use_cache: bool = True,
):
    """One pyramid level for a named layer, built on demand and cached.

    ``name`` is a field channel, or one of the special layers ``'__land__'``,
    ``'__fronts__'``, ``'__labels__'``.
    """
    XC, YC = provider.coords(date)
    width = usable_width(width, XC.shape)
    key = _key(date, name, width, reduce, XC.shape)
    path = _cache_path(key)

    if use_cache and path.exists():
        with np.load(path) as z:
            lon, lat, arr = z["lon"], z["lat"], z["arr"]
    else:
        values = _layer_values(provider, date, name)
        lon, lat, arr = regrid(values, XC, YC, width, reduce=reduce)
        if use_cache:
            try:
                np.savez_compressed(path, lon=lon, lat=lat, arr=arr)
            except OSError:
                pass                      # a full cache dir is not fatal

    if pacific:
        lon, arr = to_pacific(lon, arr)
    return lon, lat, arr


def _layer_values(provider, date: str, name: str) -> np.ndarray:
    if name == "__land__":
        return provider.land_mask(date).astype(np.float32)
    if name == "__fronts__":
        return provider.front_binary(date).astype(np.float32)
    if name == "__labels__":
        return provider.labels(date).astype(np.float64)
    return provider.field(date, name)


def clear_cache() -> int:
    """Delete every cached pyramid level.  Returns how many files went."""
    d = config.CACHE_DIR / "pyramid"
    if not d.exists():
        return 0
    n = 0
    for p in d.glob("*.npz"):
        p.unlink()
        n += 1
    return n
