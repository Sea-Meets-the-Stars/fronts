"""Pre-generated 3-D tiles, kept as zarr on S3.

Generating a tile costs about 15 s of dask over the raw depth stores, which
is fine once and painful on every page load.  The result is a plain
``xr.Dataset``, so it stores as zarr with no conversion::

    s3://dbof/tiles/{YYYYMMDD_HHMMSS}/{region}/{field}.zarr

The region is the name from :mod:`~fronts.viz.apps.common.regions`, lower
case with spaces as underscores, so a path is readable from the outside.

The S3 filesystem comes from the preprocessing repo's
``create_s3_filesystems`` -- the same client the zarr readers use, with the
endpoint and signing quirks already sorted out.
"""

from __future__ import annotations

import re

from fronts.viz.apps import config


def region_key(name: str) -> str:
    """``'California Current System'`` -> ``'california_current_system'``."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return slug or "region"


def path(date: str, region: str, field: str) -> str:
    """The zarr store for one (date, region, field), without a scheme."""
    return "/".join((
        config.S3_BUCKET,
        config.TILE_STORE_FOLDER,
        config.date_to_prefix(date),
        region_key(region),
        f"{field}.zarr",
    ))


def _filesystems():
    from fronts.viz.apps.common.s3source import _filesystems as fs
    return fs()


def exists(date: str, region: str, field: str) -> bool:
    _, fs_sync = _filesystems()
    return bool(fs_sync.exists(path(date, region, field) + "/zarr.json")
                or fs_sync.exists(path(date, region, field) + "/.zgroup"))


def read(date: str, region: str, field: str):
    """Open a stored tile.  Raises ``FileNotFoundError`` when absent."""
    import xarray as xr

    fs, _ = _filesystems()
    store = path(date, region, field)
    if not exists(date, region, field):
        raise FileNotFoundError(f"no stored tile at s3://{store}")
    ds = xr.open_zarr(fs.get_mapper(store))
    return ds.load()


def write(ds, date: str, region: str, field: str, *, clobber: bool = False):
    """Store a generated tile.  Returns the path written, or ``None``.

    Tile attrs carry provenance (``rect_i_start``, ``face_index``, ...)
    that the page needs to line labels up with the data, so they must
    survive the round trip -- zarr keeps them, but only if they are plain
    scalars, which is what ``_plain_attrs`` guarantees.
    """
    fs, _ = _filesystems()
    store = path(date, region, field)

    if not clobber and exists(date, region, field):
        return None

    out = ds.copy()
    out.attrs = _plain_attrs(ds.attrs)
    for name in out.variables:
        out[name].attrs = _plain_attrs(out[name].attrs)

    # A tile has a handful of variables, so consolidated metadata
    # buys little and its status differs between zarr 2 and 3.
    out.to_zarr(fs.get_mapper(store), mode="w")
    return store


def _plain_attrs(attrs) -> dict:
    """Coerce attrs to things zarr can serialise.

    ``tile_utils`` writes some attrs as 0-d numpy arrays or numpy scalars;
    zarr's JSON encoder rejects them, and the failure comes out as an
    unhelpful TypeError at write time.
    """
    import numpy as np

    out = {}
    for key, value in attrs.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist() if value.ndim else value.item()
        else:
            out[key] = value
    return out
