""" Library-route access to the two OSN LLC4320 surface stores for one tile.

Everything here is a thin wrapper on ``dbof`` (branch ``tiles-surface-only``):
resolve the tile, pull the static grid once, and pull one hour at a time from
the ``llc_surf`` (Eta, U, V, W, Theta, Salt) and ``llc_wind`` (KPPhbl,
oceTAUX, oceTAUY, ...) kerchunk stores.  No physics lives here.

Both stores are keyed by the *OSN* iteration number (``osn_date_to_iteration``,
i.e. MIT iteration + 10368) and both decode ``time`` from
``seconds since 2011-09-10``; the loaders assert the decoded time matches the
requested timestamp so a silent off-by-72-hours cannot creep in.
"""

from datetime import datetime

import numpy as np
import xarray as xr

from dbof.tiles.tile_mapping import rect_ij_to_tile, TileInfo
from dbof.tiles.tile_utils import _tile_indexer
from dbof.llc4320_ingestion.date_iterations import osn_date_to_iteration, DATE_FMT
from dbof.llc4320_ingestion.get_raw_data import (
    get_remote_gridfile, get_remote_llc_data, get_remote_llc_wind_data)
from dbof.llc4320_ingestion.grid import ensure_comodo_attrs, set_xgcm_grid
from dbof.preprocessing.preproc_llc_core_data import process_llc4320_grid

# Tile 330 (planning §4): rect (i, j) -> face 10, j 0:720, i 2880:3600
TILE_RECT_I, TILE_RECT_J = 13320, 9720
OSN_ENDPOINT = "https://mghp.osn.xsede.org"

# What we keep from each store (coding doc §3.2)
CORE_VARS = ('Theta', 'Salt', 'U', 'V', 'W', 'Eta')
WIND_VARS = ('KPPhbl', 'oceTAUX', 'oceTAUY')


def tile_spec(i_rect: int = TILE_RECT_I, j_rect: int = TILE_RECT_J) -> TileInfo:
    """Resolve the study tile.

    Parameters
    ----------
    i_rect, j_rect : int
        Rect-grid pixel; note ``rect_ij_to_tile`` takes ``(i, j)`` in that
        order.  Defaults are tile 330.

    Returns
    -------
    TileInfo
        Face index and face-local ``j``/``i`` slices (each 720 wide).
    """
    return rect_ij_to_tile(i_rect, j_rect)


def tile_indexer(ds: xr.Dataset, tile: TileInfo) -> dict:
    """``isel`` indexer restricting ``ds`` to the tile (all horizontal dims).

    Wraps ``dbof``'s private ``_tile_indexer``.  Staggered dims (``i_g``,
    ``j_g``) take the *same* slice as the centred ones, so the tile carries
    the staggered point on its low edge only -- the high-edge derivative rim
    is invalid.
    """
    return _tile_indexer(ds, tile)


def _subset_tile(ds: xr.Dataset, tile: TileInfo) -> xr.Dataset:
    """Cut one face + tile window out of a multi-face dataset, keeping a
    length-1 ``face`` dim (the dbof operators expect ``(face, j, i)``)."""
    if ds.sizes.get('face', 1) > 1:
        ds = ds.isel(face=[tile.face_idx])
    return ds.isel(**tile_indexer(ds, tile))


def load_grid(endpoint: str = OSN_ENDPOINT, tile: TileInfo = None) -> xr.Dataset:
    """Pull the static grid, subset to the tile, and annotate for xgcm.

    Parameters
    ----------
    endpoint : str
        OSN S3 endpoint.
    tile : TileInfo, optional
        Defaults to :func:`tile_spec`.

    Returns
    -------
    xarray.Dataset
        In memory, dims ``(face: 1, j, i)`` plus ``i_g``/``j_g``; variables
        ``XC, YC, dxC, dyC, dxG, dyG, rAz, rA, Depth, hFacC, SN, CS``.
        Comodo ``axis`` / ``c_grid_axis_shift`` attrs are on the horizontal
        dims -- ``process_llc4320_grid`` calls ``reset_coords()`` which may
        drop them, so ``ensure_comodo_attrs`` runs *after* it and *before*
        any ``set_xgcm_grid``.
    """
    tile = tile_spec() if tile is None else tile
    g = process_llc4320_grid(get_remote_gridfile(endpoint))
    g_tile = _subset_tile(g, tile).compute()
    g_tile = ensure_comodo_attrs(g_tile)
    g_tile.attrs.update(
        face_index=int(tile.face_idx),
        j_face_start=int(tile.j_face_slice.start),
        i_face_start=int(tile.i_face_slice.start),
        rect_i=int(TILE_RECT_I), rect_j=int(TILE_RECT_J),
        source='OSN', endpoint=endpoint)
    return g_tile


def build_xgcm(grid_ds: xr.Dataset):
    """xgcm grid for a single tile: no face connections, ``padding='fill'``."""
    return set_xgcm_grid(grid_ds, use_connections=False)


def _finish_hour(ds: xr.Dataset, ts: str, it: int, tile: TileInfo,
                 keep: tuple, store: str, compute: bool) -> xr.Dataset:
    """Common tail of the two hourly loaders: select vars, subset, check time."""
    ds = _subset_tile(ds[list(keep)], tile)
    # time comes back as a scalar coord (upstream isel(time=0)); restore it
    # as a length-1 dim so hours concatenate along it later
    ds = ds.expand_dims('time')
    if compute:
        ds = ds.compute()
    # The store's own clock is the ground truth for the iteration conversion
    got = np.datetime64(ds['time'].values[0], 's')
    want = np.datetime64(datetime.strptime(ts, DATE_FMT), 's')
    if got != want:
        raise ValueError(
            f"{store}: requested {ts} (OSN iter {it}) but store time is {got}")
    ds.attrs.update(store=store, endpoint=OSN_ENDPOINT, iteration=int(it),
                    face_index=int(tile.face_idx), timestamp=ts)
    return ds


def load_hour(ts: str, tile: TileInfo = None, endpoint: str = OSN_ENDPOINT,
              keep: tuple = CORE_VARS, compute: bool = True) -> xr.Dataset:
    """One hour of the core surface fields from the ``llc_surf`` store.

    Parameters
    ----------
    ts : str
        Timestamp in ``dbof`` format, ``'%Y-%m-%d %H:%M:%S'`` (colons; the
        ``front_tracking`` format with underscores is *not* accepted here).
    tile : TileInfo, optional
        Defaults to :func:`tile_spec`.
    endpoint : str
    keep : tuple of str
        Variables to retain; default ``Theta, Salt, U, V, W, Eta``.
    compute : bool
        Load into memory (default).  ``False`` returns the lazy dask view.

    Returns
    -------
    xarray.Dataset
        Dims ``(time: 1, face: 1, j, i)``; ``U`` on ``i_g``, ``V`` on
        ``j_g`` (not pre-interpolated).  ``attrs['iteration']`` is the OSN
        iteration used.
    """
    tile = tile_spec() if tile is None else tile
    it = osn_date_to_iteration(ts)
    ds = get_remote_llc_data(endpoint, it, [tile.face_idx])
    return _finish_hour(ds, ts, it, tile, keep, 'llc_surf', compute)


def load_wind_hour(ts: str, tile: TileInfo = None, endpoint: str = OSN_ENDPOINT,
                   keep: tuple = WIND_VARS, compute: bool = True) -> xr.Dataset:
    """One hour of ``KPPhbl, oceTAUX, oceTAUY`` from the ``llc_wind`` store.

    Same conventions as :func:`load_hour`; the wind store uses the same OSN
    iteration numbering, so ``osn_date_to_iteration`` applies to both.
    ``keep`` may also name ``PhiBot`` and ``SIarea``.
    """
    tile = tile_spec() if tile is None else tile
    it = osn_date_to_iteration(ts)
    ds = get_remote_llc_wind_data(endpoint, it, [tile.face_idx])
    return _finish_hour(ds, ts, it, tile, keep, 'llc_wind', compute)
