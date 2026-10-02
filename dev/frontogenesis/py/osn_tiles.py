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

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

import dbof
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

# The twelve grid vars process_llc4320_grid keeps (coding doc §3.1) ...
CORE_GRID_VARS = ('XC', 'YC', 'dxC', 'dyC', 'dxG', 'dyG', 'rA', 'rAz', 'CS', 'SN',
                  'hFacC', 'Depth')
# ... plus the pieces it drops but §3.1 wants: the staggered
# land fractions (the U/V masks) and the top-cell vertical scalars (0-d in
# the OSN gridfile: drF=1.0, Z=-0.5, Zl=0.0)
GRID_EXTRA_VARS = ('hFacW', 'hFacS', 'drF', 'Z', 'Zl')

# Face 10 is rotated: CS=0, SN=-1 (M0 task 3).  write_grid asserts this
# before recording it, so the attr cannot outlive a change of tile.
ORIENTATION = ('CS=0, SN=-1: j/V/dyC zonal (eastward), i/U/dxC meridional '
               '(i increasing southward); u_east=V, v_north=-U')

# All products go under dev/frontogenesis/data/ (coding doc §3)
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


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
    the staggered point on its low edge only.  With xgcm ``padding='fill'``
    (fill value 0) the derivative rim is then invalid -- and *finite* -- on
    all four tile edges: G one cell everywhere (low edges ~1e6x from
    differencing against 0, high edges ~0.5x from interpolating with 0), the
    Jacobian one cell on the low edges and two on the high (M0 task 5).
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
        In memory, dims ``(face: 1, j, i)`` plus ``i_g``/``j_g``; the twelve
        ``CORE_GRID_VARS`` plus ``hFacW(j, i_g)``, ``hFacS(j_g, i)`` and the
        0-d ``drF, Z, Zl`` (``GRID_EXTRA_VARS``, cut from the raw gridfile
        with the same tile indexer); the §3.1 attrs (tile location,
        orientation, spacing at 37N, ``land_fill``).
        Comodo ``axis`` / ``c_grid_axis_shift`` attrs are on the horizontal
        dims -- ``process_llc4320_grid`` calls ``reset_coords()`` which may
        drop them, so ``ensure_comodo_attrs`` runs *after* it and *before*
        any ``set_xgcm_grid``.
    """
    tile = tile_spec() if tile is None else tile
    raw = get_remote_gridfile(endpoint)
    g = process_llc4320_grid(raw)
    # everything in the raw gridfile arrives as a coordinate; promote the
    # extras to data variables so they merge like the twelve processed ones
    extra = raw.reset_coords()[list(GRID_EXTRA_VARS)]
    g_tile = xr.merge([_subset_tile(g, tile), _subset_tile(extra, tile)]).compute()
    g_tile = ensure_comodo_attrs(g_tile)
    # XC/YC are coords, as in the raw gridfile and the hourly stores
    # (process_llc4320_grid's reset_coords() demotes them); otherwise a
    # plain xr.merge([hour, grid]) raises MergeError (M0 task 4/5)
    g_tile = g_tile.set_coords(['XC', 'YC'])
    # spacing at 37N, from the data (task-3 log: 1.71 x 1.85 km).  Masked
    # positionally: dxC/dyC sit on the staggered dims, and an xarray
    # ``where`` against a (j, i) band would broadcast to 4-D instead
    band = (g_tile.YC.values >= 36.5) & (g_tile.YC.values <= 37.5)
    dx_km = float(np.median(g_tile.dxC.values[band])) / 1e3
    dy_km = float(np.median(g_tile.dyC.values[band])) / 1e3
    g_tile.attrs.update(
        face_index=int(tile.face_idx),
        j_face_start=int(tile.j_face_slice.start),
        i_face_start=int(tile.i_face_slice.start),
        rect_i=int(TILE_RECT_I), rect_j=int(TILE_RECT_J),
        source='OSN', endpoint=endpoint, orientation=ORIENTATION,
        dx_km_37N=round(dx_km, 2), dy_km_37N=round(dy_km, 2), land_fill='NaN')
    return g_tile


def _git_commit(path) -> str:
    """Short commit of the repo containing ``path`` ('+dirty' if modified)."""
    run = lambda *a: subprocess.run(['git', '-C', str(path), *a], capture_output=True,
                                    text=True, check=True).stdout.strip()
    sha = run('rev-parse', '--short', 'HEAD')
    return sha + ('+dirty' if run('status', '--porcelain') else '')


def _provenance() -> dict:
    """Attrs every product carries: fronts and dbof commits, creation time."""
    return dict(git_commit=_git_commit(Path(__file__).parent),
                dbof_commit=_git_commit(Path(dbof.__file__).parent),
                created=datetime.now(timezone.utc).isoformat(timespec='seconds'))


def _drop_face(ds: xr.Dataset) -> xr.Dataset:
    """Stored layout is ``(j, i)`` / ``(time, j, i)`` (coding doc §1.2, §3):
    the length-1 ``face`` dim becomes a scalar coord, which ``expand_dims``
    restores for the dbof operators."""
    return ds.squeeze('face') if 'face' in ds.dims else ds


def _clean_encoding(ds: xr.Dataset, time_chunk: bool = False) -> dict:
    """Strip the kerchunk source encoding (its chunks still carry the face
    dim) and chunk each variable as one (720, 720) slab, per hour if there
    is a time dim -- the append unit for the series pull."""
    enc = {}
    for name, var in ds.variables.items():
        var.encoding = {}
        if name in ds.data_vars and var.ndim:
            enc[name] = {'chunks': tuple(1 if d == 'time' else n
                                         for d, n in zip(var.dims, var.shape))}
    if time_chunk and 'time' in ds:
        enc['time'] = {'units': 'seconds since 2011-09-10', 'dtype': 'int64'}
    return enc


def _out_path(out, default_name: str, clobber: bool) -> Path:
    if out is None and default_name is None:
        raise ValueError('an output path is required')
    out = DATA_DIR / default_name if out is None else Path(out)
    if out.exists() and not clobber:
        raise FileExistsError(f'{out} exists; pass clobber=True to overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_grid(grid_ds: xr.Dataset = None, out=None, clobber: bool = False) -> str:
    """Write the static tile grid (coding doc §3.1) to ``tile330_grid.zarr``.

    Parameters
    ----------
    grid_ds : xarray.Dataset, optional
        From :func:`load_grid` (pulled if omitted).
    out : path-like, optional
        Default ``DATA_DIR / 'tile330_grid.zarr'``.
    clobber : bool

    Returns
    -------
    str
        Path written.  Dims ``(j, i)`` + ``i_g``/``j_g``, ``face`` a scalar
        coord, ``XC``/``YC`` coords (not data vars, so an hour merges with
        the grid without ``MergeError``); comodo attrs on the four horizontal
        dims; provenance attrs.
    """
    g = load_grid() if grid_ds is None else grid_ds
    missing = set(CORE_GRID_VARS + GRID_EXTRA_VARS) - set(g.variables)
    if missing:
        raise ValueError(f'grid is missing {sorted(missing)}')
    # the orientation attr is a statement about this face; check it
    if not (np.all(g.SN.values == -1) and np.all(np.abs(g.CS.values) < 1e-6)):
        raise ValueError('grid is not CS=0, SN=-1; ORIENTATION attr would be wrong')
    g = _drop_face(g)
    g.attrs.update(_provenance())
    out = _out_path(out, 'tile330_grid.zarr', clobber)
    g.to_zarr(out, mode='w', encoding=_clean_encoding(g))
    return str(out)


def open_grid(path=None, with_face: bool = True) -> xr.Dataset:
    """Re-open :func:`write_grid`'s store, in memory, comodo-annotated.

    ``with_face`` restores the length-1 ``face`` dim the dbof operators
    expect (``(face, j, i)``); ``False`` gives the stored ``(j, i)`` layout.
    """
    path = DATA_DIR / 'tile330_grid.zarr' if path is None else path
    g = ensure_comodo_attrs(xr.open_zarr(path).load())
    return g.expand_dims('face') if with_face else g


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
    # both hourly stores decode every index coord as float64 (kerchunk fill
    # value); the gridfile gives int64 -- match it so merges with the grid
    # align exactly.  astype keeps the comodo attrs.
    for c in ('i', 'i_g', 'j', 'j_g', 'k', 'k_l'):
        if c in ds.coords:
            ds[c] = ds[c].astype('int64')
    # the store's iteration counter rides along time, so hours concatenate
    ds = ds.assign_coords(niter=('time', [int(ds['niter'])]))
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


def load_hours(timestamps, tile: TileInfo = None, endpoint: str = OSN_ENDPOINT,
               include_wind: bool = True, grid_ds: xr.Dataset = None) -> xr.Dataset:
    """Consecutive hours from both stores, concatenated along ``time``
    in the §3.2 layout ``(time, j, i)`` (+ ``i_g``/``j_g``), in memory.

    This is the per-hour building block of ``pull_series`` (M2), which adds
    resumability on top; here every hour is pulled fresh.

    Parameters
    ----------
    timestamps : sequence of str
        ``dbof`` format, ``'%Y-%m-%d %H:%M:%S'``.
    tile, endpoint
        As in :func:`load_hour`.
    include_wind : bool
        Also pull ``KPPhbl, oceTAUX, oceTAUY`` from ``llc_wind``.
    grid_ds : xarray.Dataset, optional
        Source of the ``XC``/``YC`` coords (§3.2); omitted if not given.

    Returns
    -------
    xarray.Dataset
        ``face`` is a scalar coord; ``niter(time)`` is the store's own
        iteration; attrs ``iterations, timestamps, endpoint, stores`` plus
        the tile / provenance attrs.
    """
    tile = tile_spec() if tile is None else tile
    hours = []
    for ts in timestamps:
        parts = [load_hour(ts, tile, endpoint)]
        if include_wind:
            parts.append(load_wind_hour(ts, tile, endpoint))
        # the two stores share index coords and time; 'override' keeps the
        # variable attrs (a 'drop' here would strip the comodo attrs too)
        hours.append(xr.merge(parts, compat='override', combine_attrs='override'))
    ds = _drop_face(xr.concat(hours, dim='time', coords='minimal',
                              compat='override', combine_attrs='override'))
    if grid_ds is not None:
        ds = ds.assign_coords(XC=_drop_face(grid_ds).XC, YC=_drop_face(grid_ds).YC)
    ds = ensure_comodo_attrs(ds)
    ds.attrs = dict(
        iterations=[int(osn_date_to_iteration(ts)) for ts in timestamps],
        timestamps=list(timestamps), endpoint=endpoint,
        stores=['llc_surf', 'llc_wind'] if include_wind else ['llc_surf'],
        face_index=int(tile.face_idx),
        j_face_start=int(tile.j_face_slice.start),
        i_face_start=int(tile.i_face_slice.start),
        rect_i=int(TILE_RECT_I), rect_j=int(TILE_RECT_J), land_fill='NaN',
        **_provenance())
    return ds


def write_raw(ds: xr.Dataset, out, clobber: bool = False) -> str:
    """Write a :func:`load_hours` product to zarr, one chunk per hour per
    variable (so M2 can append hour by hour).  Returns the path written."""
    out = _out_path(out, None, clobber)
    ds.to_zarr(out, mode='w', encoding=_clean_encoding(ds, time_chunk=True))
    return str(out)
