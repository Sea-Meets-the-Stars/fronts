"""Co-locate global fronts with properties computed on a single LLC tile.

A tile is a 720x720 window on the same rectangular grid the label map lives on
(``RECT_H`` x ``RECT_W`` = 12960 x 17280), so no regridding is involved.  The
tile's *data*, though, is computed in face-local ``(j, i)`` space, which on some
LLC faces is a rotation of the rect window -- hence
:func:`labels_for_tile` scatters through the per-pixel lookup maps rather than
slicing.

Two sources of field data:

* :func:`tile_loader` -- ``dbof.tiles.tile_utils.run`` slices a tile out of the
  global full-depth store and writes one small NetCDF per property (a cache).
* :func:`chunk_loader` -- a ``LLC4320_RAW/CHUNKS/{name}`` store, which *is*
  already the tile, so properties are computed in memory with nothing written.
"""
import os
from functools import lru_cache

import numpy as np
import xarray as xr

from dbof.tiles import tile_utils
from dbof.tiles.field_registry import resolve_property
from dbof.tiles.tile_mapping import (
    TILE_SIZE, _build_lookup_arrays, rect_ij_to_tile,
)

CHUNKS_PREFIX = 'LLC4320_RAW/CHUNKS'

#: comodo annotations xgcm needs to find its horizontal axes.  The transfer
#: writes a chunk's grid.zarr straight from the source, which may not carry
#: them; ``_build_tile_context`` raises if they are absent.
#:
#: Copied from ``dbof.llc4320_ingestion.get_raw_data.get_llc_depth_gridfile``,
#: which is where this dataset's convention is defined.  The shift sign decides
#: which ``dxC`` a staggered difference is paired with, so it is load-bearing:
#: get it wrong and every gradient field is off by one cell's metric while
#: pointwise fields stay exact.
_COMODO = {
    'j':   {'axis': 'Y'},
    'j_g': {'axis': 'Y', 'c_grid_axis_shift': 0.5},
    'i':   {'axis': 'X'},
    'i_g': {'axis': 'X', 'c_grid_axis_shift': 0.5},
}


@lru_cache(maxsize=1)
def lookup_maps():
    """Cached rect-grid ``(face_id, j_face, i_face)`` maps.

    Building these stitches three 13 x 4320 x 4320 index arrays, so it is
    cached for the life of the process -- a per-tile loop would otherwise pay
    for it on every call.
    """
    return _build_lookup_arrays()


def tile_for(i_rect: int = None, j_rect: int = None,
             lon: float = None, lat: float = None):
    """Resolve a rect pixel or a geographic point to its enclosing tile.

    Parameters
    ----------
    i_rect, j_rect : int, optional
        Any pixel inside the wanted tile.  Mutually exclusive with lon/lat.
    lon, lat : float, optional
        Geographic point, resolved to the nearest rect pixel.

    Returns
    -------
    dbof.tiles.tile_mapping.TileInfo
    """
    if lon is not None or lat is not None:
        i_rect, j_rect = tile_utils.latlon_to_rect_ij(
            lon, lat, tile_utils._resolve_s3_source(None))
    if i_rect is None or j_rect is None:
        raise ValueError("give either (i_rect, j_rect) or (lon, lat)")
    return rect_ij_to_tile(i_rect, j_rect)


def labels_for_tile(labeled_global: np.ndarray, tile,
                    edge_margin: int = 0) -> np.ndarray:
    """Reorient the global label map onto the tile's face-local grid.

    Labels stay global, so results join to the geometry table on ``flabel``.
    Fronts crossing the tile edge are clipped: ``npix`` from a tile run counts
    only the pixels inside it.

    Parameters
    ----------
    labeled_global : np.ndarray
        Full ``(RECT_H, RECT_W)`` label map from ``group_fronts``.
    tile : TileInfo
    edge_margin : int
        Zero this many cells at the tile rim.  ``compute_tile_property``
        already NaNs an ``edge_margin`` rim for stencil-based fields and
        ``nan_policy='omit'`` drops those cells from the statistics, so this is
        only needed to keep ``npix`` from counting them.
    """
    _, j_map, i_map = lookup_maps()
    win = (tile.rect_j_slice, tile.rect_i_slice)

    out = np.zeros((TILE_SIZE, TILE_SIZE), dtype=labeled_global.dtype)
    out[j_map[win] - tile.j_face_slice.start,
        i_map[win] - tile.i_face_slice.start] = labeled_global[win]

    if edge_margin:
        m = edge_margin
        out[:m, :] = 0
        out[-m:, :] = 0
        out[:, :m] = 0
        out[:, -m:] = 0
    return out


def tile_loader(timestamp: str, tile, cache_dir: str, clobber: bool = False,
                level: int = 0):
    """Return ``loader(property_name) -> 2D array`` on the tile's grid.

    Each property is computed by ``dbof.tiles.tile_utils.run`` into
    *cache_dir* and read back at *level* (0 = surface for depth-resolved
    fields; inherently-2D fields have no ``k`` dimension).

    Parameters
    ----------
    timestamp : str
        Snapshot timestamp, e.g. '2012-07-03T12_00_00'.
    tile : TileInfo
    cache_dir : str
        Directory for the per-property tile NetCDFs.
    clobber : bool
        Recompute even if the tile NetCDF exists.
    level : int
        ``k`` index to co-locate.  Defaults to 0 (surface).
    """
    os.makedirs(cache_dir, exist_ok=True)
    date_str = timestamp.replace('T', ' ').replace('_', ':')   # dbof DATE_FMT

    def loader(name):
        prop = resolve_property(name)
        path = os.path.join(
            cache_dir, f'tile{tile.tile_idx:03d}_{timestamp}_{name}.nc')
        if clobber or not os.path.isfile(path):
            print(f"  computing {name} on tile {tile.tile_idx}")
            tile_utils.run(i_rect=tile.rect_i_slice.start,
                           j_rect=tile.rect_j_slice.start,
                           timestamp=date_str, property=name,
                           output=path, clobber=clobber)
        with xr.open_dataset(path) as ds:
            da = ds[prop.out_name]
            if 'k' in da.dims:
                da = da.isel(k=level)
            return da.values.astype(np.float32)

    return loader


# ---------------------------------------------------------------------------
# CHUNKS stores -- the chunk already is the tile
# ---------------------------------------------------------------------------

def _chunk_uri(chunk_name: str, leaf: str, bucket: str = 'dbof') -> str:
    return f"s3://{bucket.strip('/')}/{CHUNKS_PREFIX}/{chunk_name}/{leaf}"


def _open_chunk(uri: str, s3_endpoint: str) -> xr.Dataset:
    """Open a chunk store lazily.  Metadata is not consolidated."""
    return xr.open_zarr(
        uri, consolidated=False,
        storage_options={'client_kwargs': {'endpoint_url': s3_endpoint}})


def tile_from_chunk_store(chunk_name: str, bucket: str = 'dbof',
                          s3_endpoint: str = 'https://s3-west.nrp-nautilus.io'):
    """Resolve the tile a CHUNKS store holds, from the store's own attrs.

    The transfer records ``resolved_face``, ``j_start``, ``i_start`` and
    ``tile_size``, so the tile is self-describing -- no lon/lat lookup.  The
    rect-grid extent is recovered by locating that face-local corner in the
    lookup maps, then re-resolved through ``rect_ij_to_tile`` so every field of
    the returned TileInfo is mutually consistent.
    """
    ds = _open_chunk(_chunk_uri(chunk_name, 'grid.zarr', bucket), s3_endpoint)
    face = int(ds.attrs['resolved_face'])
    j0, i0 = int(ds.attrs['j_start']), int(ds.attrs['i_start'])
    size = int(ds.attrs.get('tile_size', TILE_SIZE))
    ds.close()

    if size != TILE_SIZE:
        raise ValueError(f"{chunk_name}: tile_size {size} != {TILE_SIZE}")

    face_id, j_map, i_map = lookup_maps()
    hit = np.argwhere((face_id == face) & (j_map == j0) & (i_map == i0))
    if not len(hit):
        raise ValueError(
            f"{chunk_name}: face {face} face-local (j={j0}, i={i0}) is not on "
            f"the rect grid")
    j_rect, i_rect = (int(x) for x in hit[0])

    tile = rect_ij_to_tile(i_rect, j_rect)
    if (tile.face_idx, tile.j_face_slice.start, tile.i_face_slice.start) \
            != (face, j0, i0):
        raise ValueError(
            f"{chunk_name}: store says face {face} (j={j0}, i={i0}) but the "
            f"rect lookup resolves to face {tile.face_idx} "
            f"(j={tile.j_face_slice.start}, i={tile.i_face_slice.start})")
    return tile


def chunk_context(chunk_name: str, timestamp: str, bucket: str = 'dbof',
                  s3_endpoint: str = 'https://s3-west.nrp-nautilus.io'):
    """Return ``(ds_merge, grid)`` for one snapshot of a CHUNKS store.

    The store is already the tile's extent, so nothing is sliced.  Exposed so a
    caller can run a compute callback directly -- comparing two formulations of
    a field, say -- instead of going through :func:`chunk_loader`.
    """
    date = timestamp[:13].replace('-', '').replace('_', '')   # YYYYMMDDTHH
    ds_t = _open_chunk(_chunk_uri(chunk_name, f'{date}.zarr', bucket), s3_endpoint)
    ds_g = _open_chunk(_chunk_uri(chunk_name, 'grid.zarr', bucket), s3_endpoint)

    if 'time' in ds_t.dims:
        ds_t = ds_t.isel(time=0)
    for dim, attrs in _COMODO.items():
        if dim in ds_g.coords and 'axis' not in ds_g[dim].attrs:
            ds_g[dim].attrs.update(attrs)

    return tile_utils._build_tile_context(ds_t, ds_g)


def surface(da, level: int = 0) -> np.ndarray:
    """Squeeze a tile field to a 2D float32 array at *level*."""
    if 'face' in getattr(da, 'dims', ()):
        da = da.squeeze('face', drop=True)
    if 'k' in getattr(da, 'dims', ()):
        da = da.isel(k=level)
    return np.asarray(da, dtype=np.float32)


def chunk_loader(chunk_name: str, timestamp: str, bucket: str = 'dbof',
                 s3_endpoint: str = 'https://s3-west.nrp-nautilus.io',
                 level: int = 0, mask_land: bool = True):
    """Return ``loader(property_name) -> 2D array`` from a CHUNKS store.

    The store is already the tile's extent, so nothing is sliced and nothing is
    written: each property is computed in memory and reduced to *level*.
    """
    ds_merge, grid = chunk_context(chunk_name, timestamp, bucket, s3_endpoint)

    def loader(name):
        prop = resolve_property(name)
        print(f"  computing {name} on chunk {chunk_name}")
        da = tile_utils.compute_tile_property(ds_merge, grid, prop,
                                             mask_land=mask_land)
        if 'k' in da.dims:
            da = da.isel(k=level)
        return da.values.astype(np.float32)

    return loader
