"""Co-locate global fronts with properties computed on a single LLC tile.

A tile is a 720x720 window on the same rectangular grid the label map lives on
(``RECT_H`` x ``RECT_W`` = 12960 x 17280), so no regridding is involved.  The
tile's *data*, though, is computed in face-local ``(j, i)`` space, which on some
LLC faces is a rotation of the rect window -- hence
:func:`labels_for_tile` scatters through the per-pixel lookup maps rather than
slicing.

Fields come from ``dbof.tiles.tile_utils.run``, which writes one small NetCDF
per property per tile (~2 MB for a 2D field).  Those act as a cache: a second
co-location of the same tile recomputes nothing.
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
