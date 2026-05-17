# Routines related to LLC coordinates

import os
import numpy as np
import xarray as xr


def load_coords(verbose=True, orig:bool=False):
    """Load LLC coordinates

    Args:
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        xarray.DataSet: contains the LLC coordinates
    """
    if orig:
        coord_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'LLC_coords.nc')
    else:
        coord_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'coords', 'LLC_coords_lat_lon.nc')
    if verbose:
        print("Loading LLC coords from {}".format(coord_file))
    coord_ds = xr.load_dataset(coord_file, engine='h5netcdf')
    return coord_ds


def ij_to_latlon(i, j, coord_ds=None):
    """Convert LLC pixel (i, j) indices to (lat, lon) degrees.

    The LLC global grid is indexed as ``(row, col) == (j, i)`` so the
    coordinate arrays are sampled at ``[j, i]``.

    Parameters
    ----------
    i : int or array-like
        Column index (x).  May be a scalar or any array-like that numpy
        can fancy-index a 2-D grid with.
    j : int or array-like
        Row index (y), broadcastable against ``i``.
    coord_ds : xarray.Dataset, optional
        Pre-loaded LLC coordinate dataset (as returned by
        :func:`load_coords`).  Loaded on demand when ``None``.

    Returns
    -------
    tuple
        ``(lat, lon)`` in degrees.  Longitudes are wrapped to ``[-180, 180]``
        to match the convention used by :func:`latlon_to_pixel_bbox`.
        Scalars in -> scalars out; arrays in -> ``np.ndarray`` out.
    """
    if coord_ds is None:
        coord_ds = load_coords()

    # Coerce to int arrays so we can fancy-index the 2-D grid
    i_arr = np.asarray(i, dtype=int)
    j_arr = np.asarray(j, dtype=int)

    lat = coord_ds.lat.values[j_arr, i_arr]
    lon = coord_ds.lon.values[j_arr, i_arr]

    # Wrap lon to [-180, 180] to match latlon_to_pixel_bbox convention
    lon = ((lon + 180) % 360) - 180

    # Preserve scalar-in / scalar-out behaviour
    if i_arr.ndim == 0 and j_arr.ndim == 0:
        return float(lat), float(lon)
    return lat, lon


def latlon_to_pixel_bbox(lat0, lon0, lat1, lon1):
    """Convert a lat/lon bounding box to pixel (x,y) indices.

    Uses the LLC coordinate file loaded via :func:`fronts.llc.io.load_coords`.

    Parameters
    ----------
    lat0, lon0 : float
        Lower-left corner (degrees).
    lat1, lon1 : float
        Upper-right corner (degrees).

    Returns
    -------
    tuple[int, int, int, int]
        (x0, y0, x1, y1) pixel indices into the LLC global grid.
    """
    print("Loading LLC coords for lat/lon -> pixel conversion...")
    coord_ds = load_coords()
    lat = coord_ds.lat.values  # shape (ny, nx)
    lon = coord_ds.lon.values

    # Wrap lon to [-180, 180] to match convention
    lon = ((lon + 180) % 360) - 180

    # Distance metric to find the nearest pixel for each corner
    def nearest(lat_val, lon_val):
        dist = (lat - lat_val) ** 2 + (lon - lon_val) ** 2
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx  # (row, col) = (y, x)

    r0, c0 = nearest(lat0, lon0)
    r1, c1 = nearest(lat1, lon1)

    x0, y0 = int(min(c0, c1)), int(min(r0, r1))
    x1, y1 = int(max(c0, c1)), int(max(r0, r1))
    return x0, y0, x1, y1
