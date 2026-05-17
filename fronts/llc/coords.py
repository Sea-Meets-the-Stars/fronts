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


def latlon_to_ij(lat, lon, coord_ds=None):
    """Convert (lat, lon) in degrees to nearest LLC pixel (i, j) indices.

    The LLC global grid is indexed as ``(row, col) == (j, i)``.  This is the
    inverse of :func:`ij_to_latlon` and uses a simple Euclidean distance in
    degree-space to pick the nearest grid point.

    Parameters
    ----------
    lat : float or array-like
        Latitude(s) in degrees.
    lon : float or array-like
        Longitude(s) in degrees.  Any convention is accepted; longitudes are
        wrapped to ``[-180, 180]`` internally to match the grid.
    coord_ds : xarray.Dataset, optional
        Pre-loaded LLC coordinate dataset (as returned by
        :func:`load_coords`).  Loaded on demand when ``None``.

    Returns
    -------
    tuple
        ``(i, j)`` pixel indices.  Scalars in -> ``int`` out; arrays in ->
        ``np.ndarray`` out (of the broadcast shape).
    """
    if coord_ds is None:
        coord_ds = load_coords()

    grid_lat = coord_ds.lat.values                              # (ny, nx)
    grid_lon = ((coord_ds.lon.values + 180) % 360) - 180        # wrap to [-180, 180]

    lat_in = np.asarray(lat, dtype=float)
    lon_in = np.asarray(lon, dtype=float)
    # Track whether the caller passed plain scalars so we can return ints
    scalar = lat_in.ndim == 0 and lon_in.ndim == 0

    # Broadcast lat/lon so we can iterate paired queries with one shape
    lat_b, lon_b = np.broadcast_arrays(lat_in, lon_in)

    i_out = np.empty(lat_b.shape, dtype=int)
    j_out = np.empty(lat_b.shape, dtype=int)
    i_flat = i_out.ravel()
    j_flat = j_out.ravel()

    # One nearest-pixel search per query point.  Cheap for small K (e.g. the
    # two corners passed by latlon_to_pixel_bbox); a broadcasted distance
    # tensor would blow up memory on the full LLC grid for large K.
    for k, (la, lo) in enumerate(zip(lat_b.ravel(), lon_b.ravel())):
        dist = (grid_lat - la) ** 2 + (grid_lon - lo) ** 2
        ridx = np.unravel_index(np.argmin(dist), dist.shape)
        j_flat[k] = int(ridx[0])  # row = j = y
        i_flat[k] = int(ridx[1])  # col = i = x

    if scalar:
        return int(i_out.item()), int(j_out.item())
    return i_out, j_out


def latlon_to_pixel_bbox(lat0, lon0, lat1, lon1):
    """Convert a lat/lon bounding box to pixel (x,y) indices.

    Uses the LLC coordinate file loaded via :func:`load_coords` and delegates
    the per-corner nearest-pixel lookup to :func:`latlon_to_ij`.

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

    # Reuse latlon_to_ij so the nearest-pixel logic lives in one place
    i0, j0 = latlon_to_ij(lat0, lon0, coord_ds=coord_ds)
    i1, j1 = latlon_to_ij(lat1, lon1, coord_ds=coord_ds)

    # Normalise into (min, max) corners regardless of input ordering
    x0, y0 = min(i0, i1), min(j0, j1)
    x1, y1 = max(i0, i1), max(j0, j1)
    return x0, y0, x1, y1
