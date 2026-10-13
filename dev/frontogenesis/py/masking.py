""" Static masks for one tile (coding doc §3.5, §4.2): ocean, land halo,
distance to the coast, offshore cut, tile-edge margin, and their
intersection, plus the ``tile330_masks.nc`` writer.

Convention (§1.2): every mask is a plain bool numpy array on the centred
``(j, i)`` grid, **True = retained**, land = False.  ``coast_distance_km``
is float, NaN on land.  A length-1 ``face`` dim on the input grid is
squeezed and a ``k`` dim is collapsed to the surface level first, so the
outputs are always 2-D whatever layout the grid arrives in.

Why each mask exists
--------------------
* ``ocean_mask``   -- ``hFacC > 0``.  Land is NaN in the OSN fields (M0
  task 3), cell for cell equal to this, so this is also where every
  stencil output is NaN.
* ``halo_mask``    -- ocean at least ``halo_cells`` from land, measured as a
  fast-marching (Euclidean) distance by dbof's ``llc_native_grid_halo_mask``.
  The budget is 7 cells: 3 for the gradient + Jacobian + interpolation reach
  (measured 1 cell for G, 2 for the Jacobian, M0 task 5) and 4 for the
  widest filter half-width.  Its job is filter support and a consistent
  distance field, not removing a gradient ribbon -- there is none.
* ``coast_distance_km`` / ``offshore_mask`` -- the >= 100 km cut for the
  primary statistics (planning §5.5): the coastal upwelling band dominates
  the upper G percentiles (M0 task 5) and is not the open-ocean physics
  the study is about.
* ``edge_mask``    -- the tile-edge rim is *finite, not NaN*: xgcm pads
  the missing high staggered point with 0, so G is ~1e6x wrong on the low
  edges and ~0.5x on the high edges, one to two cells deep (M0 task 5).
  Neither the land halo (skfmm measures from ``hFacC == 0``) nor the
  offshore cut (the west and north edges are open ocean) removes it.

The dbof helper takes a distance in **km**; our API takes **cells** and
converts with the *measured* median ``dxC`` (never a hard-coded km value).
On face 10 ``dxC`` is the meridional spacing (1.68-1.90 km) and ``dyC`` the
zonal one, 8.6% larger (M0 task 3).
"""

from pathlib import Path

import numpy as np
import xarray as xr
import skfmm

from dbof.preprocessing.halo_mask import llc_native_grid_halo_mask
from osn_tiles import DATA_DIR, _provenance

# The §3.5 variables, in the order they are written
MASK_VARS = ('mask_ocean', 'mask_halo', 'mask_offshore', 'mask_edge',
             'mask_analysis', 'coast_distance_km')


# ---------------------------------------------------------------------------
# grid access: positional 2-D arrays whatever the layout
# ---------------------------------------------------------------------------
def _positional(grid_ds, name: str) -> np.ndarray:
    """``grid_ds[name]`` as a float64 ``(j, i)``-like array: a length-1
    ``face`` squeezed, a ``k`` dim collapsed to the surface level."""
    da = grid_ds[name]
    if 'k' in da.dims:
        # a 3-D grid (the chunk store's) carries hFacC(k, j, i); the surface
        # study wants the top cell.  Collapsing here is what keeps the mask
        # 2-D and skfmm alive (coding doc §2.4).
        da = da.isel(k=0)
    if 'face' in da.dims:
        if da.sizes['face'] != 1:
            raise ValueError(f'{name}: one tile expected, got {da.sizes["face"]} faces')
        da = da.squeeze('face')
    if da.ndim != 2:
        raise ValueError(f'{name}: expected 2-D after collapsing face/k, got dims {da.dims}')
    return np.asarray(da.values, dtype='float64')


def _assert_shape(arr, shape, what: str):
    """Raise unconditionally (not the ``assert`` statement, which ``-O``
    strips) if ``arr`` is not the bool array of the expected shape."""
    if tuple(arr.shape) != tuple(shape) or arr.dtype != bool:
        raise AssertionError(f'{what}: expected bool {tuple(shape)}, got '
                             f'{arr.dtype} {tuple(arr.shape)}')


def spacing_km(grid_ds) -> dict:
    """Grid spacing summary in km: the medians the cell->km conversion uses
    and the means the fast-marching distance uses (the helper's recipe)."""
    dx = _positional(grid_ds, 'dxC') / 1e3
    dy = _positional(grid_ds, 'dyC') / 1e3
    return dict(dxC_median_km=float(np.median(dx)), dyC_median_km=float(np.median(dy)),
                dxC_mean_km=float(dx.mean()), dyC_mean_km=float(dy.mean()),
                dxC_min_km=float(dx.min()), dxC_max_km=float(dx.max()))


def halo_width_km(grid_ds, halo_cells: int = 7) -> float:
    """``halo_cells`` converted to km with the measured median ``dxC``
    (§4.2: never hard-code the km value).  ``dxC`` is the smaller of the
    two spacings on this face, so the halo is at least ``halo_cells`` wide
    in the ``i`` direction and ``dxC/dyC`` (0.92) of that in ``j``."""
    return float(halo_cells * spacing_km(grid_ds)['dxC_median_km'])


# ---------------------------------------------------------------------------
# the masks
# ---------------------------------------------------------------------------
def ocean_mask(grid_ds) -> np.ndarray:
    """True where the top cell is wet (``hFacC > 0``).  ``hFacC`` is binary
    at k=0 on this tile (M0 task 3), so this is exactly ``isfinite(Theta)``."""
    return _positional(grid_ds, 'hFacC') > 0


def _distance_km(ocean: np.ndarray, dx_km: float, dy_km: float) -> np.ndarray:
    """Fast-marching distance from the coast, km, NaN on land -- the same
    recipe as ``llc_native_grid_halo_mask``: a level set ``phi = +1`` on
    ocean, ``-1`` on land (so the zero level, the coastline, lies half a
    cell from the last land centre), marched on a uniform grid with the
    per-axis *mean* spacing.  Distances are therefore accurate to the
    spread of the spacing across the tile (about 6%), which is immaterial
    for a nominal 100 km cut and a 7-cell halo."""
    d = np.full(ocean.shape, np.nan)
    if not ocean.any():
        return d                       # no coast to measure from
    if ocean.all():
        d[:] = np.inf                  # no land anywhere: infinitely offshore
        return d
    phi = np.where(ocean, 1.0, -1.0)
    # axis 0 is j (spacing dyC), axis 1 is i (spacing dxC)
    dist = skfmm.distance(phi, dx=(dy_km, dx_km))
    d[ocean] = dist[ocean]
    return d


def coast_distance_km(grid_ds) -> np.ndarray:
    """Distance to the nearest land cell, km (float, NaN on land)."""
    sp = spacing_km(grid_ds)
    return _distance_km(ocean_mask(grid_ds), sp['dxC_mean_km'], sp['dyC_mean_km'])


def halo_mask(grid_ds, halo_cells: int = 7) -> np.ndarray:
    """Ocean at least ``halo_cells`` (as km, see :func:`halo_width_km`)
    from land; True = retained.

    Wraps dbof's ``llc_native_grid_halo_mask`` (§2.4) and guards its two
    known defects: (1) a face with no ocean makes it ``return mask_f`` from
    inside its face loop (``halo_mask.py:74-75``) -- a 2-D, all-True,
    convention-inverted array -- so that case is answered here without
    calling it; (2) a ``k``-carrying ``hFacC`` makes the mask 4-D and skfmm
    raises ``ValueError``, so ``k`` is collapsed first.  The output shape is
    asserted after the call.  The result is identical to
    ``ocean & (coast_distance_km >= halo_km)`` (same recipe; checked in
    ``test_masking.py``).
    """
    oc = ocean_mask(grid_ds)
    nj, ni = oc.shape
    if not oc.any():
        return np.zeros((nj, ni), dtype=bool)   # nothing to retain
    halo_km = halo_width_km(grid_ds, halo_cells)
    # the helper indexes mask[face].values and dxC[face].mean(): hand it
    # xarray objects with a leading length-1 face axis and land = True
    land = xr.DataArray(~oc[None], dims=('face', 'j', 'i'))
    dxc = xr.DataArray(_positional(grid_ds, 'dxC')[None], dims=('face', 'j', 'i_g'))
    dyc = xr.DataArray(_positional(grid_ds, 'dyC')[None], dims=('face', 'j_g', 'i'))
    out = np.asarray(llc_native_grid_halo_mask(land, dxc, dyc, halo_km))
    _assert_shape(out, (1, nj, ni), 'llc_native_grid_halo_mask')
    out = out[0]
    if out[~oc].any():
        raise AssertionError('halo mask retains land cells')
    return out


def offshore_mask(grid_ds, min_km: float = 100.0) -> np.ndarray:
    """Ocean at least ``min_km`` from any land; True = retained."""
    d = coast_distance_km(grid_ds)
    return np.where(np.isfinite(d) | np.isposinf(d), d >= min_km, False)


def edge_mask(grid_ds, edge_cells: int = 7) -> np.ndarray:
    """False within ``edge_cells`` of ANY tile edge, True elsewhere (pure
    geometry, not intersected with the ocean).  The rim that xgcm's zero
    padding corrupts is 1 cell (G) / 2 cells (Jacobian) deep, so 2 is the
    minimum for the raw operators; 7 matches the land halo once the filter
    half-width (4) is counted (§4.2)."""
    nj, ni = ocean_mask(grid_ds).shape
    m = np.zeros((nj, ni), dtype=bool)
    if 2 * edge_cells < min(nj, ni):
        m[edge_cells:nj - edge_cells, edge_cells:ni - edge_cells] = True
    return m


def analysis_mask(grid_ds, halo_cells: int = 7, min_km: float = 100.0,
                  edge_cells: int = 7) -> np.ndarray:
    """``ocean & halo & offshore & edge``: the cells the primary statistics
    use.  (The halo is a subset of the offshore cut for any ``min_km`` above
    ~13 km; it is kept in the product so the stages stay visible.)"""
    m = (ocean_mask(grid_ds) & halo_mask(grid_ds, halo_cells)
         & offshore_mask(grid_ds, min_km) & edge_mask(grid_ds, edge_cells))
    _assert_shape(m, ocean_mask(grid_ds).shape, 'analysis_mask')
    return m


# ---------------------------------------------------------------------------
# tile330_masks.nc (§3.5)
# ---------------------------------------------------------------------------
def build_masks(grid_ds, halo_cells: int = 7, min_km: float = 100.0,
                edge_cells: int = 7) -> xr.Dataset:
    """The §3.5 dataset: the five masks and ``coast_distance_km`` on
    ``(j, i)`` with ``XC``/``YC`` coords, and attrs recording every
    parameter, the spacing used, the convention, and the retained count at
    each stage."""
    oc = ocean_mask(grid_ds)
    halo = halo_mask(grid_ds, halo_cells)
    dist = coast_distance_km(grid_ds)
    off = offshore_mask(grid_ds, min_km)
    edge = edge_mask(grid_ds, edge_cells)
    ana = oc & halo & off & edge
    for m in (halo, off, edge, ana):
        _assert_shape(m, oc.shape, 'build_masks')
    sp = spacing_km(grid_ds)
    g2 = grid_ds.squeeze('face') if 'face' in grid_ds.dims else grid_ds
    coords = {'j': g2['j'], 'i': g2['i']}
    for c in ('XC', 'YC'):
        if c in g2.coords:
            coords[c] = (('j', 'i'), g2[c].values)
    dims = ('j', 'i')
    ds = xr.Dataset(
        {'mask_ocean': (dims, oc, {'long_name': 'hFacC > 0 at the surface'}),
         'mask_halo': (dims, halo, {'long_name': f'ocean >= {halo_cells} cells '
                                    f'({halo_width_km(grid_ds, halo_cells):.2f} km) from land'}),
         'mask_offshore': (dims, off, {'long_name': f'ocean >= {min_km:g} km from land'}),
         'mask_edge': (dims, edge, {'long_name': f'>= {edge_cells} cells from every tile '
                                    'edge (geometry only, not intersected with the ocean)'}),
         'mask_analysis': (dims, ana, {'long_name': 'ocean & halo & offshore & edge'}),
         'coast_distance_km': (dims, dist, {'units': 'km', 'long_name':
                               'fast-marching distance to the nearest land cell; NaN on land'})},
        coords=coords)
    ds.attrs.update(
        convention='bool masks: True = retained, land = False; coast_distance_km NaN on land',
        halo_cells=int(halo_cells), halo_km=halo_width_km(grid_ds, halo_cells),
        halo_km_rule='halo_cells * median(dxC), dxC measured from the grid (coding doc §4.2)',
        edge_cells=int(edge_cells), offshore_km=float(min_km),
        distance_method=('skfmm.distance on phi=+1 ocean / -1 land, uniform per-axis mean '
                         'spacing dx=(mean dyC, mean dxC) -- the llc_native_grid_halo_mask recipe'),
        **{k: round(v, 5) for k, v in sp.items()},
        n_cells=int(oc.size), n_ocean=int(oc.sum()), n_halo=int(halo.sum()),
        n_offshore=int(off.sum()), n_edge_ocean=int((edge & oc).sum()),
        n_analysis=int(ana.sum()),
        **{k: grid_ds.attrs[k] for k in ('face_index', 'j_face_start', 'i_face_start',
                                         'rect_i', 'rect_j', 'orientation')
           if k in grid_ds.attrs},
        source_grid_commit=grid_ds.attrs.get('git_commit', 'unknown'),
        **_provenance())
    return ds


def write_masks(masks_ds: xr.Dataset, out=None, clobber: bool = False) -> str:
    """Write :func:`build_masks`'s dataset to ``tile330_masks.nc`` (netCDF4;
    bools are stored as int8 with ``dtype='bool'`` and decode back).
    Returns the path written."""
    out = DATA_DIR / 'tile330_masks.nc' if out is None else Path(out)
    if out.exists() and not clobber:
        raise FileExistsError(f'{out} exists; pass clobber=True to overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {'zlib': True, 'complevel': 4} for v in masks_ds.data_vars}
    masks_ds.to_netcdf(out, encoding=enc)
    return str(out)


def open_masks(path=None) -> xr.Dataset:
    """Re-open :func:`write_masks`'s file in memory; masks come back bool."""
    path = DATA_DIR / 'tile330_masks.nc' if path is None else path
    ds = xr.load_dataset(path)
    for v in MASK_VARS[:-1]:
        if ds[v].dtype != bool:
            ds[v] = ds[v].astype(bool)
    return ds
