""" Tests for ``masking.py`` (coding doc §5, prompt 2 task 1).

Synthetic-grid tests run offline.  Tests on the real tile are marked
``needs_grid`` and use the M0 stores from ``conftest.py``.

Guards: the halo width in cells and km; the two ``llc_native_grid_halo_mask``
defects (an all-land face; a ``k``-carrying ``hFacC``); the True = retained
convention; ``ocean_mask == isfinite(Theta)`` cell for cell; and the tile-edge
margin covering the rim ``m0_qa_checks.check_edge_rim`` measures.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

import masking as mk
from dbof.preprocessing.halo_mask import llc_native_grid_halo_mask


# ---------------------------------------------------------------------------
# synthetic grids
# ---------------------------------------------------------------------------
def synthetic_grid(nj=60, ni=80, land_rows=10, land_cols=0, dx_m=1800.0, dy_m=None,
                   with_face=True, k_levels=0, hfacc=None):
    """A tile-like grid: land in the first ``land_rows`` rows (a coast
    perpendicular to ``j``) and the first ``land_cols`` columns; uniform
    ``dxC``/``dyC``; ``XC``/``YC`` on a plain lon/lat ladder.  ``k_levels``
    > 0 gives ``hFacC`` a leading ``k`` dim (the chunk-store layout)."""
    dy_m = dx_m if dy_m is None else dy_m
    if hfacc is None:
        hfacc = np.ones((nj, ni))
        hfacc[:land_rows, :] = 0.0
        hfacc[:, :land_cols] = 0.0
    dims = ('j', 'i')
    ds = xr.Dataset(
        {'hFacC': (dims, hfacc),
         'dxC': (('j', 'i_g'), np.full((nj, ni), dx_m)),
         'dyC': (('j_g', 'i'), np.full((nj, ni), dy_m))},
        coords={'j': np.arange(nj), 'i': np.arange(ni),
                'j_g': np.arange(nj), 'i_g': np.arange(ni),
                'XC': (dims, -125.0 + 0.02 * np.arange(nj)[:, None] + 0 * np.arange(ni)),
                'YC': (dims, 35.0 - 0.02 * np.arange(ni)[None, :] + 0 * np.arange(nj)[:, None])})
    if k_levels:
        ds['hFacC'] = ds['hFacC'].expand_dims(k=np.arange(k_levels)).transpose('k', 'j', 'i')
    if with_face:
        ds = ds.expand_dims('face')
    return ds


# ---------------------------------------------------------------------------
# convention and shapes
# ---------------------------------------------------------------------------
def test_true_is_retained_land_is_false():
    g = synthetic_grid()
    oc = mk.ocean_mask(g)
    assert oc.dtype == bool and oc.shape == (60, 80)
    assert not oc[:10].any() and oc[10:].all()
    for fn in (mk.halo_mask, mk.offshore_mask, mk.analysis_mask):
        m = fn(g)
        assert m.dtype == bool and m.shape == oc.shape
        assert not m[~oc].any(), f'{fn.__name__} retains land'
        assert m.sum() < oc.sum()                     # every cut removes something
    d = mk.coast_distance_km(g)
    assert np.isnan(d[~oc]).all() and np.isfinite(d[oc]).all()
    # the ocean cell touching the coast is half a cell from the interface
    assert d[10, 40] == pytest.approx(0.9, abs=1e-6)     # 0.5 * 1.8 km


def test_analysis_mask_is_the_intersection():
    g = synthetic_grid(land_cols=5)
    ana = mk.analysis_mask(g, halo_cells=3, min_km=10.0, edge_cells=2)
    expect = (mk.ocean_mask(g) & mk.halo_mask(g, 3) & mk.offshore_mask(g, 10.0)
              & mk.edge_mask(g, 2))
    assert np.array_equal(ana, expect)
    assert ana.any()


def test_edge_mask_geometry():
    g = synthetic_grid()
    e = mk.edge_mask(g, edge_cells=7)
    assert e.shape == (60, 80) and e.dtype == bool
    assert not e[:7].any() and not e[-7:].any() and not e[:, :7].any() and not e[:, -7:].any()
    assert e[7:-7, 7:-7].all()
    assert mk.edge_mask(g, edge_cells=0).all()
    assert e.sum() == (60 - 14) * (80 - 14)


# ---------------------------------------------------------------------------
# halo width, in cells and km
# ---------------------------------------------------------------------------
def test_halo_width_km_uses_measured_median_dxC():
    g = synthetic_grid(dx_m=1800.0, dy_m=1950.0)
    # dxC with a skewed tail: the median must win over the mean
    dx = np.full((60, 80), 1800.0)
    dx[:, :5] = 3000.0
    g['dxC'] = (('face', 'j', 'i_g'), dx[None])
    assert mk.halo_width_km(g, 7) == pytest.approx(7 * 1.8)
    assert mk.halo_width_km(g, 7) != pytest.approx(7 * dx.mean() / 1e3)
    assert mk.halo_width_km(g, 3) == pytest.approx(5.4)


@pytest.mark.parametrize('halo_cells', [3, 7])
def test_halo_width_in_cells_uniform_grid(halo_cells):
    """Straight coasts, dx == dy: the fast-marching distance of the ocean
    cell at index distance ``d`` from land is ``(d - 0.5) dx`` (the
    interface lies half a cell from the last land centre), so the halo
    excludes ``d <= halo_cells`` and retains ``d >= halo_cells + 1`` -- in
    both grid directions."""
    g = synthetic_grid(land_rows=10, land_cols=12)
    oc = mk.ocean_mask(g)
    h = mk.halo_mask(g, halo_cells)
    taxi = ndimage.distance_transform_cdt(oc, metric='taxicab')
    assert taxi[h].min() == halo_cells + 1
    assert taxi[oc & ~h].max() == halo_cells
    # the same, by row / column
    assert not h[10:10 + halo_cells, 40].any() and h[10 + halo_cells:, 40].all()
    assert not h[40, 12:12 + halo_cells].any() and h[40, 12 + halo_cells:].all()
    # and in km, against coast_distance_km
    d = mk.coast_distance_km(g)
    halo_km = mk.halo_width_km(g, halo_cells)
    assert halo_km == pytest.approx(halo_cells * 1.8)
    assert np.nanmin(d[h]) >= halo_km > np.nanmax(d[oc & ~h])
    assert np.array_equal(h, oc & (np.nan_to_num(d, nan=-1) >= halo_km))


def test_halo_anisotropic_spacing_converts_with_dxC():
    """``halo_km = halo_cells * median(dxC)``; with ``dyC`` 8.6% larger
    (face 10) the halo is ``halo_cells`` wide along ``i`` and about
    ``halo_cells * dxC/dyC`` along ``j``."""
    g = synthetic_grid(land_rows=10, land_cols=12, dx_m=1800.0, dy_m=1955.0)
    h = mk.halo_mask(g, 7)
    assert mk.halo_width_km(g, 7) == pytest.approx(12.6)
    # along i (spacing dxC): (d - 0.5) * 1.8 >= 12.6  ->  d >= 7.5  ->  d = 8
    assert not h[40, 12:12 + 7].any() and h[40, 12 + 7:].all()
    # along j (spacing dyC): (d - 0.5) * 1.955 >= 12.6  ->  d >= 6.94  ->  d = 7
    assert not h[10:10 + 6, 40].any() and h[10 + 6:, 40].all()


# ---------------------------------------------------------------------------
# the two llc_native_grid_halo_mask defects
# ---------------------------------------------------------------------------
def test_all_land_face_guard():
    """``halo_mask.py:74-75``: a face with no ocean makes the helper
    ``return mask_f`` -- 2-D, all True, i.e. 'keep everything' in the
    output convention.  Our wrapper answers all-False, 2-D, without
    reaching it."""
    g = synthetic_grid(hfacc=np.zeros((60, 80)))
    # the defect itself, on record
    raw = llc_native_grid_halo_mask(g['hFacC'] == 0, g['dxC'], g['dyC'], 12.0)
    assert raw.shape == (60, 80) and raw.all()
    # the wrapper
    h = mk.halo_mask(g)
    assert h.shape == (60, 80) and h.dtype == bool and not h.any()
    assert not mk.ocean_mask(g).any()
    assert np.isnan(mk.coast_distance_km(g)).all()
    assert not mk.offshore_mask(g).any() and not mk.analysis_mask(g).any()


def test_k_carrying_hfacc_is_collapsed():
    """A ``k``-carrying ``hFacC`` makes the helper's per-face mask 3-D and
    skfmm rejects it (``dx must be of length len(phi.shape)``).  Our
    functions collapse ``k`` to the surface level first and give the same
    2-D answer as the 2-D grid."""
    g2 = synthetic_grid(land_rows=10, land_cols=12)
    g3 = synthetic_grid(land_rows=10, land_cols=12, k_levels=3)
    assert g3['hFacC'].dims == ('face', 'k', 'j', 'i')
    with pytest.raises(ValueError):
        llc_native_grid_halo_mask(g3['hFacC'] == 0, g3['dxC'], g3['dyC'], 12.0)
    for fn in (mk.ocean_mask, mk.halo_mask, mk.offshore_mask, mk.edge_mask, mk.analysis_mask):
        a, b = fn(g2), fn(g3)
        assert a.shape == (60, 80) and np.array_equal(a, b), fn.__name__
    assert np.array_equal(mk.coast_distance_km(g2), mk.coast_distance_km(g3), equal_nan=True)


def test_no_face_dim_and_no_land():
    g = synthetic_grid(land_rows=0, with_face=False)
    assert mk.ocean_mask(g).all() and mk.halo_mask(g).all() and mk.offshore_mask(g).all()
    assert np.isposinf(mk.coast_distance_km(g)).all()
    assert np.array_equal(mk.analysis_mask(g), mk.edge_mask(g))


def test_multi_face_grid_rejected():
    g = xr.concat([synthetic_grid(), synthetic_grid()], dim='face')
    with pytest.raises(ValueError):
        mk.ocean_mask(g)


# ---------------------------------------------------------------------------
# the real tile
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_ocean_mask_equals_isfinite_theta(grid_ds, raw_ds):
    """Land is NaN in the OSN fields, cell for cell where ``hFacC == 0``
    (M0 tasks 3-5: 0 mismatches in 518,400 cells)."""
    oc = mk.ocean_mask(grid_ds)
    assert oc.shape == (720, 720) and oc.sum() == 356_877
    for t in range(raw_ds.sizes['time']):
        fin = np.isfinite(raw_ds['Theta'].isel(time=t).values)
        assert np.array_equal(oc, fin), f'{(oc != fin).sum()} mismatches at t{t}'


@pytest.mark.needs_grid
def test_halo_matches_dbof_helper_and_distance(grid_ds):
    """The wrapper reproduces the raw helper (shape asserted, 3-D -> 2-D)
    and equals ``ocean & (coast_distance_km >= halo_km)``."""
    oc = mk.ocean_mask(grid_ds)
    halo_km = mk.halo_width_km(grid_ds, 7)
    assert halo_km == pytest.approx(7 * np.median(grid_ds['dxC'].values) / 1e3)
    assert 12.0 < halo_km < 14.0                       # §4.2: "roughly 12-14 km"
    raw = llc_native_grid_halo_mask(grid_ds['hFacC'] == 0, grid_ds['dxC'], grid_ds['dyC'],
                                    halo_km)
    assert raw.shape == (1, 720, 720)
    h = mk.halo_mask(grid_ds, 7)
    assert h.shape == (720, 720) and np.array_equal(h, raw[0])
    d = mk.coast_distance_km(grid_ds)
    assert np.array_equal(h, oc & (np.nan_to_num(d, nan=-1.0) >= halo_km))
    assert np.nanmin(d[h]) >= halo_km > np.nanmax(d[oc & ~h])


@pytest.mark.needs_grid
def test_halo_width_real_grid(grid_ds):
    """Every retained cell is at least ``halo_cells`` (taxicab) from land,
    and the halo is not wider than it needs to be."""
    oc = mk.ocean_mask(grid_ds)
    h = mk.halo_mask(grid_ds, 7)
    taxi = ndimage.distance_transform_cdt(oc, metric='taxicab')
    assert taxi[h].min() >= 7
    assert taxi[h].min() <= 8                          # halo_cells or halo_cells + 1
    assert taxi[oc & ~h].max() <= 10                   # diagonal reach of a Euclidean halo
    assert 0.9 < h.sum() / oc.sum() < 1.0


@pytest.mark.needs_grid
def test_offshore_and_analysis_counts(grid_ds):
    oc = mk.ocean_mask(grid_ds)
    off = mk.offshore_mask(grid_ds, 100.0)
    ana = mk.analysis_mask(grid_ds)
    h = mk.halo_mask(grid_ds)
    assert not (off & ~h).any()                        # 100 km cut contains the 12.6 km halo
    assert not (ana & ~off).any() and not (ana & ~mk.edge_mask(grid_ds)).any()
    assert 0.5 < ana.sum() / oc.sum() < 0.9
    # inland water the model treats as ocean (Gulf of California, Salton Sea,
    # the Delta) is gone: one connected component survives
    _, n = ndimage.label(ana)
    assert n == 1


@pytest.mark.needs_grid
def test_gulf_of_california_removed_by_offshore_cut(grid_ds):
    """The tile's east edge clips the Gulf; no polygon -- the >= 100 km cut
    must remove it on its own."""
    from m1_write_masks import gulf_of_california_check
    masks = mk.build_masks(grid_ds)
    res = gulf_of_california_check(grid_ds, masks)
    assert res['gulf_is_separate_from_pacific']
    assert res['n_gulf'] > 5000
    assert res['gulf_max_coast_km'] < 100.0
    assert res['gulf_n_ge_offshore'] == 0 and res['gulf_n_analysis'] == 0


@pytest.mark.needs_grid
def test_edge_margin_covers_crop_test_rim(grid_ds):
    """``m0_qa_checks.check_edge_rim``: the rows/cols from each tile edge
    whose G or Jacobian changes when the edge moves.  The margin must cover
    every one of them (measured: G 1 cell everywhere, Jacobian 1 low / 2
    high), and the default 7 has room for the filter half-width."""
    from validate import _snapshot_fields
    import m0_qa_checks as qc
    ds, grid, b, G = _snapshot_fields(grid_ds)
    rim = qc.check_edge_rim(ds, grid, b)
    offsets = {k: v for r in ('G_components', 'jacobian')
               for k, v in rim[r].items() if k in ('low_j', 'high_j', 'low_i', 'high_i')}
    rim_max = max(max(v) for v in offsets.values() if v)
    assert rim_max == 1                                # Jacobian, high edges: offsets [0, 1]
    for edge_cells in (2, 7):
        e = mk.edge_mask(grid_ds, edge_cells)
        assert rim_max < edge_cells
        assert not e[:rim_max + 1].any() and not e[-(rim_max + 1):].any()
        assert not e[:, :rim_max + 1].any() and not e[:, -(rim_max + 1):].any()
    # the rim is finite and wrong, not NaN: median |G| at offset 0 relative
    # to offsets 1-3 is ~1e6x on the low edges (diff against the 0 fill)
    # and ~0.5x on the high edges (interp with it); offset 2 is clean
    from m0_qa_plot import edge_profile
    prof = edge_profile(G, mk.ocean_mask(grid_ds))
    assert np.isfinite(G[0, :]).any() and np.isfinite(G[:, 0]).any()
    assert prof['low j (j=0)'][0] > 1e3 and prof['low i (i=2880)'][0] > 1e3
    assert prof['high j (j=719)'][0] < 0.8 and prof['high i (i=3599)'][0] < 0.8
    for name in prof:
        assert 0.7 < prof[name][2] < 1.4
