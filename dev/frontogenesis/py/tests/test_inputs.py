""" Tests for ``inputs.py``, the merged input layer (M3 task 1), offline.

Two synthetic stores mimic coding §3.2 (OSN: nine surface vars, staggered
``U``/``V``/``oceTAU*``, scalar ``face``/``k``/``k_l``, ``niter``) and §3.3
(chunk: ``Theta``/``Salt`` on ``k = 0..2``, ``W`` on ``k_l = 0..2``, the
three downward-positive fluxes with their sign attrs, ``drF``, ``Z``,
``Zl``, ``mit_iteration``) on a 16 x 16 synthetic C-grid with a land
block; the chunk's level 0 is bit-identical to the OSN surface, as the real
stores are (M2 tasks 4-6).  They are written to zarr under ``tmp_path`` so
the real open / merge path runs; defect variants are built in memory.  One
``needs_grid`` test opens the real stores and reproduces the V3 / M2
numbers (hour 0; 16 vars; the k = 0 identity; ``drF``; ``n_valid`` 262,925;
``n_front`` 26,293).
"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

import inputs as inp
import operators as op
import osn_tiles as ot
import synthetic as sy
from dbof.llc4320_ingestion.date_iterations import osn_date_to_iteration

N = 16
NH = 6
TS = [f'2012-07-02 {h:02d}:00:00' for h in range(NH)]
DRF = np.array([1.0, 1.14, 1.30], 'f4')
Z = np.array([-0.5, -1.57, -2.79], 'f4')
ZL = np.array([0.0, -1.0, -2.14], 'f4')
FLUX_ATTRS = {v: dict(units=u, sign_convention=f'positive downward (into the ocean); {s}',
                      source_sign_convention='(+=down)', sign_conversion='negated at write time',
                      forcing_note='6-hourly forcing, linearly interpolated')
              for v, u, s in (('oceQnet', 'W/m^2', '>0 increases theta'),
                              ('oceQsw', 'W/m^2', '>0 increases theta'),
                              ('oceFWflx', 'kg/m^2/s', '>0 decreases salinity'))}


# ---------------------------------------------------------------------------
# the synthetic tile: grid, masks, two stores
# ---------------------------------------------------------------------------
def _land():
    c = np.zeros((N, N), bool)
    c[:4, :5] = True                           # a land block in the corner
    w = c.copy(); w[4, :5] = True              # the U mask differs by a row (hFacW)
    s = c.copy(); s[:4, 5] = True              # the V mask by a column (hFacS)
    return dict(hFacC=c, hFacW=w, hFacS=s)


def make_grid():
    land = _land()
    g, grid, pos = sy.synthetic_cgrid(nj=N, ni=N, dx=1800.0, dy=1800.0, land=land['hFacC'])
    g['hFacW'] = (('face', 'j', 'i_g'), (~land['hFacW']).astype('f8')[None])
    g['hFacS'] = (('face', 'j_g', 'i'), (~land['hFacS']).astype('f8')[None])
    x, y = pos('c')
    g = g.assign_coords(XC=(('j', 'i'), (-125.0 + x[0] / 1e5).astype('f4')),
                        YC=(('j', 'i'), (36.0 + y[0] / 1e5).astype('f4')), face=('face', [10]))
    return g


def make_masks():
    land = _land()
    ana = ~land['hFacC'] & sy.inner((N, N), 3)[...]
    return xr.Dataset({'mask_ocean': (('j', 'i'), ~land['hFacC']),
                       'mask_analysis': (('j', 'i'), ana)},
                      coords={'j': np.arange(N), 'i': np.arange(N)})


def _field(ts, name, lo, hi, shape=(N, N)):
    """Deterministic per (hour, variable) float32 field, ocean values in [lo, hi]."""
    t = int(np.datetime64(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), 's').astype('int64'))
    rng = np.random.default_rng([t, abs(hash(name)) % 1000])
    return (lo + (hi - lo) * rng.random(shape)).astype('f4')


def make_stores():
    """``(osn, chunk)`` in memory with the §3.2 / §3.3 layout; the chunk's
    level 0 equals the OSN surface bit for bit, levels 1-2 differ."""
    land = _land()
    coords = {'time': np.array([np.datetime64(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), 'ns')
                                for ts in TS]),
              'niter': ('time', [osn_date_to_iteration(ts) for ts in TS]),
              'j': np.arange(N), 'i': 2880 + np.arange(N), 'face': 10}
    xc = (-125.0 + 1800.0 * np.arange(N)[None, :] / 1e5 + 0 * np.arange(N)[:, None]).astype('f4')
    yc = (36.0 + 1800.0 * np.arange(N)[:, None] / 1e5 + 0 * np.arange(N)[None, :]).astype('f4')
    coords.update(XC=(('j', 'i'), xc), YC=(('j', 'i'), yc))
    rng_ = {'Theta': (13.0, 18.0), 'Salt': (33.2, 33.9), 'U': (-0.3, 0.3), 'V': (-0.3, 0.3),
            'W': (-1e-4, 1e-4), 'Eta': (-0.5, 0.5), 'KPPhbl': (5.0, 40.0),
            'oceTAUX': (-0.1, 0.1), 'oceTAUY': (-0.1, 0.1)}
    stag = {'U': ('j', 'i_g'), 'oceTAUX': ('j', 'i_g'), 'V': ('j_g', 'i'), 'oceTAUY': ('j_g', 'i')}
    osn_vars, surf = {}, {}
    for v, (lo, hi) in rng_.items():
        a = np.stack([_field(ts, v, lo, hi) for ts in TS])
        # land NaN: U on hFacW, V on hFacS, everything else -- oceTAU* included, as the
        # real store delivers them (M0 task 3) -- on the centred mask
        m = {'U': land['hFacW'], 'V': land['hFacS']}.get(v, land['hFacC'])
        a[:, m] = np.nan
        osn_vars[v] = (('time',) + stag.get(v, ('j', 'i')), a, {'units': 'x'})
        surf[v] = a
    osn = xr.Dataset(osn_vars, coords={**coords, 'i_g': 2880 + np.arange(N), 'j_g': np.arange(N),
                                       'k': 0, 'k_l': 0})
    # the chunk store: level 0 is the OSN surface, levels 1-2 a smooth modification
    def levels(v):
        a = np.stack([surf[v], surf[v] - 0.3 * (1 if v != 'W' else 1e-4),
                      surf[v] - 0.6 * (1 if v != 'W' else 1e-4)], axis=1).astype('f4')
        a[:, :, land['hFacC']] = np.nan
        return a
    ch_vars = {'Theta': (('time', 'k', 'j', 'i'), levels('Theta')),
               'Salt': (('time', 'k', 'j', 'i'), levels('Salt')),
               'W': (('time', 'k_l', 'j', 'i'), levels('W'), {'source_dim': 'k_p1'})}
    for v, (lo, hi) in (('oceQnet', (-150.0, 450.0)), ('oceQsw', (0.0, 600.0)),
                        ('oceFWflx', (-3e-5, -1e-5))):
        a = np.stack([_field(ts, v, lo, hi) for ts in TS])
        if v == 'oceQsw':
            a[:, 8, 8] = 0.0                       # the night-time "zero" lasts one instant
        a[:, land['hFacC']] = np.nan
        ch_vars[v] = (('time', 'j', 'i'), a, FLUX_ATTRS[v])
    ch_vars['drF'] = (('k',), DRF, {'units': 'm'})
    chunk = xr.Dataset(ch_vars, coords={**coords, 'k': np.arange(3), 'k_l': np.arange(3),
                                        'mit_iteration': ('time', [osn_date_to_iteration(ts) - 10368
                                                                   for ts in TS]),
                                        'Z': ('k', Z), 'Zl': ('k_l', ZL)})
    chunk.attrs.update(forcing_note=FLUX_ATTRS['oceQsw']['forcing_note'],
                       flux_sign_convention='positive downward', source='synthetic')
    osn.attrs.update(stores=['llc_surf', 'llc_wind'], source='synthetic')
    for d in (osn, chunk):
        for c, ax in (('j', 'Y'), ('i', 'X')):
            d[c].attrs['axis'] = ax
    return osn, chunk


@pytest.fixture(scope='module')
def stores(tmp_path_factory):
    """The two synthetic stores on disk (zarr), plus the grid and masks."""
    d = tmp_path_factory.mktemp('inputs')
    osn, chunk = make_stores()
    enc = {'time': {'units': 'seconds since 2011-09-10', 'dtype': 'int64'}}
    osn.to_zarr(d / 'raw.zarr', encoding=enc)
    chunk.to_zarr(d / 'chunk.zarr', encoding=enc)
    return dict(raw=d / 'raw.zarr', chunk=d / 'chunk.zarr', grid=make_grid(), masks=make_masks())


def open_(stores, **kw):
    return inp.open_inputs(stores['raw'], stores['chunk'], stores['grid'], stores['masks'], **kw)


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------
def test_merge_by_rename_schema(stores):
    ds, g, grid, masks = open_(stores)
    assert sorted(ds.data_vars) == sorted(inp.MERGED_VARS) and len(ds.data_vars) == 16
    assert set(ds.dims) == {'time', 'j', 'i', 'i_g', 'j_g', 'k', 'k_l'}
    assert ds.sizes['k'] == ds.sizes['k_l'] == 3 and ds.sizes['time'] == NH
    assert ds['Theta_k'].dims == ('time', 'k', 'j', 'i') and ds['W_k'].dims == ('time', 'k_l', 'j', 'i')
    assert ds['Theta'].dims == ('time', 'j', 'i') and ds['U'].dims == ('time', 'j', 'i_g')
    assert 'k' not in [c for c in ds.coords if ds[c].ndim == 0]       # the OSN scalar k is gone
    assert 'forcing_note' in ds.attrs and ds.attrs['k0_identity_checked'] == [0]   # lazy default
    assert 'rename' in ds.attrs['merge']
    np.testing.assert_allclose(inp.drF(ds), [1.0, 1.14, 1.30], rtol=1e-6)
    np.testing.assert_allclose(inp.Z(ds), [-0.5, -1.57, -2.79], rtol=1e-6)
    assert 'face' in g.dims and grid is not None and masks['mask_analysis'].dtype == bool


def test_plain_merge_raises(stores):
    """The M2 task 6 finding, pinned: Theta/Salt/W under one name, 2-D vs 3-D."""
    osn, chunk = xr.open_zarr(stores['raw']), xr.open_zarr(stores['chunk'])
    with pytest.raises(xr.MergeError):
        xr.merge([osn, chunk])


def test_hours_subset_checks_every_hour_requested(stores):
    ds, *_ = open_(stores, hours=slice(1, 4))
    assert ds.sizes['time'] == 3 and ds.attrs['k0_identity_checked'] == [0, 1, 2]
    ds_all, *_ = open_(stores, check_hours='all')
    assert ds_all.attrs['k0_identity_checked'] == list(range(NH))


def test_shared_coord_mismatch_raises(stores):
    osn, chunk = xr.open_zarr(stores['raw']), xr.open_zarr(stores['chunk'])
    shifted = chunk.assign_coords(time=chunk.time + np.timedelta64(1, 'h'))
    with pytest.raises(ValueError, match='time'):
        inp.open_inputs(osn, shifted, stores['grid'], stores['masks'])
    bad_face = chunk.assign_coords(face=11)
    with pytest.raises(ValueError, match='face'):
        inp.open_inputs(osn, bad_face, stores['grid'], stores['masks'])


# ---------------------------------------------------------------------------
# the k = 0 invariant
# ---------------------------------------------------------------------------
def _rolled_chunk(stores, hours):
    """A chunk store whose level-0 data at ``hours`` belong to the next hour
    (time coords untouched): the one-hour shift the identity must catch."""
    chunk = xr.open_zarr(stores['chunk']).load()
    for v in ('Theta', 'Salt', 'W'):
        a = chunk[v].values.copy()
        for h in hours:
            a[h] = chunk[v].values[(h + 1) % NH]
        chunk[v] = (chunk[v].dims, a, chunk[v].attrs)
    return chunk


def test_k0_identity_catches_a_store_shifted_by_one_hour(stores):
    osn = xr.open_zarr(stores['raw'])
    with pytest.raises(ValueError, match='k=0 identity'):
        inp.open_inputs(osn, _rolled_chunk(stores, range(NH)), stores['grid'], stores['masks'])


def test_k0_identity_lazy_policy(stores):
    """A defect at hour 3 only: the lazy open (hour 0) passes, 'all' refuses,
    and hour_pair refuses as soon as hour 3 is loaded."""
    osn = xr.open_zarr(stores['raw'])
    bad = _rolled_chunk(stores, [3])
    ds, *_ = inp.open_inputs(osn, bad, stores['grid'], stores['masks'])          # hour 0 fine
    assert ds.attrs['k0_identity_checked'] == [0]
    with pytest.raises(ValueError, match='time index 3'):
        inp.open_inputs(osn, bad, stores['grid'], stores['masks'], check_hours='all')
    with pytest.raises(ValueError, match='k=0 identity'):
        inp.open_inputs(osn, bad, stores['grid'], stores['masks'], hours=slice(2, 5))
    inp.hour_pair(ds, 0)                                                        # hours 0, 1: fine
    with pytest.raises(ValueError, match='k=0 identity'):
        inp.hour_pair(ds, 2)                                                    # loads hour 3
    assert inp.assert_k0_identity(ds, [0, 1, 2]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# the flux sign guard
# ---------------------------------------------------------------------------
def test_sign_guard_refuses_upward_positive_and_missing_attr(stores):
    osn, chunk = xr.open_zarr(stores['raw']), xr.open_zarr(stores['chunk']).load()
    up = chunk.copy()
    for v in inp.FLUX_VARS:                       # the source's convention: oceQsw <= 0
        up[v] = (-chunk[v]).assign_attrs(chunk[v].attrs)
    with pytest.raises(ValueError, match='oceQsw < 0'):
        inp.open_inputs(osn, up, stores['grid'], stores['masks'])
    no_attr = chunk.copy()
    no_attr['oceQnet'].attrs.pop('sign_convention')
    with pytest.raises(ValueError, match='sign_convention'):
        inp.open_inputs(osn, no_attr, stores['grid'], stores['masks'])
    wrong = chunk.copy()
    wrong['oceFWflx'].attrs['sign_convention'] = 'positive upward'
    with pytest.raises(ValueError, match='sign_convention'):
        inp.open_inputs(osn, wrong, stores['grid'], stores['masks'])
    # a single negative cell in a later hour: caught when that hour is loaded
    one = chunk.copy()
    a = one['oceQsw'].values.copy(); a[4, 9, 9] = -2.0
    one['oceQsw'] = (one['oceQsw'].dims, a, chunk['oceQsw'].attrs)
    ds, *_ = inp.open_inputs(osn, one, stores['grid'], stores['masks'])
    with pytest.raises(ValueError, match='oceQsw < 0'):
        inp.hour_pair(ds, 3)


def test_fluxes_as_stored_no_negation(stores):
    ds, *_ = open_(stores)
    qnet, qsw, fw = inp.fluxes(ds)
    for da, v in ((qnet, 'oceQnet'), (qsw, 'oceQsw'), (fw, 'oceFWflx')):
        assert da.name == v and np.array_equal(da.values, ds[v].values, equal_nan=True)
        assert da.attrs['forcing_note'] and da.attrs['sign_convention'].startswith('positive downward')
        assert 'do not negate' in da.attrs['negated_here']
    assert np.nanmin(qsw.values) >= 0 and np.nanmax(fw.values) < 0 and np.nanmax(qnet.values) > 0


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------
def test_W_k1_picks_k_l_1_never_0(stores):
    ds, *_ = open_(stores)
    w = inp.W_k1(ds)
    assert 'k_l' not in w.dims and 'k_l' not in w.coords and 'Zl' not in w.coords
    assert w.dims == ('time', 'j', 'i') and w.name == 'W_k1' and w.attrs['k_l'] == 1
    assert np.array_equal(w.values, ds['W_k'].isel(k_l=1).values, equal_nan=True)
    k0 = ds['W_k'].isel(k_l=0).values
    assert not np.array_equal(w.values, k0, equal_nan=True)
    assert np.array_equal(k0, ds['W'].values, equal_nan=True)         # k_l = 0 is the OSN W = dEta/dt
    assert abs(w.attrs['Zl_m'] + 1.0) < 1e-6
    with pytest.raises(ValueError):
        inp.W_k1(ds.assign(W_k=ds['W_k'].isel(k_l=1, drop=True)))      # no level dim: refuse


def test_wind_remask(stores):
    ds, g, *_ = open_(stores)
    land = _land()
    before = int(np.sum(np.isfinite(ds['oceTAUX'].values) & land['hFacW']))
    assert before > 0                                                  # finite on the U land, as stored
    tx, ty, kpp = inp.wind(ds, g)
    assert tx.dims == ds['oceTAUX'].dims and ty.dims == ds['oceTAUY'].dims
    assert int(np.sum(np.isfinite(tx.values) & land['hFacW'])) == 0
    assert int(np.sum(np.isfinite(ty.values) & land['hFacS'])) == 0
    assert np.array_equal(kpp.values, ds['KPPhbl'].values, equal_nan=True)
    assert tx.attrs['remasked_with'] == 'hFacW' and ty.attrs['remasked_with'] == 'hFacS'
    # the ocean values are untouched
    wet = ~land['hFacW']
    assert np.array_equal(tx.values[:, wet], ds['oceTAUX'].values[:, wet])
    # on an hour pair (face dim) too
    h0, _ = inp.hour_pair(ds, 0)
    tx0, _, _ = inp.wind(h0, g)
    assert tx0.dims == ('face', 'j', 'i_g') and int(np.sum(np.isfinite(tx0.values[0]) & land['hFacW'])) == 0


# ---------------------------------------------------------------------------
# hour pair, midpoint, time_mid
# ---------------------------------------------------------------------------
def test_hour_pair_midpoint_time_mid(stores):
    ds, *_ = open_(stores)
    h0, h1 = inp.hour_pair(ds, 2)
    for h, t in ((h0, 2), (h1, 3)):
        assert h['Theta'].dims == ('face', 'j', 'i') and h['U'].dims == ('face', 'j', 'i_g')
        assert h['Theta_k'].dims == ('face', 'k', 'j', 'i') and h.sizes['face'] == 1
        assert all(h[v].dtype == np.float64 for v in h.data_vars)
        assert h.attrs['time_index'] == t and h.attrs['time'] == TS[t].replace(' ', 'T')
        assert int(h['face'].values[0]) == 10
    assert np.array_equal(h0['Theta'].values[0], ds['Theta'].isel(time=2).values, equal_nan=True)
    b_mid = inp.midpoint(h0['Theta'], h1['Theta'])
    np.testing.assert_array_equal(b_mid.values, 0.5 * (h0['Theta'].values + h1['Theta'].values))
    assert inp.time_mid(ds, 2) == np.datetime64('2012-07-02T02:30:00')
    with pytest.raises(IndexError):
        inp.time_mid(ds, NH - 1)
    with pytest.raises(IndexError):
        inp.hour_pair(ds, NH - 1)
    gap = ds.assign_coords(time=ds.time.where(ds.time != ds.time[3], ds.time[3] + np.timedelta64(1, 'm')))
    with pytest.raises(ValueError, match='apart'):
        inp.time_mid(gap, 2)
    # Figure 6's axis: 00:30 UTC is 16.5 h local solar at lon -120.5
    assert abs(inp.local_solar_hour(inp.time_mid(ds, 0)) - 16.5) < 1e-9
    assert inp.drF(h0)[0] == 1.0 and inp.Z(h0)[0] == -0.5            # accessors on an hour too


# ---------------------------------------------------------------------------
# the same filter on b, U, V
# ---------------------------------------------------------------------------
def test_filtered_same_L_identity_at_zero(stores):
    ds, g, grid, _ = open_(stores)
    h0, _ = inp.hour_pair(ds, 0)
    b, U, V = inp.filtered(h0, 0)
    assert np.array_equal(b.values, op.buoyancy(h0).values, equal_nan=True)   # JMD95, unfiltered
    assert np.array_equal(U.values, h0['U'].values, equal_nan=True)
    assert np.array_equal(V.values, h0['V'].values, equal_nan=True)
    assert b.dims == ('face', 'j', 'i') and U.dims == ('face', 'j', 'i_g') and V.dims == ('face', 'j_g', 'i')
    assert all(f.attrs['L_cells'] == 0 for f in (b, U, V)) and 'lowpass_L_cells' not in b.attrs
    b2, U2, V2 = inp.filtered(h0, 2)
    for f, raw in ((b2, b), (U2, U), (V2, V)):
        assert f.attrs['lowpass_L_cells'] == 2 and f.attrs['L_cells'] == 2 and f.dims == raw.dims
        assert np.isnan(f.values[0, 0]).all() and np.isnan(f.values[0, -1]).all()   # the NaN rim
        inner = np.isfinite(f.values) & np.isfinite(raw.values)
        assert inner.sum() > 0 and not np.allclose(f.values[inner], raw.values[inner])
    # the mean over the finite interior is preserved to the level of the removed rim
    assert np.isfinite(b2.values[0, 8, 8])
    with pytest.raises(ValueError):
        inp.filtered(h0, 3)                                                  # odd L: lowpass refuses


# ---------------------------------------------------------------------------
# the reduction rule
# ---------------------------------------------------------------------------
def test_valid_drops_nan_and_counts(stores):
    masks = stores['masks']
    ana = masks['mask_analysis'].values
    n_ana = int(ana.sum())
    f = np.ones((N, N))
    v, lost = inp.valid(masks, f)
    assert v.dtype == bool and lost == 0 and v.sum() == n_ana
    f2 = f.copy()
    jj, ii = np.where(ana)
    f2[jj[:5], ii[:5]] = np.nan                     # five analysis cells lost
    f2[0, 0] = np.nan                               # one outside the mask: not counted
    v, lost = inp.valid(masks, f, f2)
    assert lost == 5 and v.sum() == n_ana - 5 and not v[jj[0], ii[0]]
    # DataArrays with a length-1 face dim, and a bare bool mask (the edge_cells = 13 sensitivity)
    da = xr.DataArray(f2[None], dims=('face', 'j', 'i'))
    v2, lost2 = inp.valid(ana, da)
    assert lost2 == 5 and np.array_equal(v2, v)
    with pytest.raises(ValueError):
        inp.valid(masks, np.ones((N + 1, N)))


# ---------------------------------------------------------------------------
# the real stores
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_real_stores_hour0_reproduces_v3_numbers():
    """Hour 0 opens; 16 vars (rename, M3-Q6); the k = 0 identity for all
    three variables; drF; n_valid at L = 0 = 262,925 and n_front (p90) =
    26,293 -- the V3 / M2 numbers."""
    for p in (inp.RAW_ZARR, inp.CHUNK_ZARR, inp.GRID_ZARR, inp.MASKS_NC):
        if not p.exists():
            pytest.skip(f'{p} not on disk')
    ds, g, grid, masks = inp.open_inputs()
    assert len(ds.data_vars) == 16 and sorted(ds.data_vars) == sorted(inp.MERGED_VARS)
    assert set(ds.dims) == {'time', 'j', 'i', 'i_g', 'j_g', 'k', 'k_l'}
    assert ds.sizes['time'] == 72 and ds.sizes['k'] == 3 and ds.attrs['k0_identity_checked'] == [0]
    np.testing.assert_allclose(inp.drF(ds), [1.0, 1.14, 1.30], rtol=1e-6)
    np.testing.assert_allclose(inp.Z(ds), [-0.5, -1.57, -2.79], rtol=1e-6)
    assert inp.assert_k0_identity(ds, [0, 1]) == [0, 1]                  # Theta, Salt, W at k = 0
    h0, h1 = inp.hour_pair(ds, 0)
    assert str(inp.time_mid(ds, 0))[:16] == '2012-07-02T00:30'
    assert h0['Theta'].dims == ('face', 'j', 'i') and h0['Theta'].dtype == np.float64
    # the midpoint fields at L = 0, G and the default (discrete) 2F
    b0, U0, V0 = inp.filtered(h0, 0)
    b1, U1, V1 = inp.filtered(h1, 0)
    b_mid, U_mid, V_mid = (inp.midpoint(a, b) for a, b in ((b0, b1), (U0, U1), (V0, V1)))
    G_mid = op.gradb2(b_mid, g, grid)
    two_F = 2.0 * op.frontogenesis(b_mid, U_mid, V_mid, g, grid)
    val, n_lost = inp.valid(masks, b_mid, G_mid, two_F)
    assert int(val.sum()) == 262_925 and n_lost == 0                      # n_valid at L = 0
    Gm = G_mid.values[0]
    front = val & (Gm >= np.percentile(Gm[val], 90.0))
    assert int(front.sum()) == 26_293                                     # n_front, p90
    # the accessors on the real hour
    w = inp.W_k1(h0)
    assert w.dims == ('face', 'j', 'i') and not np.array_equal(w.values, h0['W'].values, equal_nan=True)
    qnet, qsw, fw = inp.fluxes(h0)
    assert np.nanmin(qsw.values) >= 0 and np.nanmean(qnet.values) > 0    # 16 LST: heating, downward-positive
    assert 'forcing_note' in qsw.attrs
    land_w = g['hFacW'].values[0] == 0
    assert int(np.sum(np.isfinite(h0['oceTAUX'].values[0]) & land_w)) == 922    # M0 task 3
    tx, ty, kpp = inp.wind(h0, g)
    assert int(np.sum(np.isfinite(tx.values[0]) & land_w)) == 0
    assert int(np.sum(np.isfinite(ty.values[0]) & (g['hFacS'].values[0] == 0))) == 0
    print(f'\ninputs on the real stores: n_valid {int(val.sum())}, n_lost {n_lost}, n_front {int(front.sum())}, '
          f'drF {inp.drF(ds)}, Z {inp.Z(ds)}, time_mid {inp.time_mid(ds, 0)}')
