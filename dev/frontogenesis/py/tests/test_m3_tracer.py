""" Tests for ``m3_tracer.py`` (M3-digging task 4), offline on ``synthetic.py`` grids.

Pinned: the sub-step loop with the piecewise velocity and no clamp IS
``fvadvect.fv_advect`` (bit for bit); the time interpolation returns the
endpoint velocities at the endpoints and the midpoint for ``piecewise``;
``linear`` and ``piecewise`` differ on a time-varying flow and agree on a
steady one; the clamp resets exactly the margin cells; an hour dataset keeps
the raw layout with only the tracers replaced; the declared reading maps the
thresholds as written.
"""
import numpy as np
import xarray as xr

import fvadvect as fv
import m3_tracer as mt
import masking as mk
import synthetic as sy

NJ, NI = 48, 64


def case(seed=0, rotated=False):
    rng = np.random.default_rng(seed)
    land = np.zeros((NJ, NI), bool)
    land[:6, :10] = True
    g, grid, pos = sy.synthetic_cgrid(nj=NJ, ni=NI, dx=1800.0, dy=1800.0, rotated=rotated, land=land)
    g['hFacW'] = (('face', 'j', 'i_g'), (~land).astype('f8')[None])
    g['hFacS'] = (('face', 'j_g', 'i'), (~land).astype('f8')[None])
    x, y = pos('c')
    b = 1e-2 * np.tanh((x[0] - x[0].mean()) / (6 * 1800.0)) + 1e-3 * rng.normal(size=(NJ, NI))
    b[land] = np.nan
    U0 = xr.DataArray(0.2 + 0.05 * rng.normal(size=(1, NJ, NI)), dims=('face', 'j', 'i_g'))
    V0 = xr.DataArray(0.1 + 0.05 * rng.normal(size=(1, NJ, NI)), dims=('face', 'j_g', 'i'))
    U1, V1 = U0 * 0.5, V0 * 1.5
    return g, grid, b, U0, V0, U1, V1


def test_substep_loop_with_piecewise_velocity_is_fv_advect_bit_for_bit():
    g, grid, b, U0, V0, U1, V1 = case()
    tile = mt.Tile(g)
    bp = [fv._pad(tile.fill(np.where(tile.ocean & np.isfinite(b), b, np.nan)))]
    mt.advect_hour(bp, tile, U0, V0, U1, V1, 'piecewise', clamp=False)
    mine = np.where(tile.ocean, bp[0][fv.PAD:-fv.PAD, fv.PAD:-fv.PAD], np.nan)
    ref = np.squeeze(fv.fv_advect(xr.DataArray(b[None], dims=('face', 'j', 'i')), 0.5 * (U0 + U1),
                                  0.5 * (V0 + V1), g, dt=mt.DT).values)
    assert np.array_equal(mine, ref, equal_nan=True)
    assert np.isfinite(mine[tile.ocean]).all()


def test_velocity_interpolation_endpoints_and_midpoint():
    g, grid, b, U0, V0, U1, V1 = case()
    u, v = mt.velocity_at(U0, V0, U1, V1, 0.0, 'linear')
    assert np.allclose(u, U0) and np.allclose(v, V0)
    u, v = mt.velocity_at(U0, V0, U1, V1, 1.0, 'linear')
    assert np.allclose(u, U1) and np.allclose(v, V1)
    u, v = mt.velocity_at(U0, V0, U1, V1, 0.1, 'piecewise')
    assert np.allclose(u, 0.5 * (U0 + U1)) and np.allclose(v, 0.5 * (V0 + V1))


def test_linear_and_piecewise_differ_on_a_varying_flow_and_agree_on_a_steady_one():
    g, grid, b, U0, V0, U1, V1 = case()
    tile = mt.Tile(g)
    out = {}
    for vel in ('linear', 'piecewise'):
        bp = [fv._pad(tile.fill(np.where(tile.ocean & np.isfinite(b), b, np.nan)))]
        mt.advect_hour(bp, tile, U0, V0, U1, V1, vel, clamp=False)
        out[vel] = bp[0].copy()
    assert not np.array_equal(out['linear'], out['piecewise'])
    for vel in ('linear', 'piecewise'):
        bp = [fv._pad(tile.fill(np.where(tile.ocean & np.isfinite(b), b, np.nan)))]
        mt.advect_hour(bp, tile, U0, V0, U0, V0, vel, clamp=False)        # steady
        out[vel] = bp[0].copy()
    assert np.allclose(out['linear'], out['piecewise'], rtol=0, atol=1e-15)


def test_clamp_resets_exactly_the_margin_cells():
    g, grid, b, U0, V0, U1, V1 = case()
    tile = mt.Tile(g)
    assert tile.margin.sum() > 0 and not (tile.margin & tile.interior).any()
    assert np.array_equal(tile.interior, mk.edge_mask(g, mt.CLAMP_CELLS))
    m0 = np.where(tile.ocean, 100.0, np.nan)
    m1 = np.where(tile.ocean, 200.0, np.nan)
    bp = [fv._pad(tile.fill(np.where(tile.ocean & np.isfinite(b), b, np.nan)))]
    mt.advect_hour(bp, tile, U0 * 0, V0 * 0, U0 * 0, V0 * 0, 'piecewise', [m0], [m1], clamp=True)
    inner = bp[0][fv.PAD:-fv.PAD, fv.PAD:-fv.PAD]
    assert np.allclose(inner[tile.margin], 200.0)                       # the model at the hour's end
    assert np.all(np.abs(inner[tile.interior & tile.ocean]) < 1.0)      # untouched by the clamp (zero flow)


def test_hour_dataset_keeps_the_raw_layout_with_only_the_tracers_replaced():
    nt, nj, ni = 3, 8, 8
    t = np.array(['2012-07-02T00', '2012-07-02T01', '2012-07-02T02'], 'datetime64[ns]')
    raw = xr.Dataset({k: (('time', 'j', 'i'), np.random.default_rng(0).normal(size=(nt, nj, ni)).astype('f4'))
                      for k in ('Theta', 'Salt', 'U', 'V', 'W', 'Eta', 'KPPhbl', 'oceTAUX', 'oceTAUY')},
                     coords=dict(time=t, face=10))
    tr = dict(Theta=np.full((nj, ni), 5.0), Salt=np.full((nj, ni), 33.0))
    ds = mt.hour_dataset(raw, 1, tr)
    assert ds.sizes['time'] == 1 and float(ds['Theta'].values.mean()) == 5.0 and ds['Theta'].dtype == 'float32'
    assert np.array_equal(ds['U'].values, raw['U'].isel(time=[1]).values)
    assert 'offline passive tracer' in ds['Theta'].attrs['tracer']
    assert sorted(ds.data_vars) == sorted(raw.data_vars)


def test_declared_reading_maps_the_thresholds():
    assert mt.reading_L0(0.6) == 'analysis-limited at the grid scale'
    assert mt.reading_L0(0.2).startswith('data-limited')
    assert mt.reading_L0(0.4) == 'both'
    assert mt.reading_L8(0.2) == 'not the pipeline' and mt.reading_L8(0.3) != 'not the pipeline'
    assert mt.DECLARED['reading']['thresholds'] == dict(hi=0.5, lo=0.25)
    assert mt.TAG == dict(linear='tile330tracer', piecewise='tile330tracerpc')
