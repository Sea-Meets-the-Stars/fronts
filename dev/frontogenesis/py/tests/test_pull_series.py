""" Tests for ``osn_tiles.pull_series``, ``zarr_series`` and
``series_verify.verify_series`` (M2 task 1), offline.

The two hourly loaders are monkeypatched with synthetic 12 x 12 hours that
carry the real schema (dims, staggering, scalar coords, ``niter``, land NaN
from a synthetic grid); nothing deeper is mocked.  One ``network`` test
pulls a real hour and is deselected by default (``pytest.ini``); run it with
``-m network``.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import osn_tiles as ot
import zarr_series as zs
import series_verify as sv
from dbof.llc4320_ingestion.date_iterations import osn_date_to_iteration

N = 12
TS = [f'2012-07-02 {h:02d}:00:00' for h in range(6)]
STAGGER = {'U': ('j', 'i_g'), 'oceTAUX': ('j', 'i_g'), 'V': ('j_g', 'i'), 'oceTAUY': ('j_g', 'i')}
COMODO = {'j': {'axis': 'Y'}, 'i': {'axis': 'X'},
          'j_g': {'axis': 'Y', 'c_grid_axis_shift': -0.5},
          'i_g': {'axis': 'X', 'c_grid_axis_shift': -0.5}}


# ---------------------------------------------------------------------------
# synthetic grid and hours
# ---------------------------------------------------------------------------
def _masks():
    hFacC = np.ones((N, N), 'f4'); hFacC[:3, :4] = 0          # a land block
    hFacW = hFacC.copy(); hFacW[3, :4] = 0                      # U mask differs by a row
    hFacS = hFacC.copy(); hFacS[:3, 4] = 0                      # V mask by a column
    return {'hFacC': hFacC, 'hFacW': hFacW, 'hFacS': hFacS}


@pytest.fixture(scope='module')
def grid():
    m = _masks()
    rng = np.random.default_rng(0)
    idx = {d: np.arange(N) for d in ('j', 'i', 'j_g', 'i_g')}
    g = xr.Dataset({'hFacC': (('j', 'i'), m['hFacC']), 'hFacW': (('j', 'i_g'), m['hFacW']),
                    'hFacS': (('j_g', 'i'), m['hFacS'])},
                   coords={**idx, 'XC': (('j', 'i'), rng.random((N, N), dtype='f4')),
                           'YC': (('j', 'i'), rng.random((N, N), dtype='f4')), 'face': 10})
    return g


def make_hour(ts, names):
    """A loader-shaped hour: ``(time, face, j, i)`` float32, land NaN from
    the synthetic masks, values a deterministic function of ``ts`` so a
    store can be compared with the truth."""
    m = _masks()
    t = np.datetime64(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), 's')
    data = {}
    for v in names:
        # seeded per (hour, variable) so truth(ts, v) matches the full hour
        rng = np.random.default_rng([int(t.astype('int64')), (ot.CORE_VARS + ot.WIND_VARS).index(v)])
        d = rng.random((1, 1, N, N), dtype='f4') + 1.0     # never all-NaN, never 0
        land = {'U': m['hFacW'], 'V': m['hFacS']}.get(v, m['hFacC']) == 0
        d[:, :, land] = np.nan
        data[v] = (('time', 'face') + STAGGER.get(v, ('j', 'i')), d, {'units': 'x'})
    ds = xr.Dataset(data, coords={'time': np.array([t]), 'face': ('face', [10]),
                                  **{d: np.arange(N) for d in ('j', 'i', 'j_g', 'i_g')},
                                  'k': 0, 'k_l': 0,
                                  'niter': ('time', [osn_date_to_iteration(ts)])})
    for c, a in COMODO.items():
        ds[c].attrs.update(a)
    return ds


@pytest.fixture
def fake_loaders(monkeypatch):
    """Monkeypatch the two OSN loaders; returns the call log."""
    calls = []
    def core(ts, tile=None, endpoint=ot.OSN_ENDPOINT, **kw):
        calls.append(('core', ts)); return make_hour(ts, ot.CORE_VARS)
    def wind(ts, tile=None, endpoint=ot.OSN_ENDPOINT, **kw):
        calls.append(('wind', ts)); return make_hour(ts, ot.WIND_VARS)
    monkeypatch.setattr(ot, 'load_hour', core)
    monkeypatch.setattr(ot, 'load_wind_hour', wind)
    return calls


def snapshot(path):
    """``{relpath: sha256}`` of every file in the store."""
    out = {}
    for r, _, fs in os.walk(path):
        for f in fs:
            p = Path(r) / f
            out[str(p.relative_to(path))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def truth(ts, v):
    return make_hour(ts, [v])[v].values[0, 0]


def pull(ts, out, grid, **kw):
    rep = {}
    ot.pull_series(ts, out, grid_ds=grid, sleep=lambda s: None, report=rep, **kw)
    return rep


# ---------------------------------------------------------------------------
# the fresh pull: schema, dtype, chunking, encoding
# ---------------------------------------------------------------------------
def test_fresh_pull_schema(tmp_path, grid, fake_loaders):
    out = tmp_path / 's.zarr'
    rep = pull(TS, out, grid)
    assert rep['pulled'] == TS and not rep['failed'] and not rep['skipped']
    ds = xr.open_zarr(out)
    assert dict(ds.sizes) == {'time': 6, 'j': N, 'i': N, 'i_g': N, 'j_g': N}
    assert set(ds.data_vars) == set(ot.CORE_VARS + ot.WIND_VARS)
    for v in ds.data_vars:
        assert ds[v].dtype == np.float32
        assert tuple(ds[v].encoding['chunks']) == (1, N, N)
        assert ds[v].dims == ('time',) + STAGGER.get(v, ('j', 'i'))
    assert ds['time'].encoding['units'] == 'seconds since 2011-09-10'
    assert np.dtype(ds['time'].encoding['dtype']) == np.int64
    assert {'XC', 'YC', 'niter', 'face', 'k', 'k_l'} <= set(ds.coords)
    assert ds.face.ndim == 0 and int(ds.face) == 10
    assert ds.attrs['iterations'] == [osn_date_to_iteration(t) for t in TS]
    assert ds.attrs['timestamps'] == TS and ds.attrs['stores'] == ['llc_surf', 'llc_wind']
    assert ds['j_g'].attrs['c_grid_axis_shift'] == -0.5
    # values are the loaders' values
    for k, ts in enumerate(TS):
        for v in ('Theta', 'U', 'oceTAUY'):
            np.testing.assert_array_equal(ds[v].isel(time=k).values, truth(ts, v))
    # one chunk file per hour per variable
    assert sorted(os.listdir(out / 'Theta' / 'c')) == [str(k) for k in range(6)]
    assert sorted(os.listdir(out / 'time' / 'c')) == [str(k) for k in range(6)]
    assert sv.verify_series(out, TS, grid_ds=grid)['ok']


@pytest.mark.needs_grid
def test_encoding_parity_with_m0_store(tmp_path, grid, fake_loaders):
    """The appended store must carry the same per-variable encoding as M0's
    ``write_raw`` product: dtype, codecs, fill value, time units."""
    m0 = ot.DATA_DIR / 'tile330_raw_20120702T00_2h.zarr'
    if not m0.exists():
        pytest.skip('M0 two-hour store not on disk')
    out = tmp_path / 's.zarr'
    pull(TS[:2], out, grid)
    for v in ot.CORE_VARS + ot.WIND_VARS + ('time', 'niter', 'face', 'XC'):
        a = json.load(open(m0 / v / 'zarr.json'))
        b = json.load(open(out / v / 'zarr.json'))
        for key in ('data_type', 'codecs', 'fill_value', 'chunk_key_encoding', 'dimension_names'):
            assert a.get(key) == b.get(key), (v, key, a.get(key), b.get(key))
        assert a['attributes'].get('_FillValue') == b['attributes'].get('_FillValue'), v
        if v in ot.CORE_VARS + ot.WIND_VARS:
            assert a['chunk_grid']['configuration']['chunk_shape'][0] == 1 == \
                b['chunk_grid']['configuration']['chunk_shape'][0]
            assert set(a['attributes']['coordinates'].split()) == \
                set(b['attributes']['coordinates'].split()), v
    assert json.load(open(m0 / 'time' / 'zarr.json'))['attributes'] == \
        json.load(open(out / 'time' / 'zarr.json'))['attributes']


# ---------------------------------------------------------------------------
# resumability
# ---------------------------------------------------------------------------
def test_rerun_is_noop_and_byte_identical(tmp_path, grid, fake_loaders):
    out = tmp_path / 's.zarr'
    pull(TS, out, grid)
    before = snapshot(out)
    n_calls = len(fake_loaders)
    rep = pull(TS, out, grid)
    assert rep['pulled'] == [] and rep['skipped'] == TS and rep['repaired'] == 0
    assert len(fake_loaders) == n_calls, 'no-op re-run must not touch the network'
    assert snapshot(out) == before


def test_resume_between_hours(tmp_path, grid, fake_loaders, monkeypatch):
    """A crash between two hours (BaseException, so it is not a retry)."""
    class Crash(BaseException):
        pass
    core = ot.load_hour
    def crashing(ts, *a, **kw):
        if ts == TS[3]:
            raise Crash('killed')
        return core(ts, *a, **kw)
    monkeypatch.setattr(ot, 'load_hour', crashing)
    out = tmp_path / 's.zarr'
    with pytest.raises(Crash):
        pull(TS, out, grid)
    assert len(zs.present_times(out)) == 3
    before = snapshot(out)
    monkeypatch.setattr(ot, 'load_hour', core)
    rep = pull(TS, out, grid)
    assert rep['skipped'] == TS[:3] and rep['pulled'] == TS[3:] and rep['repaired'] == 0
    after = snapshot(out)
    # hours 0-2 were not rewritten
    for p, h in before.items():
        if '/c/' in p and not p.startswith('zarr.json'):
            assert after[p] == h, p
    ds = xr.open_zarr(out)
    for k, ts in enumerate(TS):
        np.testing.assert_array_equal(ds.Salt.isel(time=k).values, truth(ts, 'Salt'))
    assert sv.verify_series(out, TS, grid_ds=grid)['ok']


def test_resume_after_crash_inside_append(tmp_path, grid, fake_loaders, monkeypatch):
    """A crash *inside* ``to_zarr(append_dim='time')``: xarray writes ``time``
    first, so the store is left with ``time`` one hour longer than most
    variables.  The resume must truncate that hour, not skip it."""
    out = tmp_path / 's.zarr'
    pull(TS[:3], out, grid)
    good = snapshot(out)
    class Crash(BaseException):
        pass
    orig = zarr.Array.__setitem__
    writes = []
    def boom(self, key, value):
        writes.append(self.path)
        if len(writes) == 3:                         # time + two variables written
            raise Crash('killed mid-append')
        return orig(self, key, value)
    monkeypatch.setattr(zarr.Array, '__setitem__', boom)
    with pytest.raises(Crash):
        pull(TS[:4], out, grid)
    monkeypatch.setattr(zarr.Array, '__setitem__', orig)
    g = zarr.open_group(out, mode='r', use_consolidated=False)
    lengths = {n: a.shape[0] for n, a in g.arrays() if (a.metadata.dimension_names or [''])[0] == 'time'}
    # genuinely half-written: some arrays extended, some not (which ones
    # depends on xarray's write order, which the repair does not rely on)
    assert 4 in lengths.values() and 3 in lengths.values(), lengths
    # repair, then the re-run appends hour 3 properly
    rep = pull(TS[:5], out, grid)
    assert rep['repaired'] == 1
    assert rep['skipped'] == TS[:3] and rep['pulled'] == TS[3:5]
    ds = xr.open_zarr(out)
    assert ds.sizes['time'] == 5
    for k, ts in enumerate(TS[:5]):
        for v in ot.CORE_VARS + ot.WIND_VARS:
            np.testing.assert_array_equal(ds[v].isel(time=k).values, truth(ts, v))
    assert ds.attrs['iterations'] == [osn_date_to_iteration(t) for t in TS[:5]]
    after = snapshot(out)
    assert all(after[p] == h for p, h in good.items() if '/c/' in p)
    assert sv.verify_series(out, TS[:5], grid_ds=grid)['ok']


def _resize_to(out, names, n):
    g = zarr.open_group(out, mode='r+', use_consolidated=False)
    for v in names:
        a = g[v]; a.resize((n,) + a.shape[1:])


@pytest.mark.parametrize('defect', ['time_extended_vars_not', 'vars_extended_time_not',
                                    'chunk_missing', 'chunk_corrupt', 'chunk_all_fill'])
def test_repair_trailing_defects(tmp_path, grid, fake_loaders, defect):
    """Truncated stores built directly, one per class of half-written hour."""
    out = tmp_path / 's.zarr'
    pull(TS[:3], out, grid)
    good = snapshot(out)
    g = zarr.open_group(out, mode='r+', use_consolidated=False)
    if defect == 'time_extended_vars_not':
        _resize_to(out, ['time', 'niter', 'Theta'], 4)
        g['time'][3] = g['time'][2] + 3600; g['niter'][3] = g['niter'][2] + 144
        g['Theta'][3] = truth(TS[3], 'Theta')[None]
    elif defect == 'vars_extended_time_not':
        _resize_to(out, ['Theta', 'U', 'KPPhbl'], 4)
        g['Theta'][3] = truth(TS[3], 'Theta')[None]
    else:
        # every array extended and written, then one chunk spoiled
        names = [n for n, a in g.arrays() if (a.metadata.dimension_names or [''])[0] == 'time']
        _resize_to(out, names, 4)
        g = zarr.open_group(out, mode='r+', use_consolidated=False)
        g['time'][3] = g['time'][2] + 3600; g['niter'][3] = g['niter'][2] + 144
        for v in ot.CORE_VARS + ot.WIND_VARS:
            g[v][3] = truth(TS[3], v)[None]
        chunk = out / 'Salt' / 'c' / '3' / '0' / '0'
        assert chunk.exists()
        if defect == 'chunk_missing':
            chunk.unlink()
        elif defect == 'chunk_corrupt':
            chunk.write_bytes(chunk.read_bytes()[:7])
        else:
            g['Salt'][3] = np.full((1, N, N), np.nan, 'f4')   # all fill -> no chunk written
    assert zs.repair_trailing(out) == 1
    assert len(zs.present_times(out)) == 3
    assert snapshot(out) == good or {p: h for p, h in snapshot(out).items() if '/c/' in p} == \
        {p: h for p, h in good.items() if '/c/' in p}
    assert zs.repair_trailing(out) == 0                # idempotent
    rep = pull(TS[:4], out, grid)
    assert rep['pulled'] == [TS[3]] and rep['skipped'] == TS[:3]
    assert sv.verify_series(out, TS[:4], grid_ds=grid)['ok']


def test_clobber(tmp_path, grid, fake_loaders):
    out = tmp_path / 's.zarr'
    pull(TS, out, grid)
    rep = pull(TS[:2], out, grid, clobber=True)
    assert rep['pulled'] == TS[:2]
    assert len(zs.present_times(out)) == 2
    assert not (out / 'Theta' / 'c' / '5').exists()


def test_duplicate_and_out_of_order_raise(tmp_path, grid, fake_loaders):
    out = tmp_path / 's.zarr'
    with pytest.raises(ValueError, match='duplicated'):
        pull([TS[0], TS[1], TS[1]], out, grid)
    with pytest.raises(ValueError, match='increasing'):
        pull([TS[1], TS[0]], out, grid)
    assert not out.exists() and fake_loaders == []
    # earlier than the store's last hour and not in it: cannot be appended
    pull(TS[2:4], out, grid)
    with pytest.raises(ValueError, match='earlier'):
        pull([TS[0]], out, grid)
    # a sub-window that is already present is a plain no-op
    assert pull([TS[2]], out, grid)['skipped'] == [TS[2]]


# ---------------------------------------------------------------------------
# retries and the gap policy
# ---------------------------------------------------------------------------
def test_retry_then_success(tmp_path, grid, fake_loaders, monkeypatch):
    wind = ot.load_wind_hour
    fails = {'n': 0}
    def flaky(ts, *a, **kw):
        if ts == TS[1] and fails['n'] < 2:
            fails['n'] += 1
            raise OSError('OSN hiccup')
        return wind(ts, *a, **kw)
    monkeypatch.setattr(ot, 'load_wind_hour', flaky)
    out = tmp_path / 's.zarr'
    slept = []
    rep = {}
    ot.pull_series(TS[:3], out, grid_ds=grid, report=rep, backoff=(0.01, 0.02, 0.03),
                   sleep=slept.append)
    assert slept == [0.01, 0.02]
    assert rep['pulled'] == TS[:3] and not rep['failed']
    assert sv.verify_series(out, TS[:3], grid_ds=grid)['ok']


def test_persistent_failure_stops_at_gap(tmp_path, grid, fake_loaders, monkeypatch, caplog):
    core = ot.load_hour
    def dead(ts, *a, **kw):
        if ts == TS[2]:
            raise OSError('down')
        return core(ts, *a, **kw)
    monkeypatch.setattr(ot, 'load_hour', dead)
    out = tmp_path / 's.zarr'
    lines = []
    rep = {}
    with caplog.at_level('INFO'):
        ot.pull_series(TS, out, grid_ds=grid, report=rep, attempts=3, backoff=(0,),
                       sleep=lambda s: None, log=lines.append)
    assert rep['pulled'] == TS[:2]
    assert rep['failed'] == [TS[2]]
    assert rep['not_attempted'] == TS[3:]
    assert sum(1 for c in fake_loaders if c == ('core', TS[2])) == 0   # dead replaced it
    assert len(zs.present_times(out)) == 2                           # no gap on disk
    assert any('FAILED' in l and TS[2] in l for l in lines)
    assert any('giving up after 3' in r.message for r in caplog.records)
    v = sv.verify_series(out, TS, grid_ds=grid)
    assert not v['ok'] and v['time']['missing'] == [t.replace(' ', 'T') for t in TS[2:]]
    # the next run, with the source back, finishes the series
    monkeypatch.setattr(ot, 'load_hour', core)
    rep = pull(TS, out, grid)
    assert rep['skipped'] == TS[:2] and rep['pulled'] == TS[2:]
    assert sv.verify_series(out, TS, grid_ds=grid)['ok']


# ---------------------------------------------------------------------------
# verify_series on each class of defect
# ---------------------------------------------------------------------------
@pytest.fixture
def good_store(tmp_path, grid, fake_loaders):
    out = tmp_path / 's.zarr'
    pull(TS, out, grid)
    return out


def test_verify_good(good_store, grid):
    r = sv.verify_series(good_store, TS, grid_ds=grid)
    assert r['ok'] and all(r[k]['ok'] for k in ('time', 'schema', 'land_nan', 'niter', 'KPPhbl'))
    assert r['land_nan']['hours_checked'] == 6 and r['niter']['steps'] == [144]
    assert 'OK' in sv.summarize(r)


def test_verify_gap_and_duplicate(good_store, grid):
    r = sv.verify_series(good_store, TS + ['2012-07-02 06:00:00'], grid_ds=grid)
    assert not r['ok'] and r['time']['missing'] == ['2012-07-02T06:00:00']
    r = sv.verify_series(good_store, TS[:5], grid_ds=grid)
    assert not r['ok'] and r['time']['extra'] == ['2012-07-02T05:00:00']
    g = zarr.open_group(good_store, mode='r+', use_consolidated=False)
    g['time'][5] = g['time'][4]                       # duplicate hour, out of order
    zarr.consolidate_metadata(good_store)
    r = sv.verify_series(good_store, TS, grid_ds=grid)
    assert r['time']['duplicates'] == ['2012-07-02T04:00:00'] and not r['time']['ordered']


def test_verify_schema_defects(good_store, grid):
    ds = xr.open_zarr(good_store).load()
    bad = ds.copy()
    bad['Theta'] = bad['Theta'].astype('float64')
    bad = bad.drop_vars('KPPhbl')
    bad['time'].encoding = {'units': 'hours since 2012-07-02', 'dtype': 'int64'}
    out = good_store.parent / 'bad.zarr'
    enc = {v: {'chunks': (2, N, N)} for v in bad.data_vars}
    bad.to_zarr(out, mode='w', encoding=enc)
    r = sv.verify_series(out, TS, grid_ds=grid)
    assert not r['ok'] and not r['schema']['ok'] and not r['KPPhbl']['ok']
    msgs = ' '.join(r['schema']['problems'])
    assert 'float64' in msgs and 'chunks' in msgs and 'KPPhbl' in msgs and 'time encoding' in msgs


def test_verify_land_nan_and_kpp(good_store, grid):
    g = zarr.open_group(good_store, mode='r+', use_consolidated=False)
    slab = g['Theta'][2]; slab[0, 0] = 1.0; g['Theta'][2] = slab          # finite on land
    slab = g['KPPhbl'][4]; slab[6, 6] = np.nan; g['KPPhbl'][4] = slab     # NaN on ocean
    r = sv.verify_series(good_store, TS, grid_ds=grid)
    assert not r['ok'] and not r['land_nan']['ok']
    assert r['land_nan']['max_mismatch_cells']['Theta'] == 1
    assert r['land_nan']['first_bad_hour']['Theta'] == '2012-07-02T02:00:00'
    assert r['KPPhbl']['nonfinite_on_ocean_hours'] == ['2012-07-02T04:00:00']
    assert r['time']['ok'] and r['schema']['ok']


def test_verify_niter_step(good_store, grid):
    g = zarr.open_group(good_store, mode='r+', use_consolidated=False)
    g['niter'][3] = g['niter'][3] + 1
    r = sv.verify_series(good_store, TS, grid_ds=grid)
    assert not r['ok'] and not r['niter']['ok']
    assert 145 in r['niter']['steps'] and not r['niter']['matches_timestamps']


def test_verify_missing_store(tmp_path, grid):
    r = sv.verify_series(tmp_path / 'none.zarr', TS, grid_ds=grid)
    assert not r['ok'] and 'does not exist' in r['error']


# ---------------------------------------------------------------------------
# one real hour (deselected by default)
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_network_one_real_hour(tmp_path):
    """Pull ``2012-07-02 02:00:00`` from OSN into a temp store and verify it."""
    import time
    ts = ['2012-07-02 02:00:00']
    out = tmp_path / 'one.zarr'
    rep = {}
    t0 = time.time()
    ot.pull_series(ts, out, report=rep)
    wall = time.time() - t0
    assert rep['pulled'] == ts and not rep['failed']
    r = sv.verify_series(out, ts)
    assert r['ok'], sv.summarize(r)
    ds = xr.open_zarr(out)
    assert ds.sizes == {'time': 1, 'j': 720, 'i': 720, 'i_g': 720, 'j_g': 720}
    assert int(ds.niter[0]) == osn_date_to_iteration(ts[0]) == 1023264
    print(f'\nnetwork smoke: {wall:.0f} s for one hour (both stores)')
