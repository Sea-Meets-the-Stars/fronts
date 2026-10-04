""" Tests for ``vertical.load_chunk_levels`` and
``series_verify.verify_chunk_series`` (M2 task 5), offline.

The source is mimicked by synthetic zarr-v3 stores written under
``tmp_path`` with the real layout (M2 task 4): one store per hour named
``YYYYMMDDTHH.zarr``, every variable a single bytes+zstd object, 51 levels
on ``k``, ``W`` on ``k_p1`` (52), a ``numpy.datetime64`` ``time``, the
group attrs (``selected_iteration`` = MIT, ``selected_date_utc``, tile), the
flux attrs saying "+=down" with upward-positive data, and a ``grid.zarr``.
They are read through a local fsspec filesystem, so the real
fetch/decode/validate path runs; only ``vertical._cat`` is monkeypatched,
to inject corrupt reads and crashes.  One ``network`` test loads a real
hour and is deselected by default (``pytest.ini``).
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path

import fsspec
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.codecs import ZstdCodec

import vertical as vt
import series_verify as sv
import zarr_series as zs
from dbof.llc4320_ingestion.date_iterations import osn_date_to_iteration

N, NLEV = 12, 51
TS = [f'2012-07-02 {h:02d}:00:00' for h in range(6)]
DRF = np.array([1.0, 1.14, 1.30] + [2.0] * (NLEV - 3), dtype='f4')
Z = (-(np.cumsum(DRF) - DRF / 2)).astype('f4')
ZL = np.concatenate([[0.0], -np.cumsum(DRF)[:-1]]).astype('f4')
SRC_FLUX_ATTRS = {
    'oceQnet': ('net surface heat flux into the ocean (+=down), >0 increases theta', 'W/m^2'),
    'oceQsw': ('net Short-Wave radiation (+=down), >0 increases theta', 'W/m^2'),
    'oceFWflx': ('net surface Fresh-Water flux into the ocean (+=down), >0 decreases salinity',
                 'kg/m^2/s')}


# ---------------------------------------------------------------------------
# the synthetic source
# ---------------------------------------------------------------------------
def hfac():
    """hFacC (51, N, N): a land block that grows with depth, so each of
    k = 0..2 has its own NaN pattern; deep levels mostly land."""
    h = np.ones((NLEV, N, N), 'f4')
    for k in range(NLEV):
        h[k, :3 + min(k, 6), :4] = 0
    return h


def truth(ts, var):
    """The raw source field (all levels), a deterministic function of (hour, var)."""
    t = int(np.datetime64(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), 's').astype('int64'))
    rng = np.random.default_rng([t, ['Theta', 'Salt', 'W', 'oceQnet', 'oceQsw', 'oceFWflx',
                                     'Eta'].index(var)])
    h = hfac()
    if var in ('Theta', 'Salt', 'W'):
        n = NLEV + 1 if var == 'W' else NLEV
        lo, hi = {'Theta': (8, 20), 'Salt': (32, 34), 'W': (-1e-3, 1e-3)}[var]
        a = (lo + (hi - lo) * rng.random((n, N, N))).astype('f4')
        land = np.concatenate([h == 0, (h[-1:] == 0)]) if var == 'W' else h == 0
        a[land] = np.nan
        return a
    lo, hi = {'oceQnet': (-450, 120), 'oceQsw': (-600, -1), 'oceFWflx': (1e-5, 4e-5),
              'Eta': (-1, 1)}[var]                   # upward-positive, as the real store
    a = (lo + (hi - lo) * rng.random((N, N))).astype('f4')
    a[h[0] == 0] = np.nan
    return a


def _arr(g, name, data, dims, attrs=None, fill=np.nan):
    a = g.create_array(name, shape=data.shape, dtype=data.dtype, chunks=data.shape,
                       compressors=ZstdCodec(level=0), fill_value=fill,
                       dimension_names=dims, attributes=attrs or {})
    a[...] = data


def write_grid(prefix: Path):
    g = zarr.open_group(prefix / 'grid.zarr', mode='w', zarr_format=3)
    rng = np.random.default_rng(1)
    _arr(g, 'drF', DRF, ['k']); _arr(g, 'Z', Z, ['k']); _arr(g, 'Zl', ZL, ['k_l'])
    _arr(g, 'hFacC', hfac()[:, None], ['k', 'face', 'j', 'i'])
    _arr(g, 'XC', rng.random((1, N, N), dtype='f4'), ['face', 'j', 'i'])
    _arr(g, 'YC', rng.random((1, N, N), dtype='f4'), ['face', 'j', 'i'])
    _arr(g, 'j', np.arange(N, dtype='i2'), ['j'], fill=0)
    _arr(g, 'i', (2880 + np.arange(N)).astype('i2'), ['i'], fill=0)


def write_hour(prefix: Path, ts, overrides=None):
    """One hourly store; ``overrides`` replaces any variable's raw data,
    ``'time'`` the stored time, ``'attrs'`` group attrs."""
    ov = overrides or {}
    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    attrs = dict(chunk_name='monterey_bay', resolved_face=10, j_start=0, i_start=2880,
                 tile_size=N, selected_iteration=osn_date_to_iteration(ts) - 10368,
                 selected_date_utc=ts)
    attrs.update(ov.get('attrs', {}))
    g = zarr.open_group(prefix / (dt.strftime('%Y%m%dT%H') + '.zarr'), mode='w', zarr_format=3,
                        attributes=attrs)
    t = np.array([ov.get('time', np.datetime64(dt, 'ns'))], dtype='datetime64[ns]')
    _arr(g, 'time', t, ['time'], fill=None)
    for v in ('Theta', 'Salt'):
        _arr(g, v, ov.get(v, truth(ts, v))[:, None], ['k', 'face', 'j', 'i'],
             {'units': 'x', 'long_name': v})
    _arr(g, 'W', ov.get('W', truth(ts, 'W'))[:, None], ['k_p1', 'face', 'j', 'i'],
         {'units': 'm s-1', 'long_name': 'Vertical Component of Velocity'})
    for v, (ln, u) in SRC_FLUX_ATTRS.items():
        _arr(g, v, ov.get(v, truth(ts, v))[None], ['face', 'j', 'i'],
             {'long_name': ln, 'standard_name': v, 'units': u})
    _arr(g, 'Eta', ov.get('Eta', truth(ts, 'Eta'))[None], ['face', 'j', 'i'], {'units': 'm'})


@pytest.fixture
def src(tmp_path):
    prefix = tmp_path / 'CHUNKS' / 'monterey_bay'
    write_grid(prefix)
    for ts in TS:
        write_hour(prefix, ts)
    return dict(prefix=str(prefix), fs=fsspec.filesystem('file'))


@pytest.fixture
def cat_log(monkeypatch):
    """Count the reads (the network primitive)."""
    calls = []
    orig = vt._cat
    def logged(fs, path):
        calls.append(path)
        return orig(fs, path)
    monkeypatch.setattr(vt, '_cat', logged)
    return calls


def levels():
    return dict(land=hfac()[:3] == 0, drF=DRF[:3])


def pull(window, out, src, **kw):
    rep = {}
    kw.setdefault('sleep', lambda s: None)
    vt.load_chunk_levels(window, 2, out, fs=src['fs'], prefix=src['prefix'], report=rep, **kw)
    return rep


def snapshot(path):
    out = {}
    for r, _, fs in os.walk(path):
        for f in fs:
            p = Path(r) / f
            out[str(p.relative_to(path))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# the product: levels, rename, sign, drF, schema
# ---------------------------------------------------------------------------
def test_levels_rename_and_schema(tmp_path, src):
    out = tmp_path / 'c.zarr'
    rep = pull(TS, out, src)
    assert rep['pulled'] == TS and not rep['failed'] and rep['repaired'] == 0
    ds = xr.open_zarr(out)
    assert dict(ds.sizes) == {'time': 6, 'k': 3, 'k_l': 3, 'j': N, 'i': N}
    assert ds.Theta.dims == ds.Salt.dims == ('time', 'k', 'j', 'i')
    assert ds.W.dims == ('time', 'k_l', 'j', 'i') and 'k_p1' not in ds.dims
    for k, ts in enumerate(TS):
        for v in ('Theta', 'Salt'):
            np.testing.assert_array_equal(ds[v].isel(time=k).values, truth(ts, v)[:3])
        # W(k_l = n) is the source W(k_p1 = n), n = 0..2
        np.testing.assert_array_equal(ds.W.isel(time=k).values, truth(ts, 'W')[:3])
    for v in ('Theta', 'Salt', 'W', 'oceQnet', 'oceQsw', 'oceFWflx', 'drF'):
        assert ds[v].dtype == np.float32
    assert tuple(ds.Theta.encoding['chunks']) == (1, 3, N, N)
    assert tuple(ds.oceQsw.encoding['chunks']) == (1, N, N)
    assert sorted(os.listdir(out / 'W' / 'c')) == [str(k) for k in range(6)]   # one chunk/hour
    assert ds['time'].encoding['units'] == 'seconds since 2011-09-10'
    assert ds.niter.values.tolist() == [osn_date_to_iteration(t) for t in TS]
    assert (ds.mit_iteration.values == ds.niter.values - 10368).all()
    assert ds.face.ndim == 0 and int(ds.face) == 10
    assert ds.i.values[0] == 2880 and ds.i.attrs['axis'] == 'X'
    np.testing.assert_array_equal(ds.Z.values, Z[:3])
    np.testing.assert_array_equal(ds.Zl.values, ZL[:3])
    assert ds.attrs['levels'] == 'k=0..2, k_l=0..2' and ds.attrs['source'] == 'CHUNKS/monterey_bay'
    assert ds.attrs['iterations'] == ds.niter.values.tolist()
    assert ds.W.attrs['source_dim'] == 'k_p1'
    assert sv.verify_chunk_series(out, TS, levels=levels())['ok']


def test_drF(tmp_path, src):
    out = tmp_path / 'c.zarr'
    pull(TS[:2], out, src)
    ds = xr.open_zarr(out)
    assert ds.drF.dims == ('k',)
    np.testing.assert_array_equal(ds.drF.values, np.array([1.0, 1.14, 1.30], 'f4'))
    assert float(ds.drF[0]) == 1.0 and float(ds.Z[0]) == -0.5
    # static: written once, not appended per hour
    assert len(os.listdir(out / 'drF' / 'c')) == 1


def test_sign_conversion_and_attrs(tmp_path, src):
    out = tmp_path / 'c.zarr'
    pull(TS[:3], out, src)
    ds = xr.open_zarr(out)
    for k, ts in enumerate(TS[:3]):
        for v in ('oceQnet', 'oceQsw', 'oceFWflx'):
            np.testing.assert_array_equal(ds[v].isel(time=k).values, -truth(ts, v))
    assert np.nanmin(ds.oceQsw.values) > 0                  # downward-positive after the flip
    for v in ('oceQnet', 'oceQsw', 'oceFWflx'):
        a = ds[v].attrs
        assert a['sign_convention'].startswith('positive downward')
        assert a['source_sign_convention'] == SRC_FLUX_ATTRS[v][0]
        assert 'negated' in a['sign_conversion'] and '+=down' not in a['long_name']
        assert a['units'] == SRC_FLUX_ATTRS[v][1]
    assert ds.attrs['flux_sign_convention'] == 'positive downward'
    assert any('NEGATED' in p for p in ds.attrs['provenance'])


def test_in_memory(src):
    ds = vt.load_chunk_levels(TS[:2], 2, None, fs=src['fs'], prefix=src['prefix'])
    assert isinstance(ds, xr.Dataset) and ds.sizes['time'] == 2 and ds.drF.dims == ('k',)
    np.testing.assert_array_equal(ds.oceFWflx.isel(time=1).values, -truth(TS[1], 'oceFWflx'))


def test_k_max(tmp_path, src):
    ds = vt.load_chunk_levels(TS[:1], 1, None, fs=src['fs'], prefix=src['prefix'])
    assert ds.sizes['k'] == ds.sizes['k_l'] == 2 and ds.attrs['levels'] == 'k=0..1, k_l=0..1'


# ---------------------------------------------------------------------------
# resumability
# ---------------------------------------------------------------------------
def test_noop_rerun_byte_identical(tmp_path, src, cat_log):
    out = tmp_path / 'c.zarr'
    pull(TS, out, src)
    before = snapshot(out)
    n = len(cat_log)
    rep = pull(TS, out, src)
    assert rep['pulled'] == [] and rep['skipped'] == TS and rep['repaired'] == 0
    assert len(cat_log) == n, 'a no-op re-run must not read the source'
    assert snapshot(out) == before


def test_resume_between_hours(tmp_path, src, monkeypatch):
    class Crash(BaseException):
        pass
    orig = vt._cat
    def crash(fs, path):
        if '20120702T03.zarr/Salt/c' in path:
            raise Crash('killed mid-fetch')
        return orig(fs, path)
    monkeypatch.setattr(vt, '_cat', crash)
    out = tmp_path / 'c.zarr'
    with pytest.raises(Crash):
        pull(TS, out, src)
    assert len(zs.present_times(out)) == 3                # nothing of hour 3 on disk
    before = snapshot(out)
    monkeypatch.setattr(vt, '_cat', orig)
    rep = pull(TS, out, src)
    assert rep['skipped'] == TS[:3] and rep['pulled'] == TS[3:] and rep['repaired'] == 0
    after = snapshot(out)
    assert all(after[p] == h for p, h in before.items() if '/c/' in p)
    assert sv.verify_chunk_series(out, TS, levels=levels())['ok']


def test_resume_after_crash_inside_append(tmp_path, src, monkeypatch):
    """Killed inside ``to_zarr(append_dim='time')``: the store is left
    half-written; the resume must repair and re-pull that hour."""
    out = tmp_path / 'c.zarr'
    pull(TS[:3], out, src)
    good = snapshot(out)
    class Crash(BaseException):
        pass
    orig = zarr.Array.__setitem__
    writes = []
    def boom(self, key, value):
        writes.append(self.path)
        if len(writes) == 3:
            raise Crash('killed mid-append')
        return orig(self, key, value)
    monkeypatch.setattr(zarr.Array, '__setitem__', boom)
    with pytest.raises(Crash):
        pull(TS[:4], out, src)
    monkeypatch.setattr(zarr.Array, '__setitem__', orig)
    g = zarr.open_group(out, mode='r', use_consolidated=False)
    lengths = {n: a.shape[0] for n, a in g.arrays()
               if (a.metadata.dimension_names or [''])[0] == 'time'}
    assert 4 in lengths.values() and 3 in lengths.values(), lengths
    rep = pull(TS, out, src)
    assert rep['repaired'] == 1 and rep['skipped'] == TS[:3] and rep['pulled'] == TS[3:]
    ds = xr.open_zarr(out)
    for k, ts in enumerate(TS):
        np.testing.assert_array_equal(ds.W.isel(time=k).values, truth(ts, 'W')[:3])
        np.testing.assert_array_equal(ds.oceQsw.isel(time=k).values, -truth(ts, 'oceQsw'))
    assert ds.attrs['iterations'] == [osn_date_to_iteration(t) for t in TS]
    after = snapshot(out)
    assert all(after[p] == h for p, h in good.items() if '/c/' in p)
    assert sv.verify_chunk_series(out, TS, levels=levels())['ok']


def test_clobber_and_order_errors(tmp_path, src):
    out = tmp_path / 'c.zarr'
    with pytest.raises(ValueError, match='duplicated'):
        pull([TS[0], TS[0]], out, src)
    with pytest.raises(ValueError, match='increasing'):
        pull([TS[1], TS[0]], out, src)
    pull(TS[2:4], out, src)
    with pytest.raises(ValueError, match='earlier'):
        pull([TS[0]], out, src)
    assert pull(TS[:2], out, src, clobber=True)['pulled'] == TS[:2]
    assert len(zs.present_times(out)) == 2


# ---------------------------------------------------------------------------
# corrupt reads, retries, validation, stop-at-gap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('corruption', ['truncated', 'garbage', 'wrong_size', 'nan_pattern'])
def test_retry_on_corrupt_read(tmp_path, src, monkeypatch, corruption):
    """The first read of hour 1's Theta object is corrupt; the retry gets
    good bytes and the store is right."""
    from numcodecs import Zstd
    orig = vt._cat
    hits = []
    def flaky(fs, path):
        blob = orig(fs, path)
        if '20120702T01.zarr/Theta/c/' in path and not hits:
            hits.append(path)
            if corruption == 'truncated':
                return blob[: len(blob) // 2]
            if corruption == 'garbage':
                return bytes(len(blob))
            raw = bytearray(Zstd().decode(blob))
            if corruption == 'wrong_size':
                return Zstd().encode(bytes(raw[:-4]))
            a = np.frombuffer(bytes(raw), '<f4').copy()
            a[np.isnan(a).argmax()] = 5.0                # a finite value on land: decodes fine
            return Zstd().encode(a.tobytes())
        return blob
    monkeypatch.setattr(vt, '_cat', flaky)
    out = tmp_path / 'c.zarr'
    slept, lines = [], []
    rep = {}
    vt.load_chunk_levels(TS[:3], 2, out, fs=src['fs'], prefix=src['prefix'], report=rep,
                         backoff=(0.01, 0.02), sleep=slept.append, log=lines.append)
    assert hits and slept == [0.01]
    assert rep['pulled'] == TS[:3] and not rep['failed']
    assert any('CorruptRead' in l and 'retrying' in l for l in lines)
    ds = xr.open_zarr(out)
    np.testing.assert_array_equal(ds.Theta.isel(time=1).values, truth(TS[1], 'Theta')[:3])


def test_persistent_corruption_stops_at_gap(tmp_path, src, monkeypatch):
    orig = vt._cat
    def dead(fs, path):
        blob = orig(fs, path)
        return blob[:10] if '20120702T02.zarr/Salt/c/' in path else blob
    monkeypatch.setattr(vt, '_cat', dead)
    out = tmp_path / 'c.zarr'
    lines = []
    rep = pull(TS, out, src, attempts=3, backoff=(0,), log=lines.append)
    assert rep['pulled'] == TS[:2] and rep['failed'] == [TS[2]]
    assert rep['not_attempted'] == TS[3:]
    assert len(zs.present_times(out)) == 2
    assert any('FAILED' in l and TS[2] in l for l in lines)
    assert sum('giving up after 3' in l for l in lines) == 1
    v = sv.verify_chunk_series(out, TS, levels=levels())
    assert not v['ok'] and v['time']['missing'] == [t.replace(' ', 'T') for t in TS[2:]]
    monkeypatch.setattr(vt, '_cat', orig)
    rep = pull(TS, out, src)
    assert rep['skipped'] == TS[:2] and rep['pulled'] == TS[2:]
    assert sv.verify_chunk_series(out, TS, levels=levels())['ok']


@pytest.mark.parametrize('defect', ['time', 'iteration', 'already_downward'])
def test_hour_validation_rejects(tmp_path, src, defect):
    """Time misalignment and a sign convention that would be double-flipped
    are refused (retried, then the run stops at that hour)."""
    prefix = Path(src['prefix'])
    if defect == 'time':
        write_hour(prefix, TS[1], {'time': np.datetime64('2012-07-02T02:00', 'ns')})
    elif defect == 'iteration':
        write_hour(prefix, TS[1], {'attrs': {'selected_iteration':
                                             osn_date_to_iteration(TS[1])}})   # OSN, not MIT
    else:
        write_hour(prefix, TS[1], {'oceQsw': -truth(TS[1], 'oceQsw')})
    out = tmp_path / 'c.zarr'
    rep = pull(TS[:3], out, src, backoff=(0,))
    assert rep['pulled'] == TS[:1] and rep['failed'] == [TS[1]]


def test_eta_alignment_check(tmp_path, src):
    """With ``osn_store``, the chunk Eta must equal the OSN Eta of that hour."""
    t = np.array([np.datetime64(datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), 'ns') for s in TS])
    eta = np.stack([truth(s, 'Eta') for s in TS])
    osn = tmp_path / 'osn.zarr'
    xr.Dataset({'Eta': (('time', 'j', 'i'), eta)}, coords={'time': t}).to_zarr(osn)
    rep = pull(TS[:3], tmp_path / 'a.zarr', src, osn_store=osn)
    assert rep['pulled'] == TS[:3]
    shifted = tmp_path / 'osn_shifted.zarr'          # OSN one hour off: hour 0 has no match
    xr.Dataset({'Eta': (('time', 'j', 'i'), eta[1:])},
               coords={'time': t[:-1]}).to_zarr(shifted)
    rep = pull(TS[:3], tmp_path / 'b.zarr', src, osn_store=shifted, backoff=(0,))
    assert rep['pulled'] == [] and rep['failed'] == [TS[0]]


# ---------------------------------------------------------------------------
# verify_chunk_series
# ---------------------------------------------------------------------------
@pytest.fixture
def good(tmp_path, src):
    out = tmp_path / 'c.zarr'
    pull(TS, out, src)
    return out


def test_verify_good(good):
    r = sv.verify_chunk_series(good, TS, levels=levels())
    assert r['ok'], r
    assert r['land_nan']['hours_checked'] == 6 and r['land_nan']['levels_checked'] == 3
    assert r['niter']['steps'] == [144] and r['niter']['mit_is_osn_minus_10368']
    assert r['drF']['values'] == np.array([1.0, 1.14, 1.30], 'f4').tolist()
    assert r['sign']['oceQsw_min'] > 0 and 'OK' in sv.summarize(r)


def test_verify_gap(good):
    r = sv.verify_chunk_series(good, TS + ['2012-07-02 06:00:00'], levels=levels())
    assert not r['ok'] and r['time']['missing'] == ['2012-07-02T06:00:00']


def test_verify_sign_and_land(good):
    g = zarr.open_group(good, mode='r+', use_consolidated=False)
    g['oceQsw'][2] = -g['oceQsw'][2]                     # un-negated hour
    slab = g['Theta'][4]; slab[2, 0, 0] = 10.0; g['Theta'][4] = slab   # finite on land, k=2
    slab = g['W'][1]; slab[1, 8, 8] = np.nan; g['W'][1] = slab         # NaN on ocean, k_l=1
    r = sv.verify_chunk_series(good, TS, levels=levels())
    assert not r['ok'] and not r['sign']['ok'] and not r['land_nan']['ok']
    assert r['sign']['hours_below_minus_tol'] == ['2012-07-02T02:00:00']
    assert r['land_nan']['max_mismatch_cells'] == dict(Theta=1, Salt=0, W=1, oceQnet=0,
                                                       oceQsw=0, oceFWflx=0)
    assert r['land_nan']['first_bad_hour'] == {'Theta': '2012-07-02T04:00:00',
                                               'W': '2012-07-02T01:00:00'}
    assert r['time']['ok'] and r['schema']['ok']


def test_verify_drF_and_schema(good, tmp_path):
    r = sv.verify_chunk_series(good, TS, levels=dict(land=hfac()[:3] == 0,
                                                     drF=np.array([1.0, 1.2, 1.3], 'f4')))
    assert not r['ok'] and not r['drF']['ok'] and r['schema']['ok']
    bad = xr.open_zarr(good).load()
    bad['Salt'] = bad['Salt'].astype('float64')
    bad['oceQnet'].attrs['sign_convention'] = '+=down'
    bad = bad.drop_vars('oceFWflx')
    out = tmp_path / 'bad.zarr'
    bad.to_zarr(out, mode='w')
    r = sv.verify_chunk_series(out, TS, levels=levels())
    msgs = ' '.join(r['schema']['problems'])
    assert not r['schema']['ok'] and 'float64' in msgs and 'oceFWflx' in msgs \
        and 'oceQnet sign_convention' in msgs


def test_verify_missing_store(tmp_path):
    r = sv.verify_chunk_series(tmp_path / 'none.zarr', TS, levels=levels())
    assert not r['ok'] and 'does not exist' in r['error']


# ---------------------------------------------------------------------------
# one real hour (deselected by default; ~5 min at the 0.55 MB/s measured here)
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_network_one_real_hour(tmp_path):
    """SLOW (~5 min, ~174 MB from Nautilus): load ``2012-07-03 00:00:00``
    into a temp store with the Eta check against the OSN store, verify it."""
    import time
    ts = ['2012-07-03 00:00:00']
    out = tmp_path / 'one.zarr'
    rep = {}
    t0 = time.time()
    osn = vt.OSN_RAW_ZARR if vt.OSN_RAW_ZARR.exists() else None
    vt.load_chunk_levels(ts, 2, out, osn_store=osn, local_grid=vt.DATA_DIR / 'tile330_grid.zarr',
                         report=rep)
    assert rep['pulled'] == ts and not rep['failed']
    r = sv.verify_chunk_series(out, ts)
    assert r['time']['ok'] and r['schema']['ok'] and r['land_nan']['ok'] and r['drF']['ok']
    ds = xr.open_zarr(out)
    assert dict(ds.sizes) == {'time': 1, 'k': 3, 'k_l': 3, 'j': 720, 'i': 720}
    assert float(np.nanmin(ds.oceQsw.values)) >= -1.0
    print(f'\nnetwork smoke: {time.time() - t0:.0f} s, {rep["bytes"][ts[0]] / 1e6:.0f} MB')
