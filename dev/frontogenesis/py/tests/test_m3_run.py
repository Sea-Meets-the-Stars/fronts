""" Tests for ``m3_run.py`` (M3 task 4) and ``series_verify.verify_derived_series``.

Offline: the derived store under test is written by ``budget.write_derived``
from a synthetic hour pair (``test_budget.pair`` + ``chunk_for``, so every
term is present and the schema is the real one), and the defect cases are
that store rewritten with one thing wrong.  Nothing here touches
``data/``.

Guards: the verifier accepts a store the sweep actually wrote and rejects
each class of defect one at a time (a missing variable, ``float64`` on disk,
a chunk per pair lost, a wrong ``time_mid``, ``front`` outside ``valid``, a
finite value on land, the wrong ``L_cells``, a gap in ``time``, and
``n_lost`` over M2 task 3's ``L = 8`` envelope); the ``n_lost`` reference is
the §3.5 analysis mask and not the store's own maximum; and the runner's
pair arithmetic, store/closure naming, resumable closure cache and
``--no-chunk`` stripping.
"""

import numpy as np
import pytest
import xarray as xr

import budget as bg
import m3_run as mr
import series_verify as sv
from test_budget import pair, chunk_for, budget_of

N = 64
TS = ['2012-07-02 00:00:00', '2012-07-02 01:00:00']


# ---------------------------------------------------------------------------
# a derived store written the way the sweep writes it
# ---------------------------------------------------------------------------
def _realistic(c):
    """Give the synthetic case the two things the §3.4 store needs and
    ``synthetic_cgrid`` does not provide: ``XC``/``YC`` on the grid, and a
    §3.5-shaped masks Dataset (``mask_analysis`` + ``coast_distance_km``),
    so the fixture exercises the same ``_pack`` path as the real sweep."""
    g = c['g'].copy()
    g = g.assign_coords(XC=(('face', 'j', 'i'), c['xc'] / 1e5 - 125.0),
                        YC=(('face', 'j', 'i'), c['yc'] / 1e5 + 33.0))
    dist = np.full(c['mask'].shape, 50.0)
    masks = xr.Dataset({'mask_analysis': (('j', 'i'), c['mask']),
                        'coast_distance_km': (('j', 'i'), dist)})
    return dict(c, g=g, masks=masks)


@pytest.fixture(scope='module')
def synth():
    c, ch = chunk_for(pair(ell_cells=8.0, n=N), w=-1e-5, dT=0.02)
    return _realistic(c), ch


@pytest.fixture(scope='module')
def two_pairs(synth, tmp_path_factory):
    """Two pairs of one synthetic budget, an hour apart, in one store --
    written through ``write_derived``, i.e. exactly what ``m3_run`` leaves
    on disk."""
    c, ch = synth
    out = tmp_path_factory.mktemp('derived') / 'tile330_derived_L0.zarr'
    bd = bg.compute_budget(c['ds'], c['g'], c['grid'], c['masks'], 0, chunk_ds=ch)
    bg.write_derived(bd, 0, out=out)
    bd1 = bd.assign_coords(time=bd['time'] + np.timedelta64(1, 'h'),
                           time_mid=('time', bd['time_mid'].values + np.timedelta64(1, 'h')))
    bg.write_derived(bd1, 0, out=out)
    return out, c


def _rewrite(ds, path, L=0):
    """Write a mutated Dataset back out with the sweep's own encoding, so
    only the defect under test differs from a good store."""
    ds.to_zarr(path, mode='w', encoding=bg._encoding(ds))
    return path


def _open(path):
    ds = xr.open_zarr(path).load()
    ds.close()
    return ds


def verify(path, c, L=0, ts=TS):
    return sv.verify_derived_series(path, ts, L, grid_ds=c['g'].squeeze('face'),
                                    masks=c['masks'])


# ---------------------------------------------------------------------------
# the good store
# ---------------------------------------------------------------------------
def test_verify_accepts_a_store_the_sweep_wrote(two_pairs):
    out, c = two_pairs
    r = verify(out, c)
    assert r['ok'], sv.summarize(r)
    assert r['time']['n_store'] == 2 and not r['time']['missing']
    assert r['schema']['missing'] == [] and r['schema']['extra'] == []
    assert r['schema']['L_cells'] == 0
    assert r['hours']['n_pairs'] == 2 and r['hours']['n_lost_over_bound'] == []
    assert r['hours']['n_analysis'] == int(c['mask'].sum())


def test_float32_on_disk_and_one_chunk_per_pair(two_pairs):
    out, _ = two_pairs
    ds = xr.open_zarr(out)
    for v in bg.DERIVED_VARS:
        want = 'bool' if v in sv.DERIVED_BOOL else 'float32'
        assert str(ds[v].dtype) == want, v
        assert ds[v].encoding['chunks'][0] == 1, v
    assert 'coast_distance_km' in ds.data_vars and 'time' not in ds['coast_distance_km'].dims
    ds.close()


# ---------------------------------------------------------------------------
# one defect at a time
# ---------------------------------------------------------------------------
def test_a_missing_variable_is_caught(two_pairs, tmp_path):
    out, c = two_pairs
    ds = _open(out).drop_vars('lap2_b')
    r = verify(_rewrite(ds, tmp_path / 'a.zarr'), c)
    assert not r['ok'] and r['schema']['missing'] == ['lap2_b']


def test_float64_on_disk_is_caught(two_pairs, tmp_path):
    out, c = two_pairs
    ds = _open(out)
    ds['two_F'] = ds['two_F'].astype('float64')
    r = verify(_rewrite(ds, tmp_path / 'b.zarr'), c)
    assert not r['ok'] and r['schema']['bad_dtype'] == {'two_F': 'float64'}


def test_a_lost_chunk_per_pair_is_caught(two_pairs, tmp_path):
    out, c = two_pairs
    ds = _open(out)
    enc = bg._encoding(ds)
    enc['two_F']['chunks'] = (2, N, N)          # both pairs in one chunk
    p = tmp_path / 'c.zarr'
    ds.to_zarr(p, mode='w', encoding=enc)
    r = verify(p, c)
    assert not r['ok'] and 'two_F' in r['schema']['bad_chunks']


def test_a_wrong_time_mid_is_caught(two_pairs, tmp_path):
    out, c = two_pairs
    ds = _open(out)
    ds = ds.assign_coords(time_mid=('time', ds['time_mid'].values + np.timedelta64(7, 'm')))
    r = verify(_rewrite(ds, tmp_path / 'd.zarr'), c)
    assert not r['ok'] and r['hours']['time_mid_bad'][0][1] == 37.0


def test_front_outside_valid_is_caught(two_pairs, tmp_path):
    out, c = two_pairs
    ds = _open(out)
    fr = ds['front'].values.copy()
    fr[0, 0, 0] = True                           # a front pixel the analysis mask excludes
    ds['front'] = (ds['front'].dims, fr, ds['front'].attrs)
    r = verify(_rewrite(ds, tmp_path / 'e.zarr'), c)
    assert not r['ok'] and r['hours']['front_outside_valid'] == [0]


def test_a_finite_value_on_land_is_caught(two_pairs, tmp_path):
    """The derived fields are built from NaN land, so NaN must *contain*
    land; a finite value there means something filled it."""
    out, c = two_pairs
    g = c['g'].squeeze('face').copy()
    hf = g['hFacC'].values.copy()
    hf[:4, :4] = 0.0                             # declare a corner to be land
    g['hFacC'] = (g['hFacC'].dims, hf)
    r = sv.verify_derived_series(out, TS, 0, grid_ds=g, masks=c['masks'])
    bad = r['hours']['land_finite']
    assert not r['ok'] and bad                      # some field is finite on the new land
    assert 'delta' in bad and bad['delta'] == [(0, 16), (1, 16)]


def test_the_wrong_L_is_caught(two_pairs):
    out, c = two_pairs
    r = verify(out, c, L=8)
    assert not r['ok'] and r['schema']['L_cells'] == 0 and r['schema']['L_expected'] == 8


def test_a_gap_in_time_is_caught(two_pairs):
    out, c = two_pairs
    r = verify(out, c, ts=TS + ['2012-07-02 02:00:00'])
    assert not r['ok'] and r['time']['missing'] == ['2012-07-02T02:00:00']


def test_n_lost_is_measured_against_the_analysis_mask_not_the_store(two_pairs):
    """With a mask two cells larger than anything the store can fill, both
    pairs must report a loss -- a within-store maximum would report zero."""
    out, c = two_pairs
    big = c['mask'].copy()
    big[0, 0] = big[0, 1] = True
    r = sv.verify_derived_series(out, TS, 0, grid_ds=c['g'].squeeze('face'),
                                 masks=xr.Dataset({'mask_analysis': (('j', 'i'), big)}))
    assert r['hours']['n_lost_max'] == 2 and r['hours']['n_lost_median'] == 2.0
    assert not r['ok'] and r['hours']['n_lost_bound'] == 0        # L = 0 tolerates none
    r8 = sv.verify_derived_series(out, TS, 8, grid_ds=c['g'].squeeze('face'),
                                  masks=xr.Dataset({'mask_analysis': (('j', 'i'), big)}))
    assert r8['hours']['n_lost_bound'] == sv.N_LOST_MAX_L8        # L >= 8 tolerates 34
    assert r8['hours']['n_lost_over_bound'] == []


def test_a_missing_store_is_reported_not_raised(tmp_path):
    r = sv.verify_derived_series(tmp_path / 'nope.zarr', TS, 0)
    assert r['ok'] is False and 'does not exist' in r['error']


# ---------------------------------------------------------------------------
# the runner's own arithmetic
# ---------------------------------------------------------------------------
def test_pair_range_parsing():
    assert mr._parse_pairs('0:3') == [0, 1, 2]
    assert mr._parse_pairs(':3') == [0, 1, 2]
    assert mr._parse_pairs('68:') == [68, 69, 70]
    assert mr._parse_pairs('0:999') == list(range(mr.N_PAIRS))    # clipped to 0..70
    with pytest.raises(Exception):
        mr._parse_pairs('5:5')


def test_store_and_closure_paths_keep_no_chunk_separate():
    """``--no-chunk`` must never be able to overwrite the real product."""
    assert str(mr.store_path(4)).endswith('tile330_derived_L4.zarr')
    assert str(mr.store_path(4, True)).endswith('tile330_derived_noChunk_L4.zarr')
    assert str(mr.closure_path(4)).endswith('m3_closure_L4.json')
    assert str(mr.closure_path(4, True)).endswith('m3_closure_noChunk_L4.json')
    assert mr.store_path(4) != mr.store_path(4, True)


def test_closure_cache_roundtrip_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(mr, 'closure_path', lambda L, nc=False: tmp_path / f'c_L{L}.json')
    assert mr.load_closure(0) == {}
    mr.save_closure(0, {'0': {'closed': True}})
    assert mr.load_closure(0) == {'0': {'closed': True}}
    assert not (tmp_path / 'c_L0.json.tmp').exists()              # the temp file is renamed away
    (tmp_path / 'c_L0.json').write_text('{ truncated')            # killed mid-write
    assert mr.load_closure(0) == {}                               # recomputed, not crashed


def test_no_chunk_strips_the_chunk_variables(synth):
    """The terms must be genuinely *absent*, not quietly zero."""
    c, ch = synth
    ds, _ = bg._merged(c['ds'], ch)
    assert all(v in ds for v in bg.CHUNK_VARS)
    stripped = mr._strip_chunk(ds)
    assert not any(v in stripped for v in bg.CHUNK_VARS)
    bd = bg.compute_budget(stripped, c['g'], c['grid'], c['masks'], 0)
    assert 'vertical' not in bd and 'surface_flux' not in bd
    assert bg.closure_report(bd)['closed'] is None


def test_present_pairs_reads_the_store(two_pairs, monkeypatch):
    out, _ = two_pairs
    monkeypatch.setattr(mr, 'store_path', lambda L, nc=False: out)
    assert mr.present_pairs(0, TS) == {0, 1}
    assert mr.present_pairs(0, TS + ['2012-07-02 02:00:00']) == {0, 1}
    monkeypatch.setattr(mr, 'store_path', lambda L, nc=False: out.parent / 'absent.zarr')
    assert mr.present_pairs(0, TS) == set()


def test_L_contract_is_the_M3_Q3_set():
    assert mr.L_CONTRACT == (0, 2, 4, 8) and mr.N_PAIRS == 71


def test_lowpass_refuses_an_odd_L():
    """M3-Q3's optional ``L = 1`` extra column is not available: the
    top-hat's half-width ``L/2`` must be an integer, so ``operators.lowpass``
    refuses an odd ``L``.  Recorded as a test so the option is not proposed
    again without changing the filter."""
    import operators as op
    with pytest.raises(ValueError, match='must be even'):
        op.lowpass(xr.DataArray(np.zeros((1, 8, 8)), dims=('face', 'j', 'i')), 1)
