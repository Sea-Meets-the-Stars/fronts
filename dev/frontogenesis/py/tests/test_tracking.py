""" ``tracking.find_fronts_series`` and its pieces (coding §4.10; prompt 5 task 1).

Offline on the 160 x 200 synthetic ridge of ``test_nan_finding.py`` (config D really runs:
'pool' threshold, sharpen, despur), plus one ``needs_grid`` test on pair 0 of the L = 0
derived store with the numbers of the task-1 log entry asserted as ranges.
"""
import numpy as np
import pytest
import xarray as xr
import yaml

import tracking as tr
from test_nan_finding import ridge_field, land_block, with_nan, config_d, NJ, NI, COAST_I

pytestmark = [pytest.mark.filterwarnings('ignore:Parameter `min_size` is deprecated'),
              pytest.mark.filterwarnings('ignore:Setting the shape on a NumPy array')]
NW = 2


# ---------------------------------------------------------------------------
# config and timestamps
# ---------------------------------------------------------------------------
def test_config_is_read_from_the_yaml():
    cfg = tr.load_finding_config('D')
    assert cfg == config_d()
    with open(tr.config_path('D')) as f:
        raw = yaml.safe_load(f)
    assert raw['label'] == 'D' and cfg == raw['binary']
    assert cfg['window'] == 64 and cfg['threshold'] == 85 and cfg['thresh_mode'] == 'pool'


def test_time_formats_round_trip():
    t = np.datetime64('2012-07-02T00:30:00')
    assert tr.to_track_time(t) == '2012-07-02T00_30_00'            # underscores: parse_time
    assert tr.to_dbof_time(t) == '2012-07-02 00:30:00'              # colons, space: dbof
    assert tr.from_track_time('2012-07-02T00_30_00') == t
    assert tr.to_track_time('2012-07-02 00:30:00') == '2012-07-02T00_30_00'
    assert tr.to_track_time('2012-07-02T00:30:00.000000000') == '2012-07-02T00_30_00'
    assert tr.to_dbof_time('2012-07-02T00_30_00') == '2012-07-02 00:30:00'
    with pytest.raises(ValueError):
        tr.to_track_time('02/07/2012 00:30')


# ---------------------------------------------------------------------------
# one field
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def ridge():
    G = with_nan(ridge_field('j'), land_block())
    labels, fronts = tr.find_fronts(G, config='D', n_workers=NW)
    return dict(G=G, labels=labels, fronts=fronts)


def test_labels_are_int32_contiguous_and_off_nan(ridge):
    lab, G = ridge['labels'], ridge['G']
    assert lab.dtype == np.int32 and lab.shape == (NJ, NI)
    n = int(lab.max())
    assert n >= 1
    assert set(np.unique(lab)) == set(range(n + 1))                  # 0 and 1..n, no gaps
    assert ((lab > 0) & ~np.isfinite(G)).sum() == 0                  # nothing on NaN
    assert np.array_equal(lab > 0, ridge['fronts'])
    # the ridge is found: most of its pixels are at i = 100 +- 2
    jj, ii = np.where(lab > 0)
    assert np.mean(np.abs(ii - 100) <= 2) > 0.9


def test_each_label_is_one_8_connected_component(ridge):
    from skimage import measure
    lab = ridge['labels']
    for k in range(1, int(lab.max()) + 1):
        assert measure.label(lab == k, connectivity=2).max() == 1, k


def test_mask_is_applied_as_nan_not_zero(ridge):
    """A masked block far from the ridge changes nothing on the ridge; the same block as
    zeros would move the local percentile (task 7a).  Here: the labels inside the untouched
    region are pixel-for-pixel equal."""
    G = ridge_field('j')
    mask = np.ones((NJ, NI), bool)
    mask[:, :COAST_I] = False                                       # the land block, as a mask
    lab_m, fr_m = tr.find_fronts(G, mask, config='D', n_workers=NW)
    assert (fr_m & ~mask).sum() == 0
    assert np.array_equal(fr_m, ridge['fronts'])                    # == NaN land, bit for bit
    lab_z, fr_z = tr.find_fronts(np.where(mask, G, 0.0), None, config='D', n_workers=NW)
    assert not np.array_equal(fr_z, fr_m)                           # zeros are not the same


def test_enclosed_nan_island_is_not_filled():
    """task 7a's open ``fronts`` defect: ``remove_small_holes`` fills a NaN island the front
    wraps around.  The caller-side rule removes exactly that."""
    G = ridge_field('j')
    G[80:86, 98:103] = np.nan                                       # a 6 x 5 island on the ridge
    lab, fr = tr.find_fronts(G, config='D', n_workers=NW)
    assert (fr & ~np.isfinite(G)).sum() == 0
    assert fr.sum() > 50


def test_empty_field_does_not_raise():
    """task 7a's other open defect: ``skan`` on an empty skeleton.  Despur is skipped."""
    G = np.full((NJ, NI), np.nan)
    lab, fr = tr.find_fronts(G, config='D', n_workers=NW)
    assert lab.max() == 0 and not fr.any() and lab.dtype == np.int32
    G = np.full((NJ, NI), 1.0)                                      # flat: nothing survives
    lab, fr = tr.find_fronts(G, config='D', n_workers=NW)
    assert lab.max() == 0


def test_bad_inputs():
    with pytest.raises(ValueError):
        tr.find_fronts(np.zeros((3, 4, 5)), config='D', n_workers=NW)
    with pytest.raises(ValueError):
        tr.find_fronts(np.zeros((NJ, NI)), np.ones((NJ, NI - 1), bool), config='D', n_workers=NW)


# ---------------------------------------------------------------------------
# the series
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def tiny_store():
    """A derived-store look-alike: ``G(time, j, i)`` for three pairs with the ridge moving
    one cell per hour, ``time`` and ``time_mid`` as §3.4."""
    t = np.array(['2012-07-02T00:00', '2012-07-02T01:00', '2012-07-02T02:00'],
                 dtype='datetime64[ns]')
    G = np.stack([with_nan(ridge_field('j', centre=100 + 3 * n, seed=n), land_block())
                  for n in range(3)])
    return xr.Dataset({'G': (('time', 'j', 'i'), G)},
                      coords={'time': t, 'time_mid': ('time', t + np.timedelta64(30, 'm')),
                              'j': np.arange(NJ), 'i': np.arange(NI)})


def test_find_fronts_series_keys_and_arrays(tiny_store):
    mask = ~land_block()
    lines = []
    out = tr.find_fronts_series(tiny_store, mask, config='D', n_workers=NW, log=lines.append)
    assert list(out) == ['2012-07-02T00_30_00', '2012-07-02T01_30_00', '2012-07-02T02_30_00']
    for k, lab in out.items():
        assert lab.dtype == np.int32 and lab.shape == (NJ, NI) and lab.max() >= 1
        assert (lab[:, :COAST_I] > 0).sum() == 0
    assert len(lines) == 3 and 'fronts' in lines[0]
    # the ridge moves: the labelled columns follow it
    cols = [np.median(np.where(lab > 0)[1]) for lab in out.values()]
    assert cols[1] - cols[0] == pytest.approx(3, abs=1) and cols[2] - cols[1] == pytest.approx(3, abs=1)
    # a subset of pairs, and the labels_at factory
    sub = tr.find_fronts_series(tiny_store, mask, config='D', n_workers=NW, pairs=[2])
    assert list(sub) == ['2012-07-02T02_30_00'] and np.array_equal(sub['2012-07-02T02_30_00'],
                                                                   out['2012-07-02T02_30_00'])
    labels_at, times = tr.labels_at_factory(out)
    assert times == list(out) and labels_at(1) is out[times[1]]


def test_series_stats_and_store_round_trip(tiny_store, tmp_path):
    mask = ~land_block()
    out = tr.find_fronts_series(tiny_store, mask, config='D', n_workers=NW)
    p90 = [tiny_store['G'].isel(time=n).values >= np.nanpercentile(tiny_store['G'].isel(time=n).values, 90)
           for n in range(3)]
    rows = tr.series_stats(out, p90)
    assert len(rows) == 3 and rows[0]['n_fronts'] == int(out[rows[0]['time']].max())
    assert rows[0]['n_pixels'] == int((out[rows[0]['time']] > 0).sum())
    assert 0.0 <= rows[0]['frac_on_p90'] <= 1.0
    path = tmp_path / 'fronts.zarr'
    tr.write_fronts_series(out, path, like=tiny_store, attrs={'config': 'D'})
    back = tr.open_fronts_series(path)
    assert list(back) == list(out)
    for k in out:
        assert np.array_equal(back[k], out[k]) and back[k].dtype == np.int32
    ds = xr.open_zarr(str(path))
    assert ds['labels'].dtype == np.int32 and ds.attrs['config'] == 'D'
    assert np.array_equal(ds['time'].values, tiny_store['time'].values)
    assert np.array_equal(ds['time_mid'].values, tiny_store['time_mid'].values)
    with pytest.raises(FileExistsError):
        tr.write_fronts_series(out, path)
    tr.write_fronts_series(out, path, clobber=True)


# ---------------------------------------------------------------------------
# the real tile
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_pair_0_of_the_L0_store(masks_ds):
    import budget
    src = budget.derived_path(0)
    if not src.exists():
        pytest.skip(f'{src} not on disk (M3 task 4)')
    ds = xr.open_zarr(str(src))
    out = tr.find_fronts_series(ds, masks_ds, config='D', n_workers=4, pairs=[0])
    key, lab = next(iter(out.items()))
    assert key == '2012-07-02T00_30_00'
    G = ds['G'].isel(time=0).values
    ana = masks_ds['mask_analysis'].values
    fr = lab > 0
    assert (fr & ~np.isfinite(G)).sum() == 0 and (fr & ~ana).sum() == 0
    n_px, n_fr = int(fr.sum()), int(lab.max())
    assert 8000 <= n_px <= 10500, n_px                   # task-1 log: 9,115 at 8 workers
    assert 150 <= n_fr <= 500, n_fr
    frac = (fr & ds['front'].isel(time=0).values).sum() / n_px
    assert 0.55 <= frac <= 0.75, frac                    # 0.64 on the p90 front pixels
