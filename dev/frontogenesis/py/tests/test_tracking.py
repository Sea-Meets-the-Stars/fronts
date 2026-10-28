""" ``tracking.find_fronts_series`` and its pieces (coding §4.10; prompt 5 task 1).

Offline on the 160 x 200 synthetic ridge of ``test_nan_finding.py`` (config D really runs:
'pool' threshold, sharpen, despur), plus one ``needs_grid`` test on pair 0 of the L = 0
derived store with the numbers of the task-1 log entry asserted as ranges.
"""
import json
import numpy as np
import pandas as pd
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


# ===========================================================================
# Task 2: flow-informed tracking
# ===========================================================================
from fronts import front_tracking as ft
from test_operators import synthetic_cgrid, da, C_DIMS, U_DIMS, V_DIMS
import semilag as sl


def _line_labels(nj, ni, cols, width=1):
    """Label array with one vertical 1-px front per entry of ``cols`` (label k+1 at column
    cols[k], rows 20..nj-20)."""
    lab = np.zeros((nj, ni), np.int32)
    for k, c in enumerate(cols):
        lab[20:nj - 20, c:c + width] = k + 1
    return lab


def test_advect_mask_by_integer_and_half_cell_shifts():
    m = _line_labels(60, 80, [40]) > 0
    adv = tr.advect_mask_by(m, 2.0, 0.0)                  # arriving parcel came from i - 2
    assert np.array_equal(adv[:, 2:], m[:, :-2]) and not adv[:, :2].any()
    adv = tr.advect_mask_by(m, 0.0, -3.0)
    assert np.array_equal(adv[:-3, :], m[3:, :])
    half = tr.advect_mask_by(m, 0.5, 0.0)                 # 0.5 on both neighbours: kept, 2 px
    assert half[:, 40].sum() == m[:, 40].sum() and half[:, 41].sum() == m[:, 40].sum()
    assert half.sum() == 2 * m.sum()
    nan = tr.advect_mask_by(m, np.full(m.shape, np.nan), 0.0)
    assert not nan.any()
    with pytest.raises(TypeError):
        tr.advect_mask(_line_labels(60, 80, [40]), None, None, None)   # the label field


def test_advect_mask_uniform_flow_is_a_whole_cell_translation():
    """``U = dxC / dt`` moves the mask by exactly one cell along ``i`` through the real
    ``departure_index`` (the semilag test's own case)."""
    g, grid, pos = synthetic_cgrid(dy=2250.0, rotated=False)
    x, _ = pos('c')
    shape = x.shape
    U = da(np.full(shape, 1800.0 / 3600.0), U_DIMS)
    V = da(np.zeros(shape), V_DIMS)
    u_c, v_c = sl.centre_velocities(U, V, g, grid)
    m = np.zeros(shape[-2:], bool)
    m[10:-10, 15] = True
    m[12, 20:30] = True
    adv = tr.advect_mask(m, u_c, v_c, g, dt=3600.0)
    inner = np.zeros_like(m)
    inner[4:-4, 4:-4] = True
    assert np.array_equal(adv[inner], np.roll(m, 1, axis=1)[inner])
    back = tr.advect_mask(adv, u_c, v_c, g, dt=-3600.0)
    assert np.array_equal(back[inner], m[inner])


def test_iou_dilated_decays_with_offset():
    a = _line_labels(60, 80, [40]) > 0
    vals = [tr.iou_dilated(a, np.roll(a, k, axis=1)) for k in range(5)]
    assert vals[0] == 1.0 and vals[1] == pytest.approx(0.5) and vals[2] == pytest.approx(0.2)
    assert vals[3] == 0.0 and vals[4] == 0.0
    assert tr.iou_dilated(a, np.roll(a, 1, axis=1), r=0) == 0.0          # raw IoU of skeletons
    assert tr.iou_dilated(np.zeros_like(a), np.zeros_like(a)) == 0.0


def test_flow_weighted_score_adds_a_term_and_keeps_the_veto():
    lab = _line_labels(60, 80, [30, 34])
    ref = ft.describe(lab, 1)
    cand = ft.describe(lab, 2)
    rm, cm = lab == 1, lab == 2
    dist = ft._distance_field(rm)
    kw = dict(ref_mask=rm, cand_mask=cm, ref_distance=dist)
    s0, t0 = ft.score_candidate(cand, ref, ref.centre, 2.0, **kw)
    s1, t1 = tr.flow_weighted_score(cand, ref, ref.centre, 2.0, predicted_mask=None, **kw)
    assert s1 == s0 and t1 == t0                                        # no mask: identical
    s2, t2 = tr.flow_weighted_score(cand, ref, ref.centre, 2.0, predicted_mask=cm, **kw)
    assert t2['flow_iou'] == 0.0 and s2 < s0                            # perfect flow match
    s3, t3 = tr.flow_weighted_score(cand, ref, ref.centre, 2.0, predicted_mask=rm, **kw)
    assert t3['flow_iou'] == 1.0 and s3 > s0                            # 4 cells off: no overlap
    # the weighted mean is the documented one
    used = sum(ft.WEIGHTS[k] for k in t0)
    assert s3 == pytest.approx((s0 * used + 1.0) / (used + 1.0))
    # zero weight: the term is not computed; the veto survives the extra term
    s4, t4 = tr.flow_weighted_score(cand, ref, ref.centre, 2.0, predicted_mask=cm,
                                    weights={**ft.WEIGHTS, 'flow_iou': 0.0}, **kw)
    assert 'flow_iou' not in t4 and s4 == s0
    far = _line_labels(60, 80, [30, 70])
    s5, _ = tr.flow_weighted_score(ft.describe(far, 2), ref, ref.centre, 2.0,
                                   predicted_mask=far == 2, ref_mask=rm, cand_mask=far == 2,
                                   ref_distance=dist)
    assert s5 == np.inf


class _UniformFlow:
    """A stand-in for :class:`tracking.Flow`: a rigid shift of ``di`` cells per step."""
    km_per_px = 1.9

    def __init__(self, di):
        self.di = di

    def advect(self, mask, s0, s1):
        return tr.advect_mask_by(mask, float(self.di * (s1 - s0)), 0.0)


@pytest.fixture(scope='module')
def moving_series():
    """Four steps.  Front A (label 1) moves +6 cells per step -- 3x the 2-cell search radius,
    so `follow`'s position veto rejects its true continuation; front B (label 2) sits still
    4 cells from A's start and is what plain `follow` grabs instead.  Front C (label 3) is a
    far-off bystander."""
    cols_A = [30, 36, 42, 48]
    series = {}
    for s in range(4):
        series[f'2012-07-02T{s:02d}_30_00'] = _line_labels(100, 120, [cols_A[s], 34, 90])
    return series, cols_A


def test_follow_flow_without_flow_equals_follow(moving_series):
    series, _ = moving_series
    labels_at, times = tr.labels_at_factory(series)
    for label in (2, 3):
        anchor = ft.anchor_at(labels_at(1), 1, label)
        plain = ft.follow(labels_at, times, anchor, km_per_px=1.9)
        mine = tr.follow_flow(labels_at, times, anchor, None, km_per_px=1.9)
        assert mine.labels == plain.labels and mine.centres == plain.centres
        for s in plain.links:
            assert mine.links[s].score == plain.links[s].score
            assert {k: v for k, v in mine.links[s].terms.items() if not k.startswith('_')} \
                == plain.links[s].terms


def test_flow_informed_walk_follows_the_fluid_where_plain_follow_cannot(moving_series):
    series, cols_A = moving_series
    labels_at, times = tr.labels_at_factory(series)
    anchor = ft.anchor_at(labels_at(0), 0, 1)                        # front A at step 0
    plain = tr.follow_flow(labels_at, times, anchor, None, km_per_px=1.9)
    flow = tr.follow_flow(labels_at, times, anchor, _UniformFlow(6.0), km_per_px=1.9)
    # plain: the true continuation (6 cells away) is vetoed; it links the stationary
    # neighbour B at step 1 and then stays on it
    assert plain.labels.get(1) == 2
    # flow-informed: the advected mask lands on A at every step
    assert [flow.labels[s] for s in range(4)] == [1, 1, 1, 1]
    for s in (1, 2, 3):
        assert flow.links[s].terms['flow_iou'] == 0.0
        assert flow.links[s].terms['_n_overlap'] == 1
        assert flow.links[s].terms['_pred_centre'][1] == pytest.approx(cols_A[s], abs=0.01)
    cmp = tr.compare_tracks([plain], [flow])
    assert int(cmp.loc[0, 'different']) >= 1


def test_track_top_n_and_table(moving_series):
    series, _ = moving_series
    tracks = tr.track_top_n(series, n=2, flow=_UniformFlow(6.0), anchor_step=1)
    assert len(tracks) == 2
    assert [t.anchor.label for t in tracks] == [1, 2] or [t.anchor.label for t in tracks] == [2, 1]
    tab = tr.track_table(tracks, list(series))
    assert set(tab.columns) >= {'anchor_label', 'n_found', 'span_h', 'n_gaps_in_span',
                                'worst_score', 'n_flow_links', 'n_split_merge'}
    a = tab.set_index('anchor_label')
    assert a.loc[1, 'n_found'] == 4 and a.loc[1, 'span_h'] == 3.0 and a.loc[1, 'n_flow_links'] == 3
    # json round trip
    back = tr.tracks_from_json(json.loads(json.dumps(tr.tracks_to_json(tracks, list(series)),
                                                     default=tr._json_default)))
    assert [t.labels for t in back] == [t.labels for t in tracks]
    assert back[0].links[0].terms.keys() == tracks[0].links[0].terms.keys()


@pytest.mark.needs_grid
def test_real_flow_sign_and_retention():
    """Transition 0 of the real stores.  The sign: advecting the pair-0 front mask with
    ``+d`` overlaps the pair-1 fronts better than with ``-d``, forwards and backwards.  What
    is NOT asserted, because it is false (task-2 log): that advection beats persistence --
    at the hourly cadence the fronts move 0.37 cells (median), so the thresholded skeleton
    is persistence plus rounding, and overlaps slightly less (0.704 vs 0.717)."""
    import inputs as inp
    p = tr.fronts_path(0)
    if not p.exists():
        pytest.skip(f'{p} not on disk (M4 task 1)')
    ds, grid_ds, grid, masks = inp.open_inputs()
    flow = tr.Flow(ds, grid_ds, grid)
    assert 1.7 < flow.km_per_px < 2.1                      # tile 330: median dxC 1.80, dyC 1.95
    labels = tr.open_fronts_series(p)
    keys = list(labels)
    m0, m1 = labels[keys[0]] > 0, labels[keys[1]] > 0
    di, dj = flow.displacement(0, +1)
    adv, wrong = tr.advect_mask_by(m0, di, dj), tr.advect_mask_by(m0, -di, -dj)
    assert 0.85 * m0.sum() < adv.sum() <= m0.sum()                  # 5% lost to rounding
    assert np.array_equal(adv, flow.advect(m0, 0, 1))
    assert tr.iou_dilated(adv, m1) > tr.iou_dilated(wrong, m1)       # the sign
    assert tr.iou_dilated(adv, m1) > 0.95 * tr.iou_dilated(m0, m1)   # not far off persistence
    di, dj = flow.displacement(0, -1)
    back, wrong = tr.advect_mask_by(m1, di, dj), tr.advect_mask_by(m1, -di, -dj)
    assert tr.iou_dilated(back, m0) > tr.iou_dilated(wrong, m0)


# ===========================================================================
# Task 3: the reconciliation on Lagrangian-matched pixels
# ===========================================================================
def test_forward_positions_is_the_inverse_of_the_arrival_map():
    """``x = pos + d(x)`` solved: the departure of the arrival point is the parcel's start."""
    nj, ni = 90, 120
    jj, ii = np.mgrid[0:nj, 0:ni].astype('f8')
    di = 1.5 + 0.8 * np.sin(jj / 11.0) * np.cos(ii / 13.0)
    dj = -0.7 + 0.5 * np.cos(ii / 9.0)
    rng = np.random.default_rng(3)
    pos = np.stack([rng.uniform(15, nj - 15, 200), rng.uniform(15, ni - 15, 200)], axis=1)
    x = tr.forward_positions(pos, di, dj)
    back = x - np.stack([tr.sample_at(dj, x), tr.sample_at(di, x)], axis=1)
    assert np.abs(back - pos).max() < 1e-8
    # a NaN displacement drops the parcel
    di2 = di.copy()
    di2[:, 60:] = np.nan
    x2 = tr.forward_positions(pos, di2, dj)
    assert np.isnan(x2[pos[:, 1] > 62]).all() and np.isfinite(x2[pos[:, 1] < 50]).all()


class _ShiftFlow(_UniformFlow):
    """Uniform ``di`` cells per step, with the displacement fields spelled out."""
    shape = (100, 120)

    def displacement(self, k, direction=+1):
        return (np.full(self.shape, float(self.di * direction)), np.zeros(self.shape))


@pytest.fixture(scope='module')
def growing_front():
    """Three steps.  A ridge in ``G`` at column 40 moves 3 cells per step with the flow and
    grows LINEARLY in time (``G_t = (1 + g t) R(x - 3t)``), so the material rate of the
    parcel starting at cell ``x`` is ``g R(x) / dt`` at every step -- the store holds exactly
    that, indexed at the parcel's cell at each step.  The label at step 1 is LONGER than at
    step 0 (it grew 20 rows at one end).  Forward and centred differences are both exact on
    this field, parcels land on cells, and the cubic sample is exact at nodes."""
    nj, ni = _ShiftFlow.shape
    jj, ii = np.mgrid[0:nj, 0:ni].astype('f8')
    g = 0.10
    R = (1.0 + 0.3 * np.sin(jj / 7.0)) * np.exp(-((ii - 40.0) / 2.5) ** 2) + 0.05
    G = [(1.0 + g * t) * np.roll(R, 3 * t, axis=1) for t in range(3)]
    D = [g * np.roll(R, 3 * t, axis=1) / 3600.0 for t in range(3)]
    labs = []
    for t, rows in enumerate(((20, 60), (20, 80), (20, 80))):
        lab = np.zeros((nj, ni), np.int32)
        lab[rows[0]:rows[1], 40 + 3 * t] = 1
        labs.append(lab)
    t = ['2012-07-02T00_30_00', '2012-07-02T01_30_00', '2012-07-02T02_30_00']
    ds = xr.Dataset({'G': (('time', 'j', 'i'), np.stack(G)),
                     'DGDt_semilag': (('time', 'j', 'i'), np.stack(D)),
                     **{k: (('time', 'j', 'i'), np.zeros((3, nj, ni))) for k in tr.BUDGET_TERMS}},
                    coords={'time': np.array(['2012-07-02T00:00', '2012-07-02T01:00',
                                              '2012-07-02T02:00'], dtype='datetime64[ns]')})
    ds['two_F'][:] = 0.5 * np.stack(D)
    return dict(labels=dict(zip(t, labs)), ds=ds, g=g)


@pytest.mark.parametrize('centred,band', [(False, 0), (True, 0), (False, 1), (True, 1)])
def test_matched_link_measures_the_material_rate_and_the_label_rate_does_not(growing_front,
                                                                              centred, band):
    gf = growing_front
    flow = _ShiftFlow(3.0)
    fields = tr.DerivedFields(gf['ds'])
    step = 1 if centred else 0
    row = tr.matched_link(step, 1, 1, gf['labels'], flow, fields, dt=3600.0,
                          centred=centred, band=band)
    n0 = (60 + 2 if centred else 40 + 2) * 3 if band else (60 if centred else 40)  # 3x3 dilation: +1 row each end
    assert row['n_parcels'] == n0 and row['n_lost'] == 0 and row['disp_median_cells'] == 3.0
    assert row['centred'] == centred and row['band'] == band
    assert row['r_matched'] == pytest.approx(row['r_store'], rel=1e-9)         # criterion 4
    assert row['r_forward'] == pytest.approx(row['r_store'], rel=1e-9)         # linear growth
    assert row['two_F'] == pytest.approx(0.5 * row['r_store'], rel=1e-9)
    if centred:
        assert row['G_mean_tp1_matched'] - row['G_mean_tm1_matched'] == pytest.approx(
            2 * gf['g'] * row['G_mean_t'] / (1 + gf['g']), rel=1e-9)
    else:
        assert row['G_mean_tp1_matched'] == pytest.approx(1.10 * row['G_mean_t'], rel=1e-9)
        # the thresholded-mask variant: an integer shift keeps every pixel
        assert row['n_mask'] == n0 and row['r_mask'] == pytest.approx(row['r_matched'], rel=1e-9)
        # the endpoint-label rate carries the set's growth (20 new rows, a different mean)
        assert row['n_label_tp1'] == (62 * 3 if band else 60)
        assert abs(row['set_evolution']) > 0.1 * abs(row['r_matched'])
        assert row['r_label'] != pytest.approx(row['r_matched'], rel=1e-3)
    # a gap at t+1: the matched rate is still defined, the label rate is not
    row2 = tr.matched_link(step, 1, None, gf['labels'], flow, fields, centred=centred, band=band)
    assert row2['r_matched'] == row['r_matched'] and np.isnan(row2['r_label'])
    # a centred link needs a predecessor
    assert tr.matched_link(0, 1, 1, gf['labels'], flow, fields, centred=True) is None


def test_reconcile_track_and_summary(growing_front):
    gf = growing_front
    flow = _ShiftFlow(3.0)
    fields = tr.DerivedFields(gf['ds'])
    anchor = ft.anchor_at(gf['labels']['2012-07-02T00_30_00'], 0, 1)
    track = ft.Track(anchor=anchor, labels={0: 1, 1: 1, 2: 1},
                     centres={0: anchor.centre, 1: (50, 43), 2: (50, 46)},
                     links={0: ft.Link(0, 1, 0.0, {}), 1: ft.Link(1, 1, 0.2, {'_n_overlap': 2}),
                            2: ft.Link(2, 1, 0.1, {'_n_overlap': 1})})
    df = tr.reconcile_track(track, gf['labels'], flow, fields)
    assert len(df) == 2 and bool(df.loc[0, 'split_merge']) and df.loc[0, 'anchor_label'] == 1
    dfc = tr.reconcile_track(track, gf['labels'], flow, fields, centred=True, band=1)
    assert len(dfc) == 1 and dfc.loc[0, 'step'] == 1 and dfc.loc[0, 'band'] == 1
    assert tr.reconcile_summary(dfc)['construction'] == {'centred': True, 'band': 1}
    df = df.iloc[:1]
    # a pooled summary needs >= 3 links: three copies with scatter stand in
    df3 = pd.concat([df] * 3, ignore_index=True)
    df3['r_store'] = df3['r_store'] * np.array([1.0, 2.0, 3.0])
    df3['r_matched'] = df3['r_store'] * np.array([1.1, 0.9, 1.0])
    s = tr.reconcile_summary(df3)
    assert s['n_links'] == 3 and s['matched_vs_store']['rms_ratio'] < 0.1 and s['passes']
    assert s['per_track'][1]['passes'] and s['n_tracks_passing'] == 1
    assert s['tolerance']['rms_ratio_max'] == tr.RECON_RMS_TOL
    df3['r_matched'] = df3['r_store'] * np.array([2.0, -1.0, 0.5])
    assert not tr.reconcile_summary(df3)['passes']


@pytest.mark.needs_grid
def test_real_forward_positions_match_the_case_study_integrator():
    """The stored-displacement inversion against ``m3_case.step_forward`` (the budget's own
    forward integrator on the raw velocities), transition 30, 300 front pixels."""
    import inputs as inp
    import m3_case
    p = tr.fronts_path(0)
    if not p.exists():
        pytest.skip(f'{p} not on disk (M4 task 1)')
    ds, grid_ds, grid, masks = inp.open_inputs()
    flow = tr.Flow(ds, grid_ds, grid)
    labels = tr.open_fronts_series(p)
    lab = labels[list(labels)[30]]
    jj, ii = np.nonzero(lab > 0)
    pos = np.stack([jj, ii], axis=1).astype('f8')[::30][:300]
    di, dj = flow.displacement(30, +1)
    x = tr.forward_positions(pos, di, dj)
    h = ds.isel(time=31).load()
    ui, vj = m3_case.index_velocity(h['U'], h['V'], grid_ds, grid)
    x_case, _ = m3_case.step_forward(pos, ui, vj)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(x_case).all(axis=1)
    assert ok.sum() > 250
    err = np.hypot(*(x - x_case)[ok].T)
    assert np.median(err) < 0.01 and np.percentile(err, 99) < 0.05


# ===========================================================================
# Task 4: tracking quality, split / merge
# ===========================================================================
def test_tracking_quality_plain_vs_flow(moving_series, tmp_path):
    """On the moving series: the flow walk follows front A (difference 0 at every link); the
    plain walk's link at step 1 is the stationary neighbour, 6 cells from where the fluid put
    the front."""
    series, cols_A = moving_series
    labels_at, times = tr.labels_at_factory(series)
    flow = _UniformFlow(6.0)
    anchor = ft.anchor_at(labels_at(0), 0, 1)
    plain = tr.follow_flow(labels_at, times, anchor, None, km_per_px=1.9)
    flowed = tr.follow_flow(labels_at, times, anchor, flow, km_per_px=1.9)
    qf = tr.tracking_quality([flowed], flow, series, name='flow')
    qp = tr.tracking_quality([plain], flow, series, name='plain')
    assert len(qf) == 3 and (qf['diff_cells'] < 1e-9).all() and (qf['mask_dist_median'] == 0).all()
    assert (qf['flow_iou'] == 1.0).all() and not qf['split'].any() and not qf['merge'].any()
    assert qf['chosen_is_majority'].all() and (qf['chosen_overlap_frac'] == 1.0).all()
    assert np.allclose(qf['pred_di'], 6.0) and np.allclose(qf['chosen_di'], 6.0)
    assert (qf['gap'] == 1).all() and (qf['direction'] == 'fwd').all()
    r1 = qp[qp['step'] == 1].iloc[0]
    assert r1['label'] == 2 and r1['diff_cells'] == pytest.approx(2.0, abs=0.05)   # B at 34, A at 36
    assert r1['mask_dist_median'] == pytest.approx(2.0) and r1['flow_iou'] == pytest.approx(0.2)  # 2 cells apart, dilated
    assert not r1['chosen_is_majority']
    sp, sf = tr.quality_summary(qp), tr.quality_summary(qf)
    assert sf['n_links'] == 3 and sf['centroid_diff_cells']['max'] < 1e-9 and sf['n_split_or_merge'] == 0
    assert sp['frac_chosen_is_majority'] < 1.0 and sp['per_track'][0]['anchor_label'] == 1
    # the figure runs on these rows and returns what it drew
    import figures_m4 as f4
    out = f4.fig_tracking_quality(qf, qp, fig_dir=tmp_path, name='tq_test')
    assert (tmp_path / 'tq_test.png').exists() and out['n_links_flow'] == 3 and out['n_split_flow'] == 0


def test_split_and_merge_flags():
    """Front 1 at step 0 is one line; at step 1 it is two pieces (labels 1 and 2) 4 cells
    apart along the line with a 10-row break: the prediction overlaps both -> split.  Run
    backwards (anchor at step 1 on one piece, step 0 one line) -> the chosen piece advected
    back overlaps one label; anchor on the whole line at step 0 after a two-piece step...
    the merge case is the reverse: pieces at step 0, one line at step 1."""
    nj, ni = 100, 120
    one = np.zeros((nj, ni), np.int32)
    one[20:80, 50] = 1
    two = np.zeros((nj, ni), np.int32)
    two[20:45, 50] = 1
    two[55:80, 50] = 2
    flow = _UniformFlow(0.0)
    # split: one line, then two pieces
    series = {'2012-07-02T00_30_00': one, '2012-07-02T01_30_00': two}
    labels_at, times = tr.labels_at_factory(series)
    track = tr.follow_flow(labels_at, times, ft.anchor_at(one, 0, 1), flow, km_per_px=1.9)
    q = tr.tracking_quality([track], flow, series)
    assert len(q) == 1 and bool(q.loc[0, 'split']) and not bool(q.loc[0, 'merge'])
    assert q.loc[0, 'n_fwd_overlap'] == 2 and sorted(q.loc[0, 'fwd_overlap_labels']) == [1, 2]
    assert 0.4 < q.loc[0, 'chosen_overlap_frac'] < 0.6
    # merge: two pieces, then one line (anchor on piece 1 at step 0)
    series = {'2012-07-02T00_30_00': two, '2012-07-02T01_30_00': one}
    labels_at, times = tr.labels_at_factory(series)
    track = tr.follow_flow(labels_at, times, ft.anchor_at(two, 0, 1), flow, km_per_px=1.9)
    q = tr.tracking_quality([track], flow, series)
    assert len(q) == 1 and bool(q.loc[0, 'merge']) and not bool(q.loc[0, 'split'])
    assert q.loc[0, 'n_bwd_overlap'] == 2
    s = tr.quality_summary(q)
    assert s['n_merge'] == 1 and s['n_split'] == 0 and s['per_track'][0]['merge_steps'] == [1]
    # a one-pixel touch is not a topology event
    touch = two.copy()
    touch[45:55, 50] = 0
    touch[44, 50] = 0
    touch[30, 50] = 2                     # one pixel of label 2 inside piece 1's span
    series = {'2012-07-02T00_30_00': one, '2012-07-02T01_30_00': touch}
    labels_at, times = tr.labels_at_factory(series)
    track = tr.follow_flow(labels_at, times, ft.anchor_at(one, 0, 1), flow, km_per_px=1.9)
    q = tr.tracking_quality([track], flow, series)
    assert bool(q.loc[0, 'split'])        # the real second piece still counts ...
    assert q.loc[0, 'n_fwd_overlap'] == 2  # ... and the one-pixel touch does not add a third
