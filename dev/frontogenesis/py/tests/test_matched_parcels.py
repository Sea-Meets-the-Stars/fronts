""" ``matched_parcels`` (track analysis, "Matched parcels" task 1, 2026-11-18): the per-parcel
form of ``tracking.matched_link`` and the two pixel sets, on ``test_tracking``'s growing front
(a ridge moving 3 cells per step with the flow; the label is 20 rows longer from step 1)."""
import numpy as np
import pytest

import tracking as tr
import matched_parcels as mp
from fronts import front_tracking as ft
from test_tracking import growing_front, _ShiftFlow              # noqa: F401  (fixture)


def _track(labels):
    keys = list(labels)
    anchor = ft.anchor_at(labels[keys[1]], 1, 1)
    return ft.Track(anchor=anchor, labels={0: 1, 1: 1, 2: 1}, centres={}, links={})


def test_parcel_link_is_matched_link_per_parcel(growing_front):
    gf, flow = growing_front, _ShiftFlow(3.0)
    fields = tr.DerivedFields(gf['ds'], names=('G', 'DGDt_semilag', 'two_F') + tr.BUDGET_TERMS)
    pl = mp.parcel_link(1, 1, gf['labels'], flow, fields, band=1)
    ref = tr.matched_link(1, 1, 1, gf['labels'], flow, fields, centred=True, band=1)
    assert pl['n_parcels'] == ref['n_parcels'] == 62 * 3 and pl['alive'].all()
    assert pl['r_matched'] == pytest.approx(ref['r_matched'], rel=1e-12)
    assert pl['r_store'] == pytest.approx(ref['r_store'], rel=1e-12)
    assert pl['two_F_mean'] == pytest.approx(ref['two_F'], rel=1e-12)
    # linear growth: every parcel's own rate is the stored rate at its cell
    assert np.allclose(pl['rate'], pl['store'], rtol=1e-9)
    assert np.allclose(pl['x_plus'][:, 1] - pl['i'], 3.0) and np.allclose(pl['x_minus'][:, 1] - pl['i'], -3.0)
    assert mp.parcel_link(0, 1, gf['labels'], flow, fields) is None
    assert mp.parcel_link(2, 1, gf['labels'], flow, fields) is None


def test_pixel_sets_and_table(growing_front):
    gf, flow = growing_front, _ShiftFlow(3.0)
    fields = tr.DerivedFields(gf['ds'], names=('G', 'DGDt_semilag', 'two_F') + tr.BUDGET_TERMS)
    track = _track(gf['labels'])
    first = mp.pixel_sets(track, 0, gf['labels'], flow)
    assert first['advected'] is None and first['npix'] == 40
    ps = mp.pixel_sets(track, 1, gf['labels'], flow)             # 40 rows advected, 60 chosen
    assert (ps['n_advected'], ps['n_both'], ps['n_chosen_only'], ps['n_advected_only']) == (40, 40, 20, 0)
    assert ps['n_gained'] == 19 and ps['n_lost'] == 0            # one new row touches the old end
    assert not ps['split'] and not ps['merge']
    ps = mp.pixel_sets(track, 2, gf['labels'], flow)
    assert ps['n_both'] == ps['npix'] == 60 and ps['n_gained'] == 0
    df = mp.analyse_track(track, gf['labels'], flow, fields)
    assert list(df['step']) == [0, 1, 2] and df['r_matched'].notna().tolist() == [False, True, False]
    s = mp.track_summary(df)
    assert s['n_rates'] == 1 and s['n_links'] == 2 and s['n_split_or_merge'] == 0
