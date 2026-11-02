""" Tests for ``m3_controls.py`` (M3-digging task 2), offline and synthetic.

A small synthetic "store" in memory -- a handful of pairs on a 96 x 96 grid
with a land block, hourly local-solar times, a p90 front mask -- is what the
controls are run on.  What is pinned: the permutation of a pool whose ``2F``
*is* the measured side gives a residual ratio of ``sqrt(2)``; the 24-h lag
keeps the local-hour label; a roll never maps a pixel onto itself; the
``G_prev`` selection reduces to the p90 selection when ``G`` is constant in
time; the moment-form estimators equal their direct forms; and the declared
block reaches the disk before any store is opened.
"""
import json

import numpy as np
import pytest

import m3_controls as ct
import stats as st

NJ = NI = 96
N_PAIRS = 30


def synth(n=N_PAIRS, seed=1, two_F_is_measured=False, G_constant=False):
    rng = np.random.default_rng(seed)
    shape = (NJ, NI)
    land = np.zeros(shape, bool)
    land[:12, :20] = True
    valid = np.repeat(~land[None], n, axis=0)
    F = {}
    F['two_F'] = rng.normal(size=(n,) + shape).astype('f4')
    F['subfilter'] = (0.2 * rng.normal(size=(n,) + shape)).astype('f4')
    F['vertical'] = (0.01 * rng.normal(size=(n,) + shape)).astype('f4')
    F['surface_flux'] = (0.1 * rng.normal(size=(n,) + shape)).astype('f4')
    if two_F_is_measured:
        meas = F['two_F'].copy()
        for k in ('subfilter', 'vertical', 'surface_flux'):
            F[k][:] = 0.0
    else:
        meas = (F['two_F'] + F['subfilter'] + F['vertical'] + F['surface_flux']
                + 0.5 * rng.normal(size=(n,) + shape)).astype('f4')
    F['DGDt_semilag'] = meas
    F['DGDt_euler'] = (meas + 0.05 * rng.normal(size=(n,) + shape)).astype('f4')
    F['residual'] = (meas - F['two_F'] - F['subfilter'] - F['vertical'] - F['surface_flux']).astype('f4')
    G = np.abs(rng.normal(size=(n,) + shape)).astype('f4')
    if G_constant:
        G[:] = G[0]
    F['G'] = G
    F['sigma_mag'] = np.abs(rng.normal(size=(n,) + shape)).astype('f4')
    for k in list(F):
        F[k][:, land] = np.nan
    F['valid'] = valid
    front = np.zeros((n,) + shape, bool)
    for p in range(n):
        front[p] = valid[p] & (G[p] >= np.nanpercentile(G[p][valid[p]], 90))
    F['front'] = front
    F['lst'] = (16.5 + np.arange(n)) % 24.0
    F['n'], F['shape'], F['L'], F['path'] = n, shape, 0, 'synthetic'
    F['pred'] = ct.PRED
    return F


def test_permutation_of_a_pool_whose_2F_is_measured_gives_sqrt2():
    F = synth(two_F_is_measured=True)
    rng = np.random.default_rng(0)
    fam = ct.control_permutation(F, rng, n_perm=6, say=lambda *a, **k: None)
    med = fam['dist']['resid_over_meas']['median']
    assert abs(med - np.sqrt(2.0)) < 0.05, med          # independent, equal variance
    real = ct.pooled_stats(ct.gather(F, None, stored_residual=True))
    assert real['resid_over_meas'] < 1e-6                 # the real pool closes exactly
    assert abs(real['independent_reference'] - np.sqrt(2.0)) < 1e-6
    for m in fam['members']:
        assert not np.any(np.asarray(m['perm']) == np.arange(F['n']))   # a derangement


def test_the_24h_lag_keeps_the_local_hour_and_drops_the_tail():
    F = synth()
    c = ct.control_lag(F, 24, n_boot=50, say=lambda *a, **k: None)
    assert c['n_pairs'] == F['n'] - 24 and c['lag_h'] == 24
    P = ct.gather(F, [n + 24 if n + 24 < F['n'] else None for n in range(F['n'])])
    assert set(np.unique(P['pair']).tolist()) == set(range(F['n'] - 24))
    for n in np.unique(P['pair']):
        assert np.all(P['lst'][P['pair'] == n] == F['lst'][n])
        assert abs((F['lst'][n + 24] - F['lst'][n]) % 24.0) < 1e-9   # the same solar hour
    assert set(c['by_local_hour']) <= set(range(24))


def test_a_roll_never_maps_a_pixel_onto_itself_and_respects_the_minimum():
    rng = np.random.default_rng(0)
    for _ in range(200):
        dj, di = ct.draw_roll(rng, (720, 720))
        assert abs(dj) >= ct.ROLL_MIN and abs(di) >= ct.ROLL_MIN
        assert abs(dj) <= 720 - ct.ROLL_MIN and abs(di) <= 720 - ct.ROLL_MIN
    F = synth()
    fam = ct.control_roll(F, rng, n_roll=3, say=lambda *a, **k: None)
    assert fam['n_draws'] == 3 and all(abs(m['roll'][0]) >= 64 for m in fam['members'])
    # a rolled pool is decorrelated: the residual grows towards the reference
    real = ct.pooled_stats(ct.gather(F, None, stored_residual=True))
    assert fam['dist']['resid_over_meas']['median'] > real['resid_over_meas']


def test_G_prev_selection_equals_p90_when_G_is_constant_in_time():
    F = synth(G_constant=True)
    for n in range(1, F['n']):
        a = ct.select_G_prev(F, n)
        b = ct.default_select(F, n)
        assert np.array_equal(a, b)
    assert not ct.select_G_prev(F, 0).any()                 # pair 0 dropped
    F2 = synth()
    assert not np.array_equal(ct.select_G_prev(F2, 5), ct.default_select(F2, 5))
    assert not ct.select_persist(F2, 0).any() and not ct.select_persist(F2, F2['n'] - 1).any()
    p = ct.select_persist(F2, 5)
    assert np.all(p <= (F2['front'][4] & F2['front'][5] & F2['front'][6]))


def test_moment_form_estimators_match_their_direct_forms():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    y = 0.4 * x + rng.normal(size=5000)
    lab = st.block_ids((50, 100))
    for est in (ct.ratio_rms, ct.explained):
        b = st.block_bootstrap(x, y, lab.ravel(), estimator=est, n=50, seed=0)
        assert b['path'] == 'sufficient statistics'
        assert abs(b['value'] - est(x, y)) < 1e-12
        assert b['lo'] <= b['value'] <= b['hi']
    assert abs(ct.ratio_rms(x, y) - np.sqrt(np.mean(y ** 2) / np.mean(x ** 2))) < 1e-12
    assert abs(ct.explained(x, y) - (1 - np.var(y) / np.var(x))) < 1e-12


def test_real_pool_reproduces_the_stored_residual_and_the_blocks_widen_or_not(tmp_path):
    F = synth()
    real, P = ct.real_pool(F, n_boot=100, say=lambda *a, **k: None)
    rec = ct.pooled_stats(ct.gather(F, None))                # recomputed residual
    assert abs(real['resid_over_meas'] - rec['resid_over_meas']) < 1e-5
    assert set(real['ci']) == {'1h', '3h', '12h', '24h'}
    b = ct.blocks(real)
    assert b['3h']['width_over_1h']['resid_over_meas'] is not None
    assert all(b[h]['resid_over_meas']['n_blocks'] > 1 for h in b)


def test_skill_rule_is_the_declared_one():
    F = synth()
    rng = np.random.default_rng(0)
    real, _ = ct.real_pool(F, n_boot=50, say=lambda *a, **k: None)
    controls = dict(permutation=ct.control_permutation(F, rng, 4, say=lambda *a, **k: None),
                    lag24=ct.control_lag(F, 24, n_boot=50, say=lambda *a, **k: None),
                    lag48=ct.control_lag(F, 25, n_boot=50, say=lambda *a, **k: None),
                    roll=ct.control_roll(F, rng, 2, say=lambda *a, **k: None))
    s = ct.skill(real, controls)
    assert s['passed'] == (s['below_perm_p5'] and s['below_lag24_by_halfwidth'])
    assert s['below_perm_p5'] == (s['real'] < s['perm_p5'])
    assert s['below_lag24_by_halfwidth'] == (s['real'] < s['lag24'] - s['real_half_width'])
    assert s['passed']                                      # a closing synthetic budget has skill
    g = ct.gaps(real, controls)
    assert g['resid_over_meas']['permutation']['gap'] < 0


def test_the_declared_block_is_written_before_any_store_is_opened(tmp_path, monkeypatch):
    out = tmp_path / 'ctrl.json'

    def boom(*a, **k):
        raise RuntimeError('store opened')
    monkeypatch.setattr(ct, 'load_fields', boom)
    with pytest.raises(RuntimeError, match='store opened'):
        ct.main(['--out', str(out), '--no-fig', '--no-case', '--L', '0'])
    d = json.loads(out.read_text())
    assert d['declared'] == ct.DECLARED and 'not yet opened' in d['status']
    assert ct.DECLARED['skill_rule']['perm_percentile'] == 5.0
    assert ct.DECLARED['seeds']['n_perm'] == 20 and ct.DECLARED['seeds']['roll_min_cells'] == 64


def test_format_table_runs_on_a_synthetic_result(tmp_path):
    F = synth()
    rng = np.random.default_rng(0)
    real, _ = ct.real_pool(F, n_boot=20, say=lambda *a, **k: None)
    controls = dict(permutation=ct.control_permutation(F, rng, 2, say=lambda *a, **k: None),
                    lag24=ct.control_lag(F, 24, n_boot=20, say=lambda *a, **k: None),
                    lag48=ct.control_lag(F, 25, n_boot=20, say=lambda *a, **k: None),
                    roll=ct.control_roll(F, rng, 2, say=lambda *a, **k: None))
    d = dict(L=0, real=real, controls=controls, gaps=ct.gaps(real, controls),
             skill=ct.skill(real, controls), selections=ct.selections(F, say=lambda *a, **k: None),
             blocks=ct.blocks(real))
    res = dict(declared=ct.DECLARED, per_L={'0': d}, verdict_note='x')
    txt = ct.format_table(res)
    assert 'G_prev' in txt and 'persist' in txt and 'lag24' in txt
    from m3_controls_fig import figure
    figure(res, tmp_path / 'f.png')
    assert (tmp_path / 'f.png').exists()
