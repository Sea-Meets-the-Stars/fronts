""" Tests for ``m3_term_tests.py`` (M3-digging task 3), offline and synthetic.

Pinned: the split identity on random (non-uniform) fluxes through the real
``compute_budget`` path -- the three components sum to the stored
``surface_flux`` (and ``surface_flux_kpp``) to 1e-10 relative; a zero flux
array keeps its attrs and passes the sign guard; ``inputs.W_k2`` picks
``k_l = 2`` and refuses a store with two interfaces; the second cell's term is
zero when ``b_k1 = b_k2``; the multiplier fit recovers ``(1, 1, 1)`` on a pool
built as the unit sum plus noise, and the declared rule reads it; the
declared block reaches the disk before any store is opened.
"""
import json

import numpy as np
import pytest
import xarray as xr

import budget as bg
import inputs as inp
import m3_term_tests as tt
import stats as st
import vertical as vt
from test_budget import pair, chunk_for


def _ramped(chunk, seed=0):
    """Non-uniform fluxes: a linear ramp plus noise, downward-positive, with
    ``oceQsw >= 0`` as the store requires."""
    rng = np.random.default_rng(seed)
    nt, nj, ni = chunk['oceQnet'].shape
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')
    for v, base, amp in (('oceQnet', 100.0, 80.0), ('oceQsw', 200.0, 120.0), ('oceFWflx', -1e-5, 2e-5)):
        a = base + amp * (ii - ni / 2) / ni + 0.2 * amp * rng.normal(size=(nt, nj, ni))
        if v == 'oceQsw':
            a = np.abs(a)
        attrs = dict(chunk[v].attrs)
        chunk[v] = (('time', 'j', 'i'), a)
        chunk[v].attrs.update(attrs)
    chunk['KPPhbl_dummy'] = chunk['oceQnet'] * 0 + 20.0
    return chunk


@pytest.fixture(scope='module')
def case():
    c = pair()
    c, ch = chunk_for(c, w=1e-4, dT=0.3)
    ch = _ramped(ch)
    kpp = 20.0 + 5.0 * np.linspace(-1, 1, c['ds'].sizes['i'])[None, None, :] * np.ones((2, c['ds'].sizes['j'], 1))
    c = dict(c, ds=c['ds'].assign(KPPhbl=(('time', 'j', 'i'), kpp)))
    ds, _ = bg._merged(c['ds'], ch)
    return c, ds


@pytest.mark.parametrize('L', [0, 2])
def test_split_identity_through_compute_budget(case, L):
    c, ds = case
    bd = bg.compute_budget(ds, c['g'], c['grid'], c['mask'], L)
    h0, h1 = bg._hour_pair(ds, 0, True)
    sp = tt.split_components(h0, h1, L, c['g'], c['grid'], inp.drF(ds))
    for depth, stored in (('top', 'surface_flux'), ('kpp', 'surface_flux_kpp')):
        assert sp[depth]['identity_rel_rms'] <= tt.IDENTITY_TOL
        s = sum(sp[depth][k] for k in tt.COMP)
        ref = np.squeeze(np.asarray(bd[stored].values, 'float64'))
        ok = np.isfinite(ref) & np.isfinite(s)
        assert ok.sum() > 100
        np.testing.assert_allclose(s[ok], ref[ok], rtol=1e-9, atol=1e-30)
        # each component is a real part of the term: none is identically zero
        for k in tt.COMP:
            assert np.nanmax(np.abs(sp[depth][k])) > 0
    chk = tt.identity_check(sp, {d: np.squeeze(bd[s].values) for d, s in tt.STORED.items()},
                            np.squeeze(bd['valid'].values))
    assert all(chk[d]['split_ok'] for d in tt.DEPTHS)
    assert all(chk[d]['vs_store_rel_rms'] < 1e-6 for d in tt.DEPTHS)


def test_zero_flux_arrays_keep_their_attrs_and_pass_the_sign_guard(case):
    c, ds = case
    h0, h1 = bg._hour_pair(ds, 0, True)
    qnet, qsw, fw = [inp.midpoint(a, b) for a, b in zip(inp.fluxes(h0), inp.fluxes(h1))]
    var = tt.flux_variants(qnet, qsw, fw)
    for name, (a, b, cc) in var.items():
        for da in (a, b, cc):
            assert str(da.attrs.get('sign_convention', '')).startswith('positive downward')
        vt._check_flux_sign(oceQnet=a, oceQsw=b, oceFWflx=cc)       # does not raise
    assert float(np.abs(var['freshwater'][0].values).max()) == 0.0
    assert float(np.abs(var['nonsolar'][1].values).max()) == 0.0
    np.testing.assert_allclose(var['nonsolar'][0].values, qnet.values - qsw.values)
    np.testing.assert_allclose(var['shortwave'][0].values, qsw.values)
    assert np.all(var['shortwave'][1].values >= 0)


def test_W_k2_picks_the_second_interface_and_refuses_two(case):
    c, ds = case
    w2 = inp.W_k2(ds)
    assert 'k_l' not in w2.dims and w2.attrs['k_l'] == 2
    np.testing.assert_array_equal(w2.values, ds['W_k'].isel(k_l=2).values)
    with pytest.raises(ValueError, match='3 interfaces'):
        inp.W_k2(ds.isel(k_l=slice(0, 2)))
    with pytest.raises(ValueError):
        inp.W_k1(ds.isel(k_l=slice(0, 1)))


def test_second_cell_term_is_zero_when_b_k1_equals_b_k2(case):
    c, ds = case                                   # chunk_for: levels 1 and 2 identical
    h0, h1 = bg._hour_pair(ds, 0, True)
    v = tt.vertical_variants(h0, h1, 0, c['g'], c['grid'], inp.drF(ds))
    assert np.nanmax(np.abs(v['V_cell2'])) == 0.0
    assert np.nanmax(np.abs(v['V_top'])) > 0.0        # dT = 0.3 stratifies the top cell
    assert abs(v['dz_cell2_m'] - 0.5 * (1.14 + 1.30)) < 1e-12 and abs(v['dz_top_m'] - 1.07) < 1e-12
    bd = bg.compute_budget(ds, c['g'], c['grid'], c['mask'], 0)
    ref = np.squeeze(bd['vertical'].values)
    ok = np.isfinite(ref)
    np.testing.assert_allclose(v['V_top'][ok], ref[ok], rtol=1e-6)
    ref1 = np.squeeze(bd['vertical_factorised'].values)
    np.testing.assert_allclose(v['V_fact_o1'][ok], ref1[ok], rtol=1e-6)


def test_fit_recovers_unit_multipliers_and_the_rule_reads_it():
    rng = np.random.default_rng(0)
    n = 40000
    S = {k: rng.normal(size=n) * s for k, s in (('nonsolar', 1.0), ('shortwave', 0.5), ('freshwater', 0.2))}
    I = sum(S.values()) + 0.3 * rng.normal(size=n)
    lab = st.block_ids((200, 200)).ravel()
    f = tt.fit_multipliers(I, S, lab, n_boot=200)
    for k in S:
        m = f['multipliers'][k]
        assert abs(m['value'] - 1.0) < 0.05
        assert m['lo'] - 3 * m['se'] < 1.0 < m['hi'] + 3 * m['se']   # within sampling of 1
        assert 0 < m['se'] < 0.05 and m['lo'] < m['value'] < m['hi']
    assert abs(f['gain']['value']) < 0.01 and f['rms']['fit'] <= f['rms']['unit'] + 1e-12
    r = tt.cancellation(f)
    assert all(r['multipliers_in_bounds'].values()) and not r['gain_exceeds_half_width'] and not r['passed']
    # a term entering at half weight: the fit finds it, the rule calls it a cancellation
    I2 = S['nonsolar'] + 0.5 * S['shortwave'] + S['freshwater'] + 0.05 * rng.normal(size=n)
    f2 = tt.fit_multipliers(I2, S, lab, n_boot=200)
    assert abs(f2['multipliers']['shortwave']['value'] - 0.5) < 0.03
    r2 = tt.cancellation(f2)
    assert r2['gain_exceeds_half_width'] and r2['passed']
    # a term that is pure noise w.r.t. I: multiplier ~0, out of bounds
    I3 = S['nonsolar'] + S['freshwater'] + 0.3 * rng.normal(size=n)
    f3 = tt.fit_multipliers(I3, S, lab, n_boot=200)
    assert abs(f3['multipliers']['shortwave']['value']) < 0.05
    assert not tt.cancellation(f3)['multipliers_in_bounds']['shortwave']


def test_the_declared_block_is_written_before_any_store_is_opened(tmp_path, monkeypatch):
    out = tmp_path / 'tt.json'

    def boom(*a, **k):
        raise RuntimeError('store opened')
    monkeypatch.setattr(inp, 'open_inputs', boom)
    with pytest.raises(RuntimeError, match='store opened'):
        tt.main(['--out', str(out), '--no-fig'])
    d = json.loads(out.read_text())
    assert d['declared'] == tt.DECLARED and 'not yet opened' in d['status']
    assert tt.DECLARED['split']['multiplier_bounds'] == [0.5, 2.0]
    assert tt.DECLARED['vertical']['pairs'] == [0, 9, 21, 30]


def test_figure_draws_from_memory_and_from_a_json_round_trip(tmp_path):
    """The run draws from the in-memory result (int local-hour keys); a redraw
    reads the JSON (str keys).  Both must work -- the first run's figure
    failed on exactly this."""
    import json as _json
    from m3_term_tests_fig import figure
    rng = np.random.default_rng(0)
    n = 3000
    S = {k: rng.normal(size=n) for k in tt.COMP}
    I = sum(S.values()) + rng.normal(size=n)
    lab = st.block_ids((50, 60)).ravel()
    f3 = tt.fit_multipliers(I, S, lab, n_boot=20)
    hour = {h: dict(n=10, I_mean_over_rms_meas=0.0, S_mean_over_rms_meas=-0.1,
                    **{f'{c}_mean_over_rms_meas': 0.01 for c in tt.COMP},
                    **{f'{c}_rms_over_meas': 0.1 for c in tt.COMP}) for h in range(24)}
    depth = dict(fit3=f3, cancellation=tt.cancellation(f3), by_local_hour=hour)
    row = dict(L=0, pair=0, lst=16.5, terms={k: dict(rms_over_meas=0.02, sign_of_mean=-1)
                                             for k in ('V_top', 'V_fact_o1', 'V_fact_o2', 'V_cell2')},
               cell2_on_top=dict(ols=1.2, corr=0.9))
    res = dict(declared=tt.DECLARED, split=dict(per_L={'0': dict(depth=dict(top=depth))}),
               vertical=dict(rows=[row]))
    figure(res, tmp_path / 'a.png')
    figure(_json.loads(_json.dumps(res)), tmp_path / 'b.png')
    assert (tmp_path / 'a.png').exists() and (tmp_path / 'b.png').exists()
