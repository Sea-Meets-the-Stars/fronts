""" Tests for ``m3_closure.py`` (M3 task 6), offline and synthetic.

The gate's arithmetic is what these pin: that the **pre-declared** constants
are the ones used, that a verdict flips exactly at the tolerance and not
near it, that ``L = 0`` is reported but never gated, that the slopes are
labelled unquotable whenever criterion 1 fails, and that the two
diagnostics task 6 added -- the optimal-multiplier test for a term and the
partial correlation of the residual on ``lap2_b`` vs ``KPPhbl`` -- say what
they are supposed to say on data built to have a known answer.

No store is opened: ``closure`` and ``overall`` are fed synthetic pools.
"""

import numpy as np
import pytest

import m3_closure as mc
import stats as st


def pool_with(n=20000, resid_frac=0.3, slope_resid=0.0, seed=0, terms=True):
    """A synthetic front pool whose residual has a prescribed size and a
    prescribed slope on ``2F`` -- so a verdict can be aimed at either side
    of either tolerance."""
    rng = np.random.default_rng(seed)
    F2 = rng.normal(size=n)
    sub = 0.2 * rng.normal(size=n)
    vert = 0.01 * rng.normal(size=n)
    sfl = 0.1 * rng.normal(size=n)
    res = slope_resid * F2 + resid_frac * rng.normal(size=n)
    meas = F2 + sub + (vert + sfl if terms else 0.0) + res
    p = dict(two_F=F2, subfilter=sub, vertical=vert, surface_flux=sfl,
             DGDt_semilag=meas, residual=meas - F2 - sub - (vert + sfl if terms else 0.0),
             DGDt_euler=meas + 0.05 * rng.normal(size=n),
             DGDt_semilag_o5=meas + 0.01 * rng.normal(size=n),
             two_F_chain=F2 * 0.8, lap2_b=rng.normal(size=n), KPPhbl=20 + rng.normal(size=n),
             front_width=np.abs(rng.normal(2.5, 1.0, n)), G=np.abs(rng.normal(size=n)),
             coast_distance_km=np.full(n, 150.0), lst=rng.uniform(0, 24, n),
             block=rng.integers(0, 200, n), block3=rng.integers(0, 70, n),
             pair=rng.integers(0, 71, n))
    return p


def rows_for(pool, n_pairs=71):
    return [dict(pair=p, resid_over_meas=0.8, catchall_over_meas=0.7, explained=0.3,
                 resid_on_2F=0.4, meas_on_2F=1.3) for p in range(n_pairs)]


# ---------------------------------------------------------------------------
# the pre-declaration is the thing actually used
# ---------------------------------------------------------------------------
def test_the_declared_constants_are_the_q_answers():
    assert mc.TOL_RESID_RATIO == 0.50 and mc.TOL_EXPLAINED == 0.75
    assert mc.TOL_RESID_SLOPE == 0.10
    assert mc.TOL_EULER_SLOPE == (0.85, 1.15) and mc.TOL_EULER_CORR == 0.90
    assert mc.GATE_L == (2, 4, 8) and 0 not in mc.GATE_L
    assert mc.FRONT_PCT == 90.0 and mc.FRONT_PCT_SENS == (80.0, 95.0)
    assert mc.DAY3_PAIRS == tuple(range(62, 69))
    assert mc.WIDTH_LABELS == ('<=1', '1-1.5', '1.5-2', '2-3', '3-4', '>4')
    d = mc.DECLARED
    assert d['criterion1']['resid_ratio_max'] == mc.TOL_RESID_RATIO
    assert d['criterion1']['judged_at'] == list(mc.GATE_L)
    assert d['baseline'] == st.BASELINE == 0.981          # never 1
    assert '2026-10-07' in d['source']


def test_budget_and_closure_constants_agree():
    """``budget.closure_report`` and ``m3_closure`` must be judging against
    the same numbers; a drift between them would be two different gates."""
    import budget as bg
    assert bg.Q1_RESID_RATIO_MAX == mc.TOL_RESID_RATIO
    assert bg.Q1_EXPLAINED_MIN == mc.TOL_EXPLAINED
    assert bg.Q1_RESID_SLOPE_TOL == mc.TOL_RESID_SLOPE
    assert bg.Q2_EULER_SLOPE == mc.TOL_EULER_SLOPE
    assert bg.Q2_EULER_CORR_MIN == mc.TOL_EULER_CORR
    assert bg.FRONT_PCT == mc.FRONT_PCT


# ---------------------------------------------------------------------------
# the verdict flips at the tolerance, not near it
# ---------------------------------------------------------------------------
def test_a_small_residual_passes_and_a_large_one_fails():
    good = mc.closure(pool_with(resid_frac=0.05, seed=1), rows_for(None), 2)
    assert good['residual']['over_measured'] < mc.TOL_RESID_RATIO
    assert good['verdict']['passed'] is True and good['verdict']['gated'] is True
    bad = mc.closure(pool_with(resid_frac=1.5, seed=2), rows_for(None), 2)
    assert bad['residual']['over_measured'] > mc.TOL_RESID_RATIO
    assert bad['verdict']['passed'] is False


def test_a_term_missing_in_proportion_to_2F_fails_the_slope_check_alone():
    """M3-Q1 (b) exists for exactly this: a residual small enough in rms but
    proportional to ``2F`` is a missing term, not noise."""
    c = mc.closure(pool_with(resid_frac=0.02, slope_resid=0.3, seed=3), rows_for(None), 4)
    ch = c['verdict']['checks']
    assert ch['resid_ratio']['passed'] is True            # small in rms
    assert ch['resid_slope']['passed'] is False           # but tracks 2F
    assert c['verdict']['passed'] is False


def test_L0_is_reported_but_never_gated():
    c = mc.closure(pool_with(resid_frac=0.05, seed=4), rows_for(None), 0)
    assert c['verdict']['gated'] is False
    assert 'NOT gated' in c['verdict']['note']
    assert mc.euler(pool_with(seed=4), pool_with(seed=5), 0)['gated'] is False
    assert mc.euler(pool_with(seed=4), pool_with(seed=5), 2)['gated'] is True


# ---------------------------------------------------------------------------
# the two diagnostics task 6 added
# ---------------------------------------------------------------------------
def test_term_contribution_separates_a_sign_error_from_an_uncorrelated_term():
    """The optimal multiplier is ~+1 for a correct term, ~-1 for one entering
    with the wrong sign, and ~0 for one that is simply uncorrelated with the
    residual it should explain -- three different findings."""
    rng = np.random.default_rng(6)
    n = 40000
    F2 = rng.normal(size=n)
    true_v = 0.3 * rng.normal(size=n)
    noise = 0.1 * rng.normal(size=n)
    base = dict(two_F=F2, subfilter=np.zeros(n), KPPhbl=np.full(n, 20.0),
                DGDt_euler=np.zeros(n), DGDt_semilag_o5=np.zeros(n))
    # (i) a correct term
    meas = F2 + true_v + noise
    p = dict(base, vertical=true_v, surface_flux=np.zeros(n), DGDt_semilag=meas,
             residual=meas - F2 - true_v)
    t = mc.closure(p, rows_for(None), 2)['term_contribution']['vertical']
    assert t['optimal_multiplier'] == pytest.approx(1.0, abs=0.05) and t['helps']
    assert 'explains part' in t['diagnosis']
    # (ii) the same term with the sign flipped on the way in
    p = dict(base, vertical=-true_v, surface_flux=np.zeros(n), DGDt_semilag=meas,
             residual=meas - F2 + true_v)
    t = mc.closure(p, rows_for(None), 2)['term_contribution']['vertical']
    assert t['optimal_multiplier'] == pytest.approx(-1.0, abs=0.05)
    assert 'sign error' in t['diagnosis'] and not t['helps']
    # (iii) a large term uncorrelated with the residual
    junk = 0.5 * rng.normal(size=n)
    meas2 = F2 + noise
    p = dict(base, vertical=np.zeros(n), surface_flux=junk, DGDt_semilag=meas2,
             residual=meas2 - F2 - junk)
    t = mc.closure(p, rows_for(None), 2)['term_contribution']['surface_flux']
    assert abs(t['optimal_multiplier']) < 0.25 and not t['helps']
    assert 'uncorrelated' in t['diagnosis']
    assert t['rms_subtracting'] > t['rms_without']        # subtracting it makes things worse


def test_partial_corr_removes_the_shared_driver():
    """``lap2_b`` and ``KPPhbl`` are themselves correlated on the real tile,
    so the raw correlations cannot answer Figure 2b's question."""
    rng = np.random.default_rng(7)
    n = 20000
    shared = rng.normal(size=n)
    lap = shared + 0.3 * rng.normal(size=n)
    kpp = shared + 0.3 * rng.normal(size=n)
    res = lap + 0.5 * rng.normal(size=n)                  # driven by lap only
    assert np.corrcoef(res, kpp)[0, 1] > 0.4              # KPPhbl looks guilty
    assert mc.partial_corr(res, kpp, lap) == pytest.approx(0.0, abs=0.05)   # it is not
    assert mc.partial_corr(res, lap, kpp) > 0.4
    assert np.isnan(mc.partial_corr(np.zeros(2), np.zeros(2), np.zeros(2)))


def test_width_bin_edges_are_the_declared_ones():
    w = np.array([0.5, 1.0, 1.2, 1.5, 1.9, 2.0, 2.9, 3.5, 9.0, np.nan])
    b = mc.width_bin(w)
    assert list(b) == [0, 0, 1, 1, 2, 2, 3, 4, 5, -1]   # (lo, hi], so 1.0 is '<=1'
    assert mc.WIDTH_EDGES[1:-1] == (1.0, 1.5, 2.0, 3.0, 4.0)


def test_catchall_is_the_no_chunk_identity():
    p = pool_with(seed=8)
    c = mc.closure(p, rows_for(None), 2)
    expect = p['residual'] + p['vertical'] + p['surface_flux']
    assert c['catchall']['rms'] == pytest.approx(
        float(np.sqrt(np.nanmean(expect ** 2))), rel=1e-12)
    assert 'no chunk store' in c['catchall']['note']


# ---------------------------------------------------------------------------
# the overall verdict and the "not quoted" rule
# ---------------------------------------------------------------------------
def _per_L(passes):
    return {str(L): dict(closure=dict(verdict=dict(passed=p, gated=L in mc.GATE_L)),
                         euler=dict(passed=True)) for L, p in passes.items()}


def test_overall_null_rule_follows_M3_Q9():
    """A failure at every gated ``L`` is planning §12's null; a failure at
    ``L = 0`` alone is not, because ``L = 0`` is not gated."""
    all_fail = mc.overall(_per_L({0: False, 2: False, 4: False, 8: False}))
    assert all_fail['null_result'] is True and all_fail['efficiency_quotable'] is False
    assert 'NO frontogenesis efficiency' in all_fail['statement']
    only_L0 = mc.overall(_per_L({0: False, 2: True, 4: True, 8: True}))
    assert only_L0['null_result'] is False and only_L0['efficiency_quotable'] is True
    one_passes = mc.overall(_per_L({0: False, 2: False, 4: True, 8: False}))
    assert one_passes['null_result'] is False
    assert one_passes['criterion1']['per_L'] == {2: False, 4: True, 8: False}


def test_the_real_summary_marks_every_slope_unquotable(tmp_path):
    """The run that was actually performed: criterion 1 failed everywhere, so
    every ``L`` must carry ``slopes_not_quoted`` and none may carry
    ``slopes``.  Skipped if the summary is not on disk."""
    import json
    p = mc.OUT_JSON
    if not p.exists():
        pytest.skip(f'{p} not on disk (task 6 not run here)')
    d = json.loads(p.read_text())
    assert d['verdict']['null_result'] is True
    for L, v in d['per_L'].items():
        assert v['slopes_quotable'] is False, L
        assert 'slopes' not in v and 'slopes_not_quoted' in v, L
        assert 'NOT QUOTED' in v['slopes_label']
    assert d['declared']['criterion1']['resid_ratio_max'] == mc.TOL_RESID_RATIO
