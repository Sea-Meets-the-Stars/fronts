""" Tests for ``stats.py`` (M3 task 5; coding §4.8, planning §11).

Offline and synthetic: every guard is a *mechanism* planning §11 names,
reproduced small enough to pin a number on.

* a known slope recovered by OLS, TLS, the bisector and the trimmed OLS on
  clean data, and by the ratio on a one-signed pool;
* **errors-in-variables** -- with noise on ``x``, OLS attenuates by
  ``var(X)/(var(X) + var(eta))`` to 1 %, while TLS (equal error variances,
  which both axes have here) does not; the bisector drifts *up*, so it is
  not a fix either;
* **leverage** -- one heavy-tailed point moves OLS by 0.14 and leaves the
  trimmed estimate unchanged to four decimals (M2's day-3 mechanism in
  miniature);
* the **ratio on a two-signed pool** reads 1.18 where OLS reads 1.000 and
  the split reads 1.0009 / 0.9991 (M1 task 6, flag 5), pinned;
* the **block bootstrap** is 15x wider than a pixel bootstrap on an
  autocorrelated field, and 3-hour blocks are wider again **only when the
  series is temporally correlated** -- on an uncorrelated one they are
  *narrower*, which is how the test shows the widening comes from the
  correlation and not from the coarser blocking;
* ``binned_conditional_mean`` returns the asymmetry put in;
* the ``validate`` equivalences, as exact equalities;
* ``slope_report`` divides by the baseline and never by 1.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

import stats as st
import validate as va

N_BOOT = 400                      # enough for a width comparison, quick enough for the suite


def smooth(rng, shape, sigma, scale=1.0):
    """An autocorrelated field: white noise through a Gaussian filter."""
    return ndimage.gaussian_filter(rng.normal(size=shape), sigma) * scale


# ---------------------------------------------------------------------------
# clean data: every estimator finds the slope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('b', [0.6, 0.9, 1.3])
def test_every_estimator_recovers_a_known_slope(b):
    rng = np.random.default_rng(0)
    x = rng.normal(size=200_000)
    y = b * x + rng.normal(scale=0.05, size=200_000)
    for name, fn in st.ESTIMATORS.items():
        assert fn(x, y) == pytest.approx(b, rel=0.01), name
    # the ratio is reliable on a one-signed pool, which is the point of the split
    pos = x > 0
    assert st.ratio_estimator(x[pos], y[pos], split_sign=False)['all'] == pytest.approx(b, rel=0.01)


def test_nan_cells_are_dropped_not_propagated():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = 0.9 * x + rng.normal(scale=0.05, size=5000)
    xn, yn = x.copy(), y.copy()
    xn[::97] = np.nan
    yn[5::89] = np.nan
    assert np.isfinite(st.slope_ols(xn, yn))
    assert st.slope_ols(xn, yn) == pytest.approx(0.9, rel=0.02)
    ok = np.isfinite(xn) & np.isfinite(yn)
    assert st.slope_ols(xn, yn) == st.slope_ols(x[ok], y[ok])


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match='shapes'):
        st.slope_ols(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError, match='labels shape'):
        st.block_bootstrap(np.zeros(10), np.zeros(10), np.zeros(9))


# ---------------------------------------------------------------------------
# planning §11's four mechanisms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('s', [0.3, 0.6])
def test_ols_attenuates_with_noise_on_x_and_tls_does_not(s):
    """Planning §11's first mechanism.  ``x = X + e1``, ``y = b X + e2`` with
    ``var(e1) = var(e2)``: OLS -> ``b var(X)/(var(X) + var(e1))`` (the
    attenuation), TLS -> ``b`` (it is consistent when the two error
    variances are equal, which is why the pair is quoted together).
    Stated tolerance: 1 % on the attenuation factor, 1 % on TLS."""
    rng = np.random.default_rng(0)
    n, b = 400_000, 0.9
    X = rng.normal(size=n)
    x = X + rng.normal(scale=s, size=n)
    y = b * X + rng.normal(scale=s, size=n)
    attenuated = b / (1.0 + s ** 2)                  # var(X) = 1
    assert st.slope_ols(x, y) == pytest.approx(attenuated, rel=0.01)
    assert st.slope_tls(x, y) == pytest.approx(b, rel=0.01)
    assert st.slope_ols(x, y) < st.slope_tls(x, y)
    # the bisector lies between the two OLS directions but is NOT a fix: it
    # overshoots, and by more as the noise grows
    assert st.slope_bisector(x, y) > b


def test_a_single_leverage_point_moves_ols_and_not_the_trimmed_estimate():
    """M2's day-3 mechanism: ``G`` is kurtotic, so a few pixels carry much of
    ``sum (x - xbar)^2``.  One point at ``|x| = 60`` sigma, off the line,
    moves OLS by 0.14; the trimmed estimate does not move at four decimals."""
    rng = np.random.default_rng(2)
    n = 20_000
    x = rng.normal(size=n)
    y = 0.9 * x + rng.normal(scale=0.2, size=n)
    ols0, trim0 = st.slope_ols(x, y), st.slope_trimmed(x, y)
    x2, y2 = x.copy(), y.copy()
    x2[0], y2[0] = 60.0, 0.0
    ols1, trim1 = st.slope_ols(x2, y2), st.slope_trimmed(x2, y2)
    assert ols0 - ols1 > 0.1                          # 0.897 -> 0.760
    assert trim1 == pytest.approx(trim0, abs=1e-4)    # 0.8980 either way
    assert st.slope_trimmed(x, y, pct=0.0) == st.slope_ols(x, y)
    with pytest.raises(ValueError, match='pct'):
        st.slope_trimmed(x, y, pct=100.0)


def test_the_ratio_is_ill_conditioned_on_a_two_signed_pool():
    """M1 task 6, flag 5, pinned: the denominator is a difference of large
    numbers, so ``all`` is wrong while the split is right."""
    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.normal(size=n)
    y = 1.0 * x + rng.normal(scale=0.1, size=n)
    r = st.ratio_estimator(x, y)
    assert r['all'] == pytest.approx(1.18, abs=0.02)      # wrong by 18 %
    assert r['pos'] == pytest.approx(1.0, abs=0.01)
    assert r['neg'] == pytest.approx(1.0, abs=0.01)
    assert st.slope_ols(x, y) == pytest.approx(1.0, abs=0.01)
    assert r['cancellation'] > 50                          # sum|x| / |sum x|
    assert 'ill-conditioned' in r['note']
    assert r['n_pos'] + r['n_neg'] == n
    assert 'pos' not in st.ratio_estimator(x, y, split_sign=False)


def test_block_bootstrap_is_much_wider_than_a_pixel_bootstrap():
    """Planning §11's effective sample size: pixels are not independent, so
    resampling them gives an interval that is far too small."""
    rng = np.random.default_rng(4)
    nj = ni = 256
    X = smooth(rng, (nj, ni), 8, 20.0)
    Y = 0.9 * X + smooth(rng, (nj, ni), 8, 6.0)
    blocks = st.block_bootstrap(X, Y, st.block_ids((nj, ni)), n=N_BOOT, seed=0)
    pixels = st.block_bootstrap(X, Y, np.arange(nj * ni).reshape(nj, ni), n=N_BOOT, seed=0)
    assert blocks['n_blocks'] == 64 and pixels['n_blocks'] == nj * ni
    assert (blocks['hi'] - blocks['lo']) > 8 * (pixels['hi'] - pixels['lo'])
    assert blocks['value'] == pixels['value'] == st.slope_ols(X, Y)


def test_three_hour_blocks_widen_the_ci_only_when_the_series_is_correlated():
    """M2 task 3 found chi^2/dof 1.8 for the pair-to-pair scatter against the
    one-hour interval.  Deepening the block to three hours recovers that
    **only if the hours really are correlated**: on a series whose per-pair
    offset alternates every hour the 3-hour block is *narrower* (fewer,
    larger units average more within each), so the widening below is the
    correlation and not the coarser blocking."""
    widths = {}
    for corr in (True, False):
        rng = np.random.default_rng(5)
        n = 64
        xs, ys, lab1, lab3 = [], [], [], []
        for p in range(12):
            group = (p // 3) if corr else p            # correlated in 3-hour groups, or not
            off = 0.12 if group % 2 == 0 else -0.12
            Xp = smooth(rng, (n, n), 4, 20.0)
            Yp = (0.9 + off) * Xp + smooth(rng, (n, n), 4, 3.0)
            xs.append(Xp.ravel())
            ys.append(Yp.ravel())
            lab1.append(st.space_time_block_ids((n, n), p, hours=1).ravel())
            lab3.append(st.space_time_block_ids((n, n), p, hours=3).ravel())
        x, y = np.concatenate(xs), np.concatenate(ys)
        a = st.block_bootstrap(x, y, np.concatenate(lab1), n=N_BOOT, seed=0)
        c = st.block_bootstrap(x, y, np.concatenate(lab3), n=N_BOOT, seed=0)
        assert c['n_blocks'] * 3 == a['n_blocks']
        widths[corr] = (c['hi'] - c['lo']) / (a['hi'] - a['lo'])
    assert widths[True] > 1.3                          # measured 1.58x
    assert widths[False] < 1.0                         # measured 0.63x
    assert widths[True] > 2 * widths[False]


def test_space_time_block_ids_group_by_hours():
    ids1 = [st.space_time_block_ids((64, 64), p, hours=1) for p in range(4)]
    ids3 = [st.space_time_block_ids((64, 64), p, hours=3) for p in range(4)]
    assert len({i.flat[0] for i in ids1}) == 4          # every pair its own blocks
    assert np.array_equal(ids3[0], ids3[1]) and np.array_equal(ids3[1], ids3[2])
    assert not np.array_equal(ids3[2], ids3[3])        # the fourth pair starts a new group
    with pytest.raises(ValueError, match='hours'):
        st.space_time_block_ids((8, 8), 0, hours=0)


def test_block_ids_match_validate_and_do_not_collide_across_pairs():
    shape = (128, 96)
    assert np.array_equal(st.block_ids(shape), va._block_ids(shape))
    assert np.array_equal(st.block_ids(shape, offset=7), va._block_ids(shape, offset=7))
    a, b = (st.space_time_block_ids(shape, p) for p in (0, 1))
    assert not set(np.unique(a)) & set(np.unique(b))


# ---------------------------------------------------------------------------
# binned conditional means
# ---------------------------------------------------------------------------
def test_binned_conditional_mean_returns_the_asymmetry_put_in():
    """Different slopes either side of zero -- exactly the diabatic asymmetry
    planning §11 says a single slope averages away."""
    rng = np.random.default_rng(6)
    n = 200_000
    x = rng.normal(size=n)
    y = np.where(x > 0, 1.2 * x, 0.4 * x) + rng.normal(scale=0.02, size=n)
    df = st.binned_conditional_mean(x, y, bins=8)
    assert isinstance(df, pd.DataFrame) and set(df['sign']) == {'pos', 'neg'}
    for sign, slope in (('pos', 1.2), ('neg', 0.4)):
        d = df[df['sign'] == sign]
        assert len(d) == 8
        fit = np.polyfit(d['x_mean'], d['y_mean'], 1)[0]
        assert fit == pytest.approx(slope, rel=0.02), sign
    # one slope over the whole pool is neither
    pooled = st.slope_ols(x, y)
    assert 0.4 < pooled < 1.2
    assert not df['se_iid'].isna().all()


def test_binned_conditional_mean_band_needs_labels():
    rng = np.random.default_rng(7)
    nj = ni = 128
    X = smooth(rng, (nj, ni), 6, 10.0)
    Y = 0.9 * X + smooth(rng, (nj, ni), 6, 2.0)
    plain = st.binned_conditional_mean(X, Y, bins=5)
    assert plain['lo'].isna().all() and (plain['n_blocks'] == 0).all()
    assert 'UNDERSTATES' in plain.attrs['se_note']
    banded = st.binned_conditional_mean(X, Y, bins=5, labels=st.block_ids((nj, ni)), n_boot=100)
    assert banded['lo'].notna().all() and (banded['n_blocks'] > 1).all()
    assert (banded['lo'] <= banded['y_mean']).all() and (banded['y_mean'] <= banded['hi']).all()
    # the block band is wider than the iid error bar it replaces
    assert ((banded['hi'] - banded['lo']) > 2 * banded['se_iid']).mean() > 0.5


def test_binned_conditional_mean_accepts_explicit_edges_and_no_split():
    rng = np.random.default_rng(8)
    x = rng.normal(size=10_000)
    y = 0.9 * x
    df = st.binned_conditional_mean(x, y, bins=[-4, -1, 0, 1, 4], split_sign=False)
    assert set(df['sign']) == {'all'} and len(df) == 4


# ---------------------------------------------------------------------------
# equivalence with validate -- reuse, not a fork
# ---------------------------------------------------------------------------
def test_slope_estimators_are_identical_to_validates():
    """Bit-identical, not merely close: ``stats`` writes the moments exactly
    as ``validate.slope_estimators`` does, so a drift between the two would
    be a real fork rather than a round-off difference."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=20_000) * 1e-18
    y = 0.9 * x + rng.normal(size=20_000) * 2e-19
    e = va.slope_estimators(x, y)
    assert st.slope_ols(x, y) == e['ols']
    assert st.slope_tls(x, y) == e['orthogonal']
    assert st.ratio_estimator(x, y, split_sign=False)['all'] == e['ratio']


def test_block_bootstrap_reproduces_validates_ci():
    """The same multinomial draw for the same seed, so the two agree to
    float round-off (1e-12 relative), not only within bootstrap noise."""
    rng = np.random.default_rng(10)
    nj = ni = 128
    X = smooth(rng, (nj, ni), 6, 1e-18)
    Y = 0.9 * X + smooth(rng, (nj, ni), 6, 2e-19)
    blk = st.block_ids((nj, ni))
    mine = st.block_bootstrap(X, Y, blk, st.slope_ols, n=N_BOOT, seed=0)
    theirs = va.block_bootstrap_ols(X.ravel(), Y.ravel(), blk.ravel(), n_boot=N_BOOT, seed=0)
    assert mine['n_blocks'] == theirs['n_blocks']
    assert mine['ci'][0] == pytest.approx(theirs['ci'][0], rel=1e-12)
    assert mine['ci'][1] == pytest.approx(theirs['ci'][1], rel=1e-12)
    assert mine['se'] == pytest.approx(theirs['se'], rel=1e-12)


def test_the_two_bootstrap_paths_agree():
    """The sufficient-statistics path and the general resampling path are
    two implementations of the same draw, so for a moment estimator they
    must give the *same* replicates -- not merely compatible intervals.
    Checked by stripping the ``moment_form`` off a copy of ``slope_ols``."""
    rng = np.random.default_rng(14)
    nj = ni = 96
    X = smooth(rng, (nj, ni), 5, 10.0)
    Y = 0.9 * X + smooth(rng, (nj, ni), 5, 2.0)
    blk = st.block_ids((nj, ni))

    def ols_no_form(a, b):                       # same arithmetic, no fast path
        return st.slope_ols(a, b)

    fast = st.block_bootstrap(X, Y, blk, st.slope_ols, n=200, seed=0)
    slow = st.block_bootstrap(X, Y, blk, ols_no_form, n=200, seed=0)
    assert fast['path'] == 'sufficient statistics' and slow['path'] == 'resampled sample'
    assert fast['ci'][0] == pytest.approx(slow['ci'][0], rel=1e-10)
    assert fast['ci'][1] == pytest.approx(slow['ci'][1], rel=1e-10)
    assert fast['se'] == pytest.approx(slow['se'], rel=1e-10)


def test_trimmed_uses_the_general_path():
    """``slope_trimmed``'s threshold depends on the resample, so it cannot be
    a function of the per-block sums -- it must not acquire a fast path by
    accident."""
    rng = np.random.default_rng(15)
    x = rng.normal(size=8192)
    y = 0.9 * x + rng.normal(scale=0.1, size=8192)
    lab = np.repeat(np.arange(64), 128)
    assert getattr(st.slope_trimmed, 'moment_form', None) is None
    r = st.block_bootstrap(x, y, lab, st.slope_trimmed, n=50, seed=0)
    assert r['path'] == 'resampled sample' and np.isfinite(r['se'])


def test_feature_bootstrap_is_the_same_call():
    rng = np.random.default_rng(11)
    x = rng.normal(size=4096)
    y = 0.9 * x + rng.normal(scale=0.1, size=4096)
    lab = np.repeat(np.arange(32), 128)
    a = st.block_bootstrap(x, y, lab, n=100, seed=0)
    b = st.feature_bootstrap(x, y, lab, n=100, seed=0)
    assert a['ci'] == b['ci'] and a['se'] == b['se']


def test_bootstrap_with_one_block_reports_no_interval():
    x = np.arange(100.0)
    r = st.block_bootstrap(x, 0.9 * x, np.zeros(100), n=10, seed=0)
    assert np.isnan(r['lo']) and r['n_boot'] == 0 and 'fewer than two blocks' in r['note']
    assert r['value'] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# the report: never against 1
# ---------------------------------------------------------------------------
def test_slope_report_divides_by_the_baseline_never_by_one():
    rng = np.random.default_rng(12)
    nj = ni = 128
    X = smooth(rng, (nj, ni), 6, 10.0)
    Y = 0.9 * X + smooth(rng, (nj, ni), 6, 2.0)
    rep = st.slope_report(X, Y, st.block_ids((nj, ni)), n_boot=N_BOOT)
    assert rep['baseline'] == st.BASELINE == 0.981
    assert set(rep['estimates']) == set(st.ESTIMATORS)
    for name, e in rep['estimates'].items():
        r = rep['relative'][name]
        assert r['value'] == pytest.approx(e['value'] / st.BASELINE, rel=1e-12)
        assert r['value'] != e['value']                # i.e. not divided by 1
        assert r['ci'] == pytest.approx([e['lo'] / st.BASELINE, e['hi'] / st.BASELINE], rel=1e-12)
    assert rep['baseline_ci'] == list(st.BASELINE_CI)
    assert rep['v3b_band'] == list(st.V3B_BAND)        # systematics carried, not folded in
    assert rep['temporal']['sd'] == st.TEMPORAL[1]
    assert 'ratio' in rep and 'pos' in rep['ratio']
    txt = st.format_report(rep)
    assert 'baseline' in txt and 'ill-conditioned' in txt and 'systematics' in txt


def test_slope_report_bootstraps_the_general_path_less_deeply():
    """``slope_trimmed`` has no ``moment_form``, so on a large pool it would
    dominate the report's cost; it gets ``n_boot_general`` replicates and the
    count is recorded so the coarser interval is visible."""
    rng = np.random.default_rng(16)
    nj = ni = 96
    X = smooth(rng, (nj, ni), 5, 10.0)
    Y = 0.9 * X + smooth(rng, (nj, ni), 5, 2.0)
    rep = st.slope_report(X, Y, st.block_ids((nj, ni)), n_boot=300, n_boot_general=50)
    assert rep['estimates']['ols']['n_boot'] == 300
    assert rep['estimates']['tls']['n_boot'] == 300
    assert rep['estimates']['trimmed']['n_boot'] == 50
    assert rep['estimates']['trimmed']['path'] == 'resampled sample'
    assert rep['estimates']['ols']['path'] == 'sufficient statistics'


def test_slope_report_verdict_is_against_the_baseline_interval():
    """A slope of 0.70 is below the baseline; 0.981 itself is consistent
    with it.  Both would be "below 1", which is the mistake the rule
    exists to prevent."""
    rng = np.random.default_rng(13)
    nj = ni = 128
    X = smooth(rng, (nj, ni), 5, 10.0)
    for slope, expect in ((0.70, 'below'), (st.BASELINE, 'consistent'), (1.20, 'above')):
        Y = slope * X + smooth(rng, (nj, ni), 5, 0.4)
        v = st.slope_report(X, Y, st.block_ids((nj, ni)), n_boot=N_BOOT)['verdict']
        assert v[f'{expect}_baseline' if expect != 'consistent' else 'consistent'] is True, slope
    assert 'never below 1' in v['rule']


def test_slope_report_refuses_a_meaningless_baseline():
    x = np.arange(100.0)
    y = 0.9 * x
    lab = np.repeat(np.arange(10), 10)
    for bad in (0.0, None, np.nan):
        with pytest.raises(ValueError, match='baseline'):
            st.slope_report(x, y, lab, baseline=bad, n_boot=10)
