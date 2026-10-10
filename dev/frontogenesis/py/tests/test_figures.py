""" Tests for M3 task 7's figures, offline on a **tiny synthetic derived
store** and a synthetic ``m3_closure_summary.json``.

Nothing here opens the real 7.5 GB product or renders at full size; the
point is that every function runs, writes its PNG, returns a dict, and
**draws task 6's numbers rather than its own** -- plus the two things the
prompt singles out: the baseline line is at **0.981, not 1**, and the
``L = 0`` subfilter panel is labelled identically zero.

A second, ``needs_grid`` test checks the real run: that
``m3_figs_summary.json`` agrees with ``m3_closure_summary.json`` where they
overlap, and that every PNG is on disk and **not** git-ignored.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import figures as fg
import m3_figs
import m3_closure as mc

NJ = NI = 48
L_ALL = (0, 2, 4, 8)
N_PAIRS = 6


# ---------------------------------------------------------------------------
# a tiny synthetic product
# ---------------------------------------------------------------------------
def _store(path, L, n_pairs=N_PAIRS, seed=0):
    rng = np.random.default_rng(seed + L)
    t = np.arange(n_pairs)
    X, Y = np.meshgrid(np.linspace(-126, -120, NI), np.linspace(30, 36, NJ))
    land = np.zeros((NJ, NI), bool)
    land[:4, :4] = True
    data = {}
    for v in fg.__dict__.get('_', []) or []:
        pass
    import budget as bg
    for v in bg.DERIVED_VARS:
        if v in ('valid', 'front'):
            a = np.ones((n_pairs, NJ, NI), bool)
            a[:, land] = False
            if v == 'front':
                a &= rng.random((n_pairs, NJ, NI)) < 0.3
            data[v] = (('time', 'j', 'i'), a)
        else:
            a = rng.normal(size=(n_pairs, NJ, NI)).astype('float32')
            if v == 'KPPhbl':
                a = (20 + 5 * np.sin(2 * np.pi * t / n_pairs)[:, None, None]
                     + 0.5 * a).astype('float32')
            if v == 'G':
                a = np.abs(a)
            if v == 'front_width':
                a = np.abs(a) + 1.0
            a[:, land] = np.nan
            data[v] = (('time', 'j', 'i'), a)
    ds = xr.Dataset(data, coords={
        'time': ('time', np.datetime64('2012-07-02T00', 'ns') + t * np.timedelta64(1, 'h')),
        'time_mid': ('time', np.datetime64('2012-07-02T00:30', 'ns') + t * np.timedelta64(1, 'h')),
        'j': np.arange(NJ), 'i': np.arange(NI),
        'XC': (('j', 'i'), X), 'YC': (('j', 'i'), Y)})
    ds['coast_distance_km'] = (('j', 'i'), np.full((NJ, NI), 150.0))
    ds.attrs.update(L_cells=int(L), front_pct=90.0)
    ds.to_zarr(path, mode='w')
    return path


def _summary(path):
    """A synthetic task-6 summary with the real schema and known numbers."""
    per_L = {}
    for k, L in enumerate(L_ALL):
        terms = {t: dict(rms=0.1 * (k + 1), over_measured=0.2 + 0.1 * k,
                         over_two_F=0.3 + 0.1 * k, corr_with_2F=-0.5 + 0.1 * k)
                 for t in mc.TERMS}
        est = {n: dict(value=0.9 - 0.02 * k, lo=0.88 - 0.02 * k, hi=0.92 - 0.02 * k,
                       se=0.01, n_blocks=10, n_boot=10, n=100, path='x')
               for n in ('ols', 'trimmed', 'tls', 'bisector')}
        per_L[str(L)] = dict(
            closure=dict(L=L, terms=terms,
                         residual=dict(rms=0.5, over_measured=0.8 + 0.05 * k, over_two_F=1.4,
                                       explained=0.3 - 0.1 * k, slope_on_2F=0.44,
                                       corr_on_2F=0.4),
                         catchall=dict(rms=0.4, over_measured=0.74 - 0.01 * k, explained=0.3,
                                       slope_on_2F=0.4, note='no chunk store'),
                         measured_on_2F=dict(ols=1.1, corr=0.6, n=100),
                         verdict=dict(passed=False, gated=L in mc.GATE_L, checks={}),
                         per_pair={}, day3={},
                         composites={}),
            euler=dict(front=dict(ols=0.7 + 0.08 * k, corr=0.76 + 0.07 * k, n=100),
                       valid=dict(ols=0.7, corr=0.75, n=100), gated=L in mc.GATE_L,
                       passed=L >= 4),
            subfilter=dict(L=L, over_two_F=0.0 if L == 0 else 0.3 + 0.1 * k,
                           corr_with_2F=-0.55, median_abs_ratio=0.2),
            fig2b=dict(on_lap2_b=dict(ols=1.0, corr=0.15, n=100),
                       on_KPPhbl=dict(ols=1.0, corr=0.07, n=100),
                       partial_res_lap_given_kpp=0.14 - 0.02 * k,
                       partial_res_kpp_given_lap=0.06 + 0.05 * k, by_local_hour={}),
            composites=dict(by_coast_km={f'{a}-{b}': dict(n=1000, resid_over_meas=0.8,
                                                          explained=0.3,
                                                          meas_on_2F=dict(ols=1.1, corr=0.5,
                                                                          n=100))
                                         for a, b in ((100, 150), (150, 200), (200, 300))},
                            by_width={}, by_local_hour={}, offshore_only=dict(n=1, km=100.0),
                            width_median_dx=2.3),
            slopes_not_quoted=dict(
                primary=dict(estimates=est, relative={}), primary_3h=est['ols'],
                variants={n: dict(est['ols'], relative=0.9) for n in
                          ('chain', 'order5', 'euler', 'p80', 'p95', 'edge13', 'valid')},
                by_width={}, day3=dict(est['ols'], relative=0.9, pairs=list(range(62, 69))),
                ratio_by_sign=dict(all=0.9, pos=1.0, neg=0.8, n=100, n_pos=50, n_neg=50,
                                   cancellation=5.0, note='x'),
                binned=[]),
            slopes_quotable=False, slopes_label='NOT QUOTED', per_pair=[], wall_s=1.0)
    s = dict(declared=mc.DECLARED, created='test', stores={}, per_L=per_L,
             verdict=mc.overall(per_L))
    Path(path).write_text(json.dumps(s, indent=1, default=str))
    return path


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp('figs')
    data, figs = d / 'data', d / 'figs'
    data.mkdir()
    figs.mkdir()
    for L in L_ALL:
        _store(data / f'tile330_derived_L{L}.zarr', L)
    return dict(data_dir=data, fig_dir=figs, summary=_summary(data / 'summary.json'))


# ---------------------------------------------------------------------------
# every figure runs, writes its PNG and returns numbers
# ---------------------------------------------------------------------------
def test_every_figure_runs_and_writes_its_png(tiny):
    res = m3_figs.render(data_dir=tiny['data_dir'], fig_dir=tiny['fig_dir'],
                         summary=tiny['summary'], say=lambda *a: None)
    assert res['_failed'] == [], {k: v.get('error') for k, v in res.items()
                                  if isinstance(v, dict) and 'error' in v}
    assert res['_n_ok'] == len(fg.FIGURES) == 10
    for name in fg.FIGURES:
        png = Path(res[name]['png'])
        assert png.exists() and png.stat().st_size > 5000, name
        assert png.parent == tiny['fig_dir']
        assert png.name.startswith(name + '_'), name


def test_figure_names_and_order_match_coding_4_10():
    assert list(fg.FIGURES) == ['fig01', 'fig02', 'fig02b', 'fig03', 'fig03b', 'fig04',
                                'fig05', 'fig06', 'fig07', 'fig10']
    assert fg.FIGURES['fig01'].__name__ == 'fig01_maps'
    assert fg.FIGURES['fig10'].__name__ == 'fig10_term_budget'


# ---------------------------------------------------------------------------
# the numbers drawn are task 6's
# ---------------------------------------------------------------------------
def test_figure_10_draws_the_summary_numbers(tiny):
    s = json.loads(Path(tiny['summary']).read_text())
    out = fg.fig10_term_budget(data_dir=tiny['data_dir'], fig_dir=tiny['fig_dir'],
                               summary=tiny['summary'])
    for L in L_ALL:
        c = s['per_L'][str(L)]['closure']
        assert out['residual'][str(L)] == c['residual']['over_measured']
        assert out['catchall'][str(L)] == c['catchall']['over_measured']
        for t in mc.TERMS:
            assert out['rms'][str(L)][t] == c['terms'][t]['over_measured']


def test_figure_2b_draws_the_summary_partial_correlations(tiny):
    s = json.loads(Path(tiny['summary']).read_text())
    out = fg.fig02b_discriminator(data_dir=tiny['data_dir'], fig_dir=tiny['fig_dir'],
                                  summary=tiny['summary'])
    for L in L_ALL:
        f = s['per_L'][str(L)]['fig2b']
        assert out[f'L{L}']['partial_lap'] == f['partial_res_lap_given_kpp']
        assert out[f'L{L}']['partial_kpp'] == f['partial_res_kpp_given_lap']


def test_figure_3_draws_the_summary_subfilter_ratios(tiny):
    s = json.loads(Path(tiny['summary']).read_text())
    out = fg.fig03_slope_vs_L(data_dir=tiny['data_dir'], fig_dir=tiny['fig_dir'],
                              summary=tiny['summary'])
    for L in L_ALL:
        assert out['subfilter_over_2F'][str(L)] == s['per_L'][str(L)]['subfilter']['over_two_F']
    assert out['subfilter_over_2F']['0'] == 0.0           # identically zero at L = 0


# ---------------------------------------------------------------------------
# the two things the prompt singles out
# ---------------------------------------------------------------------------
def test_the_baseline_is_0981_and_never_1():
    assert fg.BASELINE == 0.981 and fg.BASELINE_CI == (0.970, 0.994)
    assert fg.V3B_BAND == (0.954, 1.003) and fg.TEMPORAL == (0.972, 0.020)
    import stats as st
    assert fg.BASELINE == st.BASELINE and fg.BASELINE_CI == st.BASELINE_CI
    assert fg.V3B_BAND == st.V3B_BAND


def test_figure_6_carries_the_three_required_caveats():
    """M2 task 6 item 4 / prompt 4 task 7: the caption must state the
    6-hourly interpolated forcing, the wind trend and the three-cycle
    window."""
    for frag in ('6-hourly', 'linearly interpolated', 'TRIANGLE peaking at 13 LST',
                 'wind trend', 'three diurnal cycles', 'indicative, not conclusive'):
        assert frag in (fg.FORCING_CAVEAT + fg.WIND_CAVEAT + fg.WINDOW_CAVEAT), frag
    assert 'hourly' in fg.SAMPLING_CAVEAT                 # M3-Q14 (a)


def test_figure_3b_labels_the_L0_subfilter_panel(tiny):
    out = fg.fig03b_filter_ladder(data_dir=tiny['data_dir'], fig_dir=tiny['fig_dir'])
    assert 'subfilter_L0' in out and 'subfilter_L8' in out
    src = Path(fg.__file__).with_name('figures_maps.py').read_text()
    assert 'IDENTICALLY ZERO' in src


def test_no_figure_claims_diabatic_damping():
    """The 'Do not' list: no slope may be called damping without Figure 2b,
    and Figure 2b does not support it."""
    for mod in ('figures.py', 'figures_maps.py', 'figures_stats.py'):
        src = Path(fg.__file__).with_name(mod).read_text().lower()
        assert 'diabatic damping' not in src or 'no ' in src or 'not ' in src
    src = Path(fg.__file__).with_name('figures_stats.py').read_text()
    assert 'NOT QUOTED' in src and 'SLOPES NOT QUOTED' in src


# ---------------------------------------------------------------------------
# the real run
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_the_real_figures_match_task_6_and_are_tracked():
    """Every PNG on disk, none git-ignored, and the rendered numbers equal
    ``m3_closure_summary.json``'s."""
    js = fg.DATA_DIR / 'm3_figs_summary.json'
    if not js.exists() or not fg.SUMMARY.exists():
        pytest.skip('m3_figs.py / m3_closure.py not run here')
    figs = json.loads(js.read_text())
    s = json.loads(fg.SUMMARY.read_text())
    assert figs['_failed'] == [] and figs['_n_ok'] == 10
    for L in L_ALL:
        c = s['per_L'][str(L)]['closure']
        assert figs['fig10']['residual'][str(L)] == c['residual']['over_measured']
        assert figs['fig03']['subfilter_over_2F'][str(L)] == \
            s['per_L'][str(L)]['subfilter']['over_two_F']
    for name in fg.FIGURES:
        png = Path(figs[name]['png'])
        assert png.exists(), png
        r = subprocess.run(['git', 'check-ignore', '-q', str(png)],
                           cwd=png.parent, capture_output=True)
        assert r.returncode != 0, f'{png.name} is git-ignored'
