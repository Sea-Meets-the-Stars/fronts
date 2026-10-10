""" M3 task 6: the closure gate.

Reads the four derived stores (§3.4) and the per-pair reports `m3_run` cached,
and answers criteria 1-4 with a **verdict per filter scale**, against
tolerances that were fixed on 2026-10-07 (M3-Q1/Q2/Q4/Q5/Q9) and are the
module constants below.  Writes `data/m3_closure_summary.json` and the
working figure `figs/m3_closure.png`; task 7 makes the publication figures
from the JSON.

**Nothing here is tuned.**  The tolerances are inputs, not outputs; if the
budget does not close that is the result, written up against planning §12
with the limiting factor named, and no efficiency is quoted (the "Do not"
list's first item).  Criterion 4's slopes are computed either way, but they
are reported under ``slopes_not_quoted`` wherever criterion 1 fails at that
``L`` -- present so the next session need not recompute them, labelled so
they cannot be mistaken for a result.

Sections, following the prompt:

(a) closure -- the five-term table, the residual ratio and its distribution
    over the 71 pairs, the explained fraction, the residual's slope on
    ``2F``, **with and without the chunk terms**, the composites by local
    solar hour and by coast distance, and the verdict;
(b) semi-Lagrangian vs Eulerian;
(c) the filter sweep's interpretability -- ``subfilter/2F`` and its
    correlation with ``2F`` against ``L``;
(d) the slopes, with every estimator, both forms, both orders, both edge
    masks, three front percentiles, per width bin, relative to the M1
    baseline;
(e) Figure 2b's data -- the residual against ``lap2_b`` and ``KPPhbl``, with
    partial correlations, which is what separates implicit numerical
    diffusion from air-sea forcing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget as bg                                            # noqa: E402
import inputs as inp                                           # noqa: E402
import m3_run as mr                                            # noqa: E402
import masking as mk                                           # noqa: E402
import osn_tiles as ot                                         # noqa: E402
import stats as st                                             # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

FIG_DIR = HERE.parent / 'figs'
OUT_JSON = DATA_DIR / 'm3_closure_summary.json'
OUT_PNG = FIG_DIR / 'm3_closure.png'

# ---------------------------------------------------------------------------
# THE PRE-DECLARATION.  Decided 2026-10-07 (M3-Q1/Q2/Q4/Q5/Q9), before the
# sweep existed.  Nothing below changes after the numbers.
# ---------------------------------------------------------------------------
TOL_RESID_RATIO = 0.50          # M3-Q1 (a): rms(residual)/rms(measured) on front & valid
TOL_EXPLAINED = 0.75            # M3-Q1 (a), equivalently
TOL_RESID_SLOPE = 0.10          # M3-Q1 (b): |OLS slope of residual on 2F|
TOL_EULER_SLOPE = (0.85, 1.15)  # M3-Q2
TOL_EULER_CORR = 0.90           # M3-Q2
GATE_L = (2, 4, 8)              # M3-Q9 (a): criterion 1 is judged at L >= 2
REPORT_L = (0, 2, 4, 8)         # L = 0 reported and interpreted, not gated
FRONT_PCT = 90.0                # M3-Q4 (a): the gate's pool
FRONT_PCT_SENS = (80.0, 95.0)   # recomputed from the stored G on valid
EDGE_CELLS = (7, 13)            # M2 task 6 item 6; 13 via masking.analysis_mask
WIDTH_EDGES = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, np.inf)           # M3-Q5 (a), in dx
WIDTH_LABELS = ('<=1', '1-1.5', '1.5-2', '2-3', '3-4', '>4')
DAY3_PAIRS = tuple(range(62, 69))                              # 07-04 14-20 UTC (M2 task 3)
OFFSHORE_KM = 100.0                                            # planning §5.6
#: V3b's advection-numerics shortfall by front width (discrete form, M1 6b):
#: subtract before attributing anything on the sharpest fronts to diffusion
WIDTH_SHORTFALL = {'<=1': -0.11, '1-1.5': -0.04, '1.5-2': -0.02}
#: V4's bar on the resolved damping, quoted with the width (M1)
V4_BAR_PCT_PER_HOUR = (0.28, 1.0)
DECLARED = dict(
    source='frontogenesis_prompt_4.md ## Q&A, answered by JXP 2026-10-07; task 6 pre-declaration',
    criterion1=dict(resid_ratio_max=TOL_RESID_RATIO, explained_min=TOL_EXPLAINED,
                    resid_slope_tol=TOL_RESID_SLOPE, pool='front & valid', judged_at=list(GATE_L)),
    criterion2=dict(slope=list(TOL_EULER_SLOPE), corr_min=TOL_EULER_CORR, pool='front',
                    judged_at=list(GATE_L)),
    front_pct=FRONT_PCT, front_pct_sensitivities=list(FRONT_PCT_SENS),
    edge_cells=list(EDGE_CELLS), width_bins=list(WIDTH_LABELS),
    day3_pairs=list(DAY3_PAIRS), baseline=st.BASELINE, baseline_ci=list(st.BASELINE_CI),
    v3b_band=list(st.V3B_BAND), temporal=list(st.TEMPORAL),
    note='fixed before m3_closure.py was run; see the log entry of 2026-10-10 task 6')

TERMS = bg.TERMS                               # two_F, subfilter, vertical, surface_flux
N_BOOT = 1000


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _rms(a):
    a = np.asarray(a, dtype='float64')
    if a.size == 0 or not np.any(np.isfinite(a)):
        return float('nan')
    return float(np.sqrt(np.nanmean(a ** 2)))


def _fit(x, y):
    """OLS slope, correlation and n over the finite cells."""
    x, y = np.asarray(x, 'float64').ravel(), np.asarray(y, 'float64').ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(ols=float('nan'), corr=float('nan'), n=int(ok.sum()))
    xs, ys = x[ok], y[ok]
    xm, ym = xs.mean(), ys.mean()
    sxx = float(np.sum((xs - xm) ** 2))
    syy = float(np.sum((ys - ym) ** 2))
    sxy = float(np.sum((xs - xm) * (ys - ym)))
    if sxx <= 0.0 or syy <= 0.0:                 # a constant axis: subfilter is identically 0
        return dict(ols=float('nan'), corr=float('nan'), n=int(ok.sum()),
                    note='x or y has zero variance (at L = 0 the subfilter term is identically 0)')
    return dict(ols=sxy / sxx, corr=sxy / np.sqrt(sxx * syy), n=int(ok.sum()))


def _explained(meas, res):
    with np.errstate(invalid='ignore'):
        return float(1.0 - np.nanvar(res) / np.nanvar(meas))


def partial_corr(a, b, c):
    """corr(a, b) with the linear part of ``c`` removed from both -- the
    statistic that separates "the residual tracks grad^4 b" from "the
    residual tracks the mixed layer" when ``lap2_b`` and ``KPPhbl`` are
    themselves correlated."""
    a, b, c = (np.asarray(v, 'float64').ravel() for v in (a, b, c))
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if ok.sum() < 4:
        return float('nan')
    a, b, c = a[ok], b[ok], c[ok]
    ra = a - np.polyval(np.polyfit(c, a, 1), c)
    rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return float(np.corrcoef(ra, rb)[0, 1])


def width_bin(w):
    """Index into :data:`WIDTH_LABELS` for a width in dx (NaN -> -1)."""
    w = np.asarray(w, 'float64')
    out = np.full(w.shape, -1, dtype='int8')
    ok = np.isfinite(w)
    # right=True so the declared labels read as they are written: '<=1' is
    # (0, 1], '1-1.5' is (1, 1.5], ... '>4' is (4, inf).  With the default
    # right=False a width of exactly 1.0 dx would land in the '1-1.5' bin.
    out[ok] = np.clip(np.digitize(w[ok], WIDTH_EDGES[1:-1], right=True),
                      0, len(WIDTH_LABELS) - 1)
    return out


# ---------------------------------------------------------------------------
# loading one L: stream the pairs, keep the front pool
# ---------------------------------------------------------------------------
POOL_VARS = ('two_F', 'two_F_chain', 'DGDt_semilag', 'DGDt_semilag_o5', 'DGDt_euler',
             'subfilter', 'vertical', 'surface_flux', 'residual', 'lap2_b', 'KPPhbl',
             'front_width', 'G', 'coast_distance_km')


def load_pool(L, edge13, n_analysis, pcts=(FRONT_PCT,) + FRONT_PCT_SENS, say=print):
    """One pass over a derived store.

    Returns ``(per_pair, pools)``.  ``per_pair`` is the pair-by-pair table
    (the five rms fractions, the ratios, the slopes, ``n_lost``, the local
    solar hour).  ``pools`` holds the **pooled front pixels** over all 71
    pairs, one entry per selection: the p90 gate pool, the p80 / p95
    sensitivities, the ``edge_cells = 13`` pool, and ``valid`` itself --
    each with every field of :data:`POOL_VARS` plus the block labels, the
    pair index and the local solar hour.
    """
    path = bg.derived_path(L)
    ds = xr.open_zarr(path)
    n = ds.sizes['time']
    sel_names = [f'p{p:g}' for p in pcts] + ['edge13_p90', 'valid']
    pools = {s: {k: [] for k in POOL_VARS + ('block', 'block3', 'pair', 'lst')}
             for s in sel_names}
    rows = []
    dist = np.asarray(ds['coast_distance_km'].values, 'float64')
    for p in range(n):
        h = ds.isel(time=p).load()
        valid = np.asarray(h['valid'].values, bool)
        front = np.asarray(h['front'].values, bool)
        G = np.asarray(h['G'].values, 'float64')
        arrays = {k: (dist if k == 'coast_distance_km'
                      else np.asarray(h[k].values, 'float64')) for k in POOL_VARS}
        lst = float(inp.local_solar_hour(h['time_mid'].values))
        blk = st.space_time_block_ids(valid.shape, p, hours=1)
        blk3 = st.space_time_block_ids(valid.shape, p, hours=3)
        sels = {}
        for pct in pcts:
            sels[f'p{pct:g}'] = (front if pct == FRONT_PCT else
                                 valid & (G >= np.percentile(G[valid], pct)))
        v13 = valid & edge13
        sels['edge13_p90'] = v13 & (G >= np.percentile(G[v13], FRONT_PCT))
        sels['valid'] = valid
        for name, m in sels.items():
            d = pools[name]
            for k in POOL_VARS:
                d[k].append(arrays[k][m])
            d['block'].append(blk[m])
            d['block3'].append(blk3[m])
            d['pair'].append(np.full(int(m.sum()), p, dtype='int16'))
            d['lst'].append(np.full(int(m.sum()), lst, dtype='float64'))
        # the per-pair row, on the gate's pool
        f = sels[f'p{FRONT_PCT:g}']
        meas, F2, res = arrays['DGDt_semilag'][f], arrays['two_F'][f], arrays['residual'][f]
        catch = res + arrays['vertical'][f] + arrays['surface_flux'][f]
        row = dict(pair=p, lst=lst, n_front=int(f.sum()), n_valid=int(valid.sum()),
                   n_lost=int(n_analysis - valid.sum()),
                   rms_measured=_rms(meas), rms_two_F=_rms(F2),
                   resid_over_meas=_rms(res) / _rms(meas),
                   catchall_over_meas=_rms(catch) / _rms(meas),
                   explained=_explained(meas, res),
                   explained_catchall=_explained(meas, catch),
                   resid_on_2F=_fit(F2, res)['ols'],
                   meas_on_2F=_fit(F2, meas)['ols'],
                   euler_slope=_fit(arrays['DGDt_semilag'][f], arrays['DGDt_euler'][f])['ols'],
                   euler_corr=_fit(arrays['DGDt_semilag'][f], arrays['DGDt_euler'][f])['corr'])
        for t in TERMS:
            row[f'{t}_over_meas'] = _rms(arrays[t][f]) / row['rms_measured']
            row[f'{t}_over_2F'] = _rms(arrays[t][f]) / row['rms_two_F']
        row['subfilter_corr_2F'] = _fit(F2, arrays['subfilter'][f])['corr']
        rows.append(row)
        if p % 20 == 0:
            say(f'  L={L} pair {p}/{n}')
    ds.close()
    for d in pools.values():
        for k in list(d):
            d[k] = np.concatenate(d[k]) if d[k] else np.array([])
    return rows, pools


# ---------------------------------------------------------------------------
# (a) closure
# ---------------------------------------------------------------------------
def closure(pool, rows, L):
    """The five-term table, the residual ratio, the catch-all comparison and
    the verdict against M3-Q1 -- pooled over the 71 pairs and summarised per
    pair."""
    meas, F2, res = pool['DGDt_semilag'], pool['two_F'], pool['residual']
    catch = res + pool['vertical'] + pool['surface_flux']
    r_meas, r_2F = _rms(meas), _rms(F2)
    out = dict(L=int(L), n=int(meas.size), n_pairs=len(rows),
               rms_measured=r_meas, rms_two_F=r_2F,
               terms={t: dict(rms=_rms(pool[t]), over_measured=_rms(pool[t]) / r_meas,
                              over_two_F=_rms(pool[t]) / r_2F,
                              corr_with_2F=_fit(F2, pool[t])['corr']) for t in TERMS},
               residual=dict(rms=_rms(res), over_measured=_rms(res) / r_meas,
                             over_two_F=_rms(res) / r_2F, explained=_explained(meas, res),
                             slope_on_2F=_fit(F2, res)['ols'], corr_on_2F=_fit(F2, res)['corr']),
               catchall=dict(rms=_rms(catch), over_measured=_rms(catch) / r_meas,
                             explained=_explained(meas, catch),
                             slope_on_2F=_fit(F2, catch)['ols'],
                             note='residual + vertical + surface_flux: what the residual would '
                                  'be with no chunk store (the Q13 comparison)'),
               measured_on_2F=_fit(F2, meas))
    # Does each term explain the residual it is supposed to?  The multiplier
    # that would MINIMISE the residual is +1 for a correct term, -1 for one
    # entering with the wrong sign, and ~0 for one that is simply
    # uncorrelated with what is left -- which is a different finding, and the
    # one the Q13 decision turns on.
    out['term_contribution'] = {}
    for t in TERMS:
        rest = meas - sum(pool[u] for u in TERMS if u != t)      # the residual without this term
        k = _fit(pool[t], rest)
        out['term_contribution'][t] = dict(
            optimal_multiplier=k['ols'], corr_with_rest=k['corr'],
            rms_over_measured=_rms(pool[t]) / r_meas,
            rms_without=_rms(rest) / r_meas,
            rms_subtracting=_rms(rest - pool[t]) / r_meas,
            rms_adding=_rms(rest + pool[t]) / r_meas,
            rms_at_optimum=_rms(rest - k['ols'] * pool[t]) / r_meas,
            helps=bool(_rms(rest - pool[t]) < _rms(rest)),
            diagnosis=('identically zero at this L (planning §5.4): not a term, a label'
                       if not np.isfinite(k['ols']) else
                       'sign error: the optimum is near -1' if k['ols'] < -0.5 else
                       'uncorrelated with the residual it should explain: subtracting it at '
                       'unit coefficient ADDS its variance' if abs(k['ols']) < 0.25 else
                       'explains part of the residual'))
    # What limits `measured`?  Three estimates of the same quantity: the
    # order-3 and order-5 semi-Lagrangian steps (interpolation) and the
    # Eulerian split (time sampling).  If they disagree, no term computed
    # from hourly snapshots can explain the difference.
    out['measured_coherence'] = dict(
        semilag_o3_vs_o5=_fit(pool['DGDt_semilag'], pool['DGDt_semilag_o5']),
        semilag_vs_euler=_fit(pool['DGDt_semilag'], pool['DGDt_euler']),
        rms_o5_minus_o3=_rms(pool['DGDt_semilag_o5'] - pool['DGDt_semilag']) / r_meas,
        rms_euler_minus_semilag=_rms(pool['DGDt_euler'] - pool['DGDt_semilag']) / r_meas,
        note='the order-5 difference is interpolation; the Eulerian difference is time '
             'sampling. Either one comparable to the residual bounds what any budget term '
             'could explain.')
    # The surface-flux term divides the flux by the 1 m top cell (drF[0]).
    # If the optimum multiplier matches drF / KPPhbl, the hourly-mean
    # tendency is set by the KPP boundary layer, not the top cell -- a
    # question for JXP (it is M3-Q10..Q12's neighbourhood, still open), NOT a
    # correction applied here.  `would_closure_change` is the point: even the
    # best-scaled version of the term leaves the residual where it was,
    # because the term is orthogonal to it.
    kpp = pool['KPPhbl']
    sf = out['term_contribution']['surface_flux']
    out['depth_scale_check'] = dict(
        optimal_multiplier=sf['optimal_multiplier'],
        drF_over_median_KPPhbl=float(1.0 / np.nanmedian(kpp)),
        median_KPPhbl_m=float(np.nanmedian(kpp)),
        implied_depth_m=(float(1.0 / sf['optimal_multiplier'])
                         if sf['optimal_multiplier'] not in (0, None)
                         and np.isfinite(sf['optimal_multiplier']) else None),
        rms_at_optimum=sf['rms_at_optimum'], rms_without=sf['rms_without'],
        would_closure_change=bool(sf['rms_at_optimum'] <= TOL_RESID_RATIO),
        hypothesis='surface_flux divides by drF[0] = 1 m; if the flux is mixed over the KPP '
                   'boundary layer within the hour the effective divisor is KPPhbl (~22 m). '
                   'REPORTED, NOT APPLIED -- and it cannot change the verdict, because the term '
                   'is near-orthogonal to the residual (see corr_with_rest): at the optimum the '
                   'residual is unchanged in the fourth decimal.')
    out['chunk_terms_help'] = dict(
        rms_ratio=out['residual']['over_measured'] / out['catchall']['over_measured'],
        explained_gain=out['residual']['explained'] - out['catchall']['explained'],
        note='rms_ratio < 1 means subtracting the measured vertical and surface-flux terms '
             'REDUCES the residual; > 1 means it makes it worse')
    per = {k: _dist([r[k] for r in rows]) for k in
           ('resid_over_meas', 'catchall_over_meas', 'explained', 'resid_on_2F', 'meas_on_2F')}
    out['per_pair'] = per
    d3 = [r for r in rows if r['pair'] in DAY3_PAIRS]
    out['day3'] = {k: _dist([r[k] for r in d3]) for k in ('resid_over_meas', 'explained',
                                                          'meas_on_2F')}
    out['verdict'] = _verdict1(out, L)
    return out


def _dist(v):
    a = np.asarray([x for x in v], dtype='float64')
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(n=0)
    return dict(n=int(a.size), median=float(np.median(a)), mean=float(a.mean()),
                sd=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                min=float(a.min()), max=float(a.max()),
                p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))


def _verdict1(c, L):
    r = c['residual']
    checks = dict(resid_ratio=dict(value=r['over_measured'], tol=TOL_RESID_RATIO,
                                   passed=bool(r['over_measured'] <= TOL_RESID_RATIO)),
                  explained=dict(value=r['explained'], tol=TOL_EXPLAINED,
                                 passed=bool(r['explained'] >= TOL_EXPLAINED)),
                  resid_slope=dict(value=r['slope_on_2F'], tol=TOL_RESID_SLOPE,
                                   passed=bool(abs(r['slope_on_2F']) <= TOL_RESID_SLOPE)))
    gated = int(L) in GATE_L
    return dict(checks=checks, gated=gated,
                passed=bool(all(x['passed'] for x in checks.values())),
                note=('judged (M3-Q9: criterion 1 at L >= 2)' if gated else
                      'L = 0: reported and interpreted, NOT gated; the explicit subfilter term '
                      'is identically zero here, so the residual is the numerics-plus-KPP '
                      'estimate Figure 2b is about'))


# ---------------------------------------------------------------------------
# (b), (c), (e)
# ---------------------------------------------------------------------------
def euler(pool_front, pool_valid, L):
    f = _fit(pool_front['DGDt_semilag'], pool_front['DGDt_euler'])
    v = _fit(pool_valid['DGDt_semilag'], pool_valid['DGDt_euler'])
    gated = int(L) in GATE_L
    ok = bool(TOL_EULER_SLOPE[0] <= f['ols'] <= TOL_EULER_SLOPE[1] and f['corr'] >= TOL_EULER_CORR)
    return dict(L=int(L), front=f, valid=v, gated=gated, passed=ok,
                note=('judged (M3-Q2 at L >= 2)' if gated else
                      'L = 0 reported, not gated: the Eulerian split is two nearly cancelling '
                      'terms at the grid scale (planning §5.3); M1 task 3 measured corr 0.74 / '
                      'slope 0.73 there'))


def subfilter_sweep(pool, L):
    return dict(L=int(L), over_two_F=_rms(pool['subfilter']) / _rms(pool['two_F']),
                corr_with_2F=_fit(pool['two_F'], pool['subfilter'])['corr'],
                median_abs_ratio=float(np.nanmedian(np.abs(pool['subfilter'] / pool['two_F']))))


def fig2b(pool):
    """(e) The residual against ``lap2_b`` (numerical diffusion) and
    ``KPPhbl`` (air-sea forcing), with partial correlations and the local-hour
    composite.  **No "diabatic damping" may be claimed without this.**"""
    res, lap, kpp, lst = pool['residual'], pool['lap2_b'], pool['KPPhbl'], pool['lst']
    out = dict(on_lap2_b=_fit(lap, res), on_KPPhbl=_fit(kpp, res),
               lap2_b_vs_KPPhbl=_fit(kpp, lap),
               partial_res_lap_given_kpp=partial_corr(res, lap, kpp),
               partial_res_kpp_given_lap=partial_corr(res, kpp, lap))
    hrs = np.floor(lst).astype(int) % 24
    out['by_local_hour'] = {int(h): dict(n=int((hrs == h).sum()),
                                         rms_residual=_rms(res[hrs == h]),
                                         mean_KPPhbl=float(np.nanmean(kpp[hrs == h])))
                            for h in np.unique(hrs)}
    out['interpretation_rule'] = (
        'planning §12: if the residual tracks lap2_b rather than KPPhbl or the diurnal cycle, '
        'no diabatic signal can be isolated and the conclusion is methodological')
    return out


def composites(pool):
    """Residual and terms by local solar hour, by coast distance, and by
    front-width bin (M3-Q5), plus the offshore-restricted statistic
    (planning §5.6)."""
    res, meas = pool['residual'], pool['DGDt_semilag']
    out = dict()
    hrs = np.floor(pool['lst']).astype(int) % 24
    out['by_local_hour'] = {int(h): dict(
        n=int((hrs == h).sum()), resid_over_meas=_rms(res[hrs == h]) / _rms(meas[hrs == h]),
        **{f'{t}_over_meas': _rms(pool[t][hrs == h]) / _rms(meas[hrs == h]) for t in TERMS})
        for h in np.unique(hrs)}
    d = pool['coast_distance_km']
    edges = [0, 100, 150, 200, 300, np.inf]
    out['by_coast_km'] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        if m.sum() > 100:
            out['by_coast_km'][f'{a:g}-{b:g}'] = dict(
                n=int(m.sum()), resid_over_meas=_rms(res[m]) / _rms(meas[m]),
                explained=_explained(meas[m], res[m]), meas_on_2F=_fit(pool['two_F'][m], meas[m]))
    off = d >= OFFSHORE_KM
    out['offshore_only'] = dict(n=int(off.sum()), km=OFFSHORE_KM,
                                resid_over_meas=_rms(res[off]) / _rms(meas[off]),
                                explained=_explained(meas[off], res[off]),
                                note='planning §5.6: the primary statistic restricted to '
                                     '>= 100 km from the coast')
    wb = width_bin(pool['front_width'])
    out['by_width'] = {}
    for k, lab in enumerate(WIDTH_LABELS):
        m = wb == k
        if m.sum() > 100:
            out['by_width'][lab] = dict(
                n=int(m.sum()), resid_over_meas=_rms(res[m]) / _rms(meas[m]),
                meas_on_2F=_fit(pool['two_F'][m], meas[m]),
                shortfall_v3b=WIDTH_SHORTFALL.get(lab),
                note='subtract shortfall_v3b from the slope before attributing anything here '
                     'to diffusion (M1 6b)' if lab in WIDTH_SHORTFALL else None)
    out['width_median_dx'] = float(np.nanmedian(pool['front_width']))
    return out


# ---------------------------------------------------------------------------
# (d) the slopes
# ---------------------------------------------------------------------------
def slopes(pools, L, n_boot=N_BOOT, say=print):
    """Criterion 4's numbers: measured on ``2F``, every estimator, both
    forms, both orders, both edge masks, three percentiles, per width bin,
    **relative to the M1 baseline**.  Computed whatever the verdict; the
    caller decides whether they may be quoted."""
    out = dict(L=int(L), baseline=st.BASELINE, baseline_ci=list(st.BASELINE_CI),
               v3b_band=list(st.V3B_BAND), temporal=list(st.TEMPORAL))
    gate = pools[f'p{FRONT_PCT:g}']
    say(f'  L={L} slope_report on {gate["two_F"].size:,} pixels')
    out['primary'] = st.slope_report(gate['two_F'], gate['DGDt_semilag'], gate['block'],
                                     n_boot=n_boot)
    out['primary_3h'] = st.block_bootstrap(gate['two_F'], gate['DGDt_semilag'], gate['block3'],
                                           st.slope_ols, n=n_boot, seed=0)
    out['variants'] = {}
    for name, (xk, yk, sel) in {
            'chain': ('two_F_chain', 'DGDt_semilag', f'p{FRONT_PCT:g}'),
            'order5': ('two_F', 'DGDt_semilag_o5', f'p{FRONT_PCT:g}'),
            'euler': ('two_F', 'DGDt_euler', f'p{FRONT_PCT:g}'),
            'p80': ('two_F', 'DGDt_semilag', 'p80'),
            'p95': ('two_F', 'DGDt_semilag', 'p95'),
            'edge13': ('two_F', 'DGDt_semilag', 'edge13_p90'),
            'valid': ('two_F', 'DGDt_semilag', 'valid')}.items():
        p = pools[sel]
        b = st.block_bootstrap(p[xk], p[yk], p['block'], st.slope_ols, n=n_boot, seed=0)
        b['relative'] = b['value'] / st.BASELINE
        b['trimmed'] = st.slope_trimmed(p[xk], p[yk])
        b['tls'] = st.slope_tls(p[xk], p[yk])
        out['variants'][name] = b
    out['ratio_by_sign'] = st.ratio_estimator(gate['two_F'], gate['DGDt_semilag'])
    df = st.binned_conditional_mean(gate['two_F'], gate['DGDt_semilag'], bins=8,
                                    labels=gate['block'], n_boot=200)
    out['binned'] = df.to_dict(orient='records')
    wb = width_bin(gate['front_width'])
    out['by_width'] = {}
    for k, lab in enumerate(WIDTH_LABELS):
        m = wb == k
        if m.sum() > 1000:
            b = st.block_bootstrap(gate['two_F'][m], gate['DGDt_semilag'][m], gate['block'][m],
                                   st.slope_ols, n=200, seed=0)
            sh = WIDTH_SHORTFALL.get(lab)
            b.update(relative=b['value'] / st.BASELINE, shortfall_v3b=sh,
                     corrected=(b['value'] - sh * st.BASELINE) / st.BASELINE if sh else None,
                     v4_bar_pct_per_hour=list(V4_BAR_PCT_PER_HOUR))
            out['by_width'][lab] = b
    d3 = np.isin(gate['pair'], DAY3_PAIRS)
    if d3.sum() > 1000:
        out['day3'] = st.block_bootstrap(gate['two_F'][d3], gate['DGDt_semilag'][d3],
                                         gate['block'][d3], st.slope_ols, n=200, seed=0)
        out['day3']['relative'] = out['day3']['value'] / st.BASELINE
        out['day3']['pairs'] = list(DAY3_PAIRS)
    return out


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def run(Ls=REPORT_L, n_boot=N_BOOT, say=print) -> dict:
    g = ot.open_grid(inp.GRID_ZARR, with_face=True)
    edge13 = mk.analysis_mask(g, edge_cells=13)
    n_analysis = int(inp._open_masks(inp.MASKS_NC)['mask_analysis'].values.sum())
    out = dict(declared=DECLARED, created=time.strftime('%Y-%m-%dT%H:%M:%S'),
               stores={str(L): str(bg.derived_path(L)) for L in Ls}, per_L={})
    for L in Ls:
        t = time.time()
        say(f'--- L = {L}')
        rows, pools = load_pool(L, edge13, n_analysis, say=say)
        gate = pools[f'p{FRONT_PCT:g}']
        c = closure(gate, rows, L)
        s = slopes(pools, L, n_boot=n_boot, say=say)
        quoted = c['verdict']['passed']
        out['per_L'][str(L)] = dict(
            closure=c, euler=euler(gate, pools['valid'], L), subfilter=subfilter_sweep(gate, L),
            fig2b=fig2b(gate), composites=composites(gate),
            **({'slopes': s} if quoted else {'slopes_not_quoted': s}),
            slopes_quotable=bool(quoted),
            slopes_label=('criterion 1 passed at this L: these may be quoted' if quoted else
                          'WHAT THE SLOPE WOULD HAVE BEEN -- NOT QUOTED. Criterion 1 failed at '
                          'this L, so no frontogenesis efficiency may be reported from it '
                          '(prompt 4 "Do not", first item)'),
            per_pair=rows, wall_s=round(time.time() - t, 1))
        say(f'--- L = {L} done in {time.time() - t:.0f} s: closure '
            f'{"PASS" if quoted else "FAIL"}')
    out['verdict'] = overall(out['per_L'])
    return out


def overall(per_L) -> dict:
    gated = {L: per_L[str(L)]['closure']['verdict']['passed'] for L in GATE_L if str(L) in per_L}
    euler_g = {L: per_L[str(L)]['euler']['passed'] for L in GATE_L if str(L) in per_L}
    any_pass = any(gated.values())
    return dict(
        criterion1=dict(per_L=gated, passed=bool(any_pass),
                        judged_at=list(GATE_L), tolerance=DECLARED['criterion1']),
        criterion2=dict(per_L=euler_g, passed=bool(all(euler_g.values()) if euler_g else False),
                        tolerance=DECLARED['criterion2']),
        L0_reported_not_gated=per_L.get('0', {}).get('closure', {}).get('verdict', {}),
        null_result=bool(not any_pass),
        null_rule='planning §12: a failure at every L is the null result; a failure at L = 0 '
                  'alone is not (M3-Q9)',
        efficiency_quotable=bool(any_pass),
        statement=('criterion 1 passed at ' + ', '.join(f'L={L}' for L, v in gated.items() if v)
                   if any_pass else
                   'CRITERION 1 FAILED AT EVERY GATED L -- planning §12 null result; the '
                   'conclusion is methodological and NO frontogenesis efficiency is quoted'))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--L', default=','.join(str(L) for L in REPORT_L),
                    type=lambda s: [int(x) for x in s.split(',') if x])
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--out', default=str(OUT_JSON))
    ap.add_argument('--no-fig', action='store_true')
    args = ap.parse_args(argv)
    res = run(args.L, n_boot=args.n_boot)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f'wrote {args.out}')
    print(format_verdict(res))
    if not args.no_fig:
        from m3_closure_fig import figure
        figure(res, OUT_PNG)
        print(f'wrote {OUT_PNG}')
    return 0


def format_verdict(res) -> str:
    w = '=' * 86
    out = [w, 'M3 CLOSURE — THE HARD GATE', w,
           f'  tolerances (pre-declared 2026-10-07): resid/meas <= {TOL_RESID_RATIO}, '
           f'explained >= {TOL_EXPLAINED}, |resid on 2F| <= {TOL_RESID_SLOPE}',
           f'  judged at L in {GATE_L}; L = 0 reported, not gated', '',
           f'  {"L":>3} {"2F":>7} {"subfil":>7} {"vert":>7} {"sflux":>7} {"resid":>7} '
           f'{"catch":>7} {"expl":>7} {"r~2F":>7} {"m~2F":>7}  verdict']
    for L, d in res['per_L'].items():
        c = d['closure']
        t = c['terms']
        out.append(
            f'  {L:>3} {t["two_F"]["over_measured"]:7.3f} {t["subfilter"]["over_measured"]:7.3f} '
            f'{t["vertical"]["over_measured"]:7.3f} {t["surface_flux"]["over_measured"]:7.3f} '
            f'{c["residual"]["over_measured"]:7.3f} {c["catchall"]["over_measured"]:7.3f} '
            f'{c["residual"]["explained"]:7.3f} {c["residual"]["slope_on_2F"]:+7.3f} '
            f'{c["measured_on_2F"]["ols"]:7.3f}  '
            + ('PASS' if c['verdict']['passed'] else 'FAIL')
            + ('' if c['verdict']['gated'] else ' (not gated)'))
    v = res['verdict']
    out += ['', '  criterion 1 (closure)  : ' + str(v['criterion1']['per_L']),
            '  criterion 2 (euler/SL) : ' + str(v['criterion2']['per_L']), '',
            '  ' + '!' * 82, f'  VERDICT: {v["statement"]}', '  ' + '!' * 82, w]
    return '\n'.join(out)


if __name__ == '__main__':
    sys.exit(main())
