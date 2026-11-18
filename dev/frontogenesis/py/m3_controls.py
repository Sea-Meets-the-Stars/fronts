""" M3-digging task 2: the controls -- a number for "no skill" -- and the pooling checks
(vet report §3, §6 items 1a, 5a, 5b; prompt M3-digging task 2).

The M3 gate reports the residual of ``measured - 2F - subfilter - vertical - surface_flux``
on the front pool as 0.813 / 0.841 / 0.898 / 1.042 of the measured amplitude at
``L = 0 / 2 / 4 / 8``.  The vet report's red team asked whether that is "what random noise
of the same variance would give".  This module gives that question a number, three ways,
and runs the pooling checks the same report asked for -- all from the four derived stores,
**reading the product and never writing it**.

(a) Three controls, at every ``L``, on the gate's pool (``front & valid`` of the *measured*
    pair):

    1. *random pair permutation* -- the measured side of pair ``n`` against the predicted
       terms of pair ``pi(n)`` at the same pixel, ``pi`` a derangement (no pair keeps its
       own terms), 20 draws -> a distribution;
    2. *the 24-h lag* (48 h as a check) -- pair ``n`` against pair ``n + 24``: the local
       solar hour is the same, so ``vertical`` and ``surface_flux`` keep their diurnal
       phase and only the flow is decorrelated.  **This is the control quoted beside the
       real number**;
    3. *the spatial roll* -- the same pair's predicted terms rolled by ``(dj, di)`` with
       ``|dj|, |di| >= 64`` cells (two bootstrap blocks), 20 draws.

    For the real pool every statistic carries its 32-cell / 3-h block-bootstrap interval;
    the controls carry their spread; the gap real - control is given in both units.  The
    back-of-envelope for independent fields with the real amplitudes,
    ``sqrt(1 + (rms 2F / rms measured)^2)``, is written beside the measured controls.

(b) Other selections -- the closure statistics on ``G_prev`` (the previous pair's ``G``,
    p90 over ``valid``: breaks selection-on-the-outcome), ``sigma_mag`` p90 (strain, not
    gradient), ``persist`` (``front`` in pairs ``n-1, n, n+1``), beside p80 / p90 / p95.

(c) Longer bootstrap blocks -- 1 / 3 / 12 / 24-h blocks: the intervals of the residual
    ratio, the explained fraction and the residual's slope on ``2F``.

(d) The case re-seeded (``m3_case.run``) at ``t0 - 12 h`` and on persistence, written to
    ``data/m3_case_controls.json``; the original summary is untouched.

**The pre-declaration** (:data:`DECLARED`) is written to the output JSON *before any store is
opened* (``main``), as ``m3_closure.py`` does.  "Skill over no-skill" at an ``L`` means the
real residual ratio lies below the **5th percentile** of the permutation distribution AND
below the 24-h-lag control by more than the real value's block-bootstrap half-width.  A
reference, not a gate: nothing here changes the M3 verdict.

Run (from dev/frontogenesis, detached; ~15 min -- estimate):
    nohup <frontogenesis env python> py/m3_controls.py > data/m3_controls.nohup 2>&1 &
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
import m3_closure as mc                                        # noqa: E402
import stats as st                                             # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

FIG_DIR = HERE.parent / 'figs'
OUT_JSON = DATA_DIR / 'm3_controls_summary.json'
OUT_PNG = FIG_DIR / 'm3_controls.png'
CASE_JSON = DATA_DIR / 'm3_case_controls.json'
PRODUCT_JSON = DATA_DIR / 'm3_closure_summary.json'
CASE_PRODUCT_JSON = DATA_DIR / 'm3_case_summary.json'

L_ALL = tuple(mc.REPORT_L)                     # (0, 2, 4, 8)
TERMS = bg.TERMS                               # two_F, subfilter, vertical, surface_flux
CHUNK_TERMS = bg.CHUNK_TERMS
PRED = ('two_F', 'subfilter', 'vertical', 'surface_flux')
MEAS = ('DGDt_semilag', 'DGDt_euler')
FIELDS = PRED + MEAS + ('residual', 'G', 'sigma_mag')
FRONT_PCT = mc.FRONT_PCT                       # 90
SENS_PCT = mc.FRONT_PCT_SENS                   # (80, 95)

# ---------------------------------------------------------------------------
# THE PRE-DECLARATION.  Fixed 2026-10-31 (prompt M3-digging task 2), before this
# module was run.  Nothing below changes after the numbers.
# ---------------------------------------------------------------------------
SEED = 0
N_PERM = 20
N_ROLL = 20
ROLL_MIN = 64                                  # cells: two 32-cell blocks, each axis
LAGS = (24, 48)                                # hours; 24 is the control quoted
BLOCK_HOURS = (1, 3, 12, 24)
REAL_CI_HOURS = 3                              # the interval quoted for the real value
SKILL_PCT = 5.0
N_BOOT = mc.N_BOOT                             # 1000
DECLARED = dict(
    source='frontogenesis_prompt_M3_digging.md task 2 (JXP, M3-Q27 (c)); fixed 2026-10-31 '
           'before m3_controls.py was run',
    skill_rule=dict(
        perm_percentile=SKILL_PCT, lag_hours=24, real_ci_block_hours=REAL_CI_HOURS,
        rule='the real residual ratio lies below the 5th percentile of the permutation '
             'distribution AND below the 24-h-lag control by more than the real value\'s '
             '32-cell / 3-h block-bootstrap half-width',
        status='a reference, not a gate: nothing here changes the M3 verdict'),
    seeds=dict(rng=f'numpy.random.default_rng({SEED})', n_perm=N_PERM,
               permutation='a derangement: no pair keeps its own predicted terms',
               n_roll=N_ROLL, roll_min_cells=ROLL_MIN, lags_h=list(LAGS), lag_wrap=False),
    bootstrap=dict(block_cells=st.BLOCK_CELLS, hours=list(BLOCK_HOURS), n_boot=N_BOOT,
                   ci=[2.5, 97.5], seed=0),
    pool='front & valid of the MEASURED pair: G >= p90 at the midpoint over valid, '
         'mask_analysis (edge_cells = 7); a control also requires valid of the source pair',
    selections=['p80', 'p90 (the gate)', 'p95', 'G_prev', 'sigma_mag', 'persist'],
    independent_reference='sqrt(1 + (rms 2F / rms measured)^2): the residual ratio of two '
                          'independent fields with the real amplitudes (prompt 4 Q&A, '
                          'Digging, 2026-10-31)',
    Ls=list(L_ALL))


# ---------------------------------------------------------------------------
# estimators with a moment form (the bootstrap's fast path)
# ---------------------------------------------------------------------------
@st._moment_form(lambda T: np.sqrt(T[..., 4] / T[..., 3]))
def ratio_rms(x, y) -> float:
    """``rms(y) / rms(x)`` over the finite cells; from the raw sums of squares."""
    x, y = np.asarray(x, 'float64').ravel(), np.asarray(y, 'float64').ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.sum(y[ok] ** 2) / np.sum(x[ok] ** 2)))


@st._moment_form(lambda T: 1.0 - st._central(T)[1] / st._central(T)[0])
def explained(x, y) -> float:
    """``1 - var(y) / var(x)`` (``x`` measured, ``y`` the residual)."""
    x, y = np.asarray(x, 'float64').ravel(), np.asarray(y, 'float64').ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(1.0 - np.var(y[ok]) / np.var(x[ok]))


BOOT = dict(resid_over_meas=('DGDt_semilag', 'residual', ratio_rms),
            explained=('DGDt_semilag', 'residual', explained),
            resid_on_2F=('two_F', 'residual', st.slope_ols))


# ---------------------------------------------------------------------------
# the fields of one L, in memory
# ---------------------------------------------------------------------------
def load_fields(L, path=None, say=print) -> dict:
    """Every field the controls need, for all pairs of one store, as
    ``(n, nj, ni)`` arrays (float32; bool masks), plus the local solar hour
    per pair from the store's own ``utc_offset_h``."""
    path = bg.derived_path(L) if path is None else Path(path)
    ds = xr.open_zarr(path)
    say(f'  L={L}: loading {path.name} ({ds.sizes["time"]} pairs)')
    offset = float(ds.attrs.get('utc_offset_h', inp.UTC_OFFSET_H))
    F = {k: np.asarray(ds[k].values, dtype='float32') for k in FIELDS if k in ds}
    for k in ('valid', 'front'):
        F[k] = np.asarray(ds[k].values, dtype=bool)
    F['lst'] = np.array([float(inp.local_solar_hour(t, offset))
                         for t in ds['time_mid'].values], dtype='float64')
    F['n'], F['shape'], F['L'], F['path'] = int(ds.sizes['time']), F['valid'].shape[1:], int(L), str(path)
    F['pred'] = tuple(k for k in PRED if k in F)
    ds.close()
    return F


def default_select(F, n):
    return F['front'][n] & F['valid'][n]


def gather(F, src=None, roll=None, select=None, stored_residual=False, block_hours=()):
    """Pool the pairs: the measured side of pair ``n`` with the predicted
    terms of pair ``src[n]`` (``None`` skips the pair) at the same pixel, or
    rolled by ``roll = (dj, di)``; ``select(F, n)`` is the pixel mask (default
    ``front & valid``), to which the source pair's ``valid`` is added.
    ``stored_residual`` pools the store's own residual column (the real pool
    only); otherwise the residual is recomputed from the pooled terms.
    ``block_hours`` adds ``block{h}`` labels (``stats.space_time_block_ids``
    on the measured pair).  Returns 1-D float64 arrays."""
    n_pairs = F['n']
    src = list(range(n_pairs)) if src is None else list(src)
    select = default_select if select is None else select
    keys = list(F['pred']) + list(MEAS) + ['pair', 'lst'] + [f'block{h}' for h in block_hours]
    if stored_residual:
        keys.append('residual')
    out = {k: [] for k in keys}
    for n in range(n_pairs):
        m = src[n]
        if m is None:
            continue
        sel = select(F, n)
        if roll is None:
            pv = F['valid'][m]
            take = (lambda k: F[k][m])
        else:
            dj, di = roll
            pv = np.roll(F['valid'][m], (dj, di), axis=(0, 1))
            take = (lambda k: np.roll(F[k][m], (dj, di), axis=(0, 1)))
        sel = sel & pv
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        for k in F['pred']:
            out[k].append(take(k)[sel].astype('float64'))
        for k in MEAS:
            out[k].append(F[k][n][sel].astype('float64'))
        if stored_residual:
            out['residual'].append(F['residual'][n][sel].astype('float64'))
        out['pair'].append(np.full(cnt, n, dtype='int16'))
        out['lst'].append(np.full(cnt, F['lst'][n], dtype='float64'))
        for h in block_hours:
            out[f'block{h}'].append(st.space_time_block_ids(F['shape'], n, hours=h)[sel])
    P = {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}
    if not stored_residual:
        P['residual'] = P['DGDt_semilag'] - sum(P[t] for t in F['pred'])
    return P


# ---------------------------------------------------------------------------
# the statistics of a pool (the task-6 set, on whatever terms are present)
# ---------------------------------------------------------------------------
def pooled_stats(P, pred=PRED) -> dict:
    meas, F2, res = P['DGDt_semilag'], P['two_F'], P['residual']
    present = [t for t in pred if t in P]
    chunk = [t for t in CHUNK_TERMS if t in P]
    catch = res + sum((P[t] for t in chunk), 0.0)
    r_m, r_F = mc._rms(meas), mc._rms(F2)
    fr, fm = mc._fit(F2, res), mc._fit(F2, meas)
    out = dict(n=int(meas.size), n_pairs=int(np.unique(P['pair']).size) if 'pair' in P else None,
               rms_measured=r_m, rms_two_F=r_F, rms_residual=mc._rms(res),
               resid_over_meas=mc._rms(res) / r_m, explained=mc._explained(meas, res),
               resid_on_2F=fr['ols'], resid_corr_2F=fr['corr'],
               meas_on_2F=fm['ols'], meas_corr_2F=fm['corr'],
               catchall_over_meas=mc._rms(catch) / r_m,
               catchall_explained=mc._explained(meas, catch),
               independent_reference=float(np.sqrt(1.0 + (r_F / r_m) ** 2)),
               terms_over_meas={t: mc._rms(P[t]) / r_m for t in present},
               optimal_multiplier={t: mc._fit(P[t], meas - sum(P[u] for u in present if u != t))['ols']
                                   for t in present})
    if 'DGDt_euler' in P:
        e = mc._fit(meas, P['DGDt_euler'])
        out['euler_on_semilag'] = e['ols']
        out['euler_corr'] = e['corr']
    return out


def by_local_hour(P) -> dict:
    meas, res = P['DGDt_semilag'], P['residual']
    hrs = np.floor(P['lst']).astype(int) % 24
    return {int(h): dict(n=int((hrs == h).sum()),
                         resid_over_meas=mc._rms(res[hrs == h]) / mc._rms(meas[hrs == h]),
                         mean_resid_over_rms_meas=float(np.nanmean(res[hrs == h]) / mc._rms(meas[hrs == h])))
            for h in np.unique(hrs)}


def bootstrap(P, hours, n_boot=N_BOOT, stats=BOOT) -> dict:
    """The three statistics with their block-bootstrap intervals at one block
    depth (``block{hours}`` labels must be in ``P``)."""
    lab = P[f'block{hours}']
    out = {}
    for k, (xk, yk, est) in stats.items():
        b = st.block_bootstrap(P[xk], P[yk], lab, estimator=est, n=n_boot, seed=0)
        out[k] = dict(value=b['value'], lo=b['lo'], hi=b['hi'], se=b['se'],
                      half_width=0.5 * (b['hi'] - b['lo']), n_blocks=b['n_blocks'])
    out['block_hours'] = int(hours)
    return out


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------
def derangement(n, rng) -> np.ndarray:
    """A random permutation with no fixed point."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def draw_roll(rng, shape) -> tuple:
    """``(dj, di)`` with ``|dj|, |di| >= ROLL_MIN`` and at least ``ROLL_MIN``
    from the wrap-around on the other side, signs random."""
    out = []
    for size in shape:
        hi = max(ROLL_MIN + 1, int(size) - ROLL_MIN + 1)    # a small (test) grid: ROLL_MIN only
        mag = int(rng.integers(ROLL_MIN, hi))
        out.append(mag * int(rng.choice([-1, 1])))
    return tuple(out)


def _dist(vals) -> dict:
    a = np.asarray(vals, 'float64')
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), median=float(np.median(a)), mean=float(a.mean()),
                sd=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                p5=float(np.percentile(a, SKILL_PCT)), p95=float(np.percentile(a, 100 - SKILL_PCT)),
                min=float(a.min()), max=float(a.max()))


DIST_KEYS = ('resid_over_meas', 'explained', 'resid_on_2F', 'resid_corr_2F', 'meas_on_2F',
             'meas_corr_2F', 'catchall_over_meas', 'catchall_explained')


def control_family(members, extra=None) -> dict:
    """A list of per-draw statistics -> the distributions plus the members."""
    out = dict(n_draws=len(members),
               dist={k: _dist([m[k] for m in members]) for k in DIST_KEYS},
               optimal_multiplier={t: _dist([m['optimal_multiplier'][t] for m in members])
                                   for t in members[0]['optimal_multiplier']},
               members=members)
    if extra:
        out.update(extra)
    return out


def control_permutation(F, rng, n_perm=N_PERM, say=print) -> dict:
    members = []
    for k in range(n_perm):
        perm = derangement(F['n'], rng)
        s = pooled_stats(gather(F, perm))
        s['perm'] = perm.tolist()
        members.append(s)
        say(f'    perm {k + 1}/{n_perm}: resid/meas {s["resid_over_meas"]:.3f}')
    return control_family(members, dict(kind='random pair permutation (derangement)'))


def control_lag(F, lag, n_boot=N_BOOT, say=print) -> dict:
    src = [n + lag if n + lag < F['n'] else None for n in range(F['n'])]
    P = gather(F, src, block_hours=(REAL_CI_HOURS,))
    s = pooled_stats(P)
    s.update(kind=f'{lag}-h lag: measured pair n against the predicted terms of pair n + {lag}',
             lag_h=int(lag), n_pairs=int(F['n'] - lag),
             ci=bootstrap(P, REAL_CI_HOURS, n_boot=n_boot),
             by_local_hour=by_local_hour(P))
    say(f'    lag {lag} h: resid/meas {s["resid_over_meas"]:.3f} '
        f'[{s["ci"]["resid_over_meas"]["lo"]:.3f}, {s["ci"]["resid_over_meas"]["hi"]:.3f}]')
    return s


def control_roll(F, rng, n_roll=N_ROLL, say=print) -> dict:
    members = []
    for k in range(n_roll):
        dj, di = draw_roll(rng, F['shape'])
        s = pooled_stats(gather(F, None, roll=(dj, di)))
        s['roll'] = [int(dj), int(di)]
        members.append(s)
        say(f'    roll {k + 1}/{n_roll} ({dj:+d}, {di:+d}): resid/meas {s["resid_over_meas"]:.3f}')
    return control_family(members, dict(kind='spatial roll of the same pair\'s predicted terms',
                                        roll_min_cells=ROLL_MIN))


# ---------------------------------------------------------------------------
# the real pool, the selections, the blocks
# ---------------------------------------------------------------------------
def real_pool(F, n_boot=N_BOOT, say=print) -> tuple:
    P = gather(F, None, stored_residual=True, block_hours=BLOCK_HOURS)
    s = pooled_stats(P)
    s['kind'] = 'the gate\'s pool, the stored residual (task 6\'s numbers)'
    s['ci'] = {f'{h}h': bootstrap(P, h, n_boot=n_boot) for h in BLOCK_HOURS}
    s['by_local_hour'] = by_local_hour(P)
    say(f'    real: resid/meas {s["resid_over_meas"]:.3f} '
        f'[{s["ci"][f"{REAL_CI_HOURS}h"]["resid_over_meas"]["lo"]:.3f}, '
        f'{s["ci"][f"{REAL_CI_HOURS}h"]["resid_over_meas"]["hi"]:.3f}] (3-h blocks); '
        f'independent-field reference {s["independent_reference"]:.3f}')
    return s, P


def _pct_select(F, n, field, pct):
    v = F['valid'][n]
    a = F[field][n]
    return v & (a >= np.percentile(a[v], pct))


def select_G_prev(F, n):
    """The previous pair's ``G`` at the same pixel, p90 over both pairs' valid."""
    if n == 0:
        return np.zeros(F['shape'], bool)
    v = F['valid'][n] & F['valid'][n - 1]
    g = F['G'][n - 1]
    return v & (g >= np.percentile(g[v], FRONT_PCT))


def select_persist(F, n):
    if n == 0 or n == F['n'] - 1:
        return np.zeros(F['shape'], bool)
    return F['front'][n - 1] & F['front'][n] & F['front'][n + 1] & F['valid'][n]


SELECTIONS = {
    'p80': lambda F, n: _pct_select(F, n, 'G', SENS_PCT[0]),
    'p90': default_select,
    'p95': lambda F, n: _pct_select(F, n, 'G', SENS_PCT[1]),
    'G_prev': select_G_prev,
    'sigma_mag': lambda F, n: _pct_select(F, n, 'sigma_mag', FRONT_PCT),
    'persist': select_persist,
}
SELECTION_NOTE = {
    'p80': 'G >= p80 at the midpoint over valid (task 6 sensitivity)',
    'p90': 'the gate\'s pool: G >= p90 at the midpoint over valid',
    'p95': 'G >= p95 at the midpoint over valid (task 6 sensitivity)',
    'G_prev': 'the PREVIOUS pair\'s G at the same pixel >= its p90 over valid: the selection '
              'no longer sees the pair\'s own outcome (vet 5a); pair 0 dropped',
    'sigma_mag': 'strain magnitude >= p90 over valid: selected on the flow, not the gradient',
    'persist': 'front in pairs n-1, n and n+1 (and valid at n): persistent front pixels; '
               'pairs 0 and 70 dropped',
}


def selections(F, say=print) -> dict:
    out = {}
    for name, fn in SELECTIONS.items():
        P = gather(F, None, select=fn, stored_residual=True)
        s = pooled_stats(P)
        s['note'] = SELECTION_NOTE[name]
        out[name] = s
        say(f'    selection {name:10s}: n {s["n"]:>9,}  resid/meas {s["resid_over_meas"]:.3f}  '
            f'explained {s["explained"]:+.3f}  resid~2F {s["resid_on_2F"]:+.3f}')
    return out


def blocks(real) -> dict:
    """The interval widths by block depth, from the real pool's bootstraps."""
    out = {}
    for h in BLOCK_HOURS:
        c = real['ci'][f'{h}h']
        out[f'{h}h'] = {k: dict(lo=c[k]['lo'], hi=c[k]['hi'], width=c[k]['hi'] - c[k]['lo'],
                                se=c[k]['se'], n_blocks=c[k]['n_blocks'])
                        for k in BOOT}
    w1 = out['1h']
    for h in BLOCK_HOURS[1:]:
        out[f'{h}h']['width_over_1h'] = {k: (out[f'{h}h'][k]['width'] / w1[k]['width']
                                             if w1[k]['width'] > 0 else None) for k in BOOT}
    return out


# ---------------------------------------------------------------------------
# the gaps and the declared reading
# ---------------------------------------------------------------------------
def gaps(real, controls) -> dict:
    """Real - control for each statistic, in units of the control's spread
    (sd over draws; the lag's bootstrap se) and of the real interval's
    half-width (3-h blocks)."""
    out = {}
    for k in DIST_KEYS:
        rv = real[k]
        hw = real['ci'][f'{REAL_CI_HOURS}h'][k]['half_width'] if k in real['ci'][f'{REAL_CI_HOURS}h'] else None
        row = {}
        for name in ('permutation', 'roll'):
            d = controls[name]['dist'][k]
            row[name] = dict(control_median=d['median'], gap=rv - d['median'],
                             in_control_sd=((rv - d['median']) / d['sd']) if d['sd'] > 0 else None,
                             in_real_halfwidth=((rv - d['median']) / hw) if hw else None)
        for lag in LAGS:
            c = controls[f'lag{lag}']
            se = c['ci'][k]['se'] if k in c['ci'] else None
            row[f'lag{lag}'] = dict(control=c[k], gap=rv - c[k],
                                    in_control_se=((rv - c[k]) / se) if se else None,
                                    in_real_halfwidth=((rv - c[k]) / hw) if hw else None)
        out[k] = row
    return out


def skill(real, controls) -> dict:
    """The declared rule, on the residual ratio."""
    rv = real['resid_over_meas']
    hw = real['ci'][f'{REAL_CI_HOURS}h']['resid_over_meas']['half_width']
    p5 = controls['permutation']['dist']['resid_over_meas']['p5']
    lag = controls['lag24']['resid_over_meas']
    c1, c2 = bool(rv < p5), bool(rv < lag - hw)
    return dict(real=rv, real_half_width=hw, perm_p5=p5, lag24=lag, lag24_minus_halfwidth=lag - hw,
                below_perm_p5=c1, below_lag24_by_halfwidth=c2, passed=bool(c1 and c2),
                rule=DECLARED['skill_rule']['rule'], status=DECLARED['skill_rule']['status'])


# ---------------------------------------------------------------------------
# the case re-seeded (d)
# ---------------------------------------------------------------------------
CASE_RUNS = {
    'seed_minus_12h': dict(t0=18, n_back=18, n_fwd=24, seed_rule='p90',
                           note='the same corridor seeded 12 h earlier (pair 18, 2012-07-02 18 UTC); '
                                'the window is -18/+24 h because pair 18 has only 18 hours before it'),
    'seed_persist': dict(t0=30, n_back=24, n_fwd=24, seed_rule='persist',
                         note='seeded on persistence: front in >= 5 of the 7 pairs t0-3..t0+3, '
                              'not on G >= p90 at t0 alone'),
}


def case_controls(say=print) -> dict:
    import m3_case
    base = json.loads(CASE_PRODUCT_JSON.read_text()) if CASE_PRODUCT_JSON.exists() else None
    keys = ('change_factor', 'predicted_over_observed_change', 'budget_over_observed_change')
    out = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),
               original=(dict(**{k: base['strength'][k] for k in keys},
                              control_change_factor=base['selection']['control_change_factor'],
                              t0=base['case']['t0'], n_seeded=base['case']['n_seeded'],
                              source=str(CASE_PRODUCT_JSON)) if base else None),
               runs={})
    for name, kw in CASE_RUNS.items():
        t = time.time()
        say(f'  case {name}: {kw}')
        note = kw.pop('note')
        s = m3_case.run(out_json=DATA_DIR / f'm3_case_controls_{name}.json',
                        out_npz=DATA_DIR / f'm3_case_controls_{name}.npz', **kw)
        kw['note'] = note
        out['runs'][name] = dict(
            params=kw, **{k: s['strength'][k] for k in keys},
            control_change_factor=s['selection']['control_change_factor'],
            n_seeded=s['case']['n_seeded'], n_followed=s['case']['n_followed'],
            t_seed=s['case']['t_seed'], residual_over_measured=s['residual_over_measured'],
            corr_measured_2F=s['rates']['corr_measured_2F'],
            checks=dict(trajectory_median_return_error_cells=s['checks']['trajectory']['median_return_error_cells'],
                        rate_corr=s['checks']['rate']['corr']),
            wall_s=round(time.time() - t, 1))
        say(f'    {name}: change x{s["strength"]["change_factor"]:.2f}, control '
            f'x{s["selection"]["control_change_factor"]:.2f}, 2F/obs '
            f'{s["strength"]["predicted_over_observed_change"]:+.2f}, budget/obs '
            f'{s["strength"]["budget_over_observed_change"]:+.2f} ({time.time() - t:.0f} s)')
    return out


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def product_reference() -> dict:
    if not PRODUCT_JSON.exists():
        return {}
    S = json.loads(PRODUCT_JSON.read_text())
    return {L: dict(resid_over_meas=S['per_L'][L]['closure']['residual']['over_measured'],
                    explained=S['per_L'][L]['closure']['residual']['explained'],
                    resid_on_2F=S['per_L'][L]['closure']['residual']['slope_on_2F'],
                    catchall_over_meas=S['per_L'][L]['closure']['catchall']['over_measured'],
                    meas_corr_2F=S['per_L'][L]['closure']['measured_on_2F']['corr'])
            for L in S['per_L']}


def run_L(L, n_perm=N_PERM, n_roll=N_ROLL, n_boot=N_BOOT, path=None, say=print) -> dict:
    t = time.time()
    rng = np.random.default_rng(SEED)
    F = load_fields(L, path=path, say=say)
    real, _ = real_pool(F, n_boot=n_boot, say=say)
    controls = dict(permutation=control_permutation(F, rng, n_perm, say=say))
    for lag in LAGS:
        controls[f'lag{lag}'] = control_lag(F, lag, n_boot=n_boot, say=say)
    controls['roll'] = control_roll(F, rng, n_roll, say=say)
    out = dict(L=int(L), store=F['path'], n_pairs=F['n'], real=real, controls=controls,
               gaps=gaps(real, controls), skill=skill(real, controls),
               selections=selections(F, say=say), blocks=blocks(real),
               wall_s=round(time.time() - t, 1))
    say(f'  L={L} done in {out["wall_s"]:.0f} s: skill {out["skill"]["passed"]} '
        f'(real {real["resid_over_meas"]:.3f} vs perm p5 {out["skill"]["perm_p5"]:.3f}, '
        f'lag24 - hw {out["skill"]["lag24_minus_halfwidth"]:.3f})')
    return out


def run(Ls=L_ALL, n_perm=N_PERM, n_roll=N_ROLL, n_boot=N_BOOT, case=True, say=print) -> dict:
    out = dict(declared=DECLARED, created=time.strftime('%Y-%m-%dT%H:%M:%S'),
               product_reference=product_reference(), per_L={})
    for L in Ls:
        say(f'--- L = {L}')
        out['per_L'][str(L)] = run_L(L, n_perm, n_roll, n_boot, say=say)
    ref = out['product_reference']
    out['product_check'] = {L: dict(real=out['per_L'][L]['real']['resid_over_meas'],
                                    product=ref[L]['resid_over_meas'],
                                    agree=bool(abs(out['per_L'][L]['real']['resid_over_meas']
                                                   - ref[L]['resid_over_meas']) < 1e-6))
                            for L in out['per_L'] if L in ref}
    if case:
        say('--- the case re-seeded')
        out['case_controls'] = case_controls(say=say)
        CASE_JSON.write_text(json.dumps(out['case_controls'], indent=1, default=str))
        say(f'wrote {CASE_JSON}')
    out['skill_summary'] = {L: out['per_L'][L]['skill']['passed'] for L in out['per_L']}
    out['verdict_note'] = ('a reference for the M3 null, not a gate: criterion 1 failed at every '
                           'L (m3_closure_summary.json) and nothing here changes that')
    return out


def format_table(res) -> str:
    w = '=' * 100
    out = [w, 'M3 DIGGING TASK 2 -- THE CONTROLS (residual / measured on front & valid)', w,
           f'  {"L":>3} {"real":>7} {"[3h ci]":>16} {"indep":>7} {"perm p5":>8} {"perm med":>9} '
           f'{"lag24":>7} {"lag48":>7} {"roll med":>9}  skill']
    for L, d in res['per_L'].items():
        r, c, s = d['real'], d['controls'], d['skill']
        ci = r['ci'][f'{REAL_CI_HOURS}h']['resid_over_meas']
        out.append(f'  {L:>3} {r["resid_over_meas"]:7.3f} [{ci["lo"]:6.3f}, {ci["hi"]:6.3f}] '
                   f'{r["independent_reference"]:7.3f} {c["permutation"]["dist"]["resid_over_meas"]["p5"]:8.3f} '
                   f'{c["permutation"]["dist"]["resid_over_meas"]["median"]:9.3f} '
                   f'{c["lag24"]["resid_over_meas"]:7.3f} {c["lag48"]["resid_over_meas"]:7.3f} '
                   f'{c["roll"]["dist"]["resid_over_meas"]["median"]:9.3f}  '
                   + ('SKILL' if s['passed'] else 'no'))
    out += ['', f'  {"L":>3} {"sel":>10} {"n":>10} {"resid":>7} {"expl":>7} {"r~2F":>7} {"m~2F":>7}']
    for L, d in res['per_L'].items():
        for name, s in d['selections'].items():
            out.append(f'  {L:>3} {name:>10} {s["n"]:>10,} {s["resid_over_meas"]:7.3f} '
                       f'{s["explained"]:+7.3f} {s["resid_on_2F"]:+7.3f} {s["meas_on_2F"]:7.3f}')
    out += ['', '  interval width of resid/meas by block depth (x the 1-h width):']
    for L, d in res['per_L'].items():
        b = d['blocks']
        out.append(f'  {L:>3} ' + '  '.join(f'{h}: {b[h]["resid_over_meas"]["width"]:.4f}'
                                          + (f' ({b[h]["width_over_1h"]["resid_over_meas"]:.2f}x)'
                                             if h != '1h' else '') for h in b))
    if res.get('case_controls'):
        cc = res['case_controls']
        out += ['', '  the case re-seeded: change x, control x, 2F/obs, budget/obs']
        if cc['original']:
            o = cc['original']
            out.append(f'  {"original":>16} {o["change_factor"]:6.2f} {o["control_change_factor"]:6.2f} '
                       f'{o["predicted_over_observed_change"]:+6.2f} {o["budget_over_observed_change"]:+6.2f}')
        for name, r in cc['runs'].items():
            out.append(f'  {name:>16} {r["change_factor"]:6.2f} {r["control_change_factor"]:6.2f} '
                       f'{r["predicted_over_observed_change"]:+6.2f} {r["budget_over_observed_change"]:+6.2f}')
    out += ['', '  ' + res.get('verdict_note', ''), w]
    return '\n'.join(out)


def write_declared(out_path):
    """The pre-declaration goes to disk BEFORE any store is opened."""
    Path(out_path).write_text(json.dumps(dict(
        declared=DECLARED, status='declared; stores not yet opened',
        created=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--L', default=','.join(str(L) for L in L_ALL),
                    type=lambda s: [int(x) for x in s.split(',') if x])
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--n-roll', type=int, default=N_ROLL)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--no-case', action='store_true', help='skip the re-seeded case runs')
    ap.add_argument('--out', default=str(OUT_JSON))
    ap.add_argument('--no-fig', action='store_true')
    args = ap.parse_args(argv)
    if args.n_perm != N_PERM or args.n_roll != N_ROLL or args.n_boot != N_BOOT:
        print(f'NOTE: draws differ from the declaration ({N_PERM} / {N_ROLL} / {N_BOOT}); '
              'this is a trial run, not the record')
    write_declared(args.out)
    print(f'declared -> {args.out}')
    res = run(args.L, args.n_perm, args.n_roll, args.n_boot, case=not args.no_case)
    res['draws'] = dict(n_perm=args.n_perm, n_roll=args.n_roll, n_boot=args.n_boot,
                        as_declared=bool(args.n_perm == N_PERM and args.n_roll == N_ROLL
                                         and args.n_boot == N_BOOT))
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f'wrote {args.out}')
    print(format_table(res))
    if not args.no_fig:
        from m3_controls_fig import figure
        figure(res, OUT_PNG)
        print(f'wrote {OUT_PNG}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
