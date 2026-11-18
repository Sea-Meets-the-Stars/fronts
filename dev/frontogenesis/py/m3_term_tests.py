""" M3-digging task 3: the two term tests -- ``S`` split into its parts, ``V`` from the
second cell (vet report §6 items 4 and 3; prompt M3-digging task 3).

Both tests re-read the raw and chunk stores through ``inputs.open_inputs`` exactly as the
sweep did, recompute the stored term on the way (and assert it), and write **no field to
disk** -- only ``data/m3_term_tests_summary.json`` and a figure.  The product is read, never
written; nothing in the derived stores' schema changes.

(a) **Split ``S``.**  ``vertical.surface_buoyancy_tendency`` is linear in the three fluxes
    at fixed ``Theta``/``Salt``, and ``lowpass``, ``grad_b`` and the dot product are linear,
    so the surface-flux term is exactly the sum of three components, obtained by calling
    ``vertical.surface_flux_term`` with the other two fluxes zeroed:

    * non-solar heat  : ``oceQnet -> oceQnet - oceQsw``, ``oceQsw -> 0``, ``oceFWflx -> 0``;
    * shortwave       : ``oceQnet -> oceQsw``, ``oceQsw`` as stored, ``oceFWflx -> 0``
                        (so ``Q_top = f_sw oceQsw``);
    * freshwater      : ``oceQnet -> 0``, ``oceQsw -> 0``, ``oceFWflx`` as stored;

    in both depth forms (``drF[0]``; ``depth = KPPhbl``, M3-Q13 (c)).  The identity -- the
    three sum to the recomputed full term to 1e-10 relative in rms, and the recomputed term
    equals the stored one -- is asserted on every pair.  On the stored ``front & valid`` the
    imbalance ``I = DGDt_semilag - two_F - subfilter - vertical`` is regressed on the three
    components (OLS with intercept, 32-cell / 3-h block bootstrap on the multipliers and on
    the rms gain), and the rms of ``I - sum a_k S_k`` is set against ``I - S`` (the unit sum,
    the product's residual) and ``I`` (no ``S``); the two-component version (heat,
    freshwater) and the single-multiplier version (task 6's optimum) beside it.

(b) **``V`` from the second cell.**  On the three smoke pairs of prompt 4 task 2 (0, 9, 21:
    16 / 01 / 13 LST) plus pair 30 (the case seed), at ``L = 0`` and ``4``: the stored
    top-cell term (recomputed and asserted), the factorised form with ``b_z`` of order 1
    (stored) and order 2 (the three-level sensitivity), and the tilting term of the **second**
    cell's budget -- ``vertical_term(b_k1, grad b_k1, b_k2, W_k2, drF[1:], ...)`` with the new
    ``inputs.W_k2`` -- on the pair's ``front & valid``: rms over measured, the slope and
    correlation of the second-cell term on the top-cell term, and the sign of each term's
    pool mean at that local hour.

**The pre-declaration** (:data:`DECLARED`) is written to the output JSON before any store
is opened.  The split reveals a **cancellation** if the three-multiplier fit lowers the
imbalance's rms below the unit-sum residual's by more than the bootstrap half-width AND
every multiplier lies in [0.5, 2]; otherwise the interim's §6.2 (orthogonality) stands.
``V`` from the second cell is a sensitivity with no threshold.

Run (from dev/frontogenesis, detached; ~20 min -- estimate):
    nohup <frontogenesis env python> py/m3_term_tests.py > data/m3_term_tests.nohup 2>&1 &
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
import operators as op                                         # noqa: E402
import stats as st                                             # noqa: E402
import vertical as vt                                          # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

FIG_DIR = HERE.parent / 'figs'
OUT_JSON = DATA_DIR / 'm3_term_tests_summary.json'
OUT_PNG = FIG_DIR / 'm3_term_tests.png'

L_ALL = tuple(mc.REPORT_L)                     # (0, 2, 4, 8)
COMP = ('nonsolar', 'shortwave', 'freshwater')
DEPTHS = ('top', 'kpp')                        # drF[0] (primary); KPPhbl (M3-Q13 (c))
STORED = dict(top='surface_flux', kpp='surface_flux_kpp')
SMOKE_PAIRS = (0, 9, 21, 30)                   # 16, 01, 13 LST; the case seed
V_LS = (0, 4)
IDENTITY_TOL = 1e-10
BOUNDS = (0.5, 2.0)
CI_HOURS = 3
N_BOOT = mc.N_BOOT
DECLARED = dict(
    source='frontogenesis_prompt_M3_digging.md task 3 (JXP, M3-Q27 (c)); fixed 2026-10-31 '
           'before m3_term_tests.py was run',
    split=dict(
        rule='CANCELLATION if rms(I - sum_k a_k S_k) < rms(I - S) by more than the 32-cell / '
             '3-h block-bootstrap half-width of that difference AND every a_k in [0.5, 2]; '
             'otherwise the interim\'s §6.2 (S near-orthogonal to the imbalance) stands',
        multiplier_bounds=list(BOUNDS), ci_block_hours=CI_HOURS, n_boot=N_BOOT, ci=[2.5, 97.5],
        imbalance='I = DGDt_semilag - two_F - subfilter - vertical, from the store',
        pool='front & valid of the stored pair', components=list(COMP), depth_forms=list(DEPTHS),
        identity_tol_relative_rms=IDENTITY_TOL, Ls=list(L_ALL)),
    vertical=dict(
        rule='a SENSITIVITY, no threshold: reported is whether the second-cell term keeps the '
             'top-cell term\'s diurnal phase and sign, with the slope and correlation between them',
        pairs=list(SMOKE_PAIRS), Ls=list(V_LS),
        readings=dict(vet='"if it moves, it is a grid artefact" (vet §6 item 3)',
                      interim='§2.4: the term IS a discretisation term (the flux through the '
                              'base of a fixed box); moving with the cell is the prediction')))


# ---------------------------------------------------------------------------
# (a) the split, per pair
# ---------------------------------------------------------------------------
def _with(da, values):
    """A copy of ``da`` with new values and the SAME attrs (the flux-sign guard
    reads ``sign_convention`` from the attrs)."""
    return da.copy(data=np.asarray(values, dtype=da.dtype))


def flux_variants(qnet, qsw, fw) -> dict:
    """The three (oceQnet, oceQsw, oceFWflx) triples whose terms sum to the full one."""
    z = (lambda da: _with(da, np.zeros_like(da.values)))
    return {'nonsolar': (_with(qnet, qnet.values - qsw.values), z(qsw), z(fw)),
            'shortwave': (_with(qnet, qsw.values), qsw, z(fw)),
            'freshwater': (z(qnet), z(qsw), fw)}


def midpoint_inputs(h0, h1, L, grid_ds, grid):
    """What ``compute_budget`` builds before the surface-flux term: the filtered
    midpoint gradient, the midpoint fluxes, ``Theta``, ``Salt``, ``KPPhbl``."""
    f0, f1 = inp.filtered(h0, L), inp.filtered(h1, L)
    b_mid = inp.midpoint(f0[0], f1[0])
    b_x, b_y = op.grad_b(b_mid, grid_ds, grid)
    fl = [inp.midpoint(a, b) for a, b in zip(inp.fluxes(h0), inp.fluxes(h1))]
    Th = inp.midpoint(h0['Theta'], h1['Theta'])
    Sa = inp.midpoint(h0['Salt'], h1['Salt'])
    kpp = inp.midpoint(h0['KPPhbl'], h1['KPPhbl'])
    return dict(b_x=b_x, b_y=b_y, fluxes=fl, Th=Th, Sa=Sa, kpp=kpp)


def split_components(h0, h1, L, grid_ds, grid, drF) -> dict:
    """``{depth: {'full': 2 S, 'nonsolar': ..., 'shortwave': ..., 'freshwater': ...}}``
    as numpy ``(j, i)`` float64, the budget's ``2 x`` convention, plus the
    identity's relative rms error per depth form."""
    m = midpoint_inputs(h0, h1, L, grid_ds, grid)
    qnet, qsw, fw = m['fluxes']
    out = {}
    for depth in DEPTHS:
        d = None if depth == 'top' else m['kpp']
        comp = {}
        for name, (a, b, c) in flux_variants(qnet, qsw, fw).items():
            comp[name] = 2.0 * vt.surface_flux_term(m['b_x'], m['b_y'], a, b, c, m['Th'], m['Sa'],
                                                    drF, grid_ds, grid, L_cells=L, depth=d)
        full = 2.0 * vt.surface_flux_term(m['b_x'], m['b_y'], qnet, qsw, fw, m['Th'], m['Sa'],
                                          drF, grid_ds, grid, L_cells=L, depth=d)
        comp = {k: np.squeeze(np.asarray(v.values, 'float64')) for k, v in comp.items()}
        full = np.squeeze(np.asarray(full.values, 'float64'))
        s = sum(comp.values())
        err = mc._rms(s - full) / mc._rms(full)
        out[depth] = dict(full=full, identity_rel_rms=float(err), **comp)
    return out


def identity_check(split, stored, valid) -> dict:
    """The split identity (components vs the recomputed full term) and the
    provenance identity (the recomputed term vs the store, float32)."""
    out = {}
    for depth in DEPTHS:
        full = split[depth]['full']
        st32 = np.asarray(stored[depth], 'float64')
        ok = valid & np.isfinite(full) & np.isfinite(st32)
        out[depth] = dict(
            split_rel_rms=split[depth]['identity_rel_rms'],
            split_ok=bool(split[depth]['identity_rel_rms'] <= IDENTITY_TOL),
            vs_store_rel_rms=float(mc._rms(full[ok].astype('float32') - st32[ok]) / mc._rms(st32[ok])),
            vs_store_max_rel=float(np.max(np.abs(full[ok].astype('float32') - st32[ok]))
                                   / np.max(np.abs(st32[ok]))))
    return out


# ---------------------------------------------------------------------------
# the fit on the pooled imbalance, with a block bootstrap
# ---------------------------------------------------------------------------
def _block_moments(X, y, y_unit, labels):
    """Per-block ``X'X``, ``X'y``, ``y'y``, ``sum (y_unit)^2`` and ``n``."""
    ids, inv = np.unique(labels, return_inverse=True)
    nb, p = ids.size, X.shape[1]
    XtX = np.zeros((nb, p, p))
    Xty = np.zeros((nb, p))
    for a in range(p):
        Xty[:, a] = np.bincount(inv, weights=X[:, a] * y, minlength=nb)
        for b in range(a, p):
            v = np.bincount(inv, weights=X[:, a] * X[:, b], minlength=nb)
            XtX[:, a, b] = XtX[:, b, a] = v
    yty = np.bincount(inv, weights=y * y, minlength=nb)
    uu = np.bincount(inv, weights=y_unit * y_unit, minlength=nb)
    n = np.bincount(inv, minlength=nb).astype('float64')
    return XtX, Xty, yty, uu, n


def fit_multipliers(I, comps: dict, labels, n_boot=N_BOOT, seed=0, ci=(2.5, 97.5)) -> dict:
    """``I ~ c + sum_k a_k S_k`` by OLS, with the rms of the fitted residual
    against the unit-sum residual ``I - sum_k S_k`` and ``I`` itself; the
    multipliers and the rms gain carry a block-bootstrap interval from the
    per-block sufficient statistics (one draw = one matrix solve)."""
    names = list(comps)
    S = np.column_stack([comps[k] for k in names])
    ok = np.isfinite(I) & np.all(np.isfinite(S), axis=1)
    I, S, lab = I[ok], S[ok], np.asarray(labels)[ok]
    X = np.column_stack([np.ones(I.size), S])
    unit = I - S.sum(axis=1)
    XtX, Xty, yty, uu, nb_n = _block_moments(X, I, unit, lab)
    A, B = XtX.sum(0), Xty.sum(0)
    a = np.linalg.solve(A, B)
    n = float(I.size)
    rss_fit = float(yty.sum() - a @ B)
    rms = dict(I=float(np.sqrt(yty.sum() / n)), unit=float(np.sqrt(uu.sum() / n)),
               fit=float(np.sqrt(max(rss_fit, 0.0) / n)))
    nb = XtX.shape[0]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=int(n_boot)).astype('float64')
    XtX_r = np.einsum('rb,bij->rij', counts, XtX)
    Xty_r = counts @ Xty
    yty_r, uu_r, n_r = counts @ yty, counts @ uu, counts @ nb_n
    a_r = np.linalg.solve(XtX_r, Xty_r[..., None])[..., 0]
    rss_r = np.maximum(yty_r - np.einsum('ri,ri->r', a_r, Xty_r), 0.0)
    rms_fit_r, rms_unit_r = np.sqrt(rss_r / n_r), np.sqrt(uu_r / n_r)
    gain_r = rms_unit_r - rms_fit_r
    lo, hi = np.nanpercentile(gain_r, ci)
    out = dict(n=int(n), n_blocks=int(nb), n_boot=int(n_boot), intercept=float(a[0]),
               multipliers={k: dict(value=float(a[i + 1]),
                                    lo=float(np.nanpercentile(a_r[:, i + 1], ci[0])),
                                    hi=float(np.nanpercentile(a_r[:, i + 1], ci[1])),
                                    se=float(np.nanstd(a_r[:, i + 1])))
                            for i, k in enumerate(names)},
               rms=rms, rms_over_I={k: v / rms['I'] for k, v in rms.items()},
               gain=dict(value=rms['unit'] - rms['fit'], lo=float(lo), hi=float(hi),
                         half_width=float(0.5 * (hi - lo)), se=float(np.nanstd(gain_r)),
                         note='rms(I - S) - rms(I - sum a_k S_k): positive means the fit beats '
                              'the unit sum'),
               corr_with_I={k: mc._fit(comps[k][ok], I)['corr'] for k in names})
    return out


def cancellation(fit) -> dict:
    """The declared rule on one fit."""
    a = {k: v['value'] for k, v in fit['multipliers'].items()}
    in_bounds = {k: bool(BOUNDS[0] <= v <= BOUNDS[1]) for k, v in a.items()}
    gain_ok = bool(fit['gain']['value'] > fit['gain']['half_width'])
    return dict(gain_exceeds_half_width=gain_ok, multipliers_in_bounds=in_bounds,
                passed=bool(gain_ok and all(in_bounds.values())),
                rule=DECLARED['split']['rule'])


# ---------------------------------------------------------------------------
# (a) over the stores
# ---------------------------------------------------------------------------
def _pool_init():
    return {k: [] for k in ('I', 'meas', 'S', 'S_kpp', 'lst', 'pair', 'block') + tuple(
        f'{d}_{c}' for d in DEPTHS for c in COMP)}


def run_split(ds, grid_ds, grid, Ls=L_ALL, pairs=None, n_boot=N_BOOT, say=print) -> dict:
    drF = inp.drF(ds)
    n = ds.sizes['time'] - 1
    pairs = list(range(n)) if pairs is None else list(pairs)
    stores = {L: xr.open_zarr(bg.derived_path(L)) for L in Ls}
    offsets = {L: float(stores[L].attrs.get('utc_offset_h', inp.UTC_OFFSET_H)) for L in Ls}
    pools = {L: _pool_init() for L in Ls}
    ident = {L: [] for L in Ls}
    walls = []
    t_all = time.time()
    for t0 in pairs:
        t = time.time()
        h0, h1 = inp.hour_pair(ds, t0)
        for L in Ls:
            d = stores[L].isel(time=t0).load()
            valid = np.asarray(d['valid'].values, bool)
            front = np.asarray(d['front'].values, bool)
            sel = valid & front
            sp = split_components(h0, h1, L, grid_ds, grid, drF)
            chk = identity_check(sp, {k: d[STORED[k]].values for k in DEPTHS}, valid)
            chk['pair'] = int(t0)
            ident[L].append(chk)
            if not all(chk[k]['split_ok'] for k in DEPTHS):
                raise AssertionError(f'split identity broken at pair {t0}, L = {L}: {chk}')
            f = lambda k: np.asarray(d[k].values, 'float64')[sel]           # noqa: E731
            I = f('DGDt_semilag') - f('two_F') - f('subfilter') - f('vertical')
            P = pools[L]
            P['I'].append(I)
            P['meas'].append(f('DGDt_semilag'))
            P['S'].append(f('surface_flux'))
            P['S_kpp'].append(f('surface_flux_kpp'))
            for depth in DEPTHS:
                for c in COMP:
                    P[f'{depth}_{c}'].append(sp[depth][c][sel])
            lst = float(inp.local_solar_hour(d['time_mid'].values, offsets[L]))
            P['lst'].append(np.full(int(sel.sum()), lst))
            P['pair'].append(np.full(int(sel.sum()), t0, dtype='int16'))
            P['block'].append(st.space_time_block_ids(valid.shape, t0, hours=CI_HOURS)[sel])
        walls.append(time.time() - t)
        if t0 % 10 == 0 or t0 == pairs[-1]:
            say(f'  split pair {t0:2d}: {walls[-1]:.1f} s/pair ({len(Ls)} L); identity max rel '
                f'{max(ident[L][-1][k]["split_rel_rms"] for L in Ls for k in DEPTHS):.1e}; '
                f'vs store max rel {max(ident[L][-1][k]["vs_store_rel_rms"] for L in Ls for k in DEPTHS):.1e}')
    for s in stores.values():
        s.close()
    out = dict(n_pairs=len(pairs), wall_s=round(time.time() - t_all, 1),
               wall_per_pair=dict(median=float(np.median(walls)), max=float(np.max(walls))),
               per_L={})
    for L in Ls:
        P = {k: np.concatenate(v) for k, v in pools[L].items()}
        out['per_L'][str(L)] = analyse_split(P, L, ident[L], n_boot=n_boot, say=say)
    return out


def analyse_split(P, L, ident, n_boot=N_BOOT, say=print) -> dict:
    I, meas, lab = P['I'], P['meas'], P['block']
    r_m = mc._rms(meas)
    out = dict(L=int(L), n=int(I.size),
               identity=dict(
                   split_max_rel_rms={d: max(c[d]['split_rel_rms'] for c in ident) for d in DEPTHS},
                   vs_store_max_rel_rms={d: max(c[d]['vs_store_rel_rms'] for c in ident) for d in DEPTHS},
                   vs_store_max_rel_cell={d: max(c[d]['vs_store_max_rel'] for c in ident) for d in DEPTHS},
                   all_ok=bool(all(c[d]['split_ok'] for c in ident for d in DEPTHS))),
               rms_measured=r_m, imbalance_over_meas=mc._rms(I) / r_m,
               residual_over_meas=mc._rms(I - P['S']) / r_m,
               residual_kpp_over_meas=mc._rms(I - P['S_kpp']) / r_m,
               depth={})
    hrs = np.floor(P['lst']).astype(int) % 24
    for depth in DEPTHS:
        comps = {c: P[f'{depth}_{c}'] for c in COMP}
        Ssum = sum(comps.values())
        stored = P['S'] if depth == 'top' else P['S_kpp']
        d = dict(stored_term=STORED[depth],
                 sum_vs_stored_rel_rms=float(mc._rms(Ssum - stored) / mc._rms(stored)),
                 components={c: dict(rms_over_meas=mc._rms(v) / r_m,
                                     rms_over_S=mc._rms(v) / mc._rms(Ssum),
                                     corr_with_I=mc._fit(v, I)['corr'],
                                     corr_with_S=mc._fit(v, Ssum)['corr'],
                                     mean_over_rms_meas=float(np.nanmean(v) / r_m))
                             for c, v in comps.items()},
                 S=dict(rms_over_meas=mc._rms(Ssum) / r_m, corr_with_I=mc._fit(Ssum, I)['corr'],
                        single_multiplier=mc._fit(Ssum, I)['ols']),
                 fit3=fit_multipliers(I, comps, lab, n_boot=n_boot),
                 fit2=fit_multipliers(I, dict(heat=comps['nonsolar'] + comps['shortwave'],
                                              freshwater=comps['freshwater']), lab, n_boot=n_boot),
                 fit1=fit_multipliers(I, dict(S=Ssum), lab, n_boot=n_boot))
        d['cancellation'] = cancellation(d['fit3'])
        d['by_local_hour'] = {int(h): dict(
            n=int((hrs == h).sum()),
            I_mean_over_rms_meas=float(np.nanmean(I[hrs == h]) / r_m),
            S_mean_over_rms_meas=float(np.nanmean(Ssum[hrs == h]) / r_m),
            **{f'{c}_mean_over_rms_meas': float(np.nanmean(v[hrs == h]) / r_m) for c, v in comps.items()},
            **{f'{c}_rms_over_meas': mc._rms(v[hrs == h]) / mc._rms(meas[hrs == h]) for c, v in comps.items()})
            for h in np.unique(hrs)}
        out['depth'][depth] = d
        f3 = d['fit3']
        say(f'  L={L} {depth:3s}: S/meas {d["S"]["rms_over_meas"]:.3f} corr(S,I) {d["S"]["corr_with_I"]:+.3f} | '
            + ' '.join(f'{c[:2]} {d["components"][c]["rms_over_meas"]:.3f}/{d["components"][c]["corr_with_I"]:+.2f}'
                       for c in COMP)
            + f' | a = ' + ', '.join(f'{f3["multipliers"][c]["value"]:+.2f}' for c in COMP)
            + f' | rms I {f3["rms_over_I"]["I"]:.3f} unit {f3["rms_over_I"]["unit"]:.3f} fit '
              f'{f3["rms_over_I"]["fit"]:.3f} gain {f3["gain"]["value"] / mc._rms(I):+.4f} '
              f'(hw {f3["gain"]["half_width"] / mc._rms(I):.4f}) -> '
            + ('CANCELLATION' if d['cancellation']['passed'] else 'orthogonality stands'))
    return out


# ---------------------------------------------------------------------------
# (b) V from the second cell
# ---------------------------------------------------------------------------
def vertical_variants(h0, h1, L, grid_ds, grid, drF) -> dict:
    """The four vertical terms on one pair at one ``L``, the budget's ``2 x``
    convention, numpy ``(j, i)`` float64."""
    f0, f1 = inp.filtered(h0, L), inp.filtered(h1, L)
    b_mid = inp.midpoint(f0[0], f1[0])
    b_x, b_y = op.grad_b(b_mid, grid_ds, grid)
    bl = inp.midpoint(vt.buoyancy_levels(h0['Theta_k'], h0['Salt_k']),
                      vt.buoyancy_levels(h1['Theta_k'], h1['Salt_k']))
    b0, b1, b2 = (bl.isel(k=k, drop=True) for k in range(3))
    W1 = inp.midpoint(inp.W_k1(h0), inp.W_k1(h1))
    W2 = inp.midpoint(inp.W_k2(h0), inp.W_k2(h1))
    V_top = 2.0 * vt.vertical_term(b0, b_x, b_y, b1, W1, drF, grid_ds, grid, L_cells=L)
    bz1 = bg._b_z_mid(bl, drF)
    V_f1 = 2.0 * vt.vertical_term_factorised(b_x, b_y, bz1, W1, grid_ds, grid, L_cells=L)
    bz2 = inp.midpoint(vt.b_z(h0['Theta_k'], h0['Salt_k'], grid_ds, drF, order=2),
                       vt.b_z(h1['Theta_k'], h1['Salt_k'], grid_ds, drF, order=2))
    V_f2 = 2.0 * vt.vertical_term_factorised(b_x, b_y, bz2, W1, grid_ds, grid, L_cells=L)
    b1f = op.lowpass(b1, L)
    b1x, b1y = op.grad_b(b1f, grid_ds, grid)
    V_c2 = 2.0 * vt.vertical_term(b1, b1x, b1y, b2, W2, drF[1:], grid_ds, grid, L_cells=L)
    sq = (lambda a: np.squeeze(np.asarray(a.values, 'float64')))
    return dict(V_top=sq(V_top), V_fact_o1=sq(V_f1), V_fact_o2=sq(V_f2), V_cell2=sq(V_c2),
                dz_top_m=float(V_top.attrs['dz_m']), dz_cell2_m=float(V_c2.attrs['dz_m']))


def run_vertical(ds, grid_ds, grid, pairs=SMOKE_PAIRS, Ls=V_LS, say=print) -> dict:
    drF = inp.drF(ds)
    out = dict(pairs=list(pairs), Ls=list(Ls), rows=[])
    t_all = time.time()
    for L in Ls:
        der = xr.open_zarr(bg.derived_path(L))
        offset = float(der.attrs.get('utc_offset_h', inp.UTC_OFFSET_H))
        for t0 in pairs:
            t = time.time()
            h0, h1 = inp.hour_pair(ds, t0)
            d = der.isel(time=t0).load()
            sel = np.asarray(d['valid'].values, bool) & np.asarray(d['front'].values, bool)
            v = vertical_variants(h0, h1, L, grid_ds, grid, drF)
            meas = np.asarray(d['DGDt_semilag'].values, 'float64')[sel]
            r_m = mc._rms(meas)
            stored = {k: np.asarray(d[s].values, 'float64') for k, s in
                      (('V_top', 'vertical'), ('V_fact_o1', 'vertical_factorised'))}
            ok = sel & np.isfinite(v['V_top']) & np.isfinite(stored['V_top'])
            row = dict(L=int(L), pair=int(t0), lst=float(inp.local_solar_hour(d['time_mid'].values, offset)),
                       n=int(sel.sum()), rms_measured=r_m,
                       vs_store={k: float(mc._rms(v[k][ok].astype('float32') - stored[k][ok])
                                          / mc._rms(stored[k][ok])) for k in stored},
                       dz_top_m=v['dz_top_m'], dz_cell2_m=v['dz_cell2_m'], terms={})
            for k in ('V_top', 'V_fact_o1', 'V_fact_o2', 'V_cell2'):
                a = v[k][sel]
                row['terms'][k] = dict(rms_over_meas=mc._rms(a) / r_m,
                                       mean_over_rms_meas=float(np.nanmean(a) / r_m),
                                       sign_of_mean=int(np.sign(np.nanmean(a))),
                                       n_finite=int(np.isfinite(a).sum()))
            row['cell2_on_top'] = mc._fit(v['V_top'][sel], v['V_cell2'][sel])
            row['fact_o2_on_o1'] = mc._fit(v['V_fact_o1'][sel], v['V_fact_o2'][sel])
            row['cell2_over_top_rms'] = row['terms']['V_cell2']['rms_over_meas'] / row['terms']['V_top']['rms_over_meas']
            row['same_sign_cell2_top'] = bool(row['terms']['V_cell2']['sign_of_mean']
                                              == row['terms']['V_top']['sign_of_mean'])
            row['wall_s'] = round(time.time() - t, 1)
            out['rows'].append(row)
            say(f'  V L={L} pair {t0:2d} ({row["lst"]:4.1f} LST): top {row["terms"]["V_top"]["rms_over_meas"]:.4f} '
                f'fact_o1 {row["terms"]["V_fact_o1"]["rms_over_meas"]:.4f} fact_o2 '
                f'{row["terms"]["V_fact_o2"]["rms_over_meas"]:.4f} cell2 {row["terms"]["V_cell2"]["rms_over_meas"]:.4f} '
                f'| cell2~top slope {row["cell2_on_top"]["ols"]:+.3f} corr {row["cell2_on_top"]["corr"]:+.3f} '
                f'| means top {row["terms"]["V_top"]["mean_over_rms_meas"]:+.4f} cell2 '
                f'{row["terms"]["V_cell2"]["mean_over_rms_meas"]:+.4f} | vs store {row["vs_store"]["V_top"]:.1e} '
                f'({row["wall_s"]} s)')
        der.close()
    rows = out['rows']
    out['summary'] = dict(
        vs_store_max_rel_rms=max(r['vs_store']['V_top'] for r in rows),
        cell2_on_top_slope=dict(min=min(r['cell2_on_top']['ols'] for r in rows),
                                max=max(r['cell2_on_top']['ols'] for r in rows)),
        cell2_on_top_corr=dict(min=min(r['cell2_on_top']['corr'] for r in rows),
                               max=max(r['cell2_on_top']['corr'] for r in rows)),
        cell2_over_top_rms=dict(min=min(r['cell2_over_top_rms'] for r in rows),
                                max=max(r['cell2_over_top_rms'] for r in rows)),
        same_sign=sum(r['same_sign_cell2_top'] for r in rows), n_rows=len(rows),
        fact_o2_on_o1_slope=dict(min=min(r['fact_o2_on_o1']['ols'] for r in rows),
                                 max=max(r['fact_o2_on_o1']['ols'] for r in rows)),
        wall_s=round(time.time() - t_all, 1))
    return out


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def run(Ls=L_ALL, pairs=None, n_boot=N_BOOT, do_split=True, do_vertical=True, say=print) -> dict:
    ds, grid_ds, grid, masks = inp.open_inputs()
    out = dict(declared=DECLARED, created=time.strftime('%Y-%m-%dT%H:%M:%S'),
               stores={str(L): str(bg.derived_path(L)) for L in Ls})
    if do_vertical:
        say('--- (b) V from the second cell')
        out['vertical'] = run_vertical(ds, grid_ds, grid, say=say)
    if do_split:
        say('--- (a) the split')
        out['split'] = run_split(ds, grid_ds, grid, Ls=Ls, pairs=pairs, n_boot=n_boot, say=say)
        out['cancellation_summary'] = {L: {d: out['split']['per_L'][L]['depth'][d]['cancellation']['passed']
                                           for d in DEPTHS} for L in out['split']['per_L']}
    out['verdict_note'] = ('term tests only: the M3 verdict (criterion 1 failed at every L) is '
                           'unchanged whatever these say')
    return out


def format_table(res) -> str:
    w = '=' * 110
    out = [w, 'M3 DIGGING TASK 3 -- THE TERM TESTS', w]
    if 'split' in res:
        out.append('  (a) S split: component rms / measured (corr with I)   |   fit3 multipliers   |   rms/I: I, unit, fit  gain(hw)  verdict')
        for L, d in res['split']['per_L'].items():
            for depth in DEPTHS:
                dd = d['depth'][depth]
                f3 = dd['fit3']
                rI = f3['rms']['I']
                out.append(
                    f'  L={L} {depth:3s} S {dd["S"]["rms_over_meas"]:.3f}({dd["S"]["corr_with_I"]:+.2f}) '
                    + ' '.join(f'{c[:3]} {dd["components"][c]["rms_over_meas"]:.3f}({dd["components"][c]["corr_with_I"]:+.2f})'
                               for c in COMP)
                    + ' | ' + ' '.join(f'{f3["multipliers"][c]["value"]:+.2f}[{f3["multipliers"][c]["lo"]:+.2f},{f3["multipliers"][c]["hi"]:+.2f}]'
                                       for c in COMP)
                    + f' | {f3["rms_over_I"]["I"]:.3f} {f3["rms_over_I"]["unit"]:.3f} {f3["rms_over_I"]["fit"]:.3f} '
                      f'{f3["gain"]["value"] / rI:+.4f}({f3["gain"]["half_width"] / rI:.4f}) '
                    + ('CANCELLATION' if dd['cancellation']['passed'] else 'orthogonality stands'))
            idn = d['identity']
            out.append(f'       identity: split max rel {max(idn["split_max_rel_rms"].values()):.1e}; '
                       f'vs store max rel rms {max(idn["vs_store_max_rel_rms"].values()):.1e}')
    if 'vertical' in res:
        out += ['', '  (b) V from the second cell: rms / measured; cell2 ~ top slope (corr); sign of pool mean top/cell2']
        for r in res['vertical']['rows']:
            t = r['terms']
            out.append(f'  L={r["L"]} pair {r["pair"]:2d} {r["lst"]:4.1f} LST  top {t["V_top"]["rms_over_meas"]:.4f} '
                       f'fact_o1 {t["V_fact_o1"]["rms_over_meas"]:.4f} fact_o2 {t["V_fact_o2"]["rms_over_meas"]:.4f} '
                       f'cell2 {t["V_cell2"]["rms_over_meas"]:.4f}  {r["cell2_on_top"]["ols"]:+.3f} '
                       f'({r["cell2_on_top"]["corr"]:+.3f})  {t["V_top"]["sign_of_mean"]:+d}/{t["V_cell2"]["sign_of_mean"]:+d}')
        s = res['vertical']['summary']
        out.append(f'       same sign {s["same_sign"]}/{s["n_rows"]}; slope {s["cell2_on_top_slope"]["min"]:+.3f}..'
                   f'{s["cell2_on_top_slope"]["max"]:+.3f}; corr {s["cell2_on_top_corr"]["min"]:+.3f}..'
                   f'{s["cell2_on_top_corr"]["max"]:+.3f}; vs store {s["vs_store_max_rel_rms"]:.1e}')
    out += ['', '  ' + res.get('verdict_note', ''), w]
    return '\n'.join(out)


def write_declared(out_path):
    Path(out_path).write_text(json.dumps(dict(
        declared=DECLARED, status='declared; stores not yet opened',
        created=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--L', default=','.join(str(L) for L in L_ALL),
                    type=lambda s: [int(x) for x in s.split(',') if x])
    ap.add_argument('--pairs', default=None, help='a:b slice of pairs for a trial run')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--no-split', action='store_true')
    ap.add_argument('--no-vertical', action='store_true')
    ap.add_argument('--out', default=str(OUT_JSON))
    ap.add_argument('--no-fig', action='store_true')
    args = ap.parse_args(argv)
    pairs = None
    if args.pairs:
        a, _, b = args.pairs.partition(':')
        pairs = list(range(int(a or 0), int(b or 71)))
        print(f'NOTE: pairs {pairs[0]}..{pairs[-1]} only -- a trial run, not the record')
    write_declared(args.out)
    print(f'declared -> {args.out}')
    res = run(args.L, pairs=pairs, n_boot=args.n_boot, do_split=not args.no_split,
              do_vertical=not args.no_vertical)
    res['trial'] = bool(pairs is not None or args.n_boot != N_BOOT)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f'wrote {args.out}')
    print(format_table(res))
    if not args.no_fig:
        from m3_term_tests_fig import figure
        figure(res, OUT_PNG)
        print(f'wrote {OUT_PNG}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
