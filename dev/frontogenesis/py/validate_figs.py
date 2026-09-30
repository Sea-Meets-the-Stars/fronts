""" The figures of the M1 gates and supporting checks (coding doc §4.9):
one ``fig_V*`` per PNG, each taking the numbers ``validate.py`` computed
and writing ``dev/frontogenesis/figs/V*_*.png``.  No numbers are computed
here; nothing here is imported by the science path.

Maps are drawn with ``pcolormesh(XC, YC, ...)`` so they come out north-up,
east-right on the rotated face 10 (``i`` runs south, ``j`` runs east).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm   # noqa: E402
from matplotlib.patches import Rectangle                     # noqa: E402

from m0_qa_plot import draw_map, edge_profile                # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / 'figs'
COL = {'order1': '#f4a259', 'order3': '#1f5fa8', 'order5': '#2a9d8f', 'red': '#c8102e',
       'black': 'black', 'grey': '#888888', 'purple': '#5b2a86'}


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# V1: Cartesian deformation
# ---------------------------------------------------------------------------
def fig_V1(res):
    """V1: (a) ``G`` along parcels over ``n_steps`` hours vs ``exp(2 a t)``
    at the reference width; (b) the per-step growth rate over ``2a`` against
    the front width, with the ``(dx/ell)^2`` convergence; (c) the
    semi-Lagrangian step alone (interpolation + departure) against the
    discrete stencil at the exact departure point; (d) the rate across the
    front for a few widths."""
    a, dt = res['alpha'], res['dt']
    w = np.array(res['widths_cells'])
    ref = res['ref_ell_cells']
    fig, ((pa, pb), (pc, pd)) = plt.subplots(2, 2, figsize=(15, 11))
    # (a) the series
    t_h = np.array(res['series_t_hours'])
    for key, col, ls in (('CS=1', COL['order3'], '-'), ('face10', COL['red'], '--')):
        s = res['series'][key]
        pa.plot(t_h, np.array(s['median']), ls, color=col, lw=2,
                label=f'{key}: median over front parcels (n = {s["n"]})')
        pa.fill_between(t_h, np.array(s['min']), np.array(s['max']), color=col, alpha=0.15,
                        label=f'{key}: min-max over front parcels')
    pa.plot(t_h, np.exp(2 * a * t_h * 3600.0), ':', color='black', lw=2, label='exact  exp(2 a t)')
    pa.set_xlabel('t [hours]  (semi-Lagrangian steps of dt = 1 h chained backwards)')
    pa.set_ylabel('G(t) / G(0) along the parcel')
    pa.set_title(f'(a) G along parcels, front ell = {ref:g} dx (tanh; sigma_G ~ {0.45 * ref:.1f} cells), '
                 f'a = {a:g} s$^{{-1}}$\nmax |G/G0 / exp(2at) - 1| over {res["n_steps"]} h: '
                 f'{100 * res["series_max_err_ref"]:.2f}%  (gate < 1%)', fontsize=10)
    pa.legend(fontsize=8, loc='upper left'); pa.grid(alpha=0.3)
    # (b) the rate vs width
    for key, col, mk_ in (('CS=1', COL['order3'], 'o'), ('face10', COL['red'], 's')):
        r = res['rate'][key]
        med = np.array(r['median']); lo = np.array(r['min']); hi = np.array(r['max'])
        pb.errorbar(w, med, yerr=[med - lo, hi - med], fmt=mk_ + '-', color=col, ms=5, lw=1.4,
                    capsize=3, label=f'{key}: median [min, max] over front pixels')
    pb.axhline(1.0, color='black', lw=1)
    pb.axhspan(0.99, 1.01, color='#e6f0fa', label='+/- 1%')
    pb.axvline(ref, color=COL['grey'], ls='--', lw=1)
    pb.set_xscale('log'); pb.set_xticks(w); pb.set_xticklabels([f'{x:g}' for x in w])
    pb.set_xlabel('front width ell / dx  (b = b0 tanh(x/ell))')
    pb.set_ylabel('measured growth rate  ln[G(x,t+dt)/G(x_d,t)] / (2 a dt)')
    pb.set_title('(b) growth rate over 2a vs front width, one 1 h step\n'
                 f'rms error: {", ".join(f"{x:g}dx {100 * e:.2f}%" for x, e in zip(w, res["rate"]["CS=1"]["rms_err"]))}',
                 fontsize=9)
    pb.legend(fontsize=8, loc='lower right'); pb.grid(alpha=0.3, which='both')
    # (c) convergence and the semilag-only error
    rms = np.array(res['rate']['CS=1']['rms_err'])
    pc.plot(w, 100 * rms, 'o-', color=COL['black'], lw=1.6, ms=5,
            label=f'rms rate error, all of it (order {res["convergence_order"]:.2f} in dx/ell)')
    pc.plot(w, 100 * rms[w == 4][0] * (4.0 / w) ** 2, ':', color=COL['grey'], lw=1.4,
            label='(dx/ell)$^2$ through 4 dx')
    for order, key in ((3, 'order3'), (5, 'order5')):
        e = np.array(res['semilag_err_max'][f'order{order}'])
        pc.plot(w, 100 * e, 'D-', color=COL[key], lw=1.4, ms=4,
                label=f'semi-Lagrangian step alone, order {order}: max |G_d / G_d,ref - 1|')
    pc.axhline(1.0, color=COL['red'], lw=1, ls='--', label='1% gate')
    pc.set_xscale('log'); pc.set_yscale('log')
    pc.set_xticks(w); pc.set_xticklabels([f'{x:g}' for x in w])
    pc.set_xlabel('front width ell / dx')
    pc.set_ylabel('error [%]')
    pc.set_title('(c) what the error is: the centred stencil\'s truncation, second order in dx/ell;\n'
                 'the departure + interpolation (G_d vs the same stencil at the exact x_d) is far below it',
                 fontsize=9)
    pc.legend(fontsize=7.5, loc='lower left'); pc.grid(alpha=0.3, which='both')
    # (d) the rate across the front
    prof = res['profiles']
    cols = (COL['red'], COL['order1'], COL['order3'], COL['order5'], COL['purple'], COL['grey'])
    for x, col in zip(prof['widths'], cols):
        pd.plot(np.array(prof['x_over_ell'][str(float(x))]), np.array(prof['rate'][str(float(x))]), '-',
                color=col, lw=1.6, label=f'ell = {x:g} dx')
    pd.axhline(1.0, color='black', lw=1); pd.axhspan(0.99, 1.01, color='#e6f0fa')
    pd.set_xlabel('x / ell  (front pixels: G > 0.2 max)')
    pd.set_ylabel('growth rate / 2a')
    pd.set_title('(d) the rate across the front (CS = 1): the deficit is largest on the flanks,\n'
                 'where the discrete gradient of tanh is attenuated most as the front sharpens', fontsize=9)
    pd.legend(fontsize=8); pd.grid(alpha=0.3)
    fig.suptitle('V1: pure deformation u = -a x, v = a y -- G grows as exp(2 a t); operators.gradb2 + '
                 'semilag (departure_index, gradb2_at_departure) on the exact solution', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, 'V1_cartesian_deformation.png')


# ---------------------------------------------------------------------------
# V2: native-grid metric
# ---------------------------------------------------------------------------
def fig_V2(res, ctx):
    """V2 from the numbers of ``validate.test_native_metric`` and the arrays
    in ``ctx`` (``X, Y, f, err_x, err_y, ana, oc, err_lin_x, err_lin_y``):
    (a) the analytic field on the tile; (b) the ``b_x`` error map; (c) the
    error distributions (sinusoid and linear function); (d) the error
    against latitude with the local truncation prediction."""
    X, Y, f, ana, oc = ctx['X'], ctx['Y'], ctx['f'], ctx['ana'], ctx['oc']
    err_x, err_y, err_lin_x, err_lin_y = ctx['err_x'], ctx['err_y'], ctx['err_lin_x'], ctx['err_lin_y']
    fig, ((pa, pb), (pc, pd)) = plt.subplots(2, 2, figsize=(15, 12))
    m = draw_map(pa, X, Y, f, 'RdBu_r', land=~oc)
    fig.colorbar(m, ax=pa, shrink=0.8, label='f')
    pa.set_title(f'(a) f = sin(2 pi (lon - lon0)/{res["L_lon_deg"]:g} deg) cos(2 pi (lat - lat0)/'
                 f'{res["L_lat_deg"]:g} deg)\n~{res["L_lon_cells"]:.0f} x {res["L_lat_cells"]:.0f} '
                 'cells; grey = land', fontsize=10)
    vmax = 100 * res['err_x_max']
    m = draw_map(pb, X, Y, 100 * np.where(ana, err_x, np.nan), 'RdBu_r', land=~oc,
                 vmin=-vmax, vmax=vmax)
    fig.colorbar(m, ax=pb, shrink=0.8, label='(b_x - f_east) / max |grad f|  [%]')
    pb.set_title(f'(b) b_x error on mask_analysis (n = {res["n_analysis"]:,}): rms {100 * res["err_x_rms"]:.3f}%, '
                 f'max {100 * res["err_x_max"]:.3f}%\nworst at {res["worst_x"]["lat"]:.2f}N '
                 f'{-res["worst_x"]["lon"]:.2f}W; b_y: rms {100 * res["err_y_rms"]:.3f}%, max '
                 f'{100 * res["err_y_max"]:.3f}%', fontsize=10)
    bins = np.linspace(-vmax, vmax, 81)
    pc.hist(100 * err_x[ana], bins=bins, color=COL['order3'], alpha=0.7, label='b_x, sinusoid')
    pc.hist(100 * err_y[ana], bins=bins, color=COL['red'], alpha=0.5, label='b_y, sinusoid')
    pc.hist(100 * err_lin_x[ana], bins=bins, color=COL['order5'], alpha=0.7, histtype='step', lw=1.6,
            label=f'b_x, linear f (metric only): max {100 * res["lin_err_x_max"]:.4f}%')
    pc.hist(100 * err_lin_y[ana], bins=bins, color=COL['purple'], alpha=0.7, histtype='step', lw=1.6,
            label=f'b_y, linear f (metric only): max {100 * res["lin_err_y_max"]:.4f}%')
    pc.axvline(100 * res['truncation_x'], color='black', ls=':', lw=1.2,
               label=f'(k dx)^2/6 truncation: {100 * res["truncation_x"]:.3f}% (x), {100 * res["truncation_y"]:.3f}% (y)')
    pc.axvline(-100 * res['truncation_x'], color='black', ls=':', lw=1.2)
    pc.set_yscale('log')
    pc.set_xlabel('error / max |grad f| on mask_analysis  [%]')
    pc.set_ylabel('cells')
    pc.set_title('(c) error distributions: the sinusoid carries the stencil truncation, the linear\n'
                 f'function only the metric (dxC, dyC, CS, SN); components swapped would be '
                 f'{100 * res["err_if_components_swapped"]:.0f}%', fontsize=10)
    pc.legend(fontsize=7.5); pc.grid(alpha=0.3, axis='y')
    lc = np.array(res['lat_bins'])
    for key, col, lab in (('x', COL['order3'], 'b_x'), ('y', COL['red'], 'b_y')):
        pd.plot(lc, 100 * np.array(res[f'rms_{key}_vs_lat']), 'o-', color=col, ms=4, lw=1.4,
                label=f'{lab}: measured rms')
        pd.plot(lc, 100 * np.array(res[f'truncation_{key}_vs_lat']), ':', color=col, lw=1.6,
                label=f'{lab}: -(theta^2/6) f\' with the local phase advance per cell (rms)')
    pd.set_xlabel('latitude'); pd.set_ylabel('rms error / max |grad f|  [%]')
    pd.set_title(f'(d) error vs latitude: it tracks the local phase advance per cell, as truncation should\n'
                 f'(dxC / dyC {res["dxC_km_north"]:.2f} / {res["dyC_km_north"]:.2f} km at the north end, '
                 f'{res["dxC_km_south"]:.2f} / {res["dyC_km_south"]:.2f} km at the south); a wrong cos(lat) '
                 f'would grow poleward (linear f: {100 * res["lin_err_x_max"]:.3f}%)', fontsize=9)
    pd.legend(fontsize=8); pd.grid(alpha=0.3)
    fig.suptitle(f'V2: analytic f(XC, YC) through operators.grad_b on the tile 330 grid (face 10, CS = 0, SN = -1): '
                 f'max error {100 * max(res["err_x_max"], res["err_y_max"]):.3f}% < 1% on mask_analysis',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, 'V2_native_metric.png')


# ---------------------------------------------------------------------------
# V3: the discrete null
# ---------------------------------------------------------------------------
def _null_panel(ax, pool, fit, title, gate=False, verdict=None):
    """A 2-D histogram of measured vs 2F on front pixels, the 1:1 line and
    the gate OLS fit, in units of a power of ten of s^-5.  ``verdict``
    replaces the PASS/FAIL text (V3b: a recorded bias, not a gate)."""
    x, y = pool['x'], pool['y']
    sc = 10.0 ** np.floor(np.log10(np.percentile(np.abs(x), 99)))
    xs, ys = x / sc, y / sc
    lim = 1.15 * np.percentile(np.abs(np.concatenate([xs, ys])), 99.5)
    ax.hexbin(xs, ys, gridsize=70, bins='log', cmap='Blues', extent=(-lim, lim, -lim, lim), mincnt=1)
    t = np.array([-lim, lim])
    ax.plot(t, t, '-', color='black', lw=1.0, label='1:1')
    ax.plot(t, fit['ols'] * t + fit['intercept'] / sc, '--', color=COL['red'], lw=1.6,
            label=f'OLS (gate): {fit["ols"]:.3f} [{fit["bootstrap"]["ci"][0]:.3f}, {fit["bootstrap"]["ci"][1]:.3f}]')
    ax.plot(t, fit['orthogonal'] * t, ':', color=COL['purple'], lw=1.2,
            label=f'orthogonal {fit["orthogonal"]:.3f}, inverse OLS {fit["ols_inverse"]:.3f}, GM {fit["geometric_mean"]:.3f}')
    ax.axhline(0, color='#888888', lw=0.5); ax.axvline(0, color='#888888', lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    e = int(np.log10(sc))
    ax.set_xlabel(f'2F at the midpoint  [1e{e} s$^{{-5}}$]')
    ax.set_ylabel(f'measured DG/Dt (semi-Lagrangian)  [1e{e} s$^{{-5}}$]')
    if verdict is None:
        verdict = ('PASS' if fit['gate']['passed'] else 'FAIL') + ' (1 +/- 0.05)'
    ax.set_title(f'{title}\nn = {fit["n"]:,}, corr {fit["corr"]:.3f}: {verdict}' + ('  [the gate]' if gate else ''),
                 fontsize=10, color=(COL['order5'] if fit['gate']['passed'] else COL['red']) if gate else 'black')
    ax.legend(fontsize=7.5, loc='upper left'); ax.grid(alpha=0.3)


def fig_V3(res_s, ctx_s, res_l, ctx_l):
    """V3: (a, b) the strain variant, first attempt (chain-rule F, bilinear
    departure velocity) and the gate (consistent F); (c, d) the same for
    the real-velocity variant; (e) the slope against the front width for
    every form, with the ``1 - (2/3)(dx/ell)^2`` prediction; (f) every
    change tried, both variants."""
    fig, axs = plt.subplots(2, 3, figsize=(20, 12.5))
    (pa, pc, pe), (pb, pd, pf) = axs
    gate = res_s['gate_form']

    def first(res, ctx):
        """The first attempt (chain-rule F, bilinear departure velocity) when
        it was recorded, else the chain form with the cubic velocity."""
        if 'first_attempt' in res and ctx.get('pool_first') is not None:
            return ctx['pool_first'], res['first_attempt'], 'bilinear departure velocity'
        f0 = res['forms'][0]
        return ctx['pool'][f0], res['fits'][f0], f'form = {f0}, cubic departure velocity'
    pool0, fit0, lab0 = first(res_s, ctx_s)
    _null_panel(pa, pool0, fit0, f'(a) strain variant, first attempt: chain-rule F ({lab0})\n'
                f'with the cubic departure velocity: {res_s["fits"]["chain"]["ols"]:.3f}')
    _null_panel(pb, ctx_s['pool'][gate], res_s['fits'][gate],
                f'(b) strain variant, consistent F (form = {gate}, cubic departure velocity)', gate=True)
    if res_l is not None:
        pool0, fit0, lab0 = first(res_l, ctx_l)
        _null_panel(pc, pool0, fit0, f'(c) LLC variant (hour-0 b, real midpoint velocity, mask_analysis), '
                    f'first attempt:\nchain-rule F ({lab0}); with the cubic velocity: '
                    f'{res_l["fits"]["chain"]["ols"]:.3f}')
        _null_panel(pd, ctx_l['pool'][gate], res_l['fits'][gate],
                    f'(d) LLC variant, consistent F (form = {gate}, cubic departure velocity)', gate=True)
    else:
        for ax in (pc, pd):
            ax.text(0.5, 0.5, 'LLC variant: M0 stores not on disk', ha='center', va='center', transform=ax.transAxes)
    # (e) slope vs front width
    w = np.array(res_s['widths_cells'])
    forms = res_s['forms']
    cols = {'chain': COL['red'], 'discrete_o2': COL['order1'], 'discrete': COL['order5']}
    labs = {'chain': 'chain-rule F (repo form)', 'discrete_o2': 'consistent F, 2nd-order neighbour gradient (tried)',
            'discrete': 'consistent F, 4th-order neighbour gradient [the default]'}
    for f in forms:
        s = np.array([res_s['per_width'][f][str(x)]['ols'] for x in w], dtype=float)
        pe.plot(w, s, 'o-', color=cols.get(f, 'black'), lw=1.6, ms=6, label=labs.get(f, f))
        if 'diag_widths' in res_s:
            wd = np.array(res_s['diag_widths']['widths'])
            sd = np.array([res_s['diag_widths']['per_width'][f][str(x)]['ols'] for x in wd], dtype=float)
            pe.plot(wd, sd, 'o', color=cols.get(f, 'black'), ms=6, mfc='none', mew=1.5)
    ww = np.linspace(1.0, 8.5, 200)
    pe.plot(ww, 1 - (2.0 / 3.0) / ww ** 2, ':', color='black', lw=1.4,
            label='1 - (2/3)(dx/ell)$^2$: the chain-rule violation at a tanh centre')
    alt = res_s['changes_tried'].get('4th-order gradient on both sides, chain rule (per width, angle 0)')
    if alt:
        pe.plot([float(k) for k in alt], list(alt.values()), 's--', color=COL['grey'], ms=5, lw=1.2,
                label='4th-order gradient on both sides, chain rule (tried; angle 0)')
    pe.axhspan(0.95, 1.05, color='#e6f0fa', label='gate: 1 +/- 0.05')
    pe.axhline(1, color='black', lw=0.8)
    if res_l is not None:
        for f in forms:
            pe.axhline(res_l['fits'][f]['ols'], color=cols.get(f, 'black'), lw=1.0, ls='-.', alpha=0.8,
                       label=f'LLC variant, {f}: {res_l["fits"][f]["ols"]:.3f}')
    pe.set_xscale('log'); pe.set_xticks([1, 1.5, 2, 3, 4, 6, 8]); pe.set_xticklabels(['1', '1.5', '2', '3', '4', '6', '8'])
    pe.set_xlabel('front width ell / dx  (tanh; sigma_G = ell/2; open symbols: out of the pool)')
    pe.set_ylabel('OLS slope, measured DG/Dt on 2F, front pixels')
    pe.set_ylim(0.75, 1.08)
    pe.set_title('(e) slope vs front width per form (strain; dash-dot: LLC)\n'
                 'the chain rule fails by (2/3)(dx/ell)$^2$; the consistent F removes it', fontsize=10)
    pe.legend(fontsize=7.5, loc='lower right'); pe.grid(alpha=0.3, which='both')
    # (f) every change tried
    rows = []
    for name, r in (('strain', res_s), ('LLC', res_l)):
        if r is None:
            continue
        for k, v in r['changes_tried'].items():
            if isinstance(v, dict):
                continue
            rows.append((f'{name}: {k}', v))
    yv = np.arange(len(rows))[::-1]
    vals = np.array([v for _, v in rows])
    cl = [COL['order5'] if abs(v - 1) <= 0.05 else COL['red'] for v in vals]
    pf.barh(yv, vals - 1, left=1, color=cl, alpha=0.8)
    pf.axvspan(0.95, 1.05, color='#e6f0fa'); pf.axvline(1, color='black', lw=0.8)
    pf.set_yticks(yv); pf.set_yticklabels([k for k, _ in rows], fontsize=7.5)
    for yy, v in zip(yv, vals):
        pf.text(v + (0.004 if v >= 1 else -0.004), yy, f'{v:.3f}', va='center', ha='left' if v >= 1 else 'right', fontsize=7.5)
    pf.set_xlim(0.7, 1.1)
    pf.set_xlabel('OLS slope on front pixels (green: within the gate)')
    seen = (f'{res_l["strain_seen"]["departure_vs_jacobian"]:.3f} / {res_l["strain_seen"]["jacobian_vs_fluxform"]:.3f} / '
            f'{res_l["strain_seen"]["departure_vs_fluxform"]:.3f}' if res_l is not None and 'strain_seen' in res_l else 'n/a')
    pf.set_title('(f) every change tried (first attempt: chain-rule F, bilinear velocity)\n'
                 f'LLC front pixels, departure / Jacobian / flux-form strain: {seen}\n'
                 '(both sides see D_h u_c: blind to the 0.85 Jacobian attenuation)', fontsize=9.5)
    pf.grid(alpha=0.3, axis='x')
    fig.suptitle('V3: the discrete null -- a tracer advected one hour by our own semi-Lagrangian step; '
                 'measured DG/Dt vs 2F at the midpoint on front pixels (G_mid >= p90; OLS, 32-cell block bootstrap)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, 'V3_discrete_null.png')


# ---------------------------------------------------------------------------
# V3b: the finite-volume null (recorded bias)
# ---------------------------------------------------------------------------
SCHEME_COL = {'semilag': COL['grey'], 'centred': COL['order3'], 'os7': COL['order5'], 'os7mp': COL['black'],
              'dst3': COL['order1']}
SCHEME_LAB = {'semilag': 'semi-Lagrangian (V3)', 'centred': 'FV centred 2nd order (stencil only)',
              'os7': 'FV OS7 (7th order, unlimited)', 'os7mp': 'FV OS7MP-like (7th order + MP limiter)',
              'dst3': 'FV DST3 (3rd order)'}


def fig_V3b(res_s, ctx_s, res_l, ctx_l):
    """V3b: (a) strain variant, OS7MP-like truth, discrete F; (b, c) LLC
    variant with the centred and the OS7MP-like truth; (d) slope vs front
    width per scheme (strain); (e) every slope with its CI, scheme x form
    x variant; (f) the attribution: stencil (centred - 1) vs implicit
    diffusion (scheme - centred), and the limiter (OS7MP - OS7)."""
    fig, axs = plt.subplots(2, 3, figsize=(20, 12.5))
    (pa, pc, pe), (pb, pd, pf) = axs
    schemes = [s for s in res_s['schemes']]
    head = res_s['headline_scheme']
    pool = ctx_s['pools'][head]
    _null_panel(pa, dict(x=pool['two_F_discrete'], y=pool['measured']), res_s['per_scheme'][head]['fits']['discrete'],
                f'(a) strain variant, truth = {SCHEME_LAB[head]}, discrete F', verdict='recorded bias (not a gate)')
    if res_l is not None:
        for ax, sch, lab in ((pb, 'centred', '(b)'), (pc, head, '(c)')):
            if sch in ctx_l['pools']:
                pool = ctx_l['pools'][sch]
                _null_panel(ax, dict(x=pool['two_F_discrete'], y=pool['measured']), res_l['per_scheme'][sch]['fits']['discrete'],
                            f'{lab} LLC variant (hour-0 b, real midpoint velocity, mask_analysis),\ntruth = {SCHEME_LAB[sch]}, discrete F',
                            verdict='recorded bias (not a gate)')
    else:
        for ax in (pb, pc):
            ax.text(0.5, 0.5, 'LLC variant: M0 stores not on disk', ha='center', va='center', transform=ax.transAxes)
    # (d) slope vs width per scheme, discrete solid / chain dashed
    w = np.array(res_s['widths_cells'])
    for sch in schemes:
        e = res_s['per_scheme'][sch]
        for form, ls, mk in (('discrete', '-', 'o'), ('chain', '--', 's')):
            if form not in e['per_width']:
                continue
            s = np.array([e['per_width'][form][str(x)]['ols'] for x in w], dtype=float)
            pd.plot(w, s, ls, marker=mk, color=SCHEME_COL[sch], lw=1.5, ms=5,
                    label=f'{SCHEME_LAB[sch]}, {form}' if form == 'discrete' else None)
            if 'diag_widths' in e:
                wd = np.array(e['diag_widths']['widths'])
                sd = np.array([e['diag_widths']['per_width'][form][str(x)]['ols'] for x in wd], dtype=float)
                pd.plot(wd, sd, marker=mk, ls='none', color=SCHEME_COL[sch], ms=5, mfc='none', mew=1.3)
    pd.plot([], [], '--', color='black', label='dashed: chain-rule F (squares); open: out of the pool')
    pd.axhline(1, color='black', lw=0.8)
    pd.axhline(1 / 0.85, color=COL['red'], lw=1.0, ls=':', label='1/0.85: if the Jacobian attenuation biased the slope')
    pd.set_xscale('log'); pd.set_xticks([1, 1.5, 2, 3, 4, 6, 8]); pd.set_xticklabels(['1', '1.5', '2', '3', '4', '6', '8'])
    pd.set_xlabel('front width ell / dx (tanh; sigma_G = ell/2)')
    pd.set_ylabel('OLS slope, measured DG/Dt on 2F, front pixels')
    pd.set_ylim(0.75, 1.2)
    pd.set_title('(d) slope vs front width per truth (strain variant)\nthe FV truth falls short of F on sharp fronts by '
                 'the scheme\'s own truncation', fontsize=10)
    pd.legend(fontsize=7.5, loc='lower right'); pd.grid(alpha=0.3, which='both')
    # (e) every slope with its CI
    rows = []
    for name, r in (('strain', res_s), ('LLC', res_l)):
        if r is None:
            continue
        for sch in r['schemes']:
            for form in r['forms']:
                t = r['table'][sch][form]
                rows.append((f'{name}: {sch}, {form}', t['slope'], t['ci'], SCHEME_COL[sch], form))
    yv = np.arange(len(rows))[::-1]
    for y, (lab, s, ci, c, form) in zip(yv, rows):
        pe.errorbar(s, y, xerr=[[s - ci[0]], [ci[1] - s]], fmt='o' if form == 'discrete' else 's', color=c,
                    mfc=c if form == 'discrete' else 'none', capsize=3, ms=6)
        pe.text(max(ci[1], s) + 0.006, y, f'{s:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]', va='center', fontsize=7.5)
    pe.set_yticks(yv); pe.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    pe.axvline(1, color='black', lw=0.8)
    pe.axvline(1 / 0.85, color=COL['red'], lw=1.0, ls=':')
    # layout (M1 task 7): room above the first row for the legend, so it never covers a row label;
    # the long title and x-label wrapped so nothing is clipped at the right edge of the axes.
    pe.set_ylim(-0.7, len(rows) + 1.3)
    if res_l is not None:
        b = res_l['bias']
        pe.axvspan(b['ci'][0], b['ci'][1], color='#e6f0fa', label=f'recorded bias (LLC, {b["scheme"]}, {b["form"]}): '
                   f'{b["slope"]:.3f} [{b["ci"][0]:.3f}, {b["ci"][1]:.3f}]')
        pe.legend(fontsize=8, loc='upper left')
    pe.set_xlim(0.6, 1.25)
    pe.set_xlabel('OLS slope on front pixels, with the 32-cell block-bootstrap CI\n(filled: discrete F; open: chain F)')
    pe.set_title('(e) every truth x form x variant\n(dotted red: 1/0.85, the attenuation that does not appear)', fontsize=10)
    pe.grid(alpha=0.3, axis='x')
    # (f) the attribution
    labels, sten, diff, cols = [], [], [], []
    for name, r in (('strain', res_s), ('LLC', res_l)):
        if r is None or 'stencil_effect' not in r:
            continue
        for sch in r['schemes']:
            if sch in ('semilag', 'centred'):
                continue
            d = r['per_scheme'][sch].get('diffusion')
            if d is None:
                continue
            labels.append(f'{name}: {sch}')
            sten.append(r['stencil_effect']['discrete']); diff.append(d['slope_shift']['discrete']); cols.append(SCHEME_COL[sch])
    yv = np.arange(len(labels))[::-1]
    pf.barh(yv, sten, color=COL['order3'], alpha=0.7, label='C-grid stencil: centred FV slope - 1')
    pf.barh(yv, diff, left=sten, color=cols, alpha=0.9, label='implicit diffusion: scheme - centred (scheme colour)')
    for y, s_, d_ in zip(yv, sten, diff):
        pf.text(min(0, s_ + d_) - 0.004, y, f'{s_:+.3f} {d_:+.3f} = {s_ + d_:+.3f}', va='center', ha='right', fontsize=7.5)
    pf.set_yticks(yv); pf.set_yticklabels(labels, fontsize=8)
    pf.axvline(0, color='black', lw=0.8)
    pf.set_xlim(-0.25, 0.1)
    pf.set_xlabel('departure of the slope from 1 (discrete F)')
    lim = res_l['per_scheme'].get('os7mp', {}).get('limiter') if res_l is not None else None
    extra = ''
    if lim is not None:
        extra = (f'\nLLC, the MP limiter alone (OS7MP - OS7): slope shift {lim["slope_shift"]["discrete"]:+.4f};\n'
                 f'its DG/Dt term: median {100 * lim["median_dt_over_G"]:+.2f}% of G/h, p10 {100 * lim["p10_dt_over_G"]:+.1f}%')
    pf.set_title('(f) stencil vs implicit diffusion, discrete F' + extra, fontsize=9.5)
    pf.legend(fontsize=8, loc='lower left'); pf.grid(alpha=0.3, axis='x')
    fig.suptitle('V3b: the finite-volume null -- the truth is a flux-form C-grid advection (MITgcm conventions, advective '
                 'form) with V3\'s velocity; V3\'s pipeline otherwise. A recorded bias for M3, not a gate (M1-Q2).',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, 'V3b_fv_null.png')


# ---------------------------------------------------------------------------
# V4: interpolation bias
# ---------------------------------------------------------------------------
def fig_V4(res, profile):
    """V4: (a) bias vs the sub-cell displacement for orders 1/3/5; (b) bias
    vs the direction of a half-cell displacement (front along the axis and
    tilted 30 degrees); (c) bias vs the front width; (d) the bias across the
    front at a half-cell shift, with the headline error bar."""
    fr = np.array(res['fractions'])
    fig, ((pa, pb), (pc, pd)) = plt.subplots(2, 2, figsize=(15, 11))
    lab = {'order1': 'order 1 (bilinear b; reference only)', 'order3': 'order 3 (cubic b) [the default]',
           'order5': 'order 5 (quintic b)'}
    for key in ('order1', 'order3', 'order5'):
        pa.plot(fr, 100 * np.array(res['vs_fraction'][key]['rms']), 'o-', color=COL[key], ms=4, lw=1.6,
                label=f'{lab[key]}: rms over front pixels')
        pa.plot(fr, 100 * np.array(res['vs_fraction'][key]['at_max']), '--', color=COL[key], lw=1.2,
                label=f'{lab[key].split(" [")[0]}: at the front maximum (signed)')
    pa.set_yscale('symlog', linthresh=0.1)
    pa.set_xlabel('fractional cross-front displacement  d - floor(d)  [cells]  (integer shifts are exact)')
    pa.set_ylabel('measured DG/Dt x dt / G  [% of G per hour]  (truth = 0)')
    pa.set_title(f'(a) uniform flow, sigma_G = {res["sigma_G_ref"]:g}-cell front: the bias vs the sub-cell displacement\n'
                 '(positive at the maximum = fabricated frontogenesis)', fontsize=10)
    pa.legend(fontsize=7, loc='upper right'); pa.grid(alpha=0.3, which='both')
    ang = np.array(res['directions_deg'])
    for tilt, ls in (('tilt0', '-'), ('tilt30', '--')):
        for key in ('order1', 'order3', 'order5'):
            pb.plot(ang, 100 * np.array(res['vs_direction'][tilt][key]), ls, marker='o', color=COL[key], ms=4,
                    lw=1.4, label=f'{key}, front {"along j" if tilt == "tilt0" else "tilted 30 deg"}')
    pb.set_yscale('symlog', linthresh=1e-3)      # the along-front shift of the 1-D front is exactly 0
    pb.set_xlabel('direction of the 0.5-cell displacement from the i axis [deg]')
    pb.set_ylabel('rms bias over front pixels [% of G per hour]')
    pb.set_title('(b) direction: only the cross-front component matters for a 1-D front;\n'
                 'a tilted front engages both axes of the tensor-product kernel', fontsize=10)
    pb.legend(fontsize=7, ncol=2); pb.grid(alpha=0.3, which='both')
    wd = np.array(res['widths_sigma_G'])
    for key in ('order1', 'order3', 'order5'):
        pc.plot(wd, 100 * np.array(res['vs_width'][key]['rms_half_cell']), 'o-', color=COL[key], ms=5, lw=1.6,
                label=f'{key}: rms at a half-cell shift')
        pc.plot(wd, 100 * np.array(res['vs_width'][key]['rms_real_hour']), 's--', color=COL[key], ms=5, lw=1.4,
                label=f'{key}: rms at the real-hour displacement distribution')
    pc.axvline(res['sigma_G_ref'], color=COL['grey'], ls='--', lw=1)
    pc.axhspan(7, 20, color='#f4a259', alpha=0.15, label='per-hour signal 2F dt/G: 7-20%')
    pc.set_xscale('log'); pc.set_yscale('log')
    pc.set_xticks(wd); pc.set_xticklabels([f'{x:g}' for x in wd])
    pc.set_xlabel('front width sigma_G [cells]  (G Gaussian; 1.5 = planning §5.3)')
    pc.set_ylabel('rms bias over front pixels [% of G per hour]')
    pc.set_title(f'(c) front width: the half-cell bias falls as sigma_G^{res["width_slope"]["order3"]:.1f} (order 3), '
                 f'^{res["width_slope"]["order5"]:.1f} (order 5), ^{res["width_slope"]["order1"]:.1f} (order 1)\n'
                 'for sigma_G >= 1.5; 1.0 is shown for reference (real fronts include 1-2-cell features, M1 task 3)',
                 fontsize=10)
    pc.legend(fontsize=7, loc='lower left'); pc.grid(alpha=0.3, which='both')
    x = profile['s_cells']
    for key in ('order1', 'order3', 'order5'):
        pd.plot(x, 100 * profile[key], '-', color=COL[key], lw=1.6, label=lab[key].split(' [')[0])
    pd.axhline(0, color='black', lw=0.8)
    pd.axvspan(-1.8 * res['sigma_G_ref'], 1.8 * res['sigma_G_ref'], color='#e6f0fa',
               label='front pixels (G >= 0.2 max)')
    h = res['headline']
    pd.text(0.02, 0.97,
            f'headline error bar (order 3, sigma_G = {res["sigma_G_ref"]:g}):\n'
            f'rms over front pixels at the real-hour\ndisplacement distribution (median '
            f'{res["real_hour_median_cells"]:.2f}, p99 {res["real_hour_p99_cells"]:.2f} cells):\n'
            f'  {100 * h["rms"]:.2f}% of G per hour\n'
            f'  = {100 * h["rms"] / 0.20:.1f}-{100 * h["rms"] / 0.07:.1f}% of the 7-20% signal\n'
            f'signed at the maximum: {100 * h["at_max_mean"]:+.2f}% of G per hour\n'
            f'order 5: {100 * res["headline_order5"]["rms"]:.2f}%;  order 1: {100 * res["headline_order1"]["rms"]:.2f}%',
            transform=pd.transAxes, va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='white', ec='#888888'))
    pd.set_xlim(-6, 6); pd.set_ylim(-4, 8)
    pd.set_xlabel('distance from the front centre [cells]')
    pd.set_ylabel('measured DG/Dt x dt / G  [% of G per hour]')
    pd.set_title('(d) across the front at a half-cell shift: the fabricated tendency is positive at the\n'
                 'maximum and negative on the flanks (the mirror of V5\'s G_d error)', fontsize=10)
    pd.legend(fontsize=8, loc='lower right'); pd.grid(alpha=0.3)
    fig.suptitle('V4: interpolation bias -- semilag.measured_DGDt under a uniform, zero-strain flow '
                 '(true DG/Dt = 0); the recorded error bar for every later slope', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, 'V4_interpolation_bias.png')


# ---------------------------------------------------------------------------
# V5: the interpolation choice, made visible
# ---------------------------------------------------------------------------
def fig_V5(c, res, widths, sweep, sigma_G, pred, k):
    col = {'truth': 'black', 'bilinear_G': '#c8102e', 'b_order1': '#f4a259',
           'b_order3': '#1f5fa8', 'b_order5': '#2a9d8f'}
    lab = {'bilinear_G': 'G interpolated bilinearly (the trap)',
           'b_order1': 'b interpolated bilinearly, then the stencil',
           'b_order3': 'b interpolated cubic (order 3), then the stencil  [the rule]',
           'b_order5': 'b interpolated quintic (order 5), then the stencil'}
    x = c['x_cells']
    win = np.abs(x) <= 6
    fig, (pa, pb, pc) = plt.subplots(1, 3, figsize=(19, 6.2))

    # (a) the profiles
    Gmax = np.nanmax(c['truth'])
    pa.plot(x[win], c['analytic'][win] / Gmax, '-', color='#999999', lw=1.0,
            label='analytic G of the shifted front (continuum; the stencil attenuates it)')
    pa.plot(x[win], c['truth'][win] / Gmax, '-', color=col['truth'], lw=2.0,
            label='truth: the stencil on the exactly shifted b')
    for name, mk_ in (('bilinear_G', 's'), ('b_order1', 'v'), ('b_order3', 'o'), ('b_order5', 'D')):
        pa.plot(x[win], c[name][win] / Gmax, mk_, color=col[name], ms=6, mfc='none', mew=1.6,
                label=lab[name])
    xk = x[k]
    pa.annotate(f'bias at the maximum:\nbilinear G {100 * res["bias_bilinear_G"]:+.2f}%\n'
                f'bilinear b {100 * res["bias_b_order1"]:+.2f}%\n'
                f'cubic b {100 * res["bias_b_order3"]:+.2f}%\nquintic b {100 * res["bias_b_order5"]:+.2f}%'
                f'\n\nprediction dx² G_xx/8G = {100 * pred:+.2f}%\n(planning §5.3: ~5.5%)',
                xy=(xk, c['bilinear_G'][k] / Gmax), xytext=(2.6, 0.55), fontsize=9,
                arrowprops=dict(arrowstyle='->', color=col['bilinear_G']),
                bbox=dict(boxstyle='round', fc='white', ec='#888888'))
    pa.set_xlabel('distance from the front centre [cells]')
    pa.set_ylabel('G / max G (truth)')
    pa.set_title(f'(a) G = |grad b|² after a half-cell shift; front sigma_G = {sigma_G} cells '
                 '(~1.5 cells wide)', fontsize=10)
    pa.set_xlim(-11.5, 6.3)                      # room for the legend over the flat left tail
    pa.legend(fontsize=7.5, loc='upper left')
    pa.grid(alpha=0.3)

    # (b) the relative error across the front
    for name in ('bilinear_G', 'b_order1', 'b_order3', 'b_order5'):
        with np.errstate(invalid='ignore', divide='ignore'):
            rel = 100 * (c[name] - c['truth']) / c['truth']
        ok = win & (c['truth'] > 0.05 * Gmax)
        pb.plot(x[ok], rel[ok], '-', color=col[name], lw=1.8, label=lab[name].split('  [')[0])
    # the bilinear prediction across the front: the chord of a convex function
    # lies above it, so interp - truth = +dx^2 G_xx / 8 -- negative at the
    # maximum (G_xx < 0), positive on the flanks
    Gt = c['truth']
    with np.errstate(invalid='ignore', divide='ignore'):
        pred_curve = 100 * (np.roll(Gt, -1) - 2 * Gt + np.roll(Gt, 1)) / (8 * Gt)
    ok = win & (Gt > 0.05 * Gmax)
    pb.plot(x[ok], pred_curve[ok], ':', color='black', lw=1.5, label='prediction dx² G_xx / 8G (bilinear)')
    pb.axhline(0, color='#888888', lw=0.8)
    pb.axvline(xk, color='#888888', lw=0.8, ls='--')
    pb.set_xlabel('distance from the front centre [cells]')
    pb.set_ylabel('(G_interp - G_truth) / G_truth  [%]')
    pb.set_title('(b) relative error: negative at the maximum, positive on the flanks --\n'
                 'bilinear G flattens the front and fabricates frontogenesis', fontsize=10)
    pb.legend(fontsize=7.5, loc='upper center')
    pb.grid(alpha=0.3)
    pb.set_ylim(-9, 9)

    # (c) the bias at the maximum vs the front width
    for name in ('bilinear_G', 'b_order1', 'b_order3', 'b_order5'):
        pc.plot(widths, -100 * np.array(sweep[name]), 'o-', color=col[name], lw=1.6, ms=5,
                label=lab[name].split('  [')[0])
    pc.plot(widths, 100 / (8 * widths ** 2), ':', color='black', lw=1.5,
            label='prediction dx² G_xx / 8G = 1 / (8 sigma_G²)')
    pc.axvline(sigma_G, color='#888888', lw=0.8, ls='--')
    pc.axhline(5.5, color='#c8102e', lw=0.8, ls=':')
    pc.text(4.1, 5.5, '~5.5% (planning §5.3)', fontsize=8, color='#c8102e', va='bottom')
    pc.set_yscale('log')
    pc.set_xlabel('front width sigma_G [cells]')
    pc.set_ylabel('-bias at the maximum [%]  (all negative)')
    pc.set_title('(c) bias at the maximum vs front width, half-cell shift\n'
                 '(per-hour signal 2F dt / G is 7-20%: bilinear G is 25-80% of it)', fontsize=10)
    pc.legend(fontsize=7.5, loc='lower left')
    pc.grid(alpha=0.3, which='both')

    fig.suptitle('V5: interpolate b (order >= 3), never G -- a synthetic front shifted by half a cell '
                 '(semilag.gradb2_at_departure vs bilinear G)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, 'V5_interp_half_cell.png')


# ---------------------------------------------------------------------------
# V6: land halo
# ---------------------------------------------------------------------------
V6_CLASSES = [
    (0, 'land (hFacC = 0)', '#b0b0b0'),
    (1, 'retained: mask_analysis', '#e6f0fa'),
    (2, 'removed by the land halo', '#c8102e'),
    (3, 'removed by the offshore cut only', '#f4a259'),
    (4, 'removed by the tile-edge margin only', '#5b2a86'),
]


def fig_V6(ctx):
    """V6 from the context ``validate.qa_land_halo`` assembled: the masks,
    the class map and counts, the distance field, the rim profile and the
    Gulf check."""
    masks, a, X, Y = ctx['masks'], ctx['attrs'], ctx['X'], ctx['Y']
    halo_km, halo_cells, edge_cells = ctx['halo_km'], ctx['halo_cells'], ctx['edge_cells']
    oc, halo, dist, cls, counts = ctx['oc'], ctx['halo'], ctx['dist'], ctx['cls'], ctx['counts']
    taxi, G, rim, rim_max, gulf, res = ctx['taxi'], ctx['G'], ctx['rim'], ctx['rim_max'], ctx['gulf'], ctx['res']
    SNAPSHOT = ctx['snapshot']
    cmap_cls = ListedColormap([c for _, _, c in V6_CLASSES])
    norm_cls = BoundaryNorm(np.arange(-0.5, len(V6_CLASSES)), cmap_cls.N)
    fig, axs = plt.subplots(2, 3, figsize=(19, 11.5))
    (pa, pb, pc), (pd, pe, pf) = axs

    # (a) whole tile: every stage of the mask
    draw_map(pa, X, Y, cls.astype(float), cmap_cls, norm=norm_cls)
    handles = [Rectangle((0, 0), 1, 1, color=col) for _, _, col in V6_CLASSES]
    labels = [lab for _, lab, _ in V6_CLASSES]
    labels[1] += f'  (n = {counts["n_analysis"]:,})'
    labels[2] += f'  (n = {counts["n_removed_halo"]:,})'
    labels[3] += f'  (n = {counts["n_removed_offshore_only"]:,})'
    labels[4] += f'  (n = {counts["n_removed_edge_only"]:,})'
    pa.legend(handles, labels, loc='lower left', fontsize=7.5, framealpha=0.95)
    pa.set_title(f'(a) mask stages: ocean {counts["n_ocean"]:,} -> halo {counts["n_halo"]:,} -> '
                 f'offshore {counts["n_offshore"]:,}\n-> & edge margin = analysis '
                 f'{counts["n_analysis"]:,} ({100 * counts["n_analysis"] / counts["n_ocean"]:.1f}% '
                 'of the ocean)', fontsize=10)

    # (b) cell-level inset, Monterey Bay: the coastline before / after the halo
    j0, i0, w = 287, 96, 24
    sl_ = (slice(j0 - w, j0 + w), slice(i0 - w, i0 + w))
    pb.pcolormesh(X[sl_], Y[sl_], cls[sl_].astype(float), cmap=cmap_cls, norm=norm_cls,
                  shading='nearest', edgecolors='#555555', linewidth=0.15)
    # the coastline 'before' is the grey/red boundary (hFacC); 'after' is
    # the halo_km contour of the distance field
    pb.contour(X[sl_], Y[sl_], np.nan_to_num(dist[sl_], nan=-1.0), levels=[halo_km],
               colors='black', linewidths=1.2)
    pb.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y[sl_]))))
    pb.set_xlabel('longitude'); pb.set_ylabel('latitude')
    pb.set_title(f'(b) Monterey Bay, cell edges drawn: coastline before the halo (grey/red, hFacC)\n'
                 f'and after (black: coast distance = {halo_km:.2f} km = {halo_cells} x median dxC '
                 f'{a["dxC_median_km"]:.3f} km)', fontsize=10)
    pa.add_patch(Rectangle((X[sl_].min(), Y[sl_].min()), np.ptp(X[sl_]), np.ptp(Y[sl_]),
                           fill=False, ec='black', lw=1.2))

    # (c) the distance field with the two thresholds
    m = draw_map(pc, X, Y, dist, 'viridis', land=~oc, vmin=0, vmax=300)
    fig.colorbar(m, ax=pc, shrink=0.85, label='coast_distance_km  (clipped at 300)')
    pc.contour(X, Y, np.nan_to_num(dist, nan=-1.0), levels=[a['offshore_km']], colors='white',
               linewidths=1.2)
    pc.contour(X, Y, np.nan_to_num(dist, nan=-1.0), levels=[halo_km], colors='#c8102e',
               linewidths=0.6)
    pc.set_title(f'(c) distance to land (skfmm, mean spacing {a["dyC_mean_km"]:.3f} x '
                 f'{a["dxC_mean_km"]:.3f} km)\nwhite: {a["offshore_km"]:g} km offshore cut; red: the '
                 f'{halo_km:.1f} km halo', fontsize=10)

    # (d) Gulf of California
    gsl = (slice(540, 720), slice(330, 620))
    m = pd.pcolormesh(X[gsl], Y[gsl], np.ma.masked_invalid(dist[gsl]), cmap='viridis',
                      vmin=0, vmax=100, shading='nearest')
    pd.pcolormesh(X[gsl], Y[gsl], np.ma.masked_where(oc[gsl], np.ones_like(X[gsl])),
                  cmap=ListedColormap(['#b0b0b0']), shading='nearest')
    ana = masks['mask_analysis'].values
    pd.pcolormesh(X[gsl], Y[gsl], np.ma.masked_where(~ana[gsl], np.ones_like(X[gsl])),
                  cmap=ListedColormap(['#e6f0fa']), shading='nearest')
    pd.contour(X[gsl], Y[gsl], np.nan_to_num(dist[gsl], nan=-1.0), levels=[halo_km],
               colors='#c8102e', linewidths=0.8)
    fig.colorbar(m, ax=pd, shrink=0.85, label='coast_distance_km')
    pd.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y[gsl]))))
    pd.set_xlabel('longitude'); pd.set_ylabel('latitude')
    pd.set_title(f'(d) Gulf of California: its own ocean component ({gulf["n_gulf"]:,} cells), '
                 f'clipped by the\neast edge at {gulf["gulf_lon"][1]:.2f}E; max coast distance '
                 f'{gulf["gulf_max_coast_km"]:.1f} km < {a["offshore_km"]:g}: cells >= '
                 f'{a["offshore_km"]:g} km: {gulf["gulf_n_ge_offshore"]},\nin mask_analysis: '
                 f'{gulf["gulf_n_analysis"]} (pale = retained Pacific)', fontsize=10)
    pa.add_patch(Rectangle((X[gsl].min(), Y[gsl].min()), np.ptp(X[gsl]), np.ptp(Y[gsl]),
                           fill=False, ec='black', lw=1.2, ls='--'))

    # (e) the finite tile-edge rim and the margin that removes it
    prof = edge_profile(G, oc, nmax=12)
    cols = {'low j (j=0)': '#1f5fa8', 'high j (j=719)': '#c8102e',
            'low i (i=2880)': '#2a9d8f', 'high i (i=3599)': '#f4a259'}
    k = np.arange(12)
    for name, col in cols.items():
        pe.plot(k, prof[name], 'o-', color=col, lw=1.5, ms=4, label=f'G, {name} edge')
    pe.axvspan(-0.5, edge_cells - 0.5, color='#5b2a86', alpha=0.15,
               label=f'edge margin: edge_cells = {edge_cells} (mask_edge False)')
    offs = {r: sorted({o for kk in ('low_j', 'high_j', 'low_i', 'high_i') for o in rim[r][kk]})
            for r in ('G_components', 'jacobian')}
    pe.axvline(rim_max + 0.5, color='black', lw=1.2, ls=':',
               label=f'crop test: G changed at offsets {offs["G_components"]}, '
                     f'Jacobian at {offs["jacobian"]}')
    pe.set_yscale('log'); pe.set_xticks(k); pe.set_xlim(-0.5, 11.5)
    pe.set_xlabel('offset from tile edge [cells]')
    pe.set_ylabel('median |G| / median over offsets 1-3')
    pe.set_title('(e) tile-edge rim: finite, not NaN (xgcm padding = 0). Low edges ~1e6x (diff '
                 'against 0),\nhigh edges ~0.5x (interp with 0); Jacobian 1-2 cells deep; '
                 f'edge_cells = {edge_cells} margin shaded', fontsize=10)
    pe.legend(fontsize=7, loc='upper right'); pe.grid(alpha=0.3)

    # (f) halo width in cells: index distance to land, excluded vs retained
    bins = np.arange(0.5, 16.5)
    pf.hist(taxi[oc & ~halo], bins=bins, color='#c8102e', alpha=0.8,
            label=f'ocean removed by the halo (n = {counts["n_removed_halo"]:,})')
    pf.hist(taxi[halo & (taxi <= 15)], bins=bins, color='#1f5fa8', alpha=0.8,
            label='retained (taxicab distance <= 15 shown)')
    pf.axvline(halo_cells, color='black', lw=1.5, ls='--', label=f'halo_cells = {halo_cells}')
    pf.set_xlabel('taxicab index distance to the nearest land cell [cells]')
    pf.set_ylabel('cells')
    pf.set_title(f'(f) halo width: {halo_km:.2f} km = {halo_cells} cells of dxC (meridional, i) = '
                 f'{halo_km / a["dyC_median_km"]:.2f} of dyC (zonal, j)\nretained: min taxicab '
                 f'{res["halo_min_taxicab_retained"]}, min chessboard {res["halo_min_chessboard_retained"]}; '
                 f'removed ocean: max taxicab {res["halo_max_taxicab_excluded"]}', fontsize=10)
    pf.legend(fontsize=8); pf.grid(alpha=0.3, axis='y')

    fig.suptitle(f'V6: land halo, offshore cut, Gulf of California and tile-edge margin; tile 330 '
                 f'(face 10), rim measured at {SNAPSHOT}; maps north-up via (XC, YC)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, 'V6_land_halo_tile330.png')
