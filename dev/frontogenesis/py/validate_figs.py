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
