"""Slides: how ``2F`` (the kinematic frontogenesis prediction) of a tracked front is computed
(frontogenesis_slides.md, Figures / 2F, prompt 2; Q&A 2F-Q1..Q6).

The same hour, front and parcels as ``slides_dGdt.py``: pair 36 (hours 36, 37 =
2012-07-03 12:00, 13:00 UTC), label 83 of M4 flow track 0, the band of the label
dilated by one cell.  ``2F`` has **no time difference** (2F-Q1): it is an instant, the
midpoint 12:30, built from the hour-averaged fields

    b_mid = (b(12:00) + b(13:00)) / 2,     u_mid = (u(12:00) + u(13:00)) / 2.

It is explained with the continuum picture (2F-Q2 (a)),

    2F = -2 (grad b)^T (grad u) (grad b) = -delta G + |sigma| G cos 2 theta,

(convergence ``-delta`` sharpens; strain sharpens when ``grad b`` lies within 45 deg of the
compressional axis), and the number quoted is the **stored, discretely consistent** ``2F``
(``operators.frontogenesis(form='discrete')``, the ``two_F`` of the derived store), which on
narrow fronts is smaller than the continuum form.

Two figures (2F-Q4):

* ``slides/2F_method.png`` -- (a) the front and the window (shared with the dG/dt slide);
  (b) one parcel close up: ``b_mid`` contours, ``grad b``, the compressional axis, the
  velocity relative to the parcel, and its numbers; (c) ``2F`` on every parcel and the band
  mean; (d)-(f) the window: relative velocity on ``G``, ``cos 2 theta`` with the compressional
  axes on the ridge, and ``2F``.  Panels (c) and (f) use the colour scales of
  ``dGdt_method.png`` (c), (f), so the two slides compare directly.
* ``slides/2F_split.png`` -- convergence ``-delta G`` + strain ``|sigma| G cos 2 theta`` =
  continuum ``2F`` (exact, from the Jacobian-consistent strain set), every parcel and the
  window; the stored discrete value is given in the footer.

Window and parcel are those of the dG/dt figures (read from ``data/slides_dGdt_36.json``).
Every field is recomputed from the 72-h raw store with the pipeline's functions and
**asserted** against the stored ``two_F``, ``two_F_chain`` and ``G`` and against ``two_F`` of
``data/matched_parcels_track_81.json``.  Run in the ``frontogenesis`` env:

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python slides_2F.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import xarray as xr
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import m2_qa                                                         # noqa: E402
import inputs as inp                                                 # noqa: E402
import operators as op                                               # noqa: E402
import slides_dGdt as sd                                             # noqa: E402
from slides_dGdt import (_ll, _corners, _box, _geo, _crop, _cells, _blank, _lasso, _mark,   # noqa: E402
                         _log, _label, _save, panel_front, fig_split,
                         WIN, CLOSE, SCALE, CMAP_G, CMAP_R, CMAP_B, COL_WIN, COL_CLOSE)
from slides_tracking import BAND, STRUCT                             # noqa: E402

DATA_DIR = HERE.parent / 'data'
SLIDE_DIR = HERE.parent / 'slides'
DGDT_JSON = DATA_DIR / 'slides_dGdt_{pair}.json'      # window, parcel and colour scales of the dG/dt slide
SUMMARY = DATA_DIR / 'slides_2F_{pair}.json'

PAIR = 36
COL_GRAD = '#d62728'         # grad b
COL_COMP = 'k'               # compressional axis
COL_VEL = '#6a3d9a'          # relative velocity
AXIS_CELLS = 1.6             # half-length of the drawn axes in the close-up [cells]
TICK_CELLS = 0.45            # half-length of the compressional-axis ticks in the window [cells]
TOL_STORED = 1e-5            # max |recomputed - stored| / rms(stored) on the band
TOL_JSON = 1e-6
TOL_SPLIT = 1e-10            # convergence + strain = continuum 2F, relative to its rms


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------
_np2 = sd._np2


def compute(pair=PAIR, data_dir=None):
    """The midpoint fields of ``pair`` and every term of ``2F`` on the full tile."""
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    raw, g, grid, _, _ = m2_qa.open_inputs()
    h0, h1 = m2_qa.hour_ds(raw, g, pair), m2_qa.hour_ds(raw, g, pair + 1)
    f0, f1 = inp.filtered(h0, 0), inp.filtered(h1, 0)       # L = 0
    bm, U, V = (inp.midpoint(a, b) for a, b in zip(f0, f1))  # b_mid, U_mid, V_mid (budget.py)
    two_F = 2.0 * _np2(op.frontogenesis(bm, U, V, h0, grid, form='discrete'))
    two_F_chain = 2.0 * _np2(op.frontogenesis(bm, U, V, h0, grid, form='chain'))
    bx, by = op.grad_b(bm, h0, grid)
    ux, uy, vx, vy = op.jacobian(U, V, h0, grid)
    delta, sn, ss, smag = op.strain_from_jacobian(ux, uy, vx, vy)
    theta = op.strain_alignment(bx, by, sn, ss)
    uc, vc = op.centred_model_velocity(U, V, grid)
    cs, sn_ = _np2(h0['CS']), _np2(h0['SN'])
    uc, vc = _np2(uc), _np2(vc)
    bx, by = _np2(bx), _np2(by)
    G = bx ** 2 + by ** 2
    delta, sn, ss, smag, theta = (_np2(a) for a in (delta, sn, ss, smag, theta))

    der = xr.open_zarr(data_dir / sd.DERIVED.name).isel(time=pair)
    track = json.load(open(data_dir / sd.PARCELS_JSON.name))
    row = next(r for r in track['rows'] if r['step'] == pair)
    lab = xr.open_zarr(data_dir / sd.FRONTS.name)['labels'].isel(time=pair).values
    skel = lab == int(row['label'])
    band = ndi.binary_dilation(skel, STRUCT, iterations=BAND)
    d = dict(pair=pair, label=int(row['label']), row=row,
             times=[str(h0['time'].values)[:16], str(h1['time'].values)[:16]],
             time_mid=str(np.datetime_as_string(der['time_mid'].values, unit='m')),
             X=np.asarray(der['XC'].values), Y=np.asarray(der['YC'].values),
             bm=_np2(bm), bx=bx, by=by, G=G, Gmid=G,
             two_F=two_F, two_F_chain=two_F_chain,
             delta=delta, sigma_n=sn, sigma_s=ss, sigma=smag, theta=theta,
             conv=-delta * G, strain=smag * G * np.cos(2 * theta),
             u_e=uc * cs - vc * sn_, v_n=uc * sn_ + vc * cs,            # geographic, m/s
             psi_c=0.5 * np.arctan2(ss, sn) + 0.5 * np.pi,             # compressional axis [rad from east]
             stored=dict(two_F=np.asarray(der['two_F'].values, 'float64'),
                         two_F_chain=np.asarray(der['two_F_chain'].values, 'float64'),
                         G=np.asarray(der['G'].values, 'float64')),
             skel=skel, band=band)
    return d


def check(d):
    band = d['band']
    out = {}
    for key, mine in (('two_F', d['two_F']), ('two_F_chain', d['two_F_chain']), ('G', d['G'])):
        st = d['stored'][key][band]
        if not (np.isfinite(st).all() and np.isfinite(mine[band]).all()):
            raise ValueError(f'{key}: NaN on the band')
        err = float(np.max(np.abs(mine[band] - st)) / np.sqrt(np.mean(st ** 2)))
        if err > TOL_STORED:
            raise AssertionError(f'recomputed {key} differs from the stored field by {err:.2e} of the rms')
        out[f'{key}_max_err_over_rms'] = err
    mean, ref = float(d['two_F'][band].mean()), float(d['row']['two_F'])
    if abs(mean - ref) > TOL_JSON * abs(ref):
        raise AssertionError(f'band mean 2F {mean:.6e} != json two_F {ref:.6e}')
    rms = np.sqrt(np.mean(d['two_F_chain'][band] ** 2))
    split = float(np.max(np.abs(d['conv'] + d['strain'] - d['two_F_chain'])[band]) / rms)
    if split > TOL_SPLIT:
        raise AssertionError(f'convergence + strain != continuum 2F ({split:.2e} of the rms)')
    out.update(band_mean_two_F=mean, json_two_F=ref, split_err_over_rms=split,
               n_band=int(band.sum()), n_skeleton=int(d['skel'].sum()))
    return out


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def _cell_deg(d, p):
    """Centre-to-centre spacing at ``p`` in degrees of latitude (screen-true length unit
    on axes whose aspect is 1 / cos(lat))."""
    j, i = p
    lat = np.deg2rad(d['Y'][j, i])
    return float(np.hypot((d['X'][j, i + 1] - d['X'][j, i]) * np.cos(lat), d['Y'][j, i + 1] - d['Y'][j, i]))


def _offset(d, p, ang, length):
    """A data-space offset of screen length ``length`` (deg of lat) at angle ``ang`` (rad
    from east)."""
    lat = np.deg2rad(d['Y'][p])
    return length * np.cos(ang) / np.cos(lat), length * np.sin(ang)


def _axis(ax, d, p, ang, half, col, lw=1.6, inward=True, z=8):
    """The compressional axis through ``p``: two arrows pointing inward along ``ang``."""
    x0, y0 = _ll(d, *p)
    x0, y0 = x0[0], y0[0]
    dx, dy = _offset(d, p, ang, half)
    for sgn in (1, -1):
        tail, head = (x0 + sgn * dx, y0 + sgn * dy), (x0 + sgn * 0.35 * dx, y0 + sgn * 0.35 * dy)
        if not inward:
            tail, head = head, tail
        ax.add_patch(FancyArrowPatch(tail, head, arrowstyle='-|>', mutation_scale=14, color=col,
                                     lw=lw, zorder=z))


def _nice(v):
    e = 10 ** np.floor(np.log10(v))
    return float(min((1, 2, 5, 10), key=lambda m: abs(m * e - v)) * e)


def _quiver(ax, d, sj, si, du, dv, scale, key_val, key_xy=(0.84, 1.03), every=1):
    X, Y = d['X'][sj, si][::every, ::every], d['Y'][sj, si][::every, ::every]
    q = ax.quiver(X, Y, du[::every, ::every], dv[::every, ::every], color=COL_VEL, angles='uv',
                  scale_units='inches', scale=scale, width=0.004, headwidth=4, zorder=7)
    ax.quiverkey(q, *key_xy, key_val, f'{key_val * 100:.0f} cm s$^{{-1}}$', labelpos='E',
                 coordinates='axes', fontproperties={'size': 9}, color=COL_VEL)
    return q


# ---------------------------------------------------------------------------
# figure 1: the method
# ---------------------------------------------------------------------------
def fig_method(d, win, p, lims, out_png):
    band = d['band']
    cj, ci = _crop(d)
    wj, wi = slice(win[0], win[0] + WIN), slice(win[1], win[1] + WIN)
    h = CLOSE // 2
    kj, ki = slice(p[0] - h, p[0] + h + 1), slice(p[1] - h, p[1] + h + 1)
    t0, t1, tm = d['times'][0][11:16], d['times'][1][11:16], d['time_mid'][11:16]
    cell = _cell_deg(d, p)

    fig, axs = plt.subplots(2, 3, figsize=(17.0, 12.0))
    (a, b, c), (dd, e, f) = axs
    panel_front(fig, a, d, win)

    # (b) one parcel: b_mid, grad b, the compressional axis, the velocity relative to the parcel
    bb = d['bm'][kj, ki]
    _cells(b, d, d['bm'], kj, ki, cmap=CMAP_B, alpha=0.85,
           vmin=bb.min() - 0.15 * np.ptp(bb), vmax=bb.max() + 0.15 * np.ptp(bb))
    Xc, Yc = _corners(d, kj.start, kj.stop, ki.start, ki.stop)
    for k in range(Xc.shape[0]):
        b.plot(Xc[k], Yc[k], color='0.45', lw=0.5, zorder=2)
    for k in range(Xc.shape[1]):
        b.plot(Xc[:, k], Yc[:, k], color='0.45', lw=0.5, zorder=2)
    kj2, ki2 = slice(kj.start - 1, kj.stop + 1), slice(ki.start - 1, ki.stop + 1)
    cs = b.contour(d['X'][kj2, ki2], d['Y'][kj2, ki2], d['bm'][kj2, ki2] * 1e3, levels=8,
                   colors='k', linewidths=0.8, zorder=3)
    b.clabel(cs, fontsize=7, fmt='%.2f')
    du = d['u_e'][kj, ki] - d['u_e'][p]
    dv = d['v_n'][kj, ki] - d['v_n'][p]
    vmax = float(np.nanmax(np.hypot(du, dv)))
    key = _nice(0.6 * vmax)
    _quiver(b, d, kj, ki, du, dv, scale=vmax / 0.55, key_val=key, key_xy=(0.10, 0.955))
    _axis(b, d, p, d['psi_c'][p], AXIS_CELLS * cell, COL_COMP, lw=1.8)
    ang_g = float(np.arctan2(d['by'][p], d['bx'][p]))
    x0, y0 = _ll(d, *p)
    gx, gy = _offset(d, p, ang_g, 1.3 * cell)
    b.add_patch(FancyArrowPatch((x0[0], y0[0]), (x0[0] + gx, y0[0] + gy), arrowstyle='-|>',
                                mutation_scale=18, color=COL_GRAD, lw=2.6, zorder=9))
    b.text(x0[0] + 1.08 * gx, y0[0] + 1.08 * gy, r'$\nabla b$', color=COL_GRAD, fontsize=12,
           fontweight='bold', ha='center', va='center', zorder=10)
    b.scatter(x0, y0, s=30, c='w', edgecolor='k', lw=1.2, zorder=10)
    _geo(b, d, kj, ki, nbins=3)
    b.set_xlim(Xc.min(), Xc.max())
    b.set_ylim(Yc.min(), Yc.max())
    b.set_title(f'one parcel at {tm} UTC: gradient, strain and relative flow', fontsize=11)
    b.legend(handles=[Line2D([], [], color=COL_GRAD, lw=2.6, label=r'$\nabla b$ (direction)'),
                      Line2D([], [], color=COL_COMP, lw=1.8, marker='>', ms=6,
                             label='compressional axis of the strain'),
                      Line2D([], [], color=COL_VEL, lw=1.2, label=r'$u - u(x)$: flow relative to the parcel'),
                      Line2D([], [], color='k', lw=0.8, label=r'$b_{mid}$ contours [10$^{-3}$ m s$^{-2}$]')],
             loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5, frameon=False)
    G, dl, sg, th = d['G'][p], d['delta'][p], d['sigma'][p], d['theta'][p]
    b.text(0.5, -0.31,
           rf'$G = {G * 1e13:.3f}\times10^{{-13}}$ s$^{{-4}}$,  $\delta = {dl * 1e5:+.2f}$,  '
           rf'$|\sigma| = {sg * 1e5:.2f}$ [10$^{{-5}}$ s$^{{-1}}$],  $\theta = {np.degrees(th):.0f}°$'
           '\n'
           rf'$2F = -\delta G + |\sigma| G\cos 2\theta = {d["conv"][p] * SCALE:+.1f} '
           rf'{d["strain"][p] * SCALE:+.1f} = {d["two_F_chain"][p] * SCALE:+.1f}$'
           '\n'
           rf'stored (discrete) $2F = {d["two_F"][p] * SCALE:+.1f}$   [10$^{{-19}}$ s$^{{-5}}$]',
           transform=b.transAxes, ha='center', va='top', fontsize=10)
    _label(b, 'b')

    # (c) every parcel: stored 2F, on the DG/Dt slide's scale
    v = d['two_F'][band] * SCALE
    norm = TwoSlopeNorm(0.0, -lims['band'], lims['band'])
    _blank(c, d, cj, ci, band)
    m = _cells(c, d, d['two_F'] * SCALE, cj, ci, mask=band, cmap=CMAP_R, norm=norm)
    _box(c, d, win[0], win[1], WIN, COL_WIN, lw=2.0)
    _geo(c, d, cj, ci)
    c.set_title(r'every parcel: $2F$ (stored, discrete)', fontsize=11)
    cb = fig.colorbar(m, ax=c, fraction=0.045, pad=0.02, extend='both')
    cb.set_label(r'$2F$ [10$^{-19}$ s$^{-5}$]  (scale of the $DG/Dt$ slide)', fontsize=9.5)
    c.text(0.97, 0.04, f'mean over {int(band.sum()):,} parcels\n= {v.mean():+.2f}'
           r' $\times 10^{-19}$ s$^{-5}$',
           transform=c.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold',
           bbox=dict(fc='w', ec='k', lw=0.8, alpha=0.9), zorder=10)
    c.text(0.97, 0.20, f'($DG/Dt$ on the same parcels: {lims["DGDt_mean"]:+.2f})',
           transform=c.transAxes, ha='right', va='bottom', fontsize=9.5, color='0.25', zorder=10)
    _label(c, 'c')

    # (d) the window: G with the flow relative to the window mean
    lg = _log(d['G'])
    vw = np.nanpercentile(lg[wj, wi][band[wj, wi]], [1, 99])
    m = _cells(dd, d, lg, wj, wi, cmap=CMAP_G, vmin=vw[0] - 1.0, vmax=vw[1])
    _lasso(dd, d, wj, wi, lw=1.2)
    du = d['u_e'][wj, wi] - np.nanmean(d['u_e'][wj, wi])
    dv = d['v_n'][wj, wi] - np.nanmean(d['v_n'][wj, wi])
    wmax = float(np.nanpercentile(np.hypot(du, dv), 98))
    _quiver(dd, d, wj, wi, du, dv, scale=wmax / 0.28, key_val=_nice(0.6 * wmax), key_xy=(0.74, -0.075))
    cb = fig.colorbar(m, ax=dd, fraction=0.045, pad=0.02)
    cb.set_label(r'$\log_{10} G$ [s$^{-4}$]', fontsize=10)
    dd.set_title(r'window: $G$, flow relative to its mean', fontsize=11)
    _label(dd, 'd')

    # (e) the window: cos 2 theta on the parcels, the compressional axis on the ridge
    _blank(e, d, wj, wi, band)
    m = _cells(e, d, np.cos(2 * d['theta']), wj, wi, mask=band, cmap='PuOr_r', vmin=-1, vmax=1)
    for jj, ii in zip(*np.where(d['skel'][wj, wi])):
        q = (wj.start + jj, wi.start + ii)
        x0, y0 = _ll(d, *q)
        dx, dy = _offset(d, q, d['psi_c'][q], TICK_CELLS * cell)
        e.plot([x0[0] - dx, x0[0] + dx], [y0[0] - dy, y0[0] + dy], color=COL_COMP, lw=1.3, zorder=7)
    cb = fig.colorbar(m, ax=e, fraction=0.045, pad=0.02)
    cb.set_label(r'$\cos 2\theta$  (+1: $\nabla b$ along the compressional axis)', fontsize=9.5)
    fr = float(np.mean(d['theta'][band] < np.pi / 4))
    e.set_title(r'window: alignment $\cos 2\theta$ (ticks: compressional axis)', fontsize=10.5)
    e.text(0.97, 0.04, f'every parcel: {100 * fr:.0f}% have θ < 45°\n'
           f'median θ = {np.degrees(np.median(d["theta"][band])):.0f}°',
           transform=e.transAxes, ha='right', va='bottom', fontsize=9.5,
           bbox=dict(fc='w', ec='0.5', lw=0.6, alpha=0.9), zorder=10)
    _label(e, 'e')

    # (f) the window: 2F, on the DG/Dt slide's window scale
    _blank(f, d, wj, wi, band)
    m = _cells(f, d, d['two_F'] * SCALE, wj, wi, mask=band, cmap=CMAP_R,
               norm=TwoSlopeNorm(0.0, -lims['window'], lims['window']))
    cb = fig.colorbar(m, ax=f, fraction=0.045, pad=0.02, extend='both')
    cb.set_label(r'$2F$ [10$^{-19}$ s$^{-5}$]  (scale of the $DG/Dt$ slide)', fontsize=9.5)
    wm = d['two_F'][wj, wi][band[wj, wi]] * SCALE
    f.set_title(r'window: $2F$ (stored, discrete)', fontsize=11)
    f.text(0.97, 0.04, f'window: {wm.size} parcels, mean {wm.mean():+.1f}\n'
           f'($DG/Dt$: {lims["DGDt_window_mean"]:+.1f})',
           transform=f.transAxes, ha='right', va='bottom', fontsize=9.5,
           bbox=dict(fc='w', ec='0.5', lw=0.6, alpha=0.9), zorder=10)
    _label(f, 'f')
    for ax in (dd, e, f):
        _box(ax, d, kj.start, ki.start, CLOSE, COL_CLOSE, lw=1.6, ls='--')
        _mark(ax, d, p)
        _geo(ax, d, wj, wi, nbins=3)

    fig.suptitle(f'The kinematic prediction $2F$: LLC4320, {d["times"][0][:10]} {tm} UTC, '
                 f'tracked front (track 0, label {d["label"]})', fontsize=14, y=0.995)
    fig.text(0.5, 0.955,
             rf'$2F$ is an instant, not a difference:  $b_{{mid}} = \frac{{1}}{{2}}[b({t0}) + b({t1})]$,  '
             rf'$u_{{mid}} = \frac{{1}}{{2}}[u({t0}) + u({t1})]$  $\rightarrow$  '
             rf'$2F = -2\,(\nabla b)^T(\nabla u)(\nabla b)$ at {tm}',
             ha='center', va='center', fontsize=11.5,
             bbox=dict(boxstyle='round,pad=0.4', fc='#fff7e0', ec='0.4', lw=0.8))
    fig.text(0.5, 0.005, 'grey cells: not parcels.  Dashed box in (d)-(f): the close-up of (b); '
             'square: its parcel.  The same hour, parcels, window and parcel as the $DG/Dt$ slide.',
             ha='center', fontsize=9.5, color='0.3')
    fig.tight_layout(rect=(0, 0.015, 1, 0.94))
    _save(fig, out_png)
    return dict(png=str(out_png), window_mean=float(wm.mean()), window_n=int(wm.size),
                band_mean=float(v.mean()), frac_theta_lt_45=fr,
                parcel=dict(j=p[0], i=p[1], G=float(G), delta=float(dl), sigma=float(sg),
                            theta_deg=float(np.degrees(th)), conv=float(d['conv'][p]),
                            strain=float(d['strain'][p]), two_F_chain=float(d['two_F_chain'][p]),
                            two_F=float(d['two_F'][p]), rel_speed_max=vmax),
                window_rel_speed_p98=wmax)


# ---------------------------------------------------------------------------
# figure 2: convergence + strain = 2F (continuum)
# ---------------------------------------------------------------------------
def fig_2F_split(d, win, out_png):
    band = d['band']
    tm = d['time_mid'][11:16]
    parts = (('conv', d['conv'], r'convergence: $-\delta\, G$'),
             ('strain', d['strain'], r'strain: $|\sigma|\, G \cos 2\theta$'),
             ('two_F_chain', d['two_F_chain'], r'$2F$ (continuum) $= -2\,(\nabla b)^T(\nabla u)(\nabla b)$'))
    disc = float(d['two_F'][band].mean() * SCALE)
    chain = float(d['two_F_chain'][band].mean() * SCALE)
    corr = float(np.corrcoef(d['two_F'][band], d['two_F_chain'][band])[0, 1])
    fr = float(np.mean(d['theta'][band] < np.pi / 4))

    def suptitle(s):
        return (f'convergence + strain = $2F$ on the {int(band.sum()):,} parcels of label '
                f'{d["label"]} ({d["times"][0][:10]} {tm} UTC):   '
                f'{s["conv"]["band_mean"]:+.2f} {s["strain"]["band_mean"]:+.2f} = '
                f'{s["two_F_chain"]["band_mean"]:+.2f}')

    def footer(s):
        return (f'Strain carries {100 * s["strain"]["band_mean"] / s["two_F_chain"]["band_mean"]:.0f}% '
                f'of $2F$ here: {100 * fr:.0f}% of the parcels have their gradient within 45° of the '
                'compressional axis.  The pipeline stores the discretely consistent $2F$: '
                f'{disc:+.2f} on these parcels\n= {disc / chain:.2f} × the continuum value '
                f'(cell-by-cell correlation {corr:.2f}); that is the number compared with $DG/Dt$ '
                '(+4.20).  Top: every parcel; bottom: the window (its own colour scale).  '
                'Units: 10$^{-19}$ s$^{-5}$.')

    res = fig_split(d, win, out_png, parts=parts, suptitle=suptitle, footer=footer)
    res.update(two_F_discrete_mean=disc, two_F_chain_mean=chain, ratio=disc / chain, corr=corr)
    return res


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pair', type=int, default=PAIR)
    p.add_argument('--window', type=str, default=None, help='j0,i0 (default: the dG/dt slide\'s)')
    p.add_argument('--parcel', type=str, default=None, help='j,i (default: the dG/dt slide\'s)')
    p.add_argument('--out-dir', default=str(SLIDE_DIR))
    a = p.parse_args(argv)
    ref = json.loads(DGDT_JSON.with_name(DGDT_JSON.name.format(pair=a.pair)).read_text())
    win = (ref['window']['j0'], ref['window']['i0'])
    par = tuple(ref['parcel_ji'])
    if a.window:
        win = tuple(int(v) for v in a.window.split(','))
    if a.parcel:
        par = tuple(int(v) for v in a.parcel.split(','))
    lims = dict(band=ref['method']['rate_limit'], window=ref['method']['window_rate_limit'],
                DGDt_mean=ref['check']['band_mean'] * SCALE, DGDt_window_mean=ref['method']['window_mean'])
    d = compute(a.pair)
    chk = check(d)
    band = d['band']
    res = dict(pair=a.pair, label=d['label'], time_mid=d['time_mid'], check=chk,
               window=ref['window'], parcel_ji=list(par), colour_limits_from=DGDT_JSON.name.format(pair=a.pair),
               band=dict(theta_median_deg=float(np.degrees(np.median(d['theta'][band]))),
                         delta_median_abs=float(np.median(np.abs(d['delta'][band]))),
                         sigma_median=float(np.median(d['sigma'][band]))))
    out = pathlib.Path(a.out_dir)
    res['method'] = fig_method(d, win, par, lims, out / '2F_method.png')
    res['split'] = fig_2F_split(d, win, out / '2F_split.png')
    path = SUMMARY.with_name(SUMMARY.name.format(pair=a.pair))
    path.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    main()
