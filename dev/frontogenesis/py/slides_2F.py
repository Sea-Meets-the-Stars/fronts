"""Slides: how ``2F`` (the kinematic frontogenesis prediction) of a tracked front is computed
(frontogenesis_slides.md, Figures / 2F, prompts 2-9; Q&A 2F-Q1..Q9, NS-Q1..Q5, DF-Q1..Q5).

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
  ``dGdt_method.png`` (c), (f), so the two slides compare directly.  Since 2F prompt 3 it is
  in the slide format of ``dGdt_method.png`` (10 x 4.45 in, 300 dpi, no title, footnote or
  parcel numbers -- those are in the JSON), panel (a) and the legend shared via
  ``slides_dGdt.slide_front`` / ``slide_legend_handles``, and (b) and (d) on one quiver scale.
  2F prompt 5: text about 20 % smaller (``FS_2F``, ``FS_2F_LEGEND``) and the panels
  enlarged; the colour bars are hung beside each panel after its aspect is applied.

The summary JSON also carries ``decomposition``: the convergence and strain growth rates
``(1/G) DG/Dt = -delta + |sigma| cos 2 theta`` at the parcel and (G-weighted) over the band --
the numbers of the report's §2.5, checked by ``report/check_frontogenesis_report.py``.
* ``slides/2F_split.png`` -- (2F prompt 7; NS-Q1..Q3) convergence + strain = the **stored,
  discrete** ``2F``, from the exact split of the discrete form (:func:`discrete_parts`: each
  one-sided velocity gradient split into squeeze, strain and rotation; the rotation leaves a
  grid remainder, mean ~0, not drawn); every parcel on top, the window below, in the slide
  format of ``2F_method.png`` and on the ``DG/Dt`` slide's colour scales.
* ``slides/2F_discrete.png`` -- (2F prompts 8-9; DF-Q1..Q5) the calculation actually compared:
  (a) the equations of the discretely consistent form, (b) the stencil at the close-up parcel
  (gradient at the centre and at the four neighbours, velocity difference to each neighbour),
  (c) the four neighbour terms against the same terms with the centre gradient, which sum to
  the chain-rule ``2F`` (asserted band-wide in :func:`check`).

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
from matplotlib.patches import FancyArrowPatch, Polygon

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import m2_qa                                                         # noqa: E402
import inputs as inp                                                 # noqa: E402
import operators as op                                               # noqa: E402
import slides_dGdt as sd                                             # noqa: E402
from slides_dGdt import (_ll, _corners, _box, _geo, _crop, _cells, _blank, _lasso, _mark,   # noqa: E402
                         _log,
                         slide_front, slide_legend_handles, _edge_labels,
                         WIN, CLOSE, SCALE, CMAP_G, CMAP_R, CMAP_B, COL_WIN, COL_CLOSE,
                         B_LEVELS, FS_DG, FS_DG_LEGEND)
from slides_tracking import BAND, STRUCT, SLIDE_SIZE, METHOD_DPI                             # noqa: E402

DATA_DIR = HERE.parent / 'data'
SLIDE_DIR = HERE.parent / 'slides'
DGDT_JSON = DATA_DIR / 'slides_dGdt_{pair}.json'      # window, parcel and colour scales of the dG/dt slide
SUMMARY = DATA_DIR / 'slides_2F_{pair}.json'

PAIR = 36
COL_GRAD = '#d62728'         # grad b
COL_COMP = 'k'               # compressional axis
COL_VEL = '#6a3d9a'          # relative velocity
AXIS_CELLS = 1.6             # half-length of the drawn axes in the close-up [cells]
FS_2F = 10.5                 # pt, panel titles of 2F_method.png (2F prompt 5: ~20 % below FS_DG)
FS_2F_LEGEND = 9             # pt, its ticks, colour bars and legend (FS_2F - 1.5)
OMEGA = 7.2921e-5            # s^-1, for f in the decomposition summary
QUIVER_SCALE = 0.10 / 0.22    # slide: m s^-1 per inch, shared by (b) and (d) (max ~10 cm/s in both)
QUIVER_KEY = 0.10             # m s^-1
TICK_CELLS = 0.45            # half-length of the compressional-axis ticks in the window [cells]
TOL_STORED = 1e-5            # max |recomputed - stored| / rms(stored) on the band
TOL_JSON = 1e-6
TOL_SPLIT = 1e-10            # convergence + strain = continuum 2F, relative to its rms


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------
_np2 = sd._np2


def discrete_parts(b, U, V, grid_ds, grid):
    """Exact split of the stored, discretely consistent ``2F`` (NS-Q1 (a), 2F prompt 7).

    ``_frontogenesis_discrete`` is ``F = -1/2 sum_sides sum_k g_k sum_j J^side_jk g4_j(x +- e_k)``
    with ``J^side`` the one-sided velocity gradient (velocity differences to the +/- neighbours
    over the face spacing) and ``g4`` the 4th-order gradient at those neighbours.  ``F`` is linear
    in ``J^side``, so splitting each one-sided ``J`` into squeeze ``delta/2 I``, strain
    ``1/2 [[sn, ss], [ss, -sn]]`` and rotation ``zeta/2 [[0, -1], [1, 0]]`` splits ``F`` exactly.
    The rotation part no longer vanishes (``g4`` is taken at the neighbours): a grid remainder,
    zero for a linear ``b``.  Same arithmetic as the operator, model basis (``F`` is invariant).
    Returns ``2F`` for ``total``, ``div``, ``strain``, ``rot`` as (nj, ni) arrays; ``terms``, the
    four neighbour terms ``{(axis, side): 2F part}`` (``axis`` 'i' or 'j', ``side`` +1 / -1), which
    sum to ``total``; ``centre`` and ``centre_terms``, the same with the centre gradient ``g(x)`` in
    place of ``g4(x +- e_k)`` -- which is the chain-rule ``2F`` (2F prompt 9, DF-Q5); and the
    model-basis gradients ``gi, gj, g4i, g4j``."""
    bx, by = op.grad_b(b, grid_ds, grid)
    g_i, g_j = op.model_basis_gradient(bx, by, grid_ds)
    u_c, v_c = op.centred_model_velocity(U, V, grid)
    gi, gj, uc, vc = (_np2(a) for a in (g_i, g_j, u_c, v_c))
    S = op._shift
    g4i = (4 * gi - 0.5 * (S(gi, 1, 1) + S(gi, 1, -1))) / 3
    g4j = (4 * gj - 0.5 * (S(gj, 0, 1) + S(gj, 0, -1))) / 3
    dxc, dyc = op._positional_metric(grid_ds, 'dxC'), op._positional_metric(grid_ds, 'dyC')
    hx, hy = {+1: S(dxc, 1, 1), -1: dxc}, {+1: S(dyc, 0, 1), -1: dyc}

    def diff(a, axis, s):
        return (S(a, axis, 1) - a) if s > 0 else (a - S(a, axis, -1))

    out = {k: np.zeros_like(gi) for k in ('total', 'div', 'strain', 'rot', 'centre')}
    out.update(terms={}, centre_terms={}, gi=gi, gj=gj, g4i=g4i, g4j=g4j)
    for s in (+1, -1):
        Jui, Jvi = diff(uc, 1, s) / hx[s], diff(vc, 1, s) / hx[s]
        Juj, Jvj = diff(uc, 0, s) / hy[s], diff(vc, 0, s) / hy[s]
        dl, zt, sn, ss = Jui + Jvj, Jvi - Juj, Jui - Jvj, Jvi + Juj
        ni, nj = (S(g4i, 1, s), S(g4j, 1, s)), (S(g4i, 0, s), S(g4j, 0, s))

        def two_f(a, b_, c, e):       # J = [[J_ui, J_uj], [J_vi, J_vj]]; returns 2F
            return -(gi * (a * ni[0] + c * ni[1]) + gj * (b_ * nj[0] + e * nj[1]))
        out['total'] += two_f(Jui, Juj, Jvi, Jvj)
        out['div'] += two_f(0.5 * dl, 0, 0, 0.5 * dl)
        out['strain'] += two_f(0.5 * sn, 0.5 * ss, 0.5 * ss, -0.5 * sn)
        out['rot'] += two_f(0, -0.5 * zt, 0.5 * zt, 0)
        out['terms'][('i', s)] = -gi * (Jui * ni[0] + Jvi * ni[1])
        out['terms'][('j', s)] = -gj * (Juj * nj[0] + Jvj * nj[1])
        out['centre_terms'][('i', s)] = -gi * (Jui * gi + Jvi * gj)
        out['centre_terms'][('j', s)] = -gj * (Juj * gi + Jvj * gj)
        out['centre'] += out['centre_terms'][('i', s)] + out['centre_terms'][('j', s)]
    return out


def compute(pair=PAIR, data_dir=None):
    """The midpoint fields of ``pair`` and every term of ``2F`` on the full tile."""
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    raw, g, grid, _, _ = m2_qa.open_inputs()
    h0, h1 = m2_qa.hour_ds(raw, g, pair), m2_qa.hour_ds(raw, g, pair + 1)
    f0, f1 = inp.filtered(h0, 0), inp.filtered(h1, 0)       # L = 0
    bm, U, V = (inp.midpoint(a, b) for a, b in zip(f0, f1))  # b_mid, U_mid, V_mid (budget.py)
    two_F = 2.0 * _np2(op.frontogenesis(bm, U, V, h0, grid, form='discrete'))
    two_F_chain = 2.0 * _np2(op.frontogenesis(bm, U, V, h0, grid, form='chain'))
    dparts = discrete_parts(bm, U, V, h0, grid)
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
             zeta=_np2(vx - uy),                                        # vorticity (intuition.py)
             conv=-delta * G, strain=smag * G * np.cos(2 * theta),
             two_F_div=dparts['div'], two_F_strain=dparts['strain'], two_F_rot=dparts['rot'],
             two_F_parts_total=dparts['total'], two_F_centre=dparts['centre'], disc=dparts,
             CS=cs, SN=sn_,
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
    # the exact discrete split: its total is the stored two_F, and its parts sum to it
    st = d['stored']['two_F'][band]
    rms_d = np.sqrt(np.mean(st ** 2))
    tot = float(np.max(np.abs(d['two_F_parts_total'][band] - st)) / rms_d)
    psum = float(np.max(np.abs(d['two_F_div'] + d['two_F_strain'] + d['two_F_rot']
                               - d['two_F_parts_total'])[band]) / rms_d)
    if tot > TOL_STORED or psum > TOL_SPLIT:
        raise AssertionError(f'discrete split: total {tot:.2e}, parts {psum:.2e} of the rms')
    out.update(discrete_parts_total_err_over_rms=tot, discrete_parts_sum_err_over_rms=psum)
    rms_c = np.sqrt(np.mean(d['two_F_chain'][band] ** 2))
    cen = float(np.max(np.abs(d['two_F_centre'] - d['two_F_chain'])[band]) / rms_c)
    if cen > TOL_SPLIT:
        raise AssertionError(f'centre-gradient sum != chain-rule 2F ({cen:.2e} of the rms)')
    out['centre_sum_vs_chain_err_over_rms'] = cen
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


def _axis(ax, d, p, ang, half, col, lw=1.6, inward=True, z=8, ms=14):
    """The compressional axis through ``p``: two arrows pointing inward along ``ang``."""
    x0, y0 = _ll(d, *p)
    x0, y0 = x0[0], y0[0]
    dx, dy = _offset(d, p, ang, half)
    for sgn in (1, -1):
        tail, head = (x0 + sgn * dx, y0 + sgn * dy), (x0 + sgn * 0.35 * dx, y0 + sgn * 0.35 * dy)
        if not inward:
            tail, head = head, tail
        ax.add_patch(FancyArrowPatch(tail, head, arrowstyle='-|>', mutation_scale=ms, color=col,
                                     lw=lw, zorder=z))


def _nice(v):
    e = 10 ** np.floor(np.log10(v))
    return float(min((1, 2, 5, 10), key=lambda m: abs(m * e - v)) * e)


def _quiver(ax, d, sj, si, du, dv, scale, key_val, key_xy=(0.84, 1.03), every=1,
            labelpos='E', fs=9, width=0.004):
    X, Y = d['X'][sj, si][::every, ::every], d['Y'][sj, si][::every, ::every]
    q = ax.quiver(X, Y, du[::every, ::every], dv[::every, ::every], color=COL_VEL, angles='uv',
                  scale_units='inches', scale=scale, width=width, headwidth=4, zorder=7)
    if key_val is None:
        return q
    ax.quiverkey(q, *key_xy, key_val, f'{key_val * 100:.0f} cm s$^{{-1}}$', labelpos=labelpos,
                 coordinates='axes', fontproperties={'size': fs}, color=COL_VEL)
    return q


# ---------------------------------------------------------------------------
# figure 1: the method
# ---------------------------------------------------------------------------
def fig_method(d, win, p, lims, out_png, fs=FS_2F):
    """The method on one 16:9 slide under its own title (2F prompt 3), in the format of
    ``dGdt_method.png``: no suptitle, footnote or parcel numbers (those stay in the JSON),
    ``SLIDE_SIZE`` inches at ``METHOD_DPI``, axes placed in inches."""
    band = d['band']
    cj, ci = _crop(d)
    wj, wi = slice(win[0], win[0] + WIN), slice(win[1], win[1] + WIN)
    h = CLOSE // 2
    kj, ki = slice(p[0] - h, p[0] + h + 1), slice(p[1] - h, p[1] + h + 1)
    t0, t1, tm = d['times'][0][11:16], d['times'][1][11:16], d['time_mid'][11:16]
    cell = _cell_deg(d, p)
    W, H = SLIDE_SIZE
    small = fs - 1.5

    with plt.rc_context({'font.size': small, 'xtick.labelsize': small,
                         'ytick.labelsize': small}):
        fig = plt.figure(figsize=SLIDE_SIZE, dpi=METHOD_DPI)   # quiver's inch scale uses fig.dpi

        def axes_in(x, y, w, hh):
            return fig.add_axes([x / W, y / H, w / W, hh / H])

        def title(ax, text):
            ax.set_title(text, fontsize=fs, pad=4)

        bars = []

        def cbar(m, ax, label, **kw):
            # placed after the draw, beside the axes as the aspect leaves it
            bars.append((m, ax, label, kw))

        # 2F prompt 5: smaller text, larger panels (anchored left; the colour bars follow)
        r1, r2, hh = 2.34, 0.55, 1.56
        cols = (0.72, 3.78, 6.88)
        a, b, c = (axes_in(x, r1, 2.30, hh) for x in cols)
        dd, e, f = (axes_in(x, r2, 2.30, hh) for x in cols)
        for ax in (a, b, c, dd, e, f):
            ax.set_anchor('W')

        # the two hours enter only through their average (2F-Q1)
        fig.text(0.5, 4.27 / H,
                 rf'$2F$ is an instant:  $b_{{mid}} = \frac{{1}}{{2}}[b({t0}) + b({t1})]$,  '
                 rf'likewise $u_{{mid}}$  $\rightarrow$  '
                 rf'$2F = -\delta G + |\sigma| G\cos 2\theta$ at {tm}',
                 ha='center', va='center', fontsize=small,
                 bbox=dict(boxstyle='round,pad=0.3', fc='#fff7e0', ec='0.4', lw=0.8))

        # (a) the front (shared with the dG/dt slide)
        m = slide_front(a, d, win, small)
        title(a, f'(a) Label {d["label"]}, $G$ at {tm}')
        cbar(m, a, r'$\log_{10} G$')

        # (b) one parcel: b_mid, grad b, the compressional axis, the velocity relative to the parcel
        bb = d['bm'][kj, ki]
        _cells(b, d, d['bm'], kj, ki, cmap=CMAP_B, alpha=0.85,
               vmin=bb.min() - 0.15 * np.ptp(bb), vmax=bb.max() + 0.15 * np.ptp(bb))
        Xc, Yc = _corners(d, kj.start, kj.stop, ki.start, ki.stop)
        for k in range(Xc.shape[0]):
            b.plot(Xc[k], Yc[k], color='0.45', lw=0.5, zorder=2)
        for k in range(Xc.shape[1]):
            b.plot(Xc[:, k], Yc[:, k], color='0.45', lw=0.5, zorder=2)
        b.set_xlim(Xc.min(), Xc.max())
        b.set_ylim(Yc.min(), Yc.max())
        x0, y0 = _ll(d, *p)
        kj2, ki2 = slice(kj.start - 1, kj.stop + 1), slice(ki.start - 1, ki.stop + 1)
        cs = b.contour(d['X'][kj2, ki2], d['Y'][kj2, ki2], d['bm'][kj2, ki2] * 1e3,
                       levels=B_LEVELS, colors='k', linewidths=0.8, zorder=3)
        _edge_labels(b, cs, (x0[0], y0[0]), fs=small - 1)
        du = d['u_e'][kj, ki] - d['u_e'][p]
        dv = d['v_n'][kj, ki] - d['v_n'][p]
        vmax = float(np.nanmax(np.hypot(du, dv)))
        _quiver(b, d, kj, ki, du, dv, scale=QUIVER_SCALE, key_val=QUIVER_KEY,
                key_xy=(1.16, 0.10), labelpos='N', fs=small, width=0.008)
        _axis(b, d, p, d['psi_c'][p], AXIS_CELLS * cell, COL_COMP, lw=1.6, ms=10)
        ang_g = float(np.arctan2(d['by'][p], d['bx'][p]))
        gx, gy = _offset(d, p, ang_g, 1.3 * cell)
        b.add_patch(FancyArrowPatch((x0[0], y0[0]), (x0[0] + gx, y0[0] + gy), arrowstyle='-|>',
                                    mutation_scale=12, color=COL_GRAD, lw=2.2, zorder=9))
        b.scatter(x0, y0, s=20, c='w', edgecolor='k', lw=1.1, zorder=10)
        b.set_xticks([]); b.set_yticks([])
        title(b, f'(b) One parcel at {tm}')

        # (c) every parcel: stored 2F, on the DG/Dt slide's scale
        v = d['two_F'][band] * SCALE
        _blank(c, d, cj, ci, band)
        m = _cells(c, d, d['two_F'] * SCALE, cj, ci, mask=band, cmap=CMAP_R,
                   norm=TwoSlopeNorm(0.0, -lims['band'], lims['band']))
        _box(c, d, win[0], win[1], WIN, COL_WIN, lw=1.8)
        _geo(c, d, cj, ci, nbins=2)
        c.set_xticks([]); c.set_yticks([])
        title(c, r'(c) $2F$, every parcel')
        cbar(m, c, r'$2F$ [10$^{-19}$ s$^{-5}$]', extend='both')
        c.text(0.97, 0.03, f'mean {v.mean():+.2f}', transform=c.transAxes, ha='right',
               va='bottom', fontsize=fs, fontweight='bold',
               bbox=dict(fc='w', ec='k', lw=0.8, alpha=0.9, pad=2), zorder=10)

        # (d) the window: G with the flow relative to the window mean
        lg = _log(d['G'])
        vw = np.nanpercentile(lg[wj, wi][band[wj, wi]], [1, 99])
        m = _cells(dd, d, lg, wj, wi, cmap=CMAP_G, vmin=vw[0] - 1.0, vmax=vw[1])
        _lasso(dd, d, wj, wi, lw=1.0)
        du = d['u_e'][wj, wi] - np.nanmean(d['u_e'][wj, wi])
        dv = d['v_n'][wj, wi] - np.nanmean(d['v_n'][wj, wi])
        wmax = float(np.nanpercentile(np.hypot(du, dv), 98))
        _quiver(dd, d, wj, wi, du, dv, scale=QUIVER_SCALE, key_val=None, width=0.006)
        cbar(m, dd, r'$\log_{10} G$')
        title(dd, r'(d) $G$, relative flow')

        # (e) the window: cos 2 theta on the parcels, the compressional axis on the ridge
        _blank(e, d, wj, wi, band)
        m = _cells(e, d, np.cos(2 * d['theta']), wj, wi, mask=band, cmap='PuOr_r', vmin=-1, vmax=1)
        for jj, ii in zip(*np.where(d['skel'][wj, wi])):
            q = (wj.start + jj, wi.start + ii)
            xq, yq = _ll(d, *q)
            dx, dy = _offset(d, q, d['psi_c'][q], TICK_CELLS * cell)
            e.plot([xq[0] - dx, xq[0] + dx], [yq[0] - dy, yq[0] + dy], color=COL_COMP, lw=1.1, zorder=7)
        cbar(m, e, r'$\cos 2\theta$')
        fr = float(np.mean(d['theta'][band] < np.pi / 4))
        title(e, r'(e) Alignment $\cos 2\theta$')

        # (f) the window: 2F, on the DG/Dt slide's window scale
        _blank(f, d, wj, wi, band)
        m = _cells(f, d, d['two_F'] * SCALE, wj, wi, mask=band, cmap=CMAP_R,
                   norm=TwoSlopeNorm(0.0, -lims['window'], lims['window']))
        cbar(m, f, r'$2F$ [10$^{-19}$ s$^{-5}$]', extend='both')
        wm = d['two_F'][wj, wi][band[wj, wi]] * SCALE
        title(f, r'(f) $2F$, window')
        f.text(0.97, 0.03, f'mean {wm.mean():+.1f}', transform=f.transAxes, ha='right',
               va='bottom', fontsize=small, bbox=dict(fc='w', ec='0.5', lw=0.6, alpha=0.9, pad=2),
               zorder=10)
        for ax in (dd, e, f):
            _box(ax, d, kj.start, ki.start, CLOSE, COL_CLOSE, lw=1.4, ls='--')
            _mark(ax, d, p, s=30)
            _geo(ax, d, wj, wi, nbins=3)
            ax.set_xticks([]); ax.set_yticks([])

        handles = slide_legend_handles() + [
            Line2D([], [], color=COL_GRAD, lw=2.2, label=r'$\nabla b$ (direction)'),
            Line2D([], [], color=COL_COMP, lw=1.6, marker='>', ms=5, label='compressional axis'),
            Line2D([], [], color=COL_VEL, lw=1.2, label=r'relative flow'),
            Line2D([], [], color='k', lw=0.8, label=r'$b_{mid}$ [10$^{-3}$ m s$^{-2}$]')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4,
                   frameon=False, fontsize=FS_2F_LEGEND, handlelength=1.5, handletextpad=0.4,
                   columnspacing=1.2)
        fig.canvas.draw()                     # apply the aspects, then hang the colour bars
        for m, ax, label, kw in bars:
            pos = ax.get_position()
            cb = fig.colorbar(m, cax=fig.add_axes([pos.x1 + 0.08 / W, pos.y0 + 0.06 * pos.height,
                                                   0.09 / W, 0.88 * pos.height]), **kw)
            cb.set_label(label, fontsize=small, labelpad=2)
            cb.ax.tick_params(labelsize=small)
        out_png = pathlib.Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METHOD_DPI)
        plt.close(fig)
    G, dl, sg, th = d['G'][p], d['delta'][p], d['sigma'][p], d['theta'][p]
    return dict(png=str(out_png), size_in=list(SLIDE_SIZE), dpi=METHOD_DPI,
                window_mean=float(wm.mean()), window_n=int(wm.size),
                band_mean=float(v.mean()), frac_theta_lt_45=fr,
                DGDt_band_mean=lims['DGDt_mean'], DGDt_window_mean=lims['DGDt_window_mean'],
                parcel=dict(j=p[0], i=p[1], G=float(G), delta=float(dl), sigma=float(sg),
                            theta_deg=float(np.degrees(th)), conv=float(d['conv'][p]),
                            strain=float(d['strain'][p]), two_F_chain=float(d['two_F_chain'][p]),
                            two_F=float(d['two_F'][p]), rel_speed_max=vmax),
                window_rel_speed_p98=wmax)


# ---------------------------------------------------------------------------
# figure 2: convergence + strain = the stored, discrete 2F (2F prompt 7; NS-Q1..Q3)
# ---------------------------------------------------------------------------
def fig_2F_split(d, win, lims, out_png, fs=FS_2F):
    """``slides/2F_split.png`` on one 16:9 slide (the format of ``2F_method.png``):
    convergence + strain = stored ``two_F``, from the exact split of the discrete form
    (:func:`discrete_parts`); every parcel on top, the window below, on the colour scales
    of the ``DG/Dt`` slide.  The rotation remainder (mean ~0) is not drawn (NS-Q1 (a))."""
    band = d['band']
    cj, ci = _crop(d)
    wj, wi = slice(win[0], win[0] + WIN), slice(win[1], win[1] + WIN)
    W, H = SLIDE_SIZE
    small = fs - 1.5
    parts = (('div', d['two_F_div'], 'Convergence'),
             ('strain', d['two_F_strain'], 'Strain'),
             ('two_F', d['two_F'], r'$2F$ (stored, discrete)'))
    res = {}
    with plt.rc_context({'font.size': small, 'xtick.labelsize': small,
                         'ytick.labelsize': small}):
        fig = plt.figure(figsize=SLIDE_SIZE, dpi=METHOD_DPI)

        def axes_in(x, y, w, hh):
            return fig.add_axes([x / W, y / H, w / W, hh / H])

        r1, r2, hh = 2.42, 0.42, 1.74
        cols = (0.30, 3.40, 6.50)
        top = [axes_in(x, r1, 2.45, hh) for x in cols]
        bot = [axes_in(x, r2, 2.45, hh) for x in cols]
        for ax in top + bot:
            ax.set_anchor('W')
        letters = iter('abcdef')
        for row, axs in (('band', top), ('window', bot)):
            sj, si = (cj, ci) if row == 'band' else (wj, wi)
            norm = TwoSlopeNorm(0.0, -lims[row], lims[row])
            m_in = band[sj, si]
            for ax, (key, fld, name) in zip(axs, parts):
                _blank(ax, d, sj, si, band)
                m = _cells(ax, d, fld * SCALE, sj, si, mask=band, cmap=CMAP_R, norm=norm)
                _geo(ax, d, sj, si, nbins=2)
                ax.set_xticks([]); ax.set_yticks([])
                v = fld[sj, si][m_in] * SCALE
                where = 'every parcel' if row == 'band' else 'window'
                ax.set_title(f'({next(letters)}) {name}, {where}', fontsize=fs, pad=4)
                if row == 'band':
                    _box(ax, d, win[0], win[1], WIN, COL_WIN, lw=1.8)
                    txt, kw = f'mean {v.mean():+.2f}', dict(fontsize=fs, fontweight='bold')
                    box = dict(fc='w', ec='k', lw=0.8, alpha=0.9, pad=2)
                else:
                    txt, kw = f'mean {v.mean():+.1f}', dict(fontsize=small)
                    box = dict(fc='w', ec='0.5', lw=0.6, alpha=0.9, pad=2)
                ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom',
                        bbox=box, zorder=10, **kw)
                res.setdefault(key, {})[f'{row}_mean'] = float(v.mean())
                res[key][f'{row}_rms'] = float(np.sqrt(np.mean(v ** 2)))
            res.setdefault('mappables', []).append((m, axs[-1]))
        fig.legend(handles=[Line2D([], [], color=COL_WIN, lw=2.0, label='window (d–f)'),
                            Line2D([], [], color='#d9d9d9', marker='s', ls='', ms=7,
                                   label='grey: not parcels')],
                   loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
                   fontsize=small, handlelength=1.5, columnspacing=2.0)
        fig.canvas.draw()                     # apply the aspects, then hang bars and signs
        for m, ax in res.pop('mappables'):
            pos = ax.get_position()
            cb = fig.colorbar(m, cax=fig.add_axes([pos.x1 + 0.08 / W, pos.y0 + 0.06 * pos.height,
                                                   0.09 / W, 0.88 * pos.height]), extend='both')
            cb.set_label(r'$2F$ [10$^{-19}$ s$^{-5}$]', fontsize=small, labelpad=2)
            cb.ax.tick_params(labelsize=small)
        for axs in (top, bot):
            p = [ax.get_position() for ax in axs]
            for sign, l, r in (('+', p[0], p[1]), ('=', p[1], p[2])):
                fig.text(0.5 * (l.x1 + r.x0), 0.5 * (l.y0 + l.y1), sign, ha='center',
                         va='center', fontsize=2 * fs, fontweight='bold')
        out_png = pathlib.Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METHOD_DPI)
        plt.close(fig)
    wb = band[wj, wi]
    res['rot'] = dict(band_mean=float(d['two_F_rot'][band].mean() * SCALE),
                      band_rms=float(np.sqrt(np.mean(d['two_F_rot'][band] ** 2)) * SCALE),
                      window_mean=float(d['two_F_rot'][wj, wi][wb].mean() * SCALE))
    res.update(png=str(out_png), size_in=list(SLIDE_SIZE), dpi=METHOD_DPI,
               strain_share=res['strain']['band_mean'] / res['two_F']['band_mean'])
    return res


# ---------------------------------------------------------------------------
# figure 3: the discretely consistent 2F, as computed (2F prompts 8-9; DF-Q1..Q5)
# ---------------------------------------------------------------------------
def _compass(cs, sn, axis, side):
    """Geographic name of the neighbour ``x + side * e_axis`` (model axes rotated by CS/SN)."""
    vx, vy = (cs, sn) if axis == 'i' else (-sn, cs)
    ang = np.degrees(np.arctan2(side * vy, side * vx)) % 360
    return 'ENWS'[int(((ang + 45) % 360) // 90)]


def fig_discrete(d, p, lims, out_png, fs=FS_2F):
    """``slides/2F_discrete.png``: (a) the equations of the discretely consistent ``2F``;
    (b) the stencil at the close-up parcel -- the gradient at the centre and at the four
    neighbours, and the velocity difference to each neighbour; (c) the stored ``2F`` at every
    parcel (2F prompt 10: the bar chart of the neighbour terms is gone; its numbers stay in the
    JSON).  ``grad b`` throughout, not ``g``."""
    W, H = SLIDE_SIZE
    small = fs - 1.5
    dc = d['disc']
    cs, sn = float(d['CS'][p]), float(d['SN'][p])
    h = CLOSE // 2
    kj, ki = slice(p[0] - h, p[0] + h + 1), slice(p[1] - h, p[1] + h + 1)
    cell = _cell_deg(d, p)
    keys = [('i', 1), ('i', -1), ('j', 1), ('j', -1)]
    nbr = {k: (p[0], p[1] + k[1]) if k[0] == 'i' else (p[0] + k[1], p[1]) for k in keys}
    name = {k: _compass(cs, sn, *k) for k in keys}
    order = sorted(keys, key=lambda k: 'NSEW'.index(name[k]))

    def geo(vi, vj):                      # model basis -> (east, north)
        return vi * cs - vj * sn, vi * sn + vj * cs

    with plt.rc_context({'font.size': small, 'xtick.labelsize': small,
                         'ytick.labelsize': small}):
        fig = plt.figure(figsize=SLIDE_SIZE, dpi=METHOD_DPI)

        def axes_in(x, y, w, hh):
            return fig.add_axes([x / W, y / H, w / W, hh / H])

        # (a) the equations
        a = axes_in(0.12, 0.62, 3.15, 3.55)
        a.axis('off')
        a.set_title('(a) The calculation', fontsize=fs, pad=4, loc='left')
        lines = [
            (r'$F = -\left(u_x\, b_x^2 + (u_y + v_x)\, b_x b_y + v_y\, b_y^2\right)$', fs),
            (r'On the grid ($\mathsf{L}_k$: the centred difference over two cells):', small - 0.5),
            (r'$\dfrac{D}{Dt}\,\mathsf{L}_k b = -\,[\mathsf{L}_k,\ \mathbf{u}\cdot\nabla]\,b \equiv -\,C_k$', fs),
            (r'$C_k = \dfrac{1}{2}\left[\dfrac{\Delta\mathbf{u}_k^{+}\cdot\nabla b(\mathbf{x}+\mathbf{e}_k)}{h}'
             r' + \dfrac{\Delta\mathbf{u}_k^{-}\cdot\nabla b(\mathbf{x}-\mathbf{e}_k)}{h}\right]$', small),
            (r'$\Delta\mathbf{u}_k^{\pm} = \pm\,[\,\mathbf{u}(\mathbf{x}\pm\mathbf{e}_k) - \mathbf{u}(\mathbf{x})\,]$', small),
            ('$k = x$ (E–W) or $y$ (N–S);  $\\mathbf{e}_k$: one cell along $k$;\n'
             '$\\mathbf{x}\\pm\\mathbf{e}_k$: the neighbours in (b);  $h$: the cell spacing', small),
            (r'$2F = -2\,\sum_k\, (\nabla b)_k(\mathbf{x})\, C_k$', fs + 1),
            ('chain rule: the same, with $\\nabla b(\\mathbf{x})$\n'
             'in place of $\\nabla b(\\mathbf{x}\\pm\\mathbf{e}_k)$', small),
        ]
        ys = (0.97, 0.86, 0.79, 0.62, 0.45, 0.37, 0.22, 0.07)
        for (txt, size), y in zip(lines, ys):
            kw = dict(bbox=dict(boxstyle='round,pad=0.35', fc='#fff7e0', ec='0.4', lw=0.8)) \
                if txt.startswith(r'$2F') else {}
            a.text(0.0, y, txt, transform=a.transAxes, ha='left', va='top', fontsize=size, **kw)

        # (b) the stencil at the parcel
        b = axes_in(3.62, 0.70, 2.95, 3.45)
        b.set_anchor('C')
        bb = d['bm'][kj, ki]
        _cells(b, d, d['bm'], kj, ki, cmap=CMAP_B, alpha=0.6,
               vmin=bb.min() - 0.15 * np.ptp(bb), vmax=bb.max() + 0.15 * np.ptp(bb))
        Xc, Yc = _corners(d, kj.start, kj.stop, ki.start, ki.stop)
        for k in range(Xc.shape[0]):
            b.plot(Xc[k], Yc[k], color='0.55', lw=0.5, zorder=2)
        for k in range(Xc.shape[1]):
            b.plot(Xc[:, k], Yc[:, k], color='0.55', lw=0.5, zorder=2)
        kj2, ki2 = slice(kj.start - 1, kj.stop + 1), slice(ki.start - 1, ki.stop + 1)
        b.contour(d['X'][kj2, ki2], d['Y'][kj2, ki2], d['bm'][kj2, ki2] * 1e3, levels=B_LEVELS,
                  colors='k', linewidths=0.7, alpha=0.6, zorder=3)
        b.set_xlim(Xc.min(), Xc.max())
        b.set_ylim(Yc.min(), Yc.max())
        b.set_xticks([]); b.set_yticks([])
        for q, lw in [(p, 2.4)] + [(nbr[k], 1.4) for k in keys]:
            Xa, Ya = _corners(d, q[0], q[0] + 1, q[1], q[1] + 1)
            b.add_patch(Polygon(np.c_[Xa[[0, 0, 1, 1], [0, 1, 1, 0]], Ya[[0, 0, 1, 1], [0, 1, 1, 0]]],
                                closed=True, fill=False, ec='k', lw=lw, zorder=6))
        gmag = float(np.hypot(dc['gi'][p], dc['gj'][p]))
        dus = {k: (d['u_e'][nbr[k]] - d['u_e'][p], d['v_n'][nbr[k]] - d['v_n'][p]) for k in keys}
        dumax = max(np.hypot(*v) for v in dus.values())

        def arrow(q, ve, vn, length, col, lw, ms):
            x0, y0 = _ll(d, *q)
            dx, dy = _offset(d, q, np.arctan2(vn, ve), length)
            b.add_patch(FancyArrowPatch((x0[0], y0[0]), (x0[0] + dx, y0[0] + dy), arrowstyle='-|>',
                                        mutation_scale=ms, color=col, lw=lw, zorder=9,
                                        shrinkA=0, shrinkB=0))
        ge, gn = geo(dc['gi'][p], dc['gj'][p])
        arrow(p, ge, gn, 0.85 * cell, COL_GRAD, 3.0, 13)
        for k in keys:
            q = nbr[k]
            ge, gn = geo(dc['g4i'][q], dc['g4j'][q])
            arrow(q, ge, gn, 0.85 * cell * np.hypot(ge, gn) / gmag, '#ff7f7f', 2.0, 10)
            arrow(q, *dus[k], 0.85 * cell * np.hypot(*dus[k]) / dumax, COL_VEL, 2.0, 10)
            Xa, Ya = _corners(d, q[0], q[0] + 1, q[1], q[1] + 1)
            b.text(Xa.min() + 0.08 * (Xa.max() - Xa.min()), Ya.max() - 0.08 * (Ya.max() - Ya.min()),
                   name[k], ha='left', va='top', fontsize=fs, fontweight='bold', zorder=10)
        b.scatter(*_ll(d, *p), s=18, c='w', edgecolor='k', lw=1.0, zorder=10)
        b.set_title('(b) The stencil at one parcel', fontsize=fs, pad=4)

        # (c) the stored 2F at every parcel, the parcel of (b) marked (2F prompt 10)
        disc = [float(dc['terms'][k][p] * SCALE) for k in order]
        cent = [float(dc['centre_terms'][k][p] * SCALE) for k in order]
        band = d['band']
        cj, ci = _crop(d)
        c = axes_in(6.62, 0.95, 2.45, 3.20)
        c.set_anchor('W')
        _blank(c, d, cj, ci, band)
        m = _cells(c, d, d['two_F'] * SCALE, cj, ci, mask=band, cmap=CMAP_R,
                   norm=TwoSlopeNorm(0.0, -lims['band'], lims['band']))
        _mark(c, d, p, s=40)
        _geo(c, d, cj, ci, nbins=2)
        c.set_xticks([]); c.set_yticks([])
        v = d['two_F'][band] * SCALE
        c.text(0.97, 0.03, f'mean {v.mean():+.2f}', transform=c.transAxes, ha='right', va='bottom',
               fontsize=fs, fontweight='bold', bbox=dict(fc='w', ec='k', lw=0.8, alpha=0.9, pad=2),
               zorder=10)
        c.set_title(r'(c) $2F$ at every parcel', fontsize=fs, pad=4)

        fig.legend(handles=[Line2D([], [], color=COL_GRAD, lw=3.0, label=r'$\nabla b$ at the centre'),
                            Line2D([], [], color='#ff7f7f', lw=2.0,
                                   label=r'$\nabla b$ at a neighbour (length $\propto |\nabla b|$)'),
                            Line2D([], [], color=COL_VEL, lw=2.0,
                                   label=rf'$\Delta\mathbf{{u}}$ to the neighbour (longest {100 * dumax:.1f} cm s$^{{-1}}$)'),
                            Line2D([], [], color='k', lw=0.7, alpha=0.6, label=r'$b_{mid}$ contours'),
                            Line2D([], [], color=COL_CLOSE, marker='s', mfc='none', ls='', ms=7,
                                   label='the parcel of (b)')],
                   loc='lower center', bbox_to_anchor=(0.62, 0.0), ncol=3,
                   frameon=False, fontsize=small, handlelength=1.5, columnspacing=1.0)
        fig.canvas.draw()
        pos = c.get_position()
        cb = fig.colorbar(m, cax=fig.add_axes([pos.x1 + 0.08 / W, pos.y0 + 0.06 * pos.height,
                                               0.09 / W, 0.88 * pos.height]), extend='both')
        cb.set_label(r'$2F$ [10$^{-19}$ s$^{-5}$]', fontsize=small, labelpad=2)
        cb.ax.tick_params(labelsize=small)
        out_png = pathlib.Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METHOD_DPI)
        plt.close(fig)
    nb = {name[k]: dict(model=f'{k[0]}{k[1]:+d}', term=float(dc['terms'][k][p] * SCALE),
                        term_centre=float(dc['centre_terms'][k][p] * SCALE),
                        g_ratio=float(np.hypot(dc['g4i'][nbr[k]], dc['g4j'][nbr[k]]) / gmag),
                        du_cm=float(100 * np.hypot(*dus[k])),
                        G_ratio=float(d['G'][nbr[k]] / d['G'][p])) for k in keys}
    return dict(png=str(out_png), size_in=list(SLIDE_SIZE), dpi=METHOD_DPI, parcel=list(p),
                neighbours=nb, sum=float(sum(disc)), sum_centre=float(sum(cent)),
                two_F=float(d['two_F'][p] * SCALE), two_F_chain=float(d['two_F_chain'][p] * SCALE),
                ratio=float(sum(disc) / sum(cent)))


# ---------------------------------------------------------------------------
# the numbers of the divergence + strain derivation (2F prompt 4, Q&A Round 3; report §2.5)
# ---------------------------------------------------------------------------
def decomposition(d, p):
    """``(1/G) DG/Dt = -delta + |sigma| cos 2 theta``: the two growth rates [s^-1] at the
    close-up parcel and over the band.  Band rates are ``G``-weighted, so that
    ``<2F> = <G> x rate`` (the band means of ``2F_split.png``).  Continuum form,
    Jacobian-consistent strain set (exact split)."""
    band, skel = d['band'], d['skel']
    G, dl, sg, th = d['G'], d['delta'], d['sigma'], d['theta']
    c2 = np.cos(2 * th)
    r_conv, r_str = -dl, sg * c2
    f = float(2 * OMEGA * np.sin(np.deg2rad(d['Y'][p])))

    def gw(a, m=band):
        return float((G * a)[m].sum() / G[m].sum())

    rate_p = float(r_conv[p] + r_str[p])
    rate_b = gw(r_conv) + gw(r_str)
    out = dict(f=f, lat=float(d['Y'][p]),
               parcel=dict(j=p[0], i=p[1], G=float(G[p]), delta=float(dl[p]), sigma=float(sg[p]),
                           theta_deg=float(np.degrees(th[p])), cos2theta=float(c2[p]),
                           rate_conv=float(r_conv[p]), rate_strain=float(r_str[p]), rate=rate_p,
                           rate_over_f=rate_p / f, efold_h=1.0 / rate_p / 3600.0,
                           conv=float(-dl[p] * G[p] * SCALE), strain=float(r_str[p] * G[p] * SCALE)),
               band=dict(n=int(band.sum()), G_mean=float(G[band].mean()),
                         rate_conv=gw(r_conv), rate_strain=gw(r_str), rate=rate_b,
                         rate_over_f=rate_b / f, efold_h=1.0 / rate_b / 3600.0,
                         abs_delta=gw(np.abs(dl)), sigma=gw(sg),
                         frac_convergent=float(np.mean(dl[band] < 0)),
                         frac_theta_lt_45=float(np.mean(th[band] < np.pi / 4)),
                         theta_median_deg=float(np.degrees(np.median(th[band]))),
                         cos2theta_mean=float(c2[band].mean()),
                         conv=float((-dl * G)[band].mean() * SCALE),
                         strain=float((sg * G * c2)[band].mean() * SCALE),
                         two_F_chain=float(d['two_F_chain'][band].mean() * SCALE),
                         two_F=float(d['two_F'][band].mean() * SCALE)))
    b = out['band']
    b['strain_share'] = b['strain'] / b['two_F_chain']
    b['net_over_abs_delta'] = b['rate_conv'] / b['abs_delta']
    b['strain_realised'] = b['rate_strain'] / b['sigma']
    b['discrete_over_chain'] = b['two_F'] / b['two_F_chain']
    # the exact split of the stored discrete 2F (2F prompt 7; report §2.5, Appendix D)
    rms = lambda a: float(np.sqrt(np.mean(a[band] ** 2)) * SCALE)    # noqa: E731
    out['discrete'] = dict(div=float(d['two_F_div'][band].mean() * SCALE),
                           strain=float(d['two_F_strain'][band].mean() * SCALE),
                           rot=float(d['two_F_rot'][band].mean() * SCALE),
                           rot_rms=rms(d['two_F_rot']), total_rms=rms(d['two_F']),
                           corr_div=float(np.corrcoef(d['two_F_div'][band], (-dl * G)[band])[0, 1]),
                           corr_strain=float(np.corrcoef(d['two_F_strain'][band], (sg * G * c2)[band])[0, 1]))
    dd = out['discrete']
    dd['strain_share'] = dd['strain'] / b['two_F']
    dd['div_over_cont'] = dd['div'] / b['conv']
    dd['strain_over_cont'] = dd['strain'] / b['strain']
    for nm, m in (('ridge', skel), ('flank', band & ~skel)):
        out[nm] = dict(n=int(m.sum()), conv=float((-dl * G)[m].mean() * SCALE),
                       strain=float((sg * G * c2)[m].mean() * SCALE),
                       d_div=float(d['two_F_div'][m].mean() * SCALE),
                       d_strain=float(d['two_F_strain'][m].mean() * SCALE))
    split_err = np.abs(-dl * G + sg * G * c2 - d['two_F_chain'])[band].max()
    out['split_max_err_over_rms'] = float(split_err / np.sqrt(np.mean(d['two_F_chain'][band] ** 2)))
    assert out['split_max_err_over_rms'] < TOL_SPLIT, out['split_max_err_over_rms']
    return out


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
    res['split'] = fig_2F_split(d, win, lims, out / '2F_split.png')
    res['decomposition'] = decomposition(d, par)
    res['discrete'] = fig_discrete(d, par, lims, out / '2F_discrete.png')
    path = SUMMARY.with_name(SUMMARY.name.format(pair=a.pair))
    path.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    main()
