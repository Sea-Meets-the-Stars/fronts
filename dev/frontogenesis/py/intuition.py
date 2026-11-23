"""Figures and numbers for ``report/frontogenesis_intuition.md`` (frontogenesis_intuition.md).

Section 2F (prompt 1): the primary equation

    (1/G) DG/Dt = 2F / G = -delta + |sigma| cos 2 theta,       G = |grad b|^2,

its derivation from ``D(grad b)/Dt = -(grad u)^T grad b`` and the split of ``grad u`` into
an isotropic part (divergence), a pure strain and a rotation.

``--what schematics`` (no data store is opened; only ``data/slides_2F_36.json`` is read, for
the two angles marked on the curve):

* ``report/figs/intuition_2F_terms.png`` -- one relative flow written as convergence +
  strain + rotation; each panel carries a ring of water and a set of isopycnals forward
  for a short time, so the change of the isopycnal spacing (that is, of ``G``) is seen.
* ``report/figs/intuition_2F_angles.png`` -- (a) the angles of the derivation: ``alpha``
  (direction of ``grad b``), ``psi`` (extensional axis), ``psi + 90 deg`` (compressional
  axis) and ``theta`` between ``grad b`` and the compressional axis; (b)-(d) one strain
  field acting on fronts at ``theta`` = 0, 45, 90 deg; (e) the factor ``cos 2 theta``.

Every rate written on a panel is computed with the pipeline's own
``operators.strain_from_jacobian`` and ``operators.strain_alignment`` and **asserted**
against the direct form ``-2 n^T (grad u) n`` and against the drawn deformation.

``--what numbers`` (about 1 min; opens the raw store through ``slides_2F.compute``):
the terms as growth rates on label 83 at pair 36 -> ``data/intuition_2F.json``, asserted
against ``data/slides_2F_36.json``.

Run in the ``frontogenesis`` env:

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python intuition.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Polygon

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import operators as op                                               # noqa: E402
import slides_2F as s2f                                              # noqa: E402
from slides_2F import COL_GRAD, COL_COMP, COL_VEL                    # noqa: E402
from slides_dGdt import SCALE                                        # noqa: E402

DATA_DIR = HERE.parent / 'data'
FIG_DIR = HERE.parent / 'report' / 'figs'
SUMMARY = DATA_DIR / 'intuition_2F.json'
SLIDES_JSON = DATA_DIR / 'slides_2F_{pair}.json'

PAIR = 36
OMEGA = 7.2921e-5            # rad s^-1
COL_EXT = '#1C7293'          # extensional axis
COL_BEFORE = '0.62'          # isopycnals and ring before the flow acts
COL_ISO = '#065A82'          # isopycnals after
COL_RING = '#A8501B'         # ring of water after
TAU = 0.5                    # time the drawn flow acts, in units of 1/|sigma|
HALF = 1.5                   # half-width of a panel
RING = 0.85                  # radius of the ring of water
ISO_STEP = 0.4               # isopycnal spacing before
TOL = 1e-12                  # identities between the forms of the rate
TOL_DRAWN = 1e-9             # drawn deformation against exp(rate * tau), pure flows
TOL_JSON = 1e-9              # band means against slides_2F_36.json
DPI = 200

# the example of Figure 1 (rates in units of |sigma|)
EX_DELTA, EX_SIGMA, EX_ZETA = -0.6, 1.0, 0.8
EX_ALPHA, EX_THETA = 0.0, 25.0                 # grad b to the east; compressional axis 25 deg from it
# the geometry of Figure 2 (a) [deg]
ANG_PSI, ANG_ALPHA = 20.0, 80.0                # so the compressional axis is at 110 and theta = 30


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------
def velocity_gradient(delta=0.0, sigma=0.0, psi_deg=0.0, zeta=0.0):
    """``grad u = [[u_x, u_y], [v_x, v_y]]`` from its three parts: divergence ``delta``,
    strain of magnitude ``sigma`` with extensional axis ``psi``, vorticity ``zeta``."""
    c, s = np.cos(2 * np.radians(psi_deg)), np.sin(2 * np.radians(psi_deg))
    return (0.5 * delta * np.eye(2) + 0.5 * sigma * np.array([[c, s], [s, -c]])
            + 0.5 * zeta * np.array([[0.0, -1.0], [1.0, 0.0]]))


def rates(A, alpha_deg):
    """The growth rate of ``G`` for a gradient in direction ``alpha``, by the pipeline's
    functions: ``-delta + |sigma| cos 2 theta``.  Asserted against ``-2 n^T A n``."""
    a = np.radians(alpha_deg)
    n = np.array([np.cos(a), np.sin(a)])
    delta, sn, ss, mag = (np.float64(v) for v in
                          op.strain_from_jacobian(A[0, 0], A[0, 1], A[1, 0], A[1, 1]))
    theta = np.float64(op.strain_alignment(np.float64(n[0]), np.float64(n[1]), sn, ss))
    strain = float(mag * np.cos(2 * theta)) if mag > 0 else 0.0
    out = dict(delta=float(delta), sigma=float(mag), zeta=float(A[1, 0] - A[0, 1]),
               theta_deg=float(np.degrees(theta)) if mag > 0 else float('nan'),
               conv=float(-delta), strain=strain, total=float(-delta) + strain)
    direct = float(-2.0 * n @ A @ n)
    if abs(out['total'] - direct) > TOL:
        raise AssertionError(f'-delta + |sigma| cos 2 theta = {out["total"]} != -2 n.A.n = {direct}')
    return out


def carried(A, alpha_deg, tau=TAU):
    """The map of the water after ``tau`` and what it does to the gradient: ``M = exp(tau A)``,
    ``grad b -> M^-T grad b``.  Returns ``M``, the new unit-gradient vector and ``G'/G``."""
    a = np.radians(alpha_deg)
    M = expm(tau * A)
    g = np.linalg.inv(M).T @ np.array([np.cos(a), np.sin(a)])
    return M, g, float(g @ g)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def _arrow(ax, p0, p1, color, lw=2.0, ms=14, z=6, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                                 linewidth=lw, zorder=z, shrinkA=0, shrinkB=0, linestyle=ls))


def _axis_line(ax, ang_deg, half, color, inward, lw=2.0, ms=13, z=5):
    """A strain axis through the origin: arrowheads pointing in (compression) or out."""
    e = half * np.array([np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))])
    for sgn in (1, -1):
        tip, tail = (0.42 * sgn * e, sgn * e) if inward else (sgn * e, 0.42 * sgn * e)
        _arrow(ax, tail, tip, color, lw=lw, ms=ms, z=z)
    ax.plot([-e[0], e[0]], [-e[1], e[1]], color=color, lw=lw * 0.5, zorder=z - 1)


def _isopycnals(ax, alpha_deg, M=None, color=COL_BEFORE, lw=1.0, ls='--', z=2):
    a = np.radians(alpha_deg)
    g, t = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    s = np.linspace(-6, 6, 2)
    for c in np.arange(-8, 9) * ISO_STEP:
        pts = c * g[:, None] + t[:, None] * s[None, :]
        if M is not None:
            pts = M @ pts
        ax.plot(pts[0], pts[1], color=color, lw=lw, ls=ls, zorder=z)


def _ring(ax, M=None, color=COL_BEFORE, lw=1.2, ls='--', fill=False, z=3):
    ph = np.linspace(0, 2 * np.pi, 181)
    pts = RING * np.vstack([np.cos(ph), np.sin(ph)])
    if M is not None:
        pts = M @ pts
    if fill:
        ax.add_patch(Polygon(pts.T, closed=True, facecolor=color, alpha=0.13, edgecolor='none', zorder=z - 1))
    ax.plot(pts[0], pts[1], color=color, lw=lw, ls=ls, zorder=z)


def _flow(ax, A, n=7, scale=None, color=COL_VEL, alpha=0.75):
    x = np.linspace(-HALF * 0.86, HALF * 0.86, n)
    X, Y = np.meshgrid(x, x)
    U, V = A[0, 0] * X + A[0, 1] * Y, A[1, 0] * X + A[1, 1] * Y
    ax.quiver(X, Y, U, V, color=color, alpha=alpha, angles='xy', scale_units='xy',
              scale=scale, width=0.008, headwidth=4, headlength=4.5, zorder=4)


def _frame(ax):
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('0.4')


def _panel(ax, A, alpha_deg, tau=TAU, flow_scale=3.2, grad_len=0.62):
    """Isopycnals, a ring of water and ``grad b``, before (grey, dashed) and after the flow
    ``A`` has acted for ``tau`` (colour), with the flow relative to the centre as quivers."""
    M, g, ratio = carried(A, alpha_deg, tau)
    _frame(ax)
    _isopycnals(ax, alpha_deg)
    _isopycnals(ax, alpha_deg, M, color=COL_ISO, lw=1.5, ls='-', z=3)
    _ring(ax)
    _ring(ax, M, color=COL_RING, lw=2.0, ls='-', fill=True, z=5)
    _flow(ax, A, scale=flow_scale)
    _arrow(ax, (0, 0), tuple(grad_len * g), COL_GRAD, lw=2.4, ms=16, z=7)
    return ratio


def _save(fig, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor='w')
    plt.close(fig)
    print(f'{path.name}  {path.stat().st_size / 1e3:.0f} kB')
    return str(path)


def _legend_handles():
    return [Line2D([], [], color=COL_BEFORE, ls='--', lw=1.2, label='isopycnals and ring of water, before'),
            Line2D([], [], color=COL_ISO, lw=1.6, label='isopycnals, after'),
            Line2D([], [], color=COL_RING, lw=2.0, label='the same water, after'),
            Line2D([], [], color=COL_GRAD, lw=2.4, marker='>', ms=7, label=r'$\nabla b$, after (length $\propto |\nabla b|$)'),
            Line2D([], [], color=COL_VEL, lw=1.6, marker='>', ms=6, label='velocity relative to the centre')]


def fig_terms(out_png):
    """Figure 1: one relative flow = convergence + strain + rotation, and what each does."""
    psi = EX_ALPHA + EX_THETA - 90.0            # extensional axis; the compressional one is theta from grad b
    parts = (('total', dict(delta=EX_DELTA, sigma=EX_SIGMA, psi_deg=psi, zeta=EX_ZETA)),
             ('conv', dict(delta=EX_DELTA)),
             ('strain', dict(sigma=EX_SIGMA, psi_deg=psi)),
             ('rot', dict(zeta=EX_ZETA)))
    res = {k: rates(velocity_gradient(**kw), EX_ALPHA) for k, kw in parts}
    if abs(res['total']['total'] - res['conv']['total'] - res['strain']['total'] - res['rot']['total']) > TOL:
        raise AssertionError('the three parts do not add to the total rate')
    if abs(res['rot']['total']) > TOL:
        raise AssertionError('rotation changes G')
    if abs(res['strain']['theta_deg'] - EX_THETA) > 1e-9:
        raise AssertionError(f'theta of the example is {res["strain"]["theta_deg"]}, not {EX_THETA}')

    titles = dict(total=r'(a) flow around a parcel: $\nabla\mathbf{u}$',
                  conv=r'(b) convergence: $\frac{1}{2}\delta\,\mathsf{I}$',
                  strain=r'(c) strain: $\frac{1}{2}|\sigma|\,\mathsf{S}(\psi)$',
                  rot=r'(d) rotation: $\frac{1}{2}\zeta\,\mathsf{J}$')
    lines = dict(
        total=r'$\frac{1}{G}\frac{DG}{Dt} = -\delta + |\sigma|\cos 2\theta = %+.2f\,|\sigma|$',
        conv=r'$-\delta = %+.2f\,|\sigma|$' + '\nevery spacing shrinks',
        strain=r'$|\sigma|\cos 2\theta = %+.2f\,|\sigma|$' + '\n' + rf'squeezed along one axis ($\theta = {EX_THETA:.0f}^\circ$)',
        rot=r'$%.0f$' + '\n' + r'turns $\nabla b$: changes $\theta$, not $G$')
    fig = plt.figure(figsize=(12.0, 4.25))
    w, h, y0 = 0.205, 0.205 * 12.0 / 4.25, 0.27
    x0s = [0.022 + k * (w + 0.0453) for k in range(4)]
    for k, (key, kw) in enumerate(parts):
        A = velocity_gradient(**kw)
        ax = fig.add_axes([x0s[k], y0, w, h])
        ratio = _panel(ax, A, EX_ALPHA)
        res[key]['G_ratio_drawn'] = ratio
        if key in ('conv', 'rot'):                 # the orientation is fixed, so the growth is exponential
            if abs(np.log(ratio) - res[key]['total'] * TAU) > TOL_DRAWN:
                raise AssertionError(f'{key}: drawn G ratio {ratio} != exp(rate tau)')
        if key in ('total', 'strain'):
            _axis_line(ax, psi + 90.0, 1.32, COL_COMP, inward=True)
            _axis_line(ax, psi, 1.32, COL_EXT, inward=False)
        ax.set_title(titles[key], fontsize=13, pad=7)
        val = res[key]['total']
        fig.text(x0s[k] + w / 2, y0 - 0.035, lines[key] % (0.0 if key == 'rot' else val),
                 ha='center', va='top', fontsize=12, linespacing=1.5)
        if k:
            fig.text(x0s[k] - 0.0227, y0 + h / 2, '=' if k == 1 else '+', ha='center', va='center', fontsize=26)
    handles = _legend_handles() + [
        Line2D([], [], color=COL_COMP, lw=2.0, marker='>', ms=6, label='compressional axis'),
        Line2D([], [], color=COL_EXT, lw=2.0, marker='>', ms=6, label='extensional axis')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.6, handlelength=2.4)
    res['example'] = dict(delta=EX_DELTA, sigma=EX_SIGMA, zeta=EX_ZETA, alpha_deg=EX_ALPHA,
                          theta_deg=EX_THETA, psi_deg=psi, tau=TAU)
    res['png'] = _save(fig, out_png)
    return res


def _arc(ax, a0, a1, r, color, text, lw=1.6, dr=0.12, fs=15, dang=0.0):
    ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=a0, theta2=a1, color=color, lw=lw, zorder=8))
    am = np.radians(0.5 * (a0 + a1) + dang)
    ax.text((r + dr) * np.cos(am), (r + dr) * np.sin(am), text, color=color, fontsize=fs,
            ha='center', va='center', zorder=9)


def fig_angles(out_png, marks):
    """Figure 2: the angles of the derivation, one strain on three fronts, and ``cos 2 theta``."""
    comp = ANG_PSI + 90.0
    A = velocity_gradient(sigma=1.0, psi_deg=ANG_PSI)
    r = rates(A, ANG_ALPHA)
    theta = comp - ANG_ALPHA
    if abs(r['theta_deg'] - theta) > 1e-9:
        raise AssertionError(f'panel (a): theta {r["theta_deg"]} != {theta}')
    if abs(np.cos(np.radians(2 * theta)) + np.cos(np.radians(2 * (ANG_ALPHA - ANG_PSI)))) > TOL:
        raise AssertionError('cos 2 theta != -cos 2(alpha - psi)')
    res = dict(a=dict(psi_deg=ANG_PSI, alpha_deg=ANG_ALPHA, compressional_deg=comp, **r))

    fig = plt.figure(figsize=(12.0, 6.3))
    # (a) the angles ------------------------------------------------------------------
    ax = fig.add_axes([0.012, 0.155, 0.79 * 6.3 / 12.0, 0.79])
    _frame(ax)
    _flow(ax, A, n=9, scale=3.4, alpha=0.30)
    a = np.radians(ANG_ALPHA)
    g, t = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    for c in np.arange(-8, 9) * ISO_STEP:
        p = c * g[:, None] + t[:, None] * np.array([-6.0, 6.0])[None, :]
        ax.plot(p[0], p[1], color=COL_ISO, lw=1.3, alpha=0.55, zorder=2)
    ax.plot([0, 1.42], [0, 0], color='0.25', lw=1.0, zorder=3)
    ax.text(1.44, -0.07, r'$x$ (east)', ha='right', va='top', fontsize=11, color='0.25')
    _axis_line(ax, comp, 1.36, COL_COMP, inward=True, lw=2.4, ms=16)
    _axis_line(ax, ANG_PSI, 1.36, COL_EXT, inward=False, lw=2.4, ms=16)
    _arrow(ax, (0, 0), tuple(1.12 * g), COL_GRAD, lw=3.0, ms=20, z=7)
    ax.text(*(1.25 * g + np.array([0.13, 0.0])), r'$\nabla b$', color=COL_GRAD, fontsize=16, ha='center', va='center')
    _arc(ax, 0, ANG_PSI, 0.95, COL_EXT, r'$\psi$', dr=0.11)
    _arc(ax, 0, ANG_ALPHA, 0.36, COL_GRAD, r'$\alpha$', dr=0.11, dang=8)
    _arc(ax, ANG_ALPHA, comp, 0.70, 'k', r'$\theta$', dr=0.12, lw=2.2)
    ec, ee = (np.array([np.cos(np.radians(v)), np.sin(np.radians(v))]) for v in (comp, ANG_PSI))
    q = 0.13                                                     # the right angle between the two axes
    ax.plot(*np.array([q * ee, q * (ee + ec), q * ec]).T, color='0.3', lw=1.0, zorder=6)
    ax.text(*(-1.22 * ec + np.array([0.33, 0.02])), 'compressional\naxis ' + r'($\psi + 90^\circ$)', fontsize=11,
            ha='left', va='center', color=COL_COMP, zorder=9,
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, pad=1.5))
    ax.text(*(-1.02 * ee + np.array([0.0, -0.17])), 'extensional axis ' + r'($\psi$)', fontsize=11,
            ha='center', va='top', color=COL_EXT, zorder=9, rotation=ANG_PSI,
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, pad=1.5))
    ax.text(*(0.93 * t + 0.62 * g), 'isopycnals\n' + r'($b$ = const)', fontsize=11, color=COL_ISO, ha='center',
            va='center', rotation=ANG_ALPHA - 90.0, zorder=9,
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, pad=1.5))
    ax.set_title(r'(a) the angles:  $\cos 2\theta = -\cos 2(\alpha - \psi)$', fontsize=13, pad=7)

    # (b)-(d) one strain, three fronts ------------------------------------------------
    S = velocity_gradient(sigma=1.0, psi_deg=90.0)              # compression along x, extension along y
    xr0, wr, gap = 0.462, 0.150, 0.030
    hr = wr * 12.0 / 6.3
    cases = ((0.0, r'(b) $\theta = 0^\circ$: sharpens', r'$+|\sigma|$'),
             (45.0, r'(c) $\theta = 45^\circ$: no change', r'$0$'),
             (90.0, r'(d) $\theta = 90^\circ$: weakens', r'$-|\sigma|$'))
    res['cases'] = []
    for k, (th, title, txt) in enumerate(cases):
        axk = fig.add_axes([xr0 + k * (wr + gap), 0.945 - 0.052 - hr, wr, hr])
        rk = rates(S, th)                                       # grad b at angle theta from the x (compressional) axis
        if abs(rk['theta_deg'] - th) > 1e-9 or abs(rk['total'] - np.cos(np.radians(2 * th))) > TOL:
            raise AssertionError(f'case theta = {th}: {rk}')
        ratio = _panel(axk, S, th, flow_scale=4.2, grad_len=0.55)
        if th in (0.0, 90.0) and abs(np.log(ratio) - rk['total'] * TAU) > TOL_DRAWN:
            raise AssertionError(f'case theta = {th}: drawn G ratio {ratio}')
        axk.set_title(title, fontsize=13, pad=7)
        axk.text(0.5, -0.045, r'$\frac{1}{G}\frac{DG}{Dt} = $' + txt, transform=axk.transAxes, ha='center',
                 va='top', fontsize=12)
        res['cases'].append(dict(theta_deg=th, rate_over_sigma=rk['total'], G_ratio_drawn=ratio))

    # (e) cos 2 theta -----------------------------------------------------------------
    axe = fig.add_axes([xr0 + 0.04, 0.115, 3 * wr + 2 * gap - 0.05, 0.30])
    th = np.linspace(0, 90, 181)
    c2 = np.cos(np.radians(2 * th))
    axe.fill_between(th, 0, c2, where=c2 >= 0, color=COL_GRAD, alpha=0.16, lw=0)
    axe.fill_between(th, 0, c2, where=c2 <= 0, color=COL_ISO, alpha=0.16, lw=0)
    axe.plot(th, c2, color='k', lw=2.0)
    axe.axhline(0, color='0.4', lw=0.8)
    axe.text(13, 0.22, 'strain sharpens\n(frontogenesis)', color=COL_GRAD, fontsize=11, ha='center', va='center')
    axe.text(77, -0.24, 'strain weakens\n(frontolysis)', color=COL_ISO, fontsize=11, ha='center', va='center')
    for key, (deg, label, mk, dy) in marks.items():
        axe.plot(deg, np.cos(np.radians(2 * deg)), mk, color='k', ms=9, mfc='w', mew=1.8, zorder=5)
        axe.annotate(label, (deg, np.cos(np.radians(2 * deg))), xytext=(deg + 9, np.cos(np.radians(2 * deg)) + dy),
                     fontsize=10, va='center', arrowprops=dict(arrowstyle='-', color='0.4', lw=0.8))
    axe.set_xlim(0, 90)
    axe.set_ylim(-1.08, 1.08)
    axe.set_xticks([0, 15, 30, 45, 60, 75, 90])
    axe.set_xlabel(r'$\theta$: angle between $\nabla b$ and the compressional axis [deg]', fontsize=11)
    axe.set_ylabel(r'$\cos 2\theta$', fontsize=12)
    axe.set_title(r'(e) the strain term is $|\sigma|\,G\cos 2\theta$', fontsize=13, pad=7)
    axe.tick_params(labelsize=10)

    fig.legend(handles=_legend_handles(), loc='lower left', ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.008, 0.0), columnspacing=1.4, handlelength=2.2)
    res['marks'] = {k: v[0] for k, v in marks.items()}
    res['png'] = _save(fig, out_png)
    return res


def schematics(pair=PAIR, out_dir=FIG_DIR):
    ref = json.loads(SLIDES_JSON.with_name(SLIDES_JSON.name.format(pair=pair)).read_text())
    marks = dict(
        parcel=(ref['method']['parcel']['theta_deg'],
                rf'close-up parcel, {ref["method"]["parcel"]["theta_deg"]:.0f}$^\circ$', 'o', 0.30),
        band=(ref['band']['theta_median_deg'],
              rf'band median, {ref["band"]["theta_median_deg"]:.0f}$^\circ$ (label {ref["label"]}, pair {pair})', 's', 0.0))
    out_dir = pathlib.Path(out_dir)
    return dict(terms=fig_terms(out_dir / 'intuition_2F_terms.png'),
                angles=fig_angles(out_dir / 'intuition_2F_angles.png', marks))


# ---------------------------------------------------------------------------
# the terms as growth rates on the tracked front
# ---------------------------------------------------------------------------
def _wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def _rates_on(d, m):
    """Means over the cells ``m``: contributions to ``<2F>`` and ``G``-weighted rates."""
    G, delta, sigma, theta = d['G'][m], d['delta'][m], d['sigma'][m], d['theta'][m]
    c2 = np.cos(2 * theta)
    return dict(
        n=int(m.sum()), G_mean=float(G.mean()),
        conv_mean=float(d['conv'][m].mean() * SCALE), strain_mean=float(d['strain'][m].mean() * SCALE),
        two_F_chain_mean=float(d['two_F_chain'][m].mean() * SCALE), two_F_mean=float(d['two_F'][m].mean() * SCALE),
        rate_conv=_wmean(-delta, G), rate_strain=_wmean(sigma * c2, G), rate_total=_wmean(-delta + sigma * c2, G),
        abs_delta_Gw=_wmean(np.abs(delta), G), sigma_Gw=_wmean(sigma, G), cos2theta_Gw=_wmean(c2, G),
        cos2theta_mean=float(c2.mean()), abs_delta_median=float(np.median(np.abs(delta))),
        sigma_median=float(np.median(sigma)), theta_median_deg=float(np.degrees(np.median(theta))),
        frac_converging=float(np.mean(delta < 0)), frac_theta_lt_45=float(np.mean(theta < 0.25 * np.pi)))


def numbers(pair=PAIR):
    """The growth-rate form of the split on the band of the tracked label at ``pair``."""
    ref = json.loads(SLIDES_JSON.with_name(SLIDES_JSON.name.format(pair=pair)).read_text())
    d = s2f.compute(pair)
    chk = s2f.check(d)
    band, skel = d['band'], d['skel']
    f = float(2 * OMEGA * np.sin(np.radians(np.mean(d['Y'][band]))))
    res = dict(pair=pair, label=d['label'], time_mid=d['time_mid'], check=chk, f=f,
               lat_mean=float(np.mean(d['Y'][band])), units=dict(rates='s-1', means='1e-19 s-5'),
               band=_rates_on(d, band), ridge=_rates_on(d, skel), flank=_rates_on(d, band & ~skel))
    b = res['band']
    for key, mine in (('conv', b['conv_mean']), ('strain', b['strain_mean']), ('two_F_chain', b['two_F_chain_mean'])):
        if abs(mine - ref['split'][key]['band_mean']) > TOL_JSON * abs(mine):
            raise AssertionError(f'{key}: band mean {mine} != slides_2F json {ref["split"][key]["band_mean"]}')
    if abs(b['rate_total'] * b['G_mean'] * SCALE - b['two_F_chain_mean']) > 1e-9 * abs(b['two_F_chain_mean']):
        raise AssertionError('<G> x (G-weighted rate) != <2F>')
    b['rate_total_over_f'] = b['rate_total'] / f
    b['efold_hours'] = 1.0 / b['rate_total'] / 3600.0
    b['strain_fraction'] = b['strain_mean'] / b['two_F_chain_mean']
    p = tuple(ref['parcel_ji'])
    c2 = float(np.cos(2 * d['theta'][p]))
    rate = float(-d['delta'][p] + d['sigma'][p] * c2)
    if abs(rate * d['G'][p] - d['two_F_chain'][p]) > 1e-9 * abs(d['two_F_chain'][p]):
        raise AssertionError('parcel: G x rate != 2F')
    res['parcel'] = dict(j=p[0], i=p[1], G=float(d['G'][p]), delta=float(d['delta'][p]), sigma=float(d['sigma'][p]),
                         theta_deg=float(np.degrees(d['theta'][p])), cos2theta=c2,
                         rate_conv=float(-d['delta'][p]), rate_strain=float(d['sigma'][p] * c2), rate_total=rate,
                         rate_total_over_f=rate / f, efold_hours=1.0 / rate / 3600.0,
                         conv=float(d['conv'][p] * SCALE), strain=float(d['strain'][p] * SCALE),
                         two_F_chain=float(d['two_F_chain'][p] * SCALE), two_F=float(d['two_F'][p] * SCALE))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--what', choices=('schematics', 'numbers', 'all'), default='all')
    p.add_argument('--pair', type=int, default=PAIR)
    a = p.parse_args(argv)
    res = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    if a.what in ('schematics', 'all'):
        res['schematics'] = schematics(a.pair)
    if a.what in ('numbers', 'all'):
        res['numbers'] = numbers(a.pair)
    SUMMARY.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    main()
