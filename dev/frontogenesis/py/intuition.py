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

* ``report/figs/intuition_2F_angle_theory.png`` (prompt 2, I-Q4 (a)) -- the equation for the
  angle, ``D phi/Dt = (|sigma|/2)(r - sin 2 phi)`` with ``phi`` the signed angle of ``grad b``
  from the compressional axis and ``r = (zeta - 2 D psi/Dt)/|sigma|``: its rate, its
  solutions and the steady ``cos 2 theta = sqrt(1 - r^2)``.  Asserted against the turning of
  a gradient carried by ``exp(tau grad u)``.

``--what numbers`` (about 1 min; opens the raw store through ``slides_2F.compute``):
the terms as growth rates on label 83 at pair 36, and the signed angle against
``zeta/|sigma|`` on the same band -> ``data/intuition_2F.json``, asserted against
``data/slides_2F_36.json``.

``--what discrete`` (prompt 3; reads two stored fields, no recomputation):
``report/figs/intuition_2F_discrete.png`` -- why the stored, discrete ``2F`` is smaller than
the continuum form on a narrow front: the gradient stencil spans two cells, and what changes
its value is the velocity difference across each half times the slope of ``b`` at its two
ends.  A 1-D tanh front in a uniform convergence; the commutator is asserted against a
finite semi-Lagrangian step.  Also the stored ``front_width`` on the label.

``--what data`` (no store; reads the summary): ``report/figs/intuition_2F_angle_data.png``,
``sin 2 phi`` against ``zeta/|sigma|`` on the band.  Also drawn after ``numbers``.

Run in the ``frontogenesis`` env:

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python intuition.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp
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
from slides_dGdt import SCALE, DERIVED, FRONTS                       # noqa: E402

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
TOL_PHI = 1e-6               # |signed phi| against operators.strain_alignment [rad]
R_MAX, R_BINS, BIN_MIN = 3.0, 12, 15   # bins of r = zeta / |sigma| for the band test
R_CASES = (0.0, 0.5, 0.9, 1.3)                 # rotation ratios drawn in the angle figure
R_COLOURS = ('k', '#1C7293', '#A8501B', '#6a3d9a')
T_END = 8.0                                    # the drawn time, in units of 1/|sigma|
ELL_DRAWN = 1.2                                # front width of the stencil drawing [cells]
ELL_RANGE = (0.9, 8.0)                         # widths of the ratio curve [cells]
ELL_MARKS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
A_DT = 1e-5                                    # convergence x time step of the finite-step check
TOL_COMMUTATOR = 1e-4                          # finite-step rate against the commutator, relative
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


# ---------------------------------------------------------------------------
# the angle equation
# ---------------------------------------------------------------------------
def dphi_dt(phi, r):
    """``D phi / Dt`` in units of ``|sigma|``: ``(r - sin 2 phi) / 2``, with ``phi`` the signed
    angle of ``grad b`` from the compressional axis (anticlockwise) and
    ``r = (zeta - 2 D psi/Dt) / |sigma|``."""
    return 0.5 * (r - np.sin(2 * phi))


def steady_angle(r):
    """The stable steady angle ``phi* = arcsin(r) / 2`` [rad]; NaN where ``|r| > 1``."""
    r = np.asarray(r, float)
    with np.errstate(invalid='ignore'):
        return 0.5 * np.arcsin(np.where(np.abs(r) <= 1, r, np.nan))


def integrate_angle(r, phi0_deg, t):
    """``phi(t)`` [rad, unwrapped] and ``ln G(t)/G(0)`` for strain alone plus rotation
    (``delta = 0``), time in units of ``1/|sigma|``."""
    sol = solve_ivp(lambda _, y: [dphi_dt(y[0], r), np.cos(2 * y[0])], (t[0], t[-1]),
                    [np.radians(phi0_deg), 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    return sol.y[0], sol.y[1]


def check_angle_equation():
    """The angle equation against the drawn kinematics (``carried``), the closed form in pure
    strain, and the steady angle."""
    out = {}
    tau, psi, alpha = 1e-6, 20.0, 80.0
    worst = 0.0
    for delta, sigma, zeta in ((0.0, 1.0, 0.0), (-0.6, 1.0, 0.8), (0.3, 0.7, -1.9)):
        A = velocity_gradient(delta, sigma, psi, zeta)
        _, g, _ = carried(A, alpha, tau)
        a0 = np.radians(alpha)
        da = np.arctan2(g[1], g[0]) - a0
        phi = a0 - np.radians(psi + 90.0)
        worst = max(worst, abs(da / tau - sigma * dphi_dt(phi, zeta / sigma)))
    if worst > 1e-5:
        raise AssertionError(f'D alpha/Dt of the carried gradient differs from the angle equation by {worst:.2e}')
    out['max_err_rate'] = float(worst)
    t = np.linspace(0, 3, 31)
    phi, lnG = integrate_angle(0.0, 45.0, t)
    err = float(np.max(np.abs(np.tan(phi) - np.exp(-t))))
    errG = float(np.max(np.abs(lnG - np.log(np.cosh(t)))))
    if max(err, errG) > 1e-7:
        raise AssertionError(f'pure strain: tan phi != exp(-|sigma| t) ({err:.2e}) or G != cosh ({errG:.2e})')
    out.update(max_err_tan=err, max_err_lnG=errG, G_ratio_45deg_at_tau=float(np.cosh(TAU)))
    for r in (0.5, 0.9):
        phi, lnG = integrate_angle(r, -80.0, np.linspace(0, 60, 3))
        if abs(phi[-1] - steady_angle(r)) > 1e-6:
            raise AssertionError(f'r = {r}: phi does not reach arcsin(r)/2')
    return out


def fig_angle_theory(out_png):
    """Figure 3: the angle equation: its rate, its solutions and the strain term it leaves."""
    res = dict(check=check_angle_equation(), cases=[])
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.3))
    fig.subplots_adjust(left=0.062, right=0.985, bottom=0.215, top=0.90, wspace=0.30)
    a, b, c = axs
    ph = np.linspace(-90, 90, 361)
    t = np.linspace(0, T_END, 401)
    a.axhline(0, color='0.4', lw=0.8)
    a.axvspan(-45, 45, color=COL_GRAD, alpha=0.07, lw=0)
    for r, col in zip(R_CASES, R_COLOURS):
        a.plot(ph, 2 * dphi_dt(np.radians(ph), r), color=col, lw=2.0, label=f'$r = {r:g}$')
        entry = dict(r=r)
        if abs(r) <= 1:
            st = float(np.degrees(steady_angle(r)))
            un = 90.0 - st
            a.plot(st, 0, 'o', color=col, ms=9, zorder=5)
            a.plot([un] if un < 90 else [-90, 90], [0] * (1 if un < 90 else 2), 'o', color=col, ms=9, mfc='w',
                   mew=1.8, zorder=5, clip_on=False)
            entry.update(phi_steady_deg=st, cos2theta_steady=float(np.cos(np.radians(2 * st))))
        res['cases'].append(entry)
        for k, phi0 in enumerate((-80.0, 0.0, 50.0) if abs(r) <= 1 else (0.0,)):
            phi, lnG = integrate_angle(r, phi0, t)
            b.plot(t, np.degrees(phi), color=col, lw=1.8 if phi0 == 0 or abs(r) <= 1 else 1.2,
                   ls='-' if abs(r) <= 1 else '--')
            if phi0 == 0.0:
                entry['lnG_end'] = float(lnG[-1])
    a.set_xlim(-90, 90)
    a.set_xticks([-90, -45, 0, 45, 90])
    a.set_xlabel(r'$\phi$: signed angle of $\nabla b$ from the compressional axis [deg]', fontsize=10.5)
    a.set_ylabel(r'$\frac{2}{|\sigma|}\frac{D\phi}{Dt} = r - \sin 2\phi$', fontsize=12)
    a.set_title(r'(a) the rate of turning', fontsize=13, pad=7)
    a.legend(fontsize=10, frameon=False, loc='upper right', ncol=2, columnspacing=1.0, handlelength=1.4)
    b.axhspan(-45, 45, color=COL_GRAD, alpha=0.07, lw=0)
    b.set_xlim(0, T_END)
    b.text(T_END * 0.97, -41, r'strain sharpens ($|\phi| < 45^\circ$)', color=COL_GRAD, fontsize=10, ha='right', va='bottom')
    b.set_ylim(-90, 180)
    b.set_yticks([-90, -45, 0, 45, 90, 135, 180])
    b.set_xlabel(r'time [$1/|\sigma|$]', fontsize=11)
    b.set_ylabel(r'$\phi$ [deg]', fontsize=11)
    b.set_title(r'(b) the angle in time', fontsize=13, pad=7)
    rr = np.linspace(-2.5, 2.5, 1001)
    eq = np.where(np.abs(rr) <= 1, np.sqrt(np.clip(1 - rr ** 2, 0, None)), 0.0)
    c.plot(rr, eq, color='k', lw=2.2)
    c.fill_between(rr, 0, eq, color=COL_GRAD, alpha=0.14, lw=0)
    for r, col in zip(R_CASES, R_COLOURS):
        c.plot(r, np.sqrt(1 - r ** 2) if abs(r) <= 1 else 0.0, 'o', color=col, ms=9, zorder=5, clip_on=False)
    c.text(0, 0.36, 'strain wins:\nthe angle\nlocks', ha='center', va='center', fontsize=10, color=COL_GRAD)
    for x in (-1.78, 1.78):
        c.text(x, 0.36, 'rotation wins:\n' + r'$\nabla b$ keeps' + '\nturning', ha='center', va='center',
               fontsize=9.5, color=COL_VEL)
    c.set_xlim(-2.5, 2.5)
    c.set_ylim(0, 1.08)
    c.set_xlabel(r'$r = (\zeta - 2\,D\psi/Dt)\,/\,|\sigma|$', fontsize=11)
    c.set_ylabel(r'long-time mean of $\cos 2\theta$', fontsize=11)
    c.set_title(r'(c) what is left of the strain term', fontsize=13, pad=7)
    for ax in axs:
        ax.tick_params(labelsize=10)
    handles = [Line2D([], [], color='0.3', marker='o', ls='', ms=9, label='stable steady angle'),
               Line2D([], [], color='0.3', marker='o', ls='', ms=9, mfc='w', mew=1.8, label='unstable steady angle'),
               Line2D([], [], color=R_COLOURS[-1], ls='--', lw=1.4, label=r'$|r| > 1$: no steady angle')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
    res['png'] = _save(fig, out_png)
    return res


def fig_angle_data(out_png, ang, label, pair):
    """Figure 6: the angle against the vorticity on the band of the tracked label."""
    phi, r = np.asarray(ang['phi_deg']), np.asarray(ang['r'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 4.5))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.135, top=0.90, wspace=0.22)
    a.axhline(0, color='0.5', lw=0.8)
    a.axvline(0, color='0.5', lw=0.8)
    a.plot(r, np.sin(np.radians(2 * phi)), '.', color='0.55', ms=3.5, alpha=0.7, label='one cell of the band', zorder=2)
    rr = np.linspace(-1, 1, 201)
    a.plot(rr, rr, color=COL_GRAD, lw=2.2, label=r'steady angle, $\sin 2\phi = r$', zorder=4)
    a.plot([-R_MAX, -1], [-1, -1], color=COL_GRAD, lw=1.2, ls=':', zorder=4)
    a.plot([1, R_MAX], [1, 1], color=COL_GRAD, lw=1.2, ls=':', zorder=4)
    bins = [q for q in ang['bins'] if q['sin2phi_median'] is not None]
    a.plot([0.5 * (q['r_lo'] + q['r_hi']) for q in bins], [q['sin2phi_median'] for q in bins], 's-', color='k', ms=7,
           lw=1.4, mfc='w', mew=1.6, label=f'median in bins of $r$ (at least {BIN_MIN} cells)', zorder=5)
    a.set_xlim(-R_MAX, R_MAX)
    a.set_ylim(-1.08, 1.08)
    a.set_xlabel(r'$r = \zeta\,/\,|\sigma|$   (the turning of the strain axes is not measured)', fontsize=11)
    a.set_ylabel(r'$\sin 2\phi$', fontsize=12)
    a.set_title(f'(a) the angle against the vorticity: label {label}, pair {pair}', fontsize=13, pad=7)
    a.legend(fontsize=9.5, loc='lower right', framealpha=0.9)
    edges = np.arange(-90, 91, 10)
    for sel, col, name in ((r > 0, COL_GRAD, r'cyclonic, $\zeta > 0$'), (r < 0, COL_ISO, r'anticyclonic, $\zeta < 0$')):
        b.hist(phi[sel], bins=edges, histtype='stepfilled', color=col, alpha=0.22, lw=0)
        b.hist(phi[sel], bins=edges, histtype='step', color=col, lw=2.0,
               label=f'{name}: {int(sel.sum())} cells, median {np.median(phi[sel]):+.0f}' + r'$^\circ$')
        b.axvline(np.median(phi[sel]), color=col, lw=1.4, ls='--')
    b.axvspan(-45, 45, color='0.5', alpha=0.08, lw=0)
    b.set_xlim(-90, 90)
    b.set_xticks([-90, -45, 0, 45, 90])
    b.set_ylim(0, b.get_ylim()[1] * 1.22)
    b.set_xlabel(r'$\phi$: signed angle of $\nabla b$ from the compressional axis [deg]', fontsize=11)
    b.set_ylabel('cells', fontsize=11)
    b.set_title('(b) the angle, by the sign of the vorticity', fontsize=13, pad=7)
    b.legend(fontsize=9.5, loc='upper left', frameon=False)
    for ax in (a, b):
        ax.tick_params(labelsize=10)
    return dict(png=_save(fig, out_png),
                phi_median_deg_cyclonic=float(np.median(phi[r > 0])), n_cyclonic=int((r > 0).sum()),
                phi_median_deg_anticyclonic=float(np.median(phi[r < 0])), n_anticyclonic=int((r < 0).sum()))


# ---------------------------------------------------------------------------
# the continuum and the discrete 2F on a 1-D front
# ---------------------------------------------------------------------------
def front_1d(ell, h=1.0, a=1.0):
    """A front ``b = tanh(x/ell)`` on a grid of spacing ``h`` in the convergence ``u = -a x``,
    at the node on its centre.  Every rate is ``DG/Dt`` of the *grid* ``G = (L b)^2``, with
    ``L`` the centred two-cell difference, divided by ``2 a G`` (the chain form is then 1).

    ``chain``       ``-2 u_x (L b)^2``, the continuum formula on the grid's gradients;
    ``commutator``  ``-2 (L b) [L, u d/dx] b`` with the exact slope of ``b`` at ``x +- h``;
    ``discrete``    the same with the 4th-order neighbour gradient of
                    ``operators._frontogenesis_discrete``;
    ``semilag``     one semi-Lagrangian step of exact advection over ``dt = A_DT / a``.
    """
    b = lambda x: np.tanh(x / ell)                                   # noqa: E731
    db = lambda x: 1.0 / (ell * np.cosh(x / ell) ** 2)               # noqa: E731
    u = lambda x: -a * x                                             # noqa: E731
    L = lambda f, x: (f(x + h) - f(x - h)) / (2 * h)                 # noqa: E731
    g = L(b, 0.0)
    chain = 2 * a * g ** 2
    comm = -2 * g * ((u(h) - u(0.0)) * db(h) + (u(0.0) - u(-h)) * db(-h)) / (2 * h)
    g4 = lambda x: (4 * L(b, x) - 0.5 * (L(b, x + h) + L(b, x - h))) / 3   # noqa: E731
    disc = -2 * g * ((u(h) - u(0.0)) * g4(h) + (u(0.0) - u(-h)) * g4(-h)) / (2 * h)
    dt = A_DT / a
    b1 = lambda x: b(x * np.exp(a * dt))                             # noqa: E731  exact advection
    semilag = (L(b1, 0.0) ** 2 - g ** 2) / dt                        # the centre node departs from itself
    return dict(ell=float(ell), G_grid_over_true=float(g ** 2 / db(0.0) ** 2), chain=1.0,
                commutator=float(comm / chain), discrete=float(disc / chain), semilag=float(semilag / chain),
                expansion=float(1 - (2.0 / 3.0) * (h / ell) ** 2),
                box_mean=float(g / db(0.0)), end_value=float(db(h) / db(0.0)))


def fig_discrete(out_png):
    """Figure 7: the stencil on a narrow front, the three gradients, and the ratio of forms."""
    res = dict(marks=[front_1d(e) for e in ELL_MARKS], drawn=front_1d(ELL_DRAWN))
    for m in res['marks'] + [res['drawn']]:
        if abs(m['semilag'] - m['commutator']) > TOL_COMMUTATOR:
            raise AssertionError(f'ell = {m["ell"]}: semi-Lagrangian step {m["semilag"]} != commutator {m["commutator"]}')
    ell, d = ELL_DRAWN, res['drawn']
    b = lambda x: np.tanh(x / ell)                                   # noqa: E731
    db = lambda x: 1.0 / (ell * np.cosh(x / ell) ** 2)               # noqa: E731
    x = np.linspace(-3.2, 3.2, 641)
    fig, (a, c, e) = plt.subplots(1, 3, figsize=(12.0, 4.5))
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.235, top=0.90, wspace=0.30)
    col_box, col_end, col_true = COL_ISO, COL_RING, COL_GRAD

    a.plot(x, b(x), color='0.25', lw=2.0, zorder=3)
    nodes = np.arange(-3, 4)
    a.plot(nodes, b(nodes), 'o', color='0.25', ms=6, mfc='w', mew=1.4, zorder=5)
    a.plot([-1, 0, 1], b(np.array([-1.0, 0.0, 1.0])), 'o', color='k', ms=8, zorder=6)
    sec = (b(1.0) - b(-1.0)) / 2
    xs = np.array([-1.9, 1.9])
    a.plot(xs, sec * xs, color=col_box, lw=2.4, zorder=4)
    xt = np.array([-0.95, 0.95])
    a.plot(xt, db(0.0) * xt, color=col_true, lw=2.0, ls='--', zorder=4)
    for x0 in (-1.0, 1.0):
        xe = x0 + np.array([-0.75, 0.75])
        a.plot(xe, b(x0) + db(x0) * (xe - x0), color=col_end, lw=2.4, zorder=4)
        _arrow(a, (x0, -1.32), (x0 * 0.45, -1.32), COL_VEL, lw=2.0, ms=13)
    a.plot(0, -1.32, 'o', color=COL_VEL, ms=5)
    a.text(0, -1.47, 'velocity relative to the cell', color=COL_VEL, fontsize=9.5, ha='center', va='top')
    a.set_xlim(-3.2, 3.2)
    a.set_ylim(-1.78, 1.3)
    a.set_xticks(nodes)
    a.set_xticklabels([r'$x-3h$', '', r'$x-h$', r'$x$', r'$x+h$', '', r'$x+3h$'])
    a.set_yticks([])
    a.set_ylabel(r'buoyancy $b$', fontsize=11)
    a.set_title(rf'(a) a front of width $\ell = {ell:g}\,h$ on the grid', fontsize=13, pad=7)

    c.fill_between(x, 0, db(x) / db(0.0), where=np.abs(x) <= 1, color=col_box, alpha=0.13, lw=0)
    c.plot(x, db(x) / db(0.0), color='0.25', lw=2.0)
    c.plot([-1, 1], [d['box_mean']] * 2, color=col_box, lw=2.6)
    c.plot([-1, 1], [d['end_value']] * 2, color=col_end, lw=1.4, ls=':')
    c.plot([-1, 1], [d['end_value']] * 2, 'o', color=col_end, ms=9, zorder=5)
    c.plot(0, 1, 'o', color=col_true, ms=9, zorder=5)
    c.text(0.22, 1.0, f'at the cell: 1', color=col_true, fontsize=10, va='center')
    c.text(1.15, d['box_mean'], f'stencil mean\n($\\mathsf{{L}}b$): {d["box_mean"]:.2f}', color=col_box, fontsize=10, va='center')
    c.text(1.15, d['end_value'] - 0.02, f'ends: {d["end_value"]:.2f}', color=col_end, fontsize=10, va='top')
    c.set_xlim(-3.2, 3.2)
    c.set_ylim(0, 1.12)
    c.set_xticks(nodes)
    c.set_xticklabels([r'$x-3h$', '', r'$x-h$', r'$x$', r'$x+h$', '', r'$x+3h$'])
    c.set_ylabel(r'true gradient $\partial b/\partial x$, relative to its peak', fontsize=10.5)
    c.set_title('(b) three values of the gradient', fontsize=13, pad=7)

    ells = np.geomspace(*ELL_RANGE, 120)
    cur = [front_1d(v) for v in ells]
    e.axhline(1.0, color=col_box, lw=2.2, label='continuum (chain) form')
    e.plot(ells, [q['commutator'] for q in cur], color=col_end, lw=2.2, label='commutator, exact slope at the ends')
    e.plot(ELL_MARKS, [q['discrete'] for q in res['marks']], 's', color='#2C6E49', ms=7, zorder=5,
           label='discrete form of the code (4th-order slope)')
    e.plot(ELL_MARKS, [q['semilag'] for q in res['marks']], 'o', color='k', ms=11, mfc='none', mew=1.5, zorder=6,
           label='one semi-Lagrangian step, exact advection')
    e.plot(ells, [q['expansion'] for q in cur], color='0.4', lw=1.2, ls=':', label=r'$1 - \frac{2}{3}(h/\ell)^2$')
    e.set_xscale('log')
    e.set_xticks(ELL_MARKS)
    e.set_xticklabels([f'{v:g}' for v in ELL_MARKS])
    e.minorticks_off()
    e.set_xlim(*ELL_RANGE)
    e.set_ylim(0.0, 1.12)
    e.set_xlabel(r'front width $\ell\,/\,h$', fontsize=11)
    e.set_ylabel(r'$DG/Dt$ of the grid $G$, relative to the chain form', fontsize=10.5)
    e.set_title('(c) the rate at the centre of the front', fontsize=13, pad=7)
    e.legend(fontsize=8.8, loc='lower right', framealpha=0.92)
    for ax in (a, c, e):
        ax.tick_params(labelsize=10)
    handles = [Line2D([], [], color=col_box, lw=2.4, label=r'stencil gradient $\mathsf{L}b$: the slope between $x-h$ and $x+h$'),
               Line2D([], [], color=col_true, lw=2.0, ls='--', label='true slope at the cell'),
               Line2D([], [], color=col_end, lw=2.4, label=r'true slope at the ends of the stencil, $x \pm h$')]
    fig.legend(handles=handles, loc='lower left', ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.03, 0.0),
               columnspacing=1.5)
    res['png'] = _save(fig, out_png)
    return res


def width_numbers(pair=PAIR):
    """The stored width proxy ``front_width`` [cells] on the pixels of the tracked label."""
    summary = json.loads(SUMMARY.read_text())['numbers']
    der = xr.open_zarr(DATA_DIR / DERIVED.name).isel(time=pair)
    lab = xr.open_zarr(DATA_DIR / FRONTS.name)['labels'].isel(time=pair).values
    skel = lab == int(summary['label'])
    if int(skel.sum()) != summary['ridge']['n']:
        raise AssertionError(f'label {summary["label"]}: {int(skel.sum())} px, summary has {summary["ridge"]["n"]}')
    w = np.asarray(der['front_width'].values, 'float64')[skel]
    w = w[np.isfinite(w)]
    out = dict(n=int(w.size), median=float(np.median(w)), p25=float(np.percentile(w, 25)),
               p75=float(np.percentile(w, 75)), frac_le_2=float(np.mean(w <= 2.0)))
    for key in ('band', 'ridge', 'flank'):
        out[f'discrete_over_chain_{key}'] = summary[key]['two_F_mean'] / summary[key]['two_F_chain_mean']
    out['at_median_width'] = front_1d(out['median'])
    return out


def schematics(pair=PAIR, out_dir=FIG_DIR):
    ref = json.loads(SLIDES_JSON.with_name(SLIDES_JSON.name.format(pair=pair)).read_text())
    marks = dict(
        parcel=(ref['method']['parcel']['theta_deg'],
                rf'close-up parcel, {ref["method"]["parcel"]["theta_deg"]:.0f}$^\circ$', 'o', 0.30),
        band=(ref['band']['theta_median_deg'],
              rf'band median, {ref["band"]["theta_median_deg"]:.0f}$^\circ$ (label {ref["label"]}, pair {pair})', 's', 0.0))
    out_dir = pathlib.Path(out_dir)
    return dict(terms=fig_terms(out_dir / 'intuition_2F_terms.png'),
                angles=fig_angles(out_dir / 'intuition_2F_angles.png', marks),
                angle_theory=fig_angle_theory(out_dir / 'intuition_2F_angle_theory.png'))


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
    res['angle'] = angle_numbers(d, f)
    return res


def angle_numbers(d, f):
    """The signed angle ``phi`` of ``grad b`` from the compressional axis (anticlockwise) and
    the rotation ratio ``r = zeta / |sigma|`` on the band: the test of ``sin 2 phi = r``."""
    band = d['band']
    alpha = np.arctan2(d['by'], d['bx'])
    phi = (alpha - d['psi_c'] + 0.5 * np.pi) % np.pi - 0.5 * np.pi          # in [-90, 90) deg
    err = float(np.nanmax(np.abs(np.abs(phi[band]) - d['theta'][band])))
    if err > TOL_PHI:
        raise AssertionError(f'|phi| != theta_align on the band (max {err:.2e} rad)')
    phi, zeta, sigma, G = phi[band], d['zeta'][band], d['sigma'][band], d['G'][band]
    r = zeta / sigma
    s2 = np.sin(2 * phi)
    strain = np.abs(r) < 1
    pred = 0.5 * np.arcsin(np.clip(r, -1, 1))                              # the steady angle where |r| < 1
    edges = np.linspace(-R_MAX, R_MAX, R_BINS + 1)
    k = np.digitize(r, edges) - 1
    bins = [dict(r_lo=float(edges[b]), r_hi=float(edges[b + 1]), n=int((k == b).sum()),
                 sin2phi_median=float(np.median(s2[k == b])) if (k == b).sum() >= BIN_MIN else None,
                 cos2phi_mean=float(np.mean(np.cos(2 * phi[k == b]))) if (k == b).sum() >= BIN_MIN else None)
            for b in range(R_BINS)]
    rank = lambda x: np.argsort(np.argsort(x)).astype(float)               # noqa: E731
    return dict(
        n=int(band.sum()), max_abs_phi_minus_theta=err,
        zeta_over_f_median=float(np.median(zeta) / f), zeta_over_f_Gw=_wmean(zeta, G) / f,
        frac_cyclonic=float(np.mean(zeta > 0)),
        r_median=float(np.median(r)), abs_r_median=float(np.median(np.abs(r))),
        frac_abs_r_lt_1=float(np.mean(strain)), frac_abs_r_lt_1_Gw=_wmean(strain.astype(float), G),
        frac_phi_sign_of_zeta=float(np.mean(np.sign(phi) == np.sign(zeta))),
        phi_median_deg=float(np.degrees(np.median(phi))),
        corr_sin2phi_r_rank=float(np.corrcoef(rank(s2), rank(r))[0, 1]),
        corr_sin2phi_r_strain_cells=float(np.corrcoef(s2[strain], r[strain])[0, 1]),
        theta_pred_median_deg=float(np.degrees(np.median(np.abs(pred[strain])))),
        theta_obs_median_deg_strain_cells=float(np.degrees(np.median(np.abs(phi[strain])))),
        theta_obs_median_deg_rotation_cells=float(np.degrees(np.median(np.abs(phi[~strain])))),
        cos2theta_mean_strain_cells=float(np.mean(np.cos(2 * phi[strain]))),
        cos2theta_mean_rotation_cells=float(np.mean(np.cos(2 * phi[~strain]))),
        cos2theta_pred_mean_strain_cells=float(np.mean(np.sqrt(1 - r[strain] ** 2))),
        efold_angle_hours=float(1.0 / np.median(sigma) / 3600.0),
        bins=bins, phi_deg=[float(v) for v in np.degrees(phi)], r=[float(v) for v in r],
        G=[float(v) for v in G])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--what', choices=('schematics', 'numbers', 'data', 'discrete', 'all'), default='all')
    p.add_argument('--pair', type=int, default=PAIR)
    a = p.parse_args(argv)
    res = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    if a.what in ('schematics', 'all'):
        res['schematics'] = schematics(a.pair)
    if a.what in ('numbers', 'all'):
        res['numbers'] = numbers(a.pair)
    if a.what in ('discrete', 'all'):
        res['discrete'] = fig_discrete(FIG_DIR / 'intuition_2F_discrete.png')
        res['discrete']['front_width'] = width_numbers(a.pair)
    if a.what in ('data', 'numbers', 'all'):                  # drawn from the summary, no store
        n = res['numbers']
        res['angle_data'] = fig_angle_data(FIG_DIR / 'intuition_2F_angle_data.png', n['angle'], n['label'], n['pair'])
    SUMMARY.write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    main()
