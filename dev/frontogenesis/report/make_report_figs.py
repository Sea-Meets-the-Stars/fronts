"""The two schematics for the M3 interim report.

Everything else the report shows is re-used from ``../figs/`` (the task-7 page figures) and
``../deck/figs_m3/`` (the task-9 slide re-plots).  The project has never had a picture of its
own method, which is what these two supply (M3-Q20 (b), JXP 2026-10-12):

``figs/schematic_pipeline.png``   the two stores -> buoyancy -> one operator set -> the filter
                                  -> the predicted and measured arms -> the five-term budget.
``figs/schematic_budget.png``     a front in cross-section with each process colour-matched to
                                  its term in the budget equation.

Both are drawings.  **No data store is opened.**  The only numbers that appear are the term
amplitudes on the right of the budget schematic, read from ``../data/m3_closure_summary.json``
(task 6) -- the report's provenance rule, M3-Q23 (a).

Run:  <frontogenesis env python> make_report_figs.py
Deps: matplotlib, numpy.
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

HERE = pathlib.Path(__file__).parent
OUT = HERE / 'figs'
OUT.mkdir(exist_ok=True)

# the deck palette, so the report and the slides read as one thing
MIDNIGHT, DEEP, TEAL = '#21295C', '#065A82', '#1C7293'
AMBER, GREEN, RED, MUTED = '#A8501B', '#2C6E49', '#B22222', '#62737F'
PAPER, CARD, INK = '#FFFFFF', '#E7EFF3', '#1B2A33'

matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 170,
                            'savefig.facecolor': PAPER})

S = json.loads((HERE.parent / 'data' / 'm3_closure_summary.json').read_text())


def box(ax, x, y, w, h, text, fc=CARD, ec=DEEP, tc=INK, size=10.5, weight='normal', lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size,
            color=tc, zorder=3, weight=weight, linespacing=1.45)


def arrow(ax, p0, p1, color=DEEP, lw=2.0, style='-|>'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 color=color, linewidth=lw, zorder=1,
                                 shrinkA=2, shrinkB=2))


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    print(f'{name}  {p.stat().st_size / 1e3:.0f} kB')


# ====================================================== 1. the pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    box(ax, 0.02, 0.775, 0.265, 0.135,
        'Surface store  (hourly, 2-D)\n'
        r'$\Theta$, $S$, $u$, $v$, $\eta$' + '\n72 h from 2 July 2012',
        fc='#DCE9F0', ec=TEAL, size=10.5)
    box(ax, 0.02, 0.585, 0.265, 0.165,
        'Full-depth chunk store (hourly)\n'
        r'$\Theta$, $S$ at $k=0,1,2$ · $w$ at the cell base' + '\n'
        r'$Q_{\rm net}$, $Q_{\rm sw}$, FW flux, $h_{\rm bl}$' + '\n51 levels to ~968 m',
        fc='#DCE9F0', ec=TEAL, size=10.5)
    ax.text(0.1525, 0.935, 'DATA  (M2)', ha='center', fontsize=10.5, color=TEAL, weight='bold')

    box(ax, 0.345, 0.715, 0.235, 0.120,
        'Buoyancy\n' + r'$b = g\,\sigma_0(\Theta,S)/\rho_0$' + '\n[m s$^{-2}$]',
        fc=CARD, ec=DEEP)
    box(ax, 0.345, 0.530, 0.235, 0.150,
        'One operator set\nnative-grid gradients,\nthe same stencil\non both sides',
        fc=CARD, ec=DEEP)
    box(ax, 0.345, 0.350, 0.235, 0.145,
        'Top-hat filter at\n' + r'$L = 0,\ 2,\ 4,\ 8$ cells' + '\n(0, 3.6, 7.2, 14.4 km)',
        fc=CARD, ec=DEEP)

    arrow(ax, (0.287, 0.843), (0.345, 0.790), TEAL)
    arrow(ax, (0.287, 0.668), (0.345, 0.745), TEAL)
    arrow(ax, (0.4625, 0.715), (0.4625, 0.680))
    arrow(ax, (0.4625, 0.530), (0.4625, 0.495))

    box(ax, 0.655, 0.760, 0.330, 0.120,
        'PREDICTED\n' + r'$2F = -2\,(u_x b_x^2 + (u_y{+}v_x)\,b_xb_y + v_y b_y^2)$',
        fc='#E6F0E9', ec=GREEN, size=10.5)
    box(ax, 0.655, 0.590, 0.330, 0.130,
        'MEASURED\n' + r'$D_hG/Dt$ by a semi-Lagrangian step over' + '\n'
        r'one hour,  $G=|\nabla_h b|^2$',
        fc='#F5E9DF', ec=AMBER, size=10.5)
    arrow(ax, (0.580, 0.432), (0.655, 0.790), GREEN, lw=1.8)
    arrow(ax, (0.580, 0.412), (0.655, 0.640), AMBER, lw=1.8)

    box(ax, 0.655, 0.350, 0.330, 0.175,
        'FIVE-TERM BUDGET, per pixel, per hour, per $L$\n\n'
        r'$\dfrac{D_hG}{Dt} - 2F - \tau - V - S \;=\; R$',
        fc=CARD, ec=MIDNIGHT, size=11, lw=1.8)
    # the predicted arm is routed around the measured box, not through it
    ax.plot([0.655, 0.628, 0.628, 0.643], [0.790, 0.790, 0.438, 0.438],
            color=GREEN, lw=1.8, zorder=1, solid_capstyle='round')
    arrow(ax, (0.633, 0.438), (0.655, 0.438), GREEN, lw=1.8)
    arrow(ax, (0.820, 0.590), (0.820, 0.525), AMBER, lw=1.8)

    box(ax, 0.655, 0.190, 0.330, 0.110,
        'RESIDUAL  $R$\nthe gate: is  rms$(R)$ / rms(measured)  $\\leq 0.50$ ?',
        fc='#F6E2E2', ec=RED, size=10.5, lw=1.8)
    arrow(ax, (0.820, 0.350), (0.820, 0.300), RED, lw=2.2)

    box(ax, 0.02, 0.190, 0.555, 0.140,
        'WHERE THE BUDGET IS EVALUATED\n'
        '7-cell land halo (~13 km) + an explicit edge margin  ·  at least 100 km offshore\n'
        'front pixels: $G$ at or above its 90th percentile at the midpoint hour\n'
        '71 hourly pairs  ·  262,925 valid and 26,293 front pixels per hour',
        fc='#F3F6F8', ec=MUTED, size=10, tc=INK)

    ax.text(0.5, 0.080,
            'One tile of LLC4320 off Monterey Bay, grid spacing 1.68-2.07 km, 72 h in July 2012. '
            'Both sides of the budget are built\nwith the same discrete operators at the same '
            'filter scale, which is what makes the comparison an identity test rather than two '
            'guesses.',
            ha='center', va='center', fontsize=9.8, color=MUTED, linespacing=1.5)
    save(fig, 'schematic_pipeline.png')


# ====================================================== 2. the budget cartoon
def budget():
    fig = plt.figure(figsize=(13.4, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    halo = dict(bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.88))

    # ---- left: a front in cross-section ------------------------------------
    ax.set_xlim(0, 10); ax.set_ylim(-40, 16); ax.axis('off')
    ax.add_patch(Rectangle((0, -40), 10, 40, facecolor='#EAF2F7', edgecolor='none'))
    ax.add_patch(Rectangle((0, -1.4), 10, 1.4, facecolor='#CBDCE8', edgecolor=TEAL, lw=1.0))
    ax.plot([0, 10], [0, 0], color=DEEP, lw=2.2, zorder=5)
    ax.text(0.2, -0.72, r'top cell,  $\Delta z = 1$ m', fontsize=9, color=DEEP, va='center',
            zorder=6)

    # isopycnals through the front
    x = np.linspace(0.25, 9.75, 240)
    for z0 in (-14.0, -19.0, -24.0, -29.0):
        ax.plot(x, z0 + 2.6 * np.tanh((x - 5.0) / 0.9), color=MUTED, lw=1.5, zorder=3)
    ax.text(9.6, -31.0, 'isopycnals', fontsize=9, color=MUTED, style='italic',
            ha='right', zorder=6)
    ax.plot([5, 5], [-1.6, -32.5], color=MIDNIGHT, lw=1.0, ls=':', zorder=4)
    ax.text(5.0, -34.0, 'the front', fontsize=10, color=MIDNIGHT, ha='center',
            weight='bold', zorder=6)

    ax.plot([0, 10], [-22, -22], color=MIDNIGHT, lw=1.6, ls='--', zorder=4)
    ax.text(9.8, -21.3, r'KPP boundary layer,  $h_{\rm bl}\approx 22$ m', fontsize=9,
            color=MIDNIGHT, ha='right', va='bottom', zorder=6, **halo)

    # S -- air-sea fluxes, in the sky on the left
    ax.add_patch(Circle((1.15, 11.4), 0.62, facecolor='#F2C14E', edgecolor=AMBER, lw=1.2,
                        zorder=4))
    for a in np.arange(0, 2 * np.pi, np.pi / 4):
        ax.plot([1.15 + 0.85 * np.cos(a), 1.15 + 1.18 * np.cos(a)],
                [11.4 + 0.85 * np.sin(a) * 1.0, 11.4 + 1.18 * np.sin(a) * 1.0],
                color=AMBER, lw=1.2, zorder=4)
    for x0 in (0.75, 1.5, 2.25):
        ax.annotate('', xy=(x0, 0.6), xytext=(x0, 8.4),
                    arrowprops=dict(arrowstyle='-|>', color=AMBER, lw=1.8, mutation_scale=13))
    ax.text(3.6, 9.2, 'air-sea heat and freshwater fluxes  $\\rightarrow\\ S$',
            fontsize=10, color=AMBER, ha='left', va='center', weight='bold', zorder=6)
    ax.annotate('', xy=(9.3, 13.0), xytext=(7.0, 13.0),
                arrowprops=dict(arrowstyle='-|>', color=AMBER, lw=2.0, mutation_scale=14))
    ax.text(8.15, 14.0, 'wind', fontsize=9, color=AMBER, ha='center')

    # 2F -- confluent surface flow
    ax.annotate('', xy=(4.5, 2.3), xytext=(1.0, 2.3),
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2.6, mutation_scale=16))
    ax.annotate('', xy=(5.5, 2.3), xytext=(9.0, 2.3),
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2.6, mutation_scale=16))
    ax.text(5.0, 4.8, 'confluent flow sharpens the front  $\\rightarrow\\ 2F$',
            fontsize=10, color=GREEN, ha='center', weight='bold', zorder=6)

    # V -- vertical advection through the cell base
    for x0 in (4.35, 5.65):
        ax.annotate('', xy=(x0, -1.9), xytext=(x0, -7.0),
                    arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=2.0, mutation_scale=13))
    ax.text(6.25, -4.6, '$w$ through the\ncell base  $\\rightarrow\\ V$', fontsize=10,
            color=TEAL, ha='left', va='center', weight='bold', zorder=6)

    # tau -- eddies below the filter scale
    for cx, cy in ((1.45, -5.2), (2.75, -8.4)):
        t = np.linspace(0, 3.4 * np.pi, 220)
        ax.plot(cx + 0.075 * t * np.cos(t), cy + 0.075 * t * np.sin(t) * 1.45,
                color=DEEP, lw=1.6, zorder=4)
    ax.text(0.35, -11.6, 'eddies below the filter scale  $\\rightarrow\\ \\tau$',
            fontsize=10, color=DEEP, ha='left', va='center', weight='bold', zorder=6)

    # R -- numerics and interior KPP
    ax.add_patch(Rectangle((0.25, -39.0), 9.5, 3.6, facecolor='none', edgecolor=RED,
                           lw=1.3, hatch='////', alpha=0.6, zorder=3))
    ax.text(5.0, -37.2, 'numerical mixing + interior KPP  $\\rightarrow\\ R$',
            fontsize=10, color=RED, ha='center', va='center', weight='bold', zorder=6, **halo)

    ax.set_title('A front at the surface, and what acts on it', fontsize=12.5,
                 color=MIDNIGHT, weight='bold', pad=8)

    # ---- right: the equation, term by term, with measured amplitudes -------
    bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.axis('off')
    c = S['per_L']['2']['closure']['terms']
    rows = [
        (r'$\dfrac{D_hG}{Dt}$', 'measured', 'how fast the front actually sharpens,\n'
         'followed along the flow for one hour', MIDNIGHT, '1.000'),
        (r'$2F$', 'kinematic', 'the sharpening the resolved strain field\n'
         'predicts - the quantity under test', GREEN, f"{c['two_F']['over_measured']:.3f}"),
        (r'$\tau$', 'sub-filter', 'what the filter hides: motions smaller\n'
         'than $L$, zero by construction at $L=0$', DEEP,
         f"{c['subfilter']['over_measured']:.3f}"),
        (r'$V$', 'vertical', 'advection through the base of the 1 m\n'
         'top cell, measured from the chunk store', TEAL,
         f"{c['vertical']['over_measured']:.3f}"),
        (r'$S$', 'surface flux', 'the gradient of the air-sea buoyancy\n'
         'forcing, measured from the chunk store', AMBER,
         f"{c['surface_flux']['over_measured']:.3f}"),
        (r'$R$', 'residual', 'everything left over: the model numerics\n'
         'and interior KPP - and whatever is wrong', RED,
         f"{S['per_L']['2']['closure']['residual']['over_measured']:.3f}"),
    ]
    bx.text(0.5, 0.995, r'$\dfrac{D_hG}{Dt} \;-\; 2F \;-\; \tau \;-\; V \;-\; S \;=\; R$',
            ha='center', va='top', fontsize=16, color=MIDNIGHT)
    bx.text(0.5, 0.878, 'r.m.s. of each term as a fraction of the measured tendency,\n'
            'on front pixels at $L = 2$ cells (71 hourly pairs)',
            ha='center', va='top', fontsize=9.5, color=MUTED, linespacing=1.4)
    y = 0.795
    for sym, name, gloss, col, val in rows:
        bx.add_patch(FancyBboxPatch((0.015, y - 0.104), 0.97, 0.100,
                                    boxstyle='round,pad=0.004,rounding_size=0.012',
                                    facecolor='#FBFCFD', edgecolor=col, lw=1.3))
        bx.add_patch(Rectangle((0.015, y - 0.104), 0.012, 0.100, facecolor=col, ec='none'))
        bx.text(0.085, y - 0.054, sym, fontsize=15, color=col, ha='center', va='center')
        bx.text(0.155, y - 0.028, name, fontsize=10.5, color=col, va='center', weight='bold')
        bx.text(0.155, y - 0.075, gloss, fontsize=9.2, color=INK, va='center', linespacing=1.3)
        bx.text(0.962, y - 0.054, val, fontsize=12, color=col, ha='right', va='center',
                weight='bold', family='monospace')
        y -= 0.118
    bx.text(0.5, 0.012,
            'The residual is not small.  It is larger than the term it was meant to correct,\n'
            'and larger than the tolerance of 0.50 declared before any of it was computed.',
            ha='center', va='bottom', fontsize=9.6, color=RED, linespacing=1.45)
    save(fig, 'schematic_budget.png')


if __name__ == '__main__':
    pipeline()
    budget()
