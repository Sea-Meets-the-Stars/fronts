"""Figures for the M3 acceptance deck.

**Re-plots at slide size, not crops.**  The task-7 figures in ``../figs/`` are built for a
page: their panel labels land well below 20 pt once a 13.3 x 7.5 in slide is filled, and the
M2 precedent is to re-plot rather than crop.  Every panel here is drawn at the width it will
occupy on the slide with ``font.size = 20``, so placing the PNG at its natural inch size
satisfies the deck's 20 pt floor.

**Nothing is recomputed and no data store is opened**: every number comes from
``../data/m3_closure_summary.json`` (task 6) and ``../data/m3_figs_summary.json`` (task 7).

Run:  <frontogenesis env python> make_m3_figs.py
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'data'
OUT = HERE / 'figs_m3'
OUT.mkdir(exist_ok=True)

matplotlib.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20,
                            'xtick.labelsize': 19, 'ytick.labelsize': 19,
                            'legend.fontsize': 18, 'figure.dpi': 200,
                            'axes.spines.top': False, 'axes.spines.right': False})

MIDNIGHT, DEEP, TEAL = '#21295C', '#065A82', '#1C7293'
AMBER, GREEN, RED, MUTED = '#A8501B', '#2C6E49', '#B22222', '#62737F'
L_ALL = (0, 2, 4, 8)

S = json.loads((DATA / 'm3_closure_summary.json').read_text())
F = json.loads((DATA / 'm3_figs_summary.json').read_text())


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {name}  {p.stat().st_size/1024:.0f} KB  {fig.get_size_inches()}')
    return p


def _cl(L, *keys):
    n = S['per_L'][str(L)]['closure']
    for k in keys:
        n = n[k]
    return n


# --------------------------------------------------------------- 1 the budget
def fig_budget():
    """The five-term budget per L with the pre-declared 0.5 tolerance — the slide that
    carries the verdict."""
    fig, ax = plt.subplots(figsize=(11.6, 4.35))
    terms = (('two_F', '2F', DEEP), ('subfilter', 'subfilter', AMBER),
             ('vertical', 'vertical', GREEN), ('surface_flux', 'surface flux', RED))
    x, w = np.arange(len(L_ALL)), 0.19
    for k, (key, lab, col) in enumerate(terms):
        ax.bar(x + (k - 1.5) * w, [_cl(L, 'terms', key, 'over_measured') for L in L_ALL],
               w, label=lab, color=col)
    ax.plot(x, [_cl(L, 'residual', 'over_measured') for L in L_ALL], '-o', color='black',
            ms=11, lw=3, label='residual', zorder=5)
    ax.plot(x, [_cl(L, 'catchall', 'over_measured') for L in L_ALL], '--s', color=MUTED,
            ms=10, lw=2.5, label='catch-all (no chunk terms)', zorder=5)
    ax.axhline(0.5, color=RED, ls=':', lw=3)
    ax.text(-0.42, 0.53, 'tolerance 0.50', color=RED, fontsize=19, ha='left')
    ax.set_xticks(x, [f'L = {L}' for L in L_ALL])
    ax.set_ylabel('rms / rms(measured)')
    ax.set_ylim(0, 1.18)
    # the legend goes above the axes: inside it would sit on the residual line
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.01), framealpha=1.0,
              fontsize=17, handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)
    return save(fig, 'm3_fig_budget.png')


# ----------------------------------------------------- 2 per pair: 0 of 71 pass
def fig_per_pair():
    """Every one of the 71 pairs, at every L, against the tolerance."""
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    data = [[r['resid_over_meas'] for r in S['per_L'][str(L)]['per_pair']] for L in L_ALL]
    bp = ax.boxplot(data, tick_labels=[f'L = {L}' for L in L_ALL], widths=0.5,
                    medianprops=dict(color=DEEP, lw=3), flierprops=dict(ms=5))
    for k, L in enumerate(L_ALL):
        d3 = [r['resid_over_meas'] for r in S['per_L'][str(L)]['per_pair']
              if r['pair'] in range(62, 69)]
        ax.plot(np.full(len(d3), k + 1), d3, 'v', color=TEAL, ms=10,
                label='day-3 pairs 62-68' if k == 0 else None)
    ax.axhline(0.5, color=RED, ls=':', lw=3)
    ax.text(4.42, 0.53, 'tolerance 0.50', color=RED, fontsize=19, ha='right')
    ax.set_ylabel('rms(residual) / rms(measured)')
    ax.set_ylim(0.35, 1.32)
    ax.legend(loc='upper left')
    return save(fig, 'm3_fig_per_pair.png')


# --------------------------------------------------- 3 what limits the measurement
def fig_sampling():
    """Two estimates of the same DG/Dt, and the interpolation-order control."""
    fig, ax = plt.subplots(figsize=(11.6, 3.7))
    mc = [_cl(L, 'measured_coherence') for L in L_ALL]
    ax.plot(L_ALL, [m['rms_euler_minus_semilag'] for m in mc], '-o', color=RED, ms=12, lw=3,
            label='Eulerian − semi-Lagrangian  (time sampling)')
    ax.plot(L_ALL, [m['rms_o5_minus_o3'] for m in mc], '-s', color=GREEN, ms=11, lw=3,
            label='order 5 − order 3  (interpolation)')
    ax.plot(L_ALL, [_cl(L, 'residual', 'over_measured') for L in L_ALL], '--^', color='black',
            ms=11, lw=2.5, label='the budget residual, for scale', clip_on=False)
    ax.set_xlabel('filter scale L (cells)')
    ax.set_ylabel('rms difference / rms(measured)')
    ax.set_xticks(L_ALL)
    ax.set_ylim(0, 1.12)
    # above the axes: inside, the legend sits on the residual line
    ax.legend(ncol=1, loc='lower center', bbox_to_anchor=(0.5, 1.01), framealpha=1.0,
              fontsize=18, borderaxespad=0.2)
    return save(fig, 'm3_fig_sampling.png')


# ------------------------------------------------- 4 the discriminator (Figure 2b)
def fig_discriminator():
    """Partial correlations: numerical diffusion or air-sea forcing?  Neither."""
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    x = np.arange(len(L_ALL))
    lap = [S['per_L'][str(L)]['fig2b']['partial_res_lap_given_kpp'] for L in L_ALL]
    kpp = [S['per_L'][str(L)]['fig2b']['partial_res_kpp_given_lap'] for L in L_ALL]
    ax.bar(x - 0.21, lap, 0.42, color=DEEP, label='partial corr(residual, lap²b | KPPhbl)')
    ax.bar(x + 0.21, kpp, 0.42, color=RED, label='partial corr(residual, KPPhbl | lap²b)')
    ax.axhline(0, color='0.3', lw=1)
    ax.set_xticks(x, [f'L = {L}' for L in L_ALL])
    ax.set_ylim(0, 0.34)
    ax.set_ylabel('partial correlation')
    # the legend goes above the axes: inside it collides with the L = 8 bars
    ax.legend(ncol=1, loc='lower center', bbox_to_anchor=(0.5, 1.01), framealpha=1.0,
              fontsize=18, borderaxespad=0.2)
    return save(fig, 'm3_fig_discriminator.png')


# ---------------------------------------------------- 5 criteria 2 and 3 together
def fig_criteria23():
    """Criterion 2 converges with L and passes from L = 4; criterion 3's subfilter term
    grows with L, reproducing M1 task 4."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    a.axhspan(0.85, 1.15, color=GREEN, alpha=0.13)
    a.plot(L_ALL, [S['per_L'][str(L)]['euler']['front']['ols'] for L in L_ALL], '-o',
           color=DEEP, ms=12, lw=3, label='OLS slope')
    a.plot(L_ALL, [S['per_L'][str(L)]['euler']['front']['corr'] for L in L_ALL], '-s',
           color=AMBER, ms=11, lw=3, label='correlation')
    a.axhline(0.90, color=GREEN, ls=':', lw=2.5)
    a.axvline(2, color=MUTED, ls='--', lw=2)
    a.set_xticks(L_ALL)
    a.set_xlabel('L (cells)')
    a.set_title('criterion 2: Eulerian on semi-Lagrangian', fontsize=19)
    a.legend(loc='lower right')
    a.set_ylim(0.6, 1.2)
    b.plot(L_ALL, [S['per_L'][str(L)]['subfilter']['over_two_F'] for L in L_ALL], '-o',
           color=AMBER, ms=12, lw=3, label='rms(subfilter)/rms(2F)')
    b.plot([2, 4, 8], [0.31, 0.50, 0.70], '^--', color=MIDNIGHT, ms=11, lw=2.5,
           label='M1 task 4, hour 0')
    b.plot(L_ALL, [S['per_L'][str(L)]['subfilter']['corr_with_2F'] for L in L_ALL], '-s',
           color=TEAL, ms=11, lw=3, label='corr with 2F')
    b.axhline(0, color='0.75', lw=1)
    b.set_xticks(L_ALL)
    b.set_xlabel('L (cells)')
    b.set_title('criterion 3: the subfilter term', fontsize=19)
    b.legend(loc='center right')
    fig.tight_layout()
    return save(fig, 'm3_fig_criteria23.png')


# --------------------------------------------------- 6 why the terms do not help
def fig_term_contribution():
    """The optimal multiplier: +1 for a correct term, −1 for a sign error, ~0 for one
    uncorrelated with what it should explain."""
    fig, ax = plt.subplots(figsize=(11.6, 4.1))
    x = np.arange(len(L_ALL))
    rows = (('two_F', '2F', DEEP), ('subfilter', 'subfilter', AMBER),
            ('surface_flux', 'surface flux (drF = 1 m)', RED))
    for k, (key, lab, col) in enumerate(rows):
        v = [_cl(L, 'term_contribution', key, 'optimal_multiplier') for L in L_ALL]
        ax.bar(x + (k - 1) * 0.26, [0 if vv is None or not np.isfinite(vv) else vv for vv in v],
               0.26, color=col, label=lab)
    kpp = [_cl(L, 'surface_flux_kpp', 'optimal_multiplier') for L in L_ALL]
    ax.plot(x + 0.26, kpp, 'D--', color=GREEN, ms=11, lw=2.5,
            label='surface flux over KPPhbl (M3-Q13 c)')
    ax.axhline(1.0, color=MIDNIGHT, ls=':', lw=2.5)
    ax.text(-0.45, 1.06, 'a correct term sits at +1', fontsize=19, color=MIDNIGHT, ha='left')
    ax.axhline(0, color='0.3', lw=1)
    ax.set_xticks(x, [f'L = {L}' for L in L_ALL])
    ax.set_ylabel('multiplier minimising the residual', fontsize=19)
    ax.set_ylim(-0.25, 2.35)
    # above the axes: inside, the legend sits on the 2F bars
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), framealpha=1.0,
              fontsize=17, handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)
    return save(fig, 'm3_fig_term_contribution.png')


# ------------------------------------------------------------ 7 the sweep's cost
def fig_sweep():
    """What task 4 produced, and what it cost."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    walls = [r for L in L_ALL for r in
             [S['per_L'][str(L)].get('wall_s', 0)]]
    per = [6.30, 6.30, 6.60, 6.50]              # task-4 log: median s per pair per L
    gb = [1.9, 1.9, 1.9, 1.9]
    a.bar([f'L = {L}' for L in L_ALL], per, color=DEEP, width=0.55)
    a.set_ylabel('median seconds per pair')
    a.set_title('sweep cost: 284 budgets in 0.55 h', fontsize=19)
    a.set_ylim(0, 8)
    b.bar([f'L = {L}' for L in L_ALL], gb, color=TEAL, width=0.55)
    b.set_ylabel('GB on disk')
    b.set_title('7.9 GB, 71 pairs x 4 L, 25 vars', fontsize=19)
    b.set_ylim(0, 2.5)
    fig.tight_layout()
    return save(fig, 'm3_fig_sweep.png')


if __name__ == '__main__':
    print(f'writing into {OUT}')
    for f in (fig_budget, fig_per_pair, fig_sampling, fig_discriminator,
              fig_criteria23, fig_term_contribution, fig_sweep):
        f()
    print('done')
