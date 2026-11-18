""" The figure for M3-digging task 2 -- ``figs/m3_controls.png``.

Three rows (the residual ratio, the explained fraction, the residual's slope on
``2F``) by one column per ``L``.  In each panel the real value is a point with
its 32-cell / 3-h block-bootstrap interval; the permutation and roll controls
are their 5-95 % bands with the median; the two lag controls are points with
their own intervals; the independent-field reference and the gate's tolerance
are drawn where they apply.  The caption states the pool, the seeds and the
pre-declaration.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

import m3_closure as mc                                        # noqa: E402
import m3_controls as ct                                       # noqa: E402

ROWS = (('resid_over_meas', 'residual / measured (rms)'),
        ('explained', 'explained fraction'),
        ('resid_on_2F', 'residual slope on 2F'))
COLS = ('real', 'perm', 'lag24', 'lag48', 'roll')
C = dict(real='k', perm='#1f5fa8', lag24='#d62728', lag48='#e89a9a', roll='#2a9d8f')


def figure(res, path, dpi=150):
    Ls = list(res['per_L'])
    fig, ax = plt.subplots(len(ROWS), len(Ls), figsize=(4.2 * len(Ls), 10.5), sharex=True,
                           squeeze=False)
    x = np.arange(len(COLS))
    for j, L in enumerate(Ls):
        d = res['per_L'][L]
        r, c, s = d['real'], d['controls'], d['skill']
        for i, (k, lab) in enumerate(ROWS):
            a = ax[i, j]
            ci = r['ci'][f'{ct.REAL_CI_HOURS}h'][k]
            a.errorbar([0], [r[k]], yerr=[[r[k] - ci['lo']], [ci['hi'] - r[k]]], fmt='o',
                       color=C['real'], capsize=4, zorder=5, label='real, 3-h block CI')
            for name, xi in (('permutation', 1), ('roll', 4)):
                dd = c[name]['dist'][k]
                a.fill_between([xi - 0.3, xi + 0.3], dd['p5'], dd['p95'],
                               color=C['perm' if name == 'permutation' else 'roll'], alpha=0.3,
                               label=f'{name} 5-95 %')
                a.plot([xi - 0.3, xi + 0.3], [dd['median']] * 2,
                       color=C['perm' if name == 'permutation' else 'roll'], lw=2)
            for lag, xi in ((24, 2), (48, 3)):
                cc = c[f'lag{lag}']
                lci = cc['ci'][k]
                a.errorbar([xi], [cc[k]], yerr=[[cc[k] - lci['lo']], [lci['hi'] - cc[k]]],
                           fmt='s', color=C[f'lag{lag}'], capsize=4, zorder=5,
                           label=f'{lag}-h lag, its own CI')
            if k == 'resid_over_meas':
                a.axhline(r['independent_reference'], ls=':', color='0.4',
                          label='independent-field reference')
                a.axhline(mc.TOL_RESID_RATIO, ls='--', color='#b22222', lw=1,
                          label=f'gate tolerance {mc.TOL_RESID_RATIO}')
            if k == 'explained':
                a.axhline(0.0, color='0.6', lw=0.8)
            if k == 'resid_on_2F':
                a.axhline(0.0, color='0.6', lw=0.8)
            a.set_xticks(x)
            a.set_xticklabels(COLS)
            a.grid(alpha=0.3)
            if j == 0:
                a.set_ylabel(lab)
            if i == 0:
                a.set_title(f'L = {L}   |   ' + ('SKILL over no-skill' if s['passed'] else
                                                  'no skill by the declared rule'),
                            fontsize=10, color=('#2a9d8f' if s['passed'] else '#b22222'))
    ax[0, 0].legend(fontsize=7, loc='lower right')
    fig.suptitle('M3 digging task 2 -- the controls: the real residual against three "no skill" '
                 'constructions', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    dec = res['declared']
    fig.text(0.5, 0.012,
             f'pool: {dec["pool"]}.  seeds: {dec["seeds"]["rng"]}, {dec["seeds"]["n_perm"]} '
             f'permutations (derangements), {dec["seeds"]["n_roll"]} rolls >= '
             f'{dec["seeds"]["roll_min_cells"]} cells, lags {dec["seeds"]["lags_h"]} h without '
             f'wrap.\npre-declared: {dec["skill_rule"]["rule"]} -- {dec["skill_rule"]["status"]}.',
             ha='center', fontsize=7.5, color='0.3', wrap=True)
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
    return path
