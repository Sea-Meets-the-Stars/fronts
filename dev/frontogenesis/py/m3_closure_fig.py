""" The working figure for M3 task 6 -- ``figs/m3_closure.png``.

Six panels off ``m3_closure.run``'s result dict: the five-term budget by
``L``, the residual ratio's distribution over the 71 pairs, the
semi-Lagrangian/Eulerian agreement, the subfilter sweep, the residual
against ``lap2_b`` and ``KPPhbl`` (Figure 2b's question), and the slopes
against the M1 baseline.  Deliberately plain -- task 7 makes the publication
figures from the same JSON.  Every panel that shows a tolerance draws it.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

import m3_closure as mc                                        # noqa: E402

TERM_COLOR = dict(two_F='#1f77b4', subfilter='#ff7f0e', vertical='#2ca02c',
                  surface_flux='#d62728')


def figure(res, path, dpi=150):
    Ls = [int(L) for L in res['per_L']]
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.5))
    _terms(ax[0, 0], res, Ls)
    _resid_dist(ax[0, 1], res, Ls)
    _euler(ax[0, 2], res, Ls)
    _subfilter(ax[1, 0], res, Ls)
    _fig2b(ax[1, 1], res, Ls)
    _slopes(ax[1, 2], res, Ls)
    v = res['verdict']
    fig.suptitle('M3 closure — the hard gate   |   ' + v['statement'], fontsize=13,
                 color=('#2ca02c' if v['criterion1']['passed'] else '#b22222'))
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.text(0.5, 0.005, 'tolerances pre-declared 2026-10-07 (M3-Q1/Q2/Q9): '
             f'resid/meas <= {mc.TOL_RESID_RATIO}, explained >= {mc.TOL_EXPLAINED}, '
             f'|resid on 2F| <= {mc.TOL_RESID_SLOPE}; judged at L >= 2',
             ha='center', fontsize=8, color='0.35')
    path = str(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _terms(a, res, Ls):
    w = 0.18
    x = np.arange(len(Ls))
    for k, t in enumerate(mc.TERMS):
        v = [res['per_L'][str(L)]['closure']['terms'][t]['over_measured'] for L in Ls]
        a.bar(x + (k - 1.5) * w, v, w, label=t, color=TERM_COLOR[t])
    r = [res['per_L'][str(L)]['closure']['residual']['over_measured'] for L in Ls]
    c = [res['per_L'][str(L)]['closure']['catchall']['over_measured'] for L in Ls]
    a.plot(x, r, 'k-o', label='residual', zorder=5)
    a.plot(x, c, '--s', color='0.5', label='catch-all (no chunk terms)', zorder=5)
    a.axhline(mc.TOL_RESID_RATIO, color='#b22222', ls=':', lw=2,
              label=f'tolerance {mc.TOL_RESID_RATIO}')
    a.set_xticks(x, [f'L={L}' for L in Ls])
    a.set_ylabel('rms / rms(measured)')
    a.set_title('(a) the five-term budget')
    a.legend(fontsize=7, ncol=2)


def _resid_dist(a, res, Ls):
    data = [[r['resid_over_meas'] for r in res['per_L'][str(L)]['per_pair']] for L in Ls]
    a.boxplot(data, tick_labels=[f'L={L}' for L in Ls], showfliers=True)
    for k, L in enumerate(Ls):
        d3 = [r['resid_over_meas'] for r in res['per_L'][str(L)]['per_pair']
              if r['pair'] in mc.DAY3_PAIRS]
        a.plot(np.full(len(d3), k + 1), d3, 'v', color='#9467bd', ms=5,
               label='day-3 pairs 62-68' if k == 0 else None)
    a.axhline(mc.TOL_RESID_RATIO, color='#b22222', ls=':', lw=2)
    a.set_ylabel('rms(residual) / rms(measured)')
    a.set_title('(a) per pair, 71 pairs')
    a.legend(fontsize=7)


def _euler(a, res, Ls):
    s = [res['per_L'][str(L)]['euler']['front']['ols'] for L in Ls]
    c = [res['per_L'][str(L)]['euler']['front']['corr'] for L in Ls]
    a.plot(Ls, s, '-o', label='OLS slope (front)')
    a.plot(Ls, c, '-s', label='corr (front)')
    a.axhspan(*mc.TOL_EULER_SLOPE, color='#2ca02c', alpha=0.12, label='slope tolerance')
    a.axhline(mc.TOL_EULER_CORR, color='#2ca02c', ls=':', label=f'corr >= {mc.TOL_EULER_CORR}')
    a.axvline(2, color='0.6', ls='--', lw=1)
    a.text(2.05, a.get_ylim()[0], ' gated from L=2', fontsize=7, color='0.4', va='bottom')
    a.set_xlabel('L (cells)')
    a.set_title('(b) Eulerian on semi-Lagrangian')
    a.legend(fontsize=7)


def _subfilter(a, res, Ls):
    o = [res['per_L'][str(L)]['subfilter']['over_two_F'] for L in Ls]
    c = [res['per_L'][str(L)]['subfilter']['corr_with_2F'] for L in Ls]
    a.plot(Ls, o, '-o', color='#ff7f0e', label='rms(subfilter)/rms(2F)')
    a.plot(Ls, c, '-s', color='#8c564b', label='corr with 2F')
    a.plot([2, 4, 8], [0.31, 0.50, 0.70], 'k^--', ms=5, label='M1 task 4, hour 0')
    a.plot([2, 4, 8], [-0.66, -0.60, -0.54], 'kv--', ms=5)
    a.axhline(0, color='0.8', lw=0.8)
    a.set_xlabel('L (cells)')
    a.set_title('(c) the subfilter term grows with L')
    a.legend(fontsize=7)


def _fig2b(a, res, Ls):
    lap = [res['per_L'][str(L)]['fig2b']['partial_res_lap_given_kpp'] for L in Ls]
    kpp = [res['per_L'][str(L)]['fig2b']['partial_res_kpp_given_lap'] for L in Ls]
    x = np.arange(len(Ls))
    a.bar(x - 0.2, lap, 0.4, label='partial corr(res, lap2_b | KPPhbl)', color='#1f77b4')
    a.bar(x + 0.2, kpp, 0.4, label='partial corr(res, KPPhbl | lap2_b)', color='#d62728')
    a.axhline(0, color='0.3', lw=0.8)
    a.set_xticks(x, [f'L={L}' for L in Ls])
    a.set_title('(e) numerical diffusion or air-sea forcing?')
    a.legend(fontsize=7)


def _slopes(a, res, Ls):
    for k, L in enumerate(Ls):
        d = res['per_L'][str(L)]
        s = d.get('slopes') or d.get('slopes_not_quoted')
        e = s['primary']['estimates']
        for j, (name, col) in enumerate((('ols', '#1f77b4'), ('trimmed', '#ff7f0e'),
                                         ('tls', '#2ca02c'))):
            a.errorbar(k + (j - 1) * 0.18, e[name]['value'],
                       yerr=[[e[name]['value'] - e[name]['lo']], [e[name]['hi'] - e[name]['value']]],
                       fmt='o', color=col, ms=5, capsize=3,
                       label=name if k == 0 else None)
        if not d['slopes_quotable']:
            a.text(k, a.get_ylim()[1], 'not\nquoted', ha='center', va='top', fontsize=7,
                   color='#b22222')
    a.axhspan(*mc.st.BASELINE_CI, color='0.6', alpha=0.25, label='M1 baseline 0.981')
    a.axhspan(*mc.st.V3B_BAND, color='#9467bd', alpha=0.12, label='V3b band')
    a.axhline(1.0, color='0.8', ls=':', lw=1)
    a.set_xticks(np.arange(len(Ls)), [f'L={L}' for L in Ls])
    a.set_ylabel('slope of measured on 2F')
    a.set_title('(d) slopes vs the baseline — NEVER vs 1')
    a.legend(fontsize=7)
