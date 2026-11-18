""" The figure for M3-digging task 3 -- ``figs/m3_term_tests.png``.

Left: per ``L``, the local-hour composites of the three components of the
surface-flux term (pool means over the measured rms) with the summed term and
the imbalance ``I``.  Right: the four smoke pairs' vertical terms -- the
top-cell term, its factorised forms with ``b_z`` of order 1 and 2, and the
second cell's term -- as rms over measured, with the slope of the second-cell
term on the top-cell term written on each bar group.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

import m3_term_tests as tt                                     # noqa: E402

COL = dict(nonsolar='#d62728', shortwave='#f4a259', freshwater='#1f5fa8', S='k', I='0.5')
VCOL = dict(V_top='#2a9d8f', V_fact_o1='#8fd3c9', V_fact_o2='#5fb8ab', V_cell2='#e76f51')


def figure(res, path, dpi=150):
    has_split, has_v = 'split' in res, 'vertical' in res
    Ls = list(res['split']['per_L']) if has_split else []
    nrow = max(len(Ls), 1)
    fig = plt.figure(figsize=(15, 3.2 * nrow + 1.5))
    gs = fig.add_gridspec(nrow, 2, width_ratios=(1.25, 1))
    if has_split:
        for i, L in enumerate(Ls):
            a = fig.add_subplot(gs[i, 0])
            d = res['split']['per_L'][L]['depth']['top']
            bl = {int(h): v for h, v in d['by_local_hour'].items()}   # int keys in memory, str from JSON
            hrs = sorted(bl)
            for c in tt.COMP:
                a.plot(hrs, [bl[h][f'{c}_mean_over_rms_meas'] for h in hrs], '-o', ms=3,
                       color=COL[c], label=c)
            a.plot(hrs, [bl[h]['S_mean_over_rms_meas'] for h in hrs], '-', color=COL['S'],
                   lw=2, label='S (sum)')
            a.plot(hrs, [bl[h]['I_mean_over_rms_meas'] for h in hrs], '--', color=COL['I'],
                   lw=2, label='imbalance I')
            a.axhline(0, color='0.7', lw=0.8)
            f3 = d['fit3']
            cz = d['cancellation']
            a.set_title(f'L = {L}   |   a = ' + ', '.join(f'{f3["multipliers"][c]["value"]:+.2f}' for c in tt.COMP)
                        + f'   |   rms/I: unit {f3["rms_over_I"]["unit"]:.3f}, fit {f3["rms_over_I"]["fit"]:.3f}'
                        + ('   CANCELLATION' if cz['passed'] else '   orthogonality stands'),
                        fontsize=9, color=('#b22222' if cz['passed'] else '0.2'))
            a.set_ylabel('pool mean / rms(measured)')
            if i == nrow - 1:
                a.set_xlabel('local solar hour')
            if i == 0:
                a.legend(fontsize=7, ncol=3)
            a.grid(alpha=0.3)
    if has_v:
        a = fig.add_subplot(gs[:, 1])
        rows = res['vertical']['rows']
        keys = ('V_top', 'V_fact_o1', 'V_fact_o2', 'V_cell2')
        x = np.arange(len(rows))
        w = 0.2
        for k, key in enumerate(keys):
            a.bar(x + (k - 1.5) * w, [r['terms'][key]['rms_over_meas'] for r in rows], w,
                  color=VCOL[key], label=key)
        for xi, r in zip(x, rows):
            a.text(xi, max(r['terms'][k]['rms_over_meas'] for k in keys) * 1.03,
                   f'slope {r["cell2_on_top"]["ols"]:+.2f}\ncorr {r["cell2_on_top"]["corr"]:+.2f}\n'
                   f'signs {r["terms"]["V_top"]["sign_of_mean"]:+d}/{r["terms"]["V_cell2"]["sign_of_mean"]:+d}',
                   ha='center', fontsize=7)
        a.set_xticks(x)
        a.set_xticklabels([f'L={r["L"]}\npair {r["pair"]}\n{r["lst"]:.0f} LST' for r in rows], fontsize=8)
        a.set_ylabel('rms / rms(measured)')
        a.set_title('V: top cell, factorised (b_z order 1, 2), second cell', fontsize=10)
        a.legend(fontsize=8)
        a.grid(alpha=0.3, axis='y')
    fig.suptitle('M3 digging task 3 -- the term tests: S split into its parts; V from the second cell',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    dec = res['declared']
    fig.text(0.5, 0.01, 'pre-declared: ' + dec['split']['rule'] + '  |  V: ' + dec['vertical']['rule'],
             ha='center', fontsize=7, color='0.3', wrap=True)
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
    return path
