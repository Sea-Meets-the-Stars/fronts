""" The figure for M3-digging task 4 -- ``figs/m3_tracer.png``: the tracer's closure
statistics beside the product's against ``L`` (top row), and the tracer's front population
and conservation against hour (bottom row)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

import m3_closure as mc                                        # noqa: E402

COL = dict(product='k', catchall='0.5', linear='#1f5fa8', piecewise='#e76f51')


def figure(res, path, dpi=150):
    Ls = [int(L) for L in res['product']]
    cols = res['columns']
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    stats = (('resid_over_meas', 'residual / measured (tracer: the catch-all)'),
             ('explained', 'explained fraction'),
             ('euler_gap', 'Eulerian - semi-Lagrangian gap / measured'))
    for a, (k, lab) in zip(ax[0], stats):
        a.plot(Ls, [res['product'][str(L)][k] for L in Ls], 'o-', color=COL['product'], label='product (5 terms)')
        if k != 'euler_gap':
            a.plot(Ls, [res['product'][str(L)]['catchall_over_meas' if k == 'resid_over_meas' else k] for L in Ls],
                   's--', color=COL['catchall'], label='product (2-term catch-all)' if k == 'resid_over_meas' else '_')
        for vel, c in cols.items():
            a.plot(Ls, [c['per_L'][str(L)][k] for L in Ls], 'D-', color=COL[vel], label=f'tracer, {vel} velocity')
        if k == 'resid_over_meas':
            th = res['declared']['reading']['thresholds']
            a.axhline(th['hi'], color='#b22222', ls='--', lw=1, label=f'{th["hi"]}: analysis-limited above')
            a.axhline(th['lo'], color='#2a9d8f', ls='--', lw=1, label=f'{th["lo"]}: data-limited below')
        if k == 'explained':
            a.axhline(0, color='0.7', lw=0.8)
        a.set_xlabel('L (cells)')
        a.set_ylabel(lab)
        a.set_xticks(Ls)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    # bottom: checks, from the linear column if present
    vel = 'linear' if 'linear' in cols else (next(iter(cols)) if cols else None)
    if vel and 'checks' in cols[vel] and 'checks_2_to_4' in cols[vel]['checks']:
        ch = cols[vel]['checks']['checks_2_to_4']
        pop = ch['front_population']
        a = ax[1, 0]
        hrs = sorted(int(h) for h in pop)
        for p, ls in ((50, ':'), (90, '-'), (99, '--')):
            a.plot(hrs, [pop[str(h)]['ratio'][str(p)] if str(p) in pop[str(h)]['ratio'] else pop[str(h)]['ratio'][p] for h in hrs],
                   'o' + ls, color=COL[vel], label=f'G p{p}: tracer / model')
        a.axhline(1, color='0.7', lw=0.8)
        a.set_xlabel('hour')
        a.set_ylabel('ratio')
        a.set_title(f'check 3: the front population ({vel})', fontsize=10)
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
        a = ax[1, 1]
        cons = ch['conservation']
        for k, c in (('Theta', '#d62728'), ('Salt', '#1f5fa8'), ('b', 'k')):
            v = np.asarray(cons[k]['var'])
            a.plot(np.arange(v.size), v / v[0], '-', color=c, label=f'{k}: variance / initial')
        a.set_xlabel('hour')
        a.set_ylabel('variance / initial (mask_analysis)')
        a.set_title('check 2: the variance over the window (Theta and b grow: fronts sharpen; Salt decays)', fontsize=10)
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
        a = ax[1, 2]
        for k, c in (('Theta', '#d62728'), ('b', 'k')):
            m = np.asarray(cons[k]['mean'])
            a.plot(np.arange(m.size), (m - m[0]) / np.sqrt(cons[k]['var'][0]), '-', color=c,
                   label=f'{k}: (mean - initial) / initial sd')
        a.plot(np.arange(len(cons['margin_inflow_proxy'])), cons['margin_inflow_proxy'], ':', color='0.4',
               label='|tracer - model| on the first interior ring [K]')
        a.set_xlabel('hour')
        a.set_title('check 2: the mean drift, against what the clamp lets in', fontsize=10)
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    for vel, c in cols.items():
        r = c['reading']
        fig.text(0.5, 0.955 - 0.02 * list(cols).index(vel),
                 f'{vel}: r_tr(0) = {r["L0"]["r_tr"]:.3f} -> {r["L0"]["reading"]};  r_tr(8) = {r["L8"]["r_tr"]:.3f} '
                 f'vs the product catch-all {r["L8"]["product_catchall"]:.3f} -> {r["L8"]["reading"]}',
                 ha='center', fontsize=9, color=COL[vel])
    fig.suptitle('M3 digging task 4 -- the offline tracer: the pipeline on a case whose budget is '
                 '2F + tau + numerics exactly', fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    d = res['declared']['design']
    fig.text(0.5, 0.008, f'Theta and Salt advected from the model\'s hour 0 by OS7MP ({d["form"]}, {d["integrator"]}, '
                         f'dt_sub {d["dt_sub_s"]:.0f} s); margin of {7} cells clamped to the model; pre-declared reading '
                         f'thresholds {res["declared"]["reading"]["thresholds"]}; slopes are calibration, never an efficiency.',
             ha='center', fontsize=7.5, color='0.3', wrap=True)
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
    return path
