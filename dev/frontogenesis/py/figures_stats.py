""" M3 task 7, the statistics family: Figures 2, 2b, 3, 4, 6 and 7.

Split out of :mod:`figures` for coding §1.3's line cap; import them from
there.  Every number drawn comes from the derived stores or from
``m3_closure_summary.json`` -- nothing is recomputed, and
``tests/test_figures.py`` checks that the two agree.

**Baselines are never 1.**  Figure 2 and Figure 3 draw the M1 discrete-null
slope 0.981 [0.970, 0.994] as the reference line, with V3b's 0.954-1.003 and
the temporal 0.972 ± 0.020 beside it as separate bands (M1-Q4, M1-Q2, M2
task 6 item 7).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import figures as fg


def _baselines(ax, horizontal=True, legend=True):
    """The three reference bands every slope figure carries."""
    span = ax.axhspan if horizontal else ax.axvspan
    line = ax.axhline if horizontal else ax.axvline
    span(*fg.V3B_BAND, color='#5b2a86', alpha=0.10,
         label='V3b model-advection band 0.954-1.003' if legend else None)
    span(*fg.BASELINE_CI, color='0.45', alpha=0.30,
         label='M1 discrete null 0.981 [0.970, 0.994]' if legend else None)
    line(fg.BASELINE, color='0.25', lw=1.6)
    m, sd = fg.TEMPORAL
    (ax.errorbar if horizontal else ax.errorbar)(
        [], [], [], color=fg.COL['grey'])
    span(m - sd, m + sd, color='#2a9d8f', alpha=0.08,
         label=f'temporal systematic {m} ± {sd}' if legend else None)
    line(1.0, color='0.8', ls=':', lw=1.0)


# ---------------------------------------------------------------------------
# Figure 2 -- the joint PDF.  NOT QUOTED (M3-Q15 (a)).
# ---------------------------------------------------------------------------
def fig02_joint_pdf(data_dir=None, fig_dir=None, summary=None, Ls=fg.L_ALL, bins=140):
    """Joint PDF of ``measured`` against ``2F`` on front pixels, all 71 pairs
    pooled, one panel per ``L``: the 1:1 line, the OLS / trimmed / orthogonal
    fits, binned ``E[Y|X]`` by sign, the **baseline drawn explicitly at
    0.981 with its band**, V3b's band beside it, the temporal systematic as
    a third marker, and the day-3 pairs as a separate marker.

    **The slopes are drawn under a NOT QUOTED banner.** Task 6's criterion 1
    failed at every ``L``, so by the prompt's "Do not" list no frontogenesis
    efficiency may be reported from them (M3-Q15 (a), JXP 2026-10-10).  The
    figure is here because the *shape* of the cloud is the evidence, not the
    slope: only interpretable beside Figure 2b.
    """
    s = fg.load_summary(summary)
    fig, axs = plt.subplots(1, len(Ls), figsize=(5.4 * len(Ls), 5.6), sharey=False)
    out = {}
    for c, L in enumerate(Ls):
        ax = axs[c]
        d = fg.front_pool(L, ('two_F', 'DGDt_semilag'), data_dir)
        x, y = d['two_F'], d['DGDt_semilag']
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        lim = fg.sym_limit(x, y, pct=99.5)
        ax.hist2d(x, y, bins=bins, range=[[-lim, lim], [-lim, lim]], norm=LogNorm(),
                  cmap='Greys')
        ax.plot([-lim, lim], [-lim, lim], color='0.4', lw=1.0, ls='--', label='1:1')
        node = s['per_L'][str(L)]
        sl = (node.get('slopes') or node.get('slopes_not_quoted'))
        e = sl['primary']['estimates']
        for name, col in (('ols', fg.COL['two_F']), ('trimmed', fg.COL['subfilter']),
                          ('tls', fg.COL['vertical'])):
            ax.plot([-lim, lim], [-lim * e[name]['value'], lim * e[name]['value']],
                    color=col, lw=1.6, label=f'{name} {e[name]["value"]:.3f}')
        ax.plot([-lim, lim], [-lim * fg.BASELINE, lim * fg.BASELINE], color='0.25', lw=1.4,
                ls='-.', label=f'M1 baseline {fg.BASELINE}')
        df = fg.binned(x, y, bins=10)
        ax.plot(df['x_mean'], df['y_mean'], 'o-', color=fg.COL['red'], ms=4, lw=1.3,
                label='binned E[Y|X]')
        # the day-3 marker comes from the summary's own fit (task 6), not from a
        # second pass over the store -- one source of numbers, as the prompt asks
        if 'day3' in sl:
            ax.plot([-lim, lim], [-lim * sl['day3']['value'], lim * sl['day3']['value']],
                    color=fg.COL['day3'], lw=1.2, ls=':',
                    label=f'day-3 pairs 62-68: {sl["day3"]["value"]:.3f}')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('2F  [s$^{-5}$]')
        if c == 0:
            ax.set_ylabel('measured  DGDt_semilag  [s$^{-5}$]')
        ax.set_title(fg.mask_note(L) + f'\nn = {x.size:,}', fontsize=8)
        ax.legend(fontsize=6.5, loc='upper left')
        ax.text(0.98, 0.02, 'SLOPES NOT QUOTED\ncriterion 1 failed at this L',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                color=fg.COL['red'], weight='bold',
                bbox=dict(fc='white', alpha=0.85, ec=fg.COL['red']))
        out[f'L{L}'] = {k: e[k]['value'] for k in ('ols', 'trimmed', 'tls')}
        out[f'L{L}']['n'] = int(x.size)
    fig.suptitle('Figure 2 — measured vs 2F on front pixels, 71 pairs pooled   |   '
                 'the slopes are NOT a frontogenesis efficiency (task 6: criterion 1 failed)',
                 fontsize=12, color=fg.COL['red'])
    fg.caption(fig, 'Baseline is the M1 discrete null 0.981, never 1. Only interpretable beside '
                    'Figure 2b, which shows the residual tracks neither grad⁴b nor KPPhbl with '
                    '|r| > 0.22 — so no slope here may be called diabatic damping.')
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out['png'] = fg.save(fig, 'fig02_joint_pdf.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 2b -- the discriminator
# ---------------------------------------------------------------------------
def fig02b_discriminator(data_dir=None, fig_dir=None, summary=None, Ls=fg.L_ALL):
    """**The discriminator.**  The residual against ``lap2_b`` (the
    ``grad^4``-like diagnostic, i.e. implicit numerical diffusion) and
    against ``KPPhbl`` (air-sea forcing), as binned means with
    block-bootstrap bands, and against local solar hour; with the partial
    correlations task 6 computed.

    Planning §12: if the residual tracks ``grad^4 b`` rather than ``KPPhbl``
    or the diurnal cycle, no diabatic signal can be isolated.  Here
    **neither correlation exceeds 0.22 at any ``L``**, and the two cross
    over between ``L = 2`` and ``L = 4``.
    """
    s = fg.load_summary(summary)
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.2))
    out = {}
    for L in Ls:
        d = fg.front_pool(L, ('residual', 'lap2_b', 'KPPhbl'), data_dir)
        blk = None                       # block labels are not pooled here; se_iid is labelled
        for ax, key, xlabel in ((axs[0], 'lap2_b', 'lap²b  (grad⁴-like)  [s$^{-2}$ m$^{-4}$]'),
                                (axs[1], 'KPPhbl', 'KPPhbl  [m]')):
            df = fg.binned(d[key], d['residual'], bins=12, labels=blk)
            ax.plot(df['x_mean'], df['y_mean'], '-o', ms=4, label=f'L = {L}')
            ax.fill_between(df['x_mean'], df['y_mean'] - df['se_iid'],
                            df['y_mean'] + df['se_iid'], alpha=0.15)
            ax.set_xlabel(xlabel)
        hrs = np.floor(d['lst']).astype(int) % 24
        hh = np.unique(hrs)
        axs[2].plot(hh, [np.nanmean(d['residual'][hrs == h]) for h in hh], '-o', ms=4,
                    label=f'L = {L}')
        f = s['per_L'][str(L)]['fig2b']
        out[f'L{L}'] = dict(partial_lap=f['partial_res_lap_given_kpp'],
                            partial_kpp=f['partial_res_kpp_given_lap'],
                            corr_lap=f['on_lap2_b']['corr'], corr_kpp=f['on_KPPhbl']['corr'])
    for ax, t in ((axs[0], 'residual vs lap²b — numerical diffusion?'),
                  (axs[1], 'residual vs KPPhbl — air-sea forcing?'),
                  (axs[2], 'residual vs local solar hour')):
        ax.axhline(0, color='0.7', lw=0.8)
        ax.set_ylabel('mean residual  [s$^{-5}$]')
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8)
    axs[2].set_xlabel('local solar hour (UTC − 8 h)')
    ax = axs[3]
    x = np.arange(len(Ls))
    ax.bar(x - 0.2, [out[f'L{L}']['partial_lap'] for L in Ls], 0.4, color=fg.COL['two_F'],
           label='partial corr(res, lap²b | KPPhbl)')
    ax.bar(x + 0.2, [out[f'L{L}']['partial_kpp'] for L in Ls], 0.4, color=fg.COL['surface_flux'],
           label='partial corr(res, KPPhbl | lap²b)')
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_xticks(x, [f'L = {L}' for L in Ls])
    ax.set_ylim(0, 0.32)
    ax.set_title('the discriminator: neither exceeds 0.22', fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle('Figure 2b — what does the residual track?  |  front & valid, 71 pairs pooled',
                 fontsize=12)
    fg.caption(fig, 'Planning §12: the residual tracks grad⁴b more than KPPhbl at L = 0-2 and '
                    'the reverse at L = 8, but NEITHER partial correlation exceeds 0.22 at any '
                    'L. No diabatic signal can be isolated, so no slope in Figure 2 is called '
                    'damping. ' + fg.SAMPLING_CAVEAT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out['png'] = fg.save(fig, 'fig02b_discriminator.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 3 -- slope and correlation vs L
# ---------------------------------------------------------------------------
def fig03_slope_vs_L(data_dir=None, fig_dir=None, summary=None, Ls=fg.L_ALL):
    """Slope and correlation against ``L`` — both forms (discrete, chain),
    both orders (3, 5), both masks (``edge_cells`` 7 and 13) — with the
    baseline and the two systematic bands drawn; plus
    ``rms(subfilter)/rms(2F)`` against ``L``.

    **No ``L = 1`` column.**  M3-Q3 allowed one if task 4's pilot had time to
    spare, and it did — but ``operators.lowpass`` refuses an odd ``L`` (the
    top-hat half-width ``L/2`` must be an integer), so ``L = 1`` would be a
    different filter, not an extra column (task 4).
    """
    s = fg.load_summary(summary)
    fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.6))
    out = {}
    def node(L):
        n = s['per_L'][str(L)]
        return n.get('slopes') or n.get('slopes_not_quoted')
    series = (('discrete, order 3 (primary)', lambda L: node(L)['primary']['estimates']['ols'],
               fg.COL['two_F'], '-o'),
              ('trimmed (top 1 % |2F|)', lambda L: node(L)['primary']['estimates']['trimmed'],
               fg.COL['subfilter'], '-s'),
              ('chain form', lambda L: node(L)['variants']['chain'], fg.COL['vertical'], '--^'),
              ('order 5', lambda L: node(L)['variants']['order5'], fg.COL['day3'], '--v'),
              ('edge_cells = 13', lambda L: node(L)['variants']['edge13'], fg.COL['grey'], ':d'))
    for name, get, col, style in series:
        v = [get(L)['value'] for L in Ls]
        lo = [get(L).get('lo', np.nan) for L in Ls]
        hi = [get(L).get('hi', np.nan) for L in Ls]
        axs[0].errorbar(Ls, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt=style,
                        color=col, ms=5, capsize=3, lw=1.4, label=name)
        out[name] = dict(zip([str(L) for L in Ls], v))
    _baselines(axs[0])
    axs[0].set_xlabel('L (cells)')
    axs[0].set_ylabel('OLS slope of measured on 2F')
    axs[0].set_title('slope vs L  |  front & valid, p90, 71 pairs\nNOT an efficiency — '
                     'criterion 1 failed at every L', fontsize=9, color=fg.COL['red'])
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].plot(Ls, [s['per_L'][str(L)]['euler']['front']['corr'] for L in Ls], '-o',
                color=fg.COL['measured'], label='corr(Eulerian, semi-Lagrangian)')
    axs[1].plot(Ls, [s['per_L'][str(L)]['euler']['front']['ols'] for L in Ls], '-s',
                color=fg.COL['two_F'], label='slope(Eulerian on semi-Lagrangian)')
    axs[1].axhspan(0.85, 1.15, color='#2a9d8f', alpha=0.12, label='M3-Q2 slope tolerance')
    axs[1].axhline(0.90, color='#2a9d8f', ls=':', label='M3-Q2 corr ≥ 0.90')
    axs[1].axvline(2, color='0.6', ls='--', lw=1)
    axs[1].set_xlabel('L (cells)')
    axs[1].set_title('criterion 2 — gated from L = 2, passes at L ≥ 4', fontsize=9)
    axs[1].legend(fontsize=7)
    sub = [s['per_L'][str(L)]['subfilter']['over_two_F'] for L in Ls]
    cor = [s['per_L'][str(L)]['subfilter']['corr_with_2F'] for L in Ls]
    axs[2].plot(Ls, sub, '-o', color=fg.COL['subfilter'], label='rms(subfilter)/rms(2F)')
    axs[2].plot(Ls, cor, '-s', color='#8c564b', label='corr(subfilter, 2F)')
    axs[2].plot([2, 4, 8], [0.31, 0.50, 0.70], 'k^--', ms=5, label='M1 task 4, hour 0')
    axs[2].plot([2, 4, 8], [-0.66, -0.60, -0.54], 'kv--', ms=5)
    axs[2].axhline(0, color='0.8', lw=0.8)
    axs[2].set_xlabel('L (cells)')
    axs[2].set_title('criterion 3 — the subfilter term grows with L', fontsize=9)
    axs[2].legend(fontsize=7)
    out['subfilter_over_2F'] = dict(zip([str(L) for L in Ls], sub))
    fig.suptitle('Figure 3 — slope, Eulerian agreement and the subfilter term against L',
                 fontsize=12)
    fg.caption(fig, 'No L = 1 column: operators.lowpass refuses an odd L (half-width L/2 must be '
                    'an integer), so it would be a different filter rather than an extra column '
                    '(task 4; M3-Q3). Baseline 0.981, never 1.')
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out['png'] = fg.save(fig, 'fig03_slope_vs_L.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 4 -- the alignment PDF
# ---------------------------------------------------------------------------
def fig04_alignment(data_dir=None, fig_dir=None, Ls=fg.L_ALL, bins=60):
    """PDF of ``theta_align`` on front pixels — the angle between ``grad b``
    and the **compressional** axis of the strain, folded to ``[0, pi/2]``.
    With this ``theta``, ``F = -½ delta G + ½ |sigma| G cos 2theta``, so
    ``theta = 0`` is maximal frontogenesis (sign final, M1-Q3; planning §2.4
    writes a minus, which holds only if ``theta`` is measured from the
    *extensional* axis)."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    out = {}
    for L in Ls:
        d = fg.front_pool(L, ('theta_align', 'two_F'), data_dir)
        th = d['theta_align'][np.isfinite(d['theta_align'])]
        axs[0].hist(np.rad2deg(th), bins=bins, range=(0, 90), density=True, histtype='step',
                    lw=1.8, label=f'L = {L} (n {th.size:,})')
        t, F2 = d['theta_align'], d['two_F']
        ok = np.isfinite(t) & np.isfinite(F2)
        edges = np.linspace(0, np.pi / 2, 19)
        k = np.clip(np.digitize(t[ok], edges[1:-1]), 0, len(edges) - 2)
        mids = np.rad2deg(0.5 * (edges[:-1] + edges[1:]))
        axs[1].plot(mids, [np.nanmean(F2[ok][k == i]) for i in range(len(mids))], '-o', ms=3,
                    label=f'L = {L}')
        out[f'L{L}'] = dict(n=int(th.size), median_deg=float(np.rad2deg(np.median(th))),
                            frac_under_45=float(np.mean(th < np.pi / 4)))
    axs[0].axvline(45, color='0.6', ls=':', label='isotropic median')
    axs[0].set_xlabel(r'$\theta$ to the compressional axis  [deg]')
    axs[0].set_ylabel('probability density')
    axs[0].set_title('alignment PDF  |  front & valid, 71 pairs', fontsize=10)
    axs[0].legend(fontsize=8)
    axs[1].axhline(0, color='0.7', lw=0.8)
    axs[1].set_xlabel(r'$\theta$  [deg]')
    axs[1].set_ylabel('mean 2F  [s$^{-5}$]')
    axs[1].set_title(r'2F against alignment: $F = -\frac{1}{2}\delta G + '
                     r'\frac{1}{2}|\sigma| G \cos 2\theta$', fontsize=10)
    axs[1].legend(fontsize=8)
    fig.suptitle('Figure 4 — gradient alignment with the compressional strain axis', fontsize=12)
    fg.caption(fig, 'theta folded to [0, pi/2]; theta = 0 is maximal frontogenesis with this '
                    'sign convention (M1-Q3, final). Mask: front & valid at each L.')
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out['png'] = fg.save(fig, 'fig04_alignment.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 6 -- the diurnal composite
# ---------------------------------------------------------------------------
def fig06_diurnal(data_dir=None, fig_dir=None, summary=None, Ls=(0, 4)):
    """``residual``, ``vertical`` and ``surface_flux`` composited by **local
    solar hour** (UTC − 8.0 h), with ``KPPhbl`` overlaid and the mixed-layer
    **minimum** (~13 h solar) marked.  The **M3-Q13 (c) sensitivity** —
    ``surface_flux_kpp``, the same term with the flux spread over ``KPPhbl``
    rather than the 1 m top cell — is drawn beside the primary.

    The caption carries the three required caveats verbatim, and the
    residual is expected to **change sign** over the cycle, which by itself
    refutes a single-signed "damping" reading.
    """
    fig, axs = plt.subplots(1, len(Ls), figsize=(9.0 * len(Ls), 5.8), squeeze=False)
    out = {}
    for c, L in enumerate(Ls):
        ax = axs[0, c]
        keys = ('residual', 'vertical', 'surface_flux', 'surface_flux_kpp', 'KPPhbl')
        d = fg.front_pool(L, keys, data_dir)
        hrs = np.floor(d['lst']).astype(int) % 24
        hh = np.unique(hrs)
        for k in ('residual', 'vertical', 'surface_flux', 'surface_flux_kpp'):
            if not np.isfinite(d[k]).any():
                continue
            m = [np.nanmean(d[k][hrs == h]) for h in hh]
            ls = '--' if k == 'surface_flux_kpp' else '-'
            lab = (k + '  (M3-Q13 (c) sensitivity: flux over KPPhbl)'
                   if k == 'surface_flux_kpp' else k)
            ax.plot(hh, m, ls + 'o', color=fg.COL[k], ms=4, lw=1.6, label=lab)
            out[f'L{L}_{k}'] = [float(v) for v in m]
        ax.axhline(0, color='0.7', lw=0.8)
        ax.axvline(13, color=fg.COL['red'], ls=':', lw=1.6, label='mixed-layer minimum ~13 LST')
        ax.set_xlabel('local solar hour (UTC − 8.0 h)')
        ax.set_ylabel('mean over front pixels  [s$^{-5}$]')
        ax.set_title(fg.mask_note(L), fontsize=9)
        ax2 = ax.twinx()
        kp = [np.nanmean(d['KPPhbl'][hrs == h]) for h in hh]
        ax2.plot(hh, kp, '-', color=fg.COL['kpp'], lw=2.4, alpha=0.45)
        ax2.set_ylabel('KPPhbl [m]', color=fg.COL['kpp'])
        ax2.invert_yaxis()
        out[f'L{L}_KPPhbl'] = [float(v) for v in kp]
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Figure 6 — the diurnal composite  |  residual, vertical and surface-flux terms '
                 'by local solar hour', fontsize=12)
    fg.caption(fig, fg.FORCING_CAVEAT + ' ' + fg.WIND_CAVEAT + ' ' + fg.WINDOW_CAVEAT
               + ' The residual changes sign over the cycle, which by itself refutes a '
                 'single-signed "damping" reading. ' + fg.SAMPLING_CAVEAT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out['png'] = fg.save(fig, 'fig06_diurnal.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 7 -- offshore
# ---------------------------------------------------------------------------
def fig07_offshore(data_dir=None, fig_dir=None, summary=None, Ls=fg.L_ALL):
    """Statistics against distance offshore (``coast_distance_km`` bins) with
    the ``>= 100 km`` cut marked: the residual fraction, the slope and ``n``.
    Planning §5.6 asks that what the cut excluded stays visible — here it
    excluded **nothing**, because ``mask_analysis`` already imposes the
    100 km offshore cut, so the pool starts at the line."""
    s = fg.load_summary(summary)
    fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.2))
    out = {}
    for L in Ls:
        by = s['per_L'][str(L)]['composites']['by_coast_km']
        labels = list(by)
        mids = [float(k.split('-')[0]) + 25 for k in labels]
        axs[0].plot(mids, [by[k]['resid_over_meas'] for k in labels], '-o', ms=4, label=f'L = {L}')
        axs[1].plot(mids, [by[k]['meas_on_2F']['ols'] for k in labels], '-o', ms=4,
                    label=f'L = {L}')
        axs[2].plot(mids, [by[k]['n'] for k in labels], '-o', ms=4, label=f'L = {L}')
        out[f'L{L}'] = {k: by[k]['resid_over_meas'] for k in labels}
    for ax in axs:
        ax.axvline(100, color=fg.COL['red'], ls='--', lw=1.4)
        ax.set_xlabel('distance from the coast [km]')
        ax.legend(fontsize=8)
    axs[0].axhline(0.5, color=fg.COL['red'], ls=':', lw=1.6)
    axs[0].set_ylabel('rms(residual) / rms(measured)')
    axs[0].set_title('residual fraction (M3-Q1 tolerance 0.5 dotted)', fontsize=10)
    _baselines(axs[1])
    axs[1].set_ylabel('OLS slope of measured on 2F')
    axs[1].set_title('slope — NOT quoted as an efficiency', fontsize=10, color=fg.COL['red'])
    axs[2].set_yscale('log')
    axs[2].set_ylabel('front pixels')
    axs[2].set_title('sample size', fontsize=10)
    fig.suptitle('Figure 7 — statistics against distance offshore  |  front & valid, 71 pairs',
                 fontsize=12)
    fg.caption(fig, 'The >= 100 km cut (red dashed) is already inside mask_analysis, so the '
                    'offshore restriction excluded nothing here and the pool begins at the line '
                    '(planning §5.6). The residual fraction is nearly flat with distance.')
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out['png'] = fg.save(fig, 'fig07_offshore.png', fig_dir=fig_dir)
    return out
