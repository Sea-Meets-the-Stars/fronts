""" M3 task 7, the map family: Figures 1, 3b, 5 and 10.

Split out of :mod:`figures` for coding §1.3's line cap; import them from
there.  Every map is ``pcolormesh(XC, YC, ...)`` so the rotated face 10
comes out north-up and east-right, land is grey, and every title states its
``L`` and its mask.  No physics is computed here.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import figures as fg


def _xy(h):
    return np.asarray(h['XC'].values), np.asarray(h['YC'].values)


def _land(h):
    """Land is where the budget fields are NaN everywhere -- the stores
    carry no ``hFacC``, and ``b`` is NaN exactly on land (plus the filter
    rim, which is the honest thing to grey out too)."""
    return ~np.isfinite(np.asarray(h['b'].values))


# ---------------------------------------------------------------------------
# Figure 1 -- do the measured and predicted tendencies look alike?
# ---------------------------------------------------------------------------
def fig01_maps(data_dir=None, fig_dir=None, hours=(fg.HOUR_DAY, fg.HOUR_NIGHT), Ls=(0, 4)):
    """``DGDt_semilag`` beside ``two_F`` and ``residual`` on a **shared
    diverging scale**, at ``L = 0`` and ``L = 4``, for a daytime hour
    (07-03 21 UTC = 13 LST, the mixed-layer minimum) and a night-time one
    (07-03 09 UTC = 01 LST).

    *If these do not look alike, nothing else matters* -- and task 6's
    verdict says they do not: the residual map carries as much structure as
    either of the first two.
    """
    rows = [(L, fg.resolve_pair(L, p, data_dir)) for L in Ls for p in hours]
    fig, axs = plt.subplots(len(rows), 3, figsize=(16.5, 4.3 * len(rows)))
    out = {}
    for r, (L, p) in enumerate(rows):
        h = fg.hour(L, p, data_dir)
        X, Y, land = *_xy(h), _land(h)
        meas = np.asarray(h['DGDt_semilag'].values)
        F2 = np.asarray(h['two_F'].values)
        res = np.asarray(h['residual'].values)
        v = fg.sym_limit(meas, F2, res, pct=99.0)
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        lst = fg._lst(h)
        for c, (name, A) in enumerate((('measured  DGDt_semilag', meas), ('2F', F2),
                                       ('residual', res))):
            ax = axs[r, c]
            m = fg.draw_map(ax, X, Y, A, norm=norm, land=land)
            ax.set_title(f'{name}\nL = {L}, pair {p} ({lst:.1f} LST)', fontsize=9)
            if c == 2:
                fig.colorbar(m, ax=axs[r, :], fraction=0.02, pad=0.01,
                             label='s$^{-5}$ (shared scale per row)')
        out[f'L{L}_pair{p}'] = dict(lst=lst, scale=v,
                                    rms={k: float(np.sqrt(np.nanmean(a ** 2)))
                                         for k, a in (('measured', meas), ('two_F', F2),
                                                      ('residual', res))})
    fig.suptitle('Figure 1 — measured, predicted and residual tendency on one hour  |  '
                 'shared diverging scale per row', fontsize=12)
    fg.caption(fig, 'Mask: all finite cells shown; land and the filter rim grey. '
                    'Task 6: the residual is 0.81-1.04 of measured in rms on front pixels, which '
                    'is why the third column is not blank. ' + fg.SAMPLING_CAVEAT)
    out['png'] = fg.save(fig, 'fig01_maps.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 3b -- the filter ladder
# ---------------------------------------------------------------------------
def fig03b_filter_ladder(data_dir=None, fig_dir=None, pair=fg.HOUR_DAY, Ls=fg.L_ALL):
    """Rows ``{b, G, 2F, tau-term}`` x columns ``{L = 0, 2, 4, 8}`` on one
    hour.  **The ``L = 0`` ``tau`` panel is identically zero and is labelled
    so** (planning §5.4, corrected): at ``L = 0`` ``lowpass`` is the identity,
    so ``mean(ub) - ubar bbar`` vanishes exactly -- that column is ``2F`` vs
    measured with the whole numerical term in the residual, not a limit of
    ``tau``."""
    pair = fg.resolve_pair(Ls[0], pair, data_dir)
    rows = ('b', 'G', 'two_F', 'subfilter')
    titles = ('b  (filtered)', 'G = |grad b|²', '2F', 'subfilter term (2 x tau)')
    fig, axs = plt.subplots(len(rows), len(Ls), figsize=(4.2 * len(Ls), 3.8 * len(rows)))
    out = {}
    fields = {}
    for c, L in enumerate(Ls):
        h = fg.hour(L, pair, data_dir)
        fields[L] = {k: np.asarray(h[k].values) for k in rows}
        fields[L]['X'], fields[L]['Y'], fields[L]['land'] = *_xy(h), _land(h)
    for r, (k, t) in enumerate(zip(rows, titles)):
        pool = [fields[L][k] for L in Ls]
        div = k in ('two_F', 'subfilter')
        v = fg.sym_limit(*pool, pct=99.0)
        for c, L in enumerate(Ls):
            ax = axs[r, c]
            A = fields[L][k]
            kw = dict(norm=TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)) if div else {}
            m = fg.draw_map(ax, fields[L]['X'], fields[L]['Y'], A,
                            cmap=fg.DIVERGING if div else 'viridis', land=fields[L]['land'], **kw)
            ax.set_title(f'{t}   L = {L}', fontsize=9)
            if k == 'subfilter' and L == 0:
                ax.text(0.5, 0.5, 'IDENTICALLY ZERO\nat L = 0\n(lowpass is the identity)',
                        transform=ax.transAxes, ha='center', va='center', fontsize=11,
                        color=fg.COL['red'], weight='bold',
                        bbox=dict(fc='white', alpha=0.85, ec=fg.COL['red']))
            if c == len(Ls) - 1:
                fig.colorbar(m, ax=axs[r, :], fraction=0.015, pad=0.01)
            out[f'{k}_L{L}'] = float(np.sqrt(np.nanmean(A ** 2)))
    fig.suptitle(f'Figure 3b — the filter ladder, pair {pair}  |  mask: all finite cells',
                 fontsize=12)
    fg.caption(fig, 'The L = 0 subfilter panel is exactly zero by construction, not small: '
                    'planning §5.4 (corrected). Task 6: rms(subfilter)/rms(2F) = 0 / 0.353 / '
                    '0.552 / 0.748 at L = 0 / 2 / 4 / 8 — the term GROWS with L.')
    out['png'] = fg.save(fig, 'fig03b_filter_ladder.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 5 -- the sharpening timescale
# ---------------------------------------------------------------------------
def fig05_sharpening_time(data_dir=None, fig_dir=None, pair=fg.HOUR_DAY, Ls=(0, 4)):
    """``t_sharp = G / (2F)`` [h] with the ``dt = 1 h`` contour marked: where
    the front would double its ``G`` in under an hour, the hourly sampling
    cannot follow it -- which is task 6's limiting factor drawn as a map.
    Only the frontogenetic side (``2F > 0``) has a timescale; ``2F < 0`` is
    greyed."""
    pair = fg.resolve_pair(Ls[0], pair, data_dir)
    fig, axs = plt.subplots(1, len(Ls) + 1, figsize=(6.0 * (len(Ls) + 1), 5.4))
    out = {}
    for c, L in enumerate(Ls):
        h = fg.hour(L, pair, data_dir)
        X, Y, land = *_xy(h), _land(h)
        G = np.asarray(h['G'].values)
        F2 = np.asarray(h['two_F'].values)
        with np.errstate(invalid='ignore', divide='ignore'):
            t_sharp = np.where(F2 > 0, G / F2 / 3600.0, np.nan)      # hours
        m = fg.draw_map(axs[c], X, Y, np.log10(t_sharp), cmap='magma_r', land=land,
                        vmin=-1, vmax=2)
        axs[c].contour(X, Y, np.log10(t_sharp), levels=[0.0], colors='cyan', linewidths=1.6)
        axs[c].set_title(f'log10 t_sharp = G / 2F  [h]   L = {L}, pair {pair}\n'
                         'cyan = the dt = 1 h contour', fontsize=9)
        fig.colorbar(m, ax=axs[c], fraction=0.035, label='log10 hours')
        fr = np.asarray(h['front'].values, bool)
        vals = t_sharp[fr & np.isfinite(t_sharp)]
        out[f'L{L}'] = dict(median_h=float(np.median(vals)) if vals.size else float('nan'),
                            frac_under_1h=float(np.mean(vals < 1.0)) if vals.size else float('nan'),
                            n=int(vals.size))
    ax = axs[-1]
    for L in Ls:
        h = fg.hour(L, pair, data_dir)
        G, F2 = np.asarray(h['G'].values), np.asarray(h['two_F'].values)
        fr = np.asarray(h['front'].values, bool)
        with np.errstate(invalid='ignore', divide='ignore'):
            t = np.where(F2 > 0, G / F2 / 3600.0, np.nan)[fr]
        t = t[np.isfinite(t) & (t > 0)]
        ax.hist(np.log10(t), bins=60, histtype='step', lw=1.8, label=f'L = {L} (n {t.size:,})')
    ax.axvline(0.0, color='cyan', lw=2)
    ax.text(0.02, 0.98, 'dt = 1 h', transform=ax.transAxes, color='teal', va='top')
    ax.set_xlabel('log10 t_sharp [h]')
    ax.set_ylabel('front pixels')
    ax.set_title('front pixels only (p90 of G at the midpoint)', fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle('Figure 5 — sharpening timescale and the one-hour sampling limit', fontsize=12)
    fg.caption(fig, 'Mask: maps show all finite cells; the histogram is front & valid. '
                    'A front with t_sharp below 1 h changes faster than the output cadence. '
                    + fg.SAMPLING_CAVEAT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out['png'] = fg.save(fig, 'fig05_sharpening_time.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 10 -- the term budget
# ---------------------------------------------------------------------------
def fig10_term_budget(data_dir=None, fig_dir=None, summary=None, pair=fg.HOUR_DAY, L_map=0,
                      Ls=fg.L_ALL):
    """The five terms side by side: maps for one hour, and rms bars per
    ``L`` with the **catch-all** residual beside the real one.  With the
    measured vertical and surface-flux terms this is a real budget rather
    than a two-term comparison — and the bars are where task 6's finding
    shows: the catch-all is *smaller* than the residual at every ``L``,
    because ``surface_flux`` is near-orthogonal to the imbalance."""
    s = fg.load_summary(summary)
    terms = ('two_F', 'subfilter', 'vertical', 'surface_flux')
    pair = fg.resolve_pair(L_map, pair, data_dir)
    h = fg.hour(L_map, pair, data_dir)
    X, Y, land = *_xy(h), _land(h)
    arrs = {k: np.asarray(h[k].values) for k in terms + ('residual', 'DGDt_semilag')}
    v = fg.sym_limit(*arrs.values(), pct=99.0)
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    fig = plt.figure(figsize=(19, 9.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
    out = {}
    for c, k in enumerate(terms + ('residual', 'DGDt_semilag')):
        ax = fig.add_subplot(gs[0, c])
        m = fg.draw_map(ax, X, Y, arrs[k], norm=norm, land=land)
        ax.set_title(f'{k}\nL = {L_map}, pair {pair}', fontsize=9)
        ax.set_xlabel('')
        if c:
            ax.set_ylabel('')
    fig.colorbar(m, ax=fig.axes[:6], fraction=0.012, pad=0.01, label='s$^{-5}$')
    ax = fig.add_subplot(gs[1, :3])
    w, x = 0.17, np.arange(len(Ls))
    for k, t in enumerate(terms):
        ax.bar(x + (k - 1.5) * w, [s['per_L'][str(L)]['closure']['terms'][t]['over_measured']
                                   for L in Ls], w, label=t, color=fg.COL[t])
    ax.plot(x, [s['per_L'][str(L)]['closure']['residual']['over_measured'] for L in Ls],
            '-o', color='black', label='residual', zorder=5)
    ax.plot(x, [s['per_L'][str(L)]['closure']['catchall']['over_measured'] for L in Ls],
            '--s', color=fg.COL['catchall'], label='catch-all (no chunk terms)', zorder=5)
    ax.axhline(0.5, color=fg.COL['red'], ls=':', lw=2, label='M3-Q1 tolerance 0.5')
    ax.set_xticks(x, [f'L = {L}' for L in Ls])
    ax.set_ylabel('rms / rms(measured)')
    ax.set_title('rms per L, pooled over 71 pairs  |  front & valid', fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax2 = fig.add_subplot(gs[1, 3:])
    for k in terms:
        ax2.plot(Ls, [s['per_L'][str(L)]['closure']['terms'][k]['corr_with_2F'] for L in Ls],
                 '-o', color=fg.COL[k], label=f'corr({k}, 2F)')
    ax2.axhline(0, color='0.7', lw=0.8)
    ax2.set_xlabel('L (cells)')
    ax2.set_ylabel('correlation with 2F')
    ax2.set_title('the subfilter term is anti-correlated with 2F; the flux term is not '
                  'correlated with anything', fontsize=9)
    ax2.legend(fontsize=8)
    out['rms'] = {str(L): {t: s['per_L'][str(L)]['closure']['terms'][t]['over_measured']
                           for t in terms} for L in Ls}
    out['residual'] = {str(L): s['per_L'][str(L)]['closure']['residual']['over_measured']
                       for L in Ls}
    out['catchall'] = {str(L): s['per_L'][str(L)]['closure']['catchall']['over_measured']
                       for L in Ls}
    fig.suptitle('Figure 10 — the five-term budget  |  maps at one hour, rms over all 71 pairs',
                 fontsize=12)
    fg.caption(fig, 'The catch-all (residual + vertical + surface_flux, i.e. no chunk store) is '
                    'SMALLER than the residual at every L: the measured surface-flux term '
                    'carries 0.36-0.80 of the measured amplitude but correlates +0.01 to +0.02 '
                    'with the imbalance, so subtracting it at unit weight adds its variance. '
                    'Task 6, M3-Q13.')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out['png'] = fg.save(fig, 'fig10_term_budget.png', fig_dir=fig_dir)
    return out
