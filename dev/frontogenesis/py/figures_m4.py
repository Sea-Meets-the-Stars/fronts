""" M4 figures: the tracking-quality diagnostic (task 4); Figures 8 and 9 follow in task 5.

Same style and conventions as :mod:`figures` (200 dpi, ``figs/``, a caption on the page, the
numbers drawn returned to the caller so the summary JSON and the PNG cannot disagree).
Nothing here computes anything: the rows come from ``tracking.tracking_quality`` as the
runner wrote them.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import pandas as pd                                            # noqa: E402

import figures as fg                                           # noqa: E402

COL_FLOW, COL_PLAIN = fg.COL['two_F'], fg.COL['grey']          # two series, fixed, never cycled
SEQ = 'Blues'


def fig_tracking_quality(q_flow: pd.DataFrame, q_plain: pd.DataFrame, fig_dir=None,
                         name='fig_tq_tracking_quality', radius=2.0) -> dict:
    """Four panels.  (a) the prompt's quantity: |chosen − flow-predicted| centroid
    displacement per link, flow-informed and plain walks; (b) the mask-based distance (median
    over the chosen label's pixels to the nearest pixel of the advected previous mask);
    (c) one strip per flow-informed track over the 71 steps: the per-link p90 mask distance
    as a sequential shade, splits (▲) and merges (▼) marked; (d) the dilated IoU of prediction
    and chosen label.  Counts on log axes in (a) and (b): both distributions are a spike at
    zero with a long tail.  Returns the numbers drawn."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    (a, b), (c, d) = axes
    edges = np.concatenate([np.arange(0, 10.5, 0.5), [15, 20, 30, 60]])
    out = {}
    for ax, col, label, cap in ((a, 'diff_cells', '|chosen − flow-predicted| centroid shift (cells)', 20),
                                (b, 'mask_dist_p90', 'p90 distance, chosen pixels → prediction (cells)', 20)):
        for q, colr, nm in ((q_plain, COL_PLAIN, 'plain follow()'), (q_flow, COL_FLOW, 'flow-informed')):
            x = np.clip(q[col].to_numpy(dtype=float), 0, cap)
            ax.hist(x, bins=np.linspace(0, cap, 41), histtype='step', lw=1.8, color=colr,
                    label=f'{nm}: median {np.nanmedian(q[col]):.2f}, p90 {np.nanpercentile(q[col], 90):.1f}, '
                          f'{np.mean(q[col] <= 1.0):.0%} ≤ 1 cell')
            out[f'{nm}_{col}_median'] = float(np.nanmedian(q[col]))
            out[f'{nm}_{col}_p90'] = float(np.nanpercentile(q[col], 90))
            out[f'{nm}_{col}_frac_le_1'] = float(np.mean(q[col] <= 1.0))
        ax.set_yscale('log')
        ax.axvline(radius, color='k', lw=0.8, ls='--')
        ax.text(radius, ax.get_ylim()[1] * 0.8, f' search radius {radius:g}', va='top', fontsize=8)
        ax.set_xlabel(label + f'; last bin ≥ {cap}')
        ax.set_ylabel('links (log)')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
    a.set_title('(a) the prompt\'s quantity, per link: centroid to centroid', fontsize=10, loc='left')
    b.set_title('(b) mask to mask, per link', fontsize=10, loc='left')
    # (c) the strip
    tracks = sorted(q_flow['track'].unique())
    n_steps = int(max(q_flow['step'].max(), q_plain['step'].max())) + 1
    M = np.full((len(tracks), n_steps), np.nan)
    for i, t in enumerate(tracks):
        g = q_flow[q_flow['track'] == t]
        M[i, g['step'].to_numpy()] = g['mask_dist_p90'].to_numpy()
    im = c.imshow(np.clip(M, 0, 5), aspect='auto', cmap=SEQ, vmin=0, vmax=5, interpolation='nearest')
    for i, t in enumerate(tracks):
        g = q_flow[q_flow['track'] == t]
        c.plot(g.loc[g['split'], 'step'], [i - 0.3] * int(g['split'].sum()), marker='^', ls='none',
               color='k', ms=4)
        c.plot(g.loc[g['merge'], 'step'], [i + 0.3] * int(g['merge'].sum()), marker='v', ls='none',
               color=fg.COL['red'], ms=4)
    c.set_yticks(range(len(tracks)))
    c.set_yticklabels([f'{t}: {int(q_flow[q_flow["track"] == t]["anchor_label"].iloc[0])}' for t in tracks],
                      fontsize=7)
    c.set_xlabel('pair (step); anchor at the middle')
    c.set_ylabel('track: anchor label')
    c.set_title('(c) flow-informed tracks: p90 mask distance per link; ▲ split  ▼ merge', fontsize=10, loc='left')
    cb = fig.colorbar(im, ax=c, fraction=0.03, pad=0.02)
    cb.set_label('cells (clipped at 5)', fontsize=8)
    # (d) IoU
    for q, colr, nm in ((q_plain, COL_PLAIN, 'plain follow()'), (q_flow, COL_FLOW, 'flow-informed')):
        d.hist(q['flow_iou'], bins=np.linspace(0, 1, 41), histtype='step', lw=1.8, color=colr,
               label=f'{nm}: median {q["flow_iou"].median():.2f}')
        out[f'{nm}_flow_iou_median'] = float(q['flow_iou'].median())
    d.set_xlabel('dilated IoU(advected previous mask, chosen label)')
    d.set_ylabel('links')
    d.legend(fontsize=8, frameon=False, loc='upper left')
    d.set_title('(d) overlap with the prediction', fontsize=10, loc='left')
    ns, nm_ = int(q_flow['split'].sum()), int(q_flow['merge'].sum())
    out.update(n_split_flow=ns, n_merge_flow=nm_, n_links_flow=int(len(q_flow)),
               n_links_plain=int(len(q_plain)))
    fg.caption(fig, (f'Tracking quality (prompt 5 task 4, criterion 3). {len(q_flow)} flow-informed and '
                     f'{len(q_plain)} plain links of the 10 largest fronts at pair 35, L = 0, mask_analysis, '
                     f'config D. The prediction is the previous found mask advected by the model flow '
                     f'(tile330_flow.zarr). Flow-informed: {ns} splits, {nm_} merges flagged '
                     f'(>= 5 overlapping pixels per label). The centroid shift (a) has a tail that is topology, '
                     f'not motion; (b), the 90th-percentile distance of the chosen pixels from the prediction, is the '
                     f'measure of following the fluid.'), fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out['path'] = fg.save(fig, name, fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 8 -- one tracked front: the time sequence and its strength
# ---------------------------------------------------------------------------
COL_LABEL, COL_MATCHED, COL_PRED = fg.COL['red'], 'black', fg.COL['two_F']
COL_STORE, COL_BUDGET = fg.COL['measured'], fg.COL['surface_flux']


def _crop(track, labels_by_time, pad=15):
    keys = list(labels_by_time)
    jj, ii = [], []
    for s in track.steps():
        j, i = np.nonzero(labels_by_time[keys[s]] == track.labels[s])
        jj += [j.min(), j.max()]
        ii += [i.min(), i.max()]
    nj, ni = labels_by_time[keys[0]].shape
    return (slice(max(min(jj) - pad, 0), min(max(jj) + pad + 1, nj)),
            slice(max(min(ii) - pad, 0), min(max(ii) + pad + 1, ni)))


def fig08_front_case(track, series: pd.DataFrame, labels_by_time, fields, *, steps=None, fig_dir=None,
                     name='fig08_front_case', band=1) -> dict:
    """Five map panels of log10 G in a fixed crop with the tracked label (dilated by ``band``)
    outlined, then four panels along the track: the strength curves (the label mean, the
    material curve from the matched rate, the 2F-predicted curve, the Phase-2 and the
    four-term budget curves), the hourly rates, the cross-front buoyancy step and the width.
    Returns the numbers drawn."""
    from scipy import ndimage
    keys = list(labels_by_time)
    st = track.steps()
    if steps is None:
        steps = [st[0], st[len(st) // 4], track.anchor.step, st[3 * len(st) // 4], st[-1]]
    sj, si = _crop(track, labels_by_time)
    ds = fields.ds
    if 'XC' in ds.coords or 'XC' in ds.data_vars:
        X, Y = np.asarray(ds['XC'].values)[sj, si], np.asarray(ds['YC'].values)[sj, si]
    else:                                                    # a synthetic store: index space
        J, I = np.mgrid[sj, si]
        X, Y = I.astype('float64'), J.astype('float64')
    fig = plt.figure(figsize=(17, 9.0))
    gs_top = fig.add_gridspec(1, 5, left=0.04, right=0.93, top=0.90, bottom=0.56, wspace=0.18)
    gs = fig.add_gridspec(1, 5, left=0.05, right=0.99, top=0.44, bottom=0.10, wspace=0.55)
    Gs = [fields.get('G', s)[sj, si] for s in steps]
    vals = np.log10(np.concatenate([g[np.isfinite(g) & (g > 0)] for g in Gs]))
    vmin, vmax = np.percentile(vals, [5, 99.5])
    for c, s in enumerate(steps):
        ax = fig.add_subplot(gs_top[0, c])
        G = Gs[c]
        land = ~np.isfinite(fields.get('b', s)[sj, si])
        with np.errstate(divide='ignore', invalid='ignore'):
            m = fg.draw_map(ax, X, Y, np.log10(G), cmap='viridis', land=land, vmin=vmin, vmax=vmax)
        mask = labels_by_time[keys[s]][sj, si] == track.labels[s]
        mask = ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=int(band))
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors=[COL_LABEL], linewidths=1.2)
        flag = series.loc[series['step'] == s, 'split_merge']
        ax.set_title(f'step {s}  {keys[s][:13].replace("T", " ")}h' + ('  split/merge' if flag.any() and bool(flag.iloc[0]) else '')
                     + ('  (anchor)' if s == track.anchor.step else ''), fontsize=9)
        if c:
            ax.set_ylabel('')
    cax = fig.add_axes([0.945, 0.60, 0.008, 0.26])
    fig.colorbar(m, cax=cax, label='log10 G [s^-2]')
    d = series.sort_values('step')
    h = (d['step'] - track.anchor.step).to_numpy(dtype=float)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(h, d['G_label'], 'o-', ms=3, lw=0.8, color=COL_LABEL, label=f'label mean (band {band}), the object as found')
    ax.plot(h, d['G_matched'], '-', lw=2.0, color=COL_MATCHED, label='material: anchor mean + ∫ matched rate')
    ax.plot(h, d['G_pred'], '-', lw=2.0, color=COL_PRED, label='predicted: anchor mean + ∫ 2F')
    ax.plot(h, d['G_store'], ':', lw=1.6, color=COL_STORE, label='Phase-2: anchor mean + ∫ stored DG/Dt')
    ax.plot(h, d['G_budget'], '--', lw=1.2, color=COL_BUDGET, label='four measured terms (may leave the axes)')
    for s in d.loc[d['split_merge'], 'step']:
        ax.axvline(s - track.anchor.step, color='0.85', lw=3, zorder=0)
    ax.axvline(0, color='k', lw=0.6)
    core = np.concatenate([d[c].to_numpy(dtype=float) for c in ('G_label', 'G_matched', 'G_pred', 'G_store')])
    core = core[np.isfinite(core)]
    lo, hi = core.min(), core.max()
    ax.set_ylim(max(lo - 0.15 * (hi - lo), 0.0), hi + 0.15 * (hi - lo))      # the budget curve may exit
    ax.set_xlabel('hours from the anchor')
    ax.set_ylabel('front-mean G [s^-2]')
    ax.set_title(f'strength of the front, track label {track.anchor.label}; grey bands = split/merge', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='best')
    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(0, color='k', lw=0.5)
    ax.plot(h, d['r_matched'], '-', lw=1.6, color=COL_MATCHED, label='matched rate')
    ax.plot(h, d['two_F'], '-', lw=1.6, color=COL_PRED, label='2F')
    ax.plot(h, d['r_store'], ':', lw=1.4, color=COL_STORE, label='stored DG/Dt')
    ax.set_xlabel('hours from the anchor')
    ax.set_ylabel('rate [s^-5]')
    ax.set_title('hourly rates, matched parcels', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(h, d['delta_b'], 'o-', ms=3, lw=0.8, color='k')
    ax.set_xlabel('hours from the anchor')
    ax.set_ylabel('cross-front Δb [m s^-2]')
    ax.set_title(f'secondary: Δb across the front\n(median of {int(d["n_transects"].median())} transects)', fontsize=9)
    ax = fig.add_subplot(gs[0, 4])
    ax.plot(h, d['width_km'], 'o-', ms=3, lw=0.8, color='k')
    ax.set_xlabel('hours from the anchor')
    ax.set_ylabel('25-75 % width [km]')
    ax.set_title('secondary: cross-front width', fontsize=9)
    ok = d.dropna(subset=['r_matched'])
    out = dict(track_label=int(track.anchor.label), steps_mapped=[int(s) for s in steps],
               G_label_start=float(d['G_label'].iloc[0]), G_label_anchor=float(d.loc[d['step'] == track.anchor.step, 'G_label'].iloc[0]),
               G_label_end=float(d['G_label'].iloc[-1]), G_matched_end=float(d['G_matched'].iloc[-1]),
               G_pred_end=float(d['G_pred'].iloc[-1]), G_store_end=float(d['G_store'].iloc[-1]),
               G_budget_end=float(d['G_budget'].iloc[-1]),
               frac_2F_positive=float(np.mean(ok['two_F'] > 0)), frac_matched_positive=float(np.mean(ok['r_matched'] > 0)),
               corr_matched_2F=float(np.corrcoef(ok['r_matched'], ok['two_F'])[0, 1]),
               delta_b_median=float(d['delta_b'].median()), width_km_median=float(d['width_km'].median()),
               n_split_merge=int(d['split_merge'].sum()))
    fig.suptitle(f'Figure 8 — one tracked front (anchor label {track.anchor.label}, pair {track.anchor.step}): '
                 f'the sequence and its strength against the prediction', fontsize=12, y=0.97)
    fg.caption(fig, (f'L = 0, mask_analysis, config D, flow-informed track, band {band} (M4-Q3 (a)). Maps: log10 G at five '
                     f'steps in a fixed crop, the tracked label outlined. Curves: the label mean is a mean over a changing '
                     f'pixel set; the material curve integrates the matched rate of the same parcels (task 3); the '
                     f'predicted curve integrates the parcel-averaged 2F; no slope is quoted (M3 not closed).'), fontsize=7)
    out['path'] = fg.save(fig, name + '.png', fig_dir=fig_dir)
    return out


# ---------------------------------------------------------------------------
# Figure 9 -- the population of ten fronts
# ---------------------------------------------------------------------------
def fig09_population(summary: dict, series: dict, *, fig_dir=None, name='fig09_population') -> dict:
    """(a) per front: the fraction of hours that are frontogenetic by the matched rate and by
    2F, with the pooled values and their fronts-and-hours bootstrap intervals; (b) lifetime
    against mean 2F, as the prompt asks (every track spans the window, so this is a line of
    points — said on the panel); (c) the net change of the material strength over the window
    against the integral of 2F, one point per front, the 1:1 line for reference and no fit;
    (d) the hourly matched rate against 2F over every link, binned means, no slope."""
    per = pd.DataFrame(summary['per_front']).sort_values('track')
    pooled = summary['pooled']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    (a, b), (c, d) = axes
    x = np.arange(len(per))
    w = 0.38
    a.bar(x - w / 2, per['frac_frontogenetic_matched'], w, color=COL_MATCHED, label='matched rate > 0')
    a.bar(x + w / 2, per['frac_frontogenetic_2F'], w, color=COL_PRED, label='2F > 0')
    for k, key, colr in ((0, 'frac_frontogenetic_matched', COL_MATCHED), (1, 'frac_frontogenetic_2F', COL_PRED)):
        p = pooled[key]
        a.errorbar(len(per) + 0.6 + 0.4 * k, p['value'], yerr=[[p['value'] - p['lo']], [p['hi'] - p['value']]],
                   fmt='o', color=colr, capsize=3)
    a.axhline(0.5, color='k', lw=0.5, ls='--')
    a.set_xticks(list(x) + [len(per) + 0.8])
    a.set_xticklabels([str(int(t)) for t in per['anchor_label']] + ['pooled\n(bootstrap)'], fontsize=8)
    a.set_xlabel('front (anchor label)')
    a.set_ylabel('fraction of hours')
    a.set_ylim(0, 1.25)
    a.set_title('(a) frontogenetic fraction of hours per front', fontsize=10, loc='left')
    a.legend(fontsize=8, frameon=False, loc='upper right', ncol=2)
    b.scatter(per['mean_2F'], per['lifetime_h'], s=36, color=COL_PRED)
    top = per.nlargest(3, 'mean_2F')
    for _, r in top.iterrows():
        b.annotate(str(int(r['anchor_label'])), (r['mean_2F'], r['lifetime_h']), fontsize=7,
                   xytext=(3, 3), textcoords='offset points')
    b.text(0.02, 0.92, 'labels on the three largest mean 2F only; the rest cluster near 0',
           transform=b.transAxes, fontsize=7, va='top')
    b.axvline(0, color='k', lw=0.5)
    b.set_xlabel('mean 2F along the track [s^-5]')
    b.set_ylabel('lifetime: span of the track [h]')
    b.set_title('(b) lifetime against mean 2F', fontsize=10, loc='left')
    b.text(0.02, 0.05, f'every track is found at all 71 steps:\nlifetime = the {per["lifetime_h"].max():.0f}-h window, '
                        'not a property of the front', transform=b.transAxes, fontsize=8, va='bottom')
    c.scatter(per['dG_pred_2F'], per['dG_matched'], s=36, color=COL_MATCHED, label='material ΔG vs ∫2F dt')
    c.scatter(per['dG_pred_2F'], per['dG_label'], s=24, marker='s', color=COL_LABEL, label='label-mean ΔG (changing pixel set)')
    far = per[np.abs(per['dG_pred_2F']) > 0.25 * np.abs(per['dG_pred_2F']).max()]
    for _, r in far.iterrows():
        c.annotate(str(int(r['anchor_label'])), (r['dG_pred_2F'], r['dG_matched']), fontsize=7,
                   xytext=(3, 3), textcoords='offset points')
    lim = np.nanmax(np.abs(np.concatenate([per['dG_pred_2F'], per['dG_matched'], per['dG_label']]))) * 1.1
    c.plot([-lim, lim], [-lim, lim], color='0.6', lw=0.8, ls='--', label='1:1 (reference, not a fit)')
    c.axhline(0, color='k', lw=0.4)
    c.axvline(0, color='k', lw=0.4)
    c.set_xlim(-lim, lim)
    c.set_ylim(-lim, lim)
    c.set_xlabel('∫ 2F dt over the window [s^-2]')
    c.set_ylabel('ΔG over the window [s^-2]')
    c.set_title('(c) net change against the integrated prediction, per front', fontsize=10, loc='left')
    c.legend(fontsize=7, frameon=False, loc='upper left')
    allr = pd.concat([df.assign(track=k) for k, df in series.items()], ignore_index=True).dropna(subset=['r_matched'])
    d.scatter(allr['two_F'], allr['r_matched'], s=6, alpha=0.35, color=COL_MATCHED)
    edges = np.nanpercentile(allr['two_F'], np.linspace(0, 100, 11))
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (allr['two_F'] >= lo_) & (allr['two_F'] <= hi_)
        if m.sum() > 3:
            d.plot(allr.loc[m, 'two_F'].mean(), allr.loc[m, 'r_matched'].mean(), 's', color=COL_PRED, ms=6)
    lim = np.nanpercentile(np.abs(np.concatenate([allr['two_F'], allr['r_matched']])), 99)
    d.plot([-lim, lim], [-lim, lim], color='0.6', lw=0.8, ls='--')
    d.set_xlim(-lim, lim)
    d.set_ylim(-lim, lim)
    d.axhline(0, color='k', lw=0.4)
    d.axvline(0, color='k', lw=0.4)
    d.set_xlabel('2F on the matched parcels [s^-5]')
    d.set_ylabel('matched rate d(front-mean G)/dt [s^-5]')
    sa = pooled['sign_agreement']
    d.set_title(f'(d) every link ({len(allr)}): sign agreement {sa["value"]:.2f} [{sa["lo"]:.2f}, {sa["hi"]:.2f}]; '
                'decile means; NO slope quoted', fontsize=9, loc='left')
    fig.suptitle('Figure 9 — the ten tracked fronts: frontogenetic or frontolytic, and against the prediction', fontsize=12)
    fg.caption(fig, ('L = 0, mask_analysis, config D, flow-informed tracks, band 1, centred rates (M4-Q3 (a)). Intervals: '
                     f'bootstrap over fronts and {pooled["frac_frontogenetic_matched"]["block_h"]}-h blocks of hours, '
                     f'{pooled["frac_frontogenetic_matched"]["n_boot"]} draws. No slope is quoted as an efficiency (M3 not closed).'),
               fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = dict(n_fronts=int(len(per)),
               pooled={k: {kk: v[kk] for kk in ('value', 'lo', 'hi')} for k, v in pooled.items() if isinstance(v, dict)},
               frac_matched_per_front=[float(v) for v in per['frac_frontogenetic_matched']],
               frac_2F_per_front=[float(v) for v in per['frac_frontogenetic_2F']],
               lifetime_h=[float(v) for v in per['lifetime_h']],
               n_fronts_dG_matched_positive=int((per['dG_matched'] > 0).sum()),
               n_fronts_dG_pred_positive=int((per['dG_pred_2F'] > 0).sum()),
               n_fronts_sign_agree_dG=int((np.sign(per['dG_matched']) == np.sign(per['dG_pred_2F'])).sum()),
               n_links=int(len(allr)))
    out['path'] = fg.save(fig, name + '.png', fig_dir=fig_dir)
    return out
