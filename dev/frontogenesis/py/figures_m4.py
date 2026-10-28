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
