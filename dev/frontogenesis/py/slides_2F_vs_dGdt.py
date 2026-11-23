"""Slides: ``2F`` against the semi-Lagrangian ``DG/Dt`` hour by hour along one tracked front
(frontogenesis_slides.md, Figures / Comparing dG/dt and 2F, prompts 2-3; Q&A CMP-Q1..Q9).

Modelled on the top panel of Figure 1 of ``tracked/matched_parcels_track_81.md``, reformatted for
a 16:9 Google Slide (``SLIDE_SIZE`` at ``METHOD_DPI``, no title or footnote):

* the two band-mean rates only (CMP-Q1 (a)): ``2F`` (``two_F``, the stored discretely consistent
  form) and the stored semi-Lagrangian ``DG/Dt`` (``r_store``), each at its pair's midpoint time;
* a small summary box: the two track means and the correlation of the hourly values (CMP-Q2; no
  ratio is shown -- none is an efficiency);
* the hour of the method slides, pair 36, marked on front 0 (CMP-Q3), and asserted equal to the
  band means of ``slides_2F_36.json`` / ``slides_dGdt_36.json``;
* the anchor hour as a vertical line; no split / merge markers (CMP-Q4).

Comparing prompt 3: ``--gallery`` also draws ``slides/2F_vs_dGdt_gallery.png``, the other nine
fronts in a 3 x 3 grid, each on its own y-scale, tagged ``k: label L`` on its colour in
``slides/tracking_top10.gif`` (``slides_tracking.TOP_COLOURS``; the tags are asserted against
``m4_tracks.json``).  The y-axis reads "mean over the front's parcels" (CMP-Q8).

Reads only ``data/m4_strength.json`` and ``data/m4_tracks.json`` (no store).  Front 0 is the front of all the slides (label
81 at the anchor); the gallery of all ten fronts is not drawn yet (CMP-Q5).  Run in the
``frontogenesis`` env:

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python slides_2F_vs_dGdt.py
"""
import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from slides_tracking import SLIDE_SIZE, METHOD_DPI, TOP_COLOURS, TRACKS   # noqa: E402

DATA_DIR = HERE.parent / 'data'
SLIDE_DIR = HERE.parent / 'slides'
STRENGTH = DATA_DIR / 'm4_strength.json'
SLIDES_2F = DATA_DIR / 'slides_2F_{pair}.json'
SLIDES_DG = DATA_DIR / 'slides_dGdt_{pair}.json'
SUMMARY = DATA_DIR / 'slides_2F_vs_dGdt_{track}.json'

TRACK = 0                    # the front of all the slides (label 81 at the anchor)
SLIDES_PAIR = 36             # the hour of the method slides
SCALE = 1e19                 # s^-5 -> 10^-19 s^-5
FS_LABEL = 13                # pt, axis labels
FS_TICK = 11                 # pt, ticks, legend, box
FS_GAL = 9                   # pt, gallery text (Comparing prompt 3)
GALLERY = SLIDE_DIR / '2F_vs_dGdt_gallery.png'
COL_2F = '#2166ac'
COL_DG = '#762a83'
TOL = 1e-6                   # relative, pair-36 values against the slide JSONs


def _time(s):
    """'2012-07-03T12_30_00' -> datetime."""
    return dt.datetime.strptime(s, '%Y-%m-%dT%H_%M_%S')


def load(track=TRACK, path=STRENGTH):
    rows = json.loads(pathlib.Path(path).read_text())['series'][str(track)]
    ok = [r for r in rows if np.isfinite(r['two_F']) and np.isfinite(r['r_store'])]
    return dict(track=track, rows=rows, anchor_step=rows[0]['anchor_step'],
                anchor_label=rows[0]['anchor_label'],
                t=np.array([_time(r['time']) for r in ok]),
                step=np.array([r['step'] for r in ok]),
                label=np.array([r['label'] for r in ok]),
                two_F=np.array([r['two_F'] for r in ok]) * SCALE,
                dgdt=np.array([r['r_store'] for r in ok]) * SCALE,
                t_anchor=_time(next(r['time'] for r in rows if r['step'] == rows[0]['anchor_step'])))


def check_slides_pair(d, pair=SLIDES_PAIR):
    """The pair-36 values must be the band means of the method slides."""
    i = int(np.where(d['step'] == pair)[0][0])
    ref_2F = json.loads(SLIDES_2F.with_name(SLIDES_2F.name.format(pair=pair)).read_text())
    ref_dg = json.loads(SLIDES_DG.with_name(SLIDES_DG.name.format(pair=pair)).read_text())
    want_2F = ref_2F['decomposition']['band']['two_F']
    want_dg = ref_dg['check']['band_mean'] * SCALE
    for name, got, want in (('2F', d['two_F'][i], want_2F), ('DG/Dt', d['dgdt'][i], want_dg)):
        if abs(got - want) > TOL * abs(want):
            raise AssertionError(f'pair {pair} {name}: series {got:.6f} != slides {want:.6f}')
    if int(d['label'][i]) != int(ref_2F['label']):
        raise AssertionError(f'pair {pair}: label {d["label"][i]} != slides {ref_2F["label"]}')
    return i


def fig_compare(d, out_png, mark=None):
    W, H = SLIDE_SIZE
    with plt.rc_context({'font.size': FS_TICK, 'xtick.labelsize': FS_TICK,
                         'ytick.labelsize': FS_TICK}):
        fig = plt.figure(figsize=SLIDE_SIZE, dpi=METHOD_DPI)
        ax = fig.add_axes([1.12 / W, 0.72 / H, 8.70 / W, 3.58 / H])
        ax.axhline(0, color='k', lw=0.7, zorder=1)
        ax.axvline(d['t_anchor'], color='0.55', lw=1.0, ls='--', zorder=1)
        ax.plot(d['t'], d['two_F'], color=COL_2F, lw=2.2, label=r'$2F$ (predicted)', zorder=4)
        ax.plot(d['t'], d['dgdt'], color=COL_DG, lw=2.2, label=r'$DG/Dt$ (measured, semi-Lagrangian)',
                zorder=3)
        ylo, yhi = ax.get_ylim()
        ax.text(d['t_anchor'], yhi, f'  anchor: label {d["anchor_label"]}', ha='left', va='top',
                fontsize=FS_TICK - 1, color='0.4')
        if mark is not None:
            t0 = d['t'][mark]
            for v, col in ((d['two_F'][mark], COL_2F), (d['dgdt'][mark], COL_DG)):
                ax.scatter([t0], [v], s=55, facecolor=col, edgecolor='w', lw=1.2, zorder=6)
            top = max(d['two_F'][mark], d['dgdt'][mark])
            ax.annotate(f'the slides\' hour, {t0:%H:%M %-d %b}\n'
                        rf'$2F$ {d["two_F"][mark]:+.2f},  $DG/Dt$ {d["dgdt"][mark]:+.2f}',
                        xy=(t0, top), xytext=(t0 - dt.timedelta(hours=14), top + 0.45 * (yhi - top)),
                        fontsize=FS_TICK - 1, ha='center', va='bottom',
                        arrowprops=dict(arrowstyle='-', color='0.3', lw=0.8, shrinkB=5))
        corr = float(np.corrcoef(d['two_F'], d['dgdt'])[0, 1])
        ax.text(0.985, 0.97, f'{len(d["t"])} hours, track means:\n'
                rf'$2F$ {d["two_F"].mean():+.2f},  $DG/Dt$ {d["dgdt"].mean():+.2f}' '\n'
                f'hourly correlation {corr:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=FS_TICK,
                bbox=dict(fc='w', ec='0.5', lw=0.6, alpha=0.9, pad=4), zorder=10)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%-d %b'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        ax.set_xlim(d['t'][0] - dt.timedelta(hours=1), d['t'][-1] + dt.timedelta(hours=1))
        ax.set_ylabel('mean over the front\'s parcels\n' r'[10$^{-19}$ s$^{-5}$]', fontsize=FS_LABEL)
        ax.set_xlabel('2012 (UTC)', fontsize=FS_LABEL, labelpad=2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', fontsize=FS_TICK, frameon=False, ncol=2)
        out_png = pathlib.Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METHOD_DPI)
        plt.close(fig)
    return dict(png=str(out_png), size_in=list(SLIDE_SIZE), dpi=METHOD_DPI, n_hours=int(len(d['t'])),
                mean_two_F=float(d['two_F'].mean()), mean_DGDt=float(d['dgdt'].mean()), corr=corr,
                frac_positive_two_F=float(np.mean(d['two_F'] > 0)),
                frac_positive_DGDt=float(np.mean(d['dgdt'] > 0)),
                frac_same_sign=float(np.mean(np.sign(d['two_F']) == np.sign(d['dgdt']))))


def gif_labels(path=TRACKS):
    """The GIF's front tags (``slides_tracking.gif_top10``): track k -> its anchor label."""
    flow = json.loads(pathlib.Path(path).read_text())['flow']
    return {k: int(t['anchor']['label']) for k, t in enumerate(flow)}


def _tag(ax, k, label, fs):
    """'k: label L' on the front's GIF colour (dark text on the light colours)."""
    col = TOP_COLOURS[k]
    r, g, b = (int(col[i:i + 2], 16) / 255 for i in (1, 3, 5))
    txt = 'k' if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else 'w'
    ax.text(0.012, 0.95, f'{k}: label {label}', transform=ax.transAxes, ha='left', va='top',
            fontsize=fs, fontweight='bold', color=txt, zorder=10,
            bbox=dict(boxstyle='round,pad=0.25', fc=col, ec='none'))


def fig_gallery(tracks, out_png, labels):
    """``slides/2F_vs_dGdt_gallery.png``: the other nine fronts, 3 x 3, each on its own y-scale,
    tagged and coloured as in ``tracking_top10.gif`` (Comparing prompt 3)."""
    W, H = SLIDE_SIZE
    ncol, nrow = 3, 3
    x0, y0, x1, y1 = 0.62, 0.52, 9.92, 4.12
    gx, gy = 0.30, 0.16
    pw, ph = (x1 - x0 - (ncol - 1) * gx) / ncol, (y1 - y0 - (nrow - 1) * gy) / nrow
    t_lo = min(d['t'][0] for d in tracks) - dt.timedelta(hours=1)
    t_hi = max(d['t'][-1] for d in tracks) + dt.timedelta(hours=1)
    out = {}
    with plt.rc_context({'font.size': FS_GAL, 'xtick.labelsize': FS_GAL - 0.5,
                         'ytick.labelsize': FS_GAL - 0.5}):
        fig = plt.figure(figsize=SLIDE_SIZE, dpi=METHOD_DPI)
        for n, d in enumerate(tracks):
            r, c = divmod(n, ncol)
            ax = fig.add_axes([(x0 + c * (pw + gx)) / W, (y1 - (r + 1) * ph - r * gy) / H, pw / W, ph / H])
            ax.axhline(0, color='k', lw=0.5, zorder=1)
            ax.axvline(d['t_anchor'], color='0.55', lw=0.8, ls='--', zorder=1)
            ax.plot(d['t'], d['two_F'], color=COL_2F, lw=1.4, zorder=4)
            ax.plot(d['t'], d['dgdt'], color=COL_DG, lw=1.4, zorder=3)
            ax.set_xlim(t_lo, t_hi)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.30 * (hi - lo))           # room for the tag
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0,)))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(12,)))
            if r == nrow - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%-d %b'))
            else:
                ax.set_xticklabels([])
            ax.tick_params(length=2.5, pad=1.5)
            ax.spines[['top', 'right']].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(TOP_COLOURS[d['track']])
                ax.spines[side].set_linewidth(1.6)
            _tag(ax, d['track'], labels[d['track']], FS_GAL)
            corr = float(np.corrcoef(d['two_F'], d['dgdt'])[0, 1])
            ax.text(0.99, 0.95, rf'means {d["two_F"].mean():+.2f} / {d["dgdt"].mean():+.2f};  r {corr:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=FS_GAL - 1, color='0.25',
                    zorder=10, bbox=dict(fc='w', ec='none', alpha=0.85, pad=1))
            out[d['track']] = dict(anchor_label=labels[d['track']], mean_two_F=float(d['two_F'].mean()),
                                   mean_DGDt=float(d['dgdt'].mean()), corr=corr, n_hours=int(len(d['t'])),
                                   ylim=[float(v) for v in ax.get_ylim()])
        fig.text(0.12 / W, (y0 + y1) / 2 / H, 'mean over the front\'s parcels  ' r'[10$^{-19}$ s$^{-5}$]',
                 rotation=90, ha='left', va='center', fontsize=FS_GAL + 1)
        fig.legend(handles=[plt.Line2D([], [], color=COL_2F, lw=1.8, label=r'$2F$ (predicted)'),
                            plt.Line2D([], [], color=COL_DG, lw=1.8, label=r'$DG/Dt$ (measured, semi-Lagrangian)'),
                            plt.Line2D([], [], color='0.55', lw=0.9, ls='--', label='anchor hour'),
                            plt.Line2D([], [], ls='', label='upper right: track means of 2F / DG/Dt;  r: hourly correlation')],
                   loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                   fontsize=FS_GAL, handlelength=1.6, columnspacing=1.4)
        fig.text(0.5, 0.04, '2012 (UTC); each front on its own scale', ha='center', va='bottom',
                 fontsize=FS_GAL)
        out_png = pathlib.Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METHOD_DPI)
        plt.close(fig)
    return dict(png=str(out_png), size_in=list(SLIDE_SIZE), dpi=METHOD_DPI, fronts=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--track', type=int, default=TRACK)
    p.add_argument('--out', default=None, help='default: slides/2F_vs_dGdt.png for track 0')
    p.add_argument('--gallery', action='store_true',
                   help='also draw slides/2F_vs_dGdt_gallery.png: fronts 1-9 (Comparing prompt 3)')
    a = p.parse_args(argv)
    d = d0 = load(a.track)
    mark = check_slides_pair(d) if a.track == TRACK else None
    out = pathlib.Path(a.out) if a.out else SLIDE_DIR / ('2F_vs_dGdt.png' if a.track == TRACK
                                                        else f'2F_vs_dGdt_front{a.track}.png')
    res = dict(track=a.track, anchor_step=d['anchor_step'], anchor_label=d['anchor_label'],
               figure=fig_compare(d, out, mark))
    if mark is not None:
        res['slides_pair'] = dict(pair=SLIDES_PAIR, label=int(d['label'][mark]),
                                  two_F=float(d['two_F'][mark]), DGDt=float(d['dgdt'][mark]))
    if a.gallery:
        labels = gif_labels()
        tracks = [load(k) for k in range(1, 10)]
        for d in tracks:                       # the GIF's tag is the series' anchor label
            if labels[d['track']] != int(d['anchor_label']):
                raise AssertionError(f'track {d["track"]}: GIF label {labels[d["track"]]} != '
                                     f'series anchor {d["anchor_label"]}')
        if labels[TRACK] != int(d0['anchor_label']):
            raise AssertionError('front 0: GIF label != series anchor')
        res['gallery'] = fig_gallery(tracks, GALLERY, labels)
        res['gallery']['n_hours'] = sorted({v['n_hours'] for v in res['gallery']['fronts'].values()})
    path = SUMMARY.with_name(SUMMARY.name.format(track=a.track))
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    main()
