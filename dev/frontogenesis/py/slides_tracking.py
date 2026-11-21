"""Slides: how well one front is tracked (frontogenesis_slides.md, Figures / Tracking, prompt 1).

Three panels at the midpoint of the 72-h analysis window and 5 h either side -- pairs 30,
35, 40 (``time_mid`` 2012-07-03 06:30, 11:30, 16:30 UTC; Tracking prompt 2) -- zoomed on the
tracked front of ``tracked/matched_parcels_track_81.md`` (M4 flow track 0, anchor label 81 at
pair 35).  Each panel shows ``log10 G`` (``L = 0``, grey, TR-Q3 (a)) and a dotted cyan curve
that **lassoes the front's parcels as defined in M4**: the label the tracker chose at that
pair, dilated by ``BAND = 1`` cell (3 x 3, 8-connected; ``tracking.BAND``, M4-Q3 (a), as in
``tracking.matched_link``) -- TR-Q1.  The curve is the 0.5 contour of that band on the
cell-centre grid, so it runs half a cell outside the outermost parcels.  Panel 2 adds panel 1's
lasso (red); panel 3 adds panel 2's (red) and panel 1's (yellow).  The earlier lassos are
drawn **where they were found**, not advected by the flow (TR-Q2 (a)).

``slides/tracking_81.gif`` (Tracking prompt 3) is the same front over the **whole window**,
one frame per pair (0-70, ``time_mid`` 2012-07-02 00:30 to 07-04 22:30 UTC): the chosen
label's skeleton pixels as red dots, the current front only (Tracking prompt 4: red, no
previous hour), on ``log10 G`` with fixed colour limits and a
fixed crop holding the front at every pair.  The frame title carries the split / merge flags
of ``data/matched_parcels_track_81.json``.

``slides/tracking_top10.gif`` (Tracking prompt 5) is the same for **all ten M4 flow tracks**
-- the ten fronts of the track analysis (``m4_tracks.json`` ``flow[0..9]``, anchored at the
ten largest labels of pair 35; ``surface_flux_tracks.py`` runs on the same ten) -- one colour
per track, on grey ``log10 G``, with the ``mask_analysis`` boundary dashed.

``slides/tracking_method.png`` (Tracking / Method prompts 2 and 4) is how the tracker makes one
link, 35 -> 36 (label 81 -> 83, TR-M-Q1): (a) the front and its 1-h displacement ``d`` (drawn
x10, TR-M-Q2 (a)); (b) a close-up of the prediction ``M(x - d) >= 0.5``; (c) the candidates at
pair 36 with the search radius and the 3R veto.  Prompt 4 dropped the score panel (it is in
``report/frontogenesis_tracking.md`` §3.4) and the title.  The figure is sized for a 16:9 Google
Slide under its own title (10 x 4.45 in, 300 dpi, fonts >= 16 pt; the legend 12 pt,
Method prompt 5).  ``compute_link`` re-runs the decision with
the tracker's own functions (``tracking.Flow`` on ``tile330_flow.zarr``,
``flow_weighted_score``, ``front_tracking``) and asserts the chosen label, its score and every
term against ``m4_tracks.json``.

The figure, the GIFs and the top-10 GIF recompute nothing. The per-pair labels come from
``data/m4_tracks.json`` (``flow[0]``), the pixels from ``data/tile330_fronts_D_L0.zarr``, and
``G`` from ``tile330_derived_L0.zarr``. The module imports ``fronts``, so it needs the
``frontogenesis`` env.

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python slides_tracking.py
        [--what fig|gif|both|top10|method|all]   # default both; top10 ~1 min; method ~10 s
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import ndimage
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import figures as fg                                                 # noqa: E402
import semilag as sl                                                 # noqa: E402
import tracking as tk                                                # noqa: E402
from fronts import front_tracking as ft                              # noqa: E402
from slides_scene import LAND, DPI                                   # noqa: E402

DATA_DIR = HERE.parent / 'data'
SLIDE_DIR = HERE.parent / 'slides'
TRACKS = DATA_DIR / 'm4_tracks.json'
FRONTS = DATA_DIR / 'tile330_fronts_D_L{L}.zarr'

L = 0
TRACK_INDEX = 0              # flow track 0 = anchor label 81 at pair 35
ANCHOR_LABEL = 81
MID_PAIR = 35                # midpoint of the 71 pairs (time_mid 2012-07-03 11:30 UTC)
OFFSET = 5                   # panels at MID_PAIR - OFFSET, MID_PAIR, MID_PAIR + OFFSET [h]
BAND = 1                     # parcels = label dilated by BAND cells, 3x3 (tracking.BAND, M4-Q3 (a))
STRUCT = np.ones((3, 3), bool)
PAD = 12                     # cells of margin around the union of the three fronts
COL_NOW = '#00e5ff'          # cyan: this panel's front
COL_PREV = '#ff2d55'         # red: one hour earlier
COL_PREV2 = '#ffd400'        # yellow: two hours earlier
CMAP = 'Greys'               # grey G (dark = strong) so cyan, red and yellow all read
LW = (1.6, 1.6, 1.6)         # lasso line width for this panel / one panel earlier / two earlier
LS = (0, (1.0, 1.2))         # dotted
SUMMARY = DATA_DIR / 'slides_tracking_{label}.json'   # what this run drew (every logged number)
PARCELS_JSON = DATA_DIR / 'matched_parcels_track_{label}.json'   # split / merge flags per step
GIF_MS = 300                 # frame duration [ms]: 71 frames ~ 21 s
GIF_HOLD = 4                 # extra frames' worth of pause on the last frame
GIF_DPI = 100
COL_GIF = COL_PREV           # GIF: the active front in red, nothing else (Tracking prompt 4)
DOT_GIF = 5.0                # GIF marker area [pt^2]
MASKS = DATA_DIR / 'tile330_masks.nc'
N_TOP = 10                   # the ten M4 flow tracks
TOP_COLOURS = ('#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
               '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#ff9f9f')   # track 0..9
TOP_DOT = 5.0                # marker area [pt^2] on the near-full-tile view
TOP_PAD = 15
TOP_DPI = 80
TOP_GREYS = 48               # grey levels in the GIF palette (the rest: exact track colours)
# grey scale squeezed to its middle so light and dark colours both read against it
CMAP_TOP = LinearSegmentedColormap.from_list(
    'greys_mid', plt.get_cmap('Greys')(np.linspace(0.12, 0.78, 256)))

# Tracking / Method: one link of the tracker, drawn with the tracker's own functions
FLOW_STORE = DATA_DIR / 'tile330_flow.zarr'
METHOD_FROM = 35             # link 35 -> 36: label 81 -> 83 (TR-M-Q1)
MAG = 10                     # displacement arrows drawn x10 (TR-M-Q2 (a))
METHOD_PAD = 8               # cells of margin around the front and the allowed candidates
ZOOM = 15                    # close-up (b): ZOOM x ZOOM cells
ARROW_EVERY = 4              # (a): one arrow per 4th skeleton pixel along the front
COL_PRED = '#ff9500'         # orange: the flow-predicted mask
COL_CHOSEN = '#ff2d55'       # red: the chosen candidate
COL_ALLOWED = '#2ca02c'      # green: scored (passed the veto), not chosen
COL_VETO = '#4c72b0'         # blue: vetoed candidates
TERM_ORDER = ('position', 'flow_iou', 'overlap', 'length', 'orientation', 'area')
SLIDE_SIZE = (10.0, 4.45)    # in: a 16:9 Google Slide (10 x 5.625 in) less its own title
METHOD_DPI = 300
FS_SLIDE = 16                # pt, the smallest font on the method slide (Method prompt 4)
FS_LEGEND = 12               # pt, the legend only (Method prompt 5: small legend fonts)
Y_TITLE, Y_EQ = 4.02, 1.50  # in: panel titles (bottom) and equations (top) on the slide
TOL_SCORE = 1e-9


def load_track(index=TRACK_INDEX, path=TRACKS):
    d = json.load(open(path))['flow'][int(index)]
    return dict(anchor=d['anchor'], labels={int(s): int(v) for s, v in d['labels'].items()},
                times=d['times'])


def load_panels(pairs, track, data_dir=None):
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    lab = xr.open_zarr(data_dir / FRONTS.name.format(L=L))['labels']
    out = []
    for p in pairs:
        if p not in track['labels']:
            raise ValueError(f'the track has no label at pair {p}')
        h = fg.hour(L, p, data_dir)
        mask = np.asarray(lab.isel(time=p).values) == track['labels'][p]
        if not mask.any():
            raise ValueError(f'label {track["labels"][p]} is empty at pair {p}')
        band = ndimage.binary_dilation(mask, STRUCT, iterations=BAND)
        out.append(dict(pair=p, label=track['labels'][p], mask=mask, band=band,
                        G=np.asarray(h['G'].values, 'float64'),
                        X=np.asarray(h['XC'].values), Y=np.asarray(h['YC'].values),
                        time_mid=str(np.datetime_as_string(h['time_mid'].values, unit='m'))))
    return out


def crop_of(panels, pad=PAD):
    jj, ii = np.nonzero(np.logical_or.reduce([p['band'] for p in panels]))
    nj, ni = panels[0]['mask'].shape
    return (slice(max(jj.min() - pad, 0), min(jj.max() + pad + 1, nj)),
            slice(max(ii.min() - pad, 0), min(ii.max() + pad + 1, ni)))


def _lasso(ax, panel, crop, col, lw, z):
    """The 0.5 contour of the parcel band: a closed curve around every parcel."""
    cs = ax.contour(panel['X'][crop], panel['Y'][crop], panel['band'][crop].astype(float),
                    levels=[0.5], colors=[col], linewidths=lw, linestyles=[LS], zorder=z)
    return cs


def draw(panels, out_png):
    crop = crop_of(panels)
    X0, Y0 = panels[0]['X'][crop], panels[0]['Y'][crop]
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = [np.log10(np.where(p['G'][crop] > 0, p['G'][crop], np.nan)) for p in panels]
    vlim = np.nanpercentile(np.concatenate([a.ravel() for a in logs]), [2, 99.5])
    aspect = 1 / np.cos(np.deg2rad(np.nanmean(Y0)))
    w = (np.nanmax(X0) - np.nanmin(X0)) / aspect
    h = np.nanmax(Y0) - np.nanmin(Y0)
    pw = 5.2
    fig, axs = plt.subplots(1, 3, figsize=(3 * pw + 1.6, pw * h / w + 1.9), sharey=True)
    colours = (COL_NOW, COL_PREV, COL_PREV2)
    for k, (ax, p) in enumerate(zip(axs, panels)):
        X, Y = p['X'][crop], p['Y'][crop]
        land = ~np.isfinite(p['G'][crop])
        m = ax.pcolormesh(X, Y, np.ma.masked_invalid(logs[k]), cmap=CMAP, vmin=vlim[0],
                          vmax=vlim[1] + 0.6, shading='nearest', rasterized=True)
        if land.any():
            ax.pcolormesh(X, Y, np.ma.masked_where(~land, np.ones_like(X)),
                          cmap=ListedColormap([LAND]), shading='nearest', rasterized=True)
        # earlier lassos first, this panel's lasso on top
        for back in range(k, -1, -1):
            _lasso(ax, panels[k - back], crop, colours[back], LW[back], z=3 + (k - back))
        ax.set_aspect(aspect)
        ax.set_xlim(np.nanmin(X0), np.nanmax(X0))
        ax.set_ylim(np.nanmin(Y0), np.nanmax(Y0))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.1f}°W'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{abs(y):.1f}°N'))
        ax.tick_params(labelsize=11)
        ax.set_xlabel('longitude', fontsize=12)
        ax.set_title(f'{p["time_mid"].replace("T", " ")} UTC\n'
                     f'pair {p["pair"]}, label {p["label"]} ({int(p["band"].sum())} parcels)',
                     fontsize=12)
    axs[0].set_ylabel('latitude', fontsize=12)
    cb = fig.colorbar(m, ax=axs, fraction=0.025, pad=0.015, extend='both')
    cb.set_label(r'$\log_{10}\,G$  [s$^{-4}$]', fontsize=12)
    dt = [panels[1]['pair'] - panels[0]['pair'], panels[2]['pair'] - panels[0]['pair']]
    handles = [Line2D([], [], color=COL_NOW, lw=2.2, ls=LS,
                      label='parcels of the tracked front, this hour'),
               Line2D([], [], color=COL_PREV, lw=2.2, ls=LS,
                      label=f'previous panel ({dt[0]} h earlier, as found)'),
               Line2D([], [], color=COL_PREV2, lw=2.2, ls=LS,
                      label=f'first panel ({dt[1]} h earlier, as found)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.2, top=0.86, wspace=0.06)
    out_png = pathlib.Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    return dict(png=str(out_png), crop_j=[int(crop[0].start), int(crop[0].stop)],
                crop_i=[int(crop[1].start), int(crop[1].stop)], log10G_limits=[float(v) for v in vlim])


def _lon(x, _):
    return f'{abs(x):.1f}°W'


def _lat(y, _):
    return f'{abs(y):.1f}°N'


def gif(track, out_gif, data_dir=None, pad=PAD):
    """One frame per pair of the track: the chosen skeleton (red), on ``log10 G`` in a fixed
    crop and colour scale."""
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    steps = sorted(track['labels'])
    lab = xr.open_zarr(data_dir / FRONTS.name.format(L=L))['labels']
    masks = {s: np.asarray(lab.isel(time=s).values) == track['labels'][s] for s in steps}
    for s in steps:
        if not masks[s].any():
            raise ValueError(f'label {track["labels"][s]} is empty at pair {s}')
    jj, ii = np.nonzero(np.logical_or.reduce(list(masks.values())))
    nj, ni = masks[steps[0]].shape
    crop = (slice(max(jj.min() - pad, 0), min(jj.max() + pad + 1, nj)),
            slice(max(ii.min() - pad, 0), min(ii.max() + pad + 1, ni)))
    ds = fg.open_derived(L, data_dir)
    X, Y = np.asarray(ds['XC'].values)[crop], np.asarray(ds['YC'].values)[crop]
    G = np.asarray(ds['G'].isel(time=steps, j=crop[0], i=crop[1]).values, 'float64')
    tmid = [str(np.datetime_as_string(t, unit='m')) for t in ds['time_mid'].isel(time=steps).values]
    ds.close()
    with np.errstate(divide='ignore', invalid='ignore'):
        logG = np.log10(np.where(G > 0, G, np.nan))
    vlim = np.nanpercentile(logG, [2, 99.5])
    flags = {}
    pj = pathlib.Path(str(PARCELS_JSON).format(label=track['anchor']['label']))
    if pj.exists():
        for r in json.load(open(pj))['rows']:
            flags[int(r['step'])] = ' '.join(k.upper() for k in ('split', 'merge') if r.get(k) is True)   # NaN at the first step
    aspect = 1 / np.cos(np.deg2rad(np.nanmean(Y)))
    w, h = (np.nanmax(X) - np.nanmin(X)) / aspect, np.nanmax(Y) - np.nanmin(Y)
    pw = 8.0
    frames, rows = [], []
    for k, s in enumerate(steps):
        fig, ax = plt.subplots(figsize=(pw + 1.6, pw * h / w + 1.7), dpi=GIF_DPI)
        m = ax.pcolormesh(X, Y, np.ma.masked_invalid(logG[k]), cmap=CMAP, vmin=vlim[0],
                          vmax=vlim[1] + 0.6, shading='nearest', rasterized=True)
        land = ~np.isfinite(logG[k])
        if land.any():
            ax.pcolormesh(X, Y, np.ma.masked_where(~land, np.ones_like(X)),
                          cmap=ListedColormap([LAND]), shading='nearest')
        cur = masks[s][crop]
        ax.scatter(X[cur], Y[cur], s=DOT_GIF, c=COL_GIF, lw=0, zorder=4)
        ax.set_aspect(aspect)
        ax.set_xlim(np.nanmin(X), np.nanmax(X))
        ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(_lat))
        ax.tick_params(labelsize=10)
        ax.set_xlabel('longitude', fontsize=11)
        ax.set_ylabel('latitude', fontsize=11)
        flag = flags.get(s, '')
        ax.set_title(f'{tmid[k].replace("T", " ")} UTC   (pair {s} of {steps[0]}–{steps[-1]})\n'
                     f'label {track["labels"][s]}, {int(masks[s].sum())} px'
                     + (f'   [{flag}]' if flag else '')
                     + ('   anchor' if s == track['anchor']['step'] else ''), fontsize=11)
        cb = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02, extend='both')
        cb.set_label(r'$\log_{10}\,G$  [s$^{-4}$]', fontsize=11)
        ax.legend(handles=[Line2D([], [], ls='none', marker='o', ms=5, mfc=COL_GIF, mec='none',
                                  label='tracked front (skeleton)')],
                  loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=10,
                  frameon=False)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.88)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
        rows.append(dict(pair=s, label=track['labels'][s], npix=int(masks[s].sum()), flag=flag))
    info = _write_gif(frames, out_gif)
    return dict(**info,
                crop_j=[int(crop[0].start), int(crop[0].stop)],
                crop_i=[int(crop[1].start), int(crop[1].stop)],
                log10G_limits=[float(v) for v in vlim],
                n_split=sum('SPLIT' in r['flag'] for r in rows),
                n_merge=sum('MERGE' in r['flag'] for r in rows),
                npix_min_med_max=[int(np.min([r['npix'] for r in rows])),
                                  int(np.median([r['npix'] for r in rows])),
                                  int(np.max([r['npix'] for r in rows]))])


def _grab(fig):
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return im


def _fixed_palette(colours, n_grey):
    """A palette image holding the exact ``colours`` plus an ``n_grey``-step grey ramp, so
    quantisation cannot shift a track's colour toward the grey background."""
    rgb = [tuple(int(round(255 * c)) for c in matplotlib.colors.to_rgb(col)) for col in colours]
    rgb += [(g, g, g) for g in np.linspace(0, 255, n_grey).round().astype(int)]
    rgb = rgb[:256]
    flat = [v for c in rgb for v in c] + [0] * 3 * (256 - len(rgb))
    pal = Image.new('P', (1, 1))
    pal.putpalette(flat)
    return pal


def _write_gif(frames, out_gif, colours=None, n_grey=TOP_GREYS):
    """One shared palette, no dither, loop, the last frame held.  The palette is the middle
    frame's median cut, or -- with ``colours`` -- those exact colours plus a grey ramp."""
    pal = (_fixed_palette(colours, n_grey) if colours is not None else
           frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT))
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    dur = [GIF_MS] * (len(q) - 1) + [GIF_MS * (1 + GIF_HOLD)]
    out_gif = pathlib.Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out_gif, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False)
    return dict(gif=str(out_gif), n_frames=len(q), frame_ms=GIF_MS,
                size_px=list(frames[0].size), bytes=out_gif.stat().st_size)


def load_tracks(n=N_TOP, path=TRACKS):
    flow = json.load(open(path))['flow']
    if len(flow) < n:
        raise ValueError(f'{path.name} has {len(flow)} flow tracks, fewer than {n}')
    return [dict(index=k, anchor=d['anchor'],
                 labels={int(s): int(v) for s, v in d['labels'].items()})
            for k, d in enumerate(flow[:n])]


def gif_top10(tracks, out_gif, data_dir=None, pad=TOP_PAD):
    """One frame per pair: every track's chosen skeleton in its own colour, numbered at its
    centroid, on grey ``log10 G`` in a fixed crop (the union of all tracks) and colour scale.
    A track with no label at a pair is simply absent from that frame."""
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    lab = xr.open_zarr(data_dir / FRONTS.name.format(L=L))['labels']
    steps = sorted(set().union(*[t['labels'] for t in tracks]))
    L_all = np.asarray(lab.isel(time=steps).values)
    masks = {}                                    # (track, step) -> bool
    for t in tracks:
        for s, v in t['labels'].items():
            m = L_all[steps.index(s)] == v
            if not m.any():
                raise ValueError(f'track {t["index"]}: label {v} is empty at pair {s}')
            masks[t['index'], s] = m
    jj, ii = np.nonzero(np.logical_or.reduce(list(masks.values())))
    nj, ni = L_all.shape[1:]
    crop = (slice(max(jj.min() - pad, 0), min(jj.max() + pad + 1, nj)),
            slice(max(ii.min() - pad, 0), min(ii.max() + pad + 1, ni)))
    ds = fg.open_derived(L, data_dir)
    X, Y = np.asarray(ds['XC'].values)[crop], np.asarray(ds['YC'].values)[crop]
    G = np.asarray(ds['G'].isel(time=steps, j=crop[0], i=crop[1]).values, 'float64')
    tmid = [str(np.datetime_as_string(t, unit='m')) for t in ds['time_mid'].isel(time=steps).values]
    ds.close()
    analysis = xr.open_dataset(data_dir / MASKS.name)['mask_analysis'].values[crop]
    with np.errstate(divide='ignore', invalid='ignore'):
        logG = np.log10(np.where(G > 0, G, np.nan))
    vlim = np.nanpercentile(logG, [2, 99.5])
    aspect = 1 / np.cos(np.deg2rad(np.nanmean(Y)))
    w, h = (np.nanmax(X) - np.nanmin(X)) / aspect, np.nanmax(Y) - np.nanmin(Y)
    pw = 9.0
    handles = [Line2D([], [], ls='none', marker='o', ms=6, mfc=TOP_COLOURS[t['index']], mec='none',
                      label=f'{t["index"]}: label {t["anchor"]["label"]}') for t in tracks]
    halo = [pe.withStroke(linewidth=2.2, foreground='black')]
    frames, rows = [], []
    for k, s in enumerate(steps):
        fig, ax = plt.subplots(figsize=(pw + 1.8, pw * h / w + 1.6), dpi=TOP_DPI)
        m = ax.pcolormesh(X, Y, np.ma.masked_invalid(logG[k]), cmap=CMAP_TOP, vmin=vlim[0],
                          vmax=vlim[1], shading='nearest', rasterized=True)
        land = ~np.isfinite(logG[k])
        if land.any():
            ax.pcolormesh(X, Y, np.ma.masked_where(~land, np.ones_like(X)),
                          cmap=ListedColormap([LAND]), shading='nearest')
        ax.contour(X, Y, analysis.astype(float), levels=[0.5], colors='k', linewidths=0.8,
                   linestyles='--')
        row = dict(pair=s, npix={})
        for t in tracks:
            if (t['index'], s) not in masks:
                continue
            cur = masks[t['index'], s][crop]
            col = TOP_COLOURS[t['index']]
            ax.scatter(X[cur], Y[cur], s=TOP_DOT, c=col, lw=0, zorder=4)
            ax.text(float(np.mean(X[cur])), float(np.mean(Y[cur])), str(t['index']), color=col,
                    fontsize=11, weight='bold', ha='center', va='center', zorder=5,
                    path_effects=halo)
            row['npix'][t['index']] = int(cur.sum())
        ax.set_aspect(aspect)
        ax.set_xlim(np.nanmin(X), np.nanmax(X))
        ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FuncFormatter(_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(_lat))
        ax.tick_params(labelsize=10)
        ax.set_xlabel('longitude', fontsize=11)
        ax.set_ylabel('latitude', fontsize=11)
        ax.set_title(f'The ten tracked fronts   {tmid[k].replace("T", " ")} UTC   '
                     f'(pair {s} of {steps[0]}–{steps[-1]})'
                     + ('   anchor' if s == tracks[0]['anchor']['step'] else ''), fontsize=12)
        cb = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02, shrink=0.6, extend='both')
        cb.set_label(r'$\log_{10}\,G$  [s$^{-4}$]', fontsize=11)
        ax.legend(handles=handles, title='track: anchor label at pair 35', loc='upper center',
                  bbox_to_anchor=(0.5, -0.045), ncol=5, fontsize=9.5, title_fontsize=9.5,
                  frameon=False, columnspacing=1.2, handletextpad=0.3)
        fig.subplots_adjust(left=0.1, right=0.93, bottom=0.1, top=0.96)
        frames.append(_grab(fig))
        rows.append(row)
    info = _write_gif(frames, out_gif, colours=list(TOP_COLOURS[:len(tracks)]) + [LAND])
    per = {t['index']: [r['npix'][t['index']] for r in rows if t['index'] in r['npix']]
           for t in tracks}
    return dict(**info, crop_j=[int(crop[0].start), int(crop[0].stop)],
                crop_i=[int(crop[1].start), int(crop[1].stop)],
                log10G_limits=[float(v) for v in vlim], colours=list(TOP_COLOURS[:len(tracks)]),
                tracks=[dict(index=t['index'], anchor_label=t['anchor']['label'],
                             anchor_step=t['anchor']['step'], n_steps=len(per[t['index']]),
                             npix_min_med_max=[int(min(per[t['index']])),
                                               int(np.median(per[t['index']])),
                                               int(max(per[t['index']]))]) for t in tracks])


# ---------------------------------------------------------------------------
# Tracking / Method: how one link is made
# ---------------------------------------------------------------------------
def compute_link(step=METHOD_FROM, track_index=TRACK_INDEX, data_dir=None):
    """Re-run the tracker's decision for the link ``step -> step + 1`` with its own functions
    (``tracking.Flow``, ``advect_mask_by``, ``flow_weighted_score``; ``front_tracking``), and
    return every ingredient of the figure."""
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    raw = json.load(open(data_dir / TRACKS.name))
    params, tr = raw['params'], raw['flow'][int(track_index)]
    lab_from, lab_to = int(tr['labels'][str(step)]), int(tr['labels'][str(step + 1)])
    link = tr['links'][str(step + 1)]
    labels = xr.open_zarr(data_dir / FRONTS.name.format(L=L))['labels']
    L0, L1 = (np.asarray(labels.isel(time=k).values) for k in (step, step + 1))
    h0, h1 = fg.hour(L, step, data_dir), fg.hour(L, step + 1, data_dir)

    flow = tk.Flow(store=data_dir / FLOW_STORE.name)
    di, dj = flow.displacement(step, +1)
    M = L0 == lab_from
    f = sl.interp_to_departure(M.astype('float64'), di, dj, tk.MASK_ORDER, allow_low_order=True)
    f = np.where(np.isfinite(f), f, 0.0)
    pred = flow.advect(M, step, step + 1)
    assert np.array_equal(pred, f >= 0.5)
    ref = ft.describe(L0, lab_from)
    radius = ft._radius_px(flow.dt, float(params['km_per_px']))
    dist = ft._distance_field(pred)
    weights = dict(params['weights'])
    cands = []
    for v in ft.fronts_present(L1, min_pixels=int(params['min_pixels'])):
        shape = ft.describe(L1, v)
        if not shape.valid:
            continue
        mask = L1 == v
        score, terms = tk.flow_weighted_score(shape, ref, ft.centroid(pred), radius,
                                              predicted_mask=pred, weights=weights,
                                              ref_mask=M, cand_mask=mask, ref_distance=dist)
        cands.append(dict(label=int(v), score=float(score), terms=terms, npix=int(mask.sum()),
                          length=shape.length, orientation=shape.orientation))
    cands.sort(key=lambda c: (c['score'], c['terms']['position']))
    best = cands[0]
    # the tracker's own record must be reproduced
    assert best['label'] == lab_to == link['label'], (best['label'], lab_to, link['label'])
    assert abs(best['score'] - link['score']) < TOL_SCORE, (best['score'], link['score'])
    for k in TERM_ORDER:
        assert abs(best['terms'][k] - link['terms'][k]) < TOL_SCORE, k
    assert best['score'] <= params['max_score']
    mag = np.hypot(di[M], dj[M])
    return dict(step=step, lab_from=lab_from, lab_to=lab_to, L0=L0, L1=L1, M=M, f=f, pred=pred,
                di=di, dj=dj, dist=dist, radius=radius, cands=cands, weights=weights,
                params=params, ref=ref, link=link,
                X=np.asarray(h0['XC'].values), Y=np.asarray(h0['YC'].values),
                G0=np.asarray(h0['G'].values, 'float64'), G1=np.asarray(h1['G'].values, 'float64'),
                t0=str(np.datetime_as_string(h0['time_mid'].values, unit='m')),
                t1=str(np.datetime_as_string(h1['time_mid'].values, unit='m')),
                d_median=float(np.median(mag)), d_max=float(mag.max()),
                n_changed=int((pred ^ M).sum()), n_pred=int(pred.sum()),
                iou_raw=tk.iou_dilated(M, L1 == lab_to, 0),
                iou_dil=tk.iou_dilated(M, L1 == lab_to, tk.FLOW_IOU_DILATE))


def _ll(d, j, i):
    """lon / lat at fractional ``(j, i)`` (bilinear on the cell centres)."""
    j, i = np.atleast_1d(np.asarray(j, float)), np.atleast_1d(np.asarray(i, float))
    return (ndimage.map_coordinates(d['X'], [j, i], order=1, mode='nearest'),
            ndimage.map_coordinates(d['Y'], [j, i], order=1, mode='nearest'))


def _corners(d, sj, si):
    jj, ii = np.meshgrid(np.arange(sj.start, sj.stop + 1) - 0.5,
                         np.arange(si.start, si.stop + 1) - 0.5, indexing='ij')
    x, y = _ll(d, jj.ravel(), ii.ravel())
    return x.reshape(jj.shape), y.reshape(jj.shape)


def _cells(ax, d, a, crop, **kw):
    X, Y = _corners(d, *crop)
    return ax.pcolormesh(X, Y, a[crop], shading='flat', rasterized=True, **kw)


def _outline(ax, d, mask, crop, col, lw=1.6, ls='-', z=4):
    """Cell-edge outline of ``mask``: the 0.5 contour on a 2x super-sampled grid."""
    sj, si = crop
    m = np.kron(mask[crop].astype(float), np.ones((2, 2)))
    jj = np.arange(sj.start, sj.stop, 0.5) - 0.25
    ii = np.arange(si.start, si.stop, 0.5) - 0.25
    J, I = np.meshgrid(jj, ii, indexing='ij')
    x, y = _ll(d, J.ravel(), I.ravel())
    ax.contour(x.reshape(J.shape), y.reshape(J.shape), m, levels=[0.5], colors=[col],
               linewidths=lw, linestyles=[ls], zorder=z)


def _dots(ax, d, mask, col, s=4.0, z=4, marker='s', **kw):
    jj, ii = np.nonzero(mask)
    x, y = _ll(d, jj, ii)
    ax.scatter(x, y, s=s, marker=marker, c=col, lw=0, zorder=z, rasterized=True, **kw)


def _arrows(ax, d, jj, ii, col='#00e5ff', width=0.006):
    """Displacement over the hour, drawn x MAG: from the cell to ``x + MAG d``."""
    jj, ii = np.asarray(jj), np.asarray(ii)
    x0, y0 = _ll(d, jj, ii)
    x1, y1 = _ll(d, jj + MAG * d['dj'][jj, ii], ii + MAG * d['di'][jj, ii])
    return ax.quiver(x0, y0, x1 - x0, y1 - y0, angles='xy', scale_units='xy', scale=1,
                     color=col, width=width, headwidth=3.5, headlength=4, zorder=6,
                     edgecolor='k', linewidth=0.3)


def _frame(ax, d, crop, nbins=4, labelsize=10):
    Yc = d['Y'][crop]
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Yc))))
    X, Y = _corners(d, *crop)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.xaxis.set_major_locator(MaxNLocator(nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins))
    fmt = 1 if crop[0].stop - crop[0].start > 30 else 2
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.{fmt}f}°W'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{abs(y):.{fmt}f}°N'))
    ax.tick_params(labelsize=labelsize)


def _eq(ax, s, y=-0.13, fs=12.5, box=True):
    ax.text(0.5, y, s, transform=ax.transAxes, ha='center', va='top', fontsize=fs,
            bbox=dict(boxstyle='round,pad=0.35', fc='#f4f4f4', ec='0.6') if box else None)


def method_crop(d, pad=METHOD_PAD):
    keep = d['M'] | d['pred'] | np.isin(d['L1'], [c['label'] for c in d['cands']
                                                  if np.isfinite(c['score'])])
    jj, ii = np.nonzero(keep)
    nj, ni = keep.shape
    return (slice(max(jj.min() - pad, 0), min(jj.max() + pad + 1, nj)),
            slice(max(ii.min() - pad, 0), min(ii.max() + pad + 1, ni)))


def zoom_crop(d, size=ZOOM):
    """The close-up: centred on the changed pixel (advection) with the most front around it;
    on the front's middle pixel if nothing changed."""
    ch = d['pred'] ^ d['M']
    cand = np.argwhere(ch) if ch.any() else np.argwhere(d['M'])[[int(d['M'].sum()) // 2]]
    nj, ni = d['M'].shape

    def box(jc, ic):
        j0 = int(np.clip(jc - size // 2, 0, nj - size))
        i0 = int(np.clip(ic - size // 2, 0, ni - size))
        return slice(j0, j0 + size), slice(i0, i0 + size)
    return max((box(*c) for c in cand), key=lambda b: int(d['M'][b].sum()))


def _box(ax, d, crop, col='k', lw=1.4):
    sj, si = crop
    jc = [sj.start - .5, sj.start - .5, sj.stop - .5, sj.stop - .5]
    ic = [si.start - .5, si.stop - .5, si.stop - .5, si.start - .5]
    x, y = _ll(d, jc, ic)
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fill=False, ec=col, lw=lw, zorder=7))


def fig_method(d, out_png, fs=FS_SLIDE):
    """Panels (a)-(c) of the method, sized for a 16:9 Google Slide under its own title
    (Method prompt 4): no suptitle, ``SLIDE_SIZE`` inches, ``METHOD_DPI``, fonts >= ``fs`` pt.
    Axes are placed in inches so the layout holds at the placed size."""
    crop, zc = method_crop(d), zoom_crop(d)
    logs = [np.log10(np.where(g > 0, g, np.nan)) for g in (d['G0'], d['G1'])]
    vlim = np.nanpercentile(np.concatenate([a[crop].ravel() for a in logs]), [2, 99.5])
    gkw = dict(cmap=CMAP, vmin=vlim[0], vmax=vlim[1] + 0.6)
    allowed = [c for c in d['cands'] if np.isfinite(c['score'])]
    chosen, others = allowed[0], allowed[1:]
    n_veto = len(d['cands']) - len(allowed)
    W, H = SLIDE_SIZE

    def axes_in(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def above(ax, text, y_in=Y_TITLE):          # titles and equations at fixed heights [in]
        x = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x, y_in / H, text, ha='center', va='bottom' if y_in == Y_TITLE else 'top',
                 fontsize=fs)

    with plt.rc_context({'font.size': fs, 'axes.titlesize': fs, 'legend.fontsize': fs,
                         'xtick.labelsize': fs, 'ytick.labelsize': fs}):
        fig = plt.figure(figsize=SLIDE_SIZE)
        y0, h = 1.95, 2.0
        ax_a = axes_in(1.0, y0, 2.75, h)
        ax_b = axes_in(4.0, y0, 1.85, h)
        cax_b = axes_in(5.95, y0 + 0.15, 0.12, h - 0.3)
        ax_c = axes_in(7.15, y0, 2.6, h)

        # (a) the front at t and its displacement
        _cells(ax_a, d, logs[0], crop, **gkw)
        _dots(ax_a, d, d['M'], COL_NOW, s=3)
        jj, ii = np.nonzero(d['M'])
        order = np.lexsort((ii, jj))[::ARROW_EVERY]
        _arrows(ax_a, d, jj[order], ii[order], col='#ffd400', width=0.010)
        _box(ax_a, d, zc, col='k', lw=1.6)
        _frame(ax_a, d, crop, nbins=2, labelsize=fs)
        above(ax_a, f'(a) Pair {d["step"]}: label {d["lab_from"]}')
        above(ax_a, r'$\mathbf{d} = \Delta t\,\mathbf{u}(\mathbf{x} - \mathbf{d}/2)$'
                    '\n' rf'$|\mathbf{{d}}|$ ≈ {d["d_median"]:.2f} cells/h', Y_EQ)

        # (b) close-up of the prediction
        fz = np.ma.masked_where(d['f'] <= 0, d['f'])
        mb = _cells(ax_b, d, fz, zc, cmap='Blues', vmin=0, vmax=1)
        _outline(ax_b, d, d['M'], zc, COL_NOW, lw=2.4)
        _outline(ax_b, d, d['pred'], zc, COL_PRED, lw=2.2, ls='--', z=5)
        ch = d['pred'] ^ d['M']
        if ch[zc].any():
            jc, ic = np.nonzero(ch[zc])
            x, y = _ll(d, jc + zc[0].start, ic + zc[1].start)
            ax_b.scatter(x, y, s=150, marker='o', facecolor='none', edgecolor=COL_CHOSEN,
                         lw=2.2, zorder=8)
        J, I = np.meshgrid(np.arange(zc[0].start + 1, zc[0].stop, 3),
                           np.arange(zc[1].start + 1, zc[1].stop, 3), indexing='ij')
        _arrows(ax_b, d, J.ravel(), I.ravel(), col='#ffd400', width=0.012)
        _frame(ax_b, d, zc, nbins=2, labelsize=fs)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        cb = fig.colorbar(mb, cax=cax_b, ticks=[0, 0.5, 1])
        cb.ax.tick_params(labelsize=fs)
        cb.set_label(r'$M_t(\mathbf{x}-\mathbf{d})$', fontsize=fs, labelpad=2)
        above(ax_b, '(b) Predict (close-up)')
        above(ax_b, r'$\hat M_{t+1} = [M_t(\mathbf{x}-\mathbf{d}) \geq 0.5]$', Y_EQ)

        # (c) the candidates at t + 1
        _cells(ax_c, d, logs[1], crop, **gkw)
        X, Y = d['X'][crop], d['Y'][crop]
        ax_c.contour(X, Y, d['dist'][crop],
                     levels=[d['radius'], ft.MAX_POSITION_FACTOR * d['radius']],
                     colors=[COL_PRED, COL_PRED], linewidths=[1.2, 1.6], linestyles=[':', '-'],
                     zorder=3)
        vetoed = np.isin(d['L1'], [c['label'] for c in d['cands'] if not np.isfinite(c['score'])])
        _dots(ax_c, d, vetoed, COL_VETO, s=3)
        for c in others:
            _dots(ax_c, d, d['L1'] == c['label'], COL_ALLOWED, s=5)
        _dots(ax_c, d, d['pred'], COL_PRED, s=14, z=4, alpha=0.55)
        _dots(ax_c, d, d['L1'] == chosen['label'], COL_CHOSEN, s=3, z=5)
        _frame(ax_c, d, crop, nbins=2, labelsize=fs)
        ax_c.set_yticklabels([])
        above(ax_c, f'(c) Pair {d["step"] + 1}: candidates')
        above(ax_c, f'veto: no pixel within\n{ft.MAX_POSITION_FACTOR:.0f}R = '
                    f'{ft.MAX_POSITION_FACTOR * d["radius"]:.0f} cells ({n_veto} of {len(d["cands"])})',
              Y_EQ)

        handles = [
            Line2D([], [], ls='', marker='s', ms=8, color=COL_NOW, label=f'label {d["lab_from"]} ($M_t$)'),
            Line2D([], [], ls='', marker='s', ms=9, color=COL_PRED, alpha=0.6, label=r'predicted $\hat M_{t+1}$'),
            Line2D([], [], ls='', marker='s', ms=8, color=COL_CHOSEN, label=f'chosen: {chosen["label"]}'),
            Line2D([], [], color='#ffd400', lw=3, label=rf'$\mathbf{{d}}$ ×{MAG}'),
            Line2D([], [], ls='', marker='o', mfc='none', mec=COL_CHOSEN, ms=11, mew=2.2,
                   label='dropped px'),
            *[Line2D([], [], ls='', marker='s', ms=8, color=COL_ALLOWED,
                     label=f'scored, not chosen: {c["label"]}') for c in others],
            Line2D([], [], ls='', marker='s', ms=8, color=COL_VETO, label='vetoed'),
            Line2D([], [], color=COL_PRED, lw=1.6, label=f'R, 3R ({d["radius"]:.0f}, '
                                                         f'{ft.MAX_POSITION_FACTOR * d["radius"]:.0f} cells)')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4,
                   fontsize=FS_LEGEND)
        out_png = pathlib.Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METHOD_DPI)
        plt.close(fig)
    return dict(png=str(out_png), size_in=list(SLIDE_SIZE), dpi=METHOD_DPI, min_font_pt=fs,
                step=d['step'], label_from=d['lab_from'], label_to=d['lab_to'],
                crop_j=[int(crop[0].start), int(crop[0].stop)], crop_i=[int(crop[1].start), int(crop[1].stop)],
                zoom_j=[int(zc[0].start), int(zc[0].stop)], zoom_i=[int(zc[1].start), int(zc[1].stop)],
                radius_cells=d['radius'], d_median_cells=d['d_median'], d_max_cells=d['d_max'],
                n_pred=d['n_pred'], n_changed=d['n_changed'], n_candidates=len(d['cands']),
                n_vetoed=n_veto, iou_raw=d['iou_raw'], iou_dilated=d['iou_dil'], magnification=MAG,
                allowed=[dict(label=c['label'], npix=c['npix'], score=c['score'],
                              terms={k: float(c['terms'][k]) for k in TERM_ORDER}) for c in allowed])

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mid', type=int, default=MID_PAIR)
    ap.add_argument('--out', default=str(SLIDE_DIR / f'tracking_{ANCHOR_LABEL}.png'))
    ap.add_argument('--gif', default=str(SLIDE_DIR / f'tracking_{ANCHOR_LABEL}.gif'))
    ap.add_argument('--top10', default=str(SLIDE_DIR / 'tracking_top10.gif'))
    ap.add_argument('--method', default=str(SLIDE_DIR / 'tracking_method.png'))
    ap.add_argument('--step', type=int, default=METHOD_FROM, help='method: link step -> step+1')
    ap.add_argument('--what', choices=('fig', 'gif', 'both', 'top10', 'method', 'all'),
                    default='both')
    a = ap.parse_args(argv)
    track = load_track()
    assert track['anchor']['label'] == ANCHOR_LABEL and track['anchor']['step'] == MID_PAIR
    res = {}
    if a.what in ('method', 'all'):
        res['method'] = fig_method(compute_link(a.step), a.method)
        print(json.dumps(res['method'], indent=1))
        if a.what == 'method':
            return _dump(res)
    if a.what in ('top10', 'all'):
        res['top10'] = gif_top10(load_tracks(), a.top10)
        print(json.dumps(res['top10'], indent=1))
        if a.what == 'top10':
            return _dump(res)
    if a.what in ('gif', 'both', 'all'):
        res['gif'] = gif(track, a.gif)
        print(json.dumps(res['gif'], indent=1))
    if a.what == 'gif':
        return _dump(res)
    pairs = [a.mid - OFFSET, a.mid, a.mid + OFFSET]
    panels = load_panels(pairs, track)
    out = draw(panels, a.out)
    out['panels'] = []
    for k, p in enumerate(panels):
        row = dict(pair=p['pair'], time_mid=p['time_mid'], label=p['label'],
                   npix=int(p['mask'].sum()), n_parcels=int(p['band'].sum()))
        for j in range(k):
            q = panels[j]['band']
            row[f'parcels_shared_with_pair{panels[j]["pair"]}'] = int((p['band'] & q).sum())
        out['panels'].append(row)
    print(json.dumps(out, indent=1))
    res['fig'] = out
    return _dump(res)


def _dump(res):
    path = pathlib.Path(str(SUMMARY).format(label=ANCHOR_LABEL))
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1))
    print(f'wrote {path}')
    return res


if __name__ == '__main__':
    main()
