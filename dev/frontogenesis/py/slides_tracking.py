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

Nothing is recomputed: the per-pair labels come from ``data/m4_tracks.json`` (``flow[0]``),
the pixels from ``data/tile330_fronts_D_L0.zarr``, ``G`` from ``tile330_derived_L0.zarr``.
The script reads the JSON directly (not ``tracking.tracks_from_json``) so it runs in the
``ocean14`` env, which has no ``fronts`` package.

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/ocean14/bin/python slides_tracking.py
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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import figures as fg                                                 # noqa: E402
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mid', type=int, default=MID_PAIR)
    ap.add_argument('--out', default=str(SLIDE_DIR / f'tracking_{ANCHOR_LABEL}.png'))
    a = ap.parse_args(argv)
    track = load_track()
    assert track['anchor']['label'] == ANCHOR_LABEL and track['anchor']['step'] == MID_PAIR
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
    return out


if __name__ == '__main__':
    main()
