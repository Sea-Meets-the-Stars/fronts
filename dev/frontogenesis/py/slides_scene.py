"""Slides: the analysis scene (frontogenesis_slides.md, Figures / Analysis scene, prompt 1).

Two maps of tile 330 (LLC4320 face 10, off California) at the **midpoint of the 72-h
analysis window** -- pair 35, ``time_mid`` = 2012-07-03 11:30 UTC (the 71 pairs run
07-02 00:30 to 07-04 22:30; pair 35 is also the anchor pair of tracked front 81):

* ``slides/scene_no_fronts.png`` -- ``G = |grad b|^2`` (``L = 0``), land, and the
  regions the analysis avoids;
* ``slides/scene_fronts.png`` -- the same, with the finder's fronts (config D, the
  M4 product ``tile330_fronts_D_L0.zarr``) drawn on top, with the tracked front --
  label 81 at pair 35, the anchor of M4 track 0 (``tracked/matched_parcels_track_81.md``)
  -- in a second colour (S-Q1 (a), S-Q2 (b) without inset).

Avoided regions are read from ``tile330_masks.nc`` (``mask_analysis = mask_ocean &
mask_halo & mask_offshore & mask_edge``) and drawn as two classes:

* within ``offshore_km`` (100 km) of the coast -- ``~mask_offshore``, which contains the
  7-cell land halo;
* the 7-cell tile-edge rim -- ``~mask_edge`` (outside the coastal band).

No physics is computed here; ``G`` is the stored field of the derived store.

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/ocean14/bin/python slides_scene.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import figures as fg                                                 # noqa: E402

DATA_DIR = HERE.parent / 'data'
SLIDE_DIR = HERE.parent / 'slides'
MASKS = DATA_DIR / 'tile330_masks.nc'
FRONTS = DATA_DIR / 'tile330_fronts_D_L{L}.zarr'

L = 0
PAIR = 35                    # midpoint pair of the 71 (time_mid 2012-07-03 11:30 UTC)
G_RANGE = (-17.0, -12.5)     # log10 G [s^-4] colour limits (G p1 / p99 = 6e-18 / 3e-13)
CMAP = 'cividis'
LAND = '#c8b89a'
AVOID_COAST = '#ffffff'      # translucent veil over the coastal band
AVOID_EDGE = '#9fd3ff'       # translucent veil over the tile-edge rim
VEIL_ALPHA = 0.6
FRONT_COL = '#ff2d55'
TRACK_LABEL = 81             # anchor label of M4 track 0 at PAIR (data/matched_parcels_track_81.json)
TRACK_COL = '#00e5ff'
DPI = 200


def _lon_fmt(x, _):
    return f'{abs(x):.0f}°{"W" if x < 0 else "E"}'


def _lat_fmt(y, _):
    return f'{abs(y):.0f}°{"N" if y >= 0 else "S"}'


def load_scene(L=L, pair=PAIR, data_dir=None):
    """``G``, the grid, the masks and the config-D labels of one pair."""
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    h = fg.hour(L, pair, data_dir)
    m = xr.open_dataset(data_dir / MASKS.name)
    lab = xr.open_zarr(data_dir / FRONTS.name.format(L=L))['labels'].isel(time=pair).values
    scene = dict(
        X=np.asarray(h['XC'].values), Y=np.asarray(h['YC'].values),
        G=np.asarray(h['G'].values, 'float64'),
        time_mid=str(np.datetime_as_string(h['time_mid'].values, unit='m')),
        lst=fg._lst(h),
        ocean=m['mask_ocean'].values, offshore=m['mask_offshore'].values,
        edge=m['mask_edge'].values, analysis=m['mask_analysis'].values,
        offshore_km=float(m.attrs['offshore_km']), halo_cells=int(m.attrs['halo_cells']),
        halo_km=float(m.attrs['halo_km']), edge_cells=int(m.attrs['edge_cells']),
        labels=np.asarray(lab))
    m.close()
    return scene


def draw_scene(s, with_fronts, out_png, track_label=TRACK_LABEL):
    X, Y = s['X'], s['Y']
    land = ~s['ocean']
    avoid_coast = s['ocean'] & ~s['offshore']
    avoid_edge = s['ocean'] & s['offshore'] & ~s['edge']
    assert not np.any(s['analysis'] & (avoid_coast | avoid_edge))

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        logG = np.log10(np.where(s['G'] > 0, s['G'], np.nan))
    m = ax.pcolormesh(X, Y, np.ma.masked_invalid(logG), cmap=CMAP, vmin=G_RANGE[0],
                      vmax=G_RANGE[1], shading='nearest', rasterized=True)
    ax.pcolormesh(X, Y, np.ma.masked_where(~land, np.ones_like(X)),
                  cmap=ListedColormap([LAND]), shading='nearest', rasterized=True)
    for sel, col in ((avoid_coast, AVOID_COAST), (avoid_edge, AVOID_EDGE)):
        ax.pcolormesh(X, Y, np.ma.masked_where(~sel, np.ones_like(X)),
                      cmap=ListedColormap([col]), alpha=VEIL_ALPHA, shading='nearest',
                      rasterized=True, linewidth=0)
    plt.rcParams['hatch.color'] = '0.55'
    plt.rcParams['hatch.linewidth'] = 0.5
    ax.contourf(X, Y, avoid_coast.astype(float), levels=[0.5, 1.5], colors='none',
                hatches=['////'])
    ax.contour(X, Y, s['analysis'].astype(float), levels=[0.5], colors='k',
               linewidths=1.0, linestyles='--')

    handles = [Patch(fc=LAND, ec='0.4', label='land'),
               Patch(fc=AVOID_COAST, ec='0.4', hatch='////',
                     label=f'avoided: < {s["offshore_km"]:.0f} km from the coast '
                           f'\n(includes the {s["halo_cells"]}-cell land halo)'),
               Patch(fc=AVOID_EDGE, ec='0.4', alpha=0.9,
                     label=f'avoided: {s["edge_cells"]}-cell tile-edge rim'),
               Line2D([], [], color='k', ls='--', lw=1.0, label='edge of the analysis region')]

    n_fronts = n_pix = 0
    if with_fronts:
        fr = s['labels'] > 0
        n_pix = int(fr.sum())
        n_fronts = int(np.unique(s['labels'][fr]).size)
        other = fr & (s['labels'] != track_label)
        ax.scatter(X[other], Y[other], s=0.6, marker='s', c=FRONT_COL, lw=0, rasterized=True)
        handles.append(Line2D([], [], color=FRONT_COL, lw=2.2,
                              label=f'fronts ({n_fronts} fronts, {n_pix:,} pixels)'))
        if track_label is not None:
            tk = s['labels'] == track_label
            if not tk.any():
                raise ValueError(f'label {track_label} does not exist at this pair')
            ax.scatter(X[tk], Y[tk], s=2.5, marker='s', c=TRACK_COL, lw=0, rasterized=True)
            handles.append(Line2D([], [], color=TRACK_COL, lw=3.0,
                                  label=f'tracked front (label {track_label}, {int(tk.sum())} pixels)'))

    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y))))
    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_fmt))
    ax.set_xlabel('longitude', fontsize=13)
    ax.set_ylabel('latitude', fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(color='0.3', lw=0.4, alpha=0.5)
    ax.set_title(f'LLC4320, {s["time_mid"].replace("T", " ")} UTC '
                 f'(midpoint of the 72-h window)', fontsize=13)
    cb = fig.colorbar(m, ax=ax, fraction=0.04, pad=0.02, extend='both')
    cb.set_label(r'$\log_{10}\,G = \log_{10}|\nabla b|^2$  [s$^{-4}$]', fontsize=13)
    cb.ax.tick_params(labelsize=11)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2,
              fontsize=10, frameon=False)
    fig.tight_layout()
    out_png = pathlib.Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    return dict(png=str(out_png), n_fronts=n_fronts, n_front_pixels=n_pix,
                track_label=track_label if with_fronts else None,
                n_track_pixels=int((s['labels'] == track_label).sum()) if with_fronts else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pair', type=int, default=PAIR)
    p.add_argument('--out-dir', default=str(SLIDE_DIR))
    a = p.parse_args(argv)
    s = load_scene(L, a.pair)
    ocean = s['ocean']
    summary = dict(
        L=L, pair=a.pair, time_mid=s['time_mid'], lst=round(s['lst'], 2),
        n_ocean=int(ocean.sum()), n_analysis=int(s['analysis'].sum()),
        n_avoid_coast=int((ocean & ~s['offshore']).sum()),
        n_avoid_edge=int((ocean & s['offshore'] & ~s['edge']).sum()),
        G_p1_p50_p99=[float(v) for v in np.nanpercentile(s['G'], [1, 50, 99])],
        G_range_log10=list(G_RANGE))
    out = pathlib.Path(a.out_dir)
    summary['no_fronts'] = draw_scene(s, False, out / 'scene_no_fronts.png')
    summary['fronts'] = draw_scene(s, True, out / 'scene_fronts.png')
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == '__main__':
    main()
