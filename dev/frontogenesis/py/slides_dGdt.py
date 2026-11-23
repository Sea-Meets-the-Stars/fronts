"""Slides: how the semi-Lagrangian ``DG/Dt`` of a tracked front is measured
(frontogenesis_slides.md, Figures / dG/dt and 2F, prompt 3; Q&A DG-Q1..Q9).

Pair 36 (hours 36 -> 37, 2012-07-03 12:00 -> 13:00 UTC; the front was found on the
midpoint ``G`` at 12:30) of M4 flow track 0 (anchor label 81 at pair 35), where the
tracker chose **label 83**.  The parcels are the label dilated by one cell (3 x 3,
``tracking.matched_link``, band 1).  At every cell ``x`` of that band the stored rate is
(``semilag.measured_DGDt``)

    DG/Dt(x) = [ G(x, h37) - G(x_d, h36) ] / dt,     x_d = x - d(x),

with ``d`` the one-hour displacement of the water arriving at the centre of ``x`` at 13:00
(``semilag.departure_index``, midpoint rule, 3 iterations) and ``G(x_d, h36)`` from
``b(h36)`` interpolated (cubic) onto the five-point stencil ``x_d, x_d +- e_i, x_d +- e_j``
(``semilag.gradb2_at_departure``) -- ``b`` is interpolated, never ``G``.

Two figures:

* ``slides/dGdt_method.png`` -- (a) the front and the zoom window; (b) one parcel close up:
  ``b(h36)`` contours, the arrival cell, its departure point and the stencil at true scale;
  (c) the rate of every parcel and the band mean (the number quoted for the hour); (d)-(f)
  the window: the interpolated ``G(x_d, h36)``, ``G(x, h37)`` and their difference / dt.
  dG/dt prompt 4 made it a Google-Slide figure: no title or footnote, 10 x 4.45 in at 300 dpi
  (``slides_tracking.SLIDE_SIZE`` / ``METHOD_DPI``), 13 pt titles, 11 pt elsewhere, one shared
  legend, the contour labels of (b) away from the parcel, and no numbers under (b) (they
  stay in the summary json).
* ``slides/dGdt_split.png`` -- the same rate split into a local part
  ``[G(x, h37) - G(x, h36)] / dt`` and an advective part ``[G(x, h36) - G(x_d, h36)] / dt``
  (the two add to the rate exactly), on the band (top) and in the window (bottom).  The two
  parts are each larger than their sum and largely cancel cell by cell.

Everything is recomputed from the 72-h raw store with the pipeline's own functions and
**asserted** against the stored ``DGDt_semilag`` and against ``r_store`` of
``data/matched_parcels_track_81.json``.  Run in the ``frontogenesis`` env (xgcm):

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python slides_dGdt.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import xarray as xr
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import m2_qa                                                         # noqa: E402
import inputs as inp                                                 # noqa: E402
import operators as op                                               # noqa: E402
import semilag as sl                                                 # noqa: E402
from slides_scene import DPI                                         # noqa: E402
from slides_tracking import BAND, STRUCT, LS, SLIDE_SIZE, METHOD_DPI   # noqa: E402

DATA_DIR = HERE.parent / 'data'
SLIDE_DIR = HERE.parent / 'slides'
DERIVED = DATA_DIR / 'tile330_derived_L0.zarr'
FRONTS = DATA_DIR / 'tile330_fronts_D_L0.zarr'
PARCELS_JSON = DATA_DIR / 'matched_parcels_track_81.json'
SUMMARY = DATA_DIR / 'slides_dGdt_{pair}.json'

PAIR = 36                    # DG-Q4: hours 36 -> 37 (12:00 -> 13:00 UTC), label 83 of track 0
DT = sl.DT
WIN = 15                     # DG-Q8: the zoom window [cells]
CLOSE = 5                    # DG-Q6: the one-parcel close-up [cells]
MIN_WIN_CELLS = 40           # parcels a window must hold
PAD = 6                      # cells around the band in the whole-front panels
SCALE = 1e19                 # rates quoted in 1e-19 s^-5
CMAP_G = 'cividis'           # G, as on the scene slide
CMAP_R = 'RdBu_r'            # rates
CMAP_B = 'YlGnBu'            # b in the close-up
COL_BAND = '#00e5ff'         # lasso of the parcels
COL_SKEL = '#ff2d55'         # the label's own pixels
COL_WIN = '#ff9500'          # window box
COL_CLOSE = '#ff2d55'        # close-up box / the chosen parcel
COL_DEP = '#ff2d55'          # departure point
NOBAND = '0.86'              # cells outside the band in the rate / window maps
FS_DG = 13                   # pt, panel titles on the method slide (DG prompt 4)
FS_DG_LEGEND = 11            # pt, its legend
B_LEVELS = 6                 # b contours in the close-up
TOL_STORED = 1e-5            # max |recomputed - stored| / rms(stored) on the band
TOL_JSON = 1e-6              # relative tolerance of the band mean against r_store


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------
def _np2(a):
    a = np.asarray(a.values if hasattr(a, 'values') else a, dtype='float64')
    while a.ndim > 2:
        a = a[0]
    return a


def compute(pair=PAIR, data_dir=None):
    """Every field of the construction at ``pair`` on the full tile, recomputed with the
    pipeline's functions, plus the stored tendency, the label and the band."""
    data_dir = pathlib.Path(data_dir or DATA_DIR)
    raw, g, grid, _, _ = m2_qa.open_inputs()
    h0, h1 = m2_qa.hour_ds(raw, g, pair), m2_qa.hour_ds(raw, g, pair + 1)
    f0, f1 = inp.filtered(h0, 0), inp.filtered(h1, 0)       # L = 0: the identity
    b0, b1 = f0[0], f1[0]
    U, V = (inp.midpoint(a, b) for a, b in zip(f0[1:], f1[1:]))
    uc, vc = sl._centred_pair(U, V, h0, grid)
    di, dj = sl.departure_index(uc, vc, h0, dt=DT)
    Gd = _np2(sl.gradb2_at_departure(b0, di, dj, h0))
    G0, G1 = _np2(op.gradb2(b0, h0, grid)), _np2(op.gradb2(b1, h0, grid))
    der = xr.open_zarr(data_dir / DERIVED.name).isel(time=pair)
    track = json.load(open(data_dir / PARCELS_JSON.name))
    row = next(r for r in track['rows'] if r['step'] == pair)
    lab = xr.open_zarr(data_dir / FRONTS.name)['labels'].isel(time=pair).values
    skel = lab == int(row['label'])
    band = ndi.binary_dilation(skel, STRUCT, iterations=BAND)
    d = dict(pair=pair, label=int(row['label']), row=row,
             times=[str(h0['time'].values)[:16], str(h1['time'].values)[:16]],
             time_mid=str(np.datetime_as_string(der['time_mid'].values, unit='m')),
             X=np.asarray(der['XC'].values), Y=np.asarray(der['YC'].values),
             b0=_np2(b0), G0=G0, G1=G1, Gd=Gd, di=_np2(di), dj=_np2(dj),
             DG=(G1 - Gd) / DT, stored=np.asarray(der['DGDt_semilag'].values, 'float64'),
             Gmid=np.asarray(der['G'].values, 'float64'), skel=skel, band=band)
    d['local'] = (G1 - G0) / DT
    d['advect'] = (G0 - Gd) / DT
    d['disp'] = np.hypot(d['di'], d['dj'])
    return d


def check(d):
    """The recomputation must be the stored tendency, and its band mean ``r_store``."""
    band = d['band']
    fin = band & np.isfinite(d['DG']) & np.isfinite(d['stored'])
    if fin.sum() != band.sum():
        raise ValueError(f'{band.sum() - fin.sum()} band cells are NaN')
    rms = np.sqrt(np.mean(d['stored'][band] ** 2))
    err = float(np.max(np.abs(d['DG'][band] - d['stored'][band])) / rms)
    if err > TOL_STORED:
        raise AssertionError(f'recomputed DG/Dt differs from DGDt_semilag by {err:.2e} of the rms')
    mean = float(d['DG'][band].mean())
    ref = float(d['row']['r_store'])
    if abs(mean - ref) > TOL_JSON * abs(ref):
        raise AssertionError(f'band mean {mean:.6e} != r_store {ref:.6e}')
    split = np.max(np.abs(d['local'] + d['advect'] - d['DG'])[band]) / rms
    assert split < 1e-10, split
    return dict(max_err_over_rms=err, band_mean=mean, r_store=ref, n_band=int(band.sum()),
                n_skeleton=int(d['skel'].sum()))


def choose_window(d, size=WIN, min_cells=MIN_WIN_CELLS):
    """The ``size`` x ``size`` window **centred on a front (skeleton) pixel** with the
    largest r.m.s. rate over its band cells.  Centring keeps the front through the middle
    of the window (an unconstrained search picks a corner the front only clips)."""
    band, DG = d['band'], d['DG']
    h = size // 2
    best = None
    for jc, ic in zip(*np.where(d['skel'])):
        j0, i0 = jc - h, ic - h
        w = band[j0:j0 + size, i0:i0 + size]
        if w.shape != (size, size) or w.sum() < min_cells:
            continue
        r = np.sqrt(np.mean(DG[j0:j0 + size, i0:i0 + size][w] ** 2))
        if best is None or r > best[0]:
            best = (r, int(j0), int(i0))
    return best[1], best[2]


def choose_parcel(d, win, size=WIN, margin=CLOSE // 2):
    """A ridge (skeleton) cell of the window with a positive rate whose displacement is
    nearest the band median, kept ``margin`` cells from the window edge."""
    j0, i0 = win
    med = np.median(d['disp'][d['band']])
    sel = np.zeros_like(d['band'])
    sel[j0 + margin:j0 + size - margin, i0 + margin:i0 + size - margin] = True
    sel &= d['skel'] & (d['DG'] > 0)
    if not sel.any():
        raise ValueError('no positive ridge cell inside the window')
    jj, ii = np.where(sel)
    k = int(np.argmin(np.abs(d['disp'][jj, ii] - med)))
    return int(jj[k]), int(ii[k])


# ---------------------------------------------------------------------------
# geometry: fractional (j, i) -> (lon, lat) on the rotated grid
# ---------------------------------------------------------------------------
def _ll(d, j, i):
    j, i = np.atleast_1d(np.asarray(j, float)), np.atleast_1d(np.asarray(i, float))
    x = ndi.map_coordinates(d['X'], [j, i], order=1, mode='nearest')
    y = ndi.map_coordinates(d['Y'], [j, i], order=1, mode='nearest')
    return x, y


def _corners(d, j0, j1, i0, i1):
    """Cell-corner lon / lat of the cells ``j0..j1-1``, ``i0..i1-1`` (for flat pcolormesh)."""
    jj, ii = np.meshgrid(np.arange(j0, j1 + 1) - 0.5, np.arange(i0, i1 + 1) - 0.5, indexing='ij')
    x, y = _ll(d, jj.ravel(), ii.ravel())
    return x.reshape(jj.shape), y.reshape(jj.shape)


def _box(ax, d, j0, i0, n, col, lw=1.8, ls='-'):
    jc = [j0 - 0.5, j0 - 0.5, j0 + n - 0.5, j0 + n - 0.5]
    ic = [i0 - 0.5, i0 + n - 0.5, i0 + n - 0.5, i0 - 0.5]
    x, y = _ll(d, jc, ic)
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fill=False, ec=col, lw=lw, ls=ls, zorder=6))


def _geo(ax, d, sl_j, sl_i, nbins=4):
    Y = d['Y'][sl_j, sl_i]
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y))))
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_fine))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_fine))
    ax.xaxis.set_major_locator(MaxNLocator(nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins))
    ax.tick_params(labelsize=9)


def _lon_fine(x, _):
    return f'{abs(x):.2f}°W' if x < 0 else f'{x:.2f}°E'


def _lat_fine(y, _):
    return f'{abs(y):.2f}°N'


def _crop(d, pad=PAD):
    jj, ii = np.where(d['band'])
    return (slice(max(jj.min() - pad, 0), jj.max() + pad + 1),
            slice(max(ii.min() - pad, 0), ii.max() + pad + 1))


def _cells(ax, d, field, sl_j, sl_i, mask=None, **kw):
    """Flat pcolormesh of ``field`` on the cells of the slices (``mask`` False -> blank)."""
    X, Y = _corners(d, sl_j.start, sl_j.stop, sl_i.start, sl_i.stop)
    a = field[sl_j, sl_i]
    if mask is not None:
        a = np.ma.masked_where(~mask[sl_j, sl_i], a)
    return ax.pcolormesh(X, Y, a, shading='flat', rasterized=True, **kw)


def _blank(ax, d, sl_j, sl_i, mask):
    """Cells outside ``mask`` in a flat light grey."""
    X, Y = _corners(d, sl_j.start, sl_j.stop, sl_i.start, sl_i.stop)
    a = np.ma.masked_where(mask[sl_j, sl_i], np.ones(mask[sl_j, sl_i].shape))
    ax.pcolormesh(X, Y, a, shading='flat', cmap=matplotlib.colors.ListedColormap([NOBAND]),
                  rasterized=True)


def _lasso(ax, d, sl_j, sl_i, col=COL_BAND, lw=1.4):
    ax.contour(d['X'][sl_j, sl_i], d['Y'][sl_j, sl_i], d['band'][sl_j, sl_i].astype(float),
               levels=[0.5], colors=[col], linewidths=lw, linestyles=[LS], zorder=5)


def _mark(ax, d, p, col=COL_CLOSE, s=60):
    x, y = _ll(d, *p)
    ax.scatter(x, y, s=s, marker='s', facecolor='none', edgecolor=col, lw=1.6, zorder=7)


def _log(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(np.where(a > 0, a, np.nan))


def _label(ax, s):
    ax.text(0.02, 0.98, s, transform=ax.transAxes, va='top', ha='left', fontsize=13,
            fontweight='bold', color='k',
            path_effects=[pe.withStroke(linewidth=3, foreground='w')], zorder=10)


def panel_front(fig, a, d, win, label='a'):
    """The whole front on midpoint ``G``: skeleton, lassoed band, window box (panel a of the
    method figures; shared with ``slides_2F``)."""
    band, skel = d['band'], d['skel']
    cj, ci = _crop(d)
    lg = _log(d['Gmid'])
    vG = np.nanpercentile(lg[cj, ci], [2, 99.5])
    m = _cells(a, d, lg, cj, ci, cmap=CMAP_G, vmin=vG[0], vmax=vG[1])
    xs, ys = d['X'][skel], d['Y'][skel]
    a.scatter(xs, ys, s=1.5, marker='s', c=COL_SKEL, lw=0, zorder=4)
    _lasso(a, d, cj, ci)
    _box(a, d, win[0], win[1], WIN, COL_WIN, lw=2.0)
    _geo(a, d, cj, ci)
    a.set_title(f'the front: label {d["label"]}, $G$ at {d["time_mid"][11:16]} UTC (pair {d["pair"]})',
                fontsize=11)
    cb = fig.colorbar(m, ax=a, fraction=0.045, pad=0.02)
    cb.set_label(r'$\log_{10} G$ [s$^{-4}$]', fontsize=10)
    a.legend(handles=[Line2D([], [], color=COL_SKEL, marker='s', ls='', ms=4,
                             label=f'front pixels ({int(skel.sum())})'),
                      Line2D([], [], color=COL_BAND, ls=':', lw=1.6,
                             label=f'parcels: front + 1 cell ({int(band.sum()):,})'),
                      Line2D([], [], color=COL_WIN, lw=2.0, label=f'window ({WIN}×{WIN} cells)')],
             loc='lower right', fontsize=8.5, framealpha=0.85)
    _label(a, label)


# ---------------------------------------------------------------------------
# figure 1: the method
# ---------------------------------------------------------------------------
def _edge_labels(ax, cs, centre, fs, keep=(0.66, 0.80)):
    """Contour labels placed as far from ``centre`` (lon, lat) as possible, inside the
    central ``keep`` = (x, y) fraction of the axes, the x margin being about half a label's
    width so no label is clipped (DG prompt 4: away from the centre pixel)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    mx, my = 0.5 * (1 - keep[0]) * (x1 - x0), 0.5 * (1 - keep[1]) * (y1 - y0)
    pos = []
    for segs in cs.allsegs:
        best, dbest = None, -1.0
        for seg in segs:
            if len(seg) < 2:
                continue
            arc = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
            t = np.linspace(0, arc[-1], 200)           # densify: the contours have few vertices
            for x, y in zip(np.interp(t, arc, seg[:, 0]), np.interp(t, arc, seg[:, 1])):
                if not (x0 + mx < x < x1 - mx and y0 + my < y < y1 - my):
                    continue
                dist = np.hypot((x - centre[0]) / (x1 - x0), (y - centre[1]) / (y1 - y0))
                if dist > dbest:
                    best, dbest = (x, y), dist
        if best is not None:
            pos.append(best)
    ax.clabel(cs, manual=pos, inline=True, fontsize=fs, fmt='%.1f')


def slide_front(a, d, win, small):
    """Panel (a) of the slide-sized method figures (DG prompt 4; shared with ``slides_2F``):
    midpoint ``G``, the front pixels, the lassoed band and the window; latitude ticks only.
    Returns the mappable for the caller's colour bar."""
    skel = d['skel']
    cj, ci = _crop(d)
    lg = _log(d['Gmid'])
    vG = np.nanpercentile(lg[cj, ci], [2, 99.5])
    m = _cells(a, d, lg, cj, ci, cmap=CMAP_G, vmin=vG[0], vmax=vG[1])
    a.scatter(d['X'][skel], d['Y'][skel], s=0.6, marker='s', c=COL_SKEL, lw=0, zorder=4)
    _lasso(a, d, cj, ci, lw=1.0)
    _box(a, d, win[0], win[1], WIN, COL_WIN, lw=1.8)
    _geo(a, d, cj, ci, nbins=2)
    a.tick_params(labelsize=small)
    a.set_xticks([])                      # longitude would collide with (d)'s title
    return m


def slide_legend_handles():
    """The legend entries common to the slide-sized method figures."""
    return [Line2D([], [], color=COL_SKEL, marker='s', ls='', ms=5, label='front pixels'),
            Line2D([], [], color=COL_BAND, ls=':', lw=1.8, label='parcels (front + 1 cell)'),
            Line2D([], [], color=COL_WIN, lw=2.0, label='window (d–f)'),
            Line2D([], [], color=COL_CLOSE, lw=1.4, ls='--', label='close-up (b)')]


def fig_method(d, win, p, out_png, fs=FS_DG):
    """The method on one 16:9 slide under its own title (DG prompt 4): no suptitle or
    footnote, ``SLIDE_SIZE`` inches at ``METHOD_DPI``, axes placed in inches."""
    band, skel = d['band'], d['skel']
    cj, ci = _crop(d)
    wj, wi = slice(win[0], win[0] + WIN), slice(win[1], win[1] + WIN)
    h = CLOSE // 2
    kj, ki = slice(p[0] - h, p[0] + h + 1), slice(p[1] - h, p[1] + h + 1)
    t0, t1 = d['times'][0][11:16], d['times'][1][11:16]
    W, H = SLIDE_SIZE
    small = fs - 2

    with plt.rc_context({'font.size': small, 'xtick.labelsize': small,
                         'ytick.labelsize': small}):
        fig = plt.figure(figsize=SLIDE_SIZE)

        def axes_in(x, y, w, hh):
            return fig.add_axes([x / W, y / H, w / W, hh / H])

        def title(ax, text):
            ax.set_title(text, fontsize=fs, pad=4)

        def cbar(m, x, y, hh, label, **kw):
            cb = fig.colorbar(m, cax=axes_in(x, y, 0.09, hh), **kw)
            cb.set_label(label, fontsize=small, labelpad=2)
            cb.ax.tick_params(labelsize=small)
            return cb

        r1, r2, hh = 2.42, 0.62, 1.55
        a = axes_in(0.72, r1, 2.05, hh)
        b = axes_in(3.72, r1, 1.85, hh)
        c = axes_in(6.45, r1, 2.05, hh)
        dd = axes_in(0.72, r2, 2.05, hh - 0.05)
        e = axes_in(3.72, r2, 1.85, hh - 0.05)
        f = axes_in(6.45, r2, 2.05, hh - 0.05)

        # (a) the front: midpoint G, the band, the window
        m = slide_front(a, d, win, small)
        title(a, f'(a) Label {d["label"]}, $G$ at {d["time_mid"][11:16]}')
        cbar(m, 2.82, r1 + 0.1, hh - 0.2, r'$\log_{10} G$')

        # (b) one parcel close up: b(h36), the arrival cell, x_d and the stencil
        bb = d['b0'][kj, ki]
        _cells(b, d, d['b0'], kj, ki, cmap=CMAP_B, alpha=0.85,
               vmin=bb.min() - 0.15 * np.ptp(bb), vmax=bb.max() + 0.15 * np.ptp(bb))
        Xc, Yc = _corners(d, kj.start, kj.stop, ki.start, ki.stop)
        for k in range(Xc.shape[0]):
            b.plot(Xc[k], Yc[k], color='0.45', lw=0.5, zorder=2)
        for k in range(Xc.shape[1]):
            b.plot(Xc[:, k], Yc[:, k], color='0.45', lw=0.5, zorder=2)
        Xa, Ya = _corners(d, p[0], p[0] + 1, p[1], p[1] + 1)
        b.add_patch(Polygon(np.c_[Xa[[0, 0, 1, 1], [0, 1, 1, 0]], Ya[[0, 0, 1, 1], [0, 1, 1, 0]]],
                            closed=True, fill=False, ec='k', lw=2.2, zorder=6))
        jd, id_ = p[0] - d['dj'][p], p[1] - d['di'][p]
        xa, ya = _ll(d, *p)
        xd, yd = _ll(d, jd, id_)
        sj = np.array([jd, jd, jd, jd + 1, jd - 1])
        si = np.array([id_, id_ + 1, id_ - 1, id_, id_])
        xs_, ys_ = _ll(d, sj, si)
        b.plot(xs_[[2, 0, 1]], ys_[[2, 0, 1]], color=COL_DEP, lw=1.2, ls='--', zorder=7)
        b.plot(xs_[[4, 0, 3]], ys_[[4, 0, 3]], color=COL_DEP, lw=1.2, ls='--', zorder=7)
        b.scatter(xs_[1:], ys_[1:], s=40, facecolor='w', edgecolor=COL_DEP, lw=1.5, zorder=8)
        b.annotate('', xy=(xa[0], ya[0]), xytext=(xd[0], yd[0]), zorder=9,
                   arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2, mutation_scale=10,
                                   shrinkA=0, shrinkB=0))
        b.scatter(xd, yd, s=20, c=COL_DEP, edgecolor='k', lw=0.6, zorder=10)
        b.scatter(xa, ya, s=20, marker='o', facecolor='w', edgecolor='k', lw=1.1, zorder=10)
        _geo(b, d, kj, ki, nbins=3)
        lo, la = _corners(d, kj.start, kj.stop, ki.start, ki.stop)
        b.set_xlim(lo.min(), lo.max())
        b.set_ylim(la.min(), la.max())
        kj2, ki2 = slice(kj.start - 1, kj.stop + 1), slice(ki.start - 1, ki.stop + 1)
        cs = b.contour(d['X'][kj2, ki2], d['Y'][kj2, ki2], d['b0'][kj2, ki2] * 1e3,
                       levels=B_LEVELS, colors='k', linewidths=0.8, zorder=3)
        _edge_labels(b, cs, (xa[0], ya[0]), fs=small - 1)
        b.set_xticks([]); b.set_yticks([])
        title(b, f'(b) One parcel: $b$ at {t0}')

        # (c) every parcel: the rate on the band, its mean
        rv = d['DG'][band] * SCALE
        vr = float(np.nanpercentile(np.abs(rv), 95))
        _blank(c, d, cj, ci, band)
        m = _cells(c, d, d['DG'] * SCALE, cj, ci, mask=band, cmap=CMAP_R,
                   norm=TwoSlopeNorm(0.0, -vr, vr))
        _box(c, d, win[0], win[1], WIN, COL_WIN, lw=1.8)
        _geo(c, d, cj, ci, nbins=2)
        c.set_xticks([]); c.set_yticks([])
        title(c, r'(c) $DG/Dt$, every parcel')
        cbar(m, 8.55, r1 + 0.1, hh - 0.2, r'$DG/Dt$ [10$^{-19}$ s$^{-5}$]', extend='both')
        c.text(0.97, 0.03, f'mean {rv.mean():+.2f}', transform=c.transAxes, ha='right',
               va='bottom', fontsize=fs, fontweight='bold',
               bbox=dict(fc='w', ec='k', lw=0.8, alpha=0.9, pad=2), zorder=10)

        # (d)-(f) the window
        lgd, lg1 = _log(d['Gd']), _log(d['G1'])
        vw = np.nanpercentile(np.r_[lgd[wj, wi][band[wj, wi]], lg1[wj, wi][band[wj, wi]]], [1, 99])
        for ax, fld in ((dd, lgd), (e, lg1)):
            _blank(ax, d, wj, wi, band)
            m = _cells(ax, d, fld, wj, wi, mask=band, cmap=CMAP_G, vmin=vw[0], vmax=vw[1])
        title(dd, rf'(d) Interpolated $G(x_d,\,{t0})$')
        title(e, rf'(e) $G(x,\,{t1})$')
        cbar(m, 5.62, r2 + 0.1, hh - 0.25, r'$\log_{10} G$')
        _blank(f, d, wj, wi, band)
        vwr = float(np.nanpercentile(np.abs(d['DG'][wj, wi][band[wj, wi]]) * SCALE, 95))
        m = _cells(f, d, d['DG'] * SCALE, wj, wi, mask=band, cmap=CMAP_R,
                   norm=TwoSlopeNorm(0.0, -vwr, vwr))
        cbar(m, 8.55, r2 + 0.1, hh - 0.25, r'$DG/Dt$ [10$^{-19}$ s$^{-5}$]', extend='both')
        title(f, r'(f) $DG/Dt$ = (e $-$ d) / 1 h')
        for ax in (dd, e, f):
            _box(ax, d, kj.start, ki.start, CLOSE, COL_CLOSE, lw=1.4, ls='--')
            _mark(ax, d, p, s=30)
            _geo(ax, d, wj, wi, nbins=3)
            ax.set_xticks([]); ax.set_yticks([])
        wm = d['DG'][wj, wi][band[wj, wi]] * SCALE
        f.text(0.97, 0.03, f'mean {wm.mean():+.1f}', transform=f.transAxes, ha='right',
               va='bottom', fontsize=small, bbox=dict(fc='w', ec='0.5', lw=0.6, alpha=0.9, pad=2),
               zorder=10)

        handles = slide_legend_handles() + [
                   Line2D([], [], color='k', marker='o', mfc='w', ls='', ms=6,
                          label=f'$x$: arrival cell, {t1}'),
                   Line2D([], [], color=COL_DEP, marker='o', mec='k', ls='', ms=6,
                          label=f'$x_d$: departure point, {t0}'),
                   Line2D([], [], color=COL_DEP, marker='o', mfc='w', ls='--', ms=6,
                          label=r'stencil for $|\nabla b|^2$ at $x_d$'),
                   Line2D([], [], color='k', lw=0.8, label=r'$b$ [10$^{-3}$ m s$^{-2}$]')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4,
                   frameon=False, fontsize=FS_DG_LEGEND, handlelength=1.5, handletextpad=0.4,
                   columnspacing=1.2)
        out_png = pathlib.Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METHOD_DPI)
        plt.close(fig)
    g1, gd = d['G1'][p], d['Gd'][p]
    return dict(png=str(out_png), size_in=list(SLIDE_SIZE), dpi=METHOD_DPI,
                window_mean=float(wm.mean()), window_n=int(wm.size),
                parcel=dict(j=p[0], i=p[1], G_x_h1=float(g1), G_xd_h0=float(gd),
                            rate=float(d['DG'][p]), disp_cells=float(d['disp'][p]),
                            di=float(d['di'][p]), dj=float(d['dj'][p])),
                rate_limit=vr, window_rate_limit=vwr)

# ---------------------------------------------------------------------------
# figure 2: local + advective = semi-Lagrangian
# ---------------------------------------------------------------------------
def fig_split(d, win, out_png, parts=None, suptitle=None, footer=None):
    """``parts[0] + parts[1] = parts[2]`` as maps, every parcel on top and the window
    below, each row on its own colour scale (+-p95 of the two parts).  ``parts`` is
    ``((key, field, title), ...)``; ``suptitle`` / ``footer`` are functions of the stats
    (defaults: the semi-Lagrangian local + advective split; ``slides_2F`` passes its own)."""
    band = d['band']
    cj, ci = _crop(d)
    wj, wi = slice(win[0], win[0] + WIN), slice(win[1], win[1] + WIN)
    t0, t1 = d['times'][0][11:16], d['times'][1][11:16]
    if parts is None:
        parts = (('local', d['local'], rf'local: $[G(x,{t1}) - G(x,{t0})]$ / 1 h'),
                 ('advect', d['advect'], rf'advective: $[G(x,{t0}) - G(x_d,{t0})]$ / 1 h'),
                 ('total', d['DG'], rf'semi-Lagrangian: $[G(x,{t1}) - G(x_d,{t0})]$ / 1 h'))
    def _lim(sel):
        return float(np.nanpercentile(np.abs(np.r_[parts[0][1][sel], parts[1][1][sel]]) * SCALE, 95))
    wband = np.zeros_like(band)
    wband[wj, wi] = band[wj, wi]
    vr, vwr = _lim(band), _lim(wband)
    norms = (TwoSlopeNorm(0.0, -vr, vr), TwoSlopeNorm(0.0, -vwr, vwr))
    stats = {}
    fig, axs = plt.subplots(2, 3, figsize=(17.0, 10.6))
    fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.09, wspace=0.28, hspace=0.18)
    ms = [None, None]
    for col, (key, fld, ttl) in enumerate(parts):
        v = fld[band] * SCALE
        vw = fld[wband] * SCALE
        stats[key] = dict(band_mean=float(v.mean()), band_rms=float(np.sqrt(np.mean(v ** 2))),
                          window_mean=float(vw.mean()), window_rms=float(np.sqrt(np.mean(vw ** 2))))
        for row, (sj, si) in enumerate(((cj, ci), (wj, wi))):
            ax = axs[row, col]
            _blank(ax, d, sj, si, band)
            ms[row] = _cells(ax, d, fld * SCALE, sj, si, mask=band, cmap=CMAP_R, norm=norms[row])
            _geo(ax, d, sj, si, nbins=4 if row == 0 else 3)
            s = stats[key]
            if row == 0:
                _box(ax, d, win[0], win[1], WIN, COL_WIN, lw=2.0)
                ax.set_title(ttl, fontsize=11.5)
                txt = f'mean {s["band_mean"]:+.2f}\nr.m.s. {s["band_rms"]:.1f}'
            else:
                txt = f'window mean {s["window_mean"]:+.1f}\nr.m.s. {s["window_rms"]:.1f}'
            ax.text(0.97, 0.04, txt, transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=11 if row == 0 else 10, fontweight='bold' if row == 0 else 'normal',
                    bbox=dict(fc='w', ec='k', lw=0.6, alpha=0.9), zorder=10)
            _label(ax, 'abcdef'[3 * row + col])
    for row, sym in ((0, ('+', '=')), (1, ('+', '='))):
        for ax, s_ in zip(axs[row, :2], sym):
            ax.text(1.07, 0.5, s_, transform=ax.transAxes, fontsize=30, fontweight='bold',
                    ha='center', va='center')
    for row, lab_ in ((0, 'every parcel'), (1, 'window')):
        pos = axs[row, 2].get_position()
        cax = fig.add_axes([0.915, pos.y0, 0.012, pos.height])
        cb = fig.colorbar(ms[row], cax=cax, extend='both')
        cb.set_label(f'rate, {lab_} [10$^{{-19}}$ s$^{{-5}}$]', fontsize=10.5)
    s = stats
    if suptitle is not None:
        fig.suptitle(suptitle(s), fontsize=14, y=0.985)
        fig.text(0.5, 0.012, footer(s), ha='center', fontsize=10.5, color='0.25')
        _save(fig, out_png)
        return dict(png=str(out_png), rate_limit=vr, window_rate_limit=vwr, **stats)
    fig.suptitle(f'local + advective = semi-Lagrangian $DG/Dt$ on the {int(band.sum()):,} parcels '
                 f'of label {d["label"]} ({d["times"][0][:10]} {t0} → {t1} UTC):   '
                 f'{s["local"]["band_mean"]:+.2f} {s["advect"]["band_mean"]:+.2f} = '
                 f'{s["total"]["band_mean"]:+.2f}', fontsize=14, y=0.985)
    fig.text(0.5, 0.012, 'Each part is larger than their sum, cell by cell (r.m.s. '
             f'{s["local"]["band_rms"]:.0f} and {s["advect"]["band_rms"]:.0f} against '
             f'{s["total"]["band_rms"]:.0f}): the front moves, and most of the local change is '
             'that motion.  Following the water removes it.\n'
             'Top: every parcel; bottom: the window (its own colour scale).  '
             'Units: 10$^{-19}$ s$^{-5}$.',
             ha='center', fontsize=10.5, color='0.25')
    _save(fig, out_png)
    return dict(png=str(out_png), rate_limit=vr, window_rate_limit=vwr, **stats)


def _save(fig, out_png):
    out_png = pathlib.Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pair', type=int, default=PAIR)
    p.add_argument('--window', type=str, default=None, help='j0,i0 of the window (default: auto)')
    p.add_argument('--parcel', type=str, default=None, help='j,i of the close-up parcel (default: auto)')
    p.add_argument('--out-dir', default=str(SLIDE_DIR))
    a = p.parse_args(argv)
    d = compute(a.pair)
    chk = check(d)
    win = tuple(int(v) for v in a.window.split(',')) if a.window else choose_window(d)
    par = tuple(int(v) for v in a.parcel.split(',')) if a.parcel else choose_parcel(d, win)
    out = pathlib.Path(a.out_dir)
    band = d['band']
    res = dict(pair=a.pair, label=d['label'], times=d['times'], time_mid=d['time_mid'],
               check=chk, window=dict(j0=win[0], i0=win[1], size=WIN), parcel_ji=list(par),
               disp_cells=dict(zip(('p50', 'p90', 'max'),
                                   [float(v) for v in np.percentile(d['disp'][band], [50, 90, 100])])),
               G_band_mean=dict(Gd_h0=float(d['Gd'][band].mean()), G_h0=float(d['G0'][band].mean()),
                                G_h1=float(d['G1'][band].mean())))
    res['method'] = fig_method(d, win, par, out / 'dGdt_method.png')
    res['split'] = fig_split(d, win, out / 'dGdt_split.png')
    path = SUMMARY.with_name(SUMMARY.name.format(pair=a.pair))
    path.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    main()
