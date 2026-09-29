""" M1 validation figures (coding doc §4.9): the four gates V1-V4 and the
two supporting figures V5-V6.  Each function writes one PNG to
``dev/frontogenesis/figs/`` and returns a dict of the numbers.

Written so far (M1 task 1): ``qa_land_halo`` -> **V6**.  The other
V-functions arrive with the modules they exercise (tasks 3, 5, 6).

Maps are drawn with ``pcolormesh(XC, YC, ...)`` so they come out north-up,
east-right on the rotated face 10 (``i`` runs south, ``j`` runs east).
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm   # noqa: E402
from matplotlib.patches import Rectangle                     # noqa: E402
from scipy import ndimage                                    # noqa: E402

import osn_tiles as ot                                       # noqa: E402
import masking as mk                                         # noqa: E402
import m0_qa_checks as qc                                    # noqa: E402
from m0_qa_plot import draw_map, edge_profile                # noqa: E402
from m1_write_masks import gulf_of_california_check          # noqa: E402
from dbof.utils import native_gradient as ng                 # noqa: E402
from dbof.preprocessing.calculate_fields import buoyancy_of_field  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / 'figs'
RAW = 'tile330_raw_20120702T00_2h.zarr'
SNAPSHOT = '2012-07-02T00:00:00'

# V6 classes: value -> (label, colour)
V6_CLASSES = [
    (0, 'land (hFacC = 0)', '#b0b0b0'),
    (1, 'retained: mask_analysis', '#e6f0fa'),
    (2, 'removed by the land halo', '#c8102e'),
    (3, 'removed by the offshore cut only', '#f4a259'),
    (4, 'removed by the tile-edge margin only', '#5b2a86'),
]


def _snapshot_fields(grid_ds):
    """Grid + hour t0 merged on ``(face, j, i)`` in float64, the xgcm grid,
    ``b`` (JMD95) and ``G = b_x^2 + b_y^2`` from the component stencil M1
    adopts (§1.1), positional ``(j, i)``.  Needs the 2 h raw store."""
    g = grid_ds if 'face' in grid_ds.dims else grid_ds.expand_dims('face')
    raw = xr.open_zarr(ot.DATA_DIR / RAW).load()
    hour = raw.sel(time=SNAPSHOT).expand_dims('face')
    # the hour and the grid share XC/YC and the index coords; 'override'
    # keeps the first and silences xarray's compat FutureWarning
    ds = xr.merge([hour, g], compat='override', combine_attrs='override').astype('float64')
    grid = ot.build_xgcm(g)
    b = buoyancy_of_field(ds).compute()
    bx, by = ng.calculate_native_gradient_tracer(b, ds, grid)
    G = (bx ** 2 + by ** 2).compute()
    if G.dims != ('face', 'j', 'i'):
        raise AssertionError(f'G dims {G.dims}')     # §8: assert dims after every dbof call
    return ds, grid, b, qc._np(G)


def _classify(masks):
    """V6 class map from the §3.5 masks, plus the counts."""
    oc, halo = masks['mask_ocean'].values, masks['mask_halo'].values
    off, edge = masks['mask_offshore'].values, masks['mask_edge'].values
    ana = masks['mask_analysis'].values
    cls = np.zeros(oc.shape, dtype=int)
    cls[ana] = 1
    cls[oc & ~halo] = 2
    cls[oc & halo & ~off & edge] = 3
    cls[oc & halo & off & ~edge] = 4
    # ocean cells inside the halo AND the edge margin, or outside both cuts,
    # are drawn with the margin colour so the rim stays visible
    cls[oc & halo & ~off & ~edge] = 4
    counts = dict(n_cells=int(oc.size), n_ocean=int(oc.sum()), n_halo=int(halo.sum()),
                  n_offshore=int(off.sum()), n_edge_ocean=int((edge & oc).sum()),
                  n_analysis=int(ana.sum()),
                  n_removed_halo=int((oc & ~halo).sum()),
                  n_removed_offshore_only=int((oc & halo & ~off & edge).sum()),
                  n_removed_edge_only=int((oc & halo & off & ~edge).sum()))
    return cls, counts


def qa_land_halo(grid_ds, png: bool = True) -> dict:
    """**V6**: the coastline before/after the land halo, the offshore cut,
    the Gulf of California, the finite tile-edge rim and the ``edge_cells``
    margin that removes it.  ``@pytest.mark.needs_grid`` (real grid + the
    2 h raw store for the rim panel).  Returns the key counts."""
    masks = mk.build_masks(grid_ds)
    a = masks.attrs
    halo_km, halo_cells, edge_cells = a['halo_km'], int(a['halo_cells']), int(a['edge_cells'])
    oc = masks['mask_ocean'].values
    halo = masks['mask_halo'].values
    dist = masks['coast_distance_km'].values
    cls, counts = _classify(masks)
    X, Y = grid_ds.XC.squeeze().values, grid_ds.YC.squeeze().values
    taxi = ndimage.distance_transform_cdt(oc, metric='taxicab')
    chess = ndimage.distance_transform_cdt(oc, metric='chessboard')
    # the tile-edge rim, measured (crop test) on the real hour, and G itself
    ds, grid, b, G = _snapshot_fields(grid_ds)
    rim = qc.check_edge_rim(ds, grid, b)
    rim_max = max(max(v) for r in ('G_components', 'jacobian')
                  for k, v in rim[r].items() if k in ('low_j', 'high_j', 'low_i', 'high_i') and v)
    gulf = gulf_of_california_check(grid_ds, masks)
    res = dict(**counts, halo_cells=halo_cells, halo_km=float(halo_km),
               dxC_median_km=float(a['dxC_median_km']), edge_cells=edge_cells,
               offshore_km=float(a['offshore_km']),
               halo_min_taxicab_retained=int(taxi[halo].min()),
               halo_min_chessboard_retained=int(chess[halo].min()),
               halo_max_taxicab_excluded=int(taxi[oc & ~halo].max()),
               edge_rim_offsets={r: {k: rim[r][k] for k in ('low_j', 'high_j', 'low_i', 'high_i')}
                                 for r in ('G_components', 'jacobian')},
               edge_rim_max_offset=int(rim_max), edge_margin_covers_rim=bool(rim_max < edge_cells),
               gulf=gulf, png=None)
    if not png:
        return res

    cmap_cls = ListedColormap([c for _, _, c in V6_CLASSES])
    norm_cls = BoundaryNorm(np.arange(-0.5, len(V6_CLASSES)), cmap_cls.N)
    fig, axs = plt.subplots(2, 3, figsize=(19, 11.5))
    (pa, pb, pc), (pd, pe, pf) = axs

    # (a) whole tile: every stage of the mask
    draw_map(pa, X, Y, cls.astype(float), cmap_cls, norm=norm_cls)
    handles = [Rectangle((0, 0), 1, 1, color=col) for _, _, col in V6_CLASSES]
    labels = [lab for _, lab, _ in V6_CLASSES]
    labels[1] += f'  (n = {counts["n_analysis"]:,})'
    labels[2] += f'  (n = {counts["n_removed_halo"]:,})'
    labels[3] += f'  (n = {counts["n_removed_offshore_only"]:,})'
    labels[4] += f'  (n = {counts["n_removed_edge_only"]:,})'
    pa.legend(handles, labels, loc='lower left', fontsize=7.5, framealpha=0.95)
    pa.set_title(f'(a) mask stages: ocean {counts["n_ocean"]:,} -> halo {counts["n_halo"]:,} -> '
                 f'offshore {counts["n_offshore"]:,}\n-> & edge margin = analysis '
                 f'{counts["n_analysis"]:,} ({100 * counts["n_analysis"] / counts["n_ocean"]:.1f}% '
                 'of the ocean)', fontsize=10)

    # (b) cell-level inset, Monterey Bay: the coastline before / after the halo
    j0, i0, w = 287, 96, 24
    sl = (slice(j0 - w, j0 + w), slice(i0 - w, i0 + w))
    pb.pcolormesh(X[sl], Y[sl], cls[sl].astype(float), cmap=cmap_cls, norm=norm_cls,
                  shading='nearest', edgecolors='#555555', linewidth=0.15)
    # the coastline 'before' is the grey/red boundary (hFacC); 'after' is
    # the halo_km contour of the distance field
    pb.contour(X[sl], Y[sl], np.nan_to_num(dist[sl], nan=-1.0), levels=[halo_km],
               colors='black', linewidths=1.2)
    pb.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y[sl]))))
    pb.set_xlabel('longitude'); pb.set_ylabel('latitude')
    pb.set_title(f'(b) Monterey Bay, cell edges drawn: coastline before the halo (grey/red, hFacC)\n'
                 f'and after (black: coast distance = {halo_km:.2f} km = {halo_cells} x median dxC '
                 f'{a["dxC_median_km"]:.3f} km)', fontsize=10)
    pa.add_patch(Rectangle((X[sl].min(), Y[sl].min()), np.ptp(X[sl]), np.ptp(Y[sl]),
                           fill=False, ec='black', lw=1.2))

    # (c) the distance field with the two thresholds
    m = draw_map(pc, X, Y, dist, 'viridis', land=~oc, vmin=0, vmax=300)
    fig.colorbar(m, ax=pc, shrink=0.85, label='coast_distance_km  (clipped at 300)')
    pc.contour(X, Y, np.nan_to_num(dist, nan=-1.0), levels=[a['offshore_km']], colors='white',
               linewidths=1.2)
    pc.contour(X, Y, np.nan_to_num(dist, nan=-1.0), levels=[halo_km], colors='#c8102e',
               linewidths=0.6)
    pc.set_title(f'(c) distance to land (skfmm, mean spacing {a["dyC_mean_km"]:.3f} x '
                 f'{a["dxC_mean_km"]:.3f} km)\nwhite: {a["offshore_km"]:g} km offshore cut; red: the '
                 f'{halo_km:.1f} km halo', fontsize=10)

    # (d) Gulf of California
    gsl = (slice(540, 720), slice(330, 620))
    m = pd.pcolormesh(X[gsl], Y[gsl], np.ma.masked_invalid(dist[gsl]), cmap='viridis',
                      vmin=0, vmax=100, shading='nearest')
    pd.pcolormesh(X[gsl], Y[gsl], np.ma.masked_where(oc[gsl], np.ones_like(X[gsl])),
                  cmap=ListedColormap(['#b0b0b0']), shading='nearest')
    ana = masks['mask_analysis'].values
    pd.pcolormesh(X[gsl], Y[gsl], np.ma.masked_where(~ana[gsl], np.ones_like(X[gsl])),
                  cmap=ListedColormap(['#e6f0fa']), shading='nearest')
    pd.contour(X[gsl], Y[gsl], np.nan_to_num(dist[gsl], nan=-1.0), levels=[halo_km],
               colors='#c8102e', linewidths=0.8)
    fig.colorbar(m, ax=pd, shrink=0.85, label='coast_distance_km')
    pd.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y[gsl]))))
    pd.set_xlabel('longitude'); pd.set_ylabel('latitude')
    pd.set_title(f'(d) Gulf of California: its own ocean component ({gulf["n_gulf"]:,} cells), '
                 f'clipped by the\neast edge at {gulf["gulf_lon"][1]:.2f}E; max coast distance '
                 f'{gulf["gulf_max_coast_km"]:.1f} km < {a["offshore_km"]:g}: cells >= '
                 f'{a["offshore_km"]:g} km: {gulf["gulf_n_ge_offshore"]},\nin mask_analysis: '
                 f'{gulf["gulf_n_analysis"]} (pale = retained Pacific)', fontsize=10)
    pa.add_patch(Rectangle((X[gsl].min(), Y[gsl].min()), np.ptp(X[gsl]), np.ptp(Y[gsl]),
                           fill=False, ec='black', lw=1.2, ls='--'))

    # (e) the finite tile-edge rim and the margin that removes it
    prof = edge_profile(G, oc, nmax=12)
    cols = {'low j (j=0)': '#1f5fa8', 'high j (j=719)': '#c8102e',
            'low i (i=2880)': '#2a9d8f', 'high i (i=3599)': '#f4a259'}
    k = np.arange(12)
    for name, col in cols.items():
        pe.plot(k, prof[name], 'o-', color=col, lw=1.5, ms=4, label=f'G, {name} edge')
    pe.axvspan(-0.5, edge_cells - 0.5, color='#5b2a86', alpha=0.15,
               label=f'edge margin: edge_cells = {edge_cells} (mask_edge False)')
    offs = {r: sorted({o for kk in ('low_j', 'high_j', 'low_i', 'high_i') for o in rim[r][kk]})
            for r in ('G_components', 'jacobian')}
    pe.axvline(rim_max + 0.5, color='black', lw=1.2, ls=':',
               label=f'crop test: G changed at offsets {offs["G_components"]}, '
                     f'Jacobian at {offs["jacobian"]}')
    pe.set_yscale('log'); pe.set_xticks(k); pe.set_xlim(-0.5, 11.5)
    pe.set_xlabel('offset from tile edge [cells]')
    pe.set_ylabel('median |G| / median over offsets 1-3')
    pe.set_title('(e) tile-edge rim: finite, not NaN (xgcm padding = 0). Low edges ~1e6x (diff '
                 'against 0),\nhigh edges ~0.5x (interp with 0); Jacobian 1-2 cells deep; '
                 f'edge_cells = {edge_cells} margin shaded', fontsize=10)
    pe.legend(fontsize=7, loc='upper right'); pe.grid(alpha=0.3)

    # (f) halo width in cells: index distance to land, excluded vs retained
    bins = np.arange(0.5, 16.5)
    pf.hist(taxi[oc & ~halo], bins=bins, color='#c8102e', alpha=0.8,
            label=f'ocean removed by the halo (n = {counts["n_removed_halo"]:,})')
    pf.hist(taxi[halo & (taxi <= 15)], bins=bins, color='#1f5fa8', alpha=0.8,
            label='retained (taxicab distance <= 15 shown)')
    pf.axvline(halo_cells, color='black', lw=1.5, ls='--', label=f'halo_cells = {halo_cells}')
    pf.set_xlabel('taxicab index distance to the nearest land cell [cells]')
    pf.set_ylabel('cells')
    pf.set_title(f'(f) halo width: {halo_km:.2f} km = {halo_cells} cells of dxC (meridional, i) = '
                 f'{halo_km / a["dyC_median_km"]:.2f} of dyC (zonal, j)\nretained: min taxicab '
                 f'{res["halo_min_taxicab_retained"]}, min chessboard {res["halo_min_chessboard_retained"]}; '
                 f'removed ocean: max taxicab {res["halo_max_taxicab_excluded"]}', fontsize=10)
    pf.legend(fontsize=8); pf.grid(alpha=0.3, axis='y')

    fig.suptitle(f'V6: land halo, offshore cut, Gulf of California and tile-edge margin; tile 330 '
                 f'(face 10), rim measured at {SNAPSHOT}; maps north-up via (XC, YC)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / 'V6_land_halo_tile330.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    res['png'] = str(out)
    return res


if __name__ == '__main__':
    import pprint
    pprint.pprint(qa_land_halo(ot.open_grid(with_face=True)))
