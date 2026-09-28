""" M0 task 5: QA plot of one snapshot (Theta, G = |grad b|^2, land mask).

Reads the two M0 stores from disk (no network), computes b with the
project's EOS path (``calculate_fields.buoyancy_of_field``, JMD95 at p=0)
and G with the dbof stencils M1 will wrap, and draws:

  (a) Theta            (b) log10 G            (c) validity map, whole tile
  (d) cell-level inset on the Monterey Bay coast
  (e) ribbon test: median G vs distance to land
  (f) tile-edge test: G and the Jacobian vs distance from each edge

plus the two trap checks from ``m0_qa_checks`` printed to stdout.  No halo
(M1), no physics module.  Run:

    /Users/xavier/miniforge3/envs/frontogenesis/bin/python m0_qa_plot.py

Orientation: face 10 has CS=0, SN=-1 -- ``i`` runs south, ``j`` runs east.
Maps are drawn with ``pcolormesh(XC, YC, ...)`` so they come out north-up,
east-right whatever the array layout; the inset is the same, with cell
edges drawn.
"""

import pprint
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import osn_tiles as ot                                   # noqa: E402
import m0_qa_checks as qc                                # noqa: E402
from dbof.utils import native_gradient as ng             # noqa: E402
from dbof.preprocessing.calculate_fields import buoyancy_of_field  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / 'figs'
RAW = 'tile330_raw_20120702T00_2h.zarr'
SNAPSHOT = '2012-07-02T00:00:00'

# validity classes for panels (c)/(d): value -> (label, colour)
CLASSES = [
    (0, 'land (hFacC = 0)', '#b0b0b0'),
    (1, 'ocean, G and Jacobian valid', '#e6f0fa'),
    (2, 'stencil NaN rim: G undefined (taxicab d = 1)', '#c8102e'),
    (3, 'Jacobian-only NaN rim (taxicab d = 2)', '#f4a259'),
    (4, 'tile-edge rim: finite but invalid (fill = 0)', '#5b2a86'),
    (5, 'isfinite(Theta) != (hFacC > 0)', '#000000'),
]


def load_snapshot():
    """Grid + hour t0 merged on (face, j, i), float64, with an xgcm grid."""
    g = ot.open_grid(with_face=True)
    raw = xr.open_zarr(ot.DATA_DIR / RAW).load()
    hour = raw.sel(time=SNAPSHOT).drop_vars(['XC', 'YC']).expand_dims('face')
    # XC/YC are data vars in the grid store (§3.1) and coords in the raw
    # store (§3.2); dropping them from the hour avoids the MergeError
    ds = xr.merge([hour, g]).astype('float64')
    return ds, ot.build_xgcm(g)


def compute_fields(ds, grid):
    """b, both G stencils, and the Jacobian union, as positional (j, i)."""
    b = buoyancy_of_field(ds).compute()
    G_sq = ng.calculate_grad_squared_tracer(b, ds, grid).compute()  # repo's gradb2
    bx, by = ng.calculate_native_gradient_tracer(b, ds, grid)
    G_comp = (bx ** 2 + by ** 2).compute()
    J = [x.compute() for x in ng.calculate_jacobian(ds.U, ds.V, ds, grid)]
    return dict(b=b, G_sq=qc._np(G_sq), G_comp=qc._np(G_comp),
                J=[qc._np(x) for x in J])


def classify(ds, f, edge_rim):
    """Validity map (j, i) with the CLASSES codes, plus the counts."""
    oc = ds.hFacC.values[0] > 0
    theta_ok = np.isfinite(ds.Theta.values[0])
    d_taxi = ndimage.distance_transform_cdt(oc, metric='taxicab')
    rim_G = oc & ~np.isfinite(f['G_sq'])
    rim_J = oc & ~np.isfinite(f['J'][0]) & ~rim_G
    cls = np.where(oc, 1, 0)
    cls[rim_G] = 2
    cls[rim_J] = 3
    n = cls.shape[0]
    e = edge_rim['jacobian']                     # the wider of the two rims
    edge = np.zeros_like(oc)
    for r in e['low_j']:
        edge[r, :] = True
    for r in e['high_j']:
        edge[n - 1 - r, :] = True
    for c in e['low_i']:
        edge[:, c] = True
    for c in e['high_i']:
        edge[:, n - 1 - c] = True
    cls[edge & oc] = 4
    mism = theta_ok != oc
    cls[mism] = 5
    counts = dict(
        n_cells=int(oc.size), n_ocean=int(oc.sum()), n_theta_finite=int(theta_ok.sum()),
        n_mismatch=int(mism.sum()),
        rim_G=int(rim_G.sum()), rim_G_taxicab=np.bincount(d_taxi[rim_G]).tolist(),
        rim_G_is_taxicab1=bool(np.array_equal(rim_G, oc & (d_taxi == 1))),
        rim_J_total=int((oc & ~np.isfinite(f['J'][0])).sum()),
        rim_J_is_taxicab_le2=bool(np.array_equal(oc & ~np.isfinite(f['J'][0]),
                                                 oc & (d_taxi <= 2))),
        rim_J_only=int(rim_J.sum()), edge_cells_ocean=int((edge & oc).sum()),
        G_sq_finite=int(np.isfinite(f['G_sq']).sum()),
        G_comp_finite=int(np.isfinite(f['G_comp']).sum()))
    return cls, d_taxi, counts


def ribbon_profile(G, oc, d_taxi, dmax=20):
    """Median / p90 of finite G by taxicab distance to land, tile edges
    excluded (3 cells), and the interior reference (d >= dmax)."""
    inner = np.zeros_like(oc)
    inner[3:-3, 3:-3] = True
    base = oc & inner & np.isfinite(G)
    ds_ = np.arange(2, dmax)
    med = np.array([np.median(G[base & (d_taxi == d)]) for d in ds_])
    p90 = np.array([np.percentile(G[base & (d_taxi == d)], 90) for d in ds_])
    ref = base & (d_taxi >= dmax)
    return ds_, med, p90, float(np.median(G[ref])), float(np.percentile(G[ref], 90))


def edge_profile(A, oc, nmax=8, ref=(1, 4)):
    """Median |A| at offset 0..nmax-1 from each tile edge, normalised by
    the median over offsets ref[0]..ref[1] from that edge."""
    out = {}
    n = A.shape[0]
    for name, take in (('low j (j=0)', lambda k: A[k, :]),
                       ('high j (j=719)', lambda k: A[n - 1 - k, :]),
                       ('low i (i=2880)', lambda k: A[:, k]),
                       ('high i (i=3599)', lambda k: A[:, n - 1 - k])):
        prof = np.array([np.nanmedian(np.abs(take(k))) for k in range(nmax)])
        base = np.nanmedian(np.abs(np.concatenate([take(k) for k in range(*ref)])))
        out[name] = prof / base
    return out


def draw_map(ax, X, Y, C, cmap, norm=None, land=None, **kw):
    if land is not None:
        ax.pcolormesh(X, Y, np.ma.masked_where(~land, np.ones_like(X)),
                      cmap=ListedColormap(['#b0b0b0']), shading='nearest')
    m = ax.pcolormesh(X, Y, np.ma.masked_invalid(C), cmap=cmap, norm=norm,
                      shading='nearest', **kw)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y))))
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
    return m


def make_figure(ds, f, cls, d_taxi, counts, edge_rim, out_png):
    X, Y = ds.XC.values[0], ds.YC.values[0]
    oc = ds.hFacC.values[0] > 0
    land = ~oc
    theta = ds.Theta.values[0]
    G = f['G_sq']
    cmap_cls = ListedColormap([c for _, _, c in CLASSES])
    norm_cls = BoundaryNorm(np.arange(-0.5, len(CLASSES)), cmap_cls.N)

    fig, axs = plt.subplots(2, 3, figsize=(19, 11.5))
    (a, b_, c), (d, e, ff) = axs

    # (a) Theta
    m = draw_map(a, X, Y, theta, 'magma', land=land,
                 vmin=np.nanpercentile(theta, 1), vmax=np.nanpercentile(theta, 99))
    fig.colorbar(m, ax=a, shrink=0.85, label='Theta [deg C]')
    a.set_title(f'(a) Theta, {SNAPSHOT}  (land = hFacC 0, grey)')

    # (b) log10 G, clipped to the interior p1..p99 so the coast is readable
    inner = oc & (d_taxi >= 2)
    inner[:2, :] = inner[-2:, :] = inner[:, :2] = inner[:, -2:] = False
    lo, hi = np.nanpercentile(np.log10(G[inner]), [1, 99.5])
    m = draw_map(b_, X, Y, np.log10(G), 'viridis', land=land, vmin=lo, vmax=hi)
    fig.colorbar(m, ax=b_, shrink=0.85, label='log10 G  [s^-4]')
    b_.set_title('(b) G = |grad b|^2, JMD95 b, calculate_grad_squared_tracer\n'
                 f'colour clipped to interior p1-p99.5; tile edges saturate (fill = 0)')

    # (c) validity map
    m = draw_map(c, X, Y, cls.astype(float), cmap_cls, norm=norm_cls)
    handles = [Rectangle((0, 0), 1, 1, color=col) for _, _, col in CLASSES]
    labels = [lab for _, lab, _ in CLASSES]
    labels[5] += f'  (n = {counts["n_mismatch"]})'
    labels[2] += f'  (n = {counts["rim_G"]})'
    labels[3] += f'  (n = {counts["rim_J_only"]})'
    c.legend(handles, labels, loc='lower left', fontsize=7.5, framealpha=0.95)
    c.set_title('(c) validity of Theta, G and the Jacobian (whole tile)')
    txt = (f'cells {counts["n_cells"]:,}; hFacC>0 {counts["n_ocean"]:,}; '
           f'isfinite(Theta) {counts["n_theta_finite"]:,}; mismatch {counts["n_mismatch"]}\n'
           f'G finite {counts["G_sq_finite"]:,}; G rim = taxicab d=1 exactly: '
           f'{counts["rim_G_is_taxicab1"]}\n'
           f'Jacobian rim {counts["rim_J_total"]:,} = taxicab d<=2 exactly: '
           f'{counts["rim_J_is_taxicab_le2"]}\n'
           f'edge rim (G / Jacobian): low edges 1 / 1 cell, high edges 1 / 2 cells')
    c.text(0.02, 0.60, txt, transform=c.transAxes, va='top', fontsize=7.5,
           bbox=dict(boxstyle='round', fc='white', alpha=0.9))

    # (d) inset: Monterey Bay, 48 x 48 cells, cell edges drawn
    j0, i0, w = 287, 96, 24
    sl = (slice(j0 - w, j0 + w), slice(i0 - w, i0 + w))
    m = d.pcolormesh(X[sl], Y[sl], cls[sl].astype(float), cmap=cmap_cls, norm=norm_cls,
                     shading='nearest', edgecolors='#555555', linewidth=0.15)
    d.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y[sl]))))
    d.set_xlabel('longitude'); d.set_ylabel('latitude')
    d.set_title(f'(d) cell-level inset, Monterey Bay: j {j0-w}..{j0+w-1}, '
                f'i {2880+i0-w}..{2880+i0+w-1}\nG NaN one cell from land (red); '
                'Jacobian NaN two cells (orange)')
    c.add_patch(Rectangle((X[sl].min(), Y[sl].min()), np.ptp(X[sl]), np.ptp(Y[sl]),
                          fill=False, ec='black', lw=1.2))

    # (e) ribbon test
    dd, med, p90, ref_med, ref_p90 = ribbon_profile(G, oc, d_taxi)
    ratio = med / ref_med
    e.plot(dd, med, 'o-', color='#1f5fa8', lw=2, ms=5, label='median G at distance d')
    e.plot(dd, p90, 's--', color='#1f5fa8', lw=1.2, ms=4, alpha=0.7, label='p90 G at distance d')
    e.axhline(ref_med, color='#444444', lw=1.5, label=f'interior median (d >= 20): {ref_med:.1e}')
    e.axhline(ref_p90, color='#444444', lw=1, ls='--', label=f'interior p90: {ref_p90:.1e}')
    b_edge = np.nanmedian(G[0, :])
    e.axhline(b_edge, color='#c8102e', lw=1.5, ls=':',
              label=f'what a b(0,0) ribbon looks like: G at the j=0 edge, {b_edge:.1e}')
    e.set_yscale('log'); e.set_xlabel('taxicab distance to land [cells]  (d = 1 is the NaN rim)')
    e.set_ylabel('G [s^-4]'); e.set_xticks(dd[::2])
    e.set_title('(e) no coastal gradient ribbon: G decays smoothly from land\n'
                f'median at d = 2, 3, 4, 10: {ratio[0]:.0f}x, {ratio[1]:.0f}x, '
                f'{ratio[2]:.0f}x, {ratio[8]:.0f}x the interior', fontsize=11)
    e.legend(fontsize=7.5, loc='center right'); e.grid(alpha=0.3)

    # (f) tile-edge test: the crop experiment -- fraction of finite cells at
    # each offset from an edge whose value changes when that edge moves
    Jmag = np.sqrt(sum(x ** 2 for x in f['J']))
    prof_G, prof_J = edge_profile(G, oc), edge_profile(Jmag, oc)
    cols = {'low j': '#1f5fa8', 'high j': '#c8102e', 'low i': '#2a9d8f', 'high i': '#f4a259'}
    k = np.arange(6)
    wbar = 0.1
    for n_, (name, col) in enumerate(cols.items()):
        fg = edge_rim['G_squared']['frac_changed'][name]
        fj = edge_rim['jacobian']['frac_changed'][name]
        ff.bar(k - 0.35 + n_ * wbar, fg, wbar, color=col, label=f'G, {name} edge')
        ff.bar(k + 0.05 + n_ * wbar, fj, wbar, color=col, alpha=0.45, hatch='//',
               label=f'Jacobian, {name} edge')
    ff.set_ylim(0, 1.15); ff.set_xticks(k)
    ff.set_xlabel('offset from tile edge [cells]  (bars left of tick: G; right, hatched: Jacobian)')
    ff.set_ylabel('fraction of cells changed by moving the edge')
    ff.set_title('(f) tile-edge rim (_tile_indexer + xgcm fill = 0), crop test\n'
                 'G: 1 cell on every edge; Jacobian: 1 cell low edges, 2 cells high edges',
                 fontsize=11)
    ff.text(0.5, 0.62, 'magnitude of the edge cells (median, vs offsets 1-3):\n'
            f'G low edges {prof_G["low j (j=0)"][0]:.1e}x / {prof_G["low i (i=2880)"][0]:.1e}x '
            '(diff against the 0 fill)\n'
            f'G high edges {prof_G["high j (j=719)"][0]:.2f}x / {prof_G["high i (i=3599)"][0]:.2f}x '
            '(interp with the 0 fill)\n'
            f'|Jacobian| low edges {prof_J["low j (j=0)"][0]:.1f}x / {prof_J["low i (i=2880)"][0]:.1f}x; '
            f'high edges {prof_J["high j (j=719)"][0]:.1f}x / {prof_J["high i (i=3599)"][0]:.1f}x',
            transform=ff.transAxes, ha='center', va='top', fontsize=7.5,
            bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    ff.legend(fontsize=7, ncol=2, loc='upper right'); ff.grid(alpha=0.3, axis='y')

    fig.suptitle(f'M0 QA: tile 330 (face 10), {SNAPSHOT}; no halo; maps north-up via (XC, YC)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    return out_png


def main():
    FIG_DIR.mkdir(exist_ok=True)
    ds, grid = load_snapshot()
    f = compute_fields(ds, grid)
    print('== trap (i): calculate_jacobian(u_x, v_y) are U, V')
    pprint.pprint(qc.check_jacobian_args(ds, grid))
    print('== trap (ii): halo_mask.py:75 not reached')
    pprint.pprint(qc.check_halo_branch(ds))
    print('== xgcm fill value is 0: max |diff_X(b)[i_g=2880] - b[i=2880]| =',
          qc.fill_value_is_zero(ds, grid, f['b']))
    print('== tile-edge rim (rows/cols from each edge that a crop changes)')
    edge_rim = qc.check_edge_rim(ds, grid, f['b'])
    pprint.pprint(edge_rim)
    cls, d_taxi, counts = classify(ds, f, edge_rim)
    print('== validity counts'); pprint.pprint(counts)
    oc = ds.hFacC.values[0] > 0
    for name in ('G_sq', 'G_comp'):
        dd, med, p90, ref_med, ref_p90 = ribbon_profile(f[name], oc, d_taxi)
        print(f'== ribbon test {name}: interior median {ref_med:.3e}; median/interior at d=2..10:',
              np.round(med[:9] / ref_med, 1).tolist())
    print('== G_comp / G_sq (interior median):',
          float(np.nanmedian((f['G_comp'] / f['G_sq'])[oc & (d_taxi >= 3)])))
    out = make_figure(ds, f, cls, d_taxi, counts, edge_rim,
                      FIG_DIR / 'm0_qa_tile330_20120702T00.png')
    print('wrote', out)


if __name__ == '__main__':
    main()
