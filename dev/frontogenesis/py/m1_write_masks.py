""" M1 task 1: write ``tile330_masks.nc`` (coding doc §3.5) from the M0
static grid, re-open it from disk and verify, and run the Gulf of
California check (the >= 100 km offshore cut must remove the head of the
Gulf, which the tile's eastern edge clips -- no polygon).

Nothing here is physics.  Every check prints a number and raises on
failure.  Offline: reads ``tile330_grid.zarr`` (and the 2 h raw store for
the ``isfinite(Theta)`` identity) only.

Run:
    /Users/xavier/miniforge3/envs/frontogenesis/bin/python m1_write_masks.py
"""

import subprocess

import numpy as np
import xarray as xr
from scipy import ndimage

import masking as mk
from osn_tiles import DATA_DIR, open_grid

GRID_PATH = DATA_DIR / 'tile330_grid.zarr'
RAW_PATH = DATA_DIR / 'tile330_raw_20120702T00_2h.zarr'
OUT_PATH = DATA_DIR / 'tile330_masks.nc'
HALO_CELLS, OFFSHORE_KM, EDGE_CELLS = 7, 100.0, 7
# a point in the upper Gulf and one in the open Pacific, to name the two
# connected components of the ocean mask
GULF_LONLAT, PACIFIC_LONLAT = (-114.0, 30.5), (-125.0, 35.0)


def du(path):
    return subprocess.run(['du', '-sh', str(path)], capture_output=True,
                          text=True).stdout.split()[0]


def hdr(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78)


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        raise AssertionError(msg)


def ocean_component(grid_ds, ocean, lon, lat):
    """Bool mask of the connected ocean component holding the cell nearest
    ``(lon, lat)``, plus that cell's (j, i)."""
    X = grid_ds.XC.squeeze().values
    Y = grid_ds.YC.squeeze().values
    j, i = np.unravel_index(np.argmin((X - lon) ** 2 + (Y - lat) ** 2), X.shape)
    if not ocean[j, i]:
        raise ValueError(f'cell nearest ({lon}, {lat}) is land')
    labels, _ = ndimage.label(ocean)
    return labels == labels[j, i], (int(j), int(i))


def gulf_of_california_check(grid_ds, masks) -> dict:
    """Is the Gulf of California removed by the offshore cut alone?

    Inside the tile (lat >= 26.66N) the Baja peninsula separates the Gulf
    from the Pacific, so the Gulf is its own connected component of the
    ocean mask; no polygon is needed to name it.  The tile's east edge
    (lon -113.01) clips it, so land east of the edge is invisible to skfmm
    and the distances inside the Gulf are *over*-estimates -- which makes
    this a conservative test of the cut.
    """
    oc = masks['mask_ocean'].values
    d = masks['coast_distance_km'].values
    gulf, _ = ocean_component(grid_ds, oc, *GULF_LONLAT)
    pac, _ = ocean_component(grid_ds, oc, *PACIFIC_LONLAT)
    X = grid_ds.XC.squeeze().values
    Y = grid_ds.YC.squeeze().values
    east = gulf[-1, :]                      # the j = 719 column is the east edge
    res = dict(
        gulf_is_separate_from_pacific=bool(not (gulf & pac).any()),
        n_gulf=int(gulf.sum()),
        gulf_lon=(float(X[gulf].min()), float(X[gulf].max())),
        gulf_lat=(float(Y[gulf].min()), float(Y[gulf].max())),
        gulf_max_coast_km=float(np.nanmax(d[gulf])),
        gulf_n_ge_offshore=int((gulf & masks['mask_offshore'].values).sum()),
        gulf_n_on_east_edge=int(east.sum()),
        gulf_max_coast_km_on_east_edge=float(np.nanmax(d[-1, :][east])) if east.any() else None,
        gulf_removed_by_halo=int((gulf & ~masks['mask_halo'].values).sum()),
        gulf_removed_by_edge=int((gulf & ~masks['mask_edge'].values).sum()),
        gulf_survive_halo_and_edge=int((gulf & masks['mask_halo'].values
                                        & masks['mask_edge'].values).sum()),
        gulf_n_analysis=int((gulf & masks['mask_analysis'].values).sum()),
        n_pacific=int(pac.sum()),
        pacific_n_analysis=int((pac & masks['mask_analysis'].values).sum()))
    return res


def main():
    hdr('1. build the masks from tile330_grid.zarr')
    g = open_grid(GRID_PATH, with_face=True)
    sp = mk.spacing_km(g)
    print('  spacing (km): ' + ', '.join(f'{k}={v:.4f}' for k, v in sp.items()))
    halo_km = mk.halo_width_km(g, HALO_CELLS)
    print(f'  halo: {HALO_CELLS} cells x median dxC {sp["dxC_median_km"]:.4f} km = {halo_km:.3f} km '
          f'(= {halo_km / sp["dyC_median_km"]:.2f} cells of dyC)')
    masks = mk.build_masks(g, HALO_CELLS, OFFSHORE_KM, EDGE_CELLS)
    for v in mk.MASK_VARS[:-1]:
        print(f'  {v:14s} retained {int(masks[v].sum()):>8,} of {masks[v].size:,}')
    a = masks.attrs
    check(a['n_ocean'] == 356_877, f'n_ocean {a["n_ocean"]:,} == 356,877 (M0)')
    check(a['n_ocean'] > a['n_halo'] > a['n_offshore'] > a['n_analysis'] > 0,
          f'stages nest: ocean {a["n_ocean"]:,} > halo {a["n_halo"]:,} > offshore '
          f'{a["n_offshore"]:,} > analysis {a["n_analysis"]:,}')

    hdr('2. write, re-open, verify')
    p = mk.write_masks(masks, OUT_PATH, clobber=True)
    print(f'wrote {p}: {du(p)} on disk')
    ms = mk.open_masks(OUT_PATH)
    check(tuple(ms.data_vars) == mk.MASK_VARS, f'vars exactly §3.5: {list(ms.data_vars)}')
    check(dict(ms.sizes) == {'j': 720, 'i': 720}, f'dims {dict(ms.sizes)}')
    check({'XC', 'YC', 'j', 'i'} <= set(ms.coords), f'coords {list(ms.coords)}')
    for v in mk.MASK_VARS[:-1]:
        check(ms[v].dtype == bool and ms[v].dims == ('j', 'i'), f'{v}: bool (j, i)')
        check(np.array_equal(ms[v].values, masks[v].values), f'{v} round-trips exactly')
    check(np.array_equal(ms.coast_distance_km.values, masks.coast_distance_km.values,
                         equal_nan=True), 'coast_distance_km round-trips exactly')
    for k in ('halo_cells', 'halo_km', 'edge_cells', 'offshore_km', 'dxC_median_km',
              'convention', 'git_commit', 'dbof_commit', 'created', 'n_analysis'):
        check(k in ms.attrs, f'attr {k} = {ms.attrs[k]!r}' if k in ms.attrs else f'attr {k} missing')
    # the functions reproduce the file
    check(np.array_equal(mk.analysis_mask(g, HALO_CELLS, OFFSHORE_KM, EDGE_CELLS), ms.mask_analysis.values),
          'analysis_mask() == mask_analysis on disk')
    check(np.array_equal(ms.mask_analysis.values,
                         ms.mask_ocean.values & ms.mask_halo.values & ms.mask_offshore.values
                         & ms.mask_edge.values), 'mask_analysis == ocean & halo & offshore & edge')
    # convention: land False in every mask, NaN in the distance
    land = ~ms.mask_ocean.values
    for v in mk.MASK_VARS[:-1]:
        if v != 'mask_edge':                # geometry only, by contract (§4.2)
            check(not ms[v].values[land].any(), f'{v}: land is False')
    check(np.isnan(ms.coast_distance_km.values[land]).all() and
          np.isfinite(ms.coast_distance_km.values[~land]).all(), 'coast_distance_km NaN on land, finite on ocean')
    # ocean == isfinite(Theta), both hours
    raw = xr.open_zarr(RAW_PATH).load()
    for t in range(raw.sizes['time']):
        fin = np.isfinite(raw.Theta.isel(time=t).values)
        check(np.array_equal(fin, ms.mask_ocean.values),
              f'mask_ocean == isfinite(Theta) at t{t}: {(fin != ms.mask_ocean.values).sum()} mismatches')
    # halo width in cells, from the index distance to land
    oc = ms.mask_ocean.values
    taxi = ndimage.distance_transform_cdt(oc, metric='taxicab')
    chess = ndimage.distance_transform_cdt(oc, metric='chessboard')
    h = ms.mask_halo.values
    print(f'  halo: min taxicab distance of a retained cell {taxi[h].min()}, chessboard {chess[h].min()}; '
          f'max taxicab of an excluded ocean cell {taxi[oc & ~h].max()}, chessboard {chess[oc & ~h].max()}')
    check(taxi[h].min() >= HALO_CELLS, f'every retained cell is >= {HALO_CELLS} cells (taxicab) from land')
    d = ms.coast_distance_km.values
    check(np.nanmin(d[h]) >= ms.attrs['halo_km'] > np.nanmax(d[oc & ~h]),
          f'halo splits at {ms.attrs["halo_km"]:.3f} km: min retained {np.nanmin(d[h]):.3f}, '
          f'max excluded {np.nanmax(d[oc & ~h]):.3f}')
    # analysis mask is one connected piece (no inland seas, no cut-off pockets)
    _, n_comp = ndimage.label(ms.mask_analysis.values)
    print(f'  analysis mask connected components: {n_comp}')

    hdr('3. Gulf of California: does the >= 100 km cut remove it?')
    res = gulf_of_california_check(g, ms)
    for k, v in res.items():
        print(f'  {k:32s} {v}')
    check(res['gulf_is_separate_from_pacific'], 'the Gulf is its own ocean component inside the tile')
    check(res['gulf_max_coast_km'] < OFFSHORE_KM,
          f'max coast distance inside the Gulf {res["gulf_max_coast_km"]:.1f} km < {OFFSHORE_KM:g} km')
    check(res['gulf_n_analysis'] == 0 and res['gulf_n_ge_offshore'] == 0,
          f'Gulf cells surviving the offshore cut: {res["gulf_n_ge_offshore"]}; in the analysis mask: '
          f'{res["gulf_n_analysis"]} (of {res["n_gulf"]:,})')
    # every other non-Pacific component (inland water the model treats as ocean) too
    _, n = ndimage.label(oc)
    pac, _ = ocean_component(g, oc, *PACIFIC_LONLAT)
    other = oc & ~pac
    print(f'  {n} ocean components; non-Pacific cells {other.sum():,}, of which in the analysis mask: '
          f'{(other & ms.mask_analysis.values).sum()}')
    check((other & ms.mask_analysis.values).sum() == 0, 'no non-Pacific ocean cell survives the analysis mask')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
