""" M0 task 4: write ``tile330_grid.zarr`` (coding doc §3.1) and a two-hour
raw product in the §3.2 layout, then re-open both from disk and verify.

Nothing here is physics.  Every check prints a number and raises on
failure, so a silent layout slip cannot reach M1.

Run:
    /Users/xavier/miniforge3/envs/frontogenesis/bin/python m0_write.py
"""

import subprocess
import time

import numpy as np
import xarray as xr

from osn_tiles import (DATA_DIR, CORE_GRID_VARS, GRID_EXTRA_VARS, CORE_VARS,
                       WIND_VARS, load_grid, write_grid, open_grid, build_xgcm,
                       load_hour, load_hours, write_raw)
from dbof.llc4320_ingestion.grid import COMODO_COORD_META

T0, T1 = '2012-07-02 00:00:00', '2012-07-02 01:00:00'
GRID_PATH = DATA_DIR / 'tile330_grid.zarr'
RAW_PATH = DATA_DIR / 'tile330_raw_20120702T00_2h.zarr'
HORIZ = ('j', 'i', 'j_g', 'i_g')


def du(path):
    return subprocess.run(['du', '-sh', str(path)], capture_output=True,
                          text=True).stdout.split()[0]


def hdr(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78)


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        raise AssertionError(msg)


walls = {}

# ---------------------------------------------------------------------------
# 1. grid: pull, write
# ---------------------------------------------------------------------------
hdr('1. tile330_grid.zarr: pull and write')
t = time.time(); g = load_grid(); walls['load_grid'] = time.time() - t
t = time.time(); p = write_grid(g, GRID_PATH, clobber=True); walls['write_grid'] = time.time() - t
print(f'wrote {p}: {du(p)} on disk; load {walls["load_grid"]:.0f} s, write {walls["write_grid"]:.1f} s')

# ---------------------------------------------------------------------------
# 2. grid: re-open and verify
# ---------------------------------------------------------------------------
hdr('2. tile330_grid.zarr: re-open from disk and verify')
gs = open_grid(GRID_PATH, with_face=False)     # the stored (j, i) layout
print(f'dims {dict(gs.sizes)}; coords {list(gs.coords)}')
check(dict(gs.sizes) == {'j': 720, 'i': 720, 'i_g': 720, 'j_g': 720},
      'dims are (j, i) + i_g, j_g, no face dim (§3.1)')
check('face' in gs.coords and gs.face.ndim == 0 and int(gs.face) == 10,
      'face kept as scalar coord = 10')
want = set(CORE_GRID_VARS + GRID_EXTRA_VARS)
check(want <= set(gs.data_vars), f'all {len(want)} §3.1 vars present: {sorted(gs.data_vars)}')
dims_expect = {'hFacW': ('j', 'i_g'), 'hFacS': ('j_g', 'i'), 'hFacC': ('j', 'i'),
               'dxC': ('j', 'i_g'), 'dyC': ('j_g', 'i'), 'rAz': ('j_g', 'i_g'),
               'drF': (), 'Z': (), 'Zl': ()}
for v, d in dims_expect.items():
    check(gs[v].dims == d, f'{v} dims {gs[v].dims}')
check(all(gs[v].dtype == np.float32 for v in want),
      'all §3.1 vars float32: ' + ', '.join(f'{v}:{gs[v].dtype}' for v in sorted(want)))
print(f'  drF={float(gs.drF)}, Z={float(gs.Z)}, Zl={float(gs.Zl)}')
check((float(gs.drF), float(gs.Z), float(gs.Zl)) == (1.0, -0.5, 0.0), '0-d drF/Z/Zl = 1.0/-0.5/0.0')
for d in HORIZ:
    a = gs[d].attrs
    exp = COMODO_COORD_META[d]
    check(all(a.get(k) == v for k, v in exp.items()),
          f'comodo attrs on {d}: axis={a.get("axis")}, shift={a.get("c_grid_axis_shift")}')
# xgcm builds from the re-opened store, both layouts
grid_nf = build_xgcm(gs)
gf = open_grid(GRID_PATH, with_face=True)
grid_f = build_xgcm(gf)
check(set(grid_f.axes) == {'X', 'Y'} and dict(gf.sizes)['face'] == 1,
      f'build_xgcm from re-opened store: axes {sorted(grid_f.axes)} (face restored: {dict(gf.sizes)})')
print('  ', grid_nf)
# attrs
need_attrs = ['face_index', 'j_face_start', 'i_face_start', 'rect_i', 'rect_j', 'source',
              'git_commit', 'dbof_commit', 'created', 'orientation', 'dx_km_37N', 'dy_km_37N',
              'land_fill']
check(all(k in gs.attrs for k in need_attrs), 'all §3.1 attrs present')
for k in need_attrs:
    print(f'    {k} = {gs.attrs[k]!r}')
check((gs.attrs['face_index'], gs.attrs['j_face_start'], gs.attrs['i_face_start'],
       gs.attrs['rect_i'], gs.attrs['rect_j']) == (10, 0, 2880, 13320, 9720), 'tile attrs values')
check((gs.attrs['dx_km_37N'], gs.attrs['dy_km_37N']) == (1.71, 1.85), 'dx/dy at 37N = 1.71/1.85 km')
# values identical to the in-memory pull
for v in sorted(want):
    check(np.array_equal(gs[v].values, g[v].squeeze('face').values if 'face' in g[v].dims else g[v].values,
                         equal_nan=True), f'{v} round-trips bit-for-bit')
# hFacW/hFacS against a real hour's U/V NaN masks; hFacC against Theta
t = time.time(); h0 = load_hour(T0); walls['load_hour_check'] = time.time() - t
h0 = h0.squeeze(('time', 'face'))
for v, hf in (('U', 'hFacW'), ('V', 'hFacS'), ('Theta', 'hFacC')):
    nan = np.isnan(h0[v].values); land = gs[hf].values == 0
    check(np.array_equal(nan, land), f'isnan({v}) == ({hf}==0) at t0: {land.sum()} land cells, '
          f'{(nan != land).sum()} mismatches')

# ---------------------------------------------------------------------------
# 3. two hours: pull, write
# ---------------------------------------------------------------------------
hdr('3. two hours from both stores: pull and write')
t = time.time(); raw = load_hours([T0, T1], grid_ds=g); walls['load_hours'] = time.time() - t
t = time.time(); p = write_raw(raw, RAW_PATH, clobber=True); walls['write_raw'] = time.time() - t
print(f'wrote {p}: {du(p)} on disk; load {walls["load_hours"]:.0f} s, write {walls["write_raw"]:.1f} s')

# ---------------------------------------------------------------------------
# 4. two hours: re-open and verify
# ---------------------------------------------------------------------------
hdr('4. two-hour product: re-open from disk and verify')
rs = xr.open_zarr(RAW_PATH).load()
print(rs)
check(dict(rs.sizes) == {'time': 2, 'j': 720, 'i': 720, 'i_g': 720, 'j_g': 720},
      f'dims {dict(rs.sizes)} (§3.2 layout, no face dim)')
check(set(CORE_VARS + WIND_VARS) == set(rs.data_vars), f'vars {sorted(rs.data_vars)}')
check(list(rs.time.values.astype("datetime64[s]").astype(str)) ==
      ['2012-07-02T00:00:00', '2012-07-02T01:00:00'], f'time coord {rs.time.values.astype("datetime64[s]")}')
check(rs.attrs['iterations'] == [1022976, 1023120] and rs.attrs['iterations'][1] - rs.attrs['iterations'][0] == 144,
      f'iterations attr {rs.attrs["iterations"]} (144/hour)')
check(list(rs.niter.values) == rs.attrs['iterations'], f'niter(time) coord {list(rs.niter.values)} == attr')
check(rs.attrs['stores'] == ['llc_surf', 'llc_wind'] and rs.attrs['endpoint'].startswith('https://'),
      f'stores {rs.attrs["stores"]}, endpoint {rs.attrs["endpoint"]}')
check(all(k in rs.attrs for k in ('git_commit', 'dbof_commit', 'created', 'timestamps')),
      f'provenance attrs: git_commit={rs.attrs["git_commit"]}, dbof_commit={rs.attrs["dbof_commit"]}')
check({'XC', 'YC', 'time'} <= set(rs.coords) and rs.XC.dims == ('j', 'i'), f'coords {list(rs.coords)}')
shape_expect = {'Theta': ('time', 'j', 'i'), 'Salt': ('time', 'j', 'i'), 'U': ('time', 'j', 'i_g'),
                'V': ('time', 'j_g', 'i'), 'W': ('time', 'j', 'i'), 'Eta': ('time', 'j', 'i'),
                'KPPhbl': ('time', 'j', 'i'), 'oceTAUX': ('time', 'j', 'i_g'),
                'oceTAUY': ('time', 'j_g', 'i')}
for v, d in shape_expect.items():
    check(rs[v].dims == d and rs[v].shape == (2, 720, 720) and rs[v].dtype == np.float32,
          f'{v} {rs[v].dims} {rs[v].shape} {rs[v].dtype}')
for d in HORIZ:
    a = rs[d].attrs
    check(all(a.get(k) == v for k, v in COMODO_COORD_META[d].items()) and rs[d].dtype == np.int64
          and np.array_equal(rs[d].values, gs[d].values),
          f'{d}: comodo attrs axis={a.get("axis")}, shift={a.get("c_grid_axis_shift")}; '
          f'{rs[d].dtype}, values == grid store')
check(rs.niter.dtype == np.int64 and rs.face.dtype == np.int64, 'niter, face int64')
# chunking: one chunk per hour per var
enc = rs.Theta.encoding
print(f'  Theta encoding chunks {enc.get("chunks")}, time encoding {rs.time.encoding.get("units")}')
check(tuple(enc.get('chunks', ())) == (1, 720, 720), 'one (720, 720) chunk per hour')
# NaN pattern vs the grid masks, both hours
masks = {'hFacC': gs.hFacC.values == 0, 'hFacW': gs.hFacW.values == 0, 'hFacS': gs.hFacS.values == 0}
for v, hf in (('Theta', 'hFacC'), ('Salt', 'hFacC'), ('W', 'hFacC'), ('Eta', 'hFacC'),
              ('KPPhbl', 'hFacC'), ('U', 'hFacW'), ('V', 'hFacS')):
    for k in range(2):
        nan = np.isnan(rs[v].isel(time=k).values)
        check(np.array_equal(nan, masks[hf]), f'{v}[t{k}] NaN == ({hf}==0): {(nan != masks[hf]).sum()} mismatches')
# oceTAU* come masked with the centred mask (task-3 finding); record, do not assert the staggered one
for v, hf in (('oceTAUX', 'hFacW'), ('oceTAUY', 'hFacS')):
    nan = np.isnan(rs[v].isel(time=0).values)
    print(f'  note {v}: NaN == (hFacC==0) {np.array_equal(nan, masks["hFacC"])}; '
          f'vs {hf}: {(nan != masks[hf]).sum()} cells differ (expected: centred mask, §3.2)')
# the two hours differ, and hour 1 matches a fresh load_hour
for v in ('Theta', 'U', 'Eta', 'KPPhbl'):
    a, b = rs[v].isel(time=0).values, rs[v].isel(time=1).values
    diff = np.nanmax(np.abs(a - b))
    check(diff > 0, f'{v}: hours differ, max |t1 - t0| = {diff:.4g}')
t = time.time(); h1 = load_hour(T1).squeeze(('time', 'face')); walls['load_hour_fresh'] = time.time() - t
for v in ('Theta', 'V'):
    check(np.array_equal(rs[v].isel(time=1).values, h1[v].values, equal_nan=True),
          f'{v}[t1] on disk == fresh load_hour({T1}) bit-for-bit')
check(np.array_equal(rs.Theta.isel(time=0).values, h0.Theta.values, equal_nan=True),
      f'Theta[t0] on disk == fresh load_hour({T0}) bit-for-bit')
check(np.array_equal(rs.XC.values, gs.XC.values) and np.array_equal(rs.YC.values, gs.YC.values),
      'XC/YC coords equal the grid store')
# the mask is static across the two hours
check(np.array_equal(np.isnan(rs.Theta.isel(time=0).values), np.isnan(rs.Theta.isel(time=1).values)),
      'Theta NaN pattern identical at t0 and t1')

# ---------------------------------------------------------------------------
# 5. sizes and wall times
# ---------------------------------------------------------------------------
hdr('5. sizes and wall times')
print(f'{GRID_PATH}: {du(GRID_PATH)} on disk, {g.nbytes/1e6:.1f} MB in memory')
print(f'{RAW_PATH}: {du(RAW_PATH)} on disk, {raw.nbytes/1e6:.1f} MB in memory')
for k, v in walls.items():
    print(f'  {k:18s} {v:7.1f} s')
print('ALL CHECKS PASSED')
