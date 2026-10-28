""" M0 reconnaissance: the five empirical questions of execution prompt 1, task 3.

Throwaway but reproducible.  Everything is a one-shot statistic against
real OSN data for tile 330; no physics module is written here and nothing
is saved except stdout (tee it).  ``dbof`` operators are *called* (the
library route) where a gradient is needed, so the numbers below are for
the same stencils M1 will use.

Run:
    /Users/xavier/miniforge3/envs/frontogenesis/bin/python m0_recon.py

Questions (numbers printed under matching headers):
  Q1  land: 0 or NaN?  cell-by-cell vs hFacC / hFacW / hFacS
  Q2  W[k_l=0]: statistics, comparison with dEta/dt, and the size of the
      tilting term  -b_z (w_x b_x + w_y b_y)  relative to F
  Q3  dxC/dyC/dxG/dyG near 37N and tile-wide; hourly displacement in cells
  Q4  rotation terms u|grad CS|, u|grad SN| and the metric term u tan(phi)/a
      against a nominal 1e-5 s^-1 strain and the measured strain
  Q5  the model configuration is documentary (MITgcm_contrib llc_hires,
      llc_4320/input/data: tempAdvScheme=saltAdvScheme=7); what is computed
      here is the implicit diffusivity of that 7th-order upwind kernel as a
      function of scale, for the tile's own dx and |u| (no data needed)
"""

import time

import numpy as np
import xarray as xr
from scipy import ndimage

from osn_tiles import (OSN_ENDPOINT, tile_spec, load_grid, build_xgcm,
                       load_hour, load_wind_hour, _subset_tile)
from dbof.llc4320_ingestion.get_raw_data import get_remote_gridfile
from dbof.utils.native_gradient import (
    calculate_native_gradient_tracer, calculate_jacobian,
    calculate_native_strain_vorticity, interp_corner_squared,
    interp_pair_to_center)
from dbof.preprocessing.calculate_fields import (
    buoyancy_of_field, _frontogenesis_formula)

T_M1, T0, T1 = '2012-07-01 23:00:00', '2012-07-02 00:00:00', '2012-07-02 01:00:00'
DT = 3600.0
A_EARTH = 6371e3       # m
NOMINAL_STRAIN = 1e-5  # s^-1, planning §5.2
BZ_BRACKET = {'well-mixed 1e-5': 1e-5, 'moderate 1e-4': 1e-4,
              'diurnal warm layer 4e-4': 4e-4}   # s^-2, planning §2.2
DRF0 = 1.0             # m, top-cell thickness (task-2 log: drF=1.0 in OSN gridfile)


def sq(da):
    """xarray -> 2-D float64 numpy (drop the length-1 time/face dims)."""
    return np.asarray(da.squeeze(drop=True).values, dtype=np.float64)


def stats(x, mask=None, pct=(50, 90, 99)):
    """(min, median, p90, p99, max) of x over mask, ignoring NaN."""
    v = x if mask is None else x[mask]
    v = v[np.isfinite(v)]
    p = np.percentile(v, pct)
    return dict(n=v.size, min=v.min(), med=p[0], p90=p[1], p99=p[2], max=v.max())


def fmt(d, unit='', scale=1.0):
    return (f"n={d['n']:7d}  min={d['min']*scale:.4g}  med={d['med']*scale:.4g}  "
            f"p90={d['p90']*scale:.4g}  p99={d['p99']*scale:.4g}  max={d['max']*scale:.4g} {unit}")


def hdr(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
t_start = time.time()
tile = tile_spec()
g = load_grid()
grid = build_xgcm(g)
# staggered land fractions live only in the raw gridfile (process_llc4320_grid drops them)
raw = get_remote_gridfile(OSN_ENDPOINT).reset_coords()
hfac_stag = _subset_tile(raw[['hFacW', 'hFacS']], tile).compute()
del raw
ds_m1 = load_hour(T_M1)
ds0 = load_hour(T0)
ds1 = load_hour(T1)
wd0 = load_wind_hour(T0, keep=('KPPhbl', 'oceTAUX', 'oceTAUY', 'PhiBot', 'SIarea'))
print(f'loaded grid + 3 core hours + 1 wind hour in {time.time()-t_start:.0f} s')

hFacC = sq(g.hFacC); hFacW = sq(hfac_stag.hFacW); hFacS = sq(hfac_stag.hFacS)
YC = sq(g.YC); XC = sq(g.XC)
ocean = hFacC > 0
ny, nx = ocean.shape

# interior mask for derivative statistics: ocean, 3 cells clear of land
# (stencil width of the centred Jacobian), and clear of the tile rim where the
# staggered high edge is missing (task-2 log) and xgcm's padding='fill' bites.
land_dil = ndimage.binary_dilation(~ocean, iterations=3)
rim = np.zeros_like(ocean); rim[:3, :] = rim[-3:, :] = rim[:, :3] = rim[:, -3:] = True
interior = ocean & ~land_dil & ~rim
print(f'tile {ny}x{nx}: ocean {ocean.sum()} ({ocean.mean():.4f}), '
      f'interior (3-cell halo + 3-cell rim removed) {interior.sum()} ({interior.mean():.4f})')

# ---------------------------------------------------------------------------
# Q1  Land: 0 or NaN?
# ---------------------------------------------------------------------------
hdr('Q1  Is OSN land stored as 0 or NaN?  (cell-by-cell against hFac)')
print(f'hFacC==0: {(~ocean).sum()}   hFacW==0: {(hFacW == 0).sum()}   hFacS==0: {(hFacS == 0).sum()}')
print(f'hFacC unique-ish: min={hFacC.min()}, max={hFacC.max()}, '
      f'n(0<hFac<1)={((hFacC > 0) & (hFacC < 1)).sum()} (partial cells)')
# consistency of the staggered masks with hFacC: hFacW[j,i] should be
# min(hFacC[j,i-1], hFacC[j,i]) (west face of cell i); same for hFacS in j
hw_expect = np.minimum(np.roll(hFacC, 1, axis=1), hFacC); hw_expect[:, 0] = np.nan
hs_expect = np.minimum(np.roll(hFacC, 1, axis=0), hFacC); hs_expect[0, :] = np.nan
okW = np.isfinite(hw_expect); okS = np.isfinite(hs_expect)
print(f'hFacW == min(hFacC[i-1],hFacC[i]) at {np.mean(np.isclose(hFacW[okW], hw_expect[okW])):.5f} '
      f'of cells;  hFacS == min(hFacC[j-1],hFacC[j]) at {np.mean(np.isclose(hFacS[okS], hs_expect[okS])):.5f}')

checks = [('Theta', sq(ds0.Theta), hFacC), ('Salt', sq(ds0.Salt), hFacC),
          ('W', sq(ds0.W), hFacC), ('Eta', sq(ds0.Eta), hFacC),
          ('U', sq(ds0.U), hFacW), ('V', sq(ds0.V), hFacS),
          ('KPPhbl', sq(wd0.KPPhbl), hFacC), ('PhiBot', sq(wd0.PhiBot), hFacC),
          ('SIarea', sq(wd0.SIarea), hFacC),
          ('oceTAUX', sq(wd0.oceTAUX), hFacW), ('oceTAUY', sq(wd0.oceTAUY), hFacS)]
print(f"{'var':8s} {'mask':6s} {'n_land':>7s} {'n_nan':>7s} {'n_zero':>7s} "
      f"{'nan&land':>9s} {'finite&land':>12s} {'nan&ocean':>10s} {'zero&ocean':>11s}  verdict")
for name, arr, hf in checks:
    land = hf == 0
    isnan = np.isnan(arr); iszero = arr == 0
    n_fin_land = (land & ~isnan).sum(); n_nan_oc = (~land & isnan).sum()
    n_zero_oc = (~land & iszero).sum()
    mname = 'hFacC' if hf is hFacC else ('hFacW' if hf is hFacW else 'hFacS')
    verdict = ('NaN==land exactly' if (n_fin_land == 0 and n_nan_oc == 0)
               else f'MISMATCH')
    print(f'{name:8s} {mname:6s} {land.sum():7d} {isnan.sum():7d} {iszero.sum():7d} '
          f'{(land & isnan).sum():9d} {n_fin_land:12d} {n_nan_oc:10d} {n_zero_oc:11d}  {verdict}')
    if n_zero_oc:
        jj, ii = np.nonzero(~land & iszero)
        print(f'    exact zeros in ocean at (j,i) e.g. {list(zip(jj[:5], ii[:5]))}; '
              f'hFac there = {hf[jj[:5], ii[:5]]}')
# U/V against the centred mask, to see how the staggered NaN pattern relates to hFacC
for name, arr in (('U', sq(ds0.U)), ('V', sq(ds0.V))):
    isnan = np.isnan(arr)
    print(f'{name} vs hFacC: NaN&ocean(C) {(ocean & isnan).sum()}, finite&land(C) {(~ocean & ~isnan).sum()}')
# same at the other two hours (is the mask static?)
for lab, d in (('t-1h', ds_m1), ('t+1h', ds1)):
    print(f'{lab}: Theta NaN pattern identical to t0: {np.array_equal(np.isnan(sq(d.Theta)), np.isnan(sq(ds0.Theta)))}; '
          f'U: {np.array_equal(np.isnan(sq(d.U)), np.isnan(sq(ds0.U)))}')

# ---------------------------------------------------------------------------
# Q2  W[k_l=0]
# ---------------------------------------------------------------------------
hdr('Q2  Is W[k_l=0] ~ 0?')
W0 = sq(ds0.W); Eta_m1, Eta0, Eta1 = sq(ds_m1.Eta), sq(ds0.Eta), sq(ds1.Eta)
print('OSN core store dims for W:', dict(ds0.W.sizes), ' coords k_l =', ds0.W.coords.get('k_l', 'absent').values
      if 'k_l' in ds0.W.coords else 'absent')
print('  -> only the k_l=0 interface is in the OSN store; no next interface to compare against')
print('|W| over ocean          :', fmt(stats(np.abs(W0), ocean), 'm/s'))
print(' W  over ocean (signed) :', fmt(stats(W0, ocean), 'm/s'))
print('Eta over ocean          :', fmt(stats(Eta0, ocean), 'm'))
dEta_fwd = (Eta1 - Eta0) / DT
dEta_ctr = (Eta1 - Eta_m1) / (2 * DT)
print('dEta/dt forward  (t0->t1):', fmt(stats(np.abs(dEta_fwd), ocean), 'm/s'))
print('dEta/dt centred (t-1->t1):', fmt(stats(np.abs(dEta_ctr), ocean), 'm/s'))
for lab, d in (('forward', dEta_fwd), ('centred', dEta_ctr)):
    m = ocean & np.isfinite(W0) & np.isfinite(d)
    x, y = d[m], W0[m]
    r = np.corrcoef(x, y)[0, 1]
    slope = np.sum(x * y) / np.sum(x * x)
    rms_w, rms_d, rms_diff = [np.sqrt(np.mean(v ** 2)) for v in (y, x, y - x)]
    print(f'  W vs dEta/dt ({lab:7s}): corr={r:.4f}  slope W/dEtadt={slope:.4f}  '
          f'rms W={rms_w:.3e}  rms dEta/dt={rms_d:.3e}  rms(W-dEta/dt)={rms_diff:.3e} m/s  '
          f'-> residual/rms W = {rms_diff/rms_w:.3f}')
# tidal signature: tile-mean Eta change over the hour
print(f'  tile-mean Eta: t-1h {np.nanmean(Eta_m1[ocean]):.4f}  t0 {np.nanmean(Eta0[ocean]):.4f}  '
      f't+1h {np.nanmean(Eta1[ocean]):.4f} m;  tile-mean W {np.nanmean(W0[ocean]):.3e} m/s;  '
      f'std of (Eta1-Eta0) {np.nanstd((Eta1-Eta0)[ocean]):.4f} m')
# a centred +-1h difference under-recovers d/dt of a tidal harmonic by
# sin(w dt)/(w dt); the expected regression slope of W on the centred
# difference is the inverse of that factor if W really is dEta/dt at t0
for lab, T_h in (('M2', 12.42), ('K1', 23.93)):
    x = 2 * np.pi / (T_h * 3600) * DT
    print(f'  {lab} (T={T_h} h): centred difference recovers sin(x)/x = {np.sin(x)/x:.4f} '
          f'of dEta/dt -> expected slope {x/np.sin(x):.4f}')

# --- horizontal gradient of W at k_l=0 and the tilting term -----------------
d0 = ds0.isel(time=0)
gm = xr.merge([d0, g])                       # ds_merge: fields + metrics + CS/SN
Wx, Wy = calculate_native_gradient_tracer(d0.W, g, grid)
Wx, Wy = sq(Wx), sq(Wy)
gradW = np.hypot(Wx, Wy)
print('|grad_h W| at k_l=0, interior:', fmt(stats(gradW, interior), '1/s'))

# buoyancy and its gradient (JMD95 via dbof), Jacobian, F -- the M1 operators
b = buoyancy_of_field(gm).compute()
bx, by = calculate_native_gradient_tracer(b, g, grid)
bx, by = sq(bx), sq(by)
G = bx ** 2 + by ** 2
ux, uy, vx, vy = [sq(a) for a in calculate_jacobian(d0.U, d0.V, gm, grid)]
F = sq(_frontogenesis_formula(*[xr.DataArray(a) for a in (ux, uy, vx, vy, bx, by)]))
delta = ux + vy
print('|grad b| interior       :', fmt(stats(np.sqrt(G), interior), 's^-2'))
print('|F|      interior       :', fmt(stats(np.abs(F), interior), 's^-5'))
print('divergence interior     :', fmt(stats(np.abs(delta), interior), 's^-1'))
front = interior & (G > np.nanpercentile(G[interior], 90))
print(f'"front" pixels = interior & G > p90(G) = {front.sum()} cells')

# (a) tilting with the *surface* w gradient (what OSN gives directly)
tilt_kin = Wx * bx + Wy * by       # s^-3; multiply by b_z for the term
# (b) tilting with the *cell-base* w.  Continuity (z up) gives
#     w(-drF) = w(0) + drF * delta, so the convergence part is +drF*delta
#     (planning §2.2 writes -drF*delta: a sign slip, harmless for the
#     magnitudes below).  Only the delta part is used here; the w(0) part
#     is (a) above, and the two add in the model's cell-base W.
delta_da = xr.DataArray(delta[None], dims=('face', 'j', 'i'), coords={'j': g.j, 'i': g.i})
dx_, dy_ = calculate_native_gradient_tracer(delta_da, g, grid)
wbx, wby = DRF0 * sq(dx_), DRF0 * sq(dy_)
print(f'|W(0)| vs drF|delta| over ocean: medians {np.nanmedian(np.abs(W0[ocean])):.3e} vs '
      f'{DRF0*np.nanmedian(np.abs(delta[interior])):.3e} m/s  (which part dominates w at the cell base)')
tilt_base = wbx * bx + wby * by
print('|grad w_base| = drF|grad delta|, interior:', fmt(stats(np.hypot(wbx, wby), interior), '1/s'))
for lab, bz in BZ_BRACKET.items():
    for kind, tk in (('surface W', tilt_kin), ('cell-base w', tilt_base)):
        T = bz * tk
        for mlab, m in (('interior', interior), ('front', front)):
            ratio = np.abs(T[m]) / np.abs(F[m])
            ratio = ratio[np.isfinite(ratio)]
            rms_ratio = np.sqrt(np.nanmean(T[m] ** 2)) / np.sqrt(np.nanmean(F[m] ** 2))
            print(f'  b_z={bz:.0e} ({lab:24s}) {kind:12s} {mlab:8s}: '
                  f'median |T|/|F| = {np.median(ratio):.3e}, p90 = {np.percentile(ratio, 90):.3e}, '
                  f'rms(T)/rms(F) = {rms_ratio:.3e}')

# ---------------------------------------------------------------------------
# Q3  grid spacing and hourly displacement
# ---------------------------------------------------------------------------
hdr('Q3  dxC / dyC / dxG / dyG (km) and hourly displacement in cells')
band = (YC >= 36.5) & (YC <= 37.5)
print(f'YC range {YC.min():.3f}..{YC.max():.3f}; band 36.5-37.5N has {band.sum()} cells '
      f'({band.any(axis=1).sum()} rows)')
# which model axis is zonal on this face?  (faces 8-13 of the LLC are rotated)
lon_along_i = abs(XC[0, -1] - XC[0, 0]) > abs(XC[-1, 0] - XC[0, 0])
print(f'XC corners (j,i)=(0,0) {XC[0,0]:.3f}, (0,-1) {XC[0,-1]:.3f}, (-1,0) {XC[-1,0]:.3f}; '
      f'YC {YC[0,0]:.3f}, {YC[0,-1]:.3f}, {YC[-1,0]:.3f}')
print(f'-> longitude varies along {"i" if lon_along_i else "j"}, latitude along '
      f'{"j" if lon_along_i else "i"}: model x (i, dxC) is '
      f'{"zonal" if lon_along_i else "MERIDIONAL"}, model y (j, dyC) is '
      f'{"meridional" if lon_along_i else "ZONAL"}')
dlon = np.median(np.abs(np.diff(XC, axis=1 if lon_along_i else 0)))
print(f'   zonal step {dlon:.6f} deg = 1/{1/dlon:.1f} deg; (1/48) deg * 111.32 km * cos(lat) = '
      + ', '.join(f'{111.32/48*np.cos(np.radians(la)):.3f} km at {la}N' for la in (26.66, 37.0, 38.27)))
for name in ('dxC', 'dyC', 'dxG', 'dyG'):
    arr = sq(g[name])
    print(f'{name} band 36.5-37.5N :', fmt(stats(arr, band), 'km', 1e-3))
    print(f'{name} full tile      :', fmt(stats(arr), 'km', 1e-3))
dxC = sq(g.dxC); dyC = sq(g.dyC)
print(f'aspect dyC/dxC: median {np.median(dyC/dxC):.4f}, range {np.min(dyC/dxC):.4f}..{np.max(dyC/dxC):.4f}')
uc, vc = interp_pair_to_center(d0.U, d0.V, grid)
uc, vc = sq(uc), sq(vc)
speed = np.hypot(uc, vc)
print('speed |u| at centres, ocean :', fmt(stats(speed, ocean), 'm/s'))
print('|U| (model x, raw), ocean   :', fmt(stats(np.abs(sq(d0.U)), np.isfinite(sq(d0.U))), 'm/s'))
print('|V| (model y, raw), ocean   :', fmt(stats(np.abs(sq(d0.V)), np.isfinite(sq(d0.V))), 'm/s'))
di = np.abs(uc) * DT / dxC; dj = np.abs(vc) * DT / dyC
disp = np.hypot(di, dj)
print('hourly |di| (cells), ocean  :', fmt(stats(di, ocean), 'cells'))
print('hourly |dj| (cells), ocean  :', fmt(stats(dj, ocean), 'cells'))
print('hourly |d| (cells), ocean   :', fmt(stats(disp, ocean), 'cells'))
print(f'fraction of ocean cells with |d| > 1 cell: {np.mean(disp[ocean] > 1):.4f}; '
      f'> 1.5 cells: {np.mean(disp[ocean] > 1.5):.4f}; > 0.5: {np.mean(disp[ocean] > 0.5):.4f}')
print('hourly |d| (cells), front   :', fmt(stats(disp, front), 'cells'))
# speed in the band, for the "at 37N" displacement
print('speed |u| in 36.5-37.5N band :', fmt(stats(speed, band & ocean), 'm/s'))
print('hourly |d| in 36.5-37.5N band:', fmt(stats(disp, band & ocean), 'cells'))

# ---------------------------------------------------------------------------
# Q4  rotation terms and spherical metric term
# ---------------------------------------------------------------------------
hdr('Q4  rotation terms u|grad CS|, u|grad SN| and metric term u tan(phi)/a')
CS = sq(g.CS); SN = sq(g.SN)
print(f'CS: {np.unique(CS).size} distinct float32 values in [{CS.min():.3e}, {CS.max():.3e}]; '
      f'SN: {np.unique(SN).size} distinct in [{SN.min()}, {SN.max()}]')
ang = np.degrees(np.arctan2(SN, CS))
print(f'grid angle atan2(SN,CS): min {ang.min():.3f}, max {ang.max():.3f} deg  '
      f'(range {ang.max()-ang.min():.3f} deg across the tile); CS^2+SN^2 in '
      f'[{np.min(CS**2+SN**2):.6f}, {np.max(CS**2+SN**2):.6f}]')
csx, csy = [sq(a) for a in calculate_native_gradient_tracer(g.CS, g, grid)]
snx, sny = [sq(a) for a in calculate_native_gradient_tracer(g.SN, g, grid)]
gCS = np.hypot(csx, csy); gSN = np.hypot(snx, sny)
# raw finite differences as a cross-check (no rotation, no interp)
gCS_raw = np.hypot(np.gradient(CS, axis=1) / dxC, np.gradient(CS, axis=0) / dyC)
print('|grad CS| (dbof op), interior :', fmt(stats(gCS, interior), '1/m'))
print('|grad CS| (np.gradient)       :', fmt(stats(gCS_raw, interior), '1/m'))
print('|grad SN| (dbof op), interior :', fmt(stats(gSN, interior), '1/m'))
# measured strain magnitude |sigma| = sqrt(sigma_n^2 + sigma_s^2) at centres
sv = calculate_native_strain_vorticity(d0.U, d0.V, g, grid)
sig_n = sq(sv['strain_normal_center'])
sig_s2 = sq(interp_corner_squared(sv['strain_shear_corner'] ** 2, grid))
sigma = np.sqrt(sig_n ** 2 + sig_s2)
print('|sigma| measured, interior    :', fmt(stats(sigma, interior), 's^-1'))
print('|sigma| measured, front       :', fmt(stats(sigma, front), 's^-1'))
print('|sigma| from Jacobian (geo), interior:', fmt(stats(np.hypot(ux - vy, vx + uy), interior), 's^-1'))
rotCS = speed * gCS; rotSN = speed * gSN
metric = speed * np.tan(np.radians(YC)) / A_EARTH
print('u|grad CS|, interior          :', fmt(stats(rotCS, interior), 's^-1'))
print('u|grad SN|, interior          :', fmt(stats(rotSN, interior), 's^-1'))
print('u tan(phi)/a, interior        :', fmt(stats(metric, interior), 's^-1'))
for lab, term in (('u|grad CS|', rotCS), ('u|grad SN|', rotSN), ('u tan(phi)/a', metric)):
    for mlab, m in (('interior', interior), ('front', front)):
        r_nom = term[m] / NOMINAL_STRAIN
        r_meas = term[m] / sigma[m]
        r_meas = r_meas[np.isfinite(r_meas)]
        print(f'  {lab:13s} {mlab:8s}: vs 1e-5: median {np.nanmedian(r_nom)*100:.4f}%  p99 {np.nanpercentile(r_nom, 99)*100:.4f}%  '
              f'max {np.nanmax(r_nom)*100:.4f}% | vs measured |sigma|: median {np.median(r_meas)*100:.4f}%  '
              f'p99 {np.percentile(r_meas, 99)*100:.4f}%')
# ratio of typical magnitudes (robust to the small-|sigma| tail)
print(f'  ratio of medians: u|grad CS| / |sigma| = {np.nanmedian(rotCS[interior])/np.nanmedian(sigma[interior])*100:.4f}%, '
      f'u|grad SN| / |sigma| = {np.nanmedian(rotSN[interior])/np.nanmedian(sigma[interior])*100:.4f}%, '
      f'metric / |sigma| = {np.nanmedian(metric[interior])/np.nanmedian(sigma[interior])*100:.4f}%')
# ---------------------------------------------------------------------------
# Q5  implicit diffusivity of the 7th-order upwind kernel (tempAdvScheme=7)
# ---------------------------------------------------------------------------
hdr('Q5  kappa_num of the 7th-order upwind kernel vs scale (no data needed)')
# Face value for u>0 from the 7th-order upwind-biased stencil i-3..i+3
# (the unlimited kernel of MITgcm's OS7MP, gad_os7mp_adv_x.F).  Its
# semi-discrete Fourier symbol gives the exact damping rate per wavenumber;
# the modified-equation coefficient |u| dx^7 k^6 / 280 is its small-kdx limit.
c7 = np.array([-3, 25, -101, 319, 214, -38, 4]) / 420.0
m7 = np.arange(-3, 4)


def kernel_symbol(kdx):
    z = np.exp(1j * kdx * m7)
    return np.sum(c7 * z) * (1 - np.exp(-1j * kdx))   # F_{i+1/2} - F_{i-1/2}


print(f"{'lambda/dx':>9s} {'kdx':>6s} {'kappa/(|u|dx) exact':>20s} {'mod.eq. k^6/280':>16s}")
for lam in (2, 3, 4, 5.6, 8, 11, 20):
    kdx = 2 * np.pi / lam
    print(f'{lam:9.1f} {kdx:6.3f} {kernel_symbol(kdx).real/kdx**2:20.3e} {kdx**6/280:16.3e}')
dx_med = np.median(dxC)
u_lab = [('median', np.nanmedian(speed[ocean])), ('p90', np.nanpercentile(speed[ocean], 90)),
         ('p99', np.nanpercentile(speed[ocean], 99))]
print(f'dx = median dxC = {dx_med:.0f} m; Courant u dt/dx at dt=25 s: '
      + ', '.join(f'{lab} {u*25/dx_med:.4f}' for lab, u in u_lab))
for lab, u in u_lab:
    print(f'|u| {lab} = {u:.3f} m/s  (|u| dx = {u*dx_med:.0f} m^2/s):')
    for lam_km in (2 * dx_med / 1e3, 4 * dx_med / 1e3, 10.0, 20.0):
        k = 2 * np.pi / (lam_km * 1e3); kdx = k * dx_med
        kap = kernel_symbol(kdx).real / kdx ** 2 * u * dx_med
        rate = 2 * kap * k ** 2                      # damping rate of G = |grad b|^2
        print(f'   lambda {lam_km:5.1f} km: kappa {kap:8.3g} m^2/s  2 kappa k^2 {rate:9.3g} s^-1  '
              f'e-fold(G) {1/rate/3600:9.3g} h  G left after 72 h {np.exp(-rate*72*3600):.3f}')
    kap1 = u * dx_med / 2
    print(f'   limiter-saturated bound (1st-order upwind, kappa=|u|dx/2={kap1:.0f} m^2/s): '
          f'e-fold(G) at 10 km {1/(2*kap1*(2*np.pi/1e4)**2)/3600:.2f} h')
print(f'\nall done in {time.time()-t_start:.0f} s')
