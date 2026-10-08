""" M2 task 3: QA of the 72-hour OSN series -> figs/m2_qa_series.png.

A time-series sanity pass over ``data/tile330_raw_20120702T00_72h.zarr``
(no physics: no ``F``, no ``G`` budgets, no slopes -- that is M3):

  * tile-mean and ocean percentiles (p5 / p50 / p95) per hour of ``Eta``
    (the tide: range and dominant period from a sinusoid fit and the FFT
    of the tile mean), ``KPPhbl`` (the diurnal cycle: 24-h amplitude and
    the local solar time of its maximum), ``Theta`` and the centred speed
    ``|u|`` (``semilag.centre_velocities``; speed is rotation-invariant,
    the tile-mean eastward / northward components use the face-10
    orientation ``u_east = V, v_north = -U``, coding §3.1);
  * the land-NaN fraction per hour and variable, which must be constant
    and equal to the ``hFacC`` / ``hFacW`` / ``hFacS`` masks;
  * ``oceTAUX`` / ``oceTAUY`` re-masked with ``hFacW`` / ``hFacS`` (M0
    task 3): finite values on land before and after (after must be 0);
  * the hourly displacement distribution over all 71 hour pairs, from
    ``semilag.departure_index`` with its defaults (``vel_order = 3``,
    ``n_iter = 3``) on the time-midpoint velocity, against M1's hour-0
    numbers (median 0.364, p99 1.25, max 2.09 cells);
  * edge support: whether a departure's order-3 support (the five-point
    tracer stencil at the departure point, each point on a 4-node Lagrange
    kernel: nodes ``floor(p) - 2 .. floor(p) + 3`` per axis) leaves the
    finite part of the tile -- the tile itself at ``L = 0``, or the
    ``L/2``-cell NaN rim ``operators.lowpass`` leaves at the tile edge --
    counted on ``mask_analysis`` per pair, and cross-checked against the
    actual NaN count of ``semilag.measured_DGDt`` on the worst pair;
  * anomaly flags: frozen fields (identical consecutive hours), a missing
    tide, NaN-pattern changes, outliers (jumps in the tile-mean series).

Per-hour and per-pair results are cached as JSON in ``data/`` (git-ignored)
so an interrupted run resumes and the figure can be redrawn without
recomputing (``--figure-only``).  Run:

    timeout 300 ~/miniforge3/envs/frontogenesis/bin/python m2_qa.py [--pairs a:b] [--figure-only]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from scipy import optimize                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import osn_tiles as ot                                           # noqa: E402
import masking as mk                                             # noqa: E402
import operators as op                                           # noqa: E402
import semilag as sl                                             # noqa: E402
from validate_figs import COL, _save                             # noqa: E402

RAW72 = ot.DATA_DIR / 'tile330_raw_20120702T00_72h.zarr'
MASKS = ot.DATA_DIR / 'tile330_masks.nc'
HOURS_JSON = ot.DATA_DIR / 'm2_qa_hours.json'
PAIRS_JSON = ot.DATA_DIR / 'm2_qa_pairs.json'
SUMMARY_JSON = ot.DATA_DIR / 'm2_qa_summary.json'
PNG = 'm2_qa_series.png'

VARS = ot.CORE_VARS + ot.WIND_VARS
LAND_OF = {v: 'hFacC' for v in VARS}
LAND_OF.update(U='hFacW', V='hFacS')              # oceTAU* arrive on the centred mask (M0 task 3)
TAU_MASK = dict(oceTAUX='hFacW', oceTAUY='hFacS')  # ... and are re-masked with these
PCT = (5, 50, 95)
ORDER = 3                                          # interp order of b (coding §4.4)
LS_FOR_EDGE = (0, 2, 4, 8)                         # lowpass scales whose edge rim is tested
EDGE_CELLS = 7                                     # masking.edge_mask default
M1_HOUR0 = dict(median=0.364, p99=1.25, max=2.09)  # M1 task 3 / task 6, ocean, hour 0-1
M2_PERIOD_H = 12.4206                              # M2 tidal period [h]
K1_PERIOD_H = 23.9345


def _f(x):
    return float(x)


def _pcts(a, mask):
    v = a[mask]
    v = v[np.isfinite(v)]
    return dict(mean=_f(v.mean()), p5=_f(np.percentile(v, 5)), p50=_f(np.percentile(v, 50)),
                p95=_f(np.percentile(v, 95)), min=_f(v.min()), max=_f(v.max()))


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def open_inputs():
    """The 72-hour store (lazy), the grid with ``face`` (for the operators)
    and without (positional masks), the xgcm grid and the analysis mask."""
    raw = xr.open_zarr(RAW72)
    g = ot.open_grid(with_face=True)
    grid = ot.build_xgcm(g)
    g2 = g.squeeze('face')
    land = {hf: np.asarray(g2[hf].values) == 0 for hf in ('hFacC', 'hFacW', 'hFacS')}
    masks = mk.open_masks(MASKS)
    ana = masks['mask_analysis'].values
    return raw, g, grid, land, ana


def hour_ds(raw, g, k):
    """Hour ``k`` merged with the grid on ``(face, j, i)``, float64 -- the
    layout every dbof operator expects (coding §3.2)."""
    h = raw.isel(time=k).load().expand_dims('face')
    return xr.merge([h, g], compat='override', combine_attrs='override').astype('float64')


# ---------------------------------------------------------------------------
# per-hour statistics
# ---------------------------------------------------------------------------
def hour_stats(ds, land, grid, prev=None, nan0=None):
    """Everything the figure needs from one hour, plus the anomaly checks
    against the previous hour (``prev``: dict var -> array) and hour 0's
    NaN pattern (``nan0``: dict var -> bool array)."""
    out = dict(time=str(ds.time.values)[:19], niter=int(ds.niter.values))
    arrays = {v: ds[v].values[0] for v in VARS}
    ocean = ~land['hFacC']
    # land-NaN fraction and pattern per variable
    nan = {v: np.isnan(arrays[v]) for v in VARS}
    out['nan_fraction'] = {v: _f(nan[v].mean()) for v in VARS}
    out['nan_mismatch_vs_hfac'] = {v: int(np.sum(nan[v] != land[LAND_OF[v]])) for v in VARS}
    out['nan_pattern_changed'] = {v: bool(np.any(nan[v] != nan0[v])) for v in VARS} if nan0 else None
    # tau re-masking (M0 task 3): finite on the staggered land before / after
    tau = {}
    for v, hf in TAU_MASK.items():
        a = arrays[v]
        before = int(np.sum(np.isfinite(a) & land[hf]))
        remasked = np.where(land[hf], np.nan, a)
        after = int(np.sum(np.isfinite(remasked) & land[hf]))
        tau[v] = dict(finite_on_land_before=before, finite_on_land_after=after,
                      finite_on_centred_land=int(np.sum(np.isfinite(a) & land['hFacC'])))
    out['tau'] = tau
    # wind-stress magnitude at the centres from the re-masked components (two-point means), ocean
    tx = np.where(land['hFacW'], np.nan, arrays['oceTAUX'])
    ty = np.where(land['hFacS'], np.nan, arrays['oceTAUY'])
    txc = np.full_like(tx, np.nan); txc[:, :-1] = 0.5 * (tx[:, :-1] + tx[:, 1:])
    tyc = np.full_like(ty, np.nan); tyc[:-1, :] = 0.5 * (ty[:-1, :] + ty[1:, :])
    tmag = np.hypot(txc, tyc)
    out['tau_mag'] = _pcts(tmag, np.isfinite(tmag))
    # tile statistics over the ocean
    for v in ('Eta', 'KPPhbl', 'Theta', 'Salt', 'W'):
        out[v] = _pcts(arrays[v], ocean)
    # centred speed (model basis; |u| is rotation-invariant); the tile-mean
    # geographic components use the face-10 orientation u_east = V, v_north = -U
    u_c, v_c = sl.centre_velocities(ds.U, ds.V, ds, grid)
    uc, vc = u_c.values[0], v_c.values[0]
    speed = np.hypot(uc, vc)
    out['speed'] = _pcts(speed, np.isfinite(speed))
    out['u_east_mean'] = _f(np.nanmean(vc))
    out['v_north_mean'] = _f(-np.nanmean(uc))
    # frozen-field check: identical to the previous hour?
    if prev is not None:
        fr = {}
        for v in VARS:
            a, b = arrays[v], prev[v]
            same = np.array_equal(a, b, equal_nan=True)
            fin = np.isfinite(a) & np.isfinite(b)
            frac = _f(np.mean(a[fin] == b[fin]))
            fr[v] = dict(identical=bool(same), identical_cell_fraction=frac,
                         rms_change=_f(np.sqrt(np.mean((a[fin] - b[fin]) ** 2))))
        out['frozen'] = fr
    return out, arrays, nan


# ---------------------------------------------------------------------------
# per-pair displacement and edge support
# ---------------------------------------------------------------------------
def support_leaves(di, dj, lo, hi_i, hi_j, cells):
    """Cells (bool array) whose departure-point order-3 tracer stencil
    support -- nodes ``floor(p) - 2 .. floor(p) + 3`` per axis, ``p`` the
    departure index -- is not inside ``[lo, hi]`` along either axis, or
    whose departure is NaN.  ``cells`` selects the cells counted."""
    nj, ni = di.shape
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')
    pi, pj = ii - di, jj - dj
    bad = ~(np.isfinite(pi) & np.isfinite(pj))
    fi, fj = np.floor(np.where(bad, 0, pi)), np.floor(np.where(bad, 0, pj))
    half = (ORDER + 1) // 2                       # 2 for the cubic kernel
    lo_i, hi_i_ = fi - half, fi + half + 1        # the +/-1 stencil neighbours widen it by 1
    lo_j, hi_j_ = fj - half, fj + half + 1
    bad |= (lo_i < lo) | (hi_i_ > hi_i) | (lo_j < lo) | (hi_j_ > hi_j)
    return bad & cells


def edge_distance(shape):
    nj, ni = shape
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')
    return np.minimum.reduce([jj, nj - 1 - jj, ii, ni - 1 - ii])


def pair_stats(ds0, ds1, grid, land, ana, measured=False):
    """Displacement statistics for the pair and the edge-support counts on
    ``mask_analysis`` for each ``L`` in ``LS_FOR_EDGE``.  ``measured=True``
    also runs ``semilag.measured_DGDt`` on JMD95 ``b`` (raw and low-passed
    ``L = 8``) and counts its NaN on the analysis mask -- the empirical
    cross-check, used on the worst pair."""
    U_mid = sl.midpoint_time(ds0.U, ds1.U)
    V_mid = sl.midpoint_time(ds0.V, ds1.V)
    u_c, v_c = sl.centre_velocities(U_mid, V_mid, ds0, grid)
    di, dj = sl.departure_index(u_c, v_c, ds0)             # defaults: dt 3600, n_iter 3, vel_order 3
    di, dj = di.values[0], dj.values[0]
    d = np.hypot(di, dj)
    ocean = ~land['hFacC']
    fin = np.isfinite(d)
    out = dict(t0=str(ds0.time.values)[:19], t1=str(ds1.time.values)[:19])
    for name, m in (('ocean', ocean & fin), ('analysis', ana & fin)):
        v = d[m]
        out[name] = dict(n=int(m.sum()), median=_f(np.median(v)), p90=_f(np.percentile(v, 90)),
                         p99=_f(np.percentile(v, 99)), max=_f(v.max()),
                         frac_gt_1=_f(np.mean(v > 1)), frac_gt_1p5=_f(np.mean(v > 1.5)))
    out['nan_departure_ocean'] = int(np.sum(ocean & ~fin))
    out['nan_departure_analysis'] = int(np.sum(ana & ~fin))
    out['max_di_analysis'] = _f(np.nanmax(np.abs(di[ana])))
    out['max_dj_analysis'] = _f(np.nanmax(np.abs(dj[ana])))
    # edge support: the finite part of a field low-passed at L is [L/2, n-1-L/2]
    nj, ni = d.shape
    edge = edge_distance(d.shape)
    near = edge <= EDGE_CELLS + 2 * ORDER                   # within reach of the tile edge
    es = {}
    for L in LS_FOR_EDGE:
        rim = L // 2
        bad = support_leaves(di, dj, rim, ni - 1 - rim, nj - 1 - rim, ana)
        bad_edge = bad & near & np.isfinite(d)              # attributable to the tile edge, not NaN departures
        es[str(L)] = dict(n_analysis_cells=int(bad_edge.sum()),
                          min_edge_distance=int(edge[bad_edge].min()) if bad_edge.any() else None)
    out['edge_support'] = es
    if measured:
        b0, b1 = op.buoyancy(ds0), op.buoyancy(ds1)
        me = {}
        for L in (0, 8):
            bt, btp1 = (op.lowpass(b0, L), op.lowpass(b1, L)) if L else (b0, b1)
            m = sl.measured_DGDt(bt, btp1, U_mid, V_mid, ds0, grid)
            nanm = ~np.isfinite(m.values[0]) & ana
            me[str(L)] = dict(nan_on_analysis=int(nanm.sum()),
                              nan_near_edge=int((nanm & near).sum()),
                              min_edge_distance=int(edge[nanm & near].min()) if (nanm & near).any() else None)
        out['measured_DGDt_nan'] = me
    return out


# ---------------------------------------------------------------------------
# the series fits
# ---------------------------------------------------------------------------
def fit_sinusoid(t_h, y, period_h, free_period=False):
    """Least-squares ``c + A cos(2 pi (t - t_max)/P)`` (+ a linear trend):
    returns the amplitude, the phase of the maximum (hours) and the
    period.  With ``free_period`` the period is fitted too."""
    t_h, y = np.asarray(t_h, float), np.asarray(y, float)

    def model(t, c, s, a, b, P):
        w = 2 * np.pi / P
        return c + s * t + a * np.cos(w * t) + b * np.sin(w * t)
    p0 = [y.mean(), 0.0, y.std(), 0.0, period_h]
    if free_period:
        popt, _ = optimize.curve_fit(model, t_h, y, p0=p0, maxfev=20000)
    else:
        X = np.column_stack([np.ones_like(t_h), t_h, np.cos(2 * np.pi * t_h / period_h),
                             np.sin(2 * np.pi * t_h / period_h)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        popt = [*coef, period_h]
    c, s, a, b, P = popt
    amp = float(np.hypot(a, b))
    t_max = float((np.arctan2(b, a) / (2 * np.pi) * P) % P)
    resid = y - model(t_h, *popt)
    return dict(amplitude=amp, t_max_h=t_max, period_h=float(P), mean=float(c), trend_per_h=float(s),
                rms_residual=float(np.sqrt(np.mean(resid ** 2))),
                r2=float(1 - np.var(resid) / np.var(y)), fitted=model(t_h, *popt).tolist())


def fft_period(t_h, y):
    """The dominant period of the demeaned, detrended series by FFT (72 h
    gives a 1/72 h^-1 resolution: the peak bin, refined by a parabolic
    interpolation of the spectrum)."""
    y = np.asarray(y, float)
    y = y - np.polyval(np.polyfit(t_h, y, 1), t_h)
    n = y.size
    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, d=t_h[1] - t_h[0])
    k = int(np.argmax(spec[1:]) + 1)
    if 1 <= k < spec.size - 1:
        a, b, c = np.log(spec[k - 1:k + 2])
        dk = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        dk = 0.0
    fpk = f[k] + dk * (f[1] - f[0])
    return dict(period_h=float(1 / fpk), bin_period_h=float(1 / f[k]), resolution_h=float(1 / f[1]),
                periods_h=(1 / f[1:]).tolist(), power=(spec[1:] / spec[1:].max()).tolist())


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def run_hours(raw, g, grid, land, say):
    hours = _load(HOURS_JSON)
    n = raw.sizes['time']
    prev, nan0 = None, None
    t0 = time.time()
    for k in range(n):
        ds = hour_ds(raw, g, k)
        st, arrays, nan = hour_stats(ds, land, grid, prev=prev, nan0=nan0)
        if nan0 is None:
            nan0 = nan
        hours[str(k)] = st
        prev = arrays
        if k % 12 == 0 or k == n - 1:
            say(f'hour {k:2d} {st["time"]}  Eta mean {st["Eta"]["mean"]:+.3f}  KPPhbl mean {st["KPPhbl"]["mean"]:.1f}  '
                f'|u| p50 {st["speed"]["p50"]:.3f}  nan(Theta) {st["nan_fraction"]["Theta"]:.4f}  '
                f'tau finite-on-land before/after {st["tau"]["oceTAUX"]["finite_on_land_before"]}/'
                f'{st["tau"]["oceTAUX"]["finite_on_land_after"]}  [{time.time() - t0:.0f} s]')
            _dump(HOURS_JSON, hours)
    _dump(HOURS_JSON, hours)
    return hours


def run_pairs(raw, g, grid, land, ana, say, first=0, last=None, measured_pairs=()):
    pairs = _load(PAIRS_JSON)
    n = raw.sizes['time']
    last = n - 1 if last is None else min(last, n - 1)
    t0 = time.time()
    ds1 = None
    for k in range(first, last):
        key = str(k)
        need_meas = k in measured_pairs and 'measured_DGDt_nan' not in pairs.get(key, {})
        if key in pairs and not need_meas:
            continue
        ds0 = ds1 if (ds1 is not None and int(ds1.niter.values) == int(raw.niter.isel(time=k).values)) else hour_ds(raw, g, k)
        ds1 = hour_ds(raw, g, k + 1)
        pairs[key] = pair_stats(ds0, ds1, grid, land, ana, measured=k in measured_pairs)
        p = pairs[key]
        say(f'pair {k:2d}-{k + 1:2d}  ocean median {p["ocean"]["median"]:.3f} p99 {p["ocean"]["p99"]:.3f} '
            f'max {p["ocean"]["max"]:.2f}  edge-support NaN on analysis L=0/8: '
            f'{p["edge_support"]["0"]["n_analysis_cells"]}/{p["edge_support"]["8"]["n_analysis_cells"]}'
            + (f'  measured NaN L=0/8: {p["measured_DGDt_nan"]["0"]["nan_on_analysis"]}/'
               f'{p["measured_DGDt_nan"]["8"]["nan_on_analysis"]}' if 'measured_DGDt_nan' in p else '')
            + f'  [{time.time() - t0:.0f} s]')
        _dump(PAIRS_JSON, pairs)
    return pairs


def summarize(hours, pairs, lon_mean):
    """The QA numbers, the fits and the anomaly flags."""
    ks = sorted(hours, key=int)
    t_h = np.arange(len(ks), dtype=float)
    S = dict(n_hours=len(ks), times=[hours[k]['time'] for k in ks])
    series = {}
    for v in ('Eta', 'KPPhbl', 'Theta', 'speed', 'Salt', 'W', 'tau_mag'):
        series[v] = {q: [hours[k][v][q] for k in ks] for q in ('mean', 'p5', 'p50', 'p95', 'min', 'max')}
    series['u_east_mean'] = [hours[k]['u_east_mean'] for k in ks]
    series['v_north_mean'] = [hours[k]['v_north_mean'] for k in ks]
    S['series'] = series
    # the tide
    eta = np.array(series['Eta']['mean'])
    S['eta'] = dict(range_m=_f(eta.max() - eta.min()), min_m=_f(eta.min()), max_m=_f(eta.max()),
                    mean_m=_f(eta.mean()),
                    fft=fft_period(t_h, eta),
                    fit_M2=fit_sinusoid(t_h, eta, M2_PERIOD_H),
                    fit_free=fit_sinusoid(t_h, eta, M2_PERIOD_H, free_period=True))
    # M2 + K1 together (72 h resolves 12.4 from 23.9 h; S2 is not separable from M2)
    X = np.column_stack([np.ones_like(t_h), t_h] + [f(2 * np.pi * t_h / P) for P in (M2_PERIOD_H, K1_PERIOD_H)
                                                      for f in (np.cos, np.sin)])
    coef, *_ = np.linalg.lstsq(X, eta, rcond=None)
    fit2 = X @ coef
    S['eta']['fit_M2_K1'] = dict(amp_M2=_f(np.hypot(coef[2], coef[3])), amp_K1=_f(np.hypot(coef[4], coef[5])),
                                 rms_residual=_f(np.sqrt(np.mean((eta - fit2) ** 2))),
                                 r2=_f(1 - np.var(eta - fit2) / np.var(eta)), fitted=fit2.tolist())
    # the diurnal cycle of KPPhbl; local solar time = UTC + lon/15
    h = np.array(series['KPPhbl']['mean'])
    fitd = fit_sinusoid(t_h, h, 24.0)
    utc_max = fitd['t_max_h'] % 24.0                        # hours after 00 UTC
    S['kpphbl'] = dict(fit_24h=fitd, lon_mean=_f(lon_mean), utc_of_max_h=_f(utc_max),
                       local_solar_of_max_h=_f((utc_max + lon_mean / 15.0) % 24.0),
                       pdt_of_max_h=_f((utc_max - 7.0) % 24.0),
                       daily=[dict(day=d, mean=_f(h[24 * d:24 * d + 24].mean()), min=_f(h[24 * d:24 * d + 24].min()),
                                   max=_f(h[24 * d:24 * d + 24].max()),
                                   utc_of_max=int(np.argmax(h[24 * d:24 * d + 24])),
                                   utc_of_min=int(np.argmin(h[24 * d:24 * d + 24]))) for d in range(len(ks) // 24)])
    # land NaN, tau
    nf = {v: sorted({round(hours[k]['nan_fraction'][v], 10) for k in ks}) for v in VARS}
    S['nan_fraction'] = {v: dict(values=nf[v], constant=len(nf[v]) == 1) for v in VARS}
    S['nan_mismatch_vs_hfac_max'] = {v: max(hours[k]['nan_mismatch_vs_hfac'][v] for k in ks) for v in VARS}
    S['nan_pattern_changed_hours'] = {v: [k for k in ks[1:] if hours[k]['nan_pattern_changed'][v]] for v in VARS}
    S['tau'] = {v: dict(before=sorted({hours[k]['tau'][v]['finite_on_land_before'] for k in ks}),
                        after=sorted({hours[k]['tau'][v]['finite_on_land_after'] for k in ks}),
                        total_before=sum(hours[k]['tau'][v]['finite_on_land_before'] for k in ks),
                        total_after=sum(hours[k]['tau'][v]['finite_on_land_after'] for k in ks))
                for v in TAU_MASK}
    # frozen fields
    S['frozen'] = {v: [k for k in ks[1:] if hours[k]['frozen'][v]['identical']] for v in VARS}
    S['max_identical_cell_fraction'] = {v: max(hours[k]['frozen'][v]['identical_cell_fraction'] for k in ks[1:])
                                        for v in VARS}
    # outliers: hourly jumps of the tile means against a robust scale
    jumps = {}
    for v in ('Eta', 'KPPhbl', 'Theta', 'speed', 'Salt', 'W'):
        y = np.array(series[v]['mean'])
        dy = np.diff(y)
        mad = np.median(np.abs(dy - np.median(dy))) * 1.4826
        z = (dy - np.median(dy)) / mad if mad > 0 else np.zeros_like(dy)
        jumps[v] = dict(max_abs_z=_f(np.max(np.abs(z))), hours_gt_5=[int(i + 1) for i in np.where(np.abs(z) > 5)[0]])
    S['jumps'] = jumps
    # displacement envelope
    pk = sorted(pairs, key=int)
    for name in ('ocean', 'analysis'):
        med = np.array([pairs[k][name]['median'] for k in pk])
        p99 = np.array([pairs[k][name]['p99'] for k in pk])
        mx = np.array([pairs[k][name]['max'] for k in pk])
        S[f'displacement_{name}'] = dict(
            n_pairs=len(pk), median=dict(min=_f(med.min()), max=_f(med.max()), mean=_f(med.mean())),
            p99=dict(min=_f(p99.min()), max=_f(p99.max()), mean=_f(p99.mean())),
            max=dict(min=_f(mx.min()), max=_f(mx.max()), mean=_f(mx.mean())),
            window_max=_f(mx.max()), worst_pair=int(pk[int(np.argmax(mx))]),
            worst_pair_time=pairs[pk[int(np.argmax(mx))]]['t0'],
            hour0=dict(median=_f(med[0]), p99=_f(p99[0]), max=_f(mx[0])),
            series=dict(median=med.tolist(), p99=p99.tolist(), max=mx.tolist()))
    es = {}
    for L in LS_FOR_EDGE:
        cnt = np.array([pairs[k]['edge_support'][str(L)]['n_analysis_cells'] for k in pk])
        es[str(L)] = dict(total=int(cnt.sum()), max=int(cnt.max()), n_pairs_nonzero=int((cnt > 0).sum()),
                          worst_pair=int(pk[int(np.argmax(cnt))]), series=cnt.tolist())
    S['edge_support'] = es
    S['measured_DGDt_nan'] = {k: pairs[k]['measured_DGDt_nan'] for k in pk if 'measured_DGDt_nan' in pairs[k]}
    S['nan_departure_analysis_max'] = max(pairs[k]['nan_departure_analysis'] for k in pk)
    # anomaly verdicts
    S['flags'] = dict(
        frozen_fields=any(S['frozen'][v] for v in VARS),
        missing_tide=S['eta']['range_m'] < 0.2 or S['eta']['fit_M2']['r2'] < 0.5,
        nan_pattern_changed=any(S['nan_pattern_changed_hours'][v] for v in VARS),
        nan_fraction_not_constant=any(not S['nan_fraction'][v]['constant'] for v in VARS),
        tau_finite_on_land_after_remask=any(S['tau'][v]['total_after'] for v in TAU_MASK),
        outlier_jumps=any(jumps[v]['hours_gt_5'] for v in jumps),
        edge_support_left_at_L8=es['8']['total'] > 0,
        edge_support_left_at_L0=es['0']['total'] > 0)
    return S


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def _band(ax, t, s, color, label):
    ax.fill_between(t, s['p5'], s['p95'], color=color, alpha=0.18, lw=0, label='p5-p95 (ocean)')
    ax.plot(t, s['p50'], '-', color=color, lw=1.0, alpha=0.8, label='p50')
    ax.plot(t, s['mean'], '-', color='black', lw=1.6, label=label)


def _days(ax, n):
    for d in range(0, n + 1, 24):
        ax.axvline(d, color='#888888', lw=0.6, ls=':')
    ax.set_xlim(0, n - 1)


def fig_qa(S, pairs):
    ser = S['series']
    n = S['n_hours']
    t = np.arange(n)
    tp = np.arange(n - 1) + 0.5
    fig, axs = plt.subplots(4, 2, figsize=(18, 17))
    (pa, pb), (pc, pd), (pe, pf), (pg, ph) = axs
    # (a) Eta
    _band(pa, t, ser['Eta'], COL['order3'], 'tile mean')
    e = S['eta']
    pa.plot(t, e['fit_M2_K1']['fitted'], '--', color=COL['red'], lw=1.2,
            label=f'M2 + K1 fit: amp {e["fit_M2_K1"]["amp_M2"]:.3f} / {e["fit_M2_K1"]["amp_K1"]:.3f} m, '
                  f'r$^2$ {e["fit_M2_K1"]["r2"]:.3f}')
    pa.set_ylabel('Eta [m]')
    pa.set_title(f'(a) Eta: tile-mean range {e["range_m"]:.3f} m ({e["min_m"]:.3f} to {e["max_m"]:.3f}); '
                 f'free-period fit {e["fit_free"]["period_h"]:.2f} h (amp {e["fit_free"]["amplitude"]:.3f} m), '
                 f'FFT peak {e["fft"]["period_h"]:.1f} h (bin {e["fft"]["bin_period_h"]:.1f} h)', fontsize=10)
    pa.legend(fontsize=7.5, loc='lower left', ncol=2); pa.grid(alpha=0.3); _days(pa, n)
    # (b) KPPhbl
    _band(pb, t, ser['KPPhbl'], COL['order5'], 'tile mean')
    k = S['kpphbl']
    pb.plot(t, k['fit_24h']['fitted'], '--', color=COL['red'], lw=1.2,
            label=f'24-h fit: amplitude {k["fit_24h"]["amplitude"]:.1f} m, max at {k["utc_of_max_h"]:.1f} UTC = '
                  f'{k["local_solar_of_max_h"]:.1f} h local solar ({k["pdt_of_max_h"]:.1f} PDT), r$^2$ {k["fit_24h"]["r2"]:.2f}')
    pb.set_ylabel('KPPhbl [m]')
    pb.set_yscale('log')
    pbt = pb.twinx()
    pbt.plot(t, ser['tau_mag']['mean'], '-', color=COL['grey'], lw=1.2, label='tile-mean |tau| (re-masked), right axis')
    pbt.set_ylabel('|tau| [N m$^{-2}$]', color='#555555'); pbt.set_ylim(0, max(ser['tau_mag']['mean']) * 1.8)
    pbt.legend(fontsize=7.5, loc='lower left')
    pb.set_title(f'(b) KPPhbl: daily tile-mean min / max '
                 + ', '.join(f'{d["min"]:.1f} / {d["max"]:.1f} m' for d in k['daily']) +
                 f'  (lon {k["lon_mean"]:.1f}: local solar = UTC {k["lon_mean"] / 15:+.1f} h)', fontsize=10)
    pb.legend(fontsize=7.5, loc='upper right', ncol=2); pb.grid(alpha=0.3, which='both'); _days(pb, n)
    # (c) Theta
    _band(pc, t, ser['Theta'], COL['order1'], 'tile mean')
    th = np.array(ser['Theta']['mean'])
    pc.set_ylabel('Theta [degC]')
    pc.set_title(f'(c) Theta: tile mean {th.min():.3f}-{th.max():.3f} degC, hourly jump max |z| {S["jumps"]["Theta"]["max_abs_z"]:.1f}',
                 fontsize=10)
    pc.legend(fontsize=7.5, loc='upper left', ncol=3); pc.grid(alpha=0.3); _days(pc, n)
    # (d) |u|
    _band(pd, t, ser['speed'], COL['purple'], 'tile mean')
    pd.plot(t, ser['speed']['max'], ':', color=COL['purple'], lw=1.0, label='max')
    pd.plot(t, ser['u_east_mean'], '-', color=COL['order5'], lw=1.0, label='tile-mean u_east (= V)')
    pd.plot(t, ser['v_north_mean'], '-', color=COL['order1'], lw=1.0, label='tile-mean v_north (= -U)')
    pd.axhline(0, color='#888888', lw=0.5)
    sp = ser['speed']
    pd.set_ylabel('|u| at centres [m/s]')
    pd.set_title(f'(d) centred speed: p50 {min(sp["p50"]):.3f}-{max(sp["p50"]):.3f}, p95 {min(sp["p95"]):.3f}-{max(sp["p95"]):.3f}, '
                 f'max {min(sp["max"]):.2f}-{max(sp["max"]):.2f} m/s (face 10: u_east = V, v_north = -U)', fontsize=10)
    pd.legend(fontsize=7.5, loc='upper left', ncol=3); pd.grid(alpha=0.3); _days(pd, n)
    # (e) land-NaN fraction and tau
    for v, c in zip(VARS, plt.cm.tab10(np.linspace(0, 1, len(VARS)))):
        pe.plot(t, [S['series_nan'][v][i] for i in range(n)], '-', color=c, lw=1.0, label=v)
    pe.set_ylabel('NaN fraction')
    nfc = all(S['nan_fraction'][v]['constant'] for v in VARS)
    pe.set_title(f'(e) land-NaN fraction per hour: {"constant" if nfc else "NOT CONSTANT"} '
                 f'(centred {S["nan_fraction"]["Theta"]["values"][0]:.4f}, U {S["nan_fraction"]["U"]["values"][0]:.4f}, '
                 f'V {S["nan_fraction"]["V"]["values"][0]:.4f}); max mismatch vs hFac {max(S["nan_mismatch_vs_hfac_max"].values())} cells\n'
                 f'oceTAUX / oceTAUY finite on hFacW / hFacS land per hour: before re-mask '
                 f'{S["tau"]["oceTAUX"]["before"]} / {S["tau"]["oceTAUY"]["before"]}, after '
                 f'{S["tau"]["oceTAUX"]["after"]} / {S["tau"]["oceTAUY"]["after"]}', fontsize=9)
    pe.legend(fontsize=7, loc='center right', ncol=3); pe.grid(alpha=0.3); _days(pe, n)
    # (f) displacement vs time
    do = S['displacement_ocean']['series']
    da = S['displacement_analysis']['series']
    pf.plot(tp, do['max'], '-', color=COL['red'], lw=1.4, label='max (ocean)')
    pf.plot(tp, do['p99'], '-', color=COL['order3'], lw=1.4, label='p99 (ocean)')
    pf.plot(tp, do['median'], '-', color='black', lw=1.4, label='median (ocean)')
    pf.plot(tp, da['max'], '--', color=COL['red'], lw=0.9, label='max (mask_analysis)')
    pf.plot(tp, da['p99'], '--', color=COL['order3'], lw=0.9, label='p99 (mask_analysis)')
    for key, ls in (('max', '-'), ('p99', '-'), ('median', '-')):
        pf.axhline(M1_HOUR0[key], color='#888888', lw=0.8, ls=':')
        pf.text(n - 1.2, M1_HOUR0[key], f'M1 hour 0: {M1_HOUR0[key]}', fontsize=7, ha='right', va='bottom', color='#555555')
    D = S['displacement_ocean']
    A = S['displacement_analysis']
    pf.set_ylabel('hourly displacement [cells]')
    pf.set_title(f'(f) departure_index (vel_order 3, n_iter 3) on the midpoint velocity, 71 pairs; ocean: '
                 f'median {D["median"]["min"]:.3f}-{D["median"]["max"]:.3f}, p99 {D["p99"]["min"]:.2f}-{D["p99"]["max"]:.2f},\n'
                 f'window max {D["window_max"]:.2f} cells (pair {D["worst_pair"]}, {D["worst_pair_time"][5:13]}, Gulf of California); '
                 f'mask_analysis: median {A["median"]["min"]:.3f}-{A["median"]["max"]:.3f}, p99 {A["p99"]["min"]:.2f}-{A["p99"]["max"]:.2f}, '
                 f'max {A["window_max"]:.2f}', fontsize=9)
    pf.legend(fontsize=7.5, loc='upper left', ncol=2); pf.grid(alpha=0.3); _days(pf, n)
    # (g) edge support
    es = S['edge_support']
    for L, c in zip(LS_FOR_EDGE, (COL['black'], COL['order5'], COL['order3'], COL['red'])):
        pg.plot(tp, es[str(L)]['series'], '-', color=c, lw=1.3,
                label=f'L = {L} (rim {L // 2}): total {es[str(L)]["total"]}, max {es[str(L)]["max"]} cells')
    pg.set_ylabel('mask_analysis cells whose departure\nsupport leaves the finite tile')
    mm = S['measured_DGDt_nan']
    agree = all(m['8']['nan_on_analysis'] == pairs[k]['edge_support']['8']['n_analysis_cells'] and m['0']['nan_on_analysis'] == 0
                for k, m in mm.items())
    pg.set_title(f'(g) order-3 departure support (nodes floor(p)-2..floor(p)+3) leaving the finite tile, on mask_analysis '
                 f'(edge_cells = {EDGE_CELLS}): 0 cells at L <= 4;\nat L = 8 (rim 4) {es["8"]["total"]} cells over 71 pairs, '
                 f'{min(es["8"]["series"])}-{es["8"]["max"]} per pair (worst pair {es["8"]["worst_pair"]}); '
                 f'semilag.measured_DGDt NaN count on pairs {", ".join(mm)} {"equals" if agree else "DIFFERS FROM"} this count', fontsize=9)
    pg.legend(fontsize=7.5, loc='upper left'); pg.grid(alpha=0.3); _days(pg, n)
    # (h) Eta spectrum and flags
    fft = e['fft']
    pers, pw = np.array(fft['periods_h']), np.array(fft['power'])
    ph.semilogx(pers, pw, 'o-', color=COL['order3'], lw=1.0, ms=3, label='FFT power of the Eta tile mean (Hann, detrended)')
    for P, lab in ((M2_PERIOD_H, 'M2 12.42 h'), (K1_PERIOD_H, 'K1 23.93 h'), (24.0, '24 h')):
        ph.axvline(P, color=COL['red'] if P < 20 else COL['order5'], lw=0.8, ls='--')
        ph.text(P, 1.02, lab, fontsize=7, ha='center', color='#555555')
    ph.set_xlabel('period [h]'); ph.set_ylabel('relative power')
    ph.set_ylim(0, 1.1)
    fl = S['flags']
    txt = '\n'.join(f'{"FLAG" if v else "ok  "}  {k}' for k, v in fl.items())
    ph.text(0.03, 0.95, txt, transform=ph.transAxes, fontsize=8, family='monospace', va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='#888888', alpha=0.9))
    ph.set_title(f'(h) Eta spectrum ({fft["resolution_h"]:.0f}-h resolution) and the anomaly flags; frozen hours: '
                 f'{sum(len(S["frozen"][v]) for v in VARS)}, NaN-pattern changes: '
                 f'{sum(len(S["nan_pattern_changed_hours"][v]) for v in VARS)}, jumps |z| > 5: '
                 f'{sum(len(S["jumps"][v]["hours_gt_5"]) for v in S["jumps"])}', fontsize=9.5)
    ph.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(0.0, 0.42)); ph.grid(alpha=0.3, which='both')
    for ax in (pa, pb, pc, pd, pe, pf, pg):
        ax.set_xlabel('hours since 2012-07-02 00:00 UTC (dotted: day boundaries)')
    fig.suptitle(f'M2 task 3: QA of tile330_raw_20120702T00_72h.zarr -- {S["times"][0]} to {S["times"][-1]} UTC, '
                 f'{n} hours, 71 hour pairs (no F / G budgets: M3)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, PNG)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pairs', default=None, help='a:b -- only pairs a..b-1 this run (resumable)')
    ap.add_argument('--hours-only', action='store_true')
    ap.add_argument('--figure-only', action='store_true')
    ap.add_argument('--measured', default='', help='comma-separated pair indices on which to run measured_DGDt')
    args = ap.parse_args(argv)

    def say(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)

    raw, g, grid, land, ana = open_inputs()
    say(f'{RAW72.name}: {raw.sizes["time"]} hours; mask_analysis {int(ana.sum())} cells')
    measured = tuple(int(x) for x in args.measured.split(',') if x)
    if not args.figure_only:
        if HOURS_JSON.exists() and len(_load(HOURS_JSON)) == raw.sizes['time'] and not args.hours_only:
            say(f'hours: {HOURS_JSON.name} complete, skipping')
        else:
            run_hours(raw, g, grid, land, say)
        if not args.hours_only:
            first, last = (0, None)
            if args.pairs:
                a, b = args.pairs.split(':')
                first, last = int(a), int(b)
            run_pairs(raw, g, grid, land, ana, say, first, last, measured_pairs=measured)
    hours, pairs = _load(HOURS_JSON), _load(PAIRS_JSON)
    if len(hours) < raw.sizes['time'] or len(pairs) < raw.sizes['time'] - 1:
        say(f'incomplete: {len(hours)} hours, {len(pairs)} pairs cached -- re-run to resume')
        return 1
    S = summarize(hours, pairs, float(np.nanmean(g.XC.values)))
    ks = sorted(hours, key=int)
    S['series_nan'] = {v: [hours[k]['nan_fraction'][v] for k in ks] for v in VARS}
    S['png'] = fig_qa(S, pairs)
    _dump(SUMMARY_JSON, S)
    say(f'wrote {S["png"]} and {SUMMARY_JSON.name}')
    e, k = S['eta'], S['kpphbl']
    say(f'Eta: range {e["range_m"]:.3f} m, free-period fit {e["fit_free"]["period_h"]:.2f} h amp {e["fit_free"]["amplitude"]:.3f} m, '
        f'FFT peak {e["fft"]["period_h"]:.1f} h; M2+K1 amp {e["fit_M2_K1"]["amp_M2"]:.3f}/{e["fit_M2_K1"]["amp_K1"]:.3f} m r2 {e["fit_M2_K1"]["r2"]:.3f}')
    say(f'KPPhbl: 24-h amplitude {k["fit_24h"]["amplitude"]:.2f} m, max at {k["utc_of_max_h"]:.2f} UTC = '
        f'{k["local_solar_of_max_h"]:.2f} local solar; daily {k["daily"]}')
    say(f'NaN fraction constant: {all(S["nan_fraction"][v]["constant"] for v in VARS)}; tau: {S["tau"]}')
    say(f'displacement ocean: {S["displacement_ocean"]["median"]} {S["displacement_ocean"]["p99"]} {S["displacement_ocean"]["max"]}; '
        f'worst pair {S["displacement_ocean"]["worst_pair"]}')
    say(f'edge support: { {L: (v["total"], v["max"], v["worst_pair"]) for L, v in S["edge_support"].items()} }')
    say(f'measured_DGDt NaN: {S["measured_DGDt_nan"]}')
    say(f'flags: {S["flags"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
