""" M1 validation gates (coding doc §4.9): the four gates V1-V4 and the two
supporting figures V5-V6.  Each function computes the numbers, returns them
as a dict and (``png=True``) hands them to ``validate_figs.fig_V*`` for the
PNG in ``dev/frontogenesis/figs/``.  Synthetic grids and exact solutions
come from ``synthetic.py``; nothing here is imported by the science path.

Written: ``test_cartesian_deformation`` -> **V1**, ``test_native_metric`` ->
**V2**, ``test_interpolation_bias`` -> **V4** (M1 task 5);
``demo_interp_half_cell`` -> **V5** (task 3); ``qa_land_halo`` -> **V6**
(task 1).  ``test_discrete_null`` -> **V3** is task 6.

The ``test_*`` names are the contract's; ``__test__ = False`` keeps pytest
from collecting them here (``tests/test_validate.py`` runs them).
"""

import numpy as np
import xarray as xr
from scipy import ndimage

import osn_tiles as ot
import masking as mk
import operators as op
import semilag as sl
import synthetic as sy
import validate_figs as vf
import m0_qa_checks as qc
from m1_write_masks import gulf_of_california_check
from dbof.utils import native_gradient as ng
from dbof.preprocessing.calculate_fields import buoyancy_of_field

FIG_DIR = vf.FIG_DIR
RAW = 'tile330_raw_20120702T00_2h.zarr'
SNAPSHOT = '2012-07-02T00:00:00'
DT = sy.DT
R_SPHERE = 6370e3            # MITgcm rSphere; the tile's dxC/dyC give 6370.0 km (V2)
REAL_HOUR_CELLS = sy.REAL_HOUR_CELLS


def _rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a, dtype='float64') ** 2)))


def _front(G, margin):
    """Front pixels: ``G >= 0.2 max`` inside ``margin`` cells of the edge."""
    m = sy.inner(G.shape, margin) & np.isfinite(G)
    return m & (G >= 0.2 * np.nanmax(np.where(m, G, np.nan)))


# ---------------------------------------------------------------------------
# V1: Cartesian deformation
# ---------------------------------------------------------------------------
def test_cartesian_deformation(alpha=1e-5, png=True, n=128, n_steps=8, widths=(2, 3, 4, 6, 8),
                               ref_ell_cells=8, dt=DT, margin=8) -> dict:
    """**V1**: pure deformation ``u = -a x, v = a y`` with the front
    ``b = b0 tanh(x/ell)`` on a uniform ``n x n`` C-grid, both orientations
    (``CS = 1`` and face 10's ``CS = 0, SN = -1``).  ``b`` is conserved, so
    along a parcel ``G = |grad b|^2`` grows exactly as ``exp(2 a t)``.

    What is compared (all ``G`` from ``operators.gradb2`` / the same stencil
    at the departure point, ``semilag.gradb2_at_departure``, order 3):

    * **the gate** -- ``G`` along the parcels arriving at every front pixel
      (``G >= 0.2 max``) after ``n_steps`` chained semi-Lagrangian hours,
      ``G(t_n)/G(t_0)`` against ``exp(2 a t_n)``, at the reference width
      ``ell = ref_ell_cells dx``: ``series_max_err_ref = max |ratio - 1|``
      over parcels and steps, required ``< 1%``;
    * the one-step growth rate ``ln[G(x, t+dt) / G(x_d, t)] / (2 a dt)``
      on front pixels against the front width (``rate``), whose error is
      the centred stencil's truncation, second order in ``dx/ell``;
    * the semi-Lagrangian step alone: ``G(x_d, t)`` against the same
      stencil applied to the analytic ``b`` at the *exact* departure point
      (``semilag_err_max``), which isolates departure + interpolation from
      the stencil's own truncation (``< 1%`` at every width, orders 3 and 5);
    * ``semilag.measured_DGDt`` reproduces ``[G_1 - G_d]/dt`` bit-for-bit.
    Offline.  Returns the numbers.
    """
    res = dict(alpha=float(alpha), dt=float(dt), n=int(n), n_steps=int(n_steps),
               widths_cells=[float(w) for w in widths], ref_ell_cells=float(ref_ell_cells),
               interp_order=3, front_definition='G >= 0.2 max, %d cells from the edge' % margin,
               series_t_hours=[k * dt / 3600.0 for k in range(n_steps + 1)],
               rate={}, series={}, semilag_err_max={'order3': [], 'order5': []},
               profiles=dict(widths=[float(w) for w in widths if w != 3], x_over_ell={}, rate={}))
    for rotated, key in ((False, 'CS=1'), (True, 'face10')):
        r_ = dict(median=[], min=[], max=[], rms_err=[], n=[])
        for w in widths:
            case = sy.deformation_case(ell_cells=w, a=alpha, rotated=rotated, n=n)
            st = sy.deformation_step(case, 0.0, dt=dt, order=3)
            G1, Gd = st['G_1'].values[0], st['G_d'].values[0]
            front = _front(G1, margin) & np.isfinite(Gd)
            with np.errstate(divide='ignore', invalid='ignore'):     # sech^4 underflows off-front
                rate = np.log(G1 / Gd) / (2 * alpha * dt)
            r_['median'].append(float(np.median(rate[front])))
            r_['min'].append(float(rate[front].min()))
            r_['max'].append(float(rate[front].max()))
            r_['rms_err'].append(_rms(rate[front] - 1.0))
            r_['n'].append(int(front.sum()))
            if not rotated:
                # the step alone, isolated from the stencil's truncation
                st5 = sy.deformation_step(case, 0.0, dt=dt, order=5)
                with np.errstate(divide='ignore', invalid='ignore'):
                    res['semilag_err_max']['order3'].append(
                        float(np.abs(Gd / st['G_d_ref'][0] - 1)[front].max()))
                    res['semilag_err_max']['order5'].append(
                        float(np.abs(st5['G_d'].values[0] / st5['G_d_ref'][0] - 1)[front].max()))
                if w in res['profiles']['widths']:
                    row = n // 2
                    sel = front[row]
                    res['profiles']['x_over_ell'][str(float(w))] = \
                        ((case['xc'][0, row] - case['xc0']) / case['ell'])[sel].tolist()
                    res['profiles']['rate'][str(float(w))] = rate[row][sel].tolist()
            if w == ref_ell_cells:
                t, G = sy.deformation_series(case, n_steps=n_steps, dt=dt, order=3)
                ratio = G[:, front] / G[0, front]
                err = ratio / np.exp(2 * alpha * t)[:, None] - 1.0
                res['series'][key] = dict(n=int(front.sum()), max_err=float(np.abs(err).max()),
                                          rms_err_final=_rms(err[-1]),
                                          median=np.median(ratio, axis=1).tolist(),
                                          min=ratio.min(axis=1).tolist(), max=ratio.max(axis=1).tolist())
                # the contract function is the same arithmetic
                DG = sl.measured_DGDt(case['b_exact'](0.0), case['b_exact'](dt), case['U'], case['V'],
                                      case['g'], case['grid'], dt=dt).values[0]
                res['series'][key]['measured_DGDt_max_rel_diff'] = float(np.nanmax(
                    np.abs(DG[front] - (G1 - Gd)[front] / dt) / np.abs((G1 - Gd)[front] / dt)))
        res['rate'][key] = r_
    w = np.array(res['widths_cells'])
    e = np.array(res['rate']['CS=1']['rms_err'])
    sel = w >= 4
    res['convergence_order'] = float(-np.polyfit(np.log(w[sel]), np.log(e[sel]), 1)[0])
    res['series_max_err_ref'] = max(s['max_err'] for s in res['series'].values())
    res['semilag_err_max_all'] = max(res['semilag_err_max']['order3'])
    res['orientation_max_diff'] = float(np.max(np.abs(np.array(res['rate']['CS=1']['median'])
                                                      - np.array(res['rate']['face10']['median']))))
    res['gate'] = dict(threshold=0.01, series_pass=bool(res['series_max_err_ref'] < 0.01),
                       semilag_pass=bool(res['semilag_err_max_all'] < 0.01))
    res['png'] = vf.fig_V1(res) if png else None
    return res


# ---------------------------------------------------------------------------
# V2: native-grid metric
# ---------------------------------------------------------------------------
def _grad_arrays(f, g, grid):
    bx, by = op.grad_b(xr.DataArray(f[None], dims=('face', 'j', 'i')), g, grid)
    return bx.values[0], by.values[0]


def test_native_metric(grid_ds, png=True, L_lon_deg=2.0, L_lat_deg=2.0, R=R_SPHERE) -> dict:
    """**V2**: the analytic ``f(XC, YC) = sin(2 pi (lon - lon0)/L_lon) cos(2 pi
    (lat - lat0)/L_lat)`` through ``operators.grad_b`` on the real tile grid,
    against its exact geographic gradient on a sphere of radius ``R``
    (6370 km, MITgcm's ``rSphere``; the tile's ``dxC``/``dyC`` imply 6370.0
    km, ``R_from_dxC``/``R_from_dyC``).  Errors are normalised by ``max |grad
    f|`` on ``mask_analysis`` (the gradient passes through zero); the gate is
    ``max |b_x - f_east|, max |b_y - f_north| < 1%`` there.  The linear
    functions ``lon - lon0`` and ``lat - lat0`` (``lin_*``) carry no
    truncation, so their error is the metric alone (``dxC``, ``dyC``, ``CS``,
    ``SN``); the sinusoid's is the stencil's ``-(k dx)^2/6``, predicted per
    cell from the local phase advance (``truncation_*``).  ``err_if_
    components_swapped`` says what a wrong ``CS``/``SN`` would look like.
    ``@pytest.mark.needs_grid``.  Returns the numbers.
    """
    g = grid_ds if 'face' in grid_ds.dims else grid_ds.expand_dims('face')
    grid = ot.build_xgcm(g)
    X, Y = g.XC.squeeze().values, g.YC.squeeze().values
    masks_path = ot.DATA_DIR / 'tile330_masks.nc'
    masks = mk.open_masks(masks_path) if masks_path.exists() else mk.build_masks(g)
    ana, oc = masks['mask_analysis'].values, masks['mask_ocean'].values
    lon0, lat0 = float(np.mean(X[ana])), float(np.mean(Y[ana]))
    f, fe, fn = sy.wave(X, Y, lon0, lat0, L_lon_deg, L_lat_deg, R)
    bx, by = _grad_arrays(f, g, grid)
    gmax = float(np.nanmax(np.hypot(fe, fn)[ana]))
    err_x, err_y = (bx - fe) / gmax, (by - fn) / gmax
    res = dict(L_lon_deg=float(L_lon_deg), L_lat_deg=float(L_lat_deg), R_m=float(R),
               lon0=lon0, lat0=lat0, n_analysis=int(ana.sum()), grad_max=gmax,
               normalisation='error / max |grad f| on mask_analysis')
    # the cell size of the wave: degrees per cell along the axis that runs east-west / north-south
    dlon = (np.abs(np.diff(X, axis=1)), np.abs(np.diff(X, axis=0)))     # along i, along j
    dlat = (np.abs(np.diff(Y, axis=1)), np.abs(np.diff(Y, axis=0)))
    ew = 0 if np.nanmedian(dlon[0]) > np.nanmedian(dlon[1]) else 1        # which axis runs east-west
    res['east_west_axis'] = 'i' if ew == 0 else 'j'
    res['L_lon_cells'] = float(L_lon_deg / np.nanmedian(dlon[ew]))
    res['L_lat_cells'] = float(L_lat_deg / np.nanmedian(dlat[1 - ew]))
    # local truncation prediction: -(theta^2/6) f' with theta the phase advance per cell
    # (the dbof stencil is the 2 dx centred difference: sin(theta)/theta - 1)
    kx, ky = 2 * np.pi / L_lon_deg, 2 * np.pi / L_lat_deg
    th_x = kx * np.abs(np.gradient(X, axis=1 - ew))       # rad of lon-phase per cell, east-west axis
    th_y = ky * np.abs(np.gradient(Y, axis=ew))           # rad of lat-phase per cell, north-south axis
    pred_x, pred_y = th_x ** 2 / 6 * np.abs(fe) / gmax, th_y ** 2 / 6 * np.abs(fn) / gmax
    for comp, e, pr in (('x', err_x, pred_x), ('y', err_y, pred_y)):
        k = int(np.nanargmax(np.abs(e[ana])))
        res[f'err_{comp}_rms'] = _rms(e[ana])
        res[f'err_{comp}_max'] = float(np.nanmax(np.abs(e[ana])))
        res[f'err_{comp}_p99'] = float(np.nanpercentile(np.abs(e[ana]), 99))
        res[f'worst_{comp}'] = dict(lat=float(Y[ana][k]), lon=float(X[ana][k]),
                                    j=int(np.flatnonzero(ana)[k] // X.shape[1]),
                                    i=int(np.flatnonzero(ana)[k] % X.shape[1]),
                                    err=float(e[ana][k]), predicted=float(pr[ana][k]))
        res[f'truncation_{comp}'] = float(np.nanmax(pr[ana]))
        res[f'truncation_{comp}_rms'] = _rms(pr[ana])
    strong = ana & (np.hypot(fe, fn) > 0.5 * gmax)
    res['err_pointwise_rel_max_strong'] = float(np.nanmax(
        np.hypot(bx - fe, by - fn)[strong] / np.hypot(fe, fn)[strong]))
    res['err_if_components_swapped'] = _rms(np.hypot(by - fe, bx - fn)[ana] / gmax)
    res['finite_on_analysis'] = bool(np.isfinite(bx[ana]).all() and np.isfinite(by[ana]).all())
    # the metric alone: linear functions of lon and lat
    deg = 180.0 / np.pi
    lin = {}
    for name, fl, fle, fln in (('lon', X - lon0, deg / (R * np.cos(np.deg2rad(Y))), 0 * X),
                               ('lat', Y - lat0, 0 * X, deg / R + 0 * X)):
        lbx, lby = _grad_arrays(fl, g, grid)
        lm = float(np.nanmax(np.hypot(fle, fln)[ana]))
        lin[name] = ((lbx - fle) / lm, (lby - fln) / lm)
        res[f'lin_{name}_err_x_max'] = float(np.nanmax(np.abs(lin[name][0][ana])))
        res[f'lin_{name}_err_y_max'] = float(np.nanmax(np.abs(lin[name][1][ana])))
        res[f'lin_{name}_scale_median'] = float(np.nanmedian(
            ((lbx / fle) if name == 'lon' else (lby / fln))[ana]))
    res['lin_err_x_max'] = res['lin_lon_err_x_max']
    res['lin_err_y_max'] = res['lin_lat_err_y_max']
    res['R_from_dxC'], res['R_from_dyC'] = sy.sphere_radius(X, Y, g.dxC.values[0], g.dyC.values[0])
    # the error against latitude (rms in bins) with the local prediction
    lat = Y[ana]
    lb = np.linspace(lat.min(), lat.max(), 25)
    res['lat_bins'] = (0.5 * (lb[1:] + lb[:-1])).tolist()
    for name, arr in (('rms_x_vs_lat', err_x), ('rms_y_vs_lat', err_y),
                      ('truncation_x_vs_lat', pred_x), ('truncation_y_vs_lat', pred_y)):
        a = arr[ana]
        res[name] = [_rms(a[(lat >= lb[k]) & (lat < lb[k + 1])]) for k in range(len(lb) - 1)]
    for end, sel in (('south', lat < lat.min() + 0.5), ('north', lat > lat.max() - 0.5)):
        res[f'dxC_km_{end}'] = float(np.median(g.dxC.values[0][ana][sel]) / 1e3)
        res[f'dyC_km_{end}'] = float(np.median(g.dyC.values[0][ana][sel]) / 1e3)
    # second-order convergence of the sinusoid's error: halve and double the wavelength
    sweep = dict(L_deg=[], err_x_max=[], err_y_max=[])
    for L in (0.5 * L_lon_deg, L_lon_deg, 2.0 * L_lon_deg):
        fs, fes, fns = sy.wave(X, Y, lon0, lat0, L, L * L_lat_deg / L_lon_deg, R)
        sbx, sby = _grad_arrays(fs, g, grid)
        sm = float(np.nanmax(np.hypot(fes, fns)[ana]))
        sweep['L_deg'].append(float(L))
        sweep['err_x_max'].append(float(np.nanmax(np.abs(sbx - fes)[ana]) / sm))
        sweep['err_y_max'].append(float(np.nanmax(np.abs(sby - fns)[ana]) / sm))
    res['sweep'] = sweep
    res['sweep_order'] = float(-np.polyfit(np.log(sweep['L_deg']), np.log(sweep['err_x_max']), 1)[0])
    res['gate'] = dict(threshold=0.01, max_err=max(res['err_x_max'], res['err_y_max']),
                       passed=bool(max(res['err_x_max'], res['err_y_max']) < 0.01))
    res['png'] = None
    if png:
        ctx = dict(X=X, Y=Y, f=f, err_x=err_x, err_y=err_y, ana=ana, oc=oc,
                   err_lin_x=lin['lon'][0], err_lin_y=lin['lat'][1])
        res['png'] = vf.fig_V2(res, ctx)
    return res


# ---------------------------------------------------------------------------
# V4: interpolation bias
# ---------------------------------------------------------------------------
def _bias_stats(u):
    """``(rms over front pixels, signed at the front maximum)`` of
    ``rel = DGDt dt / G`` [fraction of G per hour]."""
    r = u['rel'][u['front']]
    k = int(np.nanargmax(np.where(u['front'], u['G'], -1.0)))
    return float(np.sqrt(np.nanmean(r ** 2))), float(u['rel'].flat[k])


def test_interpolation_bias(png=True, sigma_G_ref=1.5, orders=(1, 3, 5), n_frac=21,
                            directions_deg=(0, 30, 45, 60, 90), widths=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                            real_hour=REAL_HOUR_CELLS) -> dict:
    """**V4**: ``semilag.measured_DGDt`` under a uniform, zero-strain flow
    that moves an ``erf`` front (``G`` Gaussian of ``sigma_G`` cells) by a
    prescribed displacement per hour; ``b_tp1`` is the exactly shifted
    ``b_t``, so the true ``DG/Dt = 0`` and whatever comes out is the
    interpolation bias.  Reported as ``rel = DGDt dt / G(t+dt)`` -- the
    fabricated tendency per hour as a fraction of ``G`` (positive at the
    front maximum = fabricated frontogenesis) -- as the rms over front
    pixels (``G >= 0.2 max``) and signed at the maximum, against: the
    sub-cell displacement fraction (``vs_fraction``); the direction of a
    half-cell displacement for a front along ``j`` and one tilted 30 deg
    (``vs_direction``); the front width (``vs_width``); and the interpolation
    order 1 / 3 / 5 everywhere.

    **Headline error bar** (``headline``): the rms over front pixels for the
    ``sigma_G = 1.5``-cell front at order 3, averaged in quadrature over the
    sub-cell cross-front displacement of the real-hour distribution
    (``synthetic.real_hour_fractions``: median 0.36, p99 1.25, max 2.09 cells,
    isotropic direction), with the all-cross-front value as the upper bound,
    and expressed against the 7-20% per-hour signal ``2F dt/G``.  Offline.
    """
    fr = np.linspace(0.0, 1.0, n_frac)
    key = {o: f'order{o}' for o in orders}
    table = {}
    for w in widths:
        table[w] = {}
        for o in orders:
            st = [_bias_stats(sy.uniform_shift_bias(sigma_G=w, di=f, order=o)) for f in fr]
            table[w][key[o]] = dict(rms=[s[0] for s in st], at_max=[s[1] for s in st])
    res = dict(sigma_G_ref=float(sigma_G_ref), orders=list(orders), fractions=fr.tolist(),
               widths_sigma_G=[float(w) for w in widths], directions_deg=list(directions_deg),
               front_definition='G >= 0.2 max (within ~1.8 sigma_G of the centre)',
               rel_definition='measured DGDt * dt / G(t+dt); truth is 0',
               vs_fraction=table[sigma_G_ref], vs_width={}, vs_direction={},
               real_hour_median_cells=real_hour['median'], real_hour_p99_cells=real_hour['p99'],
               real_hour_max_cells=real_hour['max'])
    f_iso, f_all = sy.real_hour_fractions(real_hour)

    def over_distribution(tab):
        rms2 = np.interp(f_iso, fr, np.array(tab['rms']) ** 2)
        return dict(rms=float(np.sqrt(rms2.mean())),
                    rms_all_cross_front=float(np.sqrt(np.interp(f_all, fr, np.array(tab['rms']) ** 2).mean())),
                    at_max_mean=float(np.interp(f_iso, fr, tab['at_max']).mean()),
                    rms_half_cell=float(np.interp(0.5, fr, tab['rms'])),
                    at_max_half_cell=float(np.interp(0.5, fr, tab['at_max'])))
    for o in orders:
        k = key[o]
        res['vs_width'][k] = dict(rms_half_cell=[float(np.interp(0.5, fr, table[w][k]['rms'])) for w in widths],
                                  at_max_half_cell=[float(np.interp(0.5, fr, table[w][k]['at_max'])) for w in widths],
                                  rms_real_hour=[over_distribution(table[w][k])['rms'] for w in widths])
        h = over_distribution(table[sigma_G_ref][k])
        h.update(order=o, sigma_G=float(sigma_G_ref), signal_fraction=[h['rms'] / 0.20, h['rms'] / 0.07],
                 definition='rms over front pixels of DGDt dt/G, sigma_G = 1.5, averaged in quadrature over '
                            'the sub-cell cross-front displacement of the real-hour distribution (lognormal '
                            'median 0.364, p99 1.252, cap 2.09 cells; isotropic direction)')
        res[f'headline_{k}'] = h
    res['headline'] = res['headline_order3']
    res['integer_shift_rms'] = max(table[sigma_G_ref][key[o]]['rms'][i] for o in orders for i in (0, -1))
    for tilt in (0.0, 30.0):
        res['vs_direction'][f'tilt{tilt:.0f}'] = {
            key[o]: [_bias_stats(sy.uniform_shift_bias(sigma_G=sigma_G_ref, di=0.5 * np.cos(np.deg2rad(a)),
                                                       dj=0.5 * np.sin(np.deg2rad(a)), order=o,
                                                       tilt_deg=tilt))[0] for a in directions_deg]
            for o in orders}
    w = np.array(res['widths_sigma_G'])
    sel = w >= 1.5
    res['width_slope'] = {k: float(np.polyfit(np.log(w[sel]), np.log(np.array(res['vs_width'][k]['rms_half_cell'])[sel]), 1)[0])
                          for k in res['vs_width']}
    profile = {}
    for o in orders:
        u = sy.uniform_shift_bias(sigma_G=sigma_G_ref, di=0.5, order=o)
        row = u['rel'].shape[0] // 2
        profile['s_cells'] = u['s_cells'][row]
        profile[key[o]] = u['rel'][row]
    res['png'] = vf.fig_V4(res, profile) if png else None
    return res


# ---------------------------------------------------------------------------
# V5: the interpolation choice, made visible
# ---------------------------------------------------------------------------
def _half_cell_curves(sigma_G, nj=16, ni=96, dx=1800.0, x0_frac=0.3, orders=(1, 3, 5)):
    """The V5 experiment for one front width: ``G`` after a half-cell shift
    along ``i`` by every route, on the middle row.  Truth is the same
    discrete stencil applied to the exactly shifted ``b`` (so the stencil's
    own truncation cancels and only the interpolation error remains)."""
    g, grid, x, _ = sy.synthetic_uniform_grid(nj=nj, ni=ni, dx=dx)
    x0 = (ni / 2 + x0_frac) * dx
    b = xr.DataArray(sy.erf_front(x, x0, sigma_G, dx), dims=('face', 'j', 'i'))
    b_shift = xr.DataArray(sy.erf_front(x - 0.5 * dx, x0, sigma_G, dx), dims=('face', 'j', 'i'))
    row = nj // 2
    out = dict(x_cells=(x[0, row] - x0) / dx - 0.5,          # distance from the shifted front centre x0 + dx/2
               truth=op.gradb2(b_shift, g, grid).values[0, row],
               analytic=(sy.erf_front(x[0, row] - 0.5 * dx + 1e-3 * dx, x0, sigma_G, dx)
                         - sy.erf_front(x[0, row] - 0.5 * dx - 1e-3 * dx, x0, sigma_G, dx)) ** 2
               / (2e-3 * dx) ** 2)
    G = op.gradb2(b, g, grid)
    # the trap: interpolate G itself, bilinearly
    out['bilinear_G'] = sl.interp_to_departure(G, 0.5, 0.0, 1, allow_low_order=True).values[0, row]
    # the rule: interpolate b onto the departure stencil, then the same stencil
    for order in orders:
        out[f'b_order{order}'] = sl.gradb2_at_departure(
            b, 0.5, 0.0, g, order, allow_low_order=True).values[0, row]
    # the tile-edge rim is finite and wrong (xgcm's 0 fill, M0 task 5) and
    # the interpolation reach is up to 3 cells: blank 4 cells on each end
    for name in list(out):
        if name != 'x_cells':
            out[name] = out[name].copy()
            out[name][:4] = np.nan
            out[name][-4:] = np.nan
    return out


def demo_interp_half_cell(png: bool = True) -> dict:
    """**V5**: a synthetic front (``G`` Gaussian, ``sigma_G = 1.5`` cells,
    i.e. ~1.5 cells wide) shifted by half a cell -- truth vs ``G`` from
    bilinear-interpolated ``G`` vs ``G`` from cubic-interpolated ``b``
    (``semilag.gradb2_at_departure``: ``b`` onto the departure stencil,
    then the ``operators.grad_b`` stencil), with the negative bias at the
    maximum annotated in % of ``G`` against the ``dx^2 G_xx / 8G`` prediction
    and planning §5.3's ~5.5%.  Offline.  Returns the numbers."""
    sigma_G = 1.5
    c = _half_cell_curves(sigma_G)
    k = int(np.argmax(np.nan_to_num(c['truth'])))
    pred = -1.0 / (8 * sigma_G ** 2)

    def bias(name):
        return float((c[name][k] - c['truth'][k]) / c['truth'][k])

    res = dict(sigma_G_cells=sigma_G, shift_cells=0.5,
               bias_bilinear_G=bias('bilinear_G'), bias_b_order1=bias('b_order1'),
               bias_b_order3=bias('b_order3'), bias_b_order5=bias('b_order5'),
               bias_predicted=float(pred),
               bias_analytic_1d=float(np.exp(-0.25 / (2 * sigma_G ** 2)) - 1.0),
               peak_ratio_bilinear_G=float(np.nanmax(c['bilinear_G']) / np.nanmax(c['truth'])),
               peak_ratio_b_order3=float(np.nanmax(c['b_order3']) / np.nanmax(c['truth'])))
    # the bias at the maximum against the front width
    widths = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
    sweep = {name: [] for name in ('bilinear_G', 'b_order1', 'b_order3', 'b_order5')}
    for w in widths:
        cw = _half_cell_curves(w)
        kw = int(np.argmax(np.nan_to_num(cw['truth'])))
        for name in sweep:
            sweep[name].append(float((cw[name][kw] - cw['truth'][kw]) / cw['truth'][kw]))
    res['sweep_sigma_G_cells'] = widths.tolist()
    res['sweep_bias'] = {k_: v for k_, v in sweep.items()}
    res['png'] = vf.fig_V5(c, res, widths, sweep, sigma_G, pred, k) if png else None
    return res


# ---------------------------------------------------------------------------
# V6: land halo
# ---------------------------------------------------------------------------
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
    if png:
        ctx = dict(masks=masks, attrs=a, X=X, Y=Y, halo_km=halo_km, halo_cells=halo_cells,
                   edge_cells=edge_cells, oc=oc, halo=halo, dist=dist, cls=cls, counts=counts,
                   taxi=taxi, G=G, rim=rim, rim_max=rim_max, gulf=gulf, res=res, snapshot=SNAPSHOT)
        res['png'] = vf.fig_V6(ctx)
    return res


# the contract names start with test_; they are gates, not pytest tests
for _f in (test_cartesian_deformation, test_native_metric, test_interpolation_bias):
    _f.__test__ = False


if __name__ == '__main__':
    import pprint
    pprint.pprint({k: v for k, v in test_cartesian_deformation().items()
                   if k in ('series_max_err_ref', 'semilag_err_max_all', 'convergence_order', 'gate', 'png')})
    pprint.pprint({k: v for k, v in test_interpolation_bias().items()
                   if k in ('headline', 'headline_order1', 'headline_order5', 'width_slope', 'png')})
    pprint.pprint({k: v for k, v in demo_interp_half_cell().items() if k.startswith('bias') or k == 'png'})
    grid_ds = ot.open_grid(with_face=True)
    pprint.pprint({k: v for k, v in test_native_metric(grid_ds).items()
                   if k.startswith(('err_', 'lin_err', 'R_from', 'gate', 'worst_x', 'png'))})
    pprint.pprint({k: v for k, v in qa_land_halo(grid_ds).items() if k.startswith(('n_', 'edge_', 'png'))})
