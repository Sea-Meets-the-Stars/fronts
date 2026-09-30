""" M1 validation gates (coding doc §4.9): the four gates V1-V4 and the two
supporting figures V5-V6.  Each function computes the numbers, returns them
as a dict and (``png=True``) hands them to ``validate_figs.fig_V*`` for the
PNG in ``dev/frontogenesis/figs/``.  Synthetic grids and exact solutions
come from ``synthetic.py``; nothing here is imported by the science path.

Written: ``test_cartesian_deformation`` -> **V1**, ``test_native_metric`` ->
**V2**, ``test_interpolation_bias`` -> **V4** (M1 task 5);
``demo_interp_half_cell`` -> **V5** (task 3); ``qa_land_halo`` -> **V6**
(task 1); ``test_discrete_null`` -> **V3** (task 6, the hard gate: it
returns the fitted slope Figure 2 draws as its baseline, and it is what
made ``operators.frontogenesis`` default to the discretely consistent form
and ``semilag.departure_index`` to the cubic velocity).

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
# V3: the discrete null
# ---------------------------------------------------------------------------
# Declared before any number was seen (M1 task 6; planning §11):
#  * front pixels: G at the trajectory midpoint time, G_mid = gradb2(0.5 (b_t +
#    b_tp1)), at or above its 90th percentile over the valid set (mask_analysis
#    & finite for 'llc'; the interior, pooled over cases, for 'strain') -- the
#    rule M3 will use, independent of both endpoints;
#  * the gate estimator: OLS slope of the measured DG/Dt on 2F (with
#    intercept) on those pixels -- 2F is the smooth, low-noise side in a null
#    whose only "error" is discretisation; the other estimators are reported;
#  * bootstrap over 32 x 32-cell spatial blocks (never pixels), 1000 draws.
FRONT_PERCENTILE = 90.0
BLOCK_CELLS = 32
N_BOOT = 1000
NULL_WIDTHS = (2.0, 3.0, 4.0, 6.0, 8.0)          # tanh ell/dx; sigma_G = ell/2 = 1-4 cells
NULL_ANGLES = (0.0, 30.0, 60.0)                  # front normal from the compressional axis
NULL_DIAG_WIDTHS = (1.0, 1.5)                    # out of the pool: where the operators break


def slope_estimators(x, y) -> dict:
    """The slope of ``y`` on ``x`` by every estimator planning §11 asks for:
    ``ols`` (y on x, with intercept: **the gate**), ``ols_origin``,
    ``ols_inverse`` (1 / the slope of x on y), ``geometric_mean`` (reduced
    major axis), ``orthogonal`` (total least squares, equal error
    variances, meaningful here because both axes share units), ``ratio``
    (``sum y / sum x``), with ``intercept``, ``corr`` and ``n``."""
    x, y = np.asarray(x, dtype='float64'), np.asarray(y, dtype='float64')
    n = x.size
    xm, ym = x.mean(), y.mean()
    sxx, syy, sxy = np.sum((x - xm) ** 2), np.sum((y - ym) ** 2), np.sum((x - xm) * (y - ym))
    ols = sxy / sxx
    b_xy = sxy / syy
    return dict(n=int(n), ols=float(ols), intercept=float(ym - ols * xm),
                ols_origin=float(np.sum(x * y) / np.sum(x * x)),
                ols_inverse=float(1.0 / b_xy),
                geometric_mean=float(np.sign(sxy) * np.sqrt(syy / sxx)),
                orthogonal=float(((syy - sxx) + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)),
                ratio=float(y.sum() / x.sum()), corr=float(sxy / np.sqrt(sxx * syy)))


def block_bootstrap_ols(x, y, block, n_boot=N_BOOT, seed=0) -> dict:
    """Bootstrap of the gate (OLS with intercept) over spatial blocks:
    ``block`` labels each pixel; blocks are resampled with replacement
    (multinomial counts) and the slope is rebuilt from per-block sums.
    Returns the 2.5 / 97.5 percentiles, the standard error and the block
    count."""
    x, y, block = (np.asarray(a) for a in (x, y, block))
    ids, inv = np.unique(block, return_inverse=True)
    nb = ids.size
    S = np.zeros((nb, 5))
    for k, col in enumerate((np.ones_like(x), x, y, x * x, x * y)):
        S[:, k] = np.bincount(inv, weights=col, minlength=nb)
    rng = np.random.default_rng(seed)
    M = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=int(n_boot)).astype('float64')
    T = M @ S                                            # (n_boot, 5): N, Sx, Sy, Sxx, Sxy
    slopes = (T[:, 0] * T[:, 4] - T[:, 1] * T[:, 2]) / (T[:, 0] * T[:, 3] - T[:, 1] ** 2)
    return dict(ci=[float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
                se=float(slopes.std()), n_blocks=int(nb), n_boot=int(n_boot),
                block_cells=BLOCK_CELLS)


def _block_ids(shape, offset=0, B=BLOCK_CELLS):
    nj, ni = shape
    jj, ii = np.meshgrid(np.arange(nj) // B, np.arange(ni) // B, indexing='ij')
    return offset + jj * (ni // B + 1) + ii


def two_F(b_mid, U, V, grid_ds, grid, form='chain'):
    """``2F`` at the midpoint from ``operators.frontogenesis`` -- the
    predicted side, named as coding §1.1 asks."""
    return 2.0 * op.frontogenesis(b_mid, U, V, grid_ds, grid, form=form).values[0]


def null_step(b_t, U, V, grid_ds, grid, order=3, dt=DT, forms=('chain',), vel_order=3, advect=None) -> dict:
    """The discrete null for one field and one steady (or time-midpoint)
    velocity: advect ``b_t`` one hour with **our own** semi-Lagrangian step
    (``b_tp1 = b_t`` interpolated at the departure points of
    ``semilag.departure_index``, order ``order``, velocity order
    ``vel_order``), so the truth satisfies our discrete advection exactly;
    then the measured side ``semilag.measured_DGDt`` and ``2F`` from
    ``operators.frontogenesis`` at the trajectory midpoint ``0.5 (b_t +
    b_tp1)``, one ``two_F_<form>`` per requested form.  Everything
    positional ``(nj, ni)`` numpy.

    ``advect`` (V3b, task 6b): an optional ``(b_t, U, V, grid_ds, grid) ->
    b_tp1`` callable that replaces the semi-Lagrangian truth -- the
    flux-form finite-volume step of ``fvadvect.advector``.  The measured
    and predicted sides are untouched; ``None`` is V3."""
    u_c, v_c = sl.centre_velocities(U, V, grid_ds, grid)
    di, dj = sl.departure_index(u_c, v_c, grid_ds, dt=dt, vel_order=vel_order)
    if advect is None:
        b_tp1 = sl.interp_to_departure(b_t, di, dj, order)
    else:
        b_tp1 = advect(b_t, U, V, grid_ds, grid)
    meas = sl.measured_DGDt(b_t, b_tp1, U, V, grid_ds, grid, dt=dt, order=order, vel_order=vel_order)
    b_mid = sl.midpoint_time(b_t, b_tp1)
    out = dict(measured=meas.values[0], G_mid=op.gradb2(b_mid, grid_ds, grid).values[0],
               G_tp1=op.gradb2(b_tp1, grid_ds, grid).values[0],
               di=di.values[0], dj=dj.values[0], b_mid=b_mid, b_tp1=b_tp1)
    for form in forms:
        out[f'two_F_{form}'] = two_F(b_mid, U, V, grid_ds, grid, form)
    return out


def front_pixels(G_mid, valid, pct=FRONT_PERCENTILE):
    """The pre-declared front-pixel rule: ``G_mid >= p_pct`` over ``valid``."""
    thr = float(np.percentile(G_mid[valid], pct))
    return valid & (G_mid >= thr), thr


def _fit(x, y, block, n_boot=N_BOOT):
    est = slope_estimators(x, y)
    est['bootstrap'] = block_bootstrap_ols(x, y, block, n_boot=n_boot)
    est['gate'] = dict(estimator='ols (measured on 2F, with intercept)', target=1.0, tol=0.05,
                       passed=bool(abs(est['ols'] - 1.0) <= 0.05))
    return est


def _binned_means(x, y, nbins=12):
    """Conditional means ``E[y|x]`` in x-quantile bins, for the figure."""
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    k = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, nbins - 1)
    xb = np.array([x[k == q].mean() for q in range(nbins)])
    yb = np.array([y[k == q].mean() for q in range(nbins)])
    return xb, yb


def _null_strain(order, forms, widths, angles, n, a, margin, n_boot, vel_order=3, advect=None):
    """The 'strain' variant: one ``synthetic.null_strain_case`` per (width,
    angle), the front pixels pooled under one threshold.  ``advect`` as in
    :func:`null_step` (V3b)."""
    pool = {f: dict(x=[], y=[], block=[], width=[]) for f in forms}
    per_case = []
    G_all, valid_all = [], []
    cases = [(w, th) for w in widths for th in angles]
    steps = []
    for c, (w, th) in enumerate(cases):
        case = sy.null_strain_case(ell_cells=w, theta_deg=th, a=a, n=n)
        st = null_step(case['b_t'], case['U'], case['V'], case['g'], case['grid'], order=order, forms=forms,
                       vel_order=vel_order, advect=advect)
        valid = sy.inner(st['G_mid'].shape, margin) & np.isfinite(st['measured']) & np.isfinite(st['G_mid'])
        for f in forms:
            valid &= np.isfinite(st[f'two_F_{f}'])
        st['valid'], st['case'] = valid, (w, th)
        st['block'] = _block_ids(valid.shape, offset=c * 10_000)
        steps.append(st)
        G_all.append(st['G_mid'][valid]); valid_all.append(valid)
    thr = float(np.percentile(np.concatenate(G_all), FRONT_PERCENTILE))
    for st in steps:
        front = st['valid'] & (st['G_mid'] >= thr)
        st['front'] = front
        w, th = st['case']
        row = dict(width=w, angle=th, n=int(front.sum()))
        for f in forms:
            x, y = st[f'two_F_{f}'][front], st['measured'][front]
            if front.sum() > 10:
                row[f] = slope_estimators(x, y)['ols']
                row[f'{f}_ratio'] = float(y.sum() / x.sum())
            pool[f]['x'].append(x); pool[f]['y'].append(y)
            pool[f]['block'].append(st['block'][front]); pool[f]['width'].append(np.full(front.sum(), w))
        per_case.append(row)
    res = dict(threshold=thr, per_case=per_case, fits={}, per_width={})
    for f in forms:
        x, y = np.concatenate(pool[f]['x']), np.concatenate(pool[f]['y'])
        blk, wd = np.concatenate(pool[f]['block']), np.concatenate(pool[f]['width'])
        res['fits'][f] = _fit(x, y, blk, n_boot)
        res['per_width'][f] = {str(w): dict(n=int((wd == w).sum()),
                                            ols=(slope_estimators(x[wd == w], y[wd == w])['ols']
                                                 if (wd == w).sum() > 10 else None))
                               for w in widths}
        res['fits'][f]['width_share'] = {str(w): float((wd == w).mean()) for w in widths}
        pool[f]['x'], pool[f]['y'] = x, y
    return res, pool, steps


def _fourth_order_everywhere(case, order=3, margin=10):
    """The prompt's other option, tried and logged: a 4th-order gradient
    stencil on **both** sides (``G4 = |D4 b|^2`` at the arrival and at the
    departure -- ``b`` interpolated onto the 9-point stencil -- and the
    chain-rule ``F4 = -(D4 b)^T (D4 u_c)(D4 b)``), uniform synthetic grid,
    numpy.  Returns the top-decile OLS slope.  It halves the deficit at
    2-4 dx but does not remove it: the product rule still fails."""
    g, grid, dx = case['g'], case['grid'], case['dx']
    u_c, v_c = sl.centre_velocities(case['U'], case['V'], g, grid)
    di, dj = sl.departure_index(u_c, v_c, g)
    b = case['b_t']

    def D4(arr, ax):
        return (-np.roll(arr, -2, ax) + 8 * np.roll(arr, -1, ax) - 8 * np.roll(arr, 1, ax)
                + np.roll(arr, 2, ax)) / (12 * dx)

    def at(k_i, k_j):
        return sl.interp_to_departure(b, di - k_i, dj - k_j, order).values[0]
    b1 = sl.interp_to_departure(b, di, dj, order).values[0]
    G1 = D4(b1, 1) ** 2 + D4(b1, 0) ** 2
    gx = (-at(2, 0) + 8 * at(1, 0) - 8 * at(-1, 0) + at(-2, 0)) / (12 * dx)
    gy = (-at(0, 2) + 8 * at(0, 1) - 8 * at(0, -1) + at(0, -2)) / (12 * dx)
    meas = (G1 - (gx ** 2 + gy ** 2)) / DT
    bm = 0.5 * (b.values[0] + b1)
    bx4, by4 = D4(bm, 1), D4(bm, 0)
    uc, vc = u_c.values[0], v_c.values[0]
    F4 = -(D4(uc, 1) * bx4 ** 2 + (D4(uc, 0) + D4(vc, 1)) * bx4 * by4 + D4(vc, 0) * by4 ** 2)
    Gm = bx4 ** 2 + by4 ** 2
    valid = sy.inner(G1.shape, margin) & np.isfinite(meas) & np.isfinite(F4)
    front = valid & (Gm >= np.percentile(Gm[valid], FRONT_PERCENTILE))
    return slope_estimators(2 * F4[front], meas[front])['ols']


def _llc_inputs(grid_ds=None) -> dict:
    """The real-velocity variant's inputs (V3 and V3b share them): the tile
    grid ``g`` and xgcm ``grid``, hour 0's JMD95 ``b_t``, the time-midpoint
    ``U``, ``V`` of M0's two hours (raw staggered), ``mask_analysis`` as
    ``ana`` and the 32-cell block labels."""
    g = grid_ds if grid_ds is not None else ot.open_grid(with_face=True)
    g = g if 'face' in g.dims else g.expand_dims('face')
    grid = ot.build_xgcm(g)
    raw = xr.open_zarr(ot.DATA_DIR / RAW).load()
    ds = [xr.merge([raw.isel(time=k).expand_dims('face'), g], compat='override',
                   combine_attrs='override').astype('float64') for k in (0, 1)]
    b_t = op.buoyancy(ds[0])
    U_mid, V_mid = sl.midpoint_time(ds[0].U, ds[1].U), sl.midpoint_time(ds[0].V, ds[1].V)
    masks_path = ot.DATA_DIR / 'tile330_masks.nc'
    masks = mk.open_masks(masks_path) if masks_path.exists() else mk.build_masks(g)
    ana = masks['mask_analysis'].values
    return dict(g=g, grid=grid, b_t=b_t, U=U_mid, V=V_mid, ana=ana, block=_block_ids(ana.shape))


def _fit_llc(inp, b, U, V, fms, order, vel_order=3, boot=N_BOOT, advect=None):
    """One null step on the tile and the pre-declared fit: valid =
    ``mask_analysis`` & finite on every side, front pixels ``G_mid >= p90``
    over valid, the estimators and block bootstrap per form."""
    st = null_step(b, U, V, inp['g'], inp['grid'], order=order, forms=fms, vel_order=vel_order, advect=advect)
    valid = inp['ana'] & np.isfinite(st['measured']) & np.isfinite(st['G_mid'])
    for f in fms:
        valid &= np.isfinite(st[f'two_F_{f}'])
    front, thr = front_pixels(st['G_mid'], valid)
    block = inp['block']
    fits = {f: _fit(st[f'two_F_{f}'][front], st['measured'][front], block[front], boot) for f in fms}
    st.update(front=front, valid=valid, threshold=thr)
    return st, fits


def _discrete_null(velocities, order, forms, widths, angles, n, a, margin, n_boot, diag_widths, grid_ds,
                   changes):
    """The numbers behind :func:`test_discrete_null` for one variant, plus
    the figure context (the pooled front-pixel arrays)."""
    forms = tuple(forms)
    res = dict(velocities=velocities, interp_order=int(order), forms=list(forms), vel_order=3,
               front_definition=f'G_mid = gradb2(0.5 (b_t + b_tp1)) >= p{FRONT_PERCENTILE:g} over the valid set '
                                + ('(mask_analysis & finite)' if velocities == 'llc'
                                   else f'(interior, {margin}-cell margin, pooled over cases)'),
               gate_estimator='OLS of measured DG/Dt on 2F with intercept, on front pixels',
               bootstrap=f'{BLOCK_CELLS}x{BLOCK_CELLS}-cell spatial blocks, {n_boot} draws, 2.5-97.5%',
               changes_tried={})
    ctx = dict(velocities=velocities, forms=forms)
    if velocities == 'strain':
        res.update(widths_cells=list(widths), angles_deg=list(angles), n=int(n), a=float(a),
                   modes=dict(sy.NULL_MODES))
        r, pool, steps = _null_strain(order, forms, widths, angles, n, a, margin, n_boot)
        res.update(r)
        if diag_widths:                          # out of the pool: where the operators break down
            rd, _, _ = _null_strain(order, forms, diag_widths, angles, n, a, margin, 50)
            res['diag_widths'] = dict(widths=list(diag_widths), per_width=rd['per_width'])
        ctx.update(pool=pool, steps=steps)
        if changes:
            # the first attempt: the chain form with the bilinear departure velocity
            r1, pool1, _ = _null_strain(order, ('chain',), widths, angles, n, a, margin, 200, vel_order=1)
            res['changes_tried']['chain, bilinear departure velocity (first attempt)'] = r1['fits']['chain']['ols']
            res['first_attempt'] = r1['fits']['chain']
            ctx['pool_first'] = pool1['chain']
            for f in forms:
                res['changes_tried'][f'{f}, cubic departure velocity'] = res['fits'][f]['ols']
            alt = {str(w): _fourth_order_everywhere(sy.null_strain_case(ell_cells=w, theta_deg=0.0, a=a, n=n))
                   for w in widths}
            res['changes_tried']['4th-order gradient on both sides, chain rule (per width, angle 0)'] = alt
    elif velocities == 'llc':
        inp = _llc_inputs(grid_ds)
        g, grid, b_t, U_mid, V_mid, ana, block = (inp[k] for k in ('g', 'grid', 'b_t', 'U', 'V', 'ana', 'block'))

        def fit_llc(b, U, V, fms, vel_order=3, boot=n_boot):
            return _fit_llc(inp, b, U, V, fms, order, vel_order=vel_order, boot=boot)
        st, fits = fit_llc(b_t, U_mid, V_mid, forms)
        front = st['front']
        res.update(threshold=st['threshold'], n_valid=int(st['valid'].sum()), n_analysis=int(ana.sum()),
                   fits=fits, tracer='hour 0 JMD95 b', velocity='0.5 (U_t + U_tp1) of M0\'s two hours')
        d = np.hypot(st['di'], st['dj'])
        res['displacement_cells'] = dict(median=float(np.nanmedian(d[front])), p99=float(np.nanpercentile(d[front], 99)),
                                         max=float(np.nanmax(d[front])))
        res['signal'] = dict(median_2F_dt_over_G=float(np.median(st[f'two_F_{forms[-1]}'][front] * DT / st['G_mid'][front])),
                             median_abs_meas_dt_over_G=float(np.median(np.abs(st['measured'][front]) * DT / st['G_mid'][front])))
        pool = {f: dict(x=st[f'two_F_{f}'][front], y=st['measured'][front]) for f in forms}
        ctx.update(pool=pool, step=st, ana=ana)
        if changes:
            st1, fits1 = fit_llc(b_t, U_mid, V_mid, forms, vel_order=1, boot=200)
            ch = res['changes_tried']
            if 'chain' in fits1:
                ch['chain, bilinear departure velocity (first attempt)'] = fits1['chain']['ols']
                res['first_attempt'] = fits1['chain']
                ctx['pool_first'] = dict(x=st1['two_F_chain'][st1['front']], y=st1['measured'][st1['front']])
            for f in forms:
                ch[f'{f}, bilinear departure velocity'] = fits1[f]['ols']
            for f in forms:
                ch[f'{f}, cubic departure velocity'] = fits[f]['ols']
            gate_form = forms[-1]
            _, fv = fit_llc(b_t, op.lowpass(U_mid, 8), op.lowpass(V_mid, 8), (gate_form,), boot=200)
            ch[f'{gate_form}, velocity low-passed L = 8 (b raw)'] = fv[gate_form]['ols']
            _, fb = fit_llc(op.lowpass(b_t, 8), U_mid, V_mid, (gate_form,), boot=200)
            ch[f'{gate_form}, b low-passed L = 8 (velocity raw)'] = fb[gate_form]['ols']
            # is the Jacobian attenuation visible?  the departure map's strain vs the
            # Jacobian's trace vs the flux-form divergence, on the front pixels
            ux, uy, vx, vy = op.jacobian(U_mid, V_mid, g, grid)
            trJ = (ux + vy).values[0]
            delta = op.strain_divergence(U_mid, V_mid, g, grid)[0].values[0]

            def D(arr, ax):
                return (np.roll(arr, -1, ax) - np.roll(arr, 1, ax)) / 2.0
            trd = (D(st['di'], 1) + D(st['dj'], 0)) / DT
            ok = front & np.isfinite(trJ) & np.isfinite(delta) & np.isfinite(trd)
            res['strain_seen'] = dict(
                departure_vs_jacobian=float(np.sum(trJ[ok] * trd[ok]) / np.sum(trJ[ok] ** 2)),
                jacobian_vs_fluxform=float(np.sum(delta[ok] * trJ[ok]) / np.sum(delta[ok] ** 2)),
                departure_vs_fluxform=float(np.sum(delta[ok] * trd[ok]) / np.sum(delta[ok] ** 2)),
                note='the null sees D_h u_c on both sides: the Jacobian trace IS the wide centred difference '
                     'of the centred velocity, (1,2,1)/4 of the flux-form divergence')
    else:
        raise ValueError(f"velocities must be 'strain' or 'llc', got {velocities!r}")
    gate = res['fits'][forms[-1]]
    res.update(slope=gate['ols'], ci=gate['bootstrap']['ci'], n_front=gate['n'], gate=gate['gate'],
               gate_form=forms[-1])
    return res, ctx


def test_discrete_null(velocities='strain', png=True, order=3, forms=('chain', 'discrete_o2', 'discrete'),
                       widths=NULL_WIDTHS, angles=NULL_ANGLES, n=128, a=1e-5, margin=8, n_boot=N_BOOT,
                       diag_widths=NULL_DIAG_WIDTHS, grid_ds=None, changes=True) -> dict:
    """**V3**: the discrete end-to-end null (criterion 3).  A tracer is
    advected one hour by our own semi-Lagrangian step, so the truth obeys
    our discrete advection exactly; the measured ``DG/Dt``
    (``semilag.measured_DGDt``) is regressed on ``2F``
    (``operators.frontogenesis`` at the trajectory midpoint) over front
    pixels.  ``velocities='strain'``: prescribed deformation plus shear /
    divergence modes, tanh fronts of width ``widths`` (cells) and normals
    at ``angles`` from the compressional axis, pooled (``synthetic.
    null_strain_case``); ``'llc'``: the real midpoint velocity of M0's two
    hours, on the tile grid, with hour 0's JMD95 ``b`` as the tracer, on
    ``mask_analysis`` (``needs_grid``).  Front pixels, the gate estimator
    and the bootstrap are the module constants above, declared in advance.

    ``forms`` are the ``frontogenesis`` forms fitted; **the last is the
    gate** (``'discrete'``, the consistent form that made the gate pass;
    ``'chain'`` is the first attempt).  ``changes=True`` also records every
    change tried (``changes_tried``: the bilinear departure velocity of the
    first attempt, each form, the low-passed inputs, the 4th-order-everywhere
    alternative).  Returns the gate slope (``slope``; Figure 2's baseline),
    its block-bootstrap CI (``ci``), the other estimators (``fits``), ``n_front``
    and the definitions.  ``png=True`` writes **V3** with *both* variants (the
    other one is computed too; the 'llc' half needs the M0 stores).
    """
    res, ctx = _discrete_null(velocities, order, forms, widths, angles, n, a, margin, n_boot, diag_widths,
                              grid_ds, changes)
    res['png'] = None
    if png:
        other = 'llc' if velocities == 'strain' else 'strain'
        try:
            res_o, ctx_o = _discrete_null(other, order, forms, widths, angles, n, a, margin, n_boot,
                                          diag_widths, grid_ds, changes)
        except FileNotFoundError:                # no M0 stores: the figure gets the strain half only
            res_o, ctx_o = None, None
        pair = {velocities: (res, ctx), other: (res_o, ctx_o)}
        res['png'] = vf.fig_V3(pair['strain'][0], pair['strain'][1], pair['llc'][0], pair['llc'][1])
    return res


# ---------------------------------------------------------------------------
# V3b: the finite-volume null (task 6b; M1-Q2 -- a recorded bias, not a gate)
# ---------------------------------------------------------------------------
# Same pre-declared front-pixel rule (G_mid >= p90), gate estimator (OLS of
# measured on 2F with intercept) and 32-cell block bootstrap as V3; only the
# truth changes: b_tp1 from fvadvect.fv_advect instead of our semi-Lagrangian
# step.  Reference scheme for the attribution: 'centred' (no dissipation --
# the pure C-grid stencil effect); the departure from it per scheme is the
# scheme's implicit diffusion, an exact DG/Dt term (one b_t, one departure).
FV_SCHEMES = ('centred', 'os7', 'os7mp', 'dst3')
FV_REFERENCE = 'centred'
FV_HEADLINE = 'os7mp'


def _fv_pool_strain(steps, forms):
    """Front-pixel pools of a strain run: ``2F`` per form, ``measured``,
    ``G_mid``, ``G_tp1``, block ids, front width."""
    out = dict(measured=[], G_mid=[], G_tp1=[], block=[], width=[], **{f'two_F_{f}': [] for f in forms})
    for st in steps:
        fr = st['front']
        for k in ('measured', 'G_mid', 'G_tp1', 'block'):
            out[k].append(st[k][fr])
        out['width'].append(np.full(fr.sum(), st['case'][0]))
        for f in forms:
            out[f'two_F_{f}'].append(st[f'two_F_{f}'][fr])
    return {k: np.concatenate(v) for k, v in out.items()}


def _fv_attribution(pool_s, pool_ref_G_tp1, forms, n_boot):
    """The scheme's implicit diffusion as a ``DG/Dt`` term on the scheme's
    own front pixels: ``diff = [G(b_tp1^scheme) - G(b_tp1^centred)]/dt``
    (exact: the departure and ``b_t`` are shared, so it is the whole
    difference of the measured sides).  Reported as its OLS slope on
    ``2F`` per form (the part of the slope it accounts for), its rms
    relative to ``2F``, and ``diff dt / G_mid`` (the fraction of ``G``
    lost per hour; ``-2 kappa k^2 dt`` for a diffusivity ``kappa``)."""
    diff = (pool_s['G_tp1'] - pool_ref_G_tp1) / DT
    out = dict(n=int(diff.size), rms_over_rms_2F={}, slope_on_2F={},
               median_dt_over_G=float(np.median(diff * DT / pool_s['G_mid'])),
               mean_dt_over_G=float(np.mean(diff * DT / pool_s['G_mid'])),
               p10_dt_over_G=float(np.percentile(diff * DT / pool_s['G_mid'], 10)),
               fraction_negative=float(np.mean(diff < 0)))
    for f in forms:
        x = pool_s[f'two_F_{f}']
        out['slope_on_2F'][f] = float(slope_estimators(x, diff)['ols'])
        out['rms_over_rms_2F'][f] = _rms(diff) / _rms(x)
    return out


def _fv_null(velocities, schemes, forms, order, widths, angles, n, a, margin, n_boot, dt_sub,
             diag_widths, grid_ds):
    """The numbers behind :func:`test_fv_null` for one variant, plus the
    figure context."""
    import fvadvect as fv
    forms, schemes = tuple(forms), tuple(schemes)
    res = dict(velocities=velocities, schemes=list(schemes), forms=list(forms), interp_order=int(order),
               vel_order=3, dt_sub=float(dt_sub), reference=FV_REFERENCE, headline_scheme=FV_HEADLINE,
               truth='flux-form finite-volume step on the C-grid (fvadvect.fv_advect), advective form '
                     '(-[div(u b) - b div u]), same midpoint velocity as V3',
               front_definition=f'G_mid = gradb2(0.5 (b_t + b_tp1)) >= p{FRONT_PERCENTILE:g} over the valid set '
                                + ('(mask_analysis & finite)' if velocities == 'llc'
                                   else f'(interior, {margin}-cell margin, pooled over cases)'),
               estimator='OLS of measured DG/Dt on 2F with intercept, on front pixels (V3\'s)',
               bootstrap=f'{BLOCK_CELLS}x{BLOCK_CELLS}-cell spatial blocks, {n_boot} draws, 2.5-97.5%',
               status='recorded bias, not a gate (M1-Q2)', per_scheme={}, table={})
    ctx = dict(velocities=velocities, forms=forms, schemes=schemes, pools={}, steps={})
    if velocities == 'strain':
        res.update(widths_cells=list(widths), angles_deg=list(angles), n=int(n), a=float(a))
    elif velocities == 'llc':
        inp = _llc_inputs(grid_ds)
        res.update(n_analysis=int(inp['ana'].sum()), tracer='hour 0 JMD95 b',
                   velocity='0.5 (U_t + U_tp1) of M0\'s two hours')
    else:
        raise ValueError(f"velocities must be 'strain' or 'llc', got {velocities!r}")
    for scheme in schemes:
        adv = fv.advector(scheme, dt_sub=dt_sub)
        if velocities == 'strain':
            r, _, steps = _null_strain(order, forms, widths, angles, n, a, margin, n_boot, advect=adv)
            entry = dict(fits=r['fits'], per_width=r['per_width'], per_case=r['per_case'], threshold=r['threshold'])
            if diag_widths:
                rd, _, _ = _null_strain(order, forms, diag_widths, angles, n, a, margin, 50, advect=adv)
                entry['diag_widths'] = dict(widths=list(diag_widths), per_width=rd['per_width'])
            pool = _fv_pool_strain(steps, forms)
            ctx['steps'][scheme] = steps
        else:
            st, fits = _fit_llc(inp, inp['b_t'], inp['U'], inp['V'], forms, order, boot=n_boot, advect=adv)
            fr = st['front']
            entry = dict(fits=fits, threshold=st['threshold'], n_valid=int(st['valid'].sum()))
            d = np.hypot(st['di'], st['dj'])
            entry['displacement_cells'] = dict(median=float(np.nanmedian(d[fr])), p99=float(np.nanpercentile(d[fr], 99)))
            pool = dict(measured=st['measured'][fr], G_mid=st['G_mid'][fr], G_tp1=st['G_tp1'][fr],
                        **{f'two_F_{f}': st[f'two_F_{f}'][fr] for f in forms})
            ctx['steps'][scheme] = st
        entry['n_front'] = int(pool['measured'].size)
        res['per_scheme'][scheme] = entry
        ctx['pools'][scheme] = pool
        res['table'][scheme] = {f: dict(slope=entry['fits'][f]['ols'], ci=entry['fits'][f]['bootstrap']['ci'],
                                        se=entry['fits'][f]['bootstrap']['se'], n=entry['fits'][f]['n'],
                                        corr=entry['fits'][f]['corr']) for f in forms}
    # attribution: stencil = centred - 1 (no dissipation); diffusion = scheme - centred,
    # with the reference's G_tp1 re-read on THIS scheme's front pixels (b_mid, hence
    # the p90 set, moves slightly with the truth)
    def ref_on(scheme, ref):
        """``G_tp1`` of scheme ``ref`` on scheme ``scheme``'s front pixels."""
        if velocities == 'strain':
            return np.concatenate([st_r['G_tp1'][st_s['front']] for st_r, st_s
                                   in zip(ctx['steps'][ref], ctx['steps'][scheme])])
        return ctx['steps'][ref]['G_tp1'][ctx['steps'][scheme]['front']]

    def attribution(scheme, ref):
        att = _fv_attribution(ctx['pools'][scheme], ref_on(scheme, ref), forms, n_boot)
        att['reference'] = ref
        att['slope_shift'] = {f: res['per_scheme'][scheme]['fits'][f]['ols'] - res['per_scheme'][ref]['fits'][f]['ols']
                              for f in forms}
        return att
    if FV_REFERENCE in schemes:
        ref = res['per_scheme'][FV_REFERENCE]
        res['stencil_effect'] = {f: ref['fits'][f]['ols'] - 1.0 for f in forms}
        for scheme in schemes:
            if scheme not in (FV_REFERENCE, 'semilag'):
                res['per_scheme'][scheme]['diffusion'] = attribution(scheme, FV_REFERENCE)
    # the limiter alone: os7mp against the unlimited os7 (the same reconstruction)
    if 'os7' in schemes and 'os7mp' in schemes:
        res['per_scheme']['os7mp']['limiter'] = attribution('os7mp', 'os7')
    head = res['per_scheme'].get(FV_HEADLINE, res['per_scheme'][schemes[-1]])
    hf = head['fits']['discrete'] if 'discrete' in forms else head['fits'][forms[-1]]
    res.update(slope=hf['ols'], ci=hf['bootstrap']['ci'], n_front=hf['n'],
               bias=dict(scheme=FV_HEADLINE if FV_HEADLINE in schemes else schemes[-1],
                         form='discrete' if 'discrete' in forms else forms[-1],
                         slope=hf['ols'], ci=hf['bootstrap']['ci'],
                         meaning='OLS slope of the measured DG/Dt on 2F when the truth is a model-like flux-form '
                                 'advection: the factor by which M3\'s slope is biased by our pipeline (1 = none)'))
    return res, ctx


def test_fv_null(velocities='strain', schemes=FV_SCHEMES, png=True, forms=('chain', 'discrete'), order=3,
                 widths=NULL_WIDTHS, angles=NULL_ANGLES, n=128, a=1e-5, margin=8, n_boot=N_BOOT,
                 dt_sub=None, diag_widths=NULL_DIAG_WIDTHS, grid_ds=None) -> dict:
    """**V3b**: the finite-volume null (task 6b; M1-Q2, option (a): a
    *recorded bias*, not a gate).  Exactly V3's pipeline -- the measured
    ``semilag.measured_DGDt`` and ``2 * operators.frontogenesis`` at the
    trajectory midpoint, ``form='discrete'`` and ``'chain'``, the
    pre-declared ``G_mid >= p90`` front pixels, the OLS estimator and the
    32-cell block bootstrap -- with the truth replaced by a **flux-form
    finite-volume advection on the C-grid** (``fvadvect``: the same
    midpoint velocity, face transports ``U dyG hFacW``, the advective form
    ``-[div(u b) - b div u]`` as MITgcm's multi-dimensional sweep), so that
    measured and predicted can disagree for the reason the model's fronts
    might: the tracer feels the flux-form strain of the *face* velocities,
    our ``F`` the ``(1,2,1)/4``-smoothed strain of the centred ones (the
    0.85x of M0 task 5 / M1 task 2) -- and the scheme's implicit diffusion.

    ``schemes``: ``'centred'`` (second order, no dissipation: the pure
    stencil effect, the reference), ``'os7'`` (unlimited seventh-order
    one-step), ``'os7mp'`` (with the MP limiter: the OS7MP-like headline),
    ``'dst3'`` (third order, the cross-check); ``'semilag'`` reproduces V3.
    ``dt_sub`` sub-step (s; default ``fvadvect.DT_SUB = 100``).  Returns
    ``table[scheme][form]`` (slope, CI, se, n), ``per_scheme`` (the
    estimators, per width for 'strain', the diffusion attribution vs the
    centred reference: ``slope_shift``, ``slope_on_2F``, ``rms_over_rms_2F``,
    ``median_dt_over_G``), ``stencil_effect`` (centred slope - 1), and the
    headline ``bias`` (``os7mp`` x ``discrete``: ``slope``, ``ci``).
    ``png=True`` writes **V3b** with both variants (the 'llc' half needs
    the M0 stores).
    """
    import fvadvect as fv
    dt_sub = fv.DT_SUB if dt_sub is None else dt_sub
    res, ctx = _fv_null(velocities, schemes, forms, order, widths, angles, n, a, margin, n_boot, dt_sub,
                        diag_widths, grid_ds)
    res['png'] = None
    if png:
        other = 'llc' if velocities == 'strain' else 'strain'
        try:
            res_o, ctx_o = _fv_null(other, schemes, forms, order, widths, angles, n, a, margin, n_boot, dt_sub,
                                    diag_widths, grid_ds)
        except FileNotFoundError:
            res_o, ctx_o = None, None
        pair = {velocities: (res, ctx), other: (res_o, ctx_o)}
        res['png'] = vf.fig_V3b(pair['strain'][0], pair['strain'][1], pair['llc'][0], pair['llc'][1])
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
for _f in (test_cartesian_deformation, test_native_metric, test_interpolation_bias, test_discrete_null):
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
