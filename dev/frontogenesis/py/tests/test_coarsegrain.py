""" Tests for ``coarsegrain.py`` (coding doc §5, prompt 2 task 4).

Synthetic C-grid tests run offline (grids from ``test_operators``); the
real-tile smoke test is marked ``needs_grid``.

Guards: ``tau -> 0`` as ``L -> 0`` (exactly 0 at ``L = 0``; the Clark /
gradient-model asymptotics ``tau ~ L (L + 2) dx^2 / 12 grad u . grad b`` for
smooth fields); the Germano identity across two filter levels (round-off
for the composite filter; a single wider top-hat is *not* the composite,
and the mismatch is shown); the flux-form divergence against
``operators.strain_divergence``; **closure** of the coarse-grained budget
on exact solutions of the advection equation for a non-divergent shear and
a divergent separable flow -- the ``bbar`` budget closes to ``O(dx^2)``
and the ``Gbar`` budget ``Dbar Gbar/Dt = 2 Fbar + 2 T`` closes to the
stated tolerance, the residual shrinks with resolution, it does not close
without the term, and on the divergent flow it does not close with the
flux form alone; NaN propagation at a synthetic coast; and the two-hour
smoke test with the subfilter / ``Fbar`` ratios by ``L``.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

import operators as op
import osn_tiles as ot
import semilag as sl
import coarsegrain as cg
from test_operators import (synthetic_cgrid, model_components, da, C_DIMS, U_DIMS, V_DIMS)

DX0, DT = 1800.0, 3600.0


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype='float64') ** 2)))


def inner(shape, m):
    out = np.zeros(shape, bool)
    out[..., m:-m, m:-m] = True
    return out


# ---------------------------------------------------------------------------
# exact solutions of d_t b + u . grad b = 0
# ---------------------------------------------------------------------------
def backtrack_sine(x, a, k, t):
    """Position at time 0 of the parcel that is at ``x`` at time ``t`` under
    ``dx/dt = a sin(k x)``: with ``theta = k x``,
    ``tan(theta0/2) = tan(theta/2) exp(-a k t)`` on each branch
    ``theta in (2 pi n - pi, 2 pi n + pi)`` (parcels never cross the
    stagnation points ``theta = pi (2n + 1)``)."""
    n = np.round(k * x / (2 * np.pi))
    xi = k * x - 2 * np.pi * n
    return (2 * np.pi * n + 2 * np.arctan(np.tan(xi / 2) * np.exp(-a * k * t))) / k


def b0_field(x, y, x0, y0):
    """A front tilted 30 degrees (width ``4 dx0 = 7.2 km``) plus a
    smaller-scale sinusoid (21.6 x 18 km), so ``b_x``, ``b_y`` and
    structure near the filter width are all present."""
    ell, phi = 4 * DX0, np.deg2rad(30.0)
    r = (x - x0) * np.cos(phi) + (y - y0) * np.sin(phi)
    return (1e-2 * np.tanh(r / ell)
            + 3e-3 * np.sin(2 * np.pi * (x - x0) / (12 * DX0)) * np.cos(2 * np.pi * (y - y0) / (10 * DX0)))


def flow(kind, x0, y0):
    """``(u, v, X0, Y0)``: the steady geographic velocity and the exact
    back-trajectory map, so ``b(x, y, t) = b0(X0(x, y, t), Y0(x, y, t))``.
    'shear': ``u = c sin(m y)``, ``v = 0`` (non-divergent, ``ubar != u``).
    'divergent': ``u = a sin(k x)``, ``v = c sin(m y)`` (separable, so the
    map is a product of the 1-D sine back-trajectories; ``delta != 0`` and
    varies at 20-29 km)."""
    a, k = 0.2, 2 * np.pi / (20 * DX0)
    c, m = 0.3, 2 * np.pi / (16 * DX0)
    if kind == 'shear':
        return (lambda x, y: c * np.sin(m * (y - y0)), lambda x, y: 0.0 * x,
                lambda x, y, t: x - c * np.sin(m * (y - y0)) * t, lambda x, y, t: y)
    if kind == 'divergent':
        return (lambda x, y: a * np.sin(k * (x - x0)), lambda x, y: c * np.sin(m * (y - y0)),
                lambda x, y, t: x0 + backtrack_sine(x - x0, a, k, t),
                lambda x, y, t: y0 + backtrack_sine(y - y0, c, m, t))
    raise ValueError(kind)


def exact_b(kind, x, y, t, x0, y0):
    u, v, X0, Y0 = flow(kind, x0, y0)
    return b0_field(X0(x, y, t), Y0(x, y, t), x0, y0)


@pytest.mark.parametrize('kind', ['shear', 'divergent'])
def test_exact_solutions_satisfy_the_advection_equation(kind):
    """Sanity: ``d_t b + u b_x + v b_y = 0`` for the analytic solutions (fine
    finite differences, 1e-3 relative)."""
    n = 400
    x = np.linspace(-40e3, 40e3, n)[None, :] + 0 * np.linspace(-30e3, 30e3, n)[:, None]
    y = 0 * x + np.linspace(-30e3, 30e3, n)[:, None]
    u, v, X0, Y0 = flow(kind, 0.0, 0.0)
    t, h, e = 1800.0, 1.0, 5.0
    bt = (exact_b(kind, x, y, t + h, 0, 0) - exact_b(kind, x, y, t - h, 0, 0)) / (2 * h)
    bx = (exact_b(kind, x + e, y, t, 0, 0) - exact_b(kind, x - e, y, t, 0, 0)) / (2 * e)
    by = (exact_b(kind, x, y + e, t, 0, 0) - exact_b(kind, x, y - e, t, 0, 0)) / (2 * e)
    res = bt + u(x, y) * bx + v(x, y) * by
    assert rms(res) < 1e-3 * rms(u(x, y) * bx + v(x, y) * by)
    assert np.allclose(exact_b(kind, x, y, 0.0, 0, 0), b0_field(x, y, 0, 0))


# ---------------------------------------------------------------------------
# tau -> 0 as L -> 0
# ---------------------------------------------------------------------------
def test_tau_vanishes_at_L0_and_follows_the_clark_asymptotics():
    """``L = 0``: ``tau`` is *exactly* 0 wherever finite (``lowpass`` is the
    identity, so ``mean(ub) - ubar bbar`` is the same product twice).
    Smooth fields: ``tau_x -> M2 (U_x b_x + U_y b_y)`` with
    ``M2 = L (L + 2) dx^2 / 12`` (the kernel's second moment; the Clark
    model), ratio 0.97 / 0.91 / 0.73 at ``L = 2 / 4 / 8`` for 50-70 km
    scales, so ``tau = O(L^2)`` as ``L -> 0``."""
    dx, dy = 1800.0, 2000.0
    g, grid, pos = synthetic_cgrid(nj=48, ni=64, dx=dx, dy=dy)
    xu, yu = pos('u')
    x, y = pos('c')
    xv, yv = pos('v')
    lb, lby, lu, luy, lv = 60e3, 50e3, 55e3, 70e3, 65e3
    b = da(1e-2 * np.sin(2 * np.pi * x / lb) * np.cos(2 * np.pi * y / lby), C_DIMS)
    U = da(0.3 * np.sin(2 * np.pi * yu / luy) + 0.1 * np.cos(2 * np.pi * xu / lu), U_DIMS)
    V = da(0.2 * np.cos(2 * np.pi * xv / lv), V_DIMS)
    tx0, ty0 = cg.subfilter_flux(b, U, V, 0, g, grid)
    assert tx0.dims == U_DIMS and ty0.dims == V_DIMS
    assert np.all(tx0.values[np.isfinite(tx0.values)] == 0.0)
    assert np.all(ty0.values[np.isfinite(ty0.values)] == 0.0)
    assert np.isnan(tx0.values[0, :, 0]).all() and np.isnan(ty0.values[0, 0, :]).all()
    # analytic grad U . grad b at the U points
    bx = 1e-2 * (2 * np.pi / lb) * np.cos(2 * np.pi * xu / lb) * np.cos(2 * np.pi * yu / lby)
    by = -1e-2 * (2 * np.pi / lby) * np.sin(2 * np.pi * xu / lb) * np.sin(2 * np.pi * yu / lby)
    Ux = -0.1 * (2 * np.pi / lu) * np.sin(2 * np.pi * xu / lu)
    Uy = 0.3 * (2 * np.pi / luy) * np.cos(2 * np.pi * yu / luy)
    prev = None
    for L, lo in ((2, 0.95), (4, 0.85), (8, 0.6)):
        tx, _ = cg.subfilter_flux(b, U, V, L, g, grid)
        clark = L * (L + 2) / 12.0 * (dx ** 2 * Ux * bx + dy ** 2 * Uy * by)
        m = inner(tx.shape, 6) & np.isfinite(tx.values) & (np.abs(clark) > 0.3 * np.abs(clark).max())
        r = tx.values[m] / clark[m]
        print(f'\nL={L}: tau_x / Clark median {np.median(r):.3f} [{r.min():.3f}, {r.max():.3f}]')
        assert lo < np.median(r) < 1.0 and r.min() > lo - 0.1
        s = rms(tx.values[m])
        if prev is not None:                              # ~ L (L + 2): x3 then x3.3
            assert 2.0 < s / prev < 4.0
        prev = s


# ---------------------------------------------------------------------------
# Germano identity
# ---------------------------------------------------------------------------
def test_germano_identity_holds_to_roundoff_for_the_composite_filter():
    """``T - hat(tau) = L``: the subfilter flux at the composite level
    (``L1`` then ``L2``) minus the ``L2``-filtered ``L1`` flux equals the
    resolved (Leonard) flux of the ``L1``-filtered fields at ``L2``.  An
    algebraic identity for any linear filter, so it holds to round-off
    (max 4e-16 relative) -- *provided* the combined filter is the
    composition; a single top-hat of scale ``L1 + L2`` is not (two
    top-hats compose to a trapezoid), and misses by 30-45% rms."""
    g, grid, pos = synthetic_cgrid(nj=48, ni=64)
    x, y = pos('c')
    xu, yu = pos('u')
    xv, yv = pos('v')
    rng = np.random.default_rng(0)
    b = da(1e-2 * np.sin(2 * np.pi * x / 20e3) * np.cos(2 * np.pi * y / 15e3)
           + 1e-3 * rng.normal(size=x.shape), C_DIMS)
    U = da(0.3 * np.sin(2 * np.pi * yu / 18e3) + 0.1 * np.cos(2 * np.pi * xu / 14e3)
           + 0.01 * rng.normal(size=x.shape), U_DIMS)
    V = da(0.2 * np.cos(2 * np.pi * xv / 16e3) * np.sin(2 * np.pi * yv / 22e3)
           + 0.01 * rng.normal(size=x.shape), V_DIMS)
    for L1, L2 in ((2, 4), (4, 2), (2, 2)):
        tau = cg.subfilter_flux(b, U, V, L1, g, grid)
        T = cg.subfilter_flux(b, U, V, (L1, L2), g, grid)
        Leo = cg.subfilter_flux(op.lowpass(b, L1), op.lowpass(U, L1), op.lowpass(V, L1), L2, g, grid)
        single = cg.subfilter_flux(b, U, V, L1 + L2, g, grid)
        for k in (0, 1):
            lhs = (T[k] - op.lowpass(tau[k], L2)).values
            rhs = Leo[k].values
            m = np.isfinite(lhs) & np.isfinite(rhs)
            assert m.sum() > 2000 and np.array_equal(np.isfinite(lhs), np.isfinite(rhs))
            assert np.abs(lhs[m] - rhs[m]).max() < 1e-13 * np.abs(rhs[m]).max()
            lhs1 = (single[k] - op.lowpass(tau[k], L2)).values
            m1 = np.isfinite(lhs1) & np.isfinite(rhs)
            miss = rms(lhs1[m1] - rhs[m1]) / rms(rhs[m1])
            assert miss > 0.1                                  # not the composite filter


def test_flux_divergence_matches_the_repo_divergence():
    """``flux_divergence(U, V)`` is ``calculate_native_strain_vorticity``'s
    ``divergence_center`` bit-for-bit away from the last row/column, which
    are NaN here (xgcm's zero padding makes them finite and wrong)."""
    g, grid, pos = synthetic_cgrid(nj=20, ni=30, rotated=True)
    rng = np.random.default_rng(1)
    U = da(rng.normal(size=(1, 20, 30)), U_DIMS)
    V = da(rng.normal(size=(1, 20, 30)), V_DIMS)
    d = cg.flux_divergence(U, V, g, grid)
    ref = op.strain_divergence(U, V, g, grid)[0].values
    assert d.dims == C_DIMS
    assert np.isnan(d.values[0, -1, :]).all() and np.isnan(d.values[0, :, -1]).all()
    assert np.array_equal(d.values[0, :-1, :-1], ref[0, :-1, :-1])
    with pytest.raises(ValueError, match='expected dims'):
        cg.flux_divergence(V, U, g, grid)
    with pytest.raises(ValueError, match='expected dims'):
        cg.subfilter_flux(U, U, V, 2, g, grid)


# ---------------------------------------------------------------------------
# closure of the coarse-grained budget
# ---------------------------------------------------------------------------
def closure_level(kind, dx, L, rotated=False, dt=DT, nj0=64, ni0=128, order=3):
    """One resolution level: the exact solution advected for ``dt`` on a
    ``(nj0, ni0) * dx0/dx`` grid, everything filtered at ``L``, and the two
    budgets evaluated on the interior front pixels (``Gbar > 0.2 max``).

    ``Gbar`` budget: ``measured = semilag.measured_DGDt(bbar_t, bbar_tp1,
    Ubar, Vbar)`` against ``2 Fbar + 2 T`` at the midpoint.  ``bbar``
    budget: ``mean(d_t b) + ubar . grad bbar + sigma`` with ``d_t b`` from
    the exact solution (60 s centred difference), ``ubar`` the centred
    filtered velocity rotated to geographic, ``grad`` from ``operators``.
    Returns the rms residual fractions with / without the term and with
    the flux form only, plus the term's size."""
    f = DX0 / dx
    nj, ni = int(nj0 * f), int(ni0 * f)
    g, grid, pos = synthetic_cgrid(nj=nj, ni=ni, dx=dx, dy=dx, rotated=rotated)
    xc, yc = pos('c')
    xu, yu = pos('u')
    xv, yv = pos('v')
    x0, y0 = xc.mean(), yc.mean()
    u, v, X0, Y0 = flow(kind, x0, y0)
    b_t = da(exact_b(kind, xc, yc, 0.0, x0, y0), C_DIMS)
    b_tp1 = da(exact_b(kind, xc, yc, dt, x0, y0), C_DIMS)
    U, _ = model_components(u(xu, yu), v(xu, yu), rotated)
    _, V = model_components(u(xv, yv), v(xv, yv), rotated)
    U, V = da(U, U_DIMS), da(V, V_DIMS)
    # the same filter on b, U and V (coding §1.2)
    bL_t, bL_tp1 = op.lowpass(b_t, L), op.lowpass(b_tp1, L)
    UL, VL = op.lowpass(U, L), op.lowpass(V, L)
    b_mid = sl.midpoint_time(b_t, b_tp1)
    bL_mid = op.lowpass(b_mid, L)                      # == midpoint of the filtered pair
    meas = sl.measured_DGDt(bL_t, bL_tp1, UL, VL, g, grid, dt=dt, order=order).values
    two_F = 2 * op.frontogenesis(bL_mid, UL, VL, g, grid).values
    tx, ty = cg.subfilter_flux(b_mid, U, V, L, g, grid)
    td = cg.subfilter_bdelta(b_mid, U, V, L, g, grid)
    T_flux = cg.subfilter_term(bL_mid, tx, ty, g, grid).values
    T_full = cg.subfilter_term(bL_mid, tx, ty, g, grid, tau_delta=td).values
    G = op.gradb2(bL_mid, g, grid).values
    fin = inner(G.shape, L // 2 + 8) & np.isfinite(meas) & np.isfinite(two_F) & np.isfinite(T_full)
    front = fin & (G > 0.2 * np.nanmax(G[fin]))
    r0 = meas - two_F
    out = dict(kind=kind, dx=dx, L=L, dt=dt, n=int(front.sum()),
               term_frac=rms(2 * T_full[front]) / rms(meas[front]),
               res_no=rms(r0[front]) / rms(meas[front]),
               res=rms((r0 - 2 * T_full)[front]) / rms(meas[front]),
               res_flux=rms((r0 - 2 * T_flux)[front]) / rms(meas[front]))
    # the bbar budget at the midpoint time
    h = 30.0
    bdot = da((exact_b(kind, xc, yc, dt / 2 + h, x0, y0)
               - exact_b(kind, xc, yc, dt / 2 - h, x0, y0)) / (2 * h), C_DIMS)
    u_c, v_c = sl.centre_velocities(UL, VL, g, grid)
    cs, sn = g['CS'], g['SN']
    bx, by = op.grad_b(op.lowpass(da(exact_b(kind, xc, yc, dt / 2, x0, y0), C_DIMS), L), g, grid)
    adv = ((u_c * cs - v_c * sn) * bx + (u_c * sn + v_c * cs) * by).values
    tx, ty = cg.subfilter_flux(da(exact_b(kind, xc, yc, dt / 2, x0, y0), C_DIMS), U, V, L, g, grid)
    td = cg.subfilter_bdelta(da(exact_b(kind, xc, yc, dt / 2, x0, y0), C_DIMS), U, V, L, g, grid)
    sig = cg.subfilter_advection(tx, ty, g, grid, td).values
    sig_flux = cg.subfilter_advection(tx, ty, g, grid).values
    rb = op.lowpass(bdot, L).values + adv
    fb = fin & np.isfinite(sig) & np.isfinite(rb)
    out.update(b_sig_frac=rms(sig[fb]) / rms(adv[fb]), b_res_no=rms(rb[fb]) / rms(adv[fb]),
               b_res=rms((rb + sig)[fb]) / rms(adv[fb]),
               b_res_flux=rms((rb + sig_flux)[fb]) / rms(adv[fb]))
    return out


def _fmt(o):
    return (f'{o["kind"]:9s} dx {o["dx"]:5.0f} L {o["L"]:2d} dt {o["dt"]:4.0f} n {o["n"]:5d} | '
            f'G: 2T/meas {o["term_frac"]:.3f}, residual/meas without {o["res_no"]:.3f}, '
            f'with {o["res"]:.4f}, flux-only {o["res_flux"]:.3f} | '
            f'b: sigma/adv {o["b_sig_frac"]:.3f}, residual/adv without {o["b_res_no"]:.3f}, '
            f'with {o["b_res"]:.4f}, flux-only {o["b_res_flux"]:.3f}')


@pytest.mark.parametrize('kind', ['shear', 'divergent'])
def test_closure_of_the_coarse_grained_budget(kind):
    """The fixed physical set-up (front 7.2 km, velocity scales 20-29 km,
    filter ~9 km) at ``dx = 3.6 / 1.8 / 0.9 km`` with ``L = 2 / 4 / 8``:

    * the ``bbar`` budget closes with ``sigma`` to O(dx^2) (residual 12% ->
      3.6% -> 0.9% of the resolved advection on the shear flow, 22% -> 5% ->
      1.4% on the divergent one; ``sigma`` is 35-54% of the advection) and
      not without (36-75%);
    * the ``Gbar`` budget closes to **< 10%** rms at 900 m (shear: 20% -> 8%
      -> 4%; divergent 33% -> 12% -> 9%) and not without the term (47-85%;
      ``2T`` is 38-60% of the measured tendency); the residual shrinks with
      resolution and, at fixed ``dx``, with ``dt`` (the 4-9% floor at 900 m
      is the midpoint-field time discretisation: ``dt = 900 s`` gives 2-3%);
    * on the divergent flow the flux form alone (no ``tau_delta``) leaves
      38-44% in the ``Gbar`` budget and 28-70% in the ``bbar`` budget -- the
      surface budget needs the dilatation part.
    """
    rows = [closure_level(kind, dx, L) for dx, L in ((3600.0, 2), (1800.0, 4), (900.0, 8))]
    rows.append(closure_level(kind, 900.0, 8, dt=900.0))
    print()
    for o in rows:
        print(_fmt(o))
    lv = rows[:3]
    for o in lv:
        assert o['n'] > 40
        assert o['term_frac'] > 0.3                                   # the term is O(1)
        assert o['res'] < 0.5 * o['res_no']                            # does not close without it
        assert o['b_res'] < 0.4 * o['b_res_no']
    assert lv[0]['res'] > lv[1]['res'] > lv[2]['res']                  # shrinks with resolution
    assert lv[0]['b_res'] > lv[1]['b_res'] > lv[2]['b_res']
    assert lv[2]['res'] < 0.10 and lv[2]['b_res'] < 0.04
    assert lv[2]['b_res'] < 0.4 * lv[0]['b_res']                        # ~ dx^2 on the b budget
    assert rows[3]['res'] < 0.6 * lv[2]['res']                         # the floor is dt
    if kind == 'divergent':
        for o in lv[1:]:                                                # flux form alone fails
            assert o['res_flux'] > 3 * o['res'] and o['b_res_flux'] > 3 * o['b_res']
        assert lv[0]['res_flux'] > 1.2 * lv[0]['res'] and lv[0]['b_res_flux'] > 3 * lv[0]['b_res']
    else:
        for o in lv:
            assert abs(o['res_flux'] - o['res']) < 1e-12                # tau_delta = 0 exactly


def test_closure_on_the_rotated_face():
    """Face-10 orientation (``CS = 0, SN = -1``): ``tau`` stays in the model
    basis and the term is an invariant, so the closure is as good as on the
    unrotated grid."""
    for kind in ('shear', 'divergent'):
        a = closure_level(kind, 1800.0, 4, rotated=True)
        b = closure_level(kind, 1800.0, 4, rotated=False)
        print('\n' + _fmt(a) + '\n' + _fmt(b))
        assert a["res"] < 0.15 and a["res"] < 0.5 * a["res_no"]
        assert abs(a['res'] - b['res']) < 0.03 and abs(a['b_res'] - b['b_res']) < 0.01


# ---------------------------------------------------------------------------
# NaN at a synthetic coast
# ---------------------------------------------------------------------------
def test_nan_propagates_at_a_synthetic_coast():
    """Land NaN in ``b``, NaN on the coast-facing ``U``/``V`` faces: every
    finite ``tau``, ``tau_delta`` and term is *identical* to the land-free
    result (nothing filled, nothing renormalised), the term is NaN within
    chessboard ``L/2 + 2`` of land and finite beyond ``L/2 + 3``, and the
    tile-edge rim is NaN ``L/2 + 2`` deep."""
    nj = ni = 48
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')
    land = (ii < jj - 24) | ((jj == 12) & (ii == 30))
    g, grid, pos = synthetic_cgrid(nj=nj, ni=ni, dx=1796.2, dy=1950.4, land=land)
    x, y = pos('c')
    xu, yu = pos('u')
    xv, yv = pos('v')
    rng = np.random.default_rng(3)
    b_c = 1e-2 * np.sin(2 * np.pi * x / 20e3) * np.cos(2 * np.pi * y / 17e3) + 1e-4 * rng.normal(size=x.shape)
    U_c = 0.4 * np.sin(2 * np.pi * yu / 15e3) + 0.3 + 0.01 * rng.normal(size=x.shape)
    V_c = 0.4 * np.cos(2 * np.pi * xv / 15e3) - 0.2 + 0.01 * rng.normal(size=x.shape)
    U_land, V_land = land | np.roll(land, 1, axis=1), land | np.roll(land, 1, axis=0)

    def fields(with_land):
        b, U, V = da(b_c.copy(), C_DIMS), da(U_c.copy(), U_DIMS), da(V_c.copy(), V_DIMS)
        if with_land:
            b.values[0][land] = np.nan
            U.values[0][U_land] = np.nan
            V.values[0][V_land] = np.nan
        return b, U, V

    chess = ndimage.distance_transform_cdt(~land, metric='chessboard')
    for L in (2, 4, 8):
        hw = L // 2
        outs = []
        for with_land in (True, False):
            b, U, V = fields(with_land)
            tx, ty = cg.subfilter_flux(b, U, V, L, g, grid)
            td = cg.subfilter_bdelta(b, U, V, L, g, grid)
            T = cg.subfilter_term(op.lowpass(b, L), tx, ty, g, grid, tau_delta=td)
            outs.append((tx.values[0], ty.values[0], td.values[0], T.values[0]))
        for a, ref in zip(*outs):
            fin = np.isfinite(a)
            assert np.array_equal(a[fin], ref[fin])                  # no contamination
            assert np.isfinite(ref).sum() > fin.sum()
        T = outs[0][3]
        fin = np.isfinite(T)
        rim = inner(T.shape, hw + 2)
        assert not fin[~rim].any()                                    # the edge rim
        assert chess[fin].min() == hw + 2                             # reach from land
        assert fin[rim & (chess > hw + 3)].all()
        assert np.isnan(T[land]).all()


# ---------------------------------------------------------------------------
# real tile (needs_grid): M0's first hour
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_first_hour_smoke(grid_ds, raw_ds, masks_ds):
    """Hour 0 at ``L = 2, 4, 8``: ``tau``, ``tau_delta`` and the term finite
    on every ``mask_analysis`` cell, dims asserted, and the size of the
    subfilter term relative to ``Fbar`` (rms ratio, median |ratio| and
    correlation on the analysis mask and on front pixels ``Gbar > p90``)
    reported by ``L``, with the dilatation part's share.  The planning's
    claim is that it stays O(1) at every ``L``; asserted only loosely
    (the ratio is between 0.02 and 50) -- the numbers are the record."""
    grid = ot.build_xgcm(grid_ds)
    ds = xr.merge([raw_ds.isel(time=0).expand_dims('face'), grid_ds],
                  compat='override', combine_attrs='override').astype('float64')
    b = op.buoyancy(ds)
    ana = masks_ds['mask_analysis'].values
    print()
    for L in (2, 4, 8):
        bL, UL, VL = op.lowpass(b, L), op.lowpass(ds.U, L), op.lowpass(ds.V, L)
        F = op.frontogenesis(bL, UL, VL, ds, grid).values[0]
        G = op.gradb2(bL, ds, grid).values[0]
        tx, ty = cg.subfilter_flux(b, ds.U, ds.V, L, ds, grid)
        td = cg.subfilter_bdelta(b, ds.U, ds.V, L, ds, grid)
        assert tx.dims == ('face', 'j', 'i_g') and ty.dims == ('face', 'j_g', 'i')
        T_full = cg.subfilter_term(bL, tx, ty, ds, grid, tau_delta=td)
        T_flux = cg.subfilter_term(bL, tx, ty, ds, grid)
        assert T_full.dims == ('face', 'j', 'i') and T_full.shape == (1, 720, 720)
        Tf, Tx = T_full.values[0], T_flux.values[0]
        for arr in (tx.values[0], ty.values[0], td.values[0], Tf, F):
            assert np.isfinite(arr[ana]).all()
        front = ana & (G > np.nanpercentile(G[ana], 90))
        for name, sel in (('analysis', ana), ('front p90', front)):
            ratio = rms(Tf[sel]) / rms(F[sel])
            med = float(np.median(np.abs(Tf[sel]) / np.abs(F[sel])))
            r = float(np.corrcoef(Tf[sel], F[sel])[0, 1])
            share = rms((Tf - Tx)[sel]) / rms(Tf[sel])
            print(f'L={L} {name:9s} n={sel.sum():6d}: rms(term)/rms(F) {ratio:.3f}, median |term/F| {med:.3f}, '
                  f'corr(term, F) {r:+.3f}, rms F {rms(F[sel]):.2e}, rms term {rms(Tf[sel]):.2e} s^-5, '
                  f'dilatation share of the term {share:.2f}, flux-only/full rms {rms(Tx[sel]) / rms(Tf[sel]):.2f}')
            assert 0.02 < ratio < 50
