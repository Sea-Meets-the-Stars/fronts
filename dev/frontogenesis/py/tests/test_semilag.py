""" Tests for ``semilag.py`` (coding doc §5, prompt 2 task 3).

Synthetic C-grid tests run offline (grids from ``test_operators``), in both
the unrotated and the face-10 orientation where the axis pairing matters.
The real-tile smoke test is marked ``needs_grid``.

Guards: the zero-velocity identity (bit-for-bit); uniform-flow translation
by a whole cell (exact); the interpolation order (``order = 1`` shows the
``dx^2 G_xx/8`` bias at a maximum, ``order >= 3`` does not, with numbers);
the ``order >= 3`` rule and its test-only override; the ``U``/``dxC``/``i``
vs ``V``/``dyC``/``j`` pairing of the departure; the midpoint iteration;
honest NaN propagation at a synthetic coast; that ``measured_DGDt``
measures ``DG/Dt`` and not the residual (the shift-then-differentiate
construction gives ~0 under strain); the Eulerian cross-check; and the M0
two-hour smoke test.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import special

import operators as op
import osn_tiles as ot
import semilag as sl
from test_operators import (synthetic_cgrid, deformation_fields, model_components, da,
                            C_DIMS, U_DIMS, V_DIMS)

DX, DY, DT = 1800.0, 2000.0, 3600.0


def inner(shape, m):
    """Mask that drops ``m`` cells on every edge (the finite tile-edge rim
    and the interpolation reach)."""
    out = np.zeros(shape, bool)
    out[..., m:-m, m:-m] = True
    return out


def erf_front(x, x0, sigma_G, b0=1e-2, dx=DX):
    """``b = b0 erf((x - x0) / (sqrt 2 sigma_b))`` with ``sigma_b = sqrt 2
    sigma_G``, so that ``G = b_x^2`` is a Gaussian of standard deviation
    ``sigma_G`` (in cells): ``sigma_G = 1.5`` is planning §5.3's "front ~1.5
    cells wide" with its ``dx^2 G_xx / 8G = -1/(8 sigma_G^2) = -5.56%``
    bilinear bias at the maximum."""
    sb = np.sqrt(2.0) * sigma_G * dx
    return b0 * special.erf((x - x0) / (np.sqrt(2.0) * sb))


# ---------------------------------------------------------------------------
# the interpolation kernel
# ---------------------------------------------------------------------------
def test_kernel_reproduces_polynomials_and_integer_shifts():
    """Lagrange of degree ``order`` reproduces polynomials of that degree
    exactly, and an integer displacement reproduces the node values
    bit-for-bit (weights exactly 1 and 0)."""
    nj, ni = 20, 30
    jj, ii = np.meshgrid(np.arange(nj, dtype=float), np.arange(ni, dtype=float), indexing='ij')
    rng = np.random.default_rng(0)
    for order in (1, 3, 5):
        coef = rng.normal(size=(order + 1, order + 1))
        poly = sum(coef[p, q] * (ii / ni) ** p * (jj / nj) ** q
                   for p in range(order + 1) for q in range(order + 1))
        di, dj = 0.37, -0.61
        got = sl.interp_to_departure(poly, di, dj, order, allow_low_order=True)
        want = sum(coef[p, q] * ((ii - di) / ni) ** p * ((jj - dj) / nj) ** q
                   for p in range(order + 1) for q in range(order + 1))
        m = inner(got.shape, order + 1)
        assert np.isfinite(got[m]).all()
        assert np.allclose(got[m], want[m], rtol=0, atol=1e-13)
        # beyond that degree the error is the truncation, not zero
        field = np.sin(ii / 3.0) * np.cos(jj / 4.0)
        got = sl.interp_to_departure(field, di, dj, order, allow_low_order=True)
        want = np.sin((ii - di) / 3.0) * np.cos((jj - dj) / 4.0)
        err = np.abs(got[m] - want[m]).max()
        assert err < {1: 3e-2, 3: 2e-3, 5: 2e-4}[order]
    # integer shifts: exact, including a leading (face) axis
    f = rng.normal(size=(1, nj, ni))
    got = sl.interp_to_departure(f, 2.0, -1.0, 3)
    m = inner(got.shape, 3)
    assert np.array_equal(got[m], np.roll(f, (-1, 2), axis=(1, 2))[m])
    assert np.isnan(got[0, :, :3]).all() and np.isnan(got[0, -1, :]).all()   # support outside
    # DataArray in, DataArray out
    fda = da(f, C_DIMS)
    out = sl.interp_to_departure(fda, 0.5, 0.25)
    assert isinstance(out, xr.DataArray) and out.dims == C_DIMS and out.attrs['interp_order'] == 3


def test_order_rule_and_override():
    """``order < 3`` raises unless the explicit test-only override is
    given; even orders raise always."""
    f = np.zeros((8, 8))
    with pytest.raises(ValueError, match='order >= 3'):
        sl.interp_to_departure(f, 0.5, 0.5, order=1)
    for bad in (0, 2, 4, -3):
        with pytest.raises(ValueError, match='odd'):
            sl.interp_to_departure(f, 0.5, 0.5, order=bad, allow_low_order=True)
    assert np.isfinite(sl.interp_to_departure(f, 0.5, 0.5, order=1, allow_low_order=True)[3, 3])
    g, grid, pos = synthetic_cgrid(nj=8, ni=8)
    b = da(np.zeros((1, 8, 8)), C_DIMS)
    U, V = da(np.zeros((1, 8, 8)), U_DIMS), da(np.zeros((1, 8, 8)), V_DIMS)
    with pytest.raises(ValueError, match='order >= 3'):
        sl.measured_DGDt(b, b, U, V, g, grid, order=1)
    with pytest.raises(ValueError, match='order >= 3'):
        sl.gradb2_at_departure(b, 0.5, 0.0, g, order=1)
    assert sl.measured_DGDt(b, b, U, V, g, grid, order=1, allow_low_order=True).dims == C_DIMS


# ---------------------------------------------------------------------------
# zero velocity and whole-cell translation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
def test_zero_velocity_identity(rotated):
    """With ``u = 0`` the departure is the arrival: ``G(x_d, t)`` is
    bit-for-bit ``operators.gradb2(b_t)`` (same stencil, same operation
    order) and the measured ``DG/Dt`` of an unchanged field is exactly 0."""
    g, grid, pos = synthetic_cgrid(rotated=rotated)
    x, y = pos('c')
    rng = np.random.default_rng(1)
    b = da(np.sin(2 * np.pi * x / 20e3) * np.cos(2 * np.pi * y / 15e3)
           + 0.1 * rng.normal(size=x.shape), C_DIMS)
    G = op.gradb2(b, g, grid)
    G_d = sl.gradb2_at_departure(b, 0.0, 0.0, g)
    m = inner(G.shape, 3) & np.isfinite(G_d.values)
    assert np.array_equal(G.values[m], G_d.values[m])              # bit-for-bit
    assert np.isfinite(G_d.values[inner(G.shape, 3)]).all()
    bx, by = op.grad_b(b, g, grid)
    bx_d, by_d = sl.grad_b_at_departure(b, 0.0, 0.0, g)
    assert np.array_equal(bx.values[m], bx_d.values[m]) and np.array_equal(by.values[m], by_d.values[m])
    U, V = da(np.zeros(b.shape), U_DIMS), da(np.zeros(b.shape), V_DIMS)
    DG = sl.measured_DGDt(b, b, U, V, g, grid)
    assert DG.dims == C_DIMS and DG.name == 'DGDt_semilag'
    v = DG.values[inner(DG.shape, 3)]
    assert np.isfinite(v).all() and np.all(v == 0.0)                 # exactly zero
    # and the Eulerian estimate of an unchanged field is zero too
    DGe = sl.eulerian_DGDt(G, G, U, V, g, grid)
    assert np.all(DGe.values[inner(DG.shape, 3)] == 0.0)


@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
@pytest.mark.parametrize('axis', ['i', 'j'])
def test_uniform_translation_by_a_whole_cell_is_exact(rotated, axis):
    """A uniform flow of exactly one cell per ``dt`` along ``i`` (``U =
    dxC/dt``) or ``j`` (``V = dyC/dt``) moves the field by one node: the
    departure lands on a node, the kernel reproduces it, and ``DG/Dt`` of
    ``b_tp1 = b_t`` shifted by one cell is exactly 0.  The two spacings
    differ (1800 vs 2250 m -- both exact binary ratios with ``dt``), so
    this also checks the ``U``/``dxC`` and ``V``/``dyC`` pairing."""
    dy = 2250.0
    g, grid, pos = synthetic_cgrid(dy=dy, rotated=rotated)
    x, y = pos('c')
    rng = np.random.default_rng(2)
    b = da(np.sin(2 * np.pi * x / 20e3) * np.cos(2 * np.pi * y / 15e3)
           + 0.1 * rng.normal(size=x.shape), C_DIMS)
    if axis == 'i':
        U, V = da(np.full(b.shape, DX / DT), U_DIMS), da(np.zeros(b.shape), V_DIMS)
        b_tp1 = da(np.roll(b.values, 1, axis=2), C_DIMS)     # b_tp1(i) = b_t(i - 1)
    else:
        U, V = da(np.zeros(b.shape), U_DIMS), da(np.full(b.shape, dy / DT), V_DIMS)
        b_tp1 = da(np.roll(b.values, 1, axis=1), C_DIMS)
    u_c, v_c = sl.centre_velocities(U, V, g, grid)
    di, dj = sl.departure_index(u_c, v_c, g, dt=DT)
    m = inner(b.shape, 4)
    want_di, want_dj = (1.0, 0.0) if axis == 'i' else (0.0, 1.0)
    assert np.all(di.values[m] == want_di) and np.all(dj.values[m] == want_dj)
    DG = sl.measured_DGDt(b, b_tp1, U, V, g, grid, dt=DT)
    v = DG.values[m]
    assert np.isfinite(v).all() and np.all(v == 0.0)
    # the shifted G equals the departure G exactly
    G_d = sl.gradb2_at_departure(b, di, dj, g)
    G_shift = np.roll(op.gradb2(b, g, grid).values, 1, axis=2 if axis == 'i' else 1)
    assert np.array_equal(G_d.values[m], G_shift[m])


# ---------------------------------------------------------------------------
# the departure: axis pairing and the midpoint iteration
# ---------------------------------------------------------------------------
def test_departure_pairing_on_face10_no_rotation():
    """On face 10 an *eastward* flow is model ``V`` (``u_east = V``): the
    displacement is along ``j``, scaled by ``dyC`` (2000 m), and ``di = 0``.
    Nothing is rotated: ``di = U dt/dxC``, ``dj = V dt/dyC``."""
    c = 0.5
    g, grid, pos = synthetic_cgrid(rotated=True)
    xu, yu = pos('u')
    xv, yv = pos('v')
    U, _ = model_components(c + 0 * xu, 0 * yu, True)
    _, V = model_components(c + 0 * xv, 0 * yv, True)
    assert np.all(U == 0) and np.all(V == c)
    u_c, v_c = sl.centre_velocities(da(U, U_DIMS), da(V, V_DIMS), g, grid)
    assert u_c.dims == C_DIMS and u_c.attrs['units'] == 'm s-1'
    assert np.isnan(u_c.values[0, :, -1]).all() and np.isnan(v_c.values[0, -1, :]).all()
    di, dj = sl.departure_index(u_c, v_c, g, dt=DT)
    m = inner(di.shape, 3)
    assert np.allclose(dj.values[m], c * DT / DY, rtol=1e-12) and np.all(di.values[m] == 0)
    assert not np.allclose(dj.values[m], c * DT / DX, rtol=0.01)     # the cross pairing
    # a northward flow is -U: displacement along -i, scaled by dxC
    U, _ = model_components(0 * xu, c + 0 * yu, True)
    _, V = model_components(0 * xv, c + 0 * yv, True)
    u_c, v_c = sl.centre_velocities(da(U, U_DIMS), da(V, V_DIMS), g, grid)
    di, dj = sl.departure_index(u_c, v_c, g, dt=DT)
    assert np.allclose(di.values[m], -c * DT / DX, rtol=1e-12) and np.all(dj.values[m] == 0)


def test_departure_midpoint_iteration_linear_flow():
    """For ``u = -a x`` the exact departure of the parcel arriving at ``x``
    is ``x e^{a dt}``; the iterated midpoint converges to
    ``d = dt u/(1 - a dt/2)`` (third-order agreement, < 1e-3 cell here) whereas
    the first guess ``dt u(x)`` is off by ``(a dt)^2 x/2`` (~0.03 cell)."""
    a = 1e-5
    g, grid, pos = synthetic_cgrid(nj=24, ni=96)
    _, U, V = deformation_fields(g, pos, False, a=a)
    x, y = pos('c')
    xc0 = x.mean()
    u_c, v_c = sl.centre_velocities(U, V, g, grid)
    di, dj = sl.departure_index(u_c, v_c, g, dt=DT, n_iter=3)
    di0, _ = sl.departure_index(u_c, v_c, g, dt=DT, n_iter=0)
    m = inner(di.shape, 4)
    d_exact = ((x - xc0) - (x - xc0) * np.exp(a * DT)) / DX          # arrival - departure, cells
    d_first = -a * DT * (x - xc0) / DX
    assert np.abs(di0.values[m] - d_first[m]).max() < 1e-12
    assert np.abs(di.values[m] - d_exact[m]).max() < 1e-3
    assert np.abs(d_first[m] - d_exact[m]).max() > 0.02             # the iteration matters
    assert di.attrs['n_iter'] == 3 and di.attrs['units'] == 'cells'


# ---------------------------------------------------------------------------
# interpolation order: the dx^2 G_xx/8 bias
# ---------------------------------------------------------------------------
def test_interpolation_order_bias_at_a_front_maximum():
    """The ``sigma_G = 1.5``-cell front shifted by half a cell.  Truth is the
    discrete ``G`` of the exactly shifted ``b``.  ``order = 1`` (bilinear
    ``b`` onto the stencil, then difference) reproduces the negative
    ``dx^2 G_xx/8`` bias at the maximum -- the same leading-order bias as
    bilinear ``G`` (planning §5.3: ~5.5%, prediction -5.56%; measured -5.0%
    / -4.9% against the discrete truth, -5.4% analytic 1-D).  ``order = 3``
    is 9x smaller (-0.54%) and ``order = 5`` 5x smaller again (-0.10%).
    The bias is negative: it *fabricates* frontogenesis."""
    nj, ni = 12, 80
    g, grid, pos = synthetic_cgrid(nj=nj, ni=ni)
    x, _ = pos('c')
    x0 = (ni / 2 + 0.3) * DX
    sigma_G = 1.5
    b = da(erf_front(x, x0, sigma_G), C_DIMS)
    # truth: the same stencil on the exactly shifted field, G_true(x) = G(x - dx/2)
    # (the parcel arriving at x came from half a cell to the left)
    G_true = op.gradb2(da(erf_front(x - 0.5 * DX, x0, sigma_G), C_DIMS), g, grid).values[0, nj // 2]
    G_grid = op.gradb2(b, g, grid)
    k = np.argmax(np.nan_to_num(G_true[4:-4])) + 4
    bias = {}
    for order in (1, 3, 5):
        G_d = sl.gradb2_at_departure(b, 0.5, 0.0, g, order, allow_low_order=True).values[0, nj // 2]
        bias[order] = (G_d[k] - G_true[k]) / G_true[k]
    G_bil = sl.interp_to_departure(G_grid, 0.5, 0.0, 1, allow_low_order=True).values[0, nj // 2]
    bias['bilinear_G'] = (G_bil[k] - G_true[k]) / G_true[k]
    pred = -1.0 / (8 * sigma_G ** 2)
    print(f'\nhalf-cell shift, sigma_G = {sigma_G}: bias at the max: order 1 {bias[1] * 100:+.3f}%, '
          f'order 3 {bias[3] * 100:+.3f}%, order 5 {bias[5] * 100:+.3f}%, bilinear G '
          f'{bias["bilinear_G"] * 100:+.3f}%; prediction dx^2 G_xx/8G = {pred * 100:+.2f}%')
    assert -0.07 < bias[1] < -0.035 and abs(bias[1] - pred) < 0.35 * abs(pred)   # the bias, negative
    assert -0.07 < bias['bilinear_G'] < -0.035
    assert abs(bias[3]) < 0.15 * abs(bias[1]) and abs(bias[3]) < 0.007          # cubic: ~9x smaller
    assert abs(bias[5]) < 0.25 * abs(bias[3]) and abs(bias[5]) < 0.0015         # quintic: ~5x smaller again
    assert bias[3] < 0 and bias[5] < 0                    # still negative at the maximum


# ---------------------------------------------------------------------------
# NaN at a synthetic coast
# ---------------------------------------------------------------------------
def _support_touches_nan(valid, di, dj, order):
    """Independent (loop-based) check of the NaN rule: for every arrival
    cell, does the ``(order+1)^2`` support of the departure point contain an
    invalid node or leave the array?"""
    nj, ni = valid.shape
    m = (order - 1) // 2
    bad = np.zeros((nj, ni), bool)
    for j in range(nj):
        for i in range(ni):
            if not (np.isfinite(di[j, i]) and np.isfinite(dj[j, i])):
                bad[j, i] = True
                continue
            pj, pi = j - dj[j, i], i - di[j, i]
            j0, i0 = int(np.floor(pj)), int(np.floor(pi))
            for kj in range(-m, order - m + 1):
                for ki in range(-m, order - m + 1):
                    jn, inn = j0 + kj, i0 + ki
                    if not (0 <= jn < nj and 0 <= inn < ni) or not valid[jn, inn]:
                        bad[j, i] = True
    return bad


def test_nan_propagates_at_a_synthetic_coast():
    """Land (NaN in ``b``, NaN on the coast-facing ``U``/``V`` faces as in
    the OSN stores): every finite ``DG/Dt`` is *identical* to the land-free
    result (nothing is filled), and the NaN set is exactly the cells whose
    departure support -- for the velocity in the midpoint iteration and for
    the five tracer-stencil points -- touches land or leaves the tile, plus
    ``G(t+dt)``'s own stencil rim."""
    nj = ni = 44
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')
    land = (ii < jj - 22) | ((jj == 10) & (ii == 28))
    g, grid, pos = synthetic_cgrid(nj=nj, ni=ni, dx=1796.2, dy=1950.4, land=land)
    x, y = pos('c')
    rng = np.random.default_rng(5)
    b_clean = 1e-2 * np.sin(2 * np.pi * x / 20e3) * np.cos(2 * np.pi * y / 17e3) \
        + 1e-4 * rng.normal(size=x.shape)
    b1_clean = np.roll(b_clean, 1, axis=2) * 1.02
    xu, yu = pos('u')
    xv, yv = pos('v')
    U_clean = 0.4 * np.sin(2 * np.pi * yu / 15e3) + 0.3
    V_clean = 0.4 * np.cos(2 * np.pi * xv / 15e3) - 0.2
    U_land = land | np.roll(land, 1, axis=1)          # U[i_g = k] sits between cells k-1 and k
    V_land = land | np.roll(land, 1, axis=0)

    def fields(with_land):
        b, b1 = da(b_clean.copy(), C_DIMS), da(b1_clean.copy(), C_DIMS)
        U, V = da(U_clean.copy(), U_DIMS), da(V_clean.copy(), V_DIMS)
        if with_land:
            b.values[0][land] = np.nan
            b1.values[0][land] = np.nan
            U.values[0][U_land] = np.nan
            V.values[0][V_land] = np.nan
        return b, b1, U, V

    order = 3
    b, b1, U, V = fields(True)
    DG = sl.measured_DGDt(b, b1, U, V, g, grid, order=order).values[0]
    DG_clean = sl.measured_DGDt(*fields(False), g, grid, order=order).values[0]
    fin = np.isfinite(DG)
    assert fin.sum() > 0.5 * (~land).sum()
    assert np.array_equal(DG[fin], DG_clean[fin])                     # no fill, no contamination
    assert np.isfinite(DG_clean).sum() > fin.sum()                    # land removed something
    # the NaN set, rebuilt independently
    u_c, v_c = sl.centre_velocities(U, V, g, grid)
    uc, vc = u_c.values[0], v_c.values[0]
    dx_c, dy_c = sl._spacing_at_centres(g)
    ui, vj = uc * DT / dx_c, vc * DT / dy_c
    di, dj = ui.copy(), vj.copy()
    vel_order = 3                       # departure_index's default since M1 task 6 (was bilinear)
    for _ in range(3):
        bad_i = _support_touches_nan(np.isfinite(ui), 0.5 * di, 0.5 * dj, vel_order)
        bad_j = _support_touches_nan(np.isfinite(vj), 0.5 * di, 0.5 * dj, vel_order)
        di_n = sl.interp_to_departure(ui, 0.5 * di, 0.5 * dj, vel_order)
        dj_n = sl.interp_to_departure(vj, 0.5 * di, 0.5 * dj, vel_order)
        assert np.array_equal(np.isnan(di_n), bad_i) and np.array_equal(np.isnan(dj_n), bad_j)
        di, dj = di_n, dj_n
    di_da, dj_da = sl.departure_index(u_c, v_c, g, dt=DT)
    assert np.array_equal(di_da.values[0], di, equal_nan=True)
    valid_b = np.isfinite(b.values[0])
    bad_G_d = np.zeros((nj, ni), bool)
    for ei, ej in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):        # x_d and its four neighbours
        bad_G_d |= _support_touches_nan(valid_b, di + ei, dj + ej, order)
    bad_G_d[:, -1] = True                                             # no high face for the last cell
    bad_G_d[-1, :] = True
    G1 = op.gradb2(b1, g, grid).values[0]
    expect_nan = bad_G_d | np.isnan(G1)
    assert np.array_equal(np.isnan(DG), expect_nan)
    assert np.isnan(DG[land]).all()


# ---------------------------------------------------------------------------
# what is measured: DG/Dt, not the residual
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
def test_deformation_measures_DGDt_not_the_residual(rotated):
    """Pure deformation ``u = -a x, v = a y`` on the exact solution
    ``b = b0 tanh(x e^{at}/ell)``: ``DG/Dt = 2aG`` along a parcel, ``F = aG``
    on the grid.  ``measured_DGDt`` (``b`` interpolated onto the stencil
    *at the departure point*) recovers ``2F`` at the midpoint time to a few
    percent on front pixels (0.95-0.98 for 4-8 dx fronts; the remainder is
    the centred-difference truncation not being conserved as the front
    sharpens -- V3's business), whereas differentiating the *shifted field*
    ``b_t(x_d(x))`` gives ~0: that field is the adiabatic prediction of
    ``b_{t+dt}``, so its ``G`` cancels the kinematic term and what is left
    is the residual.  Same on face 10 (departures in native index space)."""
    a, b0 = 1e-5, 1e-2
    for ell_cells in (4, 8):
        ell = ell_cells * DX
        g, grid, pos = synthetic_cgrid(nj=48, ni=96, rotated=rotated)
        xc, yc = pos('c')
        xc0 = xc.mean()
        b_t = da(b0 * np.tanh((xc - xc0) / ell), C_DIMS)
        b_tp1 = da(b0 * np.tanh((xc - xc0) * np.exp(a * DT) / ell), C_DIMS)
        _, U, V = deformation_fields(g, pos, rotated, a=a, b0=b0, ell=ell)
        DG = sl.measured_DGDt(b_t, b_tp1, U, V, g, grid, dt=DT).values
        b_mid = sl.midpoint_time(b_t, b_tp1)
        two_F = 2 * op.frontogenesis(b_mid, U, V, g, grid).values
        G_mid = op.gradb2(b_mid, g, grid).values
        m = inner(G_mid.shape, 4)
        front = m & (G_mid > 0.2 * np.nanmax(G_mid[m])) & np.isfinite(DG)
        assert front.sum() > 100
        ratio = DG[front] / two_F[front]
        # the naive construction: shift the field, then differentiate
        u_c, v_c = sl.centre_velocities(U, V, g, grid)
        di, dj = sl.departure_index(u_c, v_c, g, dt=DT)
        b_d = sl.interp_to_departure(b_t, di, dj)
        naive = (op.gradb2(b_tp1, g, grid).values - op.gradb2(b_d, g, grid).values) / DT
        print(f'\ndeformation ({"face10" if rotated else "CS=1"}, ell = {ell_cells} dx, n = {front.sum()}): '
              f'measured/2F median {np.median(ratio):.4f} [{ratio.min():.4f}, {ratio.max():.4f}]; '
              f'shift-then-differentiate/2F median {np.median(naive[front] / two_F[front]):+.4f}')
        assert 0.90 < np.median(ratio) < 1.05 and ratio.min() > 0.85 and ratio.max() < 1.1
        assert np.abs(naive[front] / two_F[front]).max() < 0.05
        # the Eulerian cross-check agrees on this smooth, slowly moving field
        G_t, G_tp1 = op.gradb2(b_t, g, grid), op.gradb2(b_tp1, g, grid)
        DGe = sl.eulerian_DGDt(G_t, G_tp1, U, V, g, grid, dt=DT)
        assert DGe.dims == C_DIMS and DGe.name == 'DGDt_euler'
        re = DGe.values[front] / two_F[front]
        assert 0.90 < np.median(re) < 1.05
        assert np.corrcoef(DGe.values[front], DG[front])[0, 1] > 0.99


# ---------------------------------------------------------------------------
# real tile (needs_grid): the two M0 hours
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_two_hours_smoke(grid_ds, raw_ds, masks_ds):
    """M0's two hours: shapes and dims; finite on every ``mask_analysis``
    cell; measured and Eulerian estimates of the same order and correlated
    (reported); displacement statistics as in M0 task 3; the midpoint
    iteration and the velocity interpolation order are sub-0.05-cell
    effects.  A quick measured-vs-``2F`` diagnostic is printed as a logged
    number only (V3 is task 6)."""
    grid = ot.build_xgcm(grid_ds)
    ds = [xr.merge([raw_ds.isel(time=k).expand_dims('face'), grid_ds],
                   compat='override', combine_attrs='override').astype('float64') for k in (0, 1)]
    b_t, b_tp1 = op.buoyancy(ds[0]), op.buoyancy(ds[1])
    U_mid, V_mid = sl.midpoint_time(ds[0].U, ds[1].U), sl.midpoint_time(ds[0].V, ds[1].V)
    ana, oc = masks_ds['mask_analysis'].values, masks_ds['mask_ocean'].values
    u_c, v_c = sl.centre_velocities(U_mid, V_mid, grid_ds, grid)
    di, dj = sl.departure_index(u_c, v_c, grid_ds)
    d = np.hypot(di.values[0], dj.values[0])
    med, p99, dmax = np.nanmedian(d[oc]), np.nanpercentile(d[oc], 99), np.nanmax(d[oc])
    di3, dj3 = sl.departure_index(u_c, v_c, grid_ds, vel_order=1)      # the default is 3 since M1 task 6
    dd = np.hypot(di3.values[0] - di.values[0], dj3.values[0] - dj.values[0])
    di0, dj0 = sl.departure_index(u_c, v_c, grid_ds, n_iter=0)
    dd0 = np.hypot(di0.values[0] - di.values[0], dj0.values[0] - dj.values[0])
    print(f'\ndisplacement (cells, ocean): median {med:.3f}, p99 {p99:.3f}, max {dmax:.3f}; '
          f'NaN in {np.isnan(d[oc]).sum()} ocean cells; vel_order 3 vs 1: max {np.nanmax(dd[ana]):.4f} '
          f'cells; midpoint iteration vs first guess: p99 {np.nanpercentile(dd0[ana], 99):.4f}, '
          f'max {np.nanmax(dd0[ana]):.4f} cells')
    assert 0.3 < med < 0.45 and 1.0 < p99 < 1.6 and dmax < 4.0      # M0 task 3: 0.37 / 1.28 / 3.5
    assert np.nanmax(dd[ana]) < 0.05 and np.nanmax(dd0[ana]) < 0.2
    DGs = sl.measured_DGDt(b_t, b_tp1, U_mid, V_mid, grid_ds, grid)
    G_t, G_tp1 = op.gradb2(b_t, grid_ds, grid), op.gradb2(b_tp1, grid_ds, grid)
    DGe = sl.eulerian_DGDt(G_t, G_tp1, U_mid, V_mid, grid_ds, grid)
    assert DGs.dims == DGe.dims == ('face', 'j', 'i') and DGs.shape == (1, 720, 720)
    assert DGs.dtype == DGe.dtype == np.float64
    s, e = DGs.values[0], DGe.values[0]
    assert np.isfinite(s[ana]).all() and np.isfinite(e[ana]).all()
    assert np.isnan(s[~oc]).all()
    r = np.corrcoef(s[ana], e[ana])[0, 1]
    slope = np.sum(s[ana] * e[ana]) / np.sum(e[ana] ** 2)
    rms_s, rms_e = np.sqrt(np.mean(s[ana] ** 2)), np.sqrt(np.mean(e[ana] ** 2))
    print(f'semilag vs Eulerian on mask_analysis (n = {ana.sum()}): corr {r:.3f}, slope {slope:.3f}, '
          f'rms {rms_s:.3e} / {rms_e:.3e} s^-5, median |.| {np.median(np.abs(s[ana])):.3e} / '
          f'{np.median(np.abs(e[ana])):.3e}; finite ocean cells {np.isfinite(s[oc]).sum()} / '
          f'{np.isfinite(e[oc]).sum()} of {oc.sum()}')
    assert r > 0.5 and 0.5 < rms_s / rms_e < 2.0
    # logged diagnostic only: measured vs 2F at the midpoint (V3 is task 6)
    two_F = 2 * op.frontogenesis(sl.midpoint_time(b_t, b_tp1), U_mid, V_mid, grid_ds, grid).values[0]
    G_mid = sl.midpoint_time(G_t, G_tp1).values[0]
    front = ana & (G_mid > np.nanpercentile(G_mid[ana], 90))
    for name, v in (('semilag', s), ('Eulerian', e)):
        print(f'{name} vs 2F (diagnostic): all analysis corr {np.corrcoef(v[ana], two_F[ana])[0, 1]:.3f}, '
              f'slope {np.sum(v[ana] * two_F[ana]) / np.sum(two_F[ana] ** 2):.3f}; front pixels '
              f'(G_mid > p90, n = {front.sum()}) corr {np.corrcoef(v[front], two_F[front])[0, 1]:.3f}, '
              f'slope {np.sum(v[front] * two_F[front]) / np.sum(two_F[front] ** 2):.3f}')
    DG1 = sl.measured_DGDt(b_t, b_tp1, U_mid, V_mid, grid_ds, grid, order=1, allow_low_order=True).values[0]
    DG5 = sl.measured_DGDt(b_t, b_tp1, U_mid, V_mid, grid_ds, grid, order=5).values[0]
    bias1 = np.median((DG1[front] - s[front]) * 3600.0 / G_mid[front])
    print(f'order 1 - order 3 on front pixels: median (dDGDt dt / G) {bias1:+.4f} (the fabricated '
          f'frontogenesis); order 5 - order 3: rms diff / rms {np.sqrt(np.mean((DG5[front] - s[front]) ** 2)) / np.sqrt(np.mean(s[front] ** 2)):.3f}')
    assert bias1 > 0.02                                             # bilinear fabricates ~5%/h of G
