""" Tests for ``vertical.py``'s physics (M3 task 2; coding §4.6), offline on
``synthetic.py`` grids: ``b_z`` of a two-level analytic profile with the
stated (code-``b``) sign; the vertical term zero for ``b_k1 = b``, equal to
the factorised form for uniform ``b_k1 - b`` and differing from it by
exactly the dropped ``-w grad(b_z) . grad b`` otherwise; the surface-flux
term zero for a uniform flux and sign-determinate for a flux gradient
along ``grad b``; ``oceQsw`` entering with ``f_sw`` and ``oceQnet - oceQsw``
with 1; JMD95 ``alpha``/``beta`` by finite differences; a 3-D ``W`` refused;
an upward-positive flux store refused (here and through ``inputs``); NaN
at land propagating; the dims guards; rotation invariance; the F-units
attrs.  One ``needs_grid`` smoke on the real stores (hour 0, 16 LST).
"""

import numpy as np
import pytest
import xarray as xr

import inputs as inp
import operators as op
import synthetic as sy
import vertical as vt
import dbof.utils.jmd95_xgcm_implementation as jmd95

C = sy.C_DIMS
DRF = np.array([1.0, 1.14, 1.30])
DZ = 0.5 * (DRF[0] + DRF[1])                      # Z[0] - Z[1] = 1.07 m
NJ, NI, DX, DY = 40, 56, 1800.0, 2000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def grid(rotated=False, land=None):
    g, xg, pos = sy.synthetic_cgrid(nj=NJ, ni=NI, dx=DX, dy=DY, rotated=rotated, land=land)
    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI), indexing='ij')
    return g, xg, (ii * DX)[None], (jj * DY)[None]      # index-space x, y (m)


def front_TS(x, T0=17.0, dT=1.0, ell=4 * DX, S0=33.6):
    """A temperature front: warm on the left, cool on the right, so code
    ``b`` (increasing with density) *rises* along +x -- the dense side is +x."""
    T = sy.da(T0 - dT * np.tanh((x - x.mean()) / ell), C)
    S = sy.da(np.full(x.shape, S0), C)
    return T, S


def levels(T0, T1, S=33.6, shape=(1, NJ, NI)):
    """``Theta_k``/``Salt_k`` on ``(face, k, j, i)`` with k = 0, 1, 2 at
    temperatures ``T0, T1`` and a linear continuation for k = 2."""
    T2 = T1 + (T1 - T0) * (vt.level_depths(DRF)[1] - vt.level_depths(DRF)[2]) / DZ
    Th = xr.DataArray(np.stack([np.broadcast_to(t, shape) for t in (T0, T1, T2)], axis=1),
                      dims=('face', 'k', 'j', 'i'))
    Sa = xr.full_like(Th, S)
    return Th, Sa


def buoy(T, S):
    return op.buoyancy(xr.Dataset({'Theta': T, 'Salt': S}))


def inner(a, m=3):
    return a[..., m:-m, m:-m]


# ---------------------------------------------------------------------------
# b_z
# ---------------------------------------------------------------------------
def test_level_depths_match_the_store():
    np.testing.assert_allclose(vt.level_depths(DRF), [-0.5, -1.57, -2.79], atol=1e-12)
    with pytest.raises(ValueError):
        vt.level_depths([1.0])


def test_b_z_two_level_profile_sign_and_value():
    """A 0.2 K warm layer over cooler water: code b is smaller at k = 0 than
    at k = 1 (lighter = smaller code b), so b_z = (b_k0 - b_k1)/dz < 0 --
    the negative of planning §2.2's textbook b_z > 0; its size is
    -g alpha dT/dz (~ -4e-4 s^-2)."""
    g, _, x, _ = grid()
    Th, Sa = levels(17.2, 17.0)
    bz = vt.b_z(Th, Sa, g, DRF)
    assert bz.dims == C and bz.name == 'b_z'
    assert float(bz.max()) < 0 and 'increases with density' in bz.attrs['sign_convention']
    b0 = buoy(Th.isel(k=0, drop=True), Sa.isel(k=0, drop=True))
    b1 = buoy(Th.isel(k=1, drop=True), Sa.isel(k=1, drop=True))
    np.testing.assert_allclose(bz.values, (b0 - b1).values / DZ, rtol=1e-12)
    alpha, _, rho = vt.expansion_coefficients(17.1, 33.6)
    expected = -vt.G * (rho / vt.RHO0_REFERENCE) * alpha * 0.2 / DZ
    np.testing.assert_allclose(bz.values, expected, rtol=2e-3)
    assert bz.attrs['dz_m'] == pytest.approx(1.07) and bz.attrs['z_eval_m'] == -1.0
    # a well-mixed column gives exactly zero
    Th, Sa = levels(17.0, 17.0)
    assert float(np.abs(vt.b_z(Th, Sa, g, DRF)).max()) == 0.0


def test_b_z_second_order_option_equals_first_on_a_linear_profile():
    """The three-level (order 2) estimate at z = -1 m equals the two-level
    one when Theta is linear in z (to the 4e-4 that JMD95's alpha(T) makes
    b not quite linear over 0.3 K), and differs when it is not."""
    g, _, _, _ = grid()
    Th, Sa = levels(17.3, 17.0)
    b1 = vt.b_z(Th, Sa, g, DRF, order=1)
    b2 = vt.b_z(Th, Sa, g, DRF, order=2)
    np.testing.assert_allclose(b2.values, b1.values, rtol=1e-3)
    assert b2.attrs['levels'] == [0, 1, 2] and b2.attrs['order'] == 2
    Th2 = Th.copy()
    Th2[{'k': 2}] = 17.0                                # a kink below k = 1
    b2k = vt.b_z(Th2, Sa, g, DRF, order=2)
    assert not np.allclose(b2k.values, b1.values, rtol=1e-3)
    with pytest.raises(ValueError, match='order'):
        vt.b_z(Th, Sa, g, DRF, order=3)


# ---------------------------------------------------------------------------
# the vertical term
# ---------------------------------------------------------------------------
def fields(rotated=False):
    g, xg, x, y = grid(rotated)
    T, S = front_TS(x)
    b = buoy(T, S)
    bx, by = op.grad_b(b, g, xg)
    # a smooth, non-uniform cell-base velocity (~5e-5 m/s, tidal scale)
    W = sy.da(5e-5 * np.sin(2 * np.pi * x / (20 * DX)) * np.cos(2 * np.pi * y / (16 * DY)), C)
    return g, xg, x, y, b, bx, by, W


def test_vertical_term_is_zero_when_b_k1_equals_b():
    g, xg, x, y, b, bx, by, W = fields()
    term = vt.vertical_term(b, bx, by, b, W, DRF, g, xg)
    assert term.dims == C and term.name == 'vertical_term'
    fin = np.isfinite(term.values) & sy.inner(term.shape, 2)
    assert fin.sum() > 0 and np.all(term.values[fin] == 0.0)


@pytest.mark.parametrize('L', [0, 2])
def test_vertical_term_equals_factorised_for_uniform_b_z(L):
    """With b_k1 - b uniform, T_v = -W (b - b_k1)/dz = +c W: grad_h T_v = c
    grad_h W and the term is -b_z (w_x b_x + w_y b_y) exactly, at L = 0 and
    with the tendency low-passed at L = 2."""
    g, xg, x, y, b, bx, by, W = fields()
    bz_val = -3e-4                                       # a warm layer, code b
    b_k1 = b - bz_val * DZ
    bxL, byL = op.grad_b(op.lowpass(b, L), g, xg)
    full = vt.vertical_term(b, bxL, byL, b_k1, W, DRF, g, xg, L_cells=L)
    bz = xr.full_like(b, bz_val)
    fac = vt.vertical_term_factorised(bxL, byL, bz, W, g, xg, L_cells=L)
    fin = np.isfinite(full.values) & np.isfinite(fac.values)
    assert fin.sum() > 0.5 * full.size
    np.testing.assert_allclose(full.values[fin], fac.values[fin],
                               rtol=1e-10, atol=1e-12 * np.nanmax(np.abs(full.values)))
    # and the sign: upwelling (W > 0) of denser water (b_k1 > b) raises b: T_v > 0
    Tv = vt.vertical_tendency(b, b_k1, W, DRF)
    assert np.all(np.sign(Tv.values[np.isfinite(Tv.values)]) == np.sign(W.values[np.isfinite(Tv.values)]))


def test_vertical_term_minus_factorised_is_the_dropped_term():
    """With b_z and W both linear in (x, y) the centred stencil obeys the
    product rule exactly, so vertical_term - vertical_term_factorised =
    -W (grad b_z . grad b) to round-off: the term the factorised form
    drops (planning §2.2), pinned."""
    g, xg, x, y, b, bx, by, _ = fields()
    bz = sy.da(-2e-4 - 1.5e-4 * (x - x.mean()) / x.max() + 1e-4 * (y - y.mean()) / y.max(), C)
    W = sy.da(4e-5 + 3e-5 * (x - x.mean()) / x.max() - 2e-5 * (y - y.mean()) / y.max(), C)
    b_k1 = b - bz * DZ
    full = vt.vertical_term(b, bx, by, b_k1, W, DRF, g, xg)
    fac = vt.vertical_term_factorised(bx, by, bz, W, g, xg)
    bzx, bzy = op.grad_b(bz, g, xg)
    dropped = -W * (bzx * bx + bzy * by)
    # interior only: the tile rim's gradient is finite but wrong (xgcm pads with 0; M0 task 5)
    fin = np.isfinite(full.values) & np.isfinite(fac.values) & np.isfinite(dropped.values) \
        & sy.inner(full.shape, 2)
    diff = (full - fac).values[fin]
    assert np.nanmax(np.abs(dropped.values)) > 0
    np.testing.assert_allclose(diff, dropped.values[fin], rtol=1e-9,
                               atol=1e-9 * np.nanmax(np.abs(full.values)))
    # it is not negligible here: a real part of the term, not round-off
    assert np.sqrt(np.mean(diff ** 2)) > 0.05 * np.sqrt(np.mean(full.values[fin] ** 2))


def test_three_d_W_raises():
    g, xg, x, y, b, bx, by, W = fields()
    W3 = xr.concat([W, W, W], dim='k_l').transpose('face', 'k_l', 'j', 'i')
    with pytest.raises(ValueError, match='level dim'):
        vt.vertical_term(b, bx, by, b, W3, DRF, g, xg)
    with pytest.raises(ValueError, match='level dim'):
        vt.vertical_term_factorised(bx, by, xr.zeros_like(b), W3, g, xg)
    with pytest.raises(ValueError, match='level dim'):
        vt.vertical_tendency(b, b, W3, DRF)


# ---------------------------------------------------------------------------
# the surface-flux term
# ---------------------------------------------------------------------------
def uniform_TS(T0=17.0, S0=33.6):
    g, xg, x, y = grid()
    T = sy.da(np.full(x.shape, T0), C)
    S = sy.da(np.full(x.shape, S0), C)
    return g, xg, x, y, T, S


def flux(a, name, note='6-hourly forcing, linearly interpolated'):
    da = sy.da(a, C)
    da.name = name
    da.attrs.update(units='W/m^2' if name != 'oceFWflx' else 'kg/m^2/s',
                    sign_convention='positive downward (into the ocean)', forcing_note=note)
    return da


def test_surface_flux_term_zero_for_uniform_flux_and_state():
    g, xg, x, y, T, S = uniform_TS()
    b = buoy(T, S)
    bx, by = op.grad_b(b, g, xg)
    q = flux(np.full(x.shape, 300.0), 'oceQnet')
    sw = flux(np.full(x.shape, 250.0), 'oceQsw')
    fw = flux(np.full(x.shape, -2e-5), 'oceFWflx')
    term = vt.surface_flux_term(bx, by, q, sw, fw, T, S, DRF, g, xg)
    fin = np.isfinite(term.values) & sy.inner(term.shape, 2)     # the rim is finite but wrong
    assert fin.sum() > 0 and np.all(term.values[fin] == 0.0)
    B = vt.surface_buoyancy_tendency(q, sw, fw, T, S, DRF)
    assert float(np.nanstd(B.values)) == 0.0 and float(B.mean()) < 0   # heating lowers code b


def test_surface_flux_sign_heating_gradient_towards_dense_side_is_frontolytic():
    """Expectation, stated before the assertion: with a front whose dense
    side is +x (code b rising along +x) and a heat flux into the ocean that
    *increases* along +x, the dense side is warmed (lightened) more than the
    light side, so the buoyancy contrast across the front is eroded --
    frontolysis -- and ``grad b . grad B_sfc`` must be **negative** on the
    front.  Reversing the flux gradient (more heating on the light side)
    sharpens the front: the term must be **positive**.  In code b: heating
    lowers b, so B_sfc is most negative on the dense (+x) side,
    grad B_sfc points to -x, opposite to grad b."""
    g, xg, x, y = grid()
    T, S = front_TS(x)
    b = buoy(T, S)
    bx, by = op.grad_b(b, g, xg)
    assert np.nanmean(inner(bx.values)) > 0                # dense side is +x
    zero = flux(np.zeros(x.shape), 'oceQsw')
    fw = flux(np.zeros(x.shape), 'oceFWflx')
    q_up = flux(300.0 + 200.0 * (x - x.mean()) / x.max(), 'oceQnet')     # more heating at +x
    term = vt.surface_flux_term(bx, by, q_up, zero, fw, T, S, DRF, g, xg)
    v = inner(term.values)
    assert np.all(v[np.isfinite(v)] <= 0) and np.nanmin(v) < 0
    q_down = flux(300.0 - 200.0 * (x - x.mean()) / x.max(), 'oceQnet')   # more heating at -x
    term2 = vt.surface_flux_term(bx, by, q_down, zero, fw, T, S, DRF, g, xg)
    v2 = inner(term2.values)
    assert np.all(v2[np.isfinite(v2)] >= 0) and np.nanmax(v2) > 0
    # fresh water INTO the ocean on the dense side is frontolytic too (it lightens that side)
    fw_up = flux(2e-5 * (x - x.mean()) / x.max(), 'oceFWflx')
    term3 = vt.surface_flux_term(bx, by, flux(np.zeros(x.shape), 'oceQnet'), zero, fw_up, T, S,
                                 DRF, g, xg)
    v3 = inner(term3.values)
    assert np.all(v3[np.isfinite(v3)] <= 0) and np.nanmin(v3) < 0


def test_qsw_enters_with_f_sw_and_nonsolar_with_one():
    """Q_top = (oceQnet - oceQsw) + f_sw oceQsw: with a uniform state the
    term is linear in Q_top, so term(Qnet = q1 + q2, Qsw = q2) =
    term(q1, 0) + f_sw term(q2, 0); and the default f_sw is the Jerlov-IA
    fraction absorbed in the 1 m cell, 0.521."""
    g, xg, x, y, T, S = uniform_TS()
    b = buoy(front_TS(x)[0], S)                           # any b with a gradient
    bx, by = op.grad_b(b, g, xg)
    fw = flux(np.zeros(x.shape), 'oceFWflx')
    q1 = 100.0 + 150.0 * (x - x.mean()) / x.max()
    q2 = 400.0 * np.cos(2 * np.pi * y / (12 * DY)) + 400.0
    zero = flux(np.zeros(x.shape), 'oceQsw')
    t1 = vt.surface_flux_term(bx, by, flux(q1, 'oceQnet'), zero, fw, T, S, DRF, g, xg)
    t2 = vt.surface_flux_term(bx, by, flux(q2, 'oceQnet'), zero, fw, T, S, DRF, g, xg)
    both = vt.surface_flux_term(bx, by, flux(q1 + q2, 'oceQnet'), flux(q2, 'oceQsw'), fw, T, S,
                                DRF, g, xg)
    f = both.attrs['f_sw']
    assert f == pytest.approx(vt.F_SW) and f == pytest.approx(0.5214, abs=5e-4)
    fin = np.isfinite(both.values)
    np.testing.assert_allclose(both.values[fin], (t1 + f * t2).values[fin], rtol=1e-10)
    # an explicit f_sw = 1 puts all of Qsw into the cell: Q_top = Qnet
    all_in = vt.surface_flux_term(bx, by, flux(q1 + q2, 'oceQnet'), flux(q2, 'oceQsw'), fw, T, S,
                                  DRF, g, xg, f_sw=1.0)
    np.testing.assert_allclose(all_in.values[fin], (t1 + t2).values[fin], rtol=1e-10)
    with pytest.raises(ValueError, match='fraction'):
        vt.surface_flux_term(bx, by, flux(q1, 'oceQnet'), zero, fw, T, S, DRF, g, xg, f_sw=1.5)


def test_sw_fraction_absorbed():
    assert vt.sw_fraction_absorbed(1.0, 2) == pytest.approx(0.52143, abs=1e-4)   # Jerlov IA
    assert vt.sw_fraction_absorbed(1.0, 1) == pytest.approx(0.56456, abs=1e-4)   # Jerlov I
    assert vt.JWTYPE == 2 and vt.F_SW == vt.sw_fraction_absorbed(1.0)
    assert vt.sw_fraction_absorbed(0.0) == 0.0 and vt.sw_fraction_absorbed(100.0) > 0.99
    assert vt.sw_fraction_absorbed(2.14) > vt.sw_fraction_absorbed(1.0)


def test_expansion_coefficients_are_jmd95_finite_differences():
    """alpha, beta at (17 degC, 33.6) from centred differences of the same
    JMD95 density ``operators.buoyancy`` wraps: 2.30e-4 K^-1 and 7.49e-4
    psu^-1 (the prompt's rounded 2.4e-4 is 4 % high), within 0.1 % of a
    ten-times finer step, and consistent with ``operators.buoyancy`` itself."""
    alpha, beta, rho = vt.expansion_coefficients(17.0, 33.6)
    assert alpha == pytest.approx(2.30e-4, rel=0.02) and beta == pytest.approx(7.49e-4, rel=0.02)
    assert rho == pytest.approx(jmd95.jmd95(33.6, 17.0, 0.0))
    a_fine, b_fine, _ = vt.expansion_coefficients(17.0, 33.6, dT=1e-3, dS=1e-3)
    assert alpha == pytest.approx(a_fine, rel=1e-3) and beta == pytest.approx(b_fine, rel=1e-3)
    # d(code b)/dT = -(g/rho0) rho alpha, from operators.buoyancy on a 1-cell dataset
    def bf(T, S):
        ds = xr.Dataset({'Theta': sy.da(np.full((1, 2, 2), T), C), 'Salt': sy.da(np.full((1, 2, 2), S), C)})
        return float(op.buoyancy(ds).values[0, 0, 0])
    db_dT = (bf(17.01, 33.6) - bf(16.99, 33.6)) / 0.02
    db_dS = (bf(17.0, 33.61) - bf(17.0, 33.59)) / 0.02
    assert db_dT == pytest.approx(-vt.G / vt.RHO0_REFERENCE * rho * alpha, rel=1e-9)
    assert db_dS == pytest.approx(vt.G / vt.RHO0_REFERENCE * rho * beta, rel=1e-9)
    # DataArray in, DataArray out, NaN propagates
    T = sy.da(np.full((1, 3, 3), 17.0), C)
    T[0, 1, 1] = np.nan
    a, b, r = vt.expansion_coefficients(T, xr.full_like(T, 33.6))
    assert a.dims == C and np.isnan(a.values[0, 1, 1]) and a.values[0, 0, 0] == pytest.approx(alpha)


def test_upward_positive_flux_store_raises_here_and_through_inputs():
    g, xg, x, y, T, S = uniform_TS()
    b = buoy(front_TS(x)[0], S)
    bx, by = op.grad_b(b, g, xg)
    fw = flux(np.zeros(x.shape), 'oceFWflx')
    q = flux(np.full(x.shape, 300.0), 'oceQnet')
    # negative shortwave: the source's upward-positive convention
    with pytest.raises(ValueError, match='oceQsw < 0'):
        vt.surface_flux_term(bx, by, q, flux(np.full(x.shape, -250.0), 'oceQsw'), fw, T, S, DRF, g, xg)
    # a wrong sign attr is refused even with plausible values
    sw = flux(np.full(x.shape, 250.0), 'oceQsw')
    sw.attrs['sign_convention'] = 'positive upward (out of the ocean)'
    with pytest.raises(ValueError, match='sign_convention'):
        vt.surface_flux_term(bx, by, q, sw, fw, T, S, DRF, g, xg)
    # through inputs: the store-level guard (missing attr, negative values)
    ds = xr.Dataset({'oceQnet': q.squeeze('face'), 'oceQsw': flux(np.full(x.shape, -1.0), 'oceQsw').squeeze('face'),
                     'oceFWflx': fw.squeeze('face')})
    with pytest.raises(ValueError, match='oceQsw < 0'):
        inp.assert_flux_sign(ds)
    ds['oceQsw'] = ds['oceQsw'] * -1
    del ds['oceQsw'].attrs['sign_convention']
    with pytest.raises(ValueError, match='sign_convention'):
        inp.fluxes(ds)


def test_nan_at_land_propagates():
    """A land block (NaN in b, b_k1, W and in the fluxes/state) makes the
    terms NaN on the block and one cell around it (the gradient's reach),
    and leaves every other value identical to the land-free result."""
    g, xg, x, y, b, bx, by, W = fields()
    T, S = front_TS(x)
    land = np.zeros((NJ, NI), bool)
    land[10:16, 20:28] = True
    bz = sy.da(-2e-4 - 1e-4 * (x - x.mean()) / x.max(), C)
    b_k1 = b - bz * DZ
    full0 = vt.vertical_term(b, bx, by, b_k1, W, DRF, g, xg)
    nanify = lambda f: f.where(~xr.DataArray(land, dims=('j', 'i')))  # noqa: E731
    bn, bk1n, Wn = nanify(b), nanify(b_k1), nanify(W)
    bxn, byn = op.grad_b(bn, g, xg)
    full = vt.vertical_term(bn, bxn, byn, bk1n, Wn, DRF, g, xg)
    from scipy import ndimage
    reach = ndimage.binary_dilation(land)                # the block plus one cell along each axis
    assert np.all(np.isnan(full.values[0][reach]))
    far = ~reach & np.isfinite(full0.values[0])
    np.testing.assert_array_equal(full.values[0][far], full0.values[0][far])
    assert np.isfinite(full.values[0][far]).all()
    # the surface term: NaN state/fluxes propagate the same way
    q = flux(300.0 + 200.0 * (x - x.mean()) / x.max(), 'oceQnet')
    sw = flux(np.full(x.shape, 100.0), 'oceQsw')
    fw = flux(np.zeros(x.shape), 'oceFWflx')
    t0 = vt.surface_flux_term(bx, by, q, sw, fw, T, S, DRF, g, xg)
    t = vt.surface_flux_term(bxn, byn, nanify(q), nanify(sw), nanify(fw), nanify(T), nanify(S), DRF, g, xg)
    assert np.all(np.isnan(t.values[0][reach]))
    np.testing.assert_array_equal(t.values[0][far], t0.values[0][far])


def test_dims_guards_before_any_dbof_call():
    g, xg, x, y, b, bx, by, W = fields()
    T, S = front_TS(x)
    U_like = bx.rename({'i': 'i_g'})                      # a staggered gradient by mistake
    with pytest.raises(ValueError, match='expected dims'):
        vt.vertical_term(b, U_like, by, b, W, DRF, g, xg)
    with pytest.raises(ValueError, match='expected dims'):
        vt.surface_flux_term(U_like, by, flux(np.zeros(x.shape), 'oceQnet'),
                             flux(np.zeros(x.shape), 'oceQsw'), flux(np.zeros(x.shape), 'oceFWflx'),
                             T, S, DRF, g, xg)
    with pytest.raises(TypeError):
        vt.vertical_term(b.values, bx, by, b, W, DRF, g, xg)
    with pytest.raises(ValueError, match='differ'):
        vt.vertical_term(b, bx, by, b, W.isel(j=slice(0, 10)), DRF, g, xg)
    with pytest.raises(ValueError, match='k dim'):
        vt.buoyancy_levels(T, S)                          # no k dim


def test_rotation_invariance_on_face_10_orientation():
    """The terms are dot products of two gradients (rotational invariants):
    the same index-space fields on the rotated (CS = 0, SN = -1) grid give
    the same values."""
    out = {}
    for rot in (False, True):
        g, xg, x, y, b, bx, by, W = fields(rot)
        T, S = front_TS(x)
        bz = sy.da(-2e-4 - 1e-4 * (x - x.mean()) / x.max(), C)
        out[rot] = (vt.vertical_term(b, bx, by, b - bz * DZ, W, DRF, g, xg).values,
                    vt.surface_flux_term(bx, by, flux(300 + 200 * (x - x.mean()) / x.max(), 'oceQnet'),
                                         flux(np.full(x.shape, 100.0), 'oceQsw'),
                                         flux(np.zeros(x.shape), 'oceFWflx'), T, S, DRF, g, xg).values)
    for a, r in zip(out[False], out[True]):
        fin = np.isfinite(a) & np.isfinite(r)
        np.testing.assert_allclose(a[fin], r[fin], rtol=1e-10, atol=1e-13 * np.nanmax(np.abs(a)))


def test_f_units_attrs_and_forcing_note():
    g, xg, x, y, b, bx, by, W = fields()
    T, S = front_TS(x)
    v = vt.vertical_term(b, bx, by, b, W, DRF, g, xg, L_cells=2)
    s = vt.surface_flux_term(bx, by, flux(np.full(x.shape, 300.0), 'oceQnet', note='TRIANGLE'),
                             flux(np.full(x.shape, 100.0), 'oceQsw', note='TRIANGLE'),
                             flux(np.zeros(x.shape), 'oceFWflx', note='TRIANGLE'), T, S, DRF, g, xg)
    for t in (v, s):
        assert t.attrs['units'] == 's-5' and 'F units' in t.attrs['convention'] and '2 x' in t.attrs['convention']
    assert v.attrs['L_cells'] == 2 and 'W(k_l=1)' in v.attrs['W_source']
    assert s.attrs['forcing_note'] == 'TRIANGLE' and s.attrs['rhoConst'] == 1027.5
    assert s.attrs['c_p'] == 3994.0 and s.attrs['convertFW2Salt'] == -1.0 and s.attrs['jwtype'] == 2
    assert 'not negated' in s.attrs['sign']


# ---------------------------------------------------------------------------
# the real stores: hour 0 (16 LST)
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_real_stores_hour0_terms_have_the_expected_shape_and_support():
    """Hour-0 pair (07-02 00-01 UTC, 16 LST) at L = 0: b_z, the vertical
    term (full and factorised) and the surface-flux term are finite on all
    of mask_analysis (n_lost = 0), dims (face, j, i); b_z is negative in the
    median (an afternoon warm layer, code b); the terms are not zero and
    not larger than 2F in rms on the analysis mask; the smoke numbers are
    printed (the log carries them -- prompt 4 task 2)."""
    for p in (inp.RAW_ZARR, inp.CHUNK_ZARR, inp.GRID_ZARR, inp.MASKS_NC):
        if not p.exists():
            pytest.skip(f'{p} not on disk')
    ds, g, grid_, masks = inp.open_inputs()
    h0, h1 = inp.hour_pair(ds, 0)
    drF = inp.drF(ds)
    b0, U0, V0 = inp.filtered(h0, 0)
    b1, U1, V1 = inp.filtered(h1, 0)
    b_mid, U_mid, V_mid = (inp.midpoint(a, c) for a, c in ((b0, b1), (U0, U1), (V0, V1)))
    bx, by = op.grad_b(b_mid, g, grid_)
    two_F = 2.0 * op.frontogenesis(b_mid, U_mid, V_mid, g, grid_)
    lev = inp.midpoint(vt.buoyancy_levels(h0['Theta_k'], h0['Salt_k']),
                       vt.buoyancy_levels(h1['Theta_k'], h1['Salt_k']))
    np.testing.assert_array_equal(lev.isel(k=0).values, b_mid.values)        # the k = 0 identity in b
    bz = inp.midpoint(vt.b_z(h0['Theta_k'], h0['Salt_k'], g, drF), vt.b_z(h1['Theta_k'], h1['Salt_k'], g, drF))
    W = inp.midpoint(inp.W_k1(h0), inp.W_k1(h1))
    vert = vt.vertical_term(b_mid, bx, by, lev.isel(k=1, drop=True), W, drF, g, grid_)
    fac = vt.vertical_term_factorised(bx, by, bz, W, g, grid_)
    fl = [inp.midpoint(a, c) for a, c in zip(inp.fluxes(h0), inp.fluxes(h1))]
    T_mid, S_mid = inp.midpoint(h0['Theta'], h1['Theta']), inp.midpoint(h0['Salt'], h1['Salt'])
    sfc = vt.surface_flux_term(bx, by, *fl, T_mid, S_mid, drF, g, grid_)
    for t in (bz, vert, fac, sfc):
        assert t.dims == ('face', 'j', 'i')
    val, n_lost = inp.valid(masks, b_mid, two_F, vert, fac, sfc, bz)
    assert int(val.sum()) == 262_925 and n_lost == 0
    ana = masks['mask_analysis'].values
    assert np.nanmedian(bz.values[0][ana]) < 0                                 # warm layer at 16 LST
    rms = lambda a: float(np.sqrt(np.mean(a[val] ** 2)))                        # noqa: E731
    r_v, r_s = rms(2 * vert.values[0]) / rms(two_F.values[0]), rms(2 * sfc.values[0]) / rms(two_F.values[0])
    assert 0 < r_v < 1 and 0 < r_s < 1
    assert sfc.attrs['forcing_note'].startswith('the surface fluxes are 6-hourly')
    print(f'\nhour 0 (16 LST), L = 0: rms(2 vert)/rms(2F) {r_v:.3f}, rms(2 sfc)/rms(2F) {r_s:.3f}, '
          f'factorised/full rms {rms(fac.values[0]) / rms(vert.values[0]):.2f}, '
          f'b_z median {np.nanmedian(bz.values[0][ana]):.2e} s^-2, f_sw {sfc.attrs["f_sw"]:.3f}')
