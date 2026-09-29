""" Tests for ``operators.py`` (coding doc §5, prompt 2 task 2).

Synthetic C-grid tests run offline, in both the unrotated (``CS = 1``) and
the face-10 (``CS = 0, SN = -1``: model x south, model y east) orientation.
Tests on the real tile are marked ``needs_grid`` and use the M0 stores and
``tile330_masks.nc`` from ``conftest.py``.

Guards: the gradient of an analytic field; the factor-of-two convention
``F = (1/2) DG/Dt`` on a pure-deformation front; the dims guard (a
mis-staggered or argument-swapped call raises *before* dbof can broadcast
to 4-D); ``lowpass`` (identity, constants, the requested scale, NaN
propagation without renormalisation, commutation with the gradient, and
the halo consequence); the strain rotation and the alignment angle; and the
criterion-7 regression against ``calculate_fields.frontogenesis_tendency``
plus the 0.911x ``gradb2`` / ``grad_b2`` ratio.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

import operators as op
import masking as mk
import osn_tiles as ot
from dbof.utils import native_gradient as ng
from dbof.preprocessing import calculate_fields as cf


# ---------------------------------------------------------------------------
# the synthetic C-grid with analytic positions lives in synthetic.py (M1
# task 5); re-exported here because test_semilag / test_coarsegrain import it
# ---------------------------------------------------------------------------
from synthetic import (synthetic_cgrid, model_components, da, deformation_fields,  # noqa: E402,F401
                       C_DIMS, U_DIMS, V_DIMS)


def interior(a, m=3):
    """Drop ``m`` cells on every edge (the finite tile-edge rim)."""
    return a[0, m:-m, m:-m]


# ---------------------------------------------------------------------------
# gradient of an analytic field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
def test_grad_b_analytic_field(rotated):
    """``grad_b`` reproduces the gradient of ``sin(kx x) cos(ky y)`` to the
    centred-difference truncation ``(k dx)^2 / 6`` (< 1% at 40 cells)."""
    g, grid, pos = synthetic_cgrid(rotated=rotated)
    x, y = pos('c')
    kx, ky = 2 * np.pi / (40 * 1800.0), 2 * np.pi / (30 * 2000.0)
    b = da(np.sin(kx * x) * np.cos(ky * y), C_DIMS)
    bx, by = op.grad_b(b, g, grid)
    assert bx.dims == C_DIMS and by.dims == C_DIMS
    bx_true, by_true = kx * np.cos(kx * x) * np.cos(ky * y), -ky * np.sin(kx * x) * np.sin(ky * y)
    for got, want, k in ((bx, bx_true, kx), (by, by_true, ky)):
        err = np.abs(interior(got.values) - interior(want)) / np.abs(want).max()
        assert np.isfinite(interior(got.values)).all()
        assert err.max() < 0.01
        # second-order: the error is the (k dx)^2/6 truncation, not larger
        assert err.max() < 2 * max(kx * 1800.0, ky * 2000.0) ** 2 / 6


@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
def test_gradb2_is_the_component_form(rotated):
    """``G = b_x^2 + b_y^2`` from the *same* ``grad_b`` -- and it is not
    the repo's ``calculate_grad_squared_tracer`` (a different stencil:
    interp-then-square attenuates a front)."""
    g, grid, pos = synthetic_cgrid(rotated=rotated)
    b, _, _ = deformation_fields(g, pos, rotated, ell=1.5 * 1800.0)
    bx, by = op.grad_b(b, g, grid)
    G = op.gradb2(b, g, grid)
    assert G.dims == C_DIMS and G.name == 'G'
    assert np.array_equal(G.values, (bx ** 2 + by ** 2).values)
    G_sq = ng.calculate_grad_squared_tracer(b, g, grid).compute().values
    Gi, Gsq = interior(G.values), interior(G_sq)
    front = Gi > 0.1 * np.nanmax(Gi)
    ratio = Gi[front] / Gsq[front]
    assert ratio.min() < 0.95                          # the stencils differ at the front
    assert np.all(ratio <= 1.0 + 1e-12)                # and the component form is the smaller


# ---------------------------------------------------------------------------
# Jacobian, factor of two, strain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
def test_jacobian_linear_flow_exact(rotated):
    """A linear velocity field is differenced exactly whatever the
    interpolation: ``u = -a x, v = a y`` gives ``(-a, 0, 0, a)``; the
    rotated grid checks the CS/SN handling (``U = -v_north``, ``V = u_east``)."""
    a = 1e-5
    g, grid, pos = synthetic_cgrid(rotated=rotated)
    _, U, V = deformation_fields(g, pos, rotated, a=a)
    ux, uy, vx, vy = op.jacobian(U, V, g, grid)
    for got, want in ((ux, -a), (uy, 0.0), (vx, 0.0), (vy, a)):
        assert got.dims == C_DIMS
        assert np.allclose(interior(got.values), want, atol=1e-12 * a, rtol=0)


@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
def test_factor_of_two_pure_deformation(rotated):
    """``F = (1/2) DG/Dt``, pinned on the exact solution.

    For ``u = -a x, v = a y`` and a conserved ``b = b0 tanh(x/ell)``, the
    exact solution is ``b(x, t) = b0 tanh(x e^{at} / ell)``, so along a
    parcel ``x(t) = x0 e^{-at}`` the front strength is
    ``G = (b0/ell)^2 e^{2at} sech^4(x0/ell)``: ``DG/Dt = 2 a G`` -- the
    factor of two (``G ~ exp(2 a t)``, criterion 1).  On the grid ``u_x = -a``
    exactly and ``G`` is built from the same ``b_x`` as ``F``, so
    ``F = a b_x^2 = a G`` to round-off, i.e. ``2F = DG/Dt``.
    """
    a, b0, ell = 1e-5, 1e-2, 4 * 1800.0
    g, grid, pos = synthetic_cgrid(rotated=rotated)
    b, U, V = deformation_fields(g, pos, rotated, a=a, b0=b0, ell=ell)
    F = op.frontogenesis(b, U, V, g, grid)
    G = op.gradb2(b, g, grid)
    assert F.dims == C_DIMS and F.name == 'F'
    Fi, Gi = interior(F.values), interior(G.values)
    assert np.allclose(Fi, a * Gi, rtol=1e-10, atol=0)         # discrete: F = a G exactly
    # the analytic material derivative along a parcel, from the exact solution
    x0 = np.linspace(-2 * ell, 2 * ell, 41)
    def G_exact(x, t):
        return (b0 / ell) ** 2 * np.exp(2 * a * t) / np.cosh(x * np.exp(a * t) / ell) ** 4
    dt = 1.0
    DGDt = (G_exact(x0 * np.exp(-a * dt), dt) - G_exact(x0, 0.0)) / dt
    assert np.allclose(DGDt, 2 * a * G_exact(x0, 0.0), rtol=1e-4)
    # so 2F is what the measured DG/Dt must be compared with:
    two_F = 2 * Fi
    assert np.allclose(two_F / Gi, 2 * a, rtol=1e-10, atol=0)
    assert np.allclose(Fi / Gi, a, rtol=1e-10, atol=0)
    assert not np.allclose(Fi / Gi, 2 * a, rtol=0.5, atol=0)    # F alone is off by 2x


@pytest.mark.parametrize('rotated', [False, True], ids=['CS=1', 'face10'])
def test_strain_divergence_rotated_to_geographic(rotated):
    """Flux-form strain of the pure deformation (``sigma_n = -2a``,
    ``sigma_s = 0``) and of a pure shear ``u = c y`` (``sigma_s = c``,
    ``sigma_n = 0``), in geographic components on both orientations: on
    face 10 the helper's model-basis pair is the negative of these."""
    a, c = 1e-5, 3e-5
    g, grid, pos = synthetic_cgrid(rotated=rotated)
    _, U, V = deformation_fields(g, pos, rotated, a=a)
    delta, sn, ss, mag = op.strain_divergence(U, V, g, grid)
    for got, want in ((delta, 0.0), (sn, -2 * a), (ss, 0.0), (mag, 2 * a)):
        assert got.dims == C_DIMS
        assert np.allclose(interior(got.values), want, atol=1e-12 * a, rtol=0)
    xu, yu = pos('u')
    xv, yv = pos('v')
    U2, _ = model_components(c * yu, 0 * xu, rotated)
    _, V2 = model_components(c * yv, 0 * xv, rotated)
    delta, sn, ss, mag = op.strain_divergence(da(U2, U_DIMS), da(V2, V_DIMS), g, grid)
    for got, want in ((delta, 0.0), (sn, 0.0), (ss, c), (mag, c)):
        assert np.allclose(interior(got.values), want, atol=1e-12 * c, rtol=0)
    # the raw helper's pair is model-basis: on face 10 it is the negative
    sv = ng.calculate_native_strain_vorticity(U, V, g, grid)
    sn_model = interior(sv['strain_normal_center'].values)
    assert np.allclose(sn_model, (2 * a) if rotated else (-2 * a), atol=1e-12 * a)


def test_strain_alignment_definition_and_decomposition():
    """``theta`` is the angle from the *compressional* axis: 0 for a front
    across the compressional axis, ``pi/2`` along it; and with the
    Jacobian-derived strain ``F = -(1/2) delta G + (1/2) |sigma| G cos 2theta``
    holds to round-off (the **plus** sign; planning §2.4's minus would
    need ``theta`` from the extensional axis)."""
    a = 1e-5
    g, grid, pos = synthetic_cgrid()
    b, U, V = deformation_fields(g, pos, False, a=a)
    bx, by = op.grad_b(b, g, grid)
    J = op.jacobian(U, V, g, grid)
    delta, sn, ss, mag = op.strain_from_jacobian(*J)
    th = op.strain_alignment(bx, by, sn, ss)
    assert th.dims == C_DIMS
    assert np.allclose(interior(th.values), 0.0, atol=1e-6)     # gradient along x = compression
    x, y = pos('c')
    b_y = da(1e-2 * np.tanh((y - y.mean()) / 7200.0), C_DIMS)         # front along the extension
    bxy, byy = op.grad_b(b_y, g, grid)
    th_y = op.strain_alignment(bxy, byy, sn, ss)
    assert np.allclose(interior(th_y.values), np.pi / 2, atol=1e-6)
    # exact decomposition on a generic smooth field and flow
    rng = np.random.default_rng(3)
    kx, ky = 2 * np.pi / (16 * 1800.0), 2 * np.pi / (12 * 2000.0)
    b2 = da(1e-2 * (np.sin(kx * x) * np.cos(ky * y) + 0.5 * np.cos(2 * kx * x + 1)), C_DIMS)
    xu, yu = pos('u')
    xv, yv = pos('v')
    U2 = da(0.3 * np.sin(ky * yu) + 0.1 * np.cos(kx * xu), U_DIMS)
    V2 = da(0.2 * np.cos(kx * xv) * np.sin(ky * yv), V_DIMS)
    F = op.frontogenesis(b2, U2, V2, g, grid)
    G = op.gradb2(b2, g, grid)
    bx2, by2 = op.grad_b(b2, g, grid)
    delta, sn, ss, mag = op.strain_from_jacobian(*op.jacobian(U2, V2, g, grid))
    th2 = op.strain_alignment(bx2, by2, sn, ss)
    F_dec = -0.5 * delta * G + 0.5 * mag * G * np.cos(2 * th2)
    Fi, Di = interior(F.values), interior(F_dec.values)
    assert np.max(np.abs(Fi - Di)) < 1e-12 * np.max(np.abs(Fi))
    assert ((interior(th2.values) >= 0) & (interior(th2.values) <= np.pi / 2)).all()
    F_minus = -0.5 * delta * G - 0.5 * mag * G * np.cos(2 * th2)     # planning §2.4's sign
    assert np.max(np.abs(Fi - interior(F_minus.values))) > 0.1 * np.max(np.abs(Fi))


# ---------------------------------------------------------------------------
# the dims guard
# ---------------------------------------------------------------------------
def test_dims_guard_raises_before_broadcasting():
    """A swapped or mis-staggered call must raise *before* the helper runs:
    on the tile the helper's 4-D+ broadcast is an OOM kill, not an error.
    Small arrays here so nothing can blow up either way."""
    g, grid, pos = synthetic_cgrid(nj=8, ni=8)
    rng = np.random.default_rng(0)
    b = da(rng.normal(size=(1, 8, 8)), C_DIMS)
    U = da(rng.normal(size=(1, 8, 8)), U_DIMS)
    V = da(rng.normal(size=(1, 8, 8)), V_DIMS)
    # the trap on record: a centred field times a staggered one is 4-D
    assert (b * U).ndim == 4
    with pytest.raises(AssertionError, match='broadcast'):
        op.assert_dims(b * U, C_DIMS, 'mask x U')
    # ours raise with a dims message, from the pre-call check
    with pytest.raises(ValueError, match='expected dims'):
        op.jacobian(V, U, g, grid)                     # swapped arguments
    with pytest.raises(ValueError, match='expected dims'):
        op.jacobian(U, U, g, grid)
    with pytest.raises(ValueError, match='expected dims'):
        op.grad_b(U, g, grid)                          # staggered field into the tracer stencil
    with pytest.raises(ValueError, match='expected dims'):
        op.gradb2(V, g, grid)
    with pytest.raises(ValueError, match='expected dims'):
        op.frontogenesis(U, U, V, g, grid)             # b on the U point
    with pytest.raises(ValueError, match='expected dims'):
        op.frontogenesis(b, V, U, g, grid)             # swapped U, V
    with pytest.raises(ValueError, match='expected dims'):
        op.strain_divergence(V, U, g, grid)
    with pytest.raises(ValueError, match='expected dims'):
        op.buoyancy(xr.Dataset({'Theta': U, 'Salt': U}))
    with pytest.raises(TypeError):
        op.grad_b(b.values, g, grid)                   # numpy is not a DataArray with dims
    # and the good call is fine
    F = op.frontogenesis(b, U, V, g, grid)
    assert F.dims == C_DIMS and F.shape == (1, 8, 8)


# ---------------------------------------------------------------------------
# lowpass
# ---------------------------------------------------------------------------
def test_lowpass_identity_constants_and_scale():
    rng = np.random.default_rng(1)
    nj, ni = 48, 64
    f = da(rng.normal(size=(1, nj, ni)), C_DIMS)
    assert op.lowpass(f, 0) is f                                        # L = 0: identity
    for bad in (3, -2, 2.5):
        with pytest.raises(ValueError):
            op.lowpass(f, bad)
    for L in (2, 4, 8):
        hw = L // 2
        # constants pass unchanged wherever the footprint is inside the domain
        c = op.lowpass(da(np.full((1, nj, ni), 3.5), C_DIMS), L)
        assert c.dims == C_DIMS and c.attrs['lowpass_L_cells'] == L
        assert np.allclose(c.values[0, hw:-hw, hw:-hw], 3.5, rtol=0, atol=1e-14)
        assert np.isnan(c.values[0, :hw]).all() and np.isnan(c.values[0, :, -hw:]).all()
        # the mean is preserved (normalised kernel)
        s = op.lowpass(f, L).values[0, hw:-hw, hw:-hw]
        assert abs(s.mean() - f.values[0].mean()) < 0.05 * f.values[0].std()
        # the requested scale: a wave of wavelength L + 1 cells is annihilated,
        # and a long wave passes with the box response sin(pi N/lam)/(N sin(pi/lam))
        ii = np.arange(ni)[None, None, :] + 0 * np.arange(nj)[None, :, None]
        wave = da(np.cos(2 * np.pi * ii / (L + 1)), C_DIMS)
        assert np.abs(op.lowpass(wave, L).values[0, hw:-hw, hw:-hw]).max() < 1e-12
        lam = 48.0
        longw = da(np.cos(2 * np.pi * ii / lam), C_DIMS)
        resp = np.sin(np.pi * (L + 1) / lam) / ((L + 1) * np.sin(np.pi / lam))
        got = op.lowpass(longw, L).values[0, hw:-hw, hw:-hw] / longw.values[0, hw:-hw, hw:-hw]
        ok = np.abs(longw.values[0, hw:-hw, hw:-hw]) > 0.5
        assert np.allclose(got[ok], resp, rtol=1e-10, atol=0)
        assert 0.9 < resp < 1.0                                # 0.944 at L = 8, 0.996 at L = 2
    # numpy in, numpy out (last two axes); staggered dims are filtered as such
    out = op.lowpass(f.values, 4)
    assert isinstance(out, np.ndarray) and out.shape == f.shape
    assert np.array_equal(out, op.lowpass(f, 4).values, equal_nan=True)
    u = da(rng.normal(size=(1, nj, ni)), U_DIMS)
    assert op.lowpass(u, 4).dims == U_DIMS
    assert np.array_equal(op.lowpass(u, 4).values, op.lowpass(u.values, 4), equal_nan=True)


def test_lowpass_nan_propagates_without_renormalising():
    """One NaN becomes an ``(L+1)^2`` NaN box; a coast makes every ocean
    cell within chessboard ``L/2`` of land NaN; and every other value is
    identical to the filtered NaN-free field (no renormalised partial
    stencils)."""
    rng = np.random.default_rng(2)
    nj, ni = 40, 50
    clean = rng.normal(size=(1, nj, ni))
    for L in (2, 8):
        hw = L // 2
        f = clean.copy()
        f[0, 20, 25] = np.nan
        out = op.lowpass(da(f, C_DIMS), L).values[0]
        ref = op.lowpass(da(clean, C_DIMS), L).values[0]
        box = np.zeros((nj, ni), bool)
        box[20 - hw:20 + hw + 1, 25 - hw:25 + hw + 1] = True
        inner = np.zeros((nj, ni), bool)
        inner[hw:nj - hw, hw:ni - hw] = True
        assert np.array_equal(np.isnan(out), box | ~inner)
        assert np.array_equal(out[~np.isnan(out)], ref[~np.isnan(out)])   # untouched elsewhere
        # a coast: land in the lower-left triangle (a diagonal coastline)
        jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')
        land = ii < jj - 10
        f = clean.copy()
        f[0][land] = np.nan
        out = op.lowpass(da(f, C_DIMS), L).values[0]
        chess = ndimage.distance_transform_cdt(~land, metric='chessboard')
        assert np.array_equal(np.isfinite(out), (chess > hw) & inner)
        assert np.array_equal(out[np.isfinite(out)], ref[np.isfinite(out)])


def test_lowpass_halo_consequence_diagonal_coast():
    """The 7-cell halo is Euclidean and sized with ``dxC`` (7.0 cells along
    ``i``, 6.45 along the wider ``j``), so on the tile's spacing it retains
    cells at chessboard 5 from land (M1 task 1, flag 1; here a diagonal
    coast plus an island on the real median spacing).  With NaN propagation
    those cells are simply NaN in ``F`` at ``L = 8`` (filter reach 4 +
    Jacobian reach 2 > 5) -- never a finite contaminated value -- so the
    halo does not need to grow; ``isfinite`` is the validity mask."""
    nj = ni = 60
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')
    land = (ii < jj - 20) | ((jj == 20) & (ii == 45))
    g, grid, pos = synthetic_cgrid(nj=nj, ni=ni, dx=1796.2, dy=1950.4, land=land)
    halo = mk.halo_mask(g, halo_cells=7)
    chess = ndimage.distance_transform_cdt(~land, metric='chessboard')
    assert chess[halo].min() == 5                      # the Euclidean halo's diagonal reach
    rng = np.random.default_rng(4)
    x, y = pos('c')
    b = da(1e-2 * np.sin(2 * np.pi * x / (20 * 1800.0)) + 1e-4 * rng.normal(size=x.shape), C_DIMS)
    b.values[0][land] = np.nan
    xu, yu = pos('u')
    xv, yv = pos('v')
    U = da(0.1 * np.sin(2 * np.pi * yu / (15 * 1800.0)), U_DIMS)
    V = da(0.1 * np.cos(2 * np.pi * xv / (15 * 1800.0)), V_DIMS)
    # U, V are NaN on every face touching land (hFacW/hFacS = 0, M0 task 3):
    # U[i_g = k] sits between cells k-1 and k, V[j_g = k] between rows k-1 and k
    U.values[0][land | np.roll(land, 1, axis=1)] = np.nan
    V.values[0][land | np.roll(land, 1, axis=0)] = np.nan
    for L, reach in ((0, 2), (2, 3), (4, 4), (8, 6)):
        F = op.frontogenesis(op.lowpass(b, L), op.lowpass(U, L), op.lowpass(V, L),
                             g, grid).values[0]
        fin = np.isfinite(F)
        inner = np.zeros((nj, ni), bool)
        m = L // 2 + 2
        inner[m:nj - m, m:ni - m] = True
        assert chess[fin & inner].min() == reach       # filter half-width + Jacobian reach
        assert fin[inner & (chess > reach + 1)].all()  # and finite beyond it
        n_halo_nan = int((halo & inner & ~fin).sum())
        if L == 8:
            assert n_halo_nan > 0                      # the chessboard-5 (and 6) halo cells
            assert not (halo & inner & ~fin & (chess >= 7)).any()
        else:
            assert n_halo_nan == 0


def test_lowpass_commutes_with_grad_b():
    """A shift-invariant normalised kernel commutes with the discrete
    gradient wherever no NaN is in reach: ``grad(lowpass b) == lowpass(grad b)``
    to round-off -- the property the coarse-grained budget rests on."""
    g, grid, pos = synthetic_cgrid(rotated=True)
    x, y = pos('c')
    rng = np.random.default_rng(5)
    b = da(np.sin(2 * np.pi * x / 20e3) * np.cos(2 * np.pi * y / 15e3)
           + 0.1 * rng.normal(size=x.shape), C_DIMS)
    for L in (2, 8):
        m = L // 2 + 2
        lhs = op.grad_b(op.lowpass(b, L), g, grid)
        rhs = [op.lowpass(c, L) for c in op.grad_b(b, g, grid)]
        for a1, a2 in zip(lhs, rhs):
            v1, v2 = a1.values[0, m:-m, m:-m], a2.values[0, m:-m, m:-m]
            assert np.isfinite(v1).all() and np.isfinite(v2).all()
            assert np.max(np.abs(v1 - v2)) < 1e-12 * np.max(np.abs(v1))


# ---------------------------------------------------------------------------
# real tile (needs_grid): the regression oracle and the 0.911x ratio
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def t0(grid_ds, raw_ds):
    """Hour 0 merged with the grid on ``(face, j, i)`` in float64, the xgcm
    grid, and ``b``."""
    ds = xr.merge([raw_ds.isel(time=0).expand_dims('face'), grid_ds],
                  compat='override', combine_attrs='override').astype('float64')
    grid = ot.build_xgcm(grid_ds)
    return ds, grid, op.buoyancy(ds)


@pytest.mark.needs_grid
def test_buoyancy_is_jmd95_sign_as_is(t0):
    ds, grid, b = t0
    assert b.dims == ('face', 'j', 'i') and b.dtype == np.float64
    ref = cf.buoyancy_of_field(ds).compute().values
    assert np.array_equal(b.values, ref, equal_nan=True)
    ok = np.isfinite(b.values)
    assert ok.sum() == 356_877                                   # land NaN, cell for cell
    assert (b.values[ok] > 0).all()                       # +g sigma0/rho0: positive, ~0.2-0.27
    assert 0.15 < np.median(b.values[ok]) < 0.30


@pytest.mark.needs_grid
def test_regression_vs_repo_frontogenesis_tendency(t0, masks_ds):
    """Criterion 7: unfiltered ``operators.frontogenesis`` against the repo's
    ``frontogenesis_tendency`` on M0's first hour, to round-off, over the
    valid analysis-mask cells.  (Measured: bit-for-bit, max |dF| = 0.)"""
    ds, grid, b = t0
    F = op.frontogenesis(b, ds.U, ds.V, ds, grid)
    F_repo = cf.frontogenesis_tendency(ds, grid).compute()
    assert F.dims == F_repo.dims == ('face', 'j', 'i')
    a, r = F.values[0], F_repo.values[0]
    ana = masks_ds['mask_analysis'].values
    assert np.array_equal(np.isfinite(a), np.isfinite(r))
    ok = ana & np.isfinite(a) & np.isfinite(r)
    assert ok.sum() == ana.sum() == 262_925               # F finite on the whole analysis mask
    d = np.abs(a[ok] - r[ok])
    scale = np.abs(r[ok]).max()
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.nanmax(d / np.abs(r[ok]))
    print(f'\nregression vs frontogenesis_tendency (t0, n={ok.sum()}): max|dF| {d.max():.3e}, '
          f'max|F| {scale:.3e}, max|dF|/max|F| {d.max() / scale:.3e}, max rel {rel:.3e}, '
          f'exact-equal {(d == 0).sum()}/{ok.sum()}')
    assert d.max() <= 1e-12 * scale
    assert rel <= 1e-9


@pytest.mark.needs_grid
def test_gradb2_ratio_to_repo_grad_b2_is_0911(t0, masks_ds):
    """``gradb2`` (component stencil) / the repo's ``grad_b2``
    (``calculate_grad_squared_tracer``): interior median 0.911 (M0 task 5)."""
    ds, grid, b = t0
    G = op.gradb2(b, ds, grid).values[0]
    G_repo = cf.grad_b2(ds, grid).compute().values[0]
    oc = masks_ds['mask_ocean'].values
    taxi = ndimage.distance_transform_cdt(oc, metric='taxicab')
    m0_set = oc & (taxi >= 3) & np.isfinite(G) & np.isfinite(G_repo)   # M0 task 5's cell set
    ratio_m0 = float(np.median(G[m0_set] / G_repo[m0_set]))
    ana = masks_ds['mask_analysis'].values
    ratio_ana = float(np.median((G / G_repo)[ana]))
    print(f'\nG_comp / G_sq median: M0 set {ratio_m0:.4f} (n={m0_set.sum()}), '
          f'analysis {ratio_ana:.4f}')
    assert abs(ratio_m0 - 0.911) < 0.002
    assert 0.90 < ratio_ana < 0.92


@pytest.mark.needs_grid
def test_strain_rotation_sign_on_the_real_face(t0, masks_ds):
    """On face 10 the flux-form strain pair must be rotated (a sign flip)
    to match the geographic Jacobian: slopes positive, ``delta`` 0.8-0.9x
    (the interpolated Jacobian's attenuation, M0 task 5: 0.80 on the
    all-ocean set), and ``sigma_s`` from the averaged corners equal to
    ``u_y + v_x`` to three figures."""
    ds, grid, b = t0
    ana = masks_ds['mask_analysis'].values
    J = op.jacobian(ds.U, ds.V, ds, grid)
    flux = op.strain_divergence(ds.U, ds.V, ds, grid)
    jac = op.strain_from_jacobian(*J)
    got = {}
    for name, x, y in zip(('delta', 'sigma_n', 'sigma_s', 'sigma_mag'), flux, jac):
        x, y = x.values[0], y.values[0]
        m = ana & np.isfinite(x) & np.isfinite(y)
        got[name] = (float(np.sum(x[m] * y[m]) / np.sum(x[m] ** 2)),
                     float(np.corrcoef(x[m], y[m])[0, 1]))
    print('\nJacobian vs flux-form (slope, corr):', got)
    assert 0.75 < got['delta'][0] < 0.92 and got['delta'][1] > 0.95
    assert 0.75 < got['sigma_n'][0] < 0.92 and got['sigma_n'][1] > 0.95
    assert abs(got['sigma_s'][0] - 1.0) < 0.01 and got['sigma_s'][1] > 0.999
    # the decomposition with the Jacobian strain closes to round-off
    bx, by = op.grad_b(b, ds, grid)
    G = op.gradb2(b, ds, grid)
    F = op.frontogenesis(b, ds.U, ds.V, ds, grid)
    th = op.strain_alignment(bx, by, jac[1], jac[2])
    F_dec = (-0.5 * jac[0] * G + 0.5 * jac[3] * G * np.cos(2 * th)).values[0]
    m = ana & np.isfinite(F_dec)
    assert np.max(np.abs(F_dec[m] - F.values[0][m])) < 1e-12 * np.max(np.abs(F.values[0][m]))


@pytest.mark.needs_grid
def test_lowpass_nan_at_the_coast_real_tile(t0, masks_ds):
    """NaN policy on the real coast: ``lowpass(b, L)`` is finite exactly
    where the chessboard distance to a NaN (land) exceeds ``L/2`` and the
    footprint is inside the tile; ``F`` at every ``L`` is finite on the whole
    analysis mask; the halo's chessboard-5 cells are NaN in ``F`` at L = 8
    (measured: 249 of 341,960), inside the offshore cut."""
    ds, grid, b = t0
    ana, halo, edge = (masks_ds[k].values for k in ('mask_analysis', 'mask_halo', 'mask_edge'))
    chess = ndimage.distance_transform_cdt(np.isfinite(b.values[0]), metric='chessboard')
    for L in (2, 4, 8):
        hw = L // 2
        bL = op.lowpass(b, L)
        inner = np.zeros(chess.shape, bool)
        inner[hw:-hw, hw:-hw] = True
        assert np.array_equal(np.isfinite(bL.values[0]), (chess > hw) & inner)
        F = op.frontogenesis(bL, op.lowpass(ds.U, L), op.lowpass(ds.V, L), ds, grid).values[0]
        assert np.isfinite(F[ana]).all()
        n_halo_nan = int((halo & edge & ~np.isfinite(F)).sum())
        print(f'\nL={L}: NaN F inside mask_halo & mask_edge: {n_halo_nan}; '
              f'min chessboard distance of a finite F: {chess[np.isfinite(F)].min()}')
        assert chess[np.isfinite(F)].min() == hw + 2                 # filter reach + Jacobian reach
        assert n_halo_nan < 0.001 * halo.sum()
