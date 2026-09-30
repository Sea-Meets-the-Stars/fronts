""" Synthetic C-grids and analytic test cases for the offline gates and
tests (coding doc §4.9, §5).  No physics choice lives here: these are the
grids and exact solutions that ``validate.py`` and ``py/tests`` measure the
operators against.

* :func:`synthetic_cgrid` -- a uniform C-grid with comodo attrs and its
  xgcm grid, in the unrotated (``CS = 1``) or the face-10 (``CS = 0,
  SN = -1``) orientation, optionally with land.
* :func:`deformation_case` -- the pure deformation ``u = -a x, v = a y``
  with the front ``b = b0 tanh(x/ell)`` and its exact solution
  ``b(x, t) = b0 tanh(x e^{at}/ell)``, for V1 and V3 (``velocities='strain'``).
* :func:`uniform_shift_bias` -- a uniform, zero-strain flow moving an
  ``erf`` front by a prescribed number of cells, the true ``DG/Dt = 0``, for
  V4 (and the same front, shifted half a cell, for V5).

Conventions: positional ``(face, j, i)`` float64 DataArrays; model
``x_M = i dx`` (axis X), ``y_M = j dy`` (axis Y); ``U`` at ``x_M - dx/2``
(``i_g``, comodo shift -0.5), ``V`` at ``y_M - dy/2`` (``j_g``).
"""

import numpy as np
import xarray as xr
from scipy import special

import osn_tiles as ot
import operators as op
import semilag as sl
from dbof.llc4320_ingestion.grid import ensure_comodo_attrs

C_DIMS, U_DIMS, V_DIMS = ('face', 'j', 'i'), ('face', 'j', 'i_g'), ('face', 'j_g', 'i')
DT = 3600.0


def da(arr, dims):
    return xr.DataArray(np.asarray(arr, dtype='float64'), dims=dims)


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
def synthetic_cgrid(nj=48, ni=64, dx=1800.0, dy=2000.0, rotated=False, land=None):
    """Uniform C-grid on ``(face, j, i)`` with comodo attrs and an xgcm grid.

    Model coordinates: ``x_M = i dx`` (axis X, dim ``i``), ``y_M = j dy``
    (axis Y, dim ``j``); ``U`` sits at ``x_M - dx/2`` (``i_g``, comodo
    shift -0.5), ``V`` at ``y_M - dy/2`` (``j_g``).  ``dx != dy`` by default
    so a swapped metric shows.  ``rotated=True`` is face 10's orientation
    (model x south, model y east).  ``land`` is an optional ``(nj, ni)``
    bool array setting ``hFacC = 0``.  Returns ``(grid_ds, grid, pos)``
    where ``pos(kind)`` gives the geographic ``(x_east, y_north)`` of the
    'c', 'u' or 'v' points as ``(1, nj, ni)`` arrays.
    """
    CS, SN = (0.0, -1.0) if rotated else (1.0, 0.0)
    one = np.ones((1, nj, ni))
    hfacc = one.copy()
    if land is not None:
        hfacc[0][land] = 0.0
    g = xr.Dataset(
        {'dxC': (('face', 'j', 'i_g'), one * dx), 'dyC': (('face', 'j_g', 'i'), one * dy),
         'dxG': (('face', 'j_g', 'i'), one * dx), 'dyG': (('face', 'j', 'i_g'), one * dy),
         'rA': (('face', 'j', 'i'), one * dx * dy), 'rAz': (('face', 'j_g', 'i_g'), one * dx * dy),
         'CS': (('face', 'j', 'i'), one * CS), 'SN': (('face', 'j', 'i'), one * SN),
         'hFacC': (('face', 'j', 'i'), hfacc)},
        coords={'j': np.arange(nj), 'i': np.arange(ni),
                'j_g': np.arange(nj), 'i_g': np.arange(ni)})
    g = ensure_comodo_attrs(g)
    grid = ot.build_xgcm(g)
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing='ij')

    def pos(kind):
        x_m = ii * dx - (dx / 2 if kind == 'u' else 0.0)
        y_m = jj * dy - (dy / 2 if kind == 'v' else 0.0)
        # r = x_M e_xM + y_M e_yM with e_xM = (CS, SN), e_yM = (-SN, CS)
        return (x_m * CS - y_m * SN)[None], (x_m * SN + y_m * CS)[None]
    return g, grid, pos


def model_components(u_east, v_north, rotated):
    """Geographic -> model components (the inverse of the CS/SN rotation):
    on face 10 ``U = -v_north``, ``V = u_east``."""
    CS, SN = (0.0, -1.0) if rotated else (1.0, 0.0)
    return u_east * CS + v_north * SN, -u_east * SN + v_north * CS


def synthetic_uniform_grid(nj=32, ni=96, dx=1800.0, dy=1800.0):
    """A uniform, unrotated (``CS = 1``) C-grid, as :func:`synthetic_cgrid`
    but returning ``(grid_ds, grid, x, y)`` with ``x``, ``y`` the
    ``(1, nj, ni)`` centre positions (the V5 convenience)."""
    g, grid, pos = synthetic_cgrid(nj=nj, ni=ni, dx=dx, dy=dy, rotated=False)
    x, y = pos('c')
    return g, grid, x, y


def inner(shape, m):
    """Mask that drops ``m`` cells on every edge of the last two axes (the
    finite tile-edge rim plus the interpolation reach)."""
    out = np.zeros(shape, bool)
    out[..., m:-m, m:-m] = True
    return out


# ---------------------------------------------------------------------------
# the fronts
# ---------------------------------------------------------------------------
def erf_front(x, x0, sigma_G_cells, dx, b0=1e-2):
    """``b = b0 erf((x - x0)/(sqrt 2 sigma_b))`` with ``sigma_b = sqrt 2
    sigma_G``: ``G = b_x^2`` is a Gaussian of standard deviation ``sigma_G``
    cells, and bilinear interpolation at a half-cell offset errs by
    ``dx^2 G_xx / 8 = -G / (8 sigma_G^2)`` at its maximum -- ``-5.56%`` for
    the ``sigma_G = 1.5`` front of planning §5.3."""
    sb = np.sqrt(2.0) * sigma_G_cells * dx
    return b0 * special.erf((x - x0) / (np.sqrt(2.0) * sb))


def deformation_fields(g, pos, rotated, a=1e-5, b0=1e-2, ell=None):
    """Pure deformation ``u = -a x, v = a y`` and a front ``b = b0 tanh(x/ell)``
    across the compressional (x) axis, on the synthetic grid: ``(b, U, V)``
    with ``U``, ``V`` the raw staggered model components."""
    ell = 4 * 1800.0 if ell is None else ell
    xc, yc = pos('c')
    xu, yu = pos('u')
    xv, yv = pos('v')
    xc0 = xc.mean()
    b = da(b0 * np.tanh((xc - xc0) / ell), C_DIMS)
    U, _ = model_components(-a * (xu - xc0), a * yu, rotated)
    _, V = model_components(-a * (xv - xc0), a * yv, rotated)
    return b, da(U, U_DIMS), da(V, V_DIMS)


# ---------------------------------------------------------------------------
# V1 / V3: pure deformation with the exact solution
# ---------------------------------------------------------------------------
def deformation_case(ell_cells=8.0, a=1e-5, rotated=False, n=128, dx=1800.0, b0=1e-2):
    """The pure deformation ``u = -a x, v = a y`` (``x`` = geographic east)
    on an ``n x n`` grid of spacing ``dx`` (square, so ``ell_cells`` means
    the same in either orientation), with the front ``b = b0 tanh(x/ell)``
    centred on the domain, ``ell = ell_cells dx``.

    ``b`` is conserved, so the exact solution is ``b(x, t) = b0 tanh(x
    e^{at}/ell)`` and along a parcel ``x(t) = x0 e^{-at}`` the continuum
    front strength is ``G = (b0/ell)^2 e^{2at} sech^4(x0/ell)``: ``G`` grows
    exactly as ``exp(2 a t)`` (criterion 1).  Returns a dict with the grid
    (``g``, ``grid``, ``pos``), the steady staggered ``U``, ``V``, the
    centre positions ``xc`` (geographic east) and ``xc0``, and
    ``b_exact(t)`` (a ``(face, j, i)`` DataArray)."""
    g, grid, pos = synthetic_cgrid(nj=n, ni=n, dx=dx, dy=dx, rotated=rotated)
    xc, yc = pos('c')
    xc0 = xc.mean()
    ell = ell_cells * dx

    def b_exact(t):
        return da(b0 * np.tanh((xc - xc0) * np.exp(a * t) / ell), C_DIMS)

    _, U, V = deformation_fields(g, pos, rotated, a=a, b0=b0, ell=ell)
    return dict(g=g, grid=grid, pos=pos, xc=xc, yc=yc, xc0=xc0, ell=ell, ell_cells=ell_cells,
                a=a, b0=b0, dx=dx, n=n, rotated=rotated, U=U, V=V, b_exact=b_exact)


def deformation_step(case, t, dt=DT, order=3):
    """One semi-Lagrangian step of the deformation case from ``t`` to
    ``t + dt`` with the machinery of ``semilag``: the departure ``(di, dj)``
    from the steady velocity, ``G_d = G(x_d, t)`` by ``gradb2_at_departure``
    on the exact ``b(t)``, and ``G_1 = G(x, t + dt)`` by ``operators.gradb2``
    on the exact ``b(t + dt)``.  Also returns ``G_d_ref``: the *same*
    centred stencil applied to the analytic ``b(t)`` at the *exact*
    departure point ``x_d = x0 + (x - x0) e^{a dt}`` -- so ``G_d / G_d_ref``
    isolates the semi-Lagrangian step (departure + interpolation) from the
    stencil's own truncation, which ``G_1 / G_d`` vs ``exp(2 a dt)`` carries.
    """
    g, grid, a, dx = case['g'], case['grid'], case['a'], case['dx']
    u_c, v_c = sl.centre_velocities(case['U'], case['V'], g, grid)
    di, dj = sl.departure_index(u_c, v_c, g, dt=dt)
    G_d = sl.gradb2_at_departure(case['b_exact'](t), di, dj, g, order)
    G_1 = op.gradb2(case['b_exact'](t + dt), g, grid)
    # the exact departure point along x (geographic east); b depends on x only
    xd = case['xc0'] + (case['xc'] - case['xc0']) * np.exp(a * dt)

    def b_t(x):
        return case['b0'] * np.tanh((x - case['xc0']) * np.exp(a * t) / case['ell'])
    G_d_ref = ((b_t(xd + dx) - b_t(xd - dx)) / (2 * dx)) ** 2
    return dict(di=di, dj=dj, G_d=G_d, G_1=G_1, G_d_ref=G_d_ref)


def deformation_series(case, n_steps=8, dt=DT, order=3):
    """``G`` along the parcels arriving at every cell at ``t_N = n_steps dt``,
    followed *backwards* through ``n_steps`` semi-Lagrangian steps: at each
    step the displacement field of ``departure_index`` is interpolated
    (bilinearly, as the velocity is in the midpoint iteration) at the
    current trajectory position and accumulated, and ``G`` at the earlier
    time is ``gradb2_at_departure`` on the exact ``b`` with the accumulated
    displacement.  Returns ``(t, G)`` with ``G`` of shape ``(n_steps + 1,
    nj, ni)``, ``G[-1]`` being ``operators.gradb2`` at the arrival cells;
    along a parcel ``G[n] / G[0]`` should be ``exp(2 a t[n])``."""
    g, grid = case['g'], case['grid']
    u_c, v_c = sl.centre_velocities(case['U'], case['V'], g, grid)
    di, dj = sl.departure_index(u_c, v_c, g, dt=dt)
    di, dj = di.values[0], dj.values[0]
    Di, Dj = np.zeros_like(di), np.zeros_like(dj)
    t = np.arange(n_steps + 1) * dt
    G = [op.gradb2(case['b_exact'](t[-1]), g, grid).values[0]]
    for k in range(n_steps, 0, -1):
        # the step from t[k-1] to t[k] started where the trajectory now is
        d_i = sl.interp_to_departure(di, Di, Dj, 1, allow_low_order=True)
        d_j = sl.interp_to_departure(dj, Di, Dj, 1, allow_low_order=True)
        Di, Dj = Di + d_i, Dj + d_j
        G.insert(0, sl.gradb2_at_departure(case['b_exact'](t[k - 1]), Di, Dj, g, order).values[0])
    return t, np.array(G)


# ---------------------------------------------------------------------------
# V3 (strain variant): a front of prescribed width in a prescribed strain
# ---------------------------------------------------------------------------
# the "realistic mix": deformation plus sinusoidal shear (vorticity + shear
# strain) and divergence modes at 36-48 dx, amplitudes 0.15-0.25 m/s, so the
# strain varies along each front (|sigma| ~ 1-3e-5 s^-1, the tile's median
# 1.9e-5) and the hourly displacement adds ~1 cell to the deformation's
NULL_MODES = dict(U1=0.25, L1_cells=40.0, U3=0.15, L3_cells=36.0, V2=0.20, L2_cells=48.0)


def null_strain_case(ell_cells=4.0, theta_deg=0.0, a=1e-5, n=128, dx=1800.0, b0=1e-2,
                     rotated=False, modes=NULL_MODES):
    """One V3 'strain' case: the tanh front ``b = b0 tanh(s/ell)`` with
    ``s = (x - x0) cos theta + (y - y0) sin theta`` (its normal at
    ``theta_deg`` from geographic east, the compressional axis of the
    deformation ``u = -a x, v = a y``: ``F = a G cos 2 theta`` for the
    deformation alone, so 0 / 30 / 60 deg give ``+aG / +aG/2 / -aG/2``),
    on an ``n x n`` grid of spacing ``dx``, with the sinusoidal modes of
    ``modes`` added to the velocity (``u += U1 sin(2 pi y/L1) + U3 sin(2 pi
    x/L3)``, ``v += V2 sin(2 pi y/L2)``: shear, vorticity and divergence
    that vary in space).  Returns a dict with the grid, the staggered
    ``U, V`` (model components), ``b_t`` and the exact velocity-gradient
    tensor at the centres (``u_x, u_y, v_x, v_y``, geographic)."""
    g, grid, pos = synthetic_cgrid(nj=n, ni=n, dx=dx, dy=dx, rotated=rotated)
    xc, yc = pos('c')
    x0, y0 = xc.mean(), yc.mean()
    ell = ell_cells * dx
    th = np.deg2rad(theta_deg)
    s = (xc - x0) * np.cos(th) + (yc - y0) * np.sin(th)
    b_t = da(b0 * np.tanh(s / ell), C_DIMS)
    m = dict(NULL_MODES, **({} if modes is None else modes))
    k1, k2, k3 = (2 * np.pi / (m[f'L{k}_cells'] * dx) for k in (1, 2, 3))

    def u_e(x, y):
        return -a * (x - x0) + m['U1'] * np.sin(k1 * (y - y0)) + m['U3'] * np.sin(k3 * (x - x0))

    def v_n(x, y):
        return a * (y - y0) + m['V2'] * np.sin(k2 * (y - y0))
    xu, yu = pos('u')
    xv, yv = pos('v')
    U, _ = model_components(u_e(xu, yu), v_n(xu, yu), rotated)
    _, V = model_components(u_e(xv, yv), v_n(xv, yv), rotated)
    exact = dict(u_x=-a + m['U3'] * k3 * np.cos(k3 * (xc - x0)),
                 u_y=m['U1'] * k1 * np.cos(k1 * (yc - y0)),
                 v_x=np.zeros_like(xc), v_y=a + m['V2'] * k2 * np.cos(k2 * (yc - y0)))
    return dict(g=g, grid=grid, pos=pos, xc=xc, yc=yc, b_t=b_t, U=da(U, U_DIMS), V=da(V, V_DIMS),
                ell=ell, ell_cells=float(ell_cells), theta_deg=float(theta_deg), a=a, b0=b0,
                dx=dx, n=n, rotated=rotated, exact=exact)


# ---------------------------------------------------------------------------
# V2: an analytic function of (lon, lat) on the sphere, and the metric's radius
# ---------------------------------------------------------------------------
def wave(X, Y, lon0, lat0, L_lon, L_lat, R):
    """``f = sin(kx (lon - lon0)) cos(ky (lat - lat0))`` on a sphere of radius
    ``R`` and its exact geographic gradient per metre: ``d/dx = (180/pi) /
    (R cos lat) d/dlon``, ``d/dy = (180/pi) / R d/dlat`` (``X``, ``Y``, ``L_*``
    in degrees).  Returns ``(f, f_east, f_north)``."""
    kx, ky = 2 * np.pi / L_lon, 2 * np.pi / L_lat
    f = np.sin(kx * (X - lon0)) * np.cos(ky * (Y - lat0))
    f_lon = kx * np.cos(kx * (X - lon0)) * np.cos(ky * (Y - lat0))
    f_lat = -ky * np.sin(kx * (X - lon0)) * np.sin(ky * (Y - lat0))
    deg = 180.0 / np.pi
    return f, f_lon * deg / (R * np.cos(np.deg2rad(Y))), f_lat * deg / R


def sphere_radius(X, Y, dxC, dyC):
    """The radius the grid metric implies: ``dxC`` over the angular
    centre-to-centre distance along ``i`` and ``dyC`` over the one along
    ``j`` (haversine on the unit sphere); the medians ``(R_i, R_j)`` in m."""
    def ang(lon1, lat1, lon2, lat2):
        p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
        h = (np.sin(0.5 * (p2 - p1)) ** 2
             + np.cos(p1) * np.cos(p2) * np.sin(0.5 * np.deg2rad(lon2 - lon1)) ** 2)
        return 2 * np.arcsin(np.sqrt(h))
    R_i = dxC[:, 1:] / ang(X[:, :-1], Y[:, :-1], X[:, 1:], Y[:, 1:])
    R_j = dyC[1:, :] / ang(X[:-1, :], Y[:-1, :], X[1:, :], Y[1:, :])
    return float(np.nanmedian(R_i)), float(np.nanmedian(R_j))


# ---------------------------------------------------------------------------
# V4 / V5: a uniform, zero-strain flow
# ---------------------------------------------------------------------------
# the real-hour displacement distribution (M1 task 3: ocean cells, hour 0)
REAL_HOUR_CELLS = dict(median=0.364, p99=1.252, max=2.09)


def real_hour_fractions(real_hour=REAL_HOUR_CELLS, n=4000, seed=0):
    """Samples of the sub-cell cross-front displacement implied by the
    real-hour distribution: ``|d|`` lognormal with the given median and
    p99, capped at ``max``; direction isotropic.  Returns ``(frac_isotropic,
    frac_all_cross_front)`` -- the fractional part of the cross-front
    component, which is all the V4 bias depends on (an integer shift is
    exact).  The real cross-front share of the displacement is unknown here
    (fronts align with jets), so the isotropic value is the estimate and
    the all-cross-front one the upper bound."""
    rng = np.random.default_rng(seed)
    sig = np.log(real_hour['p99'] / real_hour['median']) / 2.3263
    d = np.minimum(np.exp(np.log(real_hour['median']) + sig * rng.standard_normal(n)),
                   real_hour['max'])
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.abs(d * np.cos(phi)) % 1.0, d % 1.0



def uniform_shift_bias(sigma_G=1.5, di=0.5, dj=0.0, order=3, tilt_deg=0.0, nj=64, ni=96,
                       dx=1800.0, dt=DT, margin=8):
    """``measured_DGDt`` under a uniform flow that moves an ``erf`` front
    (``G`` Gaussian of ``sigma_G`` cells, tilted ``tilt_deg`` from the
    ``j`` axis) by exactly ``(di, dj)`` cells per ``dt``, where the true
    ``DG/Dt`` is identically zero: ``b_tp1(x) = b_t(x - u dt)`` exactly.
    Whatever comes out is the interpolation bias of ``G(x_d, t)`` (planning
    §5.3, V4).  Returns a dict with ``rel = DGDt dt / G_tp1`` (the bias per
    hour as a fraction of ``G``; NaN outside ``margin`` cells of the edge),
    the truth ``G`` (``operators.gradb2`` of the exactly shifted front) and
    ``front`` (cells with ``G >= 0.2 max``, within ~1.8 sigma_G of the
    centre)."""
    g, grid, pos = synthetic_cgrid(nj=nj, ni=ni, dx=dx, dy=dx, rotated=False)
    x, y = pos('c')
    th = np.deg2rad(tilt_deg)
    # the across-front coordinate; the centre off a node so a zero shift is not special
    s = (x - (ni / 2 + 0.3) * dx) * np.cos(th) + (y - (nj / 2) * dx) * np.sin(th)
    b_t = da(erf_front(s, 0.0, sigma_G, dx), C_DIMS)
    shift = (di * np.cos(th) + dj * np.sin(th)) * dx          # across-front displacement
    b_tp1 = da(erf_front(s - shift, 0.0, sigma_G, dx), C_DIMS)
    U = da(np.full((1, nj, ni), di * dx / dt), U_DIMS)
    V = da(np.full((1, nj, ni), dj * dx / dt), V_DIMS)
    DG = sl.measured_DGDt(b_t, b_tp1, U, V, g, grid, dt=dt, order=order,
                          allow_low_order=True).values[0]
    G = op.gradb2(b_tp1, g, grid).values[0]
    ok = inner(G.shape, margin) & np.isfinite(DG) & (G > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.where(ok, DG * dt / G, np.nan)
    front = ok & (G >= 0.2 * np.nanmax(np.where(ok, G, np.nan)))
    return dict(rel=rel, G=G, front=front, s_cells=s[0] / dx - shift / dx, ok=ok)
