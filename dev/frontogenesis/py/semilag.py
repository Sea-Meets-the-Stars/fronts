""" The measured side of the budget (coding doc §4.4, planning §5.3): the
semi-Lagrangian ``D_h G / Dt`` over one hour, and the Eulerian estimate that
cross-checks it.

    DG/Dt  ~  [ G(x, t+dt) - G(x_d, t) ] / dt

with ``x_d`` the departure point of the parcel arriving at ``x``, found by
iterated midpoint from the time-centred velocity ``0.5 (u_t + u_tp1)``.

Rules (all of them load-bearing; see the tests)
-----------------------------------------------
* **Interpolate ``b``, never ``G``.**  ``G(x_d, t)`` is built by interpolating
  ``b_t`` onto the five-point tracer stencil centred at ``x_d`` (``x_d``,
  ``x_d +- e_i``, ``x_d +- e_j``, the displacement held fixed across the
  stencil) and applying the *same* diff / dxC / interp / rotate stencil as
  ``operators.grad_b`` (``calculate_native_gradient_tracer``), so ``G_d`` and
  ``G(t+dt)`` share one stencil and the discrete identity ``F = (1/2) DG/Dt``
  can hold.  Interpolating ``G`` bilinearly errs by ``dx^2 G_xx / 8``,
  negative at maxima -- it fabricates frontogenesis (V5).
  **Not** "shift the field, then differentiate": ``grad[b_t(x_d(x))]`` carries
  the strain of the departure map, ``(I - grad d)^T grad b``, and in the
  adiabatic limit equals ``grad b_{t+dt}`` exactly, so that construction
  measures ``DG/Dt - 2F`` (the residual), not ``DG/Dt``
  (``test_semilag.py::test_deformation_measures_DGDt_not_the_residual``).
* **``order >= 3``** for ``b`` (:func:`interp_to_departure`); lower orders
  raise unless the explicit test-only override ``allow_low_order=True`` is
  given (the order-1 bias test and V5 need it).
* **Departures in native index space**: ``di = U dt / dxC``, ``dj = V dt / dyC``
  from the raw model-x / model-y components interpolated to the centres.  No
  rotation.  Verified on the tile grid (M1 task 3): ``dxC[j, i_g = i]`` is the
  centre-to-centre distance along ``i`` (haversine, to 0.05%) and ``dyC`` the
  one along ``j``; the cross pairing would be 8-9% off (``dyC/dxC = 1.086``).
  On face 10 ``i`` runs south and ``j`` east, which is irrelevant here: ``U``
  is the component along ``i`` whatever ``i`` points at.
* **``F`` and front selection at the trajectory midpoint time**,
  :func:`midpoint_time`: ``0.5 (f_t + f_tp1)``.
* **NaN is honest.**  A departure whose interpolation support touches a NaN
  (land, or the staggered NaN faces of the coast in ``U``/``V``) or leaves
  the tile is NaN; nothing is filled, so a finite value is never contaminated
  (``test_nan_propagates_at_a_synthetic_coast``).  The last centre along each
  interpolated axis of ``centre_velocities`` is NaN too: the tile carries only
  the low staggered point and xgcm would silently pad the missing one with 0
  (M0 task 5).

Interpolation method (:func:`interp_to_departure`)
--------------------------------------------------
Tensor-product **Lagrange** interpolation of odd degree ``order`` on the
``order + 1`` nearest nodes per axis (2 x 2 bilinear, 4 x 4 cubic, 6 x 6
quintic), evaluated at ``(j - dj, i - di)``.  Chosen over
``scipy.ndimage.map_coordinates`` (B-splines) deliberately:

* a B-spline of order >= 2 needs a *prefilter* -- a recursive filter along
  each line whose response decays as ``0.268^k`` for the cubic.  A NaN must
  be filled before it, and the fill then leaks into the coefficients 1 / 2 /
  3 / 4 / 5 nodes away at 46% / 12% / 3.3% / 0.9% / 0.24% of the jump
  (measured, M1 task 3), so "the stencil touches NaN -> NaN" could not be
  made exact without masking ~6 cells from every NaN; ``prefilter=False`` would turn
  the spline into a smoothing approximation that attenuates fronts;
* the Lagrange kernel is local: the support is exactly the ``(order+1)^2``
  nodes, the NaN rule is exact, and the kernel reproduces polynomials of
  degree ``order`` (cubic: error ``O(dx^4 b'''')``).  Integer displacements
  are reproduced exactly (weights 1 and 0 in floating point), which is what
  makes the whole-cell translation test exact.  Since the five stencil
  points share one fractional offset, the same weights apply to all five, so
  "interpolate then difference" equals "difference then interpolate" to
  round-off away from NaN.

Measured on the ``sigma_G = 1.5``-cell front at a half-cell shift (relative
error of ``G`` at the maximum, discrete truth): order 1 **-4.8%** (the
``dx^2 G_xx/8`` bias; bilinear ``G`` gives -4.5%, analytic -5.4%, prediction
-5.6%), order 3 **-0.49%**, order 5 **-0.09%**.

Conventions: everything float64; ``(face, j, i)`` DataArrays in, the same
out (numpy arrays accepted by :func:`interp_to_departure`); ``di``/``dj`` are
the parcel's displacement in cells over ``dt`` (departure = arrival - d).
"""

import numpy as np
import xarray as xr

import operators as op
from operators import (require_centred, require_u_point, require_v_point,
                       assert_dims, X_STAG, Y_STAG)
from masking import _positional
from dbof.utils import native_gradient as ng

DT = 3600.0          # s; hourly snapshots (coding §1.2)
MIN_ORDER = 3        # coding §4.4: cubic+ REQUIRED for b


# ---------------------------------------------------------------------------
# midpoint time
# ---------------------------------------------------------------------------
def midpoint_time(f_t, f_tp1):
    """``0.5 (f_t + f_tp1)``: the field at the trajectory midpoint time
    (coding §1.2), for ``F``, ``G`` and front selection.  M2 strain rotates
    ~29 degrees per hour, so an endpoint would both add noise and correlate
    it with the measured side (planning §5.3, item 3)."""
    return 0.5 * (f_t + f_tp1)


# ---------------------------------------------------------------------------
# velocities at the centres
# ---------------------------------------------------------------------------
def centre_velocities(U, V, grid_ds, grid):
    """Raw model-x ``U (j, i_g)`` and model-y ``V (j_g, i)`` interpolated to
    the cell centres [m s^-1], **model basis** (no rotation: the departure
    is formed in index space).  The same two-point interpolation as the
    Jacobian's first step (``interp_pair_to_center``).

    The last centre along each interpolated axis is set to NaN: the tile
    holds the low staggered point of every cell but not the high one, and
    xgcm's ``padding='fill'`` would average the last velocity with 0 (M0
    task 5) -- finite and wrong, so it is NaN here instead.  Coast-facing
    faces are NaN in the OSN ``U``/``V`` (M0 task 3), so ``u_c`` is NaN in
    the ocean cell adjacent to land along each axis.
    """
    udims = require_u_point(U, 'U')
    require_v_point(V, 'V')
    cdims = op._centred_dims_like(udims)
    u_c, v_c = ng.interp_pair_to_center(U, V, grid)
    u_c = assert_dims(u_c.compute().astype('float64'), cdims, 'interp_pair_to_center[0]')
    v_c = assert_dims(v_c.compute().astype('float64'), cdims, 'interp_pair_to_center[1]')
    u_c = u_c.copy(); v_c = v_c.copy()
    u_c[{'i': -1}] = np.nan                 # the missing i_g = ni face
    v_c[{'j': -1}] = np.nan                 # the missing j_g = nj face
    u_c.name, v_c.name = 'u_c', 'v_c'
    for da, ln in ((u_c, 'model-x'), (v_c, 'model-y')):
        da.attrs.clear()
        da.attrs.update(units='m s-1', long_name=f'{ln} velocity at cell centres (model basis)')
    return u_c, v_c


def _centred_pair(u, v, grid_ds, grid):
    """Accept either the raw staggered pair or an already-centred one."""
    if X_STAG in u.dims or Y_STAG in v.dims:
        return centre_velocities(u, v, grid_ds, grid)
    require_centred(u, 'u_mid')
    require_centred(v, 'v_mid')
    return u, v


def _spacing_at_centres(grid_ds):
    """``(dxC, dyC)`` at the cell centres, positional ``(j, i)``: the mean
    of the two faces of each cell, NaN at the last cell along the axis (its
    high face is outside the tile).  ``dxC[j, i_g = i]`` is the distance
    from centre ``i-1`` to centre ``i`` along ``i``; ``dxG == dxC`` and
    ``dyG == dyC`` to 1e-4 on the tile and both vary by 0.02% per cell, so
    face vs centre is a 0.01% choice -- but it must be ``dxC`` with ``i``
    and ``dyC`` with ``j`` (verified against the haversine centre distances,
    M1 task 3)."""
    dxc = _positional(grid_ds, 'dxC')
    dyc = _positional(grid_ds, 'dyC')
    dx_c = np.full_like(dxc, np.nan)
    dy_c = np.full_like(dyc, np.nan)
    dx_c[:, :-1] = 0.5 * (dxc[:, :-1] + dxc[:, 1:])
    dy_c[:-1, :] = 0.5 * (dyc[:-1, :] + dyc[1:, :])
    return dx_c, dy_c


# ---------------------------------------------------------------------------
# interpolation kernel
# ---------------------------------------------------------------------------
def _check_order(order, allow_low_order):
    order = int(order)
    if order < 1 or order % 2 == 0:
        raise ValueError(f'interp order must be a positive odd integer, got {order}')
    if order < MIN_ORDER and not allow_low_order:
        raise ValueError(f'interp order {order} < {MIN_ORDER}: bilinear interpolation of a '
                         f'front biases G by dx^2 G_xx/8 at maxima (planning §5.3); '
                         'order >= 3 is required. Pass allow_low_order=True only in a test.')
    return order


def _kernel_nodes(order):
    """Node offsets relative to ``floor(p)``: ``-(order-1)/2 .. (order+1)/2``."""
    m = (order - 1) // 2
    return np.arange(-m, order - m + 1)


def _lagrange_weights(t, ks):
    """Lagrange weights on the nodes ``ks`` for the fractional position
    ``t`` in ``[0, 1)``: ``w_k = prod_{m != k} (t - m) / (k - m)``.  At
    ``t = 0`` they are exactly ``1`` on node 0 and ``0`` elsewhere."""
    out = []
    for k in ks:
        w = np.ones_like(t)
        for m in ks:
            if m != k:
                w = w * ((t - m) / (k - m))
        out.append(w)
    return out


def _index_field(d, shape):
    """``di``/``dj`` as a float64 ``(nj, ni)`` array (scalar, numpy or a
    ``(face, j, i)`` DataArray accepted)."""
    a = np.asarray(d.values if isinstance(d, xr.DataArray) else d, dtype='float64')
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return np.broadcast_to(a, shape)


def interp_to_departure(field, di, dj, order=3, *, allow_low_order=False):
    """``field`` evaluated at the departure points ``(j - dj, i - di)`` by
    tensor-product Lagrange interpolation of odd degree ``order`` on the
    ``order + 1`` nearest nodes per axis (see the module docstring for why
    not ``map_coordinates``).  ``field`` is a ``(..., j, i)`` DataArray or
    array; ``di``, ``dj`` are scalars or ``(j, i)`` fields (cells, positive
    = the parcel moved towards larger ``i`` / ``j``).  Same type and shape
    out.

    NaN wherever the support (the ``(order+1)^2`` nodes around the
    departure point, including nodes whose weight happens to be zero at an
    integer offset) contains a NaN or lies outside the array, or where
    ``di``/``dj`` is NaN.  Nothing is filled.  ``order < 3`` raises unless
    ``allow_low_order=True`` (tests and V5 only).
    """
    order = _check_order(order, allow_low_order)
    is_da = isinstance(field, xr.DataArray)
    a = np.asarray(field.values if is_da else field, dtype='float64')
    if a.ndim < 2:
        raise ValueError(f'interp_to_departure: need a (..., j, i) array, got ndim {a.ndim}')
    nj, ni = a.shape[-2:]
    di2, dj2 = _index_field(di, (nj, ni)), _index_field(dj, (nj, ni))
    jj, ii = np.meshgrid(np.arange(nj, dtype='float64'), np.arange(ni, dtype='float64'),
                         indexing='ij')
    pj, pi = jj - dj2, ii - di2                       # the departure position
    ok = np.isfinite(pj) & np.isfinite(pi)
    pj = np.where(ok, pj, 0.0)
    pi = np.where(ok, pi, 0.0)
    j0, i0 = np.floor(pj).astype(int), np.floor(pi).astype(int)
    tj, ti = pj - j0, pi - i0                         # fractional parts in [0, 1)
    ks = _kernel_nodes(order)
    Wj, Wi = _lagrange_weights(tj, ks), _lagrange_weights(ti, ks)
    out = np.zeros(a.shape)
    bad = np.broadcast_to(~ok, a.shape).copy()
    for kj, wj in zip(ks, Wj):
        jn = j0 + kj
        inside_j = (jn >= 0) & (jn < nj)
        jc = np.clip(jn, 0, nj - 1)
        for ki, wi in zip(ks, Wi):
            inn = i0 + ki
            inside = inside_j & (inn >= 0) & (inn < ni)
            vals = a[..., jc, np.clip(inn, 0, ni - 1)]
            finite = np.isfinite(vals)
            bad |= ~inside | ~finite
            out += (wj * wi) * np.where(finite, vals, 0.0)
    out[bad] = np.nan
    if is_da:
        res = field.copy(data=out)
        res.attrs = dict(field.attrs, interp_order=order)
        return res
    return out


# ---------------------------------------------------------------------------
# departure points
# ---------------------------------------------------------------------------
def departure_index(u_c, v_c, grid_ds, dt=DT, n_iter=3, vel_order=1):
    """Displacement ``(di, dj)`` in cells of the parcel arriving at each
    centre over ``dt``, from the centred model-basis velocities: the
    departure point is ``(j - dj, i - di)``.

    ``di = u_c dt / dxC``, ``dj = v_c dt / dyC`` (native index space, no
    rotation), refined by ``n_iter`` iterations of the midpoint rule
    ``d <- dt * u(x - d/2)`` with the velocity interpolated at the
    trajectory midpoint (second order in ``dt``; for a linear velocity the
    iteration converges to ``d = dt u / (1 - dt grad u / 2)``).  ``u_c`` is
    already the *time*-midpoint velocity when the caller passes
    ``0.5 (u_t + u_tp1)``.

    The velocity is interpolated with ``vel_order = 1`` (bilinear) by
    default: the order rule protects the sharp front in ``b``, whereas the
    velocity is smooth at the grid scale and the displacement error from
    bilinear interpolation, ``dx^2 (grad^2 u) dt / 8``, is far below 0.01
    cell (on the real hour the change from ``vel_order = 3`` is logged, M1
    task 3).  NaN where the velocity or its midpoint support is NaN.
    Returns ``(di, dj)`` as DataArrays shaped like ``u_c``.
    """
    dims = require_centred(u_c, 'u_c')
    require_centred(v_c, 'v_c')
    dx_c, dy_c = _spacing_at_centres(grid_ds)
    ui = _index_field(u_c, dx_c.shape) * dt / dx_c     # cells per dt, along i
    vj = _index_field(v_c, dy_c.shape) * dt / dy_c     # cells per dt, along j
    di, dj = ui.copy(), vj.copy()
    for _ in range(int(n_iter)):
        # the velocity at the trajectory midpoint x - d/2
        di_new = interp_to_departure(ui, 0.5 * di, 0.5 * dj, vel_order, allow_low_order=True)
        dj_new = interp_to_departure(vj, 0.5 * di, 0.5 * dj, vel_order, allow_low_order=True)
        di, dj = di_new, dj_new
    shape = u_c.shape
    di_da = u_c.copy(data=di.reshape(shape))
    dj_da = v_c.copy(data=dj.reshape(shape))
    for da, name, ax in ((di_da, 'di', 'i'), (dj_da, 'dj', 'j')):
        da.name = name
        da.attrs.clear()
        da.attrs.update(units='cells', long_name=f'parcel displacement along {ax} over dt '
                        '(departure = arrival - d)', dt=float(dt), n_iter=int(n_iter))
    assert_dims(di_da, dims, 'departure_index')
    return di_da, dj_da


# ---------------------------------------------------------------------------
# the gradient at the departure point
# ---------------------------------------------------------------------------
def grad_b_at_departure(b, di, dj, grid_ds, order=3, *, allow_low_order=False):
    """Geographic ``(b_x, b_y)`` at the departure points: ``b`` interpolated
    onto the five-point tracer stencil centred at ``x_d`` (the displacement
    held fixed across the stencil), then the **same** stencil as
    ``operators.grad_b`` / ``calculate_native_gradient_tracer`` -- difference
    to the two faces, divide by ``dxC`` there, average back to the centre,
    rotate with ``CS``/``SN`` -- in the same operation order, so at zero
    displacement the result is bit-for-bit ``operators.grad_b`` away from
    the tile edge (``test_zero_velocity_identity``).  ``dxC``, ``dyC``,
    ``CS``, ``SN`` are taken at the arrival cell (they vary by 0.02% per
    cell; the rotation is uniform on the tile).  NaN where any of the five
    interpolations is NaN, or at the last cell along an axis (its high face
    is outside the tile; the dbof stencil would use xgcm's 0 fill there).
    """
    dims = require_centred(b, 'b')
    order = _check_order(order, allow_low_order)
    kw = dict(order=order, allow_low_order=True)
    # b at x_d and at the four stencil neighbours x_d +- e_i, x_d +- e_j:
    # the neighbour at i+1 sits at position (i + 1) - di = i - (di - 1)
    b0 = interp_to_departure(b, di, dj, **kw)
    bE = interp_to_departure(b, di - 1, dj, **kw)      # i + 1
    bW = interp_to_departure(b, di + 1, dj, **kw)      # i - 1
    bN = interp_to_departure(b, di, dj - 1, **kw)      # j + 1
    bS = interp_to_departure(b, di, dj + 1, **kw)      # j - 1
    dxc, dyc = _positional(grid_ds, 'dxC'), _positional(grid_ds, 'dyC')
    dx_lo, dx_hi = dxc, np.full_like(dxc, np.nan)      # dxC at the low / high face of cell i
    dx_hi[:, :-1] = dxc[:, 1:]
    dy_lo, dy_hi = dyc, np.full_like(dyc, np.nan)
    dy_hi[:-1, :] = dyc[1:, :]
    nd = b.ndim - 2
    ex = (np.newaxis,) * nd + (slice(None), slice(None))
    # diff to the faces / dxC, then xgcm's 0.5 * (low + high) back to the centre
    gX = ((b0 - bW) / dx_lo[ex] + (bE - b0) / dx_hi[ex]) * 0.5
    gY = ((b0 - bS) / dy_lo[ex] + (bN - b0) / dy_hi[ex]) * 0.5
    cs, sn = _positional(grid_ds, 'CS')[ex], _positional(grid_ds, 'SN')[ex]
    bx = gX * cs - gY * sn                             # rotate_vector_to_geographic
    by = gX * sn + gY * cs
    bx = assert_dims(bx, dims, 'grad_b_at_departure[0]')
    by = assert_dims(by, dims, 'grad_b_at_departure[1]')
    bx.name, by.name = 'b_x_d', 'b_y_d'
    for da, ln in ((bx, 'zonal'), (by, 'meridional')):
        da.attrs.clear()
        da.attrs.update(units='s-2', long_name=f'{ln} buoyancy gradient at the departure point '
                        '(geographic; b interpolated onto the stencil)', interp_order=order)
    return bx, by


def gradb2_at_departure(b, di, dj, grid_ds, order=3, *, allow_low_order=False):
    """``G(x_d) = b_x^2 + b_y^2`` from :func:`grad_b_at_departure` -- the
    component stencil, as ``operators.gradb2``.  Never an interpolated ``G``."""
    bx, by = grad_b_at_departure(b, di, dj, grid_ds, order, allow_low_order=allow_low_order)
    G = bx ** 2 + by ** 2
    G.name = 'G_d'
    G.attrs.update(units='s-4', long_name='|grad_h b|^2 at the departure point',
                   interp_order=int(order))
    return G


# ---------------------------------------------------------------------------
# the two estimates of DG/Dt
# ---------------------------------------------------------------------------
def measured_DGDt(b_t, b_tp1, u_mid, v_mid, grid_ds, grid, dt=DT, order=3, n_iter=3, *,
                  allow_low_order=False):
    """Semi-Lagrangian ``D_h G / Dt`` [s^-5] at the cell centres over one
    interval: ``[G(x, t+dt) - G(x_d, t)] / dt`` with ``G(t+dt) =
    operators.gradb2(b_tp1)`` and ``G(x_d, t)`` from
    :func:`gradb2_at_departure` (``b_t`` interpolated onto the departure
    stencil at ``order >= 3``, then the same stencil).  ``u_mid``, ``v_mid``
    are the time-midpoint velocities ``0.5 (u_t + u_tp1)`` (raw staggered or
    already centred, model basis); the departure is :func:`departure_index`.

    Compare with ``2F`` at the midpoint time (``F = (1/2) DG/Dt``).  NaN
    where either side is (land within the stencil or interpolation reach,
    the coastal NaN faces of ``U``/``V``, or a stencil leaving the tile).
    """
    dims = require_centred(b_t, 'b_t')
    if tuple(b_tp1.dims) != dims:
        raise ValueError(f'measured_DGDt: b_tp1 dims {b_tp1.dims} != b_t dims {dims}')
    u_c, v_c = _centred_pair(u_mid, v_mid, grid_ds, grid)
    di, dj = departure_index(u_c, v_c, grid_ds, dt=dt, n_iter=n_iter)
    G_d = gradb2_at_departure(b_t, di, dj, grid_ds, order, allow_low_order=allow_low_order)
    G_tp1 = op.gradb2(b_tp1, grid_ds, grid)
    DGDt = (G_tp1 - G_d) / dt
    DGDt = assert_dims(DGDt, dims, 'measured_DGDt')
    DGDt.name = 'DGDt_semilag'
    DGDt.attrs.clear()
    DGDt.attrs.update(units='s-5', long_name='semi-Lagrangian D_h G/Dt: [G(x, t+dt) - G(x_d, t)]/dt',
                      interp_order=int(order), n_iter=int(n_iter), dt=float(dt),
                      convention='compare with 2F at the midpoint time')
    return DGDt


def eulerian_DGDt(G_t, G_tp1, u_mid, v_mid, grid_ds, grid, dt=DT):
    """Eulerian ``D_h G / Dt = dG/dt + u . grad G`` [s^-5] at the midpoint
    time: ``(G_tp1 - G_t) / dt + u_mid . grad[0.5 (G_t + G_tp1)]``, with
    ``grad G`` from ``operators.grad_b``'s stencil (geographic) and the
    centred model-basis velocity rotated to geographic for the dot product
    (an invariant).  The independent cross-check of :func:`measured_DGDt`,
    not the primary estimate (planning §5.3): in the California Current the
    hourly displacement is under a cell, so the two large cancelling terms
    of the Eulerian split are tolerable here.
    """
    dims = require_centred(G_t, 'G_t')
    if tuple(G_tp1.dims) != dims:
        raise ValueError(f'eulerian_DGDt: G_tp1 dims {G_tp1.dims} != G_t dims {dims}')
    u_c, v_c = _centred_pair(u_mid, v_mid, grid_ds, grid)
    Gx, Gy = op.grad_b(midpoint_time(G_t, G_tp1), grid_ds, grid)
    cs, sn = grid_ds['CS'], grid_ds['SN']
    u_east = u_c * cs - v_c * sn                         # rotate_vector_to_geographic
    v_north = u_c * sn + v_c * cs
    DGDt = (G_tp1 - G_t) / dt + u_east * Gx + v_north * Gy
    DGDt = assert_dims(DGDt, dims, 'eulerian_DGDt')
    DGDt.name = 'DGDt_euler'
    DGDt.attrs.clear()
    DGDt.attrs.update(units='s-5', long_name='Eulerian D_h G/Dt = (G_tp1 - G_t)/dt + u.grad G '
                      'at the midpoint time', dt=float(dt))
    return DGDt
