""" Flux-form finite-volume tracer advection on the C-grid: the truth of
**V3b, the finite-volume null** (prompt 2 task 6b; M1-Q2, a recorded bias,
not a gate).

Why.  V3's truth is advected by *our own* semi-Lagrangian step, so both
sides of that null see the same centred velocity ``u_c`` and it is blind,
by construction, to the 0.80-0.85x attenuation of the interpolated Jacobian
relative to the model's flux-form strain (M0 task 5, M1 tasks 2 and 6).
LLC4320 advects its tracers in **flux form** with the seventh-order
one-step monotonicity-preserving scheme OS7MP (``tempAdvScheme = 7``,
``multiDimAdvection``, ``deltaT = 25 s``; planning §2.3).  Here the truth
is a flux-form step on the same C-grid, with the same face velocities, so
that the measured and predicted sides can disagree for the reason the
model's fronts might.  The number V3b produces is the *bias of our
pipeline relative to a model-like advection*.

The step (MITgcm conventions, ``gad_advection.F`` / ``gad_calc_rhs.F``)
--------------------------------------------------------------------
Face transports ``uTrans = U dyG hFacW`` on ``(j, i_g)``, ``vTrans = V dxG
hFacS`` on ``(j_g, i)`` (``drF`` cancels in a single layer); face tracer
values ``b_f`` from one of the reconstructions below; the tendency of cell
``(j, i)`` per directional sweep is

    db/dt = -[ Delta_X(uTrans b_f) - b Delta_X(uTrans) ] / (rA hFacC)      (X sweep)

and the same along Y.  **Divergence treatment (decision):** the ``- b
Delta(uTrans)`` term makes each sweep the *advective* form built from flux
differences, ``d_t b + div(u b) = b div u``, i.e. ``D b/Dt = 0`` for the
tracer, which is (i) what a surface tracer obeys when only its horizontal
kinematics are modelled -- the null is about the kinematic identity
``F = (1/2) DG/Dt``; (ii) exactly MITgcm's default multi-dimensional
branch (``GAD_MULTIDIM_COMPRESSIBLE`` undefined): each 1-D sweep subtracts
``tracer * (uTrans(i+1) - uTrans(i))`` so a uniform tracer stays uniform
under a divergent flow; and (iii) the model's top-cell budget under the
linear free surface, checked in the source in M0 task 3: the surface
transport is zero, the vertical sweep contributes ``-w_base (b_base -
b)/drF`` with ``w_base = W(0) + drF delta``, so the horizontal part of the
top-cell tendency is precisely the advective form above and the vertical
part is the separately measured top-cell term of M3 (``vertical.py``), not
part of this null.  The alternative, the conservative ``-div(u b)`` alone
(``form='conservative'``), adds ``-b delta`` to the tendency -- with
``delta ~ 1e-5 s^-1`` that is 4% of ``b`` per hour and ``-2 delta G`` in
the ``G`` budget, a spurious signal larger than the strain's; it is kept
only to show its size.

Sweeps are directionally split, forward in time, X then Y on even
sub-steps and Y then X on odd ones, as MITgcm alternates them
(``integrator='split'``); the second sweep acts on the field the first
updated.  The one-step schemes below are exact in time for a constant
velocity, so the splitting is the only time error (second order in the
Courant number; the model's own is ``|u| dt/dx <= 0.02`` at 25 s, here
``dt_sub = 100 s`` by default, ``c <= 0.06`` on the tile).  The centred
scheme is not stable forward-in-time (anti-diffusive at ``O(c^2)``), so it
is integrated unsplit with the three-stage SSP Runge-Kutta
(``integrator='rk3'``; amplitude error ``< (c)^4/12`` per step).

Schemes (``scheme``)
--------------------
* ``'centred'`` -- second-order centred fluxes, ``b_f = (b_{i-1} + b_i)/2``
  (``tempAdvScheme = 2`` in space).  No dissipation: the cleanest test of
  the pure C-grid stencil effect -- for a uniform velocity the tendency is
  ``-u (b_{i+1} - b_{i-1})/(2 dx)``, the same ``2 dx`` difference our
  gradient stencil ``L`` uses, while the strain the tracer feels is the
  one-sided ``(uTrans(i+1) - uTrans(i))`` of the *face* velocities, i.e.
  the flux-form divergence that the interpolated Jacobian sees at 0.85x.
* ``'dst3'`` -- MITgcm's third-order direct-space-time scheme
  (``tempAdvScheme = 30``, ``gad_dst3_adv_x.F``, no limiter): for ``u > 0``
  ``b_f = b_{i-1} + d0 (b_i - b_{i-1}) + d1 (b_{i-1} - b_{i-2})`` with
  ``d0 = (2 - c)(1 - c)/6``, ``d1 = (1 - c^2)/6`` and ``c = |u| dt/dxC``;
  mirrored for ``u < 0``.  The cross-check with a *larger* implicit
  diffusion (third order: ``kappa ~ |u| dx^3 k^2 / 12``).
* ``'os7'`` -- the unlimited seventh-order one-step scheme of Daru &
  Tenaud (2004): the face value is the exact time-average over the
  sub-step of the degree-6 polynomial reconstruction through the seven
  cell means ``b_{i-4} .. b_{i+2}`` (four upwind, three downwind), a
  degree-6 polynomial in ``c`` whose ``c -> 0`` limit is the classical
  seventh-order upwind-biased face value ``(-3, 25, -101, 319, 214, -38,
  4)/420`` and whose ``c = 1`` limit is the exact shift ``b_{i-1}``
  (:func:`os7_weight_matrix`, derived from the primitive function, not
  transcribed).  Its dissipation is the ``kappa_num`` of planning §2.3
  (``|u| dx^7 k^6 / 280`` at small ``k dx``).
* ``'os7mp'`` -- ``'os7'`` followed by the monotonicity-preserving limiter
  of Suresh & Huynh (1997) (``alpha = 4``; the MP bounds from the
  curvature-limited ``d^{M4}`` and the ``UL``/``LC``/``MD`` values), which
  is the construction Daru & Tenaud's OS7MP and MITgcm's
  ``gad_os7mp_adv_x.F`` follow.  **How it differs from MITgcm's OS7MP:**
  the model's limiter bounds carry the Courant number (its TVD region is
  ``[0, 2/(1 - c)]``-like and ``alpha`` tends to ``(1 - c)/c``), which at
  the model's ``c <= 0.02`` and our ``c <= 0.06`` is a second-order-in-``c``
  difference in *where* the limiter engages, not in the unlimited scheme;
  and MITgcm reduces the stencil order with ``maskW`` next to land, where
  here land is filled by its nearest ocean value (below).  So: an honest
  OS7MP-like scheme, exact in the smooth-flow limit, approximate in the
  limiter's bounds at finite ``c``.

Boundaries and land
-------------------
Land (``hFacC = 0`` or NaN ``b``) is filled with the nearest ocean value
for the reconstruction only; its faces carry ``hFacW = hFacS = 0`` (the
OSN ``U``/``V`` are NaN exactly there, M0 task 3 -- set to zero
transport), so no flux crosses them and land cells never change; they are
NaN again on output.  The tile edge is padded with edge values
(zero-gradient) for ``b``, the transports and the metrics -- the flow
crosses the tile edge, so the padded flux is the extrapolated one.  The
reach of either choice is measured, not assumed (the crop and fill tests
in V3b: nothing on ``mask_analysis`` moves).  Everything positional
``(nj, ni)`` float64 numpy inside; ``(face, j, i)`` DataArrays in and out.
"""

import numpy as np
import xarray as xr
from numpy.polynomial import Polynomial
from scipy import ndimage

from masking import _positional
from operators import require_centred, require_u_point, require_v_point, assert_dims

DT = 3600.0
PAD = 4                              # the seven-point stencil's reach
SCHEMES = ('centred', 'dst3', 'os7', 'os7mp')
MP_ALPHA = 4.0                       # Suresh & Huynh's alpha
DT_SUB = 100.0                       # s; c <= 0.06 on the tile (the model: 25 s)


# ---------------------------------------------------------------------------
# the OS7 reconstruction: weights as polynomials in the Courant number
# ---------------------------------------------------------------------------
def os7_weight_matrix():
    """``A[k, m]`` such that, for ``u > 0`` at the face between cells
    ``i-1`` and ``i``, the time-averaged face value over a sub-step of
    Courant number ``c`` is ``sum_k w_k(c) b_{i+k}``, ``k = -4..2``, with
    ``w_k(c) = sum_m A[k, m] c^m`` (``m = 0..6``).

    Derivation: with the face at ``x = 0`` and cell ``k`` on ``[k, k+1]``,
    the primitive ``P(x) = int_{-4}^{x} b`` is known at the eight cell
    boundaries ``x = -4..3`` (``P(n) = sum_{k < n} b_k``); ``Q`` is its
    degree-7 Lagrange interpolant and the tracer swept through the face in
    one sub-step is ``[Q(0) - Q(-c)]/c`` -- the exact one-step (space-time)
    scheme for a constant velocity.  With ``l_n`` the Lagrange basis and
    ``l_n(x) = sum_p a_{n p} x^p``, ``[l_n(0) - l_n(-c)]/c = -sum_{p >= 1}
    a_{n p} (-1)^p c^{p-1}``, and the weight of ``b_k`` is the sum over the
    nodes ``n > k``.  ``c -> 0`` gives ``Q'(0)``, the seventh-order face
    value ``(-3, 25, -101, 319, 214, -38, 4)/420``; ``c = 1`` gives the
    exact shift ``(0, 0, 0, 1, 0, 0, 0)``."""
    nodes = np.arange(-4, 4, dtype='float64')
    A = np.zeros((7, 7))
    for n_idx, n in enumerate(nodes):
        others = np.delete(nodes, n_idx)
        ln = Polynomial.fromroots(others) / np.prod(n - others)
        a = ln.coef                                        # a_p, p = 0..7
        # -sum_{p>=1} a_p (-1)^p c^{p-1}: coefficient of c^m is -a_{m+1} (-1)^{m+1}
        contrib = np.array([-a[m + 1] * (-1) ** (m + 1) for m in range(7)])
        for k_idx, k in enumerate(range(-4, 3)):
            if n > k:
                A[k_idx] += contrib
    return A


_OS7_A = os7_weight_matrix()


def os7_weights(c):
    """The seven OS7 weights ``w_k(c)``, ``k = -4..2`` (upwind orientation),
    for an array of Courant numbers ``c >= 0``."""
    powers = [np.ones_like(c)]
    for _ in range(6):
        powers.append(powers[-1] * c)
    return [sum(_OS7_A[k, m] * powers[m] for m in range(7)) for k in range(7)]


# ---------------------------------------------------------------------------
# the MP limiter (Suresh & Huynh 1997)
# ---------------------------------------------------------------------------
def _minmod2(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def _minmod4(a, b, c, d):
    sa, sb, sc, sd = np.sign(a), np.sign(b), np.sign(c), np.sign(d)
    return (0.125 * (sa + sb) * np.abs((sa + sc) * (sa + sd))
            * np.minimum(np.minimum(np.abs(a), np.abs(b)), np.minimum(np.abs(c), np.abs(d))))


def mp_limit(vL, vm2, vm1, v0, vp1, vp2, alpha=MP_ALPHA, eps=0.0):
    """Suresh & Huynh's monotonicity-preserving limiter applied to the
    high-order face value ``vL`` of the face downwind of cell ``v0``
    (upwind orientation: ``vm1``, ``vm2`` further upwind, ``vp1``, ``vp2``
    downwind).  ``vL`` is kept when it lies between ``v0`` and ``v^MP = v0 +
    minmod(vp1 - v0, alpha (v0 - vm1))``; otherwise it is clipped to the
    interval ``[v_min, v_max]`` built from the median-of-curvature
    ``d^M4`` values (``v^MD``, ``v^LC``, ``v^UL``).  Accuracy-preserving
    at smooth extrema, monotone across a jump."""
    d_m = vm2 - 2 * vm1 + v0
    d_0 = vm1 - 2 * v0 + vp1
    d_p = v0 - 2 * vp1 + vp2
    dM4p = _minmod4(4 * d_0 - d_p, 4 * d_p - d_0, d_0, d_p)
    dM4m = _minmod4(4 * d_m - d_0, 4 * d_0 - d_m, d_m, d_0)
    vUL = v0 + alpha * (v0 - vm1)
    vMD = 0.5 * (v0 + vp1) - 0.5 * dM4p
    vLC = v0 + 0.5 * (v0 - vm1) + (4.0 / 3.0) * dM4m
    vmin = np.maximum(np.minimum(np.minimum(v0, vp1), vMD), np.minimum(np.minimum(v0, vUL), vLC))
    vmax = np.minimum(np.maximum(np.maximum(v0, vp1), vMD), np.maximum(np.maximum(v0, vUL), vLC))
    vMP = v0 + _minmod2(vp1 - v0, alpha * (v0 - vm1))
    need = (vL - v0) * (vL - vMP) > eps
    clipped = vL + _minmod2(vmin - vL, vmax - vL)          # median(vL, vmin, vmax)
    return np.where(need, clipped, vL)


# ---------------------------------------------------------------------------
# face values, sweeps, the step
# ---------------------------------------------------------------------------
def _take(a, axis, start, stop):
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(start, stop)
    return a[tuple(idx)]


def _pad(a):
    """Zero-gradient padding by ``PAD`` on both axes."""
    return np.pad(a, PAD, mode='edge')


def _repad(bp):
    """Reset the pad of a padded array to the edge values (zero gradient)."""
    n0, n1 = bp.shape
    bp[:PAD, :] = bp[PAD:PAD + 1, :]
    bp[n0 - PAD:, :] = bp[n0 - PAD - 1:n0 - PAD, :]
    bp[:, :PAD] = bp[:, PAD:PAD + 1]
    bp[:, n1 - PAD:] = bp[:, n1 - PAD - 1:n1 - PAD]
    return bp


def face_values(bp, c, axis, scheme):
    """Tracer values on the ``n + 1`` faces of the ``n`` interior cells along
    ``axis`` of the padded field ``bp`` (face ``i`` sits between cells
    ``i - 1`` and ``i``), from the signed Courant numbers ``c`` at those
    faces (shape of the face array), by ``scheme``."""
    n = bp.shape[axis] - 2 * PAD
    m = bp.shape[1 - axis] - 2 * PAD

    def cell(k):                                   # cell i + k for faces i = PAD .. PAD + n
        return _take(_take(bp, axis, PAD + k, PAD + k + n + 1), 1 - axis, PAD, PAD + m)
    if scheme == 'centred':
        return 0.5 * (cell(-1) + cell(0))
    ac, up = np.abs(c), c >= 0
    if scheme == 'dst3':
        d0 = (2 - ac) * (1 - ac) / 6.0
        d1 = (1 - ac ** 2) / 6.0
        bm2, bm1, b0, bp1 = cell(-2), cell(-1), cell(0), cell(1)
        fp = bm1 + d0 * (b0 - bm1) + d1 * (bm1 - bm2)
        fm = b0 - d0 * (b0 - bm1) - d1 * (bp1 - b0)
        return np.where(up, fp, fm)
    if scheme not in ('os7', 'os7mp'):
        raise ValueError(f'fvadvect: scheme must be one of {SCHEMES}, got {scheme!r}')
    W = os7_weights(ac)
    cells = {k: cell(k) for k in range(-4, 4)}
    fp = sum(W[k + 4] * cells[k] for k in range(-4, 3))          # upwind cell i - 1
    fm = sum(W[k + 4] * cells[-1 - k] for k in range(-4, 3))     # mirrored: upwind cell i
    f = np.where(up, fp, fm)
    if scheme == 'os7mp':
        # upwind-oriented neighbours v_{j+s}: cell(-1 + s) for u > 0, cell(-s) for u < 0
        v = [np.where(up, cells[-1 + s], cells[-s]) for s in (-2, -1, 0, 1, 2)]
        f = mp_limit(f, *v)
    return f


def _sweep_tendency(bp, uT, c, axis, r_dt, scheme, form):
    """``db/dt`` of the interior cells from one directional sweep: ``-[
    Delta(uTrans b_f) - b Delta(uTrans)] / (rA hFacC)`` (``form =
    'advective'``, the model's default) or ``-Delta(uTrans b_f) / (rA
    hFacC)`` (``'conservative'``).  ``uT``, ``c`` on the ``n + 1`` faces;
    ``r_dt = 1/(rA hFacC)`` on the interior (0 on land)."""
    flux = uT * face_values(bp, c, axis, scheme)
    dF = _take(flux, axis, 1, None) - _take(flux, axis, 0, -1)
    if form == 'advective':
        dU = _take(uT, axis, 1, None) - _take(uT, axis, 0, -1)
        b_int = bp[PAD:-PAD, PAD:-PAD]
        return -(dF - b_int * dU) * r_dt
    if form == 'conservative':
        return -dF * r_dt
    raise ValueError(f"fvadvect: form must be 'advective' or 'conservative', got {form!r}")


def _faces_x(a):
    """A ``(nj, ni)`` field on ``(j, i_g)`` extended to the ``ni + 1``
    X-faces of the interior: the high face of the last cell is not in the
    tile and takes the edge value."""
    return np.concatenate([a, a[:, -1:]], axis=1)


def _faces_y(a):
    return np.concatenate([a, a[-1:, :]], axis=0)


def fv_step_arrays(b, uT, vT, cX, cY, r_inv, dt, scheme='os7mp', dt_sub=DT_SUB, form='advective',
                   integrator=None):
    """The step on positional arrays: ``b (nj, ni)`` with land already
    filled, transports ``uT (nj, ni+1)``, ``vT (nj+1, ni)`` on the faces,
    ``cX``, ``cY`` the signed Courant numbers per second (``u/dxC``) on
    the same faces, ``r_inv = 1/(rA hFacC)`` (0 on land).  ``n_sub =
    ceil(dt/dt_sub)`` equal sub-steps.  Returns ``b`` after ``dt``."""
    integrator = integrator or ('rk3' if scheme == 'centred' else 'split')
    n_sub = int(np.ceil(dt / dt_sub))
    h = dt / n_sub
    r_dt = r_inv
    kX, kY = cX * h, cY * h
    bp = _pad(np.asarray(b, dtype='float64'))

    def tend(bp_, axis):
        if axis == 1:
            return _sweep_tendency(bp_, uT, kX, 1, r_dt, scheme, form)
        return _sweep_tendency(bp_, vT, kY, 0, r_dt, scheme, form)

    if integrator == 'split':
        for s in range(n_sub):
            for axis in ((1, 0) if s % 2 == 0 else (0, 1)):
                bp[PAD:-PAD, PAD:-PAD] += h * tend(bp, axis)
                _repad(bp)
    elif integrator == 'rk3':
        def L(bp_):
            return tend(bp_, 1) + tend(bp_, 0)
        for _ in range(n_sub):
            b0 = bp[PAD:-PAD, PAD:-PAD].copy()
            b1 = b0 + h * L(bp)
            bp[PAD:-PAD, PAD:-PAD] = b1; _repad(bp)
            b2 = 0.75 * b0 + 0.25 * (b1 + h * L(bp))
            bp[PAD:-PAD, PAD:-PAD] = b2; _repad(bp)
            bp[PAD:-PAD, PAD:-PAD] = b0 / 3.0 + (2.0 / 3.0) * (b2 + h * L(bp)); _repad(bp)
    else:
        raise ValueError(f"fvadvect: integrator must be 'split' or 'rk3', got {integrator!r}")
    return bp[PAD:-PAD, PAD:-PAD].copy()


def _grid_arrays(grid_ds, U, V):
    """Transports, Courant rates and ``1/(rA hFacC)`` from the grid and the
    raw staggered velocities (NaN velocity = zero transport = a coast face,
    ``hFacW = 0`` there cell for cell on the tile, M0 task 3)."""
    dxC, dyC = _positional(grid_ds, 'dxC'), _positional(grid_ds, 'dyC')
    dxG, dyG = _positional(grid_ds, 'dxG'), _positional(grid_ds, 'dyG')
    rA, hC = _positional(grid_ds, 'rA'), _positional(grid_ds, 'hFacC')
    hW = _positional(grid_ds, 'hFacW') if 'hFacW' in grid_ds else np.ones_like(dxC)
    hS = _positional(grid_ds, 'hFacS') if 'hFacS' in grid_ds else np.ones_like(dyC)
    u = np.asarray(U.values if isinstance(U, xr.DataArray) else U, dtype='float64')
    v = np.asarray(V.values if isinstance(V, xr.DataArray) else V, dtype='float64')
    u, v = u.reshape(dxC.shape), v.reshape(dyC.shape)
    hW = np.where(np.isfinite(u), hW, 0.0)
    hS = np.where(np.isfinite(v), hS, 0.0)
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    uT, vT = _faces_x(u * dyG * hW), _faces_y(v * dxG * hS)
    cX, cY = _faces_x(u / dxC), _faces_y(v / dyC)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_inv = np.where(hC > 0, 1.0 / (rA * hC), 0.0)
    return dict(uT=uT, vT=vT, cX=cX, cY=cY, r_inv=r_inv, ocean=hC > 0)


def fill_land(b, ocean, how='nearest'):
    """``b`` with the non-ocean cells filled: ``'nearest'`` ocean value
    (the default), or ``'mean'`` (the ocean mean; the alternative fill
    used to measure the fill's reach)."""
    b = np.array(b, dtype='float64')
    valid = ocean & np.isfinite(b)
    if valid.all():
        return b
    if how == 'nearest':
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        return b[idx[0], idx[1]]
    if how == 'mean':
        return np.where(valid, b, np.nanmean(b[valid]))
    raise ValueError(f"fill_land: how must be 'nearest' or 'mean', got {how!r}")


def fv_advect(b, U, V, grid_ds, dt=DT, scheme='os7mp', dt_sub=DT_SUB, form='advective',
              integrator=None, fill='nearest'):
    """``b`` advected for ``dt`` by the flux-form finite-volume step with the
    steady staggered velocity ``U (j, i_g)``, ``V (j_g, i)`` (raw model
    components; for V3b the time-midpoint velocity V3 uses).  ``b`` is a
    ``(face, j, i)`` DataArray; the result has its dims, land NaN.  See the
    module docstring for ``scheme``, ``form``, ``integrator`` and the land
    ``fill``."""
    dims = require_centred(b, 'b')
    require_u_point(U, 'U')
    require_v_point(V, 'V')
    ga = _grid_arrays(grid_ds, U, V)
    b0 = np.asarray(b.values, dtype='float64').reshape(ga['r_inv'].shape)
    ocean = ga['ocean'] & np.isfinite(b0)
    bf = fill_land(b0, ocean, fill)
    out = fv_step_arrays(bf, ga['uT'], ga['vT'], ga['cX'], ga['cY'], ga['r_inv'], dt,
                         scheme=scheme, dt_sub=dt_sub, form=form, integrator=integrator)
    out = np.where(ocean, out, np.nan)
    res = b.copy(data=out.reshape(b.shape))
    res = assert_dims(res, dims, 'fv_advect')
    res.name = 'b_fv'
    res.attrs.clear()
    res.attrs.update(units=str(b.attrs.get('units', '')), scheme=scheme, form=form,
                     integrator=integrator or ('rk3' if scheme == 'centred' else 'split'),
                     dt=float(dt), dt_sub=float(dt_sub), n_sub=int(np.ceil(dt / dt_sub)),
                     long_name=f'b advected {dt:g} s by the flux-form C-grid step ({scheme})')
    return res


def advector(scheme='os7mp', dt_sub=DT_SUB, form='advective', integrator=None, fill='nearest'):
    """A ``(b_t, U, V, grid_ds, grid) -> b_tp1`` callable for
    ``validate.null_step(advect=...)``: the finite-volume truth of V3b.
    ``scheme='semilag'`` returns ``None`` (V3's own step)."""
    if scheme == 'semilag':
        return None

    def step(b_t, U, V, grid_ds, grid, dt=DT):
        return fv_advect(b_t, U, V, grid_ds, dt=dt, scheme=scheme, dt_sub=dt_sub, form=form,
                         integrator=integrator, fill=fill)
    step.scheme = scheme
    return step
