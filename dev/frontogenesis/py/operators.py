""" The single shared operator (coding doc §4.3, planning §5.1): buoyancy,
the low-pass filter, the buoyancy gradient, front strength G, the velocity
Jacobian, the kinematic frontogenesis function F, the flux-form strain and
divergence, and the strain-alignment angle.

**Both sides of the comparison go through this module.**  The measured
tendency (semilag.py) and the predicted one (``frontogenesis``) share
``buoyancy``, ``lowpass`` and ``grad_b``, so no stencil or filter can differ
between them.  Filtering happens *once, at the call site*: every operator
here takes fields that are already filtered (or not) and never filters
internally.

Conventions (coding §1)
-----------------------
* ``b = +g sigma0 / rho0`` (JMD95 at p = 0, ``calculate_fields.buoyancy_of_field``):
  it *increases with density*, the negative of textbook buoyancy.  Left as
  is -- ``G`` and ``F`` are quadratic in ``grad b`` and the alignment angle
  enters as ``cos 2 theta``.
* ``G = b_x^2 + b_y^2`` from the **same** ``b_x, b_y`` that enter ``F``
  (``calculate_native_gradient_tracer``: diff, interp to centres, rotate).
  Never the repo's ``calculate_grad_squared_tracer`` (squares on the
  staggered points first; 0.911x in the interior median, M0 task 5): the
  discrete identity ``F = (1/2) DG/Dt`` needs one set of ``b_x, b_y``.
* ``F = -(u_x b_x^2 + (u_y + v_x) b_x b_y + v_y b_y^2)`` and
  ``F = (1/2) DG/Dt``: compare ``2F`` with the measured ``DG/Dt``, always.
* ``grad b`` and the Jacobian are both in the **geographic** basis (the
  dbof helpers rotate with ``CS``/``SN``); ``G``, ``F``, ``delta`` and
  ``|sigma|`` are rotational invariants.  The flux-form strain pair from
  ``calculate_native_strain_vorticity`` is model-basis and is rotated here
  so that ``strain_alignment`` compares like with like.
* float64 throughout; masks are the caller's business (land is already NaN
  in the OSN fields and every stencil propagates it).

Dims are asserted before and after every dbof call.  A centred mask times a
staggered field, or ``calculate_jacobian`` with its arguments swapped,
broadcasts ``(j, i) x (j_g, i_g)`` to 4-D+ -- on a 720 x 720 tile that is an
out-of-memory kill, not an exception (M0 task 5) -- so the staggering is
checked *before* the helper is called, and the output dims after.
"""

import numpy as np
import xarray as xr
from scipy import ndimage

from dbof.utils import native_gradient as ng
from dbof.preprocessing.calculate_fields import buoyancy_of_field

CENTRE_DIMS = ('j', 'i')          # tracer points
X_STAG, Y_STAG = 'i_g', 'j_g'     # U lives on (j, i_g), V on (j_g, i)
HORIZONTAL = ('j', 'j_g', 'i', 'i_g')


# ---------------------------------------------------------------------------
# dims guards (raise unconditionally -- `assert` would be stripped by -O)
# ---------------------------------------------------------------------------
def _dims_of(da, what):
    if not isinstance(da, xr.DataArray):
        raise TypeError(f'{what}: expected an xarray.DataArray, got {type(da).__name__}')
    return tuple(da.dims)


def _require(da, what, need, forbid):
    """``da`` must carry every dim in ``need`` and none in ``forbid``."""
    dims = _dims_of(da, what)
    missing = [d for d in need if d not in dims]
    extra = [d for d in forbid if d in dims]
    if missing or extra:
        raise ValueError(f'{what}: expected dims with {need} and without {forbid}, '
                         f'got {dims}')
    return dims


def require_centred(da, what='field'):
    """A tracer-point field: has ``j`` and ``i``, no staggered dim."""
    return _require(da, what, CENTRE_DIMS, (X_STAG, Y_STAG))


def require_u_point(da, what='U'):
    """The model-x velocity point ``(j, i_g)``."""
    return _require(da, what, ('j', X_STAG), ('i', Y_STAG))


def require_v_point(da, what='V'):
    """The model-y velocity point ``(j_g, i)``."""
    return _require(da, what, (Y_STAG, 'i'), ('j', X_STAG))


def assert_dims(da, expected, what):
    """Raise if ``da.dims`` is not exactly ``expected`` (the §8 rule: after
    every dbof operator call)."""
    dims = _dims_of(da, what)
    if dims != tuple(expected):
        raise AssertionError(f'{what}: expected dims {tuple(expected)}, got {dims} '
                             f'(shape {da.shape}) -- a staggered/centred mismatch has '
                             'broadcast')
    return da


def _centred_dims_like(dims):
    """The centred dims a staggered field's output should land on."""
    return tuple({X_STAG: 'i', Y_STAG: 'j'}.get(d, d) for d in dims)


# ---------------------------------------------------------------------------
# buoyancy
# ---------------------------------------------------------------------------
def buoyancy(ds) -> xr.DataArray:
    """``b = g sigma0 / rho0`` [m s^-2] from ``Theta`` and ``Salt``, JMD95 at
    p = 0 -- the equation of state the model advected (planning §5.1), via
    ``calculate_fields.buoyancy_of_field`` with ``g = 9.81``, ``rho0 = 1000``.

    The sign is the repo's (``+g sigma0/rho0``, increasing with density) and
    is deliberately left alone.  Computed eagerly in float64; dims equal
    ``Theta``'s.  Land stays NaN.
    """
    for v in ('Theta', 'Salt'):
        if v not in ds:
            raise KeyError(f'buoyancy: dataset has no {v}')
    require_centred(ds['Theta'], 'Theta')
    ts = ds[['Theta', 'Salt']].astype('float64')      # §1.2: float64 on the compute path
    b = buoyancy_of_field(ts).compute().astype('float64')
    assert_dims(b, ds['Theta'].dims, 'buoyancy_of_field')
    b.name = 'b'
    b.attrs.update(units='m s-2',
                   long_name='buoyancy b = +g sigma0/rho0 (JMD95, p=0); increases with density')
    return b


# ---------------------------------------------------------------------------
# low-pass filter
# ---------------------------------------------------------------------------
def lowpass(field, L_cells: int):
    """Separable top-hat (box) filter of scale ``L_cells`` in index space;
    ``L_cells = 0`` is the identity.  Returns the same type as ``field``
    (``xarray.DataArray`` or ``numpy.ndarray``), float64.

    Kernel.  Weights ``1/(L+1)`` on ``|k| <= L/2`` along each horizontal
    axis, i.e. half-width ``L/2`` and full support ``L + 1`` cells (``L = 8``
    is the "widest filter half-width 4" that sizes the 7-cell land halo,
    coding §4.2).  ``L_cells`` must be even.  The kernel is normalised, so
    constants pass unchanged, and *shift-invariant*, so it commutes exactly
    with the discrete diff/interp stencils wherever no NaN is in reach --
    which is what makes the coarse-grained budget (planning §5.4) and the
    Germano identity hold on the grid.  A sinusoid of wavelength ``L + 1``
    cells is annihilated exactly.

    Applied along whichever two of ``j``/``j_g`` and ``i``/``i_g`` the field
    carries, so the *same* kernel filters ``b`` (centres), ``U`` (``i_g``)
    and ``V`` (``j_g``) -- §1.2's "same filter on ``b``, ``U`` and ``V``".
    A numpy array is filtered along its last two axes.

    NaN policy: **propagate, never renormalise.**  A cell whose ``(L+1)^2``
    footprint touches a NaN (land, a stencil rim, or the tile edge, which is
    padded with NaN) becomes NaN.  Renormalising over the valid part of the
    footprint would return finite values near the coast that are averages
    over a truncated, cell-dependent kernel -- a different filter at every
    coastal cell, which would break the commutation above and would make
    ``lowpass`` the only thing in the pipeline that turns a land neighbour
    into a finite number.  With propagation the land halo never has to
    *hide* a contaminated value; it only sizes the filter support, so its
    Euclidean 7 cells (chessboard reach 5, M1 task 1) stay adequate: the
    worst case is a few more NaN cells inside ``mask_halo`` at ``L = 8``,
    which NaN-aware reductions drop.  Validity is ``isfinite(field)``, not
    the halo.
    """
    L = int(L_cells)
    if L != L_cells or L < 0:
        raise ValueError(f'lowpass: L_cells must be a non-negative integer, got {L_cells!r}')
    if L == 0:
        return field                                   # identity, by contract
    if L % 2:
        raise ValueError(f'lowpass: L_cells must be even (half-width L/2), got {L}')
    w = np.full(L + 1, 1.0 / (L + 1))                  # normalised top-hat, half-width L/2

    def box(a, axes):
        a = np.asarray(a, dtype='float64')
        for ax in axes:
            # a direct (not running-sum) convolution: NaN reaches only the
            # outputs whose window contains it; the tile edge is padded NaN
            a = ndimage.convolve1d(a, w, axis=ax, mode='constant', cval=np.nan)
        return a

    if isinstance(field, xr.DataArray):
        axes = [field.get_axis_num(d) for d in field.dims if d in HORIZONTAL]
        if len(axes) != 2:
            raise ValueError(f'lowpass: need exactly one X and one Y dim among '
                             f'{HORIZONTAL}, got dims {field.dims}')
        out = field.copy(data=box(field.values, axes))
        out.attrs['lowpass_L_cells'] = L
        return out
    a = np.asarray(field)
    if a.ndim < 2:
        raise ValueError(f'lowpass: need a 2-D+ array, got ndim {a.ndim}')
    return box(a, [a.ndim - 2, a.ndim - 1])


# ---------------------------------------------------------------------------
# gradients and front strength
# ---------------------------------------------------------------------------
def grad_b(b, grid_ds, grid):
    """Geographic ``(b_x, b_y)`` [s^-2] at cell centres via
    ``calculate_native_gradient_tracer``: difference to the staggered
    points, divide by ``dxC``/``dyC``, interpolate back to centres, rotate
    with ``CS``/``SN``.  NaN one centre from any NaN (M0 task 5)."""
    dims = require_centred(b, 'b')
    bx, by = ng.calculate_native_gradient_tracer(b, grid_ds, grid)
    bx = assert_dims(bx.compute(), dims, 'calculate_native_gradient_tracer[0]')
    by = assert_dims(by.compute(), dims, 'calculate_native_gradient_tracer[1]')
    bx.name, by.name = 'b_x', 'b_y'
    for da, ln in ((bx, 'zonal'), (by, 'meridional')):
        da.attrs.clear()
        da.attrs.update(units='s-2', long_name=f'{ln} buoyancy gradient (geographic)')
    return bx, by


def gradb2(b, grid_ds, grid):
    """Front strength ``G = b_x^2 + b_y^2`` [s^-4] from :func:`grad_b` --
    the *component* stencil, the same ``b_x, b_y`` that enter ``F``.  Not
    ``calculate_grad_squared_tracer`` (§1.1; that one is for front finding
    only)."""
    bx, by = grad_b(b, grid_ds, grid)
    G = bx ** 2 + by ** 2
    assert_dims(G, b.dims, 'gradb2')
    G.name = 'G'
    G.attrs.update(units='s-4', long_name='|grad_h b|^2 = b_x^2 + b_y^2 (component stencil)')
    return G


# ---------------------------------------------------------------------------
# velocity gradients
# ---------------------------------------------------------------------------
def jacobian(U, V, grid_ds, grid):
    """Geographic velocity-gradient tensor ``(u_x, u_y, v_x, v_y)`` [s^-1]
    at cell centres via ``calculate_jacobian`` (ECCO recipe: interpolate the
    staggered pair to centres, rotate, difference, interpolate, rotate).

    ``U`` must be the raw model-x velocity on ``(j, i_g)`` and ``V`` the raw
    model-y velocity on ``(j_g, i)`` -- the helper's parameters are named
    ``u_x, v_y`` but that is what they are (M0 task 5, bit-for-bit against a
    numpy replica).  The staggering is checked *before* the call: swapped
    arguments broadcast to 4-D+ inside the helper (OOM on the tile).  The
    interpolated stencil attenuates the trace ~0.80x relative to the
    flux-form divergence (M0 task 5); V3 measures that.
    """
    udims = require_u_point(U, 'U')
    require_v_point(V, 'V')
    out_dims = _centred_dims_like(udims)
    J = ng.calculate_jacobian(U, V, grid_ds, grid)
    names = ('u_x', 'u_y', 'v_x', 'v_y')
    out = []
    for k, (da, name) in enumerate(zip(J, names)):
        da = assert_dims(da.compute(), out_dims, f'calculate_jacobian[{k}]')
        da.name = name
        da.attrs.clear()
        da.attrs.update(units='s-1', long_name=f'{name} (geographic, centre-interpolated)')
        out.append(da)
    return tuple(out)


# ---------------------------------------------------------------------------
# frontogenesis
# ---------------------------------------------------------------------------
def frontogenesis(b, U, V, grid_ds, grid):
    """Kinematic frontogenesis function
    ``F = -(u_x b_x^2 + (u_y + v_x) b_x b_y + v_y b_y^2)`` [s^-5] at cell
    centres, from :func:`grad_b` and :func:`jacobian`.

    ``F = (1/2) DG/Dt`` in the adiabatic limit: the comparison is always
    ``2F`` against the measured ``DG/Dt`` (name it ``two_F``, coding §1.1).
    Inputs are **already filtered** (or not) -- this function never filters.
    ``b`` on centres, ``U``/``V`` raw staggered.  NaN within 2 cells of land
    (the Jacobian's reach).
    """
    bdims = require_centred(b, 'b')
    bx, by = grad_b(b, grid_ds, grid)
    ux, uy, vx, vy = jacobian(U, V, grid_ds, grid)
    if ux.dims != bdims:
        raise AssertionError(f'frontogenesis: Jacobian dims {ux.dims} != b dims {bdims}')
    # the strain acting on the gradient: -(grad b)^T (grad u) (grad b).
    # u_x compresses/extends along x, v_y along y, u_y + v_x shears --
    # the same three combinations as delta, sigma_n, sigma_s below.
    F = -(ux * bx ** 2 + (uy + vx) * bx * by + vy * by ** 2)
    assert_dims(F, bdims, 'frontogenesis')
    F.name = 'F'
    F.attrs.update(units='s-5',
                   long_name='kinematic frontogenesis '
                             'F = -(u_x b_x^2 + (u_y+v_x) b_x b_y + v_y b_y^2)',
                   convention='F = (1/2) DG/Dt: compare 2F with the measured DG/Dt')
    return F


# ---------------------------------------------------------------------------
# strain, divergence, alignment
# ---------------------------------------------------------------------------
def _corner_to_centre(q, grid, what):
    """Four-point mean of a corner ``(j_g, i_g)`` field onto the centres:
    interpolate along X then Y (``padding='fill'``).  Signed fields are fine
    here -- this is a plain bilinear average, not a squared-quantity trick."""
    require = _require(q, what, (Y_STAG, X_STAG), ('j', 'i'))
    out = grid.interp(grid.interp(q, 'X', padding='fill'), 'Y', padding='fill')
    return assert_dims(out, _centred_dims_like(require), f'{what} -> centres')


def strain_divergence(U, V, grid_ds, grid):
    """Flux-form ``(delta, sigma_n, sigma_s, sigma_mag)`` [s^-1] at cell
    centres from ``calculate_native_strain_vorticity`` (each quantity
    differenced at its natural C-grid point, no interpolation before the
    difference), then

    * ``strain_shear_corner`` (on ``(j_g, i_g)``) is averaged to the centres
      (``vorticity_corner`` is not needed here);
    * the strain pair, which the helper leaves in the **model** basis, is
      rotated to geographic so that it shares a basis with :func:`grad_b`.
      ``(sigma_n, sigma_s)`` is a symmetric traceless tensor and rotates by
      ``2 alpha`` (``cos alpha = CS``, ``sin alpha = SN``):
      ``sigma_n' = cos2a sigma_n - sin2a sigma_s``,
      ``sigma_s' = sin2a sigma_n + cos2a sigma_s``.  On face 10
      (``CS = 0``, ``SN = -1``) that is a sign flip of both.

    ``delta = u_x + v_y`` and ``|sigma| = sqrt(sigma_n^2 + sigma_s^2)`` are
    invariants.  ``delta`` here is the flux-form divergence the interpolated
    Jacobian trace is 0.80x of (M0 task 5); :func:`strain_from_jacobian`
    gives the Jacobian-consistent set that decomposes ``F`` exactly.
    """
    udims = require_u_point(U, 'U')
    require_v_point(V, 'V')
    cdims = _centred_dims_like(udims)
    sv = ng.calculate_native_strain_vorticity(U, V, grid_ds, grid)
    if not isinstance(sv, dict):
        raise TypeError('calculate_native_strain_vorticity no longer returns a dict')
    delta = assert_dims(sv['divergence_center'].compute(), cdims, 'divergence_center')
    sn_m = assert_dims(sv['strain_normal_center'].compute(), cdims, 'strain_normal_center')
    ss_m = _corner_to_centre(sv['strain_shear_corner'], grid, 'strain_shear_corner').compute()
    cs, sn = grid_ds['CS'], grid_ds['SN']
    cos2a, sin2a = cs ** 2 - sn ** 2, 2.0 * cs * sn
    sigma_n = assert_dims(cos2a * sn_m - sin2a * ss_m, cdims, 'sigma_n')
    sigma_s = assert_dims(sin2a * sn_m + cos2a * ss_m, cdims, 'sigma_s')
    sigma_mag = np.sqrt(sigma_n ** 2 + sigma_s ** 2)
    out = (delta, sigma_n, sigma_s, sigma_mag)
    for da, name, ln in zip(out, ('delta', 'sigma_n', 'sigma_s', 'sigma_mag'),
                            ('divergence u_x + v_y (flux form)',
                             'normal strain u_x - v_y (flux form, geographic)',
                             'shear strain v_x + u_y (flux form, corners averaged to '
                             'centres, geographic)',
                             '|sigma| = sqrt(sigma_n^2 + sigma_s^2)')):
        da.name = name
        da.attrs.clear()
        da.attrs.update(units='s-1', long_name=ln)
    return out


def strain_from_jacobian(u_x, u_y, v_x, v_y):
    """``(delta, sigma_n, sigma_s, sigma_mag)`` from :func:`jacobian`'s
    components -- the set that decomposes :func:`frontogenesis` *exactly*
    on the grid (``F = -(1/2) delta G + (1/2) |sigma| G cos 2 theta``), used
    to check :func:`strain_alignment` and for the V3 diagnostics."""
    delta = u_x + v_y
    sigma_n = u_x - v_y
    sigma_s = v_x + u_y
    return delta, sigma_n, sigma_s, np.sqrt(sigma_n ** 2 + sigma_s ** 2)


def strain_alignment(b_x, b_y, sigma_n, sigma_s):
    """Angle ``theta`` [rad, in ``[0, pi/2]``] between ``grad_h b`` and the
    **compressional** axis of the strain (Figure 4).  All four inputs in the
    same (geographic) basis.

    Writing ``sigma_n = |sigma| cos 2 psi``, ``sigma_s = |sigma| sin 2 psi``
    makes ``psi`` the extensional axis; the compressional axis is
    ``psi + pi/2``, and with ``alpha`` the direction of ``grad b``,
    ``cos 2 theta = -cos 2(alpha - psi)
                  = -(sigma_n (b_x^2 - b_y^2) + 2 sigma_s b_x b_y) / (|sigma| G)``.
    Then ``F = -(1/2) delta G + (1/2) |sigma| G cos 2 theta``: ``theta = 0``
    (gradient along the compressional axis) is maximal frontogenesis.
    Note the **plus** sign on the strain term with this ``theta``: planning
    §2.4 writes a minus, which holds only if ``theta`` is measured from the
    *extensional* axis.  ``cos 2 theta`` is invariant under the sign of
    ``b`` and of the axis, so folding to ``[0, pi/2]`` loses nothing.
    NaN where ``|sigma| = 0`` or ``G = 0`` (the angle is undefined).
    """
    G = b_x ** 2 + b_y ** 2
    mag = np.sqrt(sigma_n ** 2 + sigma_s ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = -(sigma_n * (b_x ** 2 - b_y ** 2) + 2.0 * sigma_s * b_x * b_y) / (mag * G)
        theta = 0.5 * np.arccos(np.clip(c, -1.0, 1.0))
    if isinstance(theta, xr.DataArray):
        theta.name = 'theta_align'
        theta.attrs.clear()
        theta.attrs.update(units='rad', long_name='angle between grad b and the strain '
                           'compressional axis, folded to [0, pi/2]')
    return theta
