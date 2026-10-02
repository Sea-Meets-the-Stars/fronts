""" Coarse-graining (coding doc §4.5, planning §5.4): the subfilter buoyancy
flux ``tau = mean(u b) - ubar bbar`` and the term it adds to the budget of
the *filtered* front strength ``Gbar = |grad bbar|^2``.

Why this module is M1's.  The filter sweep (planning §5.4) is a change of
budget, not noise control: filtering ``b`` and ``u`` identically and then
comparing ``Dbar Gbar/Dt`` with ``2 F(ubar, bbar)`` leaves out the scale
transfer between the resolved and the subfilter part of the flux.  That
term is ``O(1)`` relative to ``F`` at every ``L`` and, being a third
derivative of the flux, lives at the filter cutoff -- exactly where
``Gbar`` has its variance.  Demonstrating that it closes the coarse-grained
budget on a synthetic field is part of validating the operators
(``tests/test_coarsegrain.py``).

Derivation: the factor, the sign and the units
-----------------------------------------------
Overbar = ``operators.lowpass`` (linear, normalised, shift-invariant, so it
commutes with the discrete gradient wherever it is finite -- M1 task 2).
Surface tracer equation, horizontal form (planning §2.2):

    d_t b + u . grad b = B                                  (1)

Filter (1) and write the advection as resolved + subfilter,

    d_t bbar + ubar . grad bbar = Bbar - sigma,
    sigma := mean(u . grad b) - ubar . grad bbar.           (2)

Take ``grad`` of (2) and dot with ``grad bbar``; with
``-grad bbar . grad(ubar . grad bbar) = F(ubar, bbar) - ubar . grad(Gbar/2)``
(the identity behind ``F = (1/2) DG/Dt``),

    Dbar/Dt (Gbar/2) = F(ubar, bbar) + grad bbar . grad Bbar
                       - grad bbar . grad sigma.            (3)

So the subfilter term in **F units** (``s^-5``, the units of
``D/Dt (G/2)``) is ``T = -grad bbar . grad sigma``, and in the ``DG/Dt``
budget the study actually compares (coding §1.1: ``two_F`` vs ``DGDt``)

    Dbar Gbar/Dt = 2 F(ubar, bbar) + 2 T + 2 grad bbar . grad Bbar.   (4)

:func:`subfilter_term` returns ``T`` (F units), matching planning §5.4's
equation, which is written for ``Dbar/Dt (Gbar/2)``.  **M3's budget field
``subfilter`` must be ``2 * subfilter_term``** to sit beside ``two_F`` and
the measured ``DGDt``.  The sign: ``sigma`` enters (2) as a sink of
``bbar``, so ``T = -grad bbar . grad sigma`` -- positive where the
subfilter advection *sharpens* the resolved gradient.

The flux form.  ``u . grad b = div(u b) - b delta`` with ``delta = div u``.
Filtering both sides and subtracting the resolved version,

    sigma = div tau - tau_delta,
    tau       = mean(u b) - ubar bbar          (subfilter buoyancy flux)
    tau_delta = mean(b delta) - bbar deltabar  (subfilter dilatation part)

For a non-divergent flow ``tau_delta = 0`` and ``T = -grad bbar . grad(div
tau)``, the form in planning §5.4 and coding §4.5.  **The LLC4320 surface
flow is divergent** -- ``delta`` is 0.1-0.5 f at fronts (planning §2.2) and
the subfilter part of ``b delta`` is of the same order as ``div tau``
(``tau_delta ~ b' delta'`` against ``div tau ~ u' b'/L``), so the flux
form alone does not close the surface budget.  :func:`subfilter_bdelta`
supplies ``tau_delta`` and :func:`subfilter_term` takes it as an optional
argument; the closure test shows the miss without it on a divergent flow.

Where ``u b`` is formed on the C-grid (decision)
------------------------------------------------
On the **staggered velocity points, flux form, model basis**: ``b`` is
averaged to the U point ``(j, i_g)`` and the V point ``(j_g, i)`` (the two
point mean, the same interpolation ``calculate_native_gradient_tracer``
uses in the other direction), the products ``U b_u`` and ``V b_v`` are
filtered *there* with the same kernel that filters ``U`` and ``V``
(``lowpass`` acts on ``i_g``/``j_g`` as it does on ``i``/``j``), and

    tau_x = mean(U b_u) - Ubar mean(b_u),   on (j, i_g)
    tau_y = mean(V b_v) - Vbar mean(b_v),   on (j_g, i)

(``mean(b_u) = interp(bbar)`` exactly, by commutation).  ``div tau`` is
then the model's own flux-form divergence at the centres,
``(Delta_X(tau_x dyG) + Delta_Y(tau_y dxG)) / rA`` -- the operator behind
``calculate_native_strain_vorticity``'s ``divergence_center`` -- with no
interpolation of the flux, and ``delta`` for ``tau_delta`` is that same
divergence of ``(U, V)``.  Reasons: (i) this is how the model advects:
``U b_face`` on the face is its tracer flux (OS7MP reconstructs the face
value at higher order, but the *resolved* flux of the filtered budget is
the second-order one); (ii) the subfilter flux is defined on the same
points as the velocity it belongs to, so "the same filter on ``b``, ``U``
and ``V``" (coding §1.2) is literal; (iii) ``div tau`` reaches the centres
in one difference, whereas a centred ``tau`` would need diff-interp and an
extra half-cell attenuation of the third derivative that the term is.
The remaining steps use the shared operators: ``grad sigma`` and
``grad bbar`` both from ``operators.grad_b`` (geographic; the dot product
is an invariant, so ``tau`` may stay in the model basis).  The alternative
-- everything at centres from ``interp_pair_to_center(U, V)`` -- differs
by the ``O(dx^2)`` product-rule mismatch between the flux and advective
forms; the closure test measures that error together with the
chain-rule violation V3 is about.

NaN: as ``lowpass`` -- propagate, never fill.  ``b_u`` is NaN on the
coast-facing faces (where ``U`` is NaN in the OSN stores anyway) and on
the first face along each axis (no low neighbour in the tile; xgcm would
pad with 0); ``div`` is NaN at the last centre along each axis (no high
face).  Reach of ``T`` from land: ``L/2 + 2`` cells (measured in the tests).
Dims are asserted before and after every dbof/xgcm call.
"""

import numpy as np
import xarray as xr

import operators as op
from operators import (require_centred, require_u_point, require_v_point, assert_dims,
                       _centred_dims_like)


# ---------------------------------------------------------------------------
# the filter, possibly composite
# ---------------------------------------------------------------------------
def filt(field, L_cells):
    """``operators.lowpass`` at scale ``L_cells``; a sequence of scales is
    applied in turn (the *composite* filter).  The Germano identity relates
    the subfilter flux at two levels through their composition, and the
    composition of two top-hats is a trapezoid, not a wider top-hat, so it
    has to be expressed this way rather than as a single ``L``."""
    if np.isscalar(L_cells):
        return op.lowpass(field, L_cells)
    for L in L_cells:
        field = op.lowpass(field, L)
    return field


# ---------------------------------------------------------------------------
# C-grid pieces: b on the velocity points, the flux-form divergence
# ---------------------------------------------------------------------------
def b_at_velocity_points(b, grid):
    """``(b_u, b_v)``: the tracer averaged onto the U point ``(j, i_g)`` and
    the V point ``(j_g, i)`` (two-point mean).  The first face along each
    axis has no low neighbour in the tile and is set to NaN (xgcm's
    ``padding='fill'`` would average with 0 -- the finite tile-edge rim of
    M0 task 5); a face touching a NaN cell is NaN."""
    dims = require_centred(b, 'b')
    b_u = grid.interp(b, 'X', padding='fill').compute().copy()
    b_v = grid.interp(b, 'Y', padding='fill').compute().copy()
    b_u = assert_dims(b_u, tuple('i_g' if d == 'i' else d for d in dims), 'interp(b, X)')
    b_v = assert_dims(b_v, tuple('j_g' if d == 'j' else d for d in dims), 'interp(b, Y)')
    b_u[{'i_g': 0}] = np.nan
    b_v[{'j_g': 0}] = np.nan
    b_u.name, b_v.name = 'b_u', 'b_v'
    return b_u, b_v


def flux_divergence(fx, fy, grid_ds, grid):
    """Flux-form divergence at the centres of a staggered pair
    ``(fx on (j, i_g), fy on (j_g, i))``:
    ``(Delta_X(fx dyG) + Delta_Y(fy dxG)) / rA`` -- the model's divergence
    operator, and ``calculate_native_strain_vorticity``'s
    ``divergence_center`` when ``(fx, fy) = (U, V)``.  Model basis, no
    interpolation.  The last centre along each axis (no high face in the
    tile) is NaN."""
    udims = require_u_point(fx, 'fx')
    require_v_point(fy, 'fy')
    cdims = _centred_dims_like(udims)
    dX = grid.diff(fx * grid_ds['dyG'], 'X', padding='fill')
    dY = grid.diff(fy * grid_ds['dxG'], 'Y', padding='fill')
    div = ((dX + dY) / grid_ds['rA']).compute().copy()
    div = assert_dims(div, cdims, 'flux_divergence')
    div[{'i': -1}] = np.nan
    div[{'j': -1}] = np.nan
    return div


# ---------------------------------------------------------------------------
# the subfilter flux and its dilatation partner
# ---------------------------------------------------------------------------
def subfilter_flux(b, U, V, L_cells, grid_ds, grid):
    """``(tau_x, tau_y) = mean(u b) - ubar bbar`` [m^2 s^-3] on the U and V
    points (model basis), from the **unfiltered** ``b`` (centres), ``U``
    ``(j, i_g)``, ``V`` ``(j_g, i)``; the filtering happens here, once, with
    ``filt`` at ``L_cells`` (an int, or a sequence for the composite filter).
    ``L_cells = 0`` gives exactly 0 (``lowpass`` is the identity).  For
    smooth fields ``tau ~ M2 (grad u . grad b)`` with ``M2 = L (L + 2)
    dx^2 / 12`` the kernel's second moment (the Clark / gradient model) --
    ``O(L^2)`` as ``L -> 0``.
    """
    require_centred(b, 'b')
    udims = require_u_point(U, 'U')
    vdims = require_v_point(V, 'V')
    b_u, b_v = b_at_velocity_points(b, grid)
    # the flux where the model carries it: velocity times the face value;
    # filtered on the same points, with the same kernel as the velocity
    tau_x = filt(U * b_u, L_cells) - filt(U, L_cells) * filt(b_u, L_cells)
    tau_y = filt(V * b_v, L_cells) - filt(V, L_cells) * filt(b_v, L_cells)
    tau_x = assert_dims(tau_x, udims, 'subfilter_flux[0]')
    tau_y = assert_dims(tau_y, vdims, 'subfilter_flux[1]')
    tau_x.name, tau_y.name = 'tau_x', 'tau_y'
    for da, ax in ((tau_x, 'x'), (tau_y, 'y')):
        da.attrs.clear()
        da.attrs.update(units='m2 s-3', L_cells=str(L_cells),
                        long_name=f'subfilter buoyancy flux, model-{ax}: '
                                  'mean(u b) - ubar bbar on the velocity point')
    return tau_x, tau_y


def subfilter_bdelta(b, U, V, L_cells, grid_ds, grid):
    """``tau_delta = mean(b delta) - bbar deltabar`` [m s^-3] at the centres,
    with ``delta = div(U, V)`` the flux-form divergence -- the part of the
    subfilter advection that the flux form misses when the flow is
    divergent (``sigma = div tau - tau_delta``; module docstring).  Zero
    for a non-divergent flow, and for a *uniform* divergence."""
    dims = require_centred(b, 'b')
    delta = flux_divergence(U, V, grid_ds, grid)
    if tuple(delta.dims) != dims:
        raise AssertionError(f'subfilter_bdelta: delta dims {delta.dims} != b dims {dims}')
    td = filt(b * delta, L_cells) - filt(b, L_cells) * filt(delta, L_cells)
    td = assert_dims(td, dims, 'subfilter_bdelta')
    td.name = 'tau_delta'
    td.attrs.clear()
    td.attrs.update(units='m s-3', L_cells=str(L_cells),
                    long_name='subfilter dilatation part: mean(b div u) - bbar div ubar')
    return td


# ---------------------------------------------------------------------------
# the subfilter advection and the budget term
# ---------------------------------------------------------------------------
def subfilter_advection(tau_x, tau_y, grid_ds, grid, tau_delta=None):
    """``sigma = div tau - tau_delta`` [m s^-3] at the centres: the subfilter
    advective tendency of ``bbar`` (``d_t bbar + ubar . grad bbar = Bbar -
    sigma``).  Without ``tau_delta`` it is the non-divergent form."""
    div = flux_divergence(tau_x, tau_y, grid_ds, grid)
    if tau_delta is not None:
        if tuple(tau_delta.dims) != tuple(div.dims):
            raise AssertionError(f'subfilter_advection: tau_delta dims {tau_delta.dims} != '
                                 f'{div.dims}')
        div = div - tau_delta
    div.name = 'sigma'
    div.attrs.clear()
    div.attrs.update(units='m s-3', long_name='subfilter advection sigma = div tau'
                     + (' - tau_delta' if tau_delta is not None else ' (non-divergent form)'))
    return div


def subfilter_term(b_bar, tau_x, tau_y, grid_ds, grid, tau_delta=None):
    """``T = -grad bbar . grad sigma`` [s^-5], **F units**: the subfilter
    term of ``Dbar/Dt (Gbar/2) = F(ubar, bbar) + T + ...`` (planning §5.4).
    For the ``DG/Dt`` budget use ``2 * T`` beside ``two_F``.

    ``b_bar`` is the **filtered** buoyancy at the centres (the caller
    filters; nothing is filtered here), ``tau_x``, ``tau_y`` from
    :func:`subfilter_flux`, ``tau_delta`` from :func:`subfilter_bdelta`
    (required for a divergent flow; omitted, the non-divergent
    ``-grad bbar . grad(div tau)`` of coding §4.5 is returned).  Both
    gradients are ``operators.grad_b`` (geographic; the dot product is
    invariant).  NaN ``L/2 + 2`` cells from land or the tile edge.
    """
    dims = require_centred(b_bar, 'b_bar')
    sigma = subfilter_advection(tau_x, tau_y, grid_ds, grid, tau_delta)
    if tuple(sigma.dims) != dims:
        raise AssertionError(f'subfilter_term: sigma dims {sigma.dims} != b_bar dims {dims}')
    sx, sy = op.grad_b(sigma, grid_ds, grid)
    bx, by = op.grad_b(b_bar, grid_ds, grid)
    # sigma is a sink of bbar in (2): its gradient along grad bbar weakens
    # the resolved front, hence the minus -- the same structure as
    # grad b . grad B for a diabatic source, with the opposite sign
    term = -(bx * sx + by * sy)
    term = assert_dims(term, dims, 'subfilter_term')
    term.name = 'subfilter_term'
    term.attrs.clear()
    term.attrs.update(units='s-5',
                      long_name='subfilter term -grad bbar . grad sigma of the coarse-grained '
                                'budget (F units)',
                      convention='D/Dt(Gbar/2) = F + term + ...; the DG/Dt budget uses 2*term '
                                 'beside two_F',
                      form='div tau - tau_delta' if tau_delta is not None
                           else 'div tau only (non-divergent form)')
    return term
