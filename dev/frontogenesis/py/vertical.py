""" The extra budget terms from the chunk store (coding §4.6, Q13): the
top-cell stratification ``b_z``, the **vertical** (finite-top-cell tilting)
term and the **surface-flux** (diabatic) term of the surface budget

    D_h/Dt ( G/2 ) = F + vertical_term + surface_flux_term + subfilter + residual

Every function here returns **F units** (s^-5), exactly as
``coarsegrain.subfilter_term`` does; the §3.4 budget fields are ``2 x``
these (``vertical = 2 * vertical_term``, ``surface_flux = 2 *
surface_flux_term``), formed in ``budget.py`` (task 3), never here.

The chunk-store **reader** (``load_chunk_levels`` and its helpers, M2 task 5)
lives in ``chunk_store.py`` since M3 task 2 (decided 2026-10-07, M3-Q7 (a));
``load_chunk_levels`` and the path constants are re-exported below so
``m2_chunk_pull.py`` keeps working.

Physics (planning §2.2; M0 task 3)
----------------------------------
From ``Db/Dt = B`` with ``w`` the vertical velocity, the *horizontal*
material derivative of ``b`` is ``D_h b/Dt = B - w b_z``; taking ``grad_h``
and dotting with ``grad_h b`` gives

    D_h/Dt (G/2) = F + grad_h b . grad_h( -w b_z ) + grad_h b . grad_h B .

In the continuum the tilting term vanishes *at* the free surface (the
velocity relative to the surface does), but the stored ``b`` is the
average of a fixed 1 m box (``drF[0] = 1.0 m``; LLC4320 has a linear free
surface) whose budget carries the advective flux through its **base**,
where the model's ``W(k_l=1)`` is ``dEta/dt + drF delta`` -- so the term is
real, diurnal (a warm layer by day, ~0 at night) and is built from the
chunk ``W(k_l=1)``, never from ``delta`` (wrong sign, missing part) and
never from the OSN ``W(k_l=0) = dEta/dt``.  The tendency ``T_v = -W_k1 b_z``
is formed **first** and then differenced, so that ``-w grad(b_z) . grad b``
is kept; the factorised ``-b_z (w_x b_x + w_y b_y)`` is a diagnostic only.

Sign of ``b``: code ``b = +g sigma0/rho0`` **increases with density**
(coding §1.1), so a warm layer over cooler water has ``b_z < 0`` here
(planning §2.2's textbook ``b_z ~ +2-4e-4 s^-2`` is the same state) and a
heat flux *into* the ocean **lowers** ``b``.  Everything below is written
in code ``b``; the dot products are even in the sign of ``b``, so the
terms are what the budget needs either way.

Filtering at ``L``: the tendencies ``T_v`` and ``B_sfc`` are low-passed
**themselves** with ``operators.lowpass`` (the kernel that filters ``b, U,
V``) and dotted with the filtered ``grad bbar``, so the subfilter
correlation ``mean(w b_z) - wbar bzbar`` sits inside the term instead of
being dropped (prompt 4 task 2).  Land and the ``W``/``b_k1`` NaN propagate.

Namelist constants (verified 2026-10-07 against ``MITgcm_contrib/llc_hires/
llc_4320/input/{data,data.pkg,data.kpp,data.exf}`` and ``code/CPP_OPTIONS.h``
on GitHub master, the files M0 task 3 cites; the production-era commit
differs only in the Leith coefficient): ``rhonil = 1027.5`` with ``rhoConst``
absent, so ``rhoConst = rhoNil = 1027.5`` (the density the model divides the
fluxes by); ``HeatCapacity_Cp`` absent -> the default **3994**;
``convertFW2Salt = -1.`` (the *local* salinity converts the fresh-water
flux) with ``useRealFreshWaterFlux = .TRUE.`` and a linear free surface
(``nonlinFreeSurf`` unset); ``eosType = 'JMD95Z'``; ``#define
SHORTWAVE_HEATING`` with no ``data.kpp``/``data.exf`` override of the
penetration, so the shortwave profile is ``model/src/swfrac.F``
(checkpoint65v), which **hard-codes Jerlov type IA** (``jwtype = 2``:
``0.62 exp(z/0.6 m) + 0.38 exp(z/20 m)``), not type I -- ``f_sw = 0.521`` of
``oceQsw`` is absorbed inside the 1 m top cell, not the 0.56 the prompt
assumed (M3-Q10).
"""

import numpy as np
import xarray as xr

import operators as op
# the reader, re-exported for m2_chunk_pull.py and the M2 tests (M3-Q7)
from chunk_store import (load_chunk_levels, CHUNK_ZARR, OSN_RAW_ZARR, CHUNK_ENDPOINT,  # noqa: F401
                         CHUNK_PREFIX, DATA_DIR, _store_name)
from dbof.preprocessing.physical_constants import G, RHO0_REFERENCE
import dbof.utils.jmd95_xgcm_implementation as jmd95

# --- the model's constants (LLC4320 namelist; see the module docstring) -----
HEAT_CAPACITY_CP = 3994.0     # J kg^-1 K^-1; MITgcm HeatCapacity_Cp default (absent from data)
RHO_CONST = 1027.5            # kg m^-3; rhoConst = rhoNil (data: rhonil=1027.5; rhoConst absent)
CONVERT_FW2SALT = -1.0        # data: convertFW2Salt=-1. -> the local salinity dilutes
# swfrac.F (checkpoint65v): swdk(z) = rfac exp(z/a1) + (1 - rfac) exp(z/a2), z <= 0 in m;
# Jerlov types I, IA, IB, II, III = 1..5; jwtype = 2 is hard-coded in the routine
JERLOV = {1: (0.58, 0.35, 23.0), 2: (0.62, 0.6, 20.0), 3: (0.67, 1.0, 17.0),
          4: (0.77, 1.5, 14.0), 5: (0.78, 1.4, 7.9)}
JWTYPE = 2
NAMELIST_SOURCE = ('MITgcm_contrib/llc_hires/llc_4320/input/data (+data.pkg, data.kpp, data.exf, '
                   'code/CPP_OPTIONS.h), GitHub master, read 2026-10-07; swfrac.F at checkpoint65v')
CONVENTION = ('F units (s^-5): D_h/Dt(G/2) = F + vertical_term + surface_flux_term + ...; the '
              '§3.4 budget fields are 2 x these (budget.py, task 3), as subfilter = 2 * subfilter_term')


def sw_fraction_absorbed(dz, jwtype: int = JWTYPE):
    """The fraction of ``oceQsw`` absorbed between the surface and depth
    ``dz`` (m), ``1 - swdk(-dz)`` with MITgcm's two-band Paulson-Simpson
    profile for Jerlov water type ``jwtype`` (``swfrac.F``).  For the 1 m
    top cell and the model's type IA this is **0.521** (type I would give
    0.565; the two-band fit is not meant below ~1 m accuracy, so the ~8 %
    between the types is the honest uncertainty of ``f_sw``).

    ``dz`` may be an array (M3-Q13 (c): the mixed-layer sensitivity needs
    ``f_sw`` at ``KPPhbl``, where it is ~1 -- essentially all the shortwave
    is absorbed within 20 m); a scalar returns a float, as before."""
    rfac, a1, a2 = JERLOV[int(jwtype)]
    scalar = np.isscalar(dz) or np.ndim(dz) == 0
    z = -np.abs(np.asarray(getattr(dz, 'values', dz), dtype='float64'))
    f = 1.0 - (rfac * np.exp(z / a1) + (1.0 - rfac) * np.exp(z / a2))
    return float(f) if scalar else (dz.copy(data=f) if hasattr(dz, 'dims') else f)


F_SW = sw_fraction_absorbed(1.0)      # 0.5214 for drF[0] = 1.0 m, Jerlov IA


# ---------------------------------------------------------------------------
# small guards and level geometry
# ---------------------------------------------------------------------------
def _drF(drF):
    v = np.asarray(getattr(drF, 'values', drF), dtype='float64').reshape(-1)
    if v.size < 2 or not np.all(v > 0):
        raise ValueError(f'drF must hold at least the two top cell thicknesses, got {v}')
    return v


def level_depths(drF):
    """Cell-centre depths ``Z_k = -(sum_{k' <= k} drF - drF_k/2)`` from the
    thicknesses: ``[-0.5, -1.57, -2.79]`` for ``[1.0, 1.14, 1.30]`` (the
    store's ``Z``, M2 task 4)."""
    v = _drF(drF)
    return -(np.cumsum(v) - 0.5 * v)


def _check_no_level_dim(da, what):
    """A field that must already be a single level: a 3-D ``W`` (``k_l``)
    or ``Theta_k`` (``k``) passed where ``W_k1`` / ``Theta`` is expected
    is refused rather than silently broadcast."""
    dims = op._dims_of(da, what)
    bad = [d for d in ('k', 'k_l', 'k_p1', 'k_u') if d in dims]
    if bad:
        raise ValueError(f'{what}: has a level dim {bad} (dims {dims}); pass one level -- '
                         'inputs.W_k1(ds) for W(k_l=1), never the 3-D W')


def _same_centred(**fields):
    """Every field centred, all on identical dims."""
    dims, shape = None, None
    for name, da in fields.items():
        d = op.require_centred(da, name)
        _check_no_level_dim(da, name)
        if dims is None:
            dims, shape = d, da.shape
        elif d != dims or da.shape != shape:
            raise ValueError(f'{name}: dims {d} {da.shape} differ from {dims} {shape}')
    return dims


def _lagrange_derivative_weights(z_nodes, z):
    """Weights ``w_i`` with ``f'(z) ~ sum_i w_i f(z_i)`` for the Lagrange
    polynomial through ``z_nodes`` (two nodes: the centred difference,
    independent of ``z``)."""
    zn = np.asarray(z_nodes, dtype='float64')
    n = zn.size
    w = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            term = 1.0 / (zn[i] - zn[j])
            for k in range(n):
                if k not in (i, j):
                    term *= (z - zn[k]) / (zn[i] - zn[k])
            w[i] += term
    return w


# ---------------------------------------------------------------------------
# b at the chunk levels and b_z
# ---------------------------------------------------------------------------
def buoyancy_levels(Theta, Salt):
    """``b`` on every level of ``Theta``/``Salt`` (the chunk ``Theta_k``,
    ``Salt_k`` on ``(face, k, j, i)``) from **one** ``operators.buoyancy``
    call: JMD95 potential density referenced to ``p = 0`` for all levels
    (planning §5.1: the ~0.5 dbar in-situ difference between ``k = 0`` and
    ``k = 1`` is negligible, and ``b_z`` must be a difference of the *same*
    density). Returns the DataArray with the ``k`` dim kept."""
    if 'k' not in Theta.dims or 'k' not in Salt.dims:
        raise ValueError(f'buoyancy_levels: Theta/Salt need a k dim, got {Theta.dims} / {Salt.dims}')
    ds = xr.Dataset({'Theta': Theta, 'Salt': Salt})
    b = op.buoyancy(ds)
    op.assert_dims(b, Theta.dims, 'buoyancy_levels')
    b.attrs['eos'] = 'JMD95, p = 0 at every level (potential density; same call for all levels)'
    return b


def b_z(Theta, Salt, grid_ds, drF, *, order: int = 1):
    """Top-cell vertical buoyancy gradient ``b_z`` [s^-2] at the base of
    the top cell from the chunk levels ``Theta(k)``, ``Salt(k)``.

    ``order = 1`` (default): ``(b_k0 - b_k1) / (Z[0] - Z[1])`` with the
    centre depths from ``drF`` (``dz = (drF[0] + drF[1])/2 = 1.07 m``; the
    store's ``Z = -0.5, -1.57``), i.e. the centred difference across the
    ``k_l = 1`` interface (``z = -1.0 m``, 3.5 cm off the midpoint).
    ``order = 2``: the quadratic through ``k = 0, 1, 2`` differentiated at
    ``z = -drF[0]`` -- the sensitivity the store's third level allows.
    Both levels' ``b`` come from one JMD95 call at ``p = 0``
    (:func:`buoyancy_levels`).

    **Sign:** code ``b`` increases with density, so a warm layer over
    cooler water (the afternoon state) gives ``b_z < 0`` here where
    planning §2.2's textbook ``b_z ~ +2-4e-4 s^-2`` is positive; recorded in
    ``attrs['sign_convention']``.  ``grid_ds`` is accepted for the §4.6
    signature and used only to check the horizontal shape.
    """
    b = buoyancy_levels(Theta, Salt)
    Z = level_depths(drF)
    n = {1: 2, 2: 3}.get(int(order))
    if n is None:
        raise ValueError(f'b_z: order must be 1 or 2, got {order!r}')
    if b.sizes['k'] < n:
        raise ValueError(f'b_z: order {order} needs {n} levels, the store has {b.sizes["k"]}')
    z_eval = -_drF(drF)[0]                              # the base of the top cell, Zl[1]
    w = _lagrange_derivative_weights(Z[:n], z_eval)     # order 1: (1/dz, -1/dz) for any z
    out = sum(float(w[k]) * b.isel(k=k, drop=True) for k in range(n))
    out_dims = tuple(d for d in b.dims if d != 'k')
    op.assert_dims(out, out_dims, 'b_z')
    nj, ni = out.sizes['j'], out.sizes['i']
    if (grid_ds.sizes.get('j'), grid_ds.sizes.get('i')) != (nj, ni):
        raise ValueError(f'b_z: grid ({grid_ds.sizes.get("j")}, {grid_ds.sizes.get("i")}) and '
                         f'field ({nj}, {ni}) shapes differ')
    out.name = 'b_z'
    out.attrs.clear()
    out.attrs.update(
        units='s-2', order=int(order), levels=list(range(n)), Z_m=Z[:n].tolist(),
        z_eval_m=float(z_eval), dz_m=float(Z[0] - Z[1]),
        long_name='top-cell vertical buoyancy gradient db/dz at the base of the top cell (code b)',
        sign_convention=('code b = +g sigma0/rho0 increases with density: a warm (light) layer over '
                         'cooler water gives b_z < 0 here; textbook b_z (planning §2.2, +2-4e-4 s^-2 '
                         'for the diurnal warm layer) is the negative of this'),
        eos='JMD95 at p = 0 for every level, one operators.buoyancy call')
    return out


# ---------------------------------------------------------------------------
# the vertical (finite-top-cell tilting) term
# ---------------------------------------------------------------------------
def vertical_tendency(b, b_k1, W_k1, drF):
    """The top-cell vertical advective tendency ``T_v = -W_k1 b_z``
    [m s^-3], with ``b_z = (b - b_k1)/dz``, ``dz = (drF[0] + drF[1])/2 =
    Z[0] - Z[1] = 1.07 m`` -- the centred difference across the cell base
    where ``W_k1`` lives.  ``W_k1`` is the chunk ``W(k_l=1)``, positive
    upward, so an upwelling (``W_k1 > 0``) of denser water from below
    (``b_k1 > b`` in code ``b``) raises the top-cell ``b``: ``T_v > 0``.

    *(deviation, M3-Q10 / M3-Q11: the prompt and coding §4.6 write
    ``-W_k1 (b_k1 - b)/drF[0]``, which is ``-w b_z`` with the sign reversed
    and ``dz`` replaced by ``drF[0] = 1.0``; the form here is the one
    planning §2.2's equation and the factorised ``-b_z (w_x b_x + w_y b_y)``
    follow from, so the two forms agree exactly for uniform ``b_z``.)*
    """
    _same_centred(b=b, b_k1=b_k1, W_k1=W_k1)
    dz = 0.5 * (_drF(drF)[0] + _drF(drF)[1])
    T_v = -W_k1 * (b - b_k1) / dz                       # -w db/dz, z up, w up
    T_v = op.assert_dims(T_v, b.dims, 'vertical_tendency')
    T_v.name = 'T_v'
    T_v.attrs.clear()
    T_v.attrs.update(units='m s-3', dz_m=float(dz),
                     long_name='top-cell vertical advective tendency T_v = -W_k1 (b - b_k1)/dz '
                               '(code b; W_k1 = chunk W(k_l=1), positive upward)')
    return T_v


def vertical_term(b, b_x, b_y, b_k1, W_k1, drF, grid_ds, grid, *, L_cells: int = 0):
    """The vertical term ``grad_h b . grad_h T_v`` [s^-5, **F units**]:
    ``T_v = -W_k1 (b - b_k1)/dz`` (:func:`vertical_tendency`) is formed
    **first**, low-passed at ``L_cells`` with the kernel that filtered
    ``b, U, V`` (so ``mean(w b_z) - wbar bzbar`` stays inside the term), then
    differenced with ``operators.grad_b`` and dotted with the **filtered**
    ``(b_x, b_y)``.  Not the factorised ``-b_z (w_x b_x + w_y b_y)``, which
    drops ``-w grad(b_z) . grad b`` (planning §2.2; see
    :func:`vertical_term_factorised`, a diagnostic).

    Inputs: ``b``, ``b_k1`` (code ``b`` at ``k = 0`` and ``k = 1``, from
    :func:`buoyancy_levels`), ``W_k1`` (``inputs.W_k1``: the chunk
    ``W(k_l=1)``, a 3-D ``W`` is refused) **unfiltered**, all at the same
    (midpoint) time; ``b_x, b_y`` the filtered gradient at ``L_cells``.
    The budget field is ``2 *`` this (task 3).  NaN (land, the ``W``/``b_k1``
    pattern) propagates; dims asserted after every dbof call.
    """
    dims = _same_centred(b=b, b_x=b_x, b_y=b_y, b_k1=b_k1, W_k1=W_k1)
    T_v = op.lowpass(vertical_tendency(b, b_k1, W_k1, drF), L_cells)
    Tx, Ty = op.grad_b(T_v, grid_ds, grid)              # asserts the centred dims
    term = op.assert_dims(b_x * Tx + b_y * Ty, dims, 'vertical_term')
    term.name = 'vertical_term'
    term.attrs.clear()
    term.attrs.update(
        units='s-5', convention=CONVENTION, L_cells=int(L_cells), dz_m=float(T_v.attrs['dz_m']),
        form='grad_h b . grad_h[ lowpass(-W_k1 (b - b_k1)/dz, L) ]: tendency first, then grad_h',
        filter_note=('T_v itself is low-passed at L with the b/U/V kernel and dotted with the '
                     'filtered grad bbar; the subfilter correlation mean(w b_z) - wbar bzbar is '
                     'inside the term, not dropped'),
        W_source='chunk W(k_l=1) = source W(k_p1=1), the cell-base velocity (dEta/dt + drF delta); '
                 'never delta, never the OSN W(k_l=0)',
        long_name='vertical (finite-top-cell tilting) term, F units')
    return term


def vertical_term_factorised(b_x, b_y, b_z, W_k1, grid_ds, grid, *, L_cells: int = 0):
    """**Diagnostic only:** the factorised form ``-b_z (w_x b_x + w_y b_y)``
    [s^-5, F units] of planning §2.2, which assumes ``b_z`` uniform across
    the front and so drops ``-w grad(b_z) . grad b``.  ``b_z`` and ``W_k1``
    are low-passed at ``L_cells`` before the gradient of ``W``; ``b_x, b_y``
    are the filtered gradient.  Equal to :func:`vertical_term` when
    ``b - b_k1`` is uniform; otherwise ``vertical_term - this`` is the
    dropped term (``-w grad(b_z) . grad b``, exactly for fields on which the
    centred stencil obeys the product rule)."""
    dims = _same_centred(b_x=b_x, b_y=b_y, b_z=b_z, W_k1=W_k1)
    bz = op.lowpass(b_z, L_cells)
    w = op.lowpass(W_k1, L_cells)
    wx, wy = op.grad_b(w, grid_ds, grid)
    term = op.assert_dims(-bz * (wx * b_x + wy * b_y), dims, 'vertical_term_factorised')
    term.name = 'vertical_term_factorised'
    term.attrs.clear()
    term.attrs.update(units='s-5', convention=CONVENTION, L_cells=int(L_cells),
                      form='-b_z (w_x b_x + w_y b_y): DIAGNOSTIC, drops -w grad(b_z) . grad b',
                      long_name='factorised vertical term (uniform-b_z approximation), F units')
    return term


# ---------------------------------------------------------------------------
# the surface-flux (diabatic) term
# ---------------------------------------------------------------------------
def expansion_coefficients(Theta, Salt, dT: float = 0.01, dS: float = 0.01):
    """``(alpha, beta, rho)`` at ``p = 0`` by **centred finite differences of
    the same JMD95 density** ``operators.buoyancy`` wraps
    (``dbof.utils.jmd95_xgcm_implementation.jmd95``): ``alpha = -(1/rho)
    drho/dTheta`` [K^-1], ``beta = (1/rho) drho/dS`` [psu^-1].  At (17 degC,
    33.6) they are ``2.30e-4`` and ``7.49e-4`` (the linear-EOS constants in
    ``physical_constants`` are 2.0e-4 / 7.4e-4 and are **not** used).
    Accepts DataArrays or arrays (NaN propagates); returns the same type."""
    T = np.asarray(getattr(Theta, 'values', Theta), dtype='float64')
    S = np.asarray(getattr(Salt, 'values', Salt), dtype='float64')
    p = np.zeros_like(T)
    with np.errstate(invalid='ignore'):
        rho = jmd95.jmd95(S, T, p)
        drho_dT = (jmd95.jmd95(S, T + dT, p) - jmd95.jmd95(S, T - dT, p)) / (2.0 * dT)
        drho_dS = (jmd95.jmd95(S + dS, T, p) - jmd95.jmd95(S - dS, T, p)) / (2.0 * dS)
        alpha, beta = -drho_dT / rho, drho_dS / rho
    if isinstance(Theta, xr.DataArray):
        wrap = lambda a, name, units: Theta.copy(data=a).rename(name).assign_attrs(  # noqa: E731
            units=units, eos='JMD95 p=0, centred finite differences dT=%g K, dS=%g' % (dT, dS))
        return (wrap(alpha, 'alpha', 'K-1'), wrap(beta, 'beta', 'psu-1'), wrap(rho, 'rho', 'kg m-3'))
    return alpha, beta, rho


def _check_flux_sign(**fluxes):
    """Refuse a flux array whose ``sign_convention`` attr is not positive
    downward, or an ``oceQsw`` with negative values (an upward-positive or
    twice-negated store; ``inputs.fluxes`` guards the same way)."""
    for name, da in fluxes.items():
        sc = da.attrs.get('sign_convention')
        if sc is not None and not str(sc).startswith('positive downward'):
            raise ValueError(f'{name}: sign_convention {sc!r} is not "positive downward" -- '
                             'the term takes the stored downward-positive fluxes, no negation')
    sw = np.asarray(fluxes['oceQsw'].values, dtype='float64')
    if np.any(sw[np.isfinite(sw)] < 0):
        raise ValueError(f'oceQsw < 0 on {int(np.sum(sw < 0))} cells: upward-positive (or negated '
                         'twice) -- refusing')


def surface_buoyancy_tendency(oceQnet, oceQsw, oceFWflx, Theta, Salt, drF, *, f_sw=None,
                              depth=None):
    """The top-cell buoyancy tendency ``B_sfc`` [m s^-3] from the stored
    **downward-positive** fluxes (no negation: the §3.3 store negated the
    source's upward-positive data at write, M2-Q6 (a); ``inputs.fluxes``
    and :func:`_check_flux_sign` refuse anything else), in **code-``b``
    sign** (heating lowers ``b``), as the model forces its top cell:

    * heat into the 1 m cell ``Q_top = (oceQnet - oceQsw) + f_sw oceQsw``:
      ``oceQnet`` *includes* the shortwave (M2 task 4), the non-solar part
      enters at the surface, and only the fraction ``f_sw`` of the shortwave
      is absorbed inside the cell (MITgcm ``SHORTWAVE_HEATING`` + ``swfrac``,
      Jerlov IA: ``f_sw = 0.521`` for ``drF[0] = 1 m``, :data:`F_SW`); the
      rest heats the water below and is not in this cell's budget;
    * ``dTheta/dt = Q_top / (rhoConst c_p drF[0])`` with the model's
      ``rhoConst = 1027.5``, ``c_p = 3994`` (M3-Q11: ``rhoConst``, not the
      ``rho0 = 1000`` of the buoyancy definition, is what the model divides by);
    * ``dS/dt = -S oceFWflx / (rhoConst drF[0])`` with the **local** ``S``
      (``convertFW2Salt = -1``): fresh water in (``oceFWflx > 0``) dilutes;
    * ``B_sfc = (g/rho0) [ drho/dT dT/dt + drho/dS dS/dt ] = (g/rho0) [ -rho
      alpha dT/dt + rho beta dS/dt ]`` with ``alpha, beta, rho`` from
      :func:`expansion_coefficients` (JMD95 finite differences at the local
      ``Theta, Salt``) and ``g = 9.81``, ``rho0 = 1000`` as in
      ``operators.buoyancy`` -- i.e. ``d(g sigma0/rho0)/dt``.

    ``Theta``/``Salt`` are the top-cell values (a ``k`` dim is reduced to
    ``k = 0``).  ``forcing_note`` (6-hourly, linearly interpolated forcing:
    the diurnal shortwave is a triangle peaking at 13 LST) is propagated.

    **``depth`` (M3-Q13 (c), decided 2026-10-10)** replaces ``drF[0]`` as the
    layer the flux is spread over, and ``f_sw`` follows it.  The default
    (``None`` -> ``drF[0] = 1 m``) is the **top cell's own** budget and stays
    primary; passing ``KPPhbl`` gives the **mixed-layer-mean** tendency,
    which is the right one if KPP redistributes the flux within the hour --
    task 6 found the multiplier that minimises the residual matches
    ``drF[0]/KPPhbl`` to 3-5 %.  An array is accepted (per pixel) and is
    floored at ``drF[0]``: a boundary layer thinner than the top cell cannot
    concentrate the flux into less than the cell the budget is written for.
    """
    if 'k' in Theta.dims:
        Theta = Theta.isel(k=0, drop=True)
    if 'k' in Salt.dims:
        Salt = Salt.isel(k=0, drop=True)
    dims = _same_centred(oceQnet=oceQnet, oceQsw=oceQsw, oceFWflx=oceFWflx, Theta=Theta, Salt=Salt)
    _check_flux_sign(oceQnet=oceQnet, oceQsw=oceQsw, oceFWflx=oceFWflx)
    drF0 = _drF(drF)[0]
    if depth is None:
        h, h_note = drF0, f'drF[0] = {drF0:g} m (the top cell; primary, M3-Q13 (c))'
    else:
        h = np.maximum(depth, drF0) if not hasattr(depth, 'dims') else depth.clip(min=drF0)
        h_note = ('per-pixel depth (the KPPhbl sensitivity, M3-Q13 (c)), floored at drF[0]')
    f_sw = sw_fraction_absorbed(h) if f_sw is None else f_sw
    if np.any((np.asarray(getattr(f_sw, 'values', f_sw)) < 0)
              | (np.asarray(getattr(f_sw, 'values', f_sw)) > 1)):
        raise ValueError(f'f_sw is not a fraction: {f_sw}')
    alpha, beta, rho = expansion_coefficients(Theta, Salt)
    # heat: the non-solar flux at the surface plus the shortwave absorbed within the cell
    Q_top = (oceQnet - oceQsw) + f_sw * oceQsw                      # W m^-2 into the 1 m cell
    dT_dt = Q_top / (RHO_CONST * HEAT_CAPACITY_CP * h)              # K s^-1
    # salt: a downward (into the ocean) fresh-water flux dilutes the local salinity
    S_conv = Salt if CONVERT_FW2SALT < 0 else CONVERT_FW2SALT
    dS_dt = -S_conv * oceFWflx / (RHO_CONST * h)                    # psu s^-1
    # code b = g sigma0/rho0: its tendency is (g/rho0) d rho(T, S)/dt; heating lowers b
    B = (G / RHO0_REFERENCE) * (-rho * alpha * dT_dt + rho * beta * dS_dt)
    B = op.assert_dims(B, dims, 'surface_buoyancy_tendency')
    B.name = 'B_sfc'
    B.attrs.clear()
    note = next((d.attrs['forcing_note'] for d in (oceQsw, oceQnet, oceFWflx)
                 if 'forcing_note' in d.attrs), 'not supplied by the caller')
    B.attrs.update(
        units='m s-3', f_sw=(float(f_sw) if np.ndim(getattr(f_sw, 'values', f_sw)) == 0
                             else float(np.nanmedian(getattr(f_sw, 'values', f_sw)))),
        jwtype=JWTYPE, c_p=HEAT_CAPACITY_CP, rhoConst=RHO_CONST,
        convertFW2Salt=CONVERT_FW2SALT, g=G, rho0=RHO0_REFERENCE, drF0_m=float(drF0),
        depth=h_note, depth_m=(float(drF0) if depth is None
                               else float(np.nanmedian(getattr(h, 'values', h)))),
        namelist_source=NAMELIST_SOURCE, forcing_note=note,
        sign='code b (increases with density): heat/fresh water INTO the ocean lowers b; fluxes '
             'taken downward-positive as stored, not negated here',
        long_name='top-cell buoyancy tendency from the surface fluxes (code b)')
    return B


def surface_flux_term(b_x, b_y, oceQnet, oceQsw, oceFWflx, Theta, Salt, drF, grid_ds, grid, *,
                      L_cells: int = 0, f_sw=None, depth=None):
    """The surface-flux term ``grad_h b . grad_h B_sfc`` [s^-5, **F units**]
    with ``B_sfc`` from :func:`surface_buoyancy_tendency` (downward-positive
    fluxes as stored, **no negation**; ``oceQsw`` separately with ``f_sw``;
    JMD95 ``alpha``/``beta`` by finite differences; code-``b`` sign),
    low-passed at ``L_cells`` as ``T_v`` is, then ``operators.grad_b`` and
    the dot with the filtered ``(b_x, b_y)``.  The budget field is ``2 *``
    this (task 3).

    Sign, in code ``b``: a heating gradient *towards the dense side* (more
    heat where ``b`` is larger) lowers ``b`` most where it is highest, so
    ``grad B_sfc`` opposes ``grad b`` and the term is **negative --
    frontolytic**; heating that favours the light side is frontogenetic.
    The dot product is even in the sign of ``b``, so this is the physical
    statement, not a convention.  ``forcing_note`` is propagated to the
    attrs; dims asserted after every dbof call.
    """
    dims = _same_centred(b_x=b_x, b_y=b_y)
    B = surface_buoyancy_tendency(oceQnet, oceQsw, oceFWflx, Theta, Salt, drF, f_sw=f_sw,
                                  depth=depth)
    if B.dims != dims:
        raise ValueError(f'surface_flux_term: flux dims {B.dims} != gradient dims {dims}')
    Bbar = op.lowpass(B, L_cells)
    Bx, By = op.grad_b(Bbar, grid_ds, grid)
    term = op.assert_dims(b_x * Bx + b_y * By, dims, 'surface_flux_term')
    term.name = 'surface_flux_term'
    term.attrs.clear()
    term.attrs.update(
        units='s-5', convention=CONVENTION, L_cells=int(L_cells),
        form='grad_h b . grad_h[ lowpass(B_sfc, L) ], B_sfc the top-cell buoyancy tendency',
        filter_note='B_sfc itself is low-passed at L with the b/U/V kernel and dotted with grad bbar',
        **{k: B.attrs[k] for k in ('f_sw', 'jwtype', 'c_p', 'rhoConst', 'convertFW2Salt',
                                   'namelist_source', 'forcing_note', 'sign', 'depth',
                                   'depth_m')},
        long_name='surface-flux (diabatic) term grad_h b . grad_h B_sfc, F units')
    return term
