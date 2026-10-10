""" The field-level buoyancy-gradient budget (M3 task 3; coding §3.4, §4.7).

One hour pair, one filter scale, every term through the M1 operators::

    measured - 2F - subfilter - vertical - surface_flux  ~  numerical + interior KPP

``measured`` is the semi-Lagrangian ``DGDt``; **everything is compared as
``2F``, never ``F``** (``F = 1/2 DG/Dt``).  :func:`compute_budget` assembles
the §3.4 fields for one pair, :func:`closure_report` turns them into a
verdict against the **pre-declared** tolerances of M3-Q1 / M3-Q2, and
:func:`write_derived` appends the pair to ``tile330_derived_L{L}.zarr``
through ``zarr_series`` (resumable, atomic per pair).

Three rules this module exists to enforce.

1. **No silent degradation.**  Without the chunk store the vertical and
   surface-flux terms are *absent*, not zero: they are left out of the
   Dataset, ``residual.attrs['terms_missing']`` names them, and
   :func:`closure_report` returns ``closed=None`` with a warning that reads
   like one.  A two-term comparison must never be able to pass for a closed
   budget (coding §4.7, §8).
2. **The midpoint is where the physics is evaluated.**  ``F``, the strain,
   the alignment, ``G``, the front selection and every diagnostic come from
   the trajectory-midpoint fields ``0.5 (f_t + f_{t+1})``; only
   ``DGDt_semilag`` sees the two endpoints, and it interpolates ``b``, never
   ``G`` (planning §11).
3. **``L = 0`` is not a limit of ``tau``.**  The explicit subfilter term is
   *identically* zero there (``lowpass`` is the identity, so ``mean(ub) -
   ubar bbar = 0`` exactly), which makes that column "2F vs measured with
   the whole numerical term in the residual" -- labelled as such in
   ``subfilter.attrs`` and in the report, not quoted as a converged
   subfilter estimate (planning §5.4, corrected).

``measured`` is stored under its own name ``DGDt_semilag`` (§3.4's var
list); :func:`measured` is the accessor, and ``attrs['measured']`` records
the choice, so the array is not duplicated 72 times per ``L`` on disk.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import coarsegrain as cg
import inputs as inp
import operators as op
import osn_tiles as ot
import semilag as sl
import validate as va
import vertical as vt
import zarr_series as zs
from osn_tiles import DATA_DIR

DT = inp.DT                                  # 3600 s
EDGE_CELLS = 7                               # M1 task 6: must not shrink
FRONT_PCT = 90.0                             # M3-Q4 (a): the primary front pool
L_CONTRACT = (0, 2, 4, 8)                    # M3-Q3 (a)
WIDTH_BINS = (1.0, 1.5, 2.0, 3.0, 4.0)       # M3-Q5 (a), in dx; task 6 bins front_width

# the pre-declared closure tolerances -- fixed before any number was seen (V3's rule)
Q1_RESID_RATIO_MAX = 0.50      # rms(residual)/rms(measured) on front & valid   (M3-Q1)
Q1_EXPLAINED_MIN = 0.75        # 1 - var(residual)/var(measured)                (M3-Q1)
Q1_RESID_SLOPE_TOL = 0.10      # |OLS slope of residual on 2F|                  (M3-Q1)
Q2_EULER_SLOPE = (0.85, 1.15)  # OLS slope of DGDt_euler on DGDt_semilag        (M3-Q2)
Q2_EULER_CORR_MIN = 0.90       # corr, on front pixels                          (M3-Q2)
Q2_GATE_L_MIN = 2              # gated at L >= 2; L = 0 reported, not gated     (M3-Q2, M3-Q9)
TOL_SOURCE = ('prompt 4 task 6, pre-declared; M3-Q1 and M3-Q2 answered by JXP 2026-10-07 '
              '(frontogenesis_prompt_4.md ## Q&A)')

#: the five budget terms, in the order the residual subtracts them
TERMS = ('two_F', 'subfilter', 'vertical', 'surface_flux')
#: terms that need the chunk store (§3.3)
CHUNK_TERMS = ('vertical', 'surface_flux')
CHUNK_VARS = ('Theta_k', 'Salt_k', 'W_k', 'oceQnet', 'oceQsw', 'oceFWflx', 'drF')
#: §3.4 plus the M3 additions (marked by task 8); the store's var order.
#: ``surface_flux_kpp`` is M3-Q13 (c)'s declared sensitivity (2026-10-10):
#: the surface-flux term with the flux spread over ``KPPhbl`` rather than the
#: 1 m top cell.  It is **not** a budget term -- ``residual`` subtracts
#: ``surface_flux`` only -- and ``TERMS`` is unchanged.
DERIVED_VARS = ('b', 'G', 'two_F', 'two_F_chain', 'DGDt_semilag', 'DGDt_semilag_o5',
                'DGDt_euler', 'subfilter', 'vertical', 'vertical_factorised', 'surface_flux',
                'surface_flux_kpp', 'residual', 'b_z', 'delta', 'sigma_n', 'sigma_s',
                'sigma_mag', 'theta_align', 'lap2_b', 'front_width', 'KPPhbl', 'valid', 'front')


# ---------------------------------------------------------------------------
# discrete helpers: the Laplacian, the biharmonic and the width proxy
# ---------------------------------------------------------------------------
def staggered_grad(f, grid_ds, grid):
    """``(f_x on (j, i_g), f_y on (j_g, i))`` [f m^-1], **model basis**: the
    first two steps of ``calculate_native_gradient_tracer`` (difference,
    divide by ``dxC``/``dyC``) *without* the interpolation back to centres
    -- the flux pair :func:`laplacian` then takes the divergence of.  The
    first face along each axis has no low neighbour in the tile and would be
    differenced against ``padding='fill'``'s zero, so it is NaN (the same
    rule as ``coarsegrain.b_at_velocity_points``)."""
    dims = op.require_centred(f, 'f')
    fx = (grid.diff(f, 'X', padding='fill') / grid_ds['dxC']).compute().copy()
    fy = (grid.diff(f, 'Y', padding='fill') / grid_ds['dyC']).compute().copy()
    fx = op.assert_dims(fx, tuple('i_g' if d == 'i' else d for d in dims), 'staggered_grad[0]')
    fy = op.assert_dims(fy, tuple('j_g' if d == 'j' else d for d in dims), 'staggered_grad[1]')
    fx[{'i_g': 0}] = np.nan
    fy[{'j_g': 0}] = np.nan
    fx.name, fy.name = 'f_x_stag', 'f_y_stag'
    return fx, fy


def laplacian(f, grid_ds, grid):
    """``div(grad f)`` [f m^-2] at the centres -- the model's own ``del^2``:
    the staggered gradient of :func:`staggered_grad` through
    ``coarsegrain.flux_divergence`` (``(Delta_X(f_x dyG) + Delta_Y(f_y dxG))
    / rA``).  Exact for a quadratic on a uniform grid; one NaN cell on each
    side beyond ``f``'s own.

    Not ``grad_b`` twice: that interpolates to the centres between the two
    differences, which widens the stencil and no longer telescopes to the
    model's flux-form operator."""
    fx, fy = staggered_grad(f, grid_ds, grid)
    lap = cg.flux_divergence(fx, fy, grid_ds, grid)
    lap.name = 'lap'
    lap.attrs.clear()
    lap.attrs.update(long_name='Laplacian div(grad f) (flux form, model del^2)')
    return lap


def cell_size(grid_ds):
    """``sqrt(rA)`` [m] at the centres: the area-equivalent cell width, the
    "dx" :func:`front_width` reports in.  Exact on a square grid; on the
    tile ``dxC`` 1796 m and ``dyC`` 1950 m bracket it by about +-4 %, which
    is well inside M3-Q5's ``{<=1, 1-1.5, 1.5-2, 2-3, 3-4, >4}`` bins."""
    return np.sqrt(grid_ds['rA'])


def front_width(G, grid_ds, grid):
    """The per-pixel width proxy ``ell = 2 sqrt(G / |lap G|)`` **in cells**
    (M3-Q5 (a)).  Exact at the maximum of ``G = (b0/ell)^2 sech^4(x/ell)``,
    i.e. for ``synthetic.py``'s ``b = b0 tanh(x/ell)`` fronts: there
    ``lap G = -4 G / ell^2`` so the expression returns ``ell`` itself.

    Meaningful **on front pixels**, where ``G`` is near a ridge maximum;
    stored everywhere it is finite, because task 6 bins it only inside the
    front pool.  NaN where ``lap G = 0`` (no curvature, no width) or where
    either field is."""
    lap = laplacian(G, grid_ds, grid)
    with np.errstate(invalid='ignore', divide='ignore'):
        ell = 2.0 * np.sqrt(G / np.abs(lap)) / cell_size(grid_ds)
    ell = op.assert_dims(ell.where(np.isfinite(ell)), tuple(G.dims), 'front_width')
    ell.name = 'front_width'
    ell.attrs.clear()
    ell.attrs.update(units='cells (dx = sqrt(rA))',
                     form='2 sqrt(G / |lap G|) / sqrt(rA); exact for G = Gmax sech^4(x/ell)',
                     decided='M3-Q5 (a), JXP 2026-10-07', bins_dx=list(WIDTH_BINS),
                     long_name='per-pixel front width proxy, in cells')
    return ell


def _b_z_mid(b_levels, drF):
    """``b_z`` at the base of the top cell from **midpoint buoyancy levels**:
    ``(b_k0 - b_k1) / (Z[0] - Z[1])``, i.e. ``vertical.b_z(order=1)``
    evaluated on ``0.5 (b_t + b_{t+1})`` rather than on the midpoint
    ``Theta``/``Salt``.  The two differ only by the EOS's curvature over the
    hour (the warm layer is 0.01 K, M3 task 2), and this one shares its
    ``b_k0``, ``b_k1`` with :func:`vertical.vertical_term`, so
    ``vertical - vertical_factorised`` is exactly the dropped
    ``-w grad(b_z) . grad b`` rather than that plus an EOS mismatch."""
    Z = vt.level_depths(drF)
    bz = (b_levels.isel(k=0, drop=True) - b_levels.isel(k=1, drop=True)) / (Z[0] - Z[1])
    bz.name = 'b_z'
    bz.attrs.clear()
    bz.attrs.update(units='s-2', order=1, dz_m=float(Z[0] - Z[1]),
                    evaluated_on='midpoint buoyancy levels (not midpoint Theta/Salt)',
                    sign_convention=('code b increases with density: a warm layer over cooler '
                                     'water gives b_z < 0 (vertical.b_z)'),
                    long_name='top-cell vertical buoyancy gradient at the cell base')
    return bz


# ---------------------------------------------------------------------------
# the hour pair
# ---------------------------------------------------------------------------
def _merged(raw_ds, chunk_ds):
    """``raw_ds`` alone (already the task-1 merge), or merged with a
    ``chunk_ds`` -- renamed with ``inputs.RENAME`` if it still carries the
    source names.  Returns ``(ds, has_chunk)``."""
    ds = raw_ds
    if chunk_ds is not None:
        ch = chunk_ds.rename({k: v for k, v in inp.RENAME.items() if k in chunk_ds})
        ds = xr.merge([raw_ds, ch], compat='override', join='exact')
    return ds, all(v in ds for v in CHUNK_VARS)


def _hour_pair(ds, t0, has_chunk):
    """``inputs.hour_pair`` when the chunk variables are present (it
    re-checks the k = 0 identity and the flux sign on both hours, which is
    the whole point of loading through ``inputs``); otherwise the same
    load -- ``isel``, ``load``, ``expand_dims('face')``, ``float64`` -- with
    those two chunk-only invariants skipped, because ``chunk_ds=None`` is
    exactly the case in which they cannot be checked."""
    if has_chunk:
        return inp.hour_pair(ds, t0)
    inp.time_mid(ds, t0)                       # the 3600 s step and the index range
    out = []
    for t in (int(t0), int(t0) + 1):
        h = ds.isel(time=t).load().expand_dims('face').astype('float64')
        h.attrs.update(ds.attrs, time_index=t, time=str(h['time'].values)[:19],
                       invariants='k=0 identity and flux sign NOT checked (no chunk vars)')
        out.append(h)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------
def _np(da):
    """The ``(face, j, i)`` array as positional ``(nj, ni)`` float64."""
    return np.asarray(da.values, dtype='float64').reshape(da.shape[-2:])


def compute_budget(raw_ds, grid_ds, grid, masks, L_cells, dt=DT, chunk_ds=None, *,
                   t0=0, forms=('discrete', 'chain'), order=3, order_sens=5,
                   front_pct=FRONT_PCT):
    """One hour pair's budget at one filter scale -> ``xr.Dataset`` on
    ``(time: 1, j, i)`` (coding §3.4, §4.7; prompt 4 task 3).

    Parameters
    ----------
    raw_ds : xarray.Dataset
        The task-1 merge (``inputs.open_inputs``), or the §3.2 store alone,
        in which case ``chunk_ds`` supplies §3.3.
    grid_ds, grid, masks
        ``osn_tiles.open_grid(with_face=True)``, ``build_xgcm``, the §3.5
        masks (Dataset or a bool ``(j, i)`` array).
    L_cells : int
        The filter scale, applied to ``b``, ``U`` **and** ``V`` alike.
        ``L = 0`` is the identity and the explicit subfilter term is then
        identically zero.
    dt : float
    chunk_ds : xarray.Dataset, optional
        §3.3.  **Without it** ``vertical`` and ``surface_flux`` are absent
        from the result and the residual is a catch-all -- see
        :func:`closure_report`.
    t0 : int
        The first hour of the pair; ``F`` and the front selection are at
        ``t0 + 30 min``.
    forms : tuple
        ``operators.frontogenesis`` forms to run.  ``'discrete'`` is the
        primary and is stored as ``two_F``; every other form is stored as
        ``two_F_<form>`` (M1-Q1: on real front pixels the discrete ``F`` is
        ~0.79x the chain ``F`` -- a stated systematic, not a bug).
    order, order_sens : int
        Semi-Lagrangian interpolation orders: the primary (3) and the
        sensitivity (5, M1-Q6), the latter costing a 6-node NaN rim per axis.
    front_pct : float
        The front pool's percentile of ``G_mid`` over ``valid`` (90, M3-Q4;
        p80 / p95 are recomputed from the stored ``G`` in task 6).
    """
    L = int(L_cells)
    ds, has_chunk = _merged(raw_ds, chunk_ds)
    if 'discrete' not in forms:
        raise ValueError(f"forms {forms} must contain 'discrete', the primary (M1 task 6)")
    h0, h1 = _hour_pair(ds, t0, has_chunk)
    t_mid = inp.time_mid(ds, t0)

    # --- the midpoint fields, filtered at L (the same kernel on b, U and V)
    f0, f1 = inp.filtered(h0, L), inp.filtered(h1, L)
    b_mid, U_mid, V_mid = (inp.midpoint(a, b) for a, b in zip(f0, f1))
    b_x, b_y = op.grad_b(b_mid, grid_ds, grid)
    G_mid = op.assert_dims(b_x ** 2 + b_y ** 2, tuple(b_mid.dims), 'G_mid')   # = op.gradb2(b_mid)

    # --- the predicted side: 2F at the midpoint, one column per form
    two_F = {f: 2.0 * op.frontogenesis(b_mid, U_mid, V_mid, grid_ds, grid, form=f) for f in forms}

    # --- the measured side: semi-Lagrangian (primary), order-5 sensitivity, Eulerian
    sem = sl.measured_DGDt(f0[0], f1[0], U_mid, V_mid, grid_ds, grid, dt=dt, order=order)
    sem5 = sl.measured_DGDt(f0[0], f1[0], U_mid, V_mid, grid_ds, grid, dt=dt, order=order_sens)
    eul = sl.eulerian_DGDt(op.gradb2(f0[0], grid_ds, grid), op.gradb2(f1[0], grid_ds, grid),
                           U_mid, V_mid, grid_ds, grid, dt=dt)

    # --- the subfilter term, from the UNFILTERED midpoint fields (it filters internally)
    r0, r1 = (f0, f1) if L == 0 else (inp.filtered(h0, 0), inp.filtered(h1, 0))
    b_r, U_r, V_r = (inp.midpoint(a, b) for a, b in zip(r0, r1))
    tau_x, tau_y = cg.subfilter_flux(b_r, U_r, V_r, L, grid_ds, grid)
    tau_d = cg.subfilter_bdelta(b_r, U_r, V_r, L, grid_ds, grid)   # mandatory: divergent flow
    sub = 2.0 * cg.subfilter_term(b_mid, tau_x, tau_y, grid_ds, grid, tau_delta=tau_d)
    sub.attrs['tau_delta'] = 'included (M1 task 4: the flux form alone overstates it 2.2x here)'
    if L == 0:
        sub.attrs['L0_note'] = ('IDENTICALLY ZERO at L = 0 (lowpass is the identity, so '
                                'mean(ub) - ubar bbar = 0 exactly) -- this column is 2F vs '
                                'measured with the whole numerical term in the residual, NOT a '
                                'limit of tau (planning §5.4, corrected)')

    terms = {'two_F': two_F['discrete'], 'subfilter': sub}
    extra = {f'two_F_{f}': v for f, v in two_F.items() if f != 'discrete'}

    # --- the chunk terms: present, or loudly absent
    if has_chunk:
        drF = inp.drF(ds)
        bl = inp.midpoint(vt.buoyancy_levels(h0['Theta_k'], h0['Salt_k']),
                          vt.buoyancy_levels(h1['Theta_k'], h1['Salt_k']))
        b_k0, b_k1 = bl.isel(k=0, drop=True), bl.isel(k=1, drop=True)
        W = inp.midpoint(inp.W_k1(h0), inp.W_k1(h1))
        bz = _b_z_mid(bl, drF)
        terms['vertical'] = 2.0 * vt.vertical_term(b_k0, b_x, b_y, b_k1, W, drF,
                                                   grid_ds, grid, L_cells=L)
        extra['vertical_factorised'] = 2.0 * vt.vertical_term_factorised(
            b_x, b_y, bz, W, grid_ds, grid, L_cells=L)
        extra['b_z'] = bz
        fl = [inp.midpoint(a, b) for a, b in zip(inp.fluxes(h0), inp.fluxes(h1))]
        kpp_mid = inp.midpoint(h0['KPPhbl'], h1['KPPhbl'])
        Th = inp.midpoint(h0['Theta'], h1['Theta'])
        Sa = inp.midpoint(h0['Salt'], h1['Salt'])
        terms['surface_flux'] = 2.0 * vt.surface_flux_term(b_x, b_y, *fl, Th, Sa, drF,
                                                           grid_ds, grid, L_cells=L)
        # M3-Q13 (c), decided 2026-10-10: the same term with the flux spread over the KPP
        # boundary layer instead of the 1 m top cell -- a declared SENSITIVITY, never the
        # primary.  It is not a rescaling of the primary: KPPhbl varies in space, so the
        # gradient of B_sfc changes shape, not just amplitude.
        extra['surface_flux_kpp'] = 2.0 * vt.surface_flux_term(
            b_x, b_y, *fl, Th, Sa, drF, grid_ds, grid, L_cells=L, depth=kpp_mid)
    missing = [t for t in CHUNK_TERMS if t not in terms]

    # --- the residual, and the reduction rule over every field it is built from
    resid = sem.copy()
    for name in TERMS:
        if name in terms:
            resid = resid - terms[name]
    resid = op.assert_dims(resid, tuple(sem.dims), 'residual')

    pool = [b_mid, G_mid, terms['two_F'], sem] + [terms[t] for t in TERMS if t in terms]
    valid, n_lost = inp.valid(masks, *pool)
    Gv = _np(G_mid)
    thr = float(np.percentile(Gv[valid], float(front_pct)))
    front = valid & (Gv >= thr)

    # --- the strain set and the alignment, all at the midpoint
    delta, sig_n, sig_s, sig_mag = op.strain_divergence(U_mid, V_mid, grid_ds, grid)
    theta = op.strain_alignment(b_x, b_y, sig_n, sig_s)

    out = dict(b=b_mid, G=G_mid, DGDt_semilag=sem, DGDt_semilag_o5=sem5, DGDt_euler=eul,
               residual=resid, delta=delta, sigma_n=sig_n, sigma_s=sig_s, sigma_mag=sig_mag,
               theta_align=theta, lap2_b=laplacian(laplacian(b_mid, grid_ds, grid), grid_ds, grid),
               front_width=front_width(G_mid, grid_ds, grid),
               KPPhbl=inp.midpoint(h0['KPPhbl'], h1['KPPhbl']),
               **terms, **extra)

    resid.attrs.clear()
    resid.attrs.update(units='s-5',
                       form='measured - ' + ' - '.join(t for t in TERMS if t in terms),
                       terms_present=[t for t in TERMS if t in terms], terms_missing=missing,
                       interpretation=('numerical diffusion + interior KPP (planning §2.3)'
                                       if not missing else
                                       'CATCH-ALL: the vertical and surface-flux terms are '
                                       'absent, so this is NOT a closed budget'),
                       long_name='budget residual')
    return _pack(out, valid, front, thr, grid_ds, masks, ds, h0,
                 L=L, t0=int(t0), t_mid=t_mid, dt=float(dt), forms=tuple(forms),
                 order=int(order), order_sens=int(order_sens), front_pct=float(front_pct),
                 n_lost=n_lost, missing=missing, has_chunk=has_chunk)


def _pack(fields, valid, front, thr, grid_ds, masks, ds, h0, **meta):
    """Assemble the ``(time: 1, j, i)`` Dataset: squeeze ``face``, add the
    ``time``/``time_mid`` coords, the bool masks, the static
    ``coast_distance_km``, and the attrs that make the store
    self-describing (§3.4)."""
    t_hour = np.asarray([h0['time'].values]).astype('datetime64[ns]')
    g2 = grid_ds.squeeze('face') if 'face' in grid_ds.dims else grid_ds
    coords = {'time': t_hour, 'time_mid': ('time', np.asarray([meta['t_mid']], 'datetime64[ns]')),
              'j': g2['j'], 'i': g2['i']}
    for c in ('XC', 'YC'):                       # absent on the synthetic grids
        if c in g2.coords or c in g2.data_vars:
            coords[c] = (('j', 'i'), np.asarray(g2[c].values))
    data = {}
    for name, da in fields.items():
        a = _np(da)[None]
        data[name] = xr.DataArray(a, dims=('time', 'j', 'i'), attrs=dict(da.attrs))
    data['valid'] = xr.DataArray(valid[None], dims=('time', 'j', 'i'), attrs=dict(
        long_name='mask_analysis & isfinite(every budget field)', n_lost=meta['n_lost'],
        rule='inputs.valid; edge_cells = 7 (M1 task 6), fields: b, G, 2F, measured and '
             'every term present'))
    data['front'] = xr.DataArray(front[None], dims=('time', 'j', 'i'), attrs=dict(
        long_name=f'G_mid >= p{meta["front_pct"]:g} over valid, at the MIDPOINT time',
        front_pct=meta['front_pct'], G_threshold=thr, n_front=int(front.sum()),
        decided='M3-Q4 (a), JXP 2026-10-07: p90 primary; p80 / p95 are task 6 sensitivities'))
    dist = masks['coast_distance_km'] if isinstance(masks, xr.Dataset) else None
    if dist is not None:
        data['coast_distance_km'] = xr.DataArray(np.asarray(dist.values), dims=('j', 'i'),
                                                 attrs=dict(dist.attrs))
    out = xr.Dataset(data, coords=coords)
    out.attrs.update(
        L_cells=meta['L'], t0=meta['t0'], dt=meta['dt'], forms=list(meta['forms']),
        order=meta['order'], order_sens=meta['order_sens'], front_pct=meta['front_pct'],
        edge_cells=EDGE_CELLS, front_percentile_pool='valid, at the midpoint time',
        time_mid=str(meta['t_mid'])[:19],
        local_solar_hour=float(inp.local_solar_hour(meta['t_mid'])),
        measured='DGDt_semilag',
        budget='measured - 2F - subfilter - vertical - surface_flux ~ numerical + interior KPP',
        terms_present=[t for t in TERMS if t in fields], terms_missing=meta['missing'],
        n_valid=int(valid.sum()), n_front=int(front.sum()), n_lost=meta['n_lost'],
        chunk_terms='present' if meta['has_chunk'] else 'ABSENT (chunk_ds is None)',
        subfilter_note=('identically zero at L = 0' if meta['L'] == 0 else
                        'flux form minus tau_delta (M1 task 4)'),
        source_osn=str(ds.attrs.get('source_osn', 'unknown')),
        source_chunk=str(ds.attrs.get('source_chunk', 'none')),
        operators='operators.frontogenesis / semilag defaults: discrete, order 3, vel_order 3, '
                  'n_iter 3 (M1 task 6, flag 10)',
        **ot._provenance())
    return out


def measured(budget_ds):
    """The measured side: ``DGDt_semilag`` (``attrs['measured']`` names it).
    Stored under its §3.4 name rather than duplicated as ``measured``."""
    return budget_ds[budget_ds.attrs.get('measured', 'DGDt_semilag')]


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def _rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a, dtype='float64') ** 2)))


def _fit(x, y):
    """OLS slope (with intercept) and correlation of ``y`` on ``x`` over the
    cells where both are finite."""
    x, y = np.asarray(x, 'float64').ravel(), np.asarray(y, 'float64').ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(ols=float('nan'), corr=float('nan'), n=int(ok.sum()))
    e = va.slope_estimators(x[ok], y[ok])
    return dict(ols=e['ols'], corr=e['corr'], intercept=e['intercept'], n=e['n'])


def _pool_stats(bd, m):
    """Every ratio, the explained fraction and the two fits on one pool."""
    meas = _np_t(measured(bd))[m]
    res = _np_t(bd['residual'])[m]
    r_meas, r_2F = _rms(meas), _rms(_np_t(bd['two_F'])[m])
    out = dict(n=int(m.sum()), rms_measured=r_meas, rms_two_F=r_2F,
               rms_residual=_rms(res),
               terms={t: dict(rms=_rms(_np_t(bd[t])[m]),
                              over_measured=_rms(_np_t(bd[t])[m]) / r_meas,
                              over_two_F=_rms(_np_t(bd[t])[m]) / r_2F)
                      for t in TERMS if t in bd})
    out['residual_over_measured'] = out['rms_residual'] / r_meas
    out['residual_over_two_F'] = out['rms_residual'] / r_2F
    with np.errstate(invalid='ignore'):
        out['explained'] = float(1.0 - np.nanvar(res) / np.nanvar(meas))
    out['residual_on_two_F'] = _fit(_np_t(bd['two_F'])[m], res)
    out['euler_on_semilag'] = _fit(_np_t(bd['DGDt_semilag'])[m], _np_t(bd['DGDt_euler'])[m])
    return out


def _np_t(da):
    """A ``(time: 1, j, i)`` variable as a positional ``(nj, ni)`` array."""
    return np.asarray(da.values, dtype='float64').reshape(da.shape[-2:])


def closure_report(budget_ds, mask=None) -> dict:
    """The verdict on one pair (coding §4.7; prompt 4 task 3).

    Computes, **on ``front & valid`` and on ``valid`` separately**: the rms
    of every term relative to ``rms(measured)`` and to ``rms(2F)``;
    ``rms(residual)`` over both; the explained fraction ``1 -
    var(residual)/var(measured)``; the OLS slope and correlation of the
    residual on ``2F`` (a term missing *in proportion to* ``2F`` shows up
    here, where a ratio alone would not); and the OLS slope and correlation
    of ``DGDt_euler`` on ``DGDt_semilag``.

    ``mask`` optionally restricts both pools (a bool ``(j, i)`` array).

    The verdict ``closed`` is ``True`` / ``False`` against the
    **pre-declared** M3-Q1 and M3-Q2 tolerances (module constants, source in
    ``TOL_SOURCE``), and **``None`` whenever a term is missing** -- with
    ``warning`` saying, in words, that the residual is then a catch-all and
    this is not a closed budget.  M3-Q2 is a gate only at ``L >= 2``; at
    ``L = 0`` it is reported and interpreted (M3-Q9).
    """
    bd = budget_ds.squeeze('time') if 'time' in budget_ds.dims else budget_ds
    valid = np.asarray(bd['valid'].values, bool).reshape(bd['valid'].shape[-2:])
    front = np.asarray(bd['front'].values, bool).reshape(bd['front'].shape[-2:])
    if mask is not None:
        m = np.asarray(getattr(mask, 'values', mask), bool)
        valid, front = valid & m, front & m
    L = int(bd.attrs.get('L_cells', 0))
    missing = list(bd.attrs.get('terms_missing', []))

    rep = dict(L_cells=L, t0=int(bd.attrs.get('t0', 0)), time_mid=bd.attrs.get('time_mid', ''),
               local_solar_hour=float(bd.attrs.get('local_solar_hour', float('nan'))),
               terms_present=list(bd.attrs.get('terms_present', [])), terms_missing=missing,
               n_valid=int(valid.sum()), n_front=int(front.sum()),
               n_lost=int(bd.attrs.get('n_lost', 0)),
               front_pct=float(bd.attrs.get('front_pct', FRONT_PCT)),
               tolerances=dict(source=TOL_SOURCE, resid_ratio_max=Q1_RESID_RATIO_MAX,
                               explained_min=Q1_EXPLAINED_MIN,
                               resid_slope_tol=Q1_RESID_SLOPE_TOL,
                               euler_slope=list(Q2_EULER_SLOPE),
                               euler_corr_min=Q2_EULER_CORR_MIN, euler_gate_L_min=Q2_GATE_L_MIN),
               front_and_valid=_pool_stats(bd, front), valid=_pool_stats(bd, valid))

    p = rep['front_and_valid']
    q1 = dict(resid_ratio=dict(value=p['residual_over_measured'], tol=Q1_RESID_RATIO_MAX,
                               passed=bool(p['residual_over_measured'] <= Q1_RESID_RATIO_MAX)),
              explained=dict(value=p['explained'], tol=Q1_EXPLAINED_MIN,
                             passed=bool(p['explained'] >= Q1_EXPLAINED_MIN)),
              resid_slope=dict(value=p['residual_on_two_F']['ols'], tol=Q1_RESID_SLOPE_TOL,
                               passed=bool(abs(p['residual_on_two_F']['ols'])
                                           <= Q1_RESID_SLOPE_TOL)))
    e = p['euler_on_semilag']
    q2 = dict(slope=e['ols'], corr=e['corr'], gated=bool(L >= Q2_GATE_L_MIN),
              passed=bool(Q2_EULER_SLOPE[0] <= e['ols'] <= Q2_EULER_SLOPE[1]
                          and e['corr'] >= Q2_EULER_CORR_MIN))
    if not q2['gated']:
        q2['note'] = ('L = 0: reported and interpreted, NOT gated (M3-Q2 / M3-Q9); the explicit '
                      'subfilter term is identically zero here')
    rep['gates'] = dict(Q1=q1, Q2=q2)
    gates_ok = all(g['passed'] for g in q1.values()) and (q2['passed'] or not q2['gated'])
    if missing:
        rep['closed'] = None
        rep['warning'] = (
            'NOT A CLOSED BUDGET: ' + ', '.join(missing) + ' absent (no chunk store). The '
            'residual is a CATCH-ALL -- everything not computed, not numerical diffusion plus '
            'interior KPP. No closure verdict and no efficiency number may be quoted from this '
            'report (prompt 4; coding §4.7, §8).')
    else:
        rep['closed'] = bool(gates_ok)
        rep['warning'] = None
    return rep


def format_closure(rep) -> str:
    """The report as a verdict, in text."""
    v = {True: 'CLOSED', False: 'NOT CLOSED', None: 'NO VERDICT'}[rep['closed']]
    w = '=' * 78
    out = [w, f'CLOSURE REPORT  L = {rep["L_cells"]}  pair t0 = {rep["t0"]}  '
              f'mid {rep["time_mid"]} ({rep["local_solar_hour"]:.1f} LST)', w,
           f'  terms present : {", ".join(rep["terms_present"]) or "none"}',
           f'  terms MISSING : {", ".join(rep["terms_missing"]) or "none"}',
           f'  n_valid {rep["n_valid"]:,}   n_front {rep["n_front"]:,} '
           f'(p{rep["front_pct"]:g})   n_lost {rep["n_lost"]}']
    for pool in ('front_and_valid', 'valid'):
        p = rep[pool]
        out += ['', f'  --- {pool} (n = {p["n"]:,}) '
                    f'rms(measured) = {p["rms_measured"]:.3e}  rms(2F) = {p["rms_two_F"]:.3e}',
                '      term            rms        /measured   /2F']
        for t, s in p['terms'].items():
            out.append(f'      {t:14s} {s["rms"]:.3e}  {s["over_measured"]:8.3f}  '
                       f'{s["over_two_F"]:8.3f}')
        out += [f'      {"residual":14s} {p["rms_residual"]:.3e}  '
                f'{p["residual_over_measured"]:8.3f}  {p["residual_over_two_F"]:8.3f}',
                f'      explained fraction 1 - var(res)/var(meas) = {p["explained"]:.3f}',
                f'      residual on 2F    : slope {p["residual_on_two_F"]["ols"]:+.3f}  '
                f'corr {p["residual_on_two_F"]["corr"]:+.3f}',
                f'      euler on semilag  : slope {p["euler_on_semilag"]["ols"]:+.3f}  '
                f'corr {p["euler_on_semilag"]["corr"]:+.3f}']
    q1, q2 = rep['gates']['Q1'], rep['gates']['Q2']
    out += ['', '  gates, pre-declared (M3-Q1, M3-Q2):']
    for k, g in q1.items():
        out.append(f'      {"PASS" if g["passed"] else "FAIL"}  Q1 {k:12s} '
                   f'{g["value"]:+.3f}  (tol {g["tol"]})')
    out.append(f'      {"PASS" if q2["passed"] else "FAIL"}  Q2 euler/semilag slope '
               f'{q2["slope"]:+.3f} corr {q2["corr"]:+.3f}'
               + ('' if q2['gated'] else '   [NOT GATED at this L]'))
    out += ['', f'  VERDICT: {v}']
    if rep['warning']:
        out += ['', '  ' + '!' * 74] + ['  !! ' + line for line in _wrap(rep['warning'])] \
               + ['  ' + '!' * 74]
    out.append(w)
    return '\n'.join(out)


def _wrap(text, width=70):
    words, lines, cur = text.split(), [], ''
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f'{cur} {word}'.strip()
    return lines + ([cur] if cur else [])


def print_closure(rep):
    """Print :func:`format_closure` and return the report unchanged."""
    print(format_closure(rep))
    return rep


# ---------------------------------------------------------------------------
# the derived store
# ---------------------------------------------------------------------------
def derived_path(L, out=None):
    return DATA_DIR / f'tile330_derived_L{int(L)}.zarr' if out is None else out


def _encoding(ds):
    """One chunk per pair per variable; ``time`` as §3.2."""
    enc = {v: {'chunks': tuple(1 if d == 'time' else n for d, n in zip(ds[v].dims, ds[v].shape))}
           for v in ds.data_vars}
    enc['time'] = {'units': 'seconds since 2011-09-10', 'dtype': 'int64'}
    enc['time_mid'] = {'units': 'seconds since 2011-09-10', 'dtype': 'int64'}
    return enc


def _truncate_from(out, n_keep: int):
    """Drop time index ``n_keep`` and everything after it, through
    ``zarr_series._truncate`` -- the clobber path.  A time-ordered,
    append-only store can only be rewritten from a prefix, so re-writing a
    pair necessarily drops the pairs after it; the sweep writes in time
    order, so on a re-run that is exactly the intended effect (the same
    policy as ``osn_tiles.pull_series``, which rewrites from scratch)."""
    import zarr
    g = zarr.open_group(Path(out), mode='r+', use_consolidated=False)
    zs._truncate(Path(g.store.root), zs._time_arrays(g), int(n_keep))
    zs.consolidate(out)


def write_derived(budget_ds, L, out=None, clobber: bool = False, log=None):
    """Append one pair to ``data/tile330_derived_L{L}.zarr`` (§3.4) through
    ``zarr_series.append_hour``: **resumable and atomic per pair**
    (``present_times`` decides "already present", ``repair_trailing``
    cleans a cut-off append), ``float32`` on disk, one chunk per pair per
    variable, ``time`` encoded as §3.2.  ``clobber`` rewrites an hour that
    is already there.  Returns the path."""
    out = Path(derived_path(L, out))
    say = (lambda m: zs._say(m, log))
    if int(budget_ds.attrs.get('L_cells', L)) != int(L):
        raise ValueError(f'budget_ds is L = {budget_ds.attrs.get("L_cells")}, not L = {L}')
    ds = budget_ds.copy()
    for v in ds.data_vars:                       # float64 in the compute path, float32 on disk
        if ds[v].dtype.kind == 'f':
            ds[v] = ds[v].astype('float32')
    zs.repair_trailing(out, log=log)
    present = zs.present_times(out)
    t = ds['time'].values.astype('datetime64[s]')[0]
    if t in present:
        if not clobber:
            say(f'{out.name}: {t} already present, skipped')          # the resume path
            return str(out)
        keep = int(np.searchsorted(present, t))
        say(f'{out.name}: clobber -- truncating from pair {keep} ({t}) and re-appending '
            f'({present.size - keep} pair(s) dropped)')
        _truncate_from(out, keep)
        present = zs.present_times(out)
    attrs = dict(ds.attrs)
    attrs.update(times=[str(x)[:19] for x in np.append(present, t)],
                 n_pairs=int(present.size + 1), vars=list(DERIVED_VARS),
                 schema='§3.4 + the M3 additions (b_z, two_F_chain, DGDt_semilag_o5, '
                        'vertical_factorised, lap2_b, front_width, valid, front, KPPhbl, '
                        'coast_distance_km)')
    zs.append_hour(out, ds, encoding=_encoding(ds), attrs=attrs)
    zs.consolidate(out)
    say(f'{out.name}: appended {t} (L = {L}); {present.size + 1} pairs on disk')
    return str(out)
