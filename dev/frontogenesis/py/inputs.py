""" The merged input layer for the M3 budget (prompt 4 task 1; coding §3.1-§3.3,
§4.6-§4.7) -- **the only place M3 opens the stores.**

The OSN surface store (§3.2) and the chunk store (§3.3) hold the same model
output (M2 tasks 4-6); a plain ``xr.merge`` fails (``Salt`` 2-D vs 3-D under
one name), so :func:`open_inputs` merges **by rename** (M3-Q6): ``Theta_k,
Salt_k, W_k`` keep the three levels, and chunk ``k = 0`` / ``k_l = 0`` being
bit-identical to OSN becomes an assertable invariant.  Asserted on every
open: the shared coords (``time`` at 3600 s steps, ``niter``, ``j``, ``i``,
``XC``, ``YC``, scalar ``face = 10``); the k = 0 identity (NaN-aware) and
the flux-sign guard (``sign_convention`` positive downward, stored
``oceQsw >= 0``, **refused otherwise** so a rewritten store is never
double-negated) on the hours checked.  The invariant check is **lazy**:
the hours requested, or hour 0 alone on a lazy full open
(``check_hours='all'`` for all 72), and :func:`hour_pair` re-checks the two
hours it loads, where the comparison costs nothing -- so every hour a budget
is computed from has passed it without 72 reads per open.

Accessors keep the physics modules from indexing the store: :func:`W_k1`
(the cell-base ``W_k(k_l=1)``, never ``k_l=0`` = ``dEta/dt``), :func:`drF`,
:func:`Z`, :func:`fluxes` (as stored, **no negation**), :func:`wind`
(``oceTAU*`` re-masked with ``hFacW``/``hFacS``, plus ``KPPhbl``);
:func:`hour_pair` / :func:`midpoint` / :func:`time_mid`; :func:`filtered`
(the same ``L`` on ``b, U, V``, coding §1.2); :func:`valid` (``mask_analysis
& isfinite(every field)`` with the cells lost, required at ``L = 8``).
Functions, not classes; float64 on the compute path.
"""

import numpy as np
import xarray as xr

import osn_tiles as ot
import masking as mk
import operators as op
import semilag as sl
from osn_tiles import DATA_DIR
from dbof.llc4320_ingestion.grid import ensure_comodo_attrs

RAW_ZARR = DATA_DIR / 'tile330_raw_20120702T00_72h.zarr'       # §3.2 (M2 task 2)
CHUNK_ZARR = DATA_DIR / 'tile330_chunk_20120702T00_72h.zarr'   # §3.3 (M2 task 5)
GRID_ZARR = DATA_DIR / 'tile330_grid.zarr'                      # §3.1 (M0 task 4)
MASKS_NC = DATA_DIR / 'tile330_masks.nc'                        # §3.5 (M1 task 1)

DT = sl.DT                                                      # 3600 s, hourly snapshots
TILE_FACE = 10
RENAME = {'Theta': 'Theta_k', 'Salt': 'Salt_k', 'W': 'W_k'}     # M3-Q6: merge by rename
# (OSN var, chunk var, its level dim): the k = 0 identity pairs
IDENTITY = (('Theta', 'Theta_k', 'k'), ('Salt', 'Salt_k', 'k'), ('W', 'W_k', 'k_l'))
FLUX_VARS = ('oceQnet', 'oceQsw', 'oceFWflx')
OSN_VARS = ot.CORE_VARS + ot.WIND_VARS                          # the nine §3.2 vars
MERGED_VARS = OSN_VARS + tuple(RENAME.values()) + FLUX_VARS + ('drF',)   # 16
MERGED_DIMS = ('time', 'j', 'i', 'i_g', 'j_g', 'k', 'k_l')
SHARED_COORDS = ('time', 'niter', 'j', 'i', 'XC', 'YC')
# local solar time for Figure 6: tile centre lon -120.5 -> UTC - 8.0 h (M2 task 3)
LON_TILE, UTC_OFFSET_H = -120.5, -8.0


# ---------------------------------------------------------------------------
# the pre-merge assertions and the two invariants
# ---------------------------------------------------------------------------
def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:                      # ints / datetimes: no NaN to care about
        return bool(np.array_equal(a, b))


def assert_shared_coords(osn, chunk):
    """``time`` (3600 s steps), ``niter``, ``j``, ``i``, ``XC``, ``YC`` equal in
    both stores, scalar ``face = 10`` in both (M2 task 6 verified them)."""
    for c in SHARED_COORDS:
        for name, d in (('OSN', osn), ('chunk', chunk)):
            if c not in d.coords:
                raise KeyError(f'{name} store has no coord {c!r}')
        if not _equal(osn[c].values, chunk[c].values):
            raise ValueError(f'coord {c!r} differs between the OSN and chunk stores')
    t = osn['time'].values
    if t.size > 1:
        steps = np.diff(t).astype('timedelta64[s]').astype('int64')
        if not np.all(steps == int(DT)):
            raise ValueError(f'time steps are not {DT:g} s: {np.unique(steps)}')
    for name, d in (('OSN', osn), ('chunk', chunk)):
        if 'face' not in d.coords or d['face'].ndim != 0 or int(d['face']) != TILE_FACE:
            raise ValueError(f'{name} store: face is not the scalar {TILE_FACE}')


def _hours_of(ds, hours):
    """Time indices to check: ``hours`` on a time-dimensioned dataset, or
    ``[None]`` (the snapshot itself) on a single hour."""
    if 'time' not in ds.dims:
        return [None]
    return [int(h) for h in (range(ds.sizes['time']) if hours is None else hours)]


def _at(da, t):
    return da if t is None else da.isel(time=t)


def assert_k0_identity(ds, hours=None):
    """Chunk ``Theta_k(k=0)``, ``Salt_k(k=0)``, ``W_k(k_l=0)`` bit-identical
    (NaN-aware) to OSN ``Theta``, ``Salt``, ``W`` for the time indices
    ``hours`` (all of them if None; the snapshot itself for a single hour).
    The two sources are the same model output (M2 tasks 4-6), so this is a
    cheap, strong check that the hours are aligned and the stores are what
    they claim.  Returns the indices checked."""
    idx = _hours_of(ds, hours)
    for t in idx:
        for osn_v, chunk_v, dim in IDENTITY:
            a = _at(ds[osn_v], t).values
            b = _at(ds[chunk_v], t).isel({dim: 0}).values
            if not _equal(a, b):
                bad = ('shapes differ' if a.shape != b.shape else
                       f'{int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))} cells differ')
                when = '' if t is None else f' at time index {t} ({str(ds["time"].values[t])[:19]})'
                raise ValueError(f'k=0 identity broken{when}: {chunk_v}({dim}=0) != {osn_v} '
                                 f'({bad}) -- a time offset or a rewritten store')
    return idx


def assert_flux_sign(ds, hours=None):
    """The three fluxes carry ``sign_convention`` = positive downward and the
    stored ``oceQsw >= 0`` wherever finite on the hours checked; otherwise
    **refuse** (an upward-positive or re-negated store must never reach
    ``surface_flux_term`` -- M2 task 6, item 2; M2-Q6 (a))."""
    for v in FLUX_VARS:
        if v not in ds:
            raise KeyError(f'no {v} in the dataset')
        sc = str(ds[v].attrs.get('sign_convention', ''))
        if not sc.startswith('positive downward'):
            raise ValueError(f'{v}: sign_convention attr {sc!r} is not "positive downward" -- '
                             'refusing (the store must be the M2 task 5 negated product)')
    for t in _hours_of(ds, hours):
        sw = _at(ds['oceQsw'], t).values
        if np.any(sw < 0):
            when = '' if t is None else f' at time index {t}'
            raise ValueError(f'oceQsw < 0 on {int(np.sum(sw < 0))} cells{when}: the store is '
                             'upward-positive (or negated twice) -- refusing')


# ---------------------------------------------------------------------------
# open and merge
# ---------------------------------------------------------------------------
def _open(x):
    return x if isinstance(x, xr.Dataset) else xr.open_zarr(x)


def _open_grid(g):
    if isinstance(g, xr.Dataset):
        g = ensure_comodo_attrs(g.load())
        return g if 'face' in g.dims else g.expand_dims('face')
    return ot.open_grid(g, with_face=True)


def _open_masks(m):
    return m if isinstance(m, xr.Dataset) else mk.open_masks(m)


def open_inputs(raw=RAW_ZARR, chunk=CHUNK_ZARR, grid=GRID_ZARR, masks=MASKS_NC, *,
                hours=None, check_hours=None):
    """Open the §3.2 and §3.3 stores and merge them by rename (M3-Q6) ->
    ``(ds, grid_ds, grid, masks_ds)``: ``ds`` lazy, 16 vars on dims
    ``time, j, i, i_g, j_g, k, k_l``; ``grid_ds`` via ``osn_tiles.open_grid
    (with_face=True)``, ``grid`` via ``build_xgcm``, ``masks_ds`` via
    ``masking.open_masks``.  Paths or already-open Datasets are accepted.

    ``hours``: an optional time ``isel`` (slice or contiguous indices) applied
    to both stores before the merge.  ``check_hours``: the time indices of
    the merged dataset on which the k = 0 identity and the flux sign are
    checked now -- ``None`` means every hour requested through ``hours``, or
    hour 0 alone when the whole store is opened lazily (the others are
    checked by :func:`hour_pair` as they are loaded); ``'all'`` checks every
    hour.  The shared-coord assertions always run on everything opened.
    """
    osn, ch = _open(raw), _open(chunk)
    if hours is not None:
        osn, ch = osn.isel(time=hours), ch.isel(time=hours)
    assert_shared_coords(osn, ch)
    missing = [v for v in OSN_VARS if v not in osn] + [v for v in list(RENAME) + list(FLUX_VARS) + ['drF']
                                                         if v not in ch]
    if missing:
        raise KeyError(f'variables missing from the stores: {missing}')
    # the OSN store carries k = 0 and k_l = 0 as SCALAR coords (its fields are the surface
    # level); the chunk store has the 3-level k / k_l index dims.  The scalar would collide
    # with the index, and the surface level is k = 0 of the merged dims anyway: drop it.
    osn = osn.drop_vars([c for c in ('k', 'k_l') if c in osn.coords and osn[c].ndim == 0])
    ch = ch.rename(RENAME)
    # compat='override': the shared coords were just asserted equal (no second load);
    # join='exact': an index mismatch is an error, never a silent outer join with NaN
    ds = xr.merge([osn, ch], compat='override', join='exact', combine_attrs='drop_conflicts')
    if sorted(ds.data_vars) != sorted(MERGED_VARS):
        raise ValueError(f'merged vars {sorted(ds.data_vars)} != {sorted(MERGED_VARS)}')
    if set(ds.dims) != set(MERGED_DIMS):
        raise ValueError(f'merged dims {tuple(ds.dims)} != {MERGED_DIMS}')
    n = ds.sizes['time']
    if isinstance(check_hours, str) and check_hours == 'all':
        idx = list(range(n))
    elif check_hours is None:
        idx = list(range(n)) if hours is not None else [0]
    else:
        idx = [int(h) for h in check_hours]
    assert_flux_sign(ds, idx)
    assert_k0_identity(ds, idx)
    ds.attrs.update(
        merge='xr.merge([osn, chunk.rename(Theta->Theta_k, Salt->Salt_k, W->W_k)]) (M3-Q6, rename)',
        source_osn='in-memory' if isinstance(raw, xr.Dataset) else str(raw),
        source_chunk='in-memory' if isinstance(chunk, xr.Dataset) else str(chunk),
        k0_identity_checked=idx,
        k0_identity_policy=('chunk k=0 / k_l=0 bit-identical to OSN; checked on open for the hours '
                            'listed, and again by inputs.hour_pair on every hour it loads'),
        flux_sign_guard='sign_convention = positive downward and stored oceQsw >= 0 on the hours checked')
    grid_ds = _open_grid(grid)
    xgrid = ot.build_xgcm(grid_ds)
    masks_ds = _open_masks(masks)
    # the grid and the masks describe this tile: same XC, same (j, i) shape
    if not _equal(grid_ds['XC'].squeeze().values, ds['XC'].values):
        raise ValueError('grid XC differs from the stores\' XC -- not the same tile')
    if tuple(masks_ds['mask_analysis'].shape) != (ds.sizes['j'], ds.sizes['i']):
        raise ValueError(f'mask_analysis shape {masks_ds["mask_analysis"].shape} != '
                         f'{(ds.sizes["j"], ds.sizes["i"])}')
    return ds, grid_ds, xgrid, masks_ds


# ---------------------------------------------------------------------------
# accessors: the physics modules never index the store themselves
# ---------------------------------------------------------------------------
def W_k1(ds):
    """``ds.W_k.isel(k_l=1)``: the model's **cell-base** vertical velocity at
    the base of the 1 m top cell (source ``k_p1 = 1``; continuity to 7e-12
    m/s, M2 task 4) -- what ``vertical_term`` takes.  It already contains
    the ``dEta/dt`` part (~5e-5 m/s, tidal) and the convergence part
    ``+drF delta`` (coding §4.6).  **Never** ``k_l = 0``: that is ``dEta/dt``
    (corr 0.998 with the centred ``Eta`` difference; 0.903 / 0.690 at
    ``k_l`` 1 / 2), a free-surface signal, not a flux through the cell base.
    The ``k_l`` / ``Zl`` coords are dropped so nothing downstream can carry
    a level dim by mistake."""
    W = ds['W_k']
    if 'k_l' not in W.dims or W.sizes['k_l'] < 2:
        raise ValueError(f'W_k needs a k_l dim with at least 2 interfaces, got dims {W.dims}')
    w = W.isel(k_l=1, drop=True)
    op.assert_dims(w, tuple(d for d in W.dims if d != 'k_l'), 'W_k1')
    w.name = 'W_k1'
    w.attrs.update(units='m s-1', k_l=1, source_dim='k_p1', source_index=1,
                   Zl_m=float(ds['Zl'].values[1]) if 'Zl' in ds.coords else -1.0,
                   long_name='vertical velocity at the base of the top cell, W(k_l=1) = source W(k_p1=1)',
                   note='the cell-base velocity for vertical_term; W(k_l=0) is dEta/dt and must not be used')
    return w


def drF(ds):
    """``drF(k)`` [m] from the store as float64, ``[1.0, 1.14, 1.30]``;
    ``drF[0] = 1.0`` asserted (M2 tasks 4-5)."""
    v = np.asarray(ds['drF'].values, dtype='float64').reshape(-1)
    if v.size != ds.sizes['k'] or not np.isclose(v[0], 1.0):
        raise ValueError(f'drF {v} is not the k-level thickness with drF[0] = 1.0 m')
    return v


def Z(ds):
    """``Z(k)`` [m] cell-centre depths from the store as float64,
    ``[-0.5, -1.57, -2.79]``; ``Z[0] = -0.5`` asserted."""
    v = np.asarray(ds['Z'].values, dtype='float64').reshape(-1)
    if v.size != ds.sizes['k'] or not np.isclose(v[0], -0.5):
        raise ValueError(f'Z {v} is not the k-level centre depth with Z[0] = -0.5 m')
    return v


def fluxes(ds):
    """``(oceQnet, oceQsw, oceFWflx)`` **as stored -- no negation**: the store
    is downward-positive (negated at write from the source's upward-positive
    data, M2-Q6 (a)), so ``oceQnet > 0`` warms the ocean, ``oceQsw >= 0``,
    ``oceFWflx < 0`` is net evaporation.  The ``sign_convention`` attr is
    re-checked and the ``forcing_note`` (6-hourly, linearly interpolated
    forcing) propagated on every array."""
    assert_flux_sign(ds, hours=())                  # attrs only; the values were checked at load
    note = ds.attrs.get('forcing_note', '')
    out = []
    for v in FLUX_VARS:
        da = ds[v]
        op.require_centred(da, v)
        da = da.copy()
        da.attrs.setdefault('forcing_note', note)
        da.attrs['negated_here'] = 'no: stored downward-positive (M2 task 5); do not negate again'
        out.append(da)
    return tuple(out)


def wind(ds, grid_ds):
    """``(oceTAUX, oceTAUY, KPPhbl)`` with the stresses **re-masked** with
    ``hFacW`` / ``hFacS``: the OSN store delivers them on the centred mask,
    leaving 922 / 565 finite values per hour on the staggered land (M0 task
    3), which any stress derivative would difference across.  ``KPPhbl`` as
    stored (metres; finite on every ocean cell, M2 task 2)."""
    tx, ty, kpp = ds['oceTAUX'], ds['oceTAUY'], ds['KPPhbl']
    op.require_u_point(tx, 'oceTAUX')
    op.require_v_point(ty, 'oceTAUY')
    op.require_centred(kpp, 'KPPhbl')
    out = []
    for da, hf in ((tx, 'hFacW'), (ty, 'hFacS')):
        wet = mk._positional(grid_ds, hf) > 0          # 2-D on the field's own staggered points
        if wet.shape != da.shape[-2:]:
            raise ValueError(f'{hf} shape {wet.shape} != {da.name} shape {da.shape[-2:]}')
        r = da.where(xr.DataArray(wet, dims=da.dims[-2:]))
        op.assert_dims(r, da.dims, f'{da.name} re-masked')
        r.attrs.update(da.attrs, remasked_with=hf,
                       note='stored on the centred mask (M0 task 3); NaN where hFac == 0 here')
        out.append(r)
    return out[0], out[1], kpp


# ---------------------------------------------------------------------------
# the hour pair and the midpoint
# ---------------------------------------------------------------------------
def time_mid(ds, t0):
    """``time[t0] + 30 min``, the trajectory-midpoint time: the derived
    store's coord and Figure 6's local-solar axis (``UTC - 8.0 h``).  Asserts
    the pair is one hour apart."""
    t = ds['time'].values
    t0 = int(t0)
    if t0 < 0 or t0 + 1 >= t.size:
        raise IndexError(f't0 = {t0} has no successor in {t.size} hours')
    step = (t[t0 + 1] - t[t0]) / np.timedelta64(1, 's')
    if step != DT:
        raise ValueError(f'hours {t0}, {t0 + 1} are {step:g} s apart, not {DT:g}')
    return t[t0] + np.timedelta64(30, 'm')


def local_solar_hour(t, utc_offset_h=UTC_OFFSET_H):
    """Hours since local solar midnight for a UTC ``datetime64`` (or array),
    ``lon -120.5 -> UTC - 8.0 h``: ``KPPhbl`` max ~01 h, min ~13 h (M2 task 3)."""
    t = np.asarray(t, dtype='datetime64[ns]')
    sec = (t - t.astype('datetime64[D]')) / np.timedelta64(1, 's')
    return (sec / 3600.0 + utc_offset_h) % 24.0


def hour_pair(ds, t0):
    """The two snapshots ``t0`` and ``t0 + 1`` loaded in memory, with
    ``expand_dims('face')`` (the ``(face, j, i)`` layout every dbof operator
    expects, coding §3.2) and ``float64`` data variables (§1.2).  Both hours
    pass the k = 0 identity and the flux-sign guard here, where the arrays
    are already in memory -- this is how the lazy :func:`open_inputs` policy
    covers every hour a budget is computed from."""
    t0 = int(t0)
    time_mid(ds, t0)                                # the 3600 s step and the index range
    out = []
    for t in (t0, t0 + 1):
        h = ds.isel(time=t).load()
        assert_flux_sign(h)
        assert_k0_identity(h)
        h = h.expand_dims('face').astype('float64')   # data vars only; coords keep their dtype
        if 'face' not in h.dims or h.sizes['face'] != 1:
            raise AssertionError('expand_dims(face) did not give a length-1 face dim')
        h.attrs.update(ds.attrs, time_index=t, time=str(h['time'].values)[:19])
        out.append(h)
    return out[0], out[1]


def midpoint(f_t, f_tp1):
    """``0.5 (f_t + f_tp1)`` -- ``semilag.midpoint_time``: the field at the
    trajectory midpoint time, where ``F`` is evaluated and fronts selected."""
    return sl.midpoint_time(f_t, f_tp1)


# ---------------------------------------------------------------------------
# the same filter on b, U and V; the reduction rule
# ---------------------------------------------------------------------------
def filtered(hour, L_cells):
    """``(b, U, V)`` of one snapshot at filter scale ``L_cells``:
    ``operators.buoyancy`` (JMD95) then ``operators.lowpass`` at the **same**
    ``L`` on all three (coding §1.2; ``L = 0`` is the identity).  Dims
    asserted: ``b`` centred, ``U`` on ``(j, i_g)``, ``V`` on ``(j_g, i)``.
    Land and the tile edge are NaN and propagate through the filter."""
    b = op.buoyancy(hour)
    U = hour['U'].astype('float64')
    V = hour['V'].astype('float64')
    b, U, V = (op.lowpass(f, L_cells) for f in (b, U, V))
    op.require_centred(b, 'b')
    op.require_u_point(U, 'U')
    op.require_v_point(V, 'V')
    for f in (b, U, V):
        f.attrs['L_cells'] = int(L_cells)
    return b, U, V


def valid(masks, *fields):
    """The reduction rule at every ``L``: ``mask_analysis & isfinite(every
    field)`` -> ``(valid, n_lost)``, ``n_lost`` the analysis cells a NaN
    removed (0 at ``L <= 4``; up to 34 per pair at ``L = 8``, where the
    order-3 departure support leaves the low-passed field's finite part at
    the first analysis row -- M2 task 3; task 4 logs it).  ``edge_cells = 7``
    must not shrink (M1 task 6); the ``edge_cells = 13`` sensitivity is a
    second mask passed as a bool array.  ``masks``: the §3.5 Dataset or a
    bool ``(j, i)`` array; fields: DataArrays or arrays, length-1 dims
    (``face``, ``time``) squeezed."""
    ana = masks['mask_analysis'].values if isinstance(masks, xr.Dataset) else masks
    ana = np.asarray(ana, dtype=bool)
    v = ana.copy()
    for k, f in enumerate(fields):
        a = np.squeeze(np.asarray(getattr(f, 'values', f), dtype='float64'))
        if a.shape != ana.shape:
            raise ValueError(f'field {k} has shape {a.shape}, mask {ana.shape}')
        v &= np.isfinite(a)
    return v, int(ana.sum() - v.sum())
