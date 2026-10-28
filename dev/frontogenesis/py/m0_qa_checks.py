""" M0 task 5: the two §2 traps still unconfirmed, checked against real data.

Read-only checks on the 2 h store; nothing here is a physics module.

1. ``calculate_jacobian(u_x, v_y, ...)``: the arguments are the raw staggered
   ``U`` (on ``i_g``) and ``V`` (on ``j_g``).  Confirmed by (a) reading the
   source -- L164 hands them to ``rotate_vector_to_geographic``, whose
   ``interp_pair_to_center`` interpolates ``u_x`` along X and ``v_y`` along
   Y; (b) replicating the whole stencil in positional numpy from ``U``/``V``
   and matching to rounding; (c) the swapped call failing; (d) the Jacobian's
   trace agreeing with the flux-form divergence, which never rotates.
2. ``halo_mask.py:75`` (``return mask_f`` for a face with no ocean) is not
   reached for the tile: both branches of the L59 condition are True.  The
   bug is reproduced on an all-land face so its signature is on record.
3. The tile-edge rim: which rows/cols the ``_tile_indexer`` slicing plus
   xgcm's ``padding='fill'`` (fill value 0) invalidate.  Found empirically by
   recomputing on a cropped tile and diffing against the full-tile result.
"""

import numpy as np
import xarray as xr

from dbof.utils import native_gradient as ng
from dbof.preprocessing.halo_mask import llc_native_grid_halo_mask
import osn_tiles as ot


def _np(da):
    """Positional float64 array of a (face, j, i)-like DataArray."""
    return np.asarray(da.values, dtype='float64')[0]


def _interp_low(a, axis):
    """xgcm interp from the low staggered point to the centre, padding='fill'
    with fill 0: c[k] = 0.5*(s[k] + s[k+1]), s[n] := 0."""
    s_next = np.roll(a, -1, axis=axis)
    idx = [slice(None)] * a.ndim
    idx[axis] = -1
    s_next[tuple(idx)] = 0.0
    return 0.5 * (a + s_next)


def _diff_low(c, axis):
    """xgcm diff from centres to the low staggered point, fill 0:
    s[k] = c[k] - c[k-1], c[-1] := 0."""
    c_prev = np.roll(c, 1, axis=axis)
    idx = [slice(None)] * c.ndim
    idx[axis] = 0
    c_prev[tuple(idx)] = 0.0
    return c - c_prev


def jacobian_numpy(U, V, CS, SN, dxC, dyC):
    """Positional-numpy replica of ``calculate_jacobian`` (ECCO recipe:
    interp to centre, rotate, diff, interp, rotate).  Axis 0 is j, 1 is i."""
    u_c = _interp_low(U, 1)
    v_c = _interp_low(V, 0)
    u_lam = u_c * CS - v_c * SN
    v_phi = u_c * SN + v_c * CS
    out = []
    for f in (u_lam, v_phi):
        gx = _interp_low(_diff_low(f, 1) / dxC, 1)
        gy = _interp_low(_diff_low(f, 0) / dyC, 0)
        out += [gx * CS - gy * SN, gx * SN + gy * CS]
    return out


def check_jacobian_args(ds, grid):
    """Trap (i): ``calculate_jacobian``'s ``u_x, v_y`` really are ``U, V``."""
    J = [x.compute() for x in ng.calculate_jacobian(ds.U, ds.V, ds, grid)]
    J_np = jacobian_numpy(_np(ds.U), _np(ds.V), _np(ds.CS), _np(ds.SN),
                          _np(ds.dxC), _np(ds.dyC))
    res = {}
    for k, name in enumerate(('du_dx', 'du_dy', 'dv_dx', 'dv_dy')):
        a, b = _np(J[k]), J_np[k]
        ok = np.isfinite(a) & np.isfinite(b)
        res[name] = dict(
            nan_pattern_equal=bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
            max_abs_diff=float(np.max(np.abs(a[ok] - b[ok]))),
            max_abs=float(np.max(np.abs(a[ok]))),
            n=int(ok.sum()))
    # face 10 has CS = 0, SN = -1: u_east = V_c, v_north = -U_c, so the
    # zonal derivative of the zonal velocity is d(V_c)/d(model y)
    dudx = _np(J[0])
    dVc_dy = _interp_low(_diff_low(_interp_low(_np(ds.V), 0), 0) / _np(ds.dyC), 0)
    ok = np.isfinite(dudx) & np.isfinite(dVc_dy)
    res['du_dx_equals_dVc_dj'] = float(np.max(np.abs(dudx[ok] - dVc_dy[ok])))
    # the swapped call.  It does NOT raise: xgcm interps V (on j_g, i)
    # along X from i to i_g, the result lands on (j_g, i_g), and the CS/SN
    # multiply then broadcasts against (j, i) into a 4-D array -- 720^4
    # doubles, which on numpy-backed data is an eager out-of-memory kill.
    # Probe it lazily (dask) and record the dims only.
    lazy = ds.chunk({})
    try:
        out = ng.calculate_jacobian(lazy.V, lazy.U, lazy, grid)[0]
        res['swapped_call'] = f'no error; result dims {out.dims}, shape {out.shape}'
    except Exception as e:  # noqa: BLE001 - we want the type on record
        res['swapped_call'] = f'raises {type(e).__name__}'
    # trace of the Jacobian vs the flux-form divergence (no rotation, no
    # interpolation): same physics, independent stencil
    sv = ng.calculate_native_strain_vorticity(ds.U, ds.V, ds, grid)
    div_flux = _np(sv['divergence_center'].compute())
    tr = _np(J[0]) + _np(J[3])
    ok = np.isfinite(tr) & np.isfinite(div_flux)
    ok[:3, :] = ok[-3:, :] = ok[:, :3] = ok[:, -3:] = False
    r = np.corrcoef(tr[ok], div_flux[ok])[0, 1]
    slope = np.sum(tr[ok] * div_flux[ok]) / np.sum(div_flux[ok] ** 2)
    res['trace_vs_flux_divergence'] = dict(corr=float(r), slope=float(slope),
                                           n=int(ok.sum()))
    return res


def check_halo_branch(grid_ds, halo_km=12.0):
    """Trap (ii): ``halo_mask.py:75`` is not reached for the tile."""
    mask = grid_ds['hFacC'] == 0            # True = masked out (input convention)
    phi = np.ones(mask.shape[1:])
    phi[mask.values[0]] = -1.0
    res = dict(n_land=int((phi == -1).sum()), n_ocean=int((phi == 1).sum()),
               L59_branch_taken=bool((phi == -1).any() and (phi == 1).any()))
    hm = llc_native_grid_halo_mask(mask, grid_ds.dxC, grid_ds.dyC, halo_km)
    res['tile_call'] = dict(shape=tuple(hm.shape), dtype=str(hm.dtype),
                            retained=int(hm.sum()), halo_km=halo_km,
                            ndim_ok=hm.ndim == 3)
    # reproduce the bug on an all-land face: L75 returns the 2-D input,
    # all True, i.e. "keep everything" in the output convention
    hm2 = llc_native_grid_halo_mask(xr.ones_like(mask), grid_ds.dxC,
                                    grid_ds.dyC, halo_km)
    res['all_land_face'] = dict(shape=tuple(hm2.shape), all_true=bool(hm2.all()))
    return res


def check_edge_rim(ds, grid, b, crop=32):
    """Trap (iii): which tile-edge rows/cols are invalid.

    Recompute G (both stencils) and the Jacobian on the tile cropped by
    ``crop`` cells on every side; where the cropped result differs from the
    full-tile result at the same cell, the cropped domain's edge has
    contaminated it.  The set of contaminated rows/cols, measured from each
    edge, is the invalid rim -- and it applies equally to the full tile's
    own edges, where no larger domain exists to compare against.
    """
    sl = dict(j=slice(crop, -crop), i=slice(crop, -crop),
              j_g=slice(crop, -crop), i_g=slice(crop, -crop))
    g_c = ds.isel(**sl)
    grid_c = ot.build_xgcm(g_c)
    b_c = b.isel(j=sl['j'], i=sl['i'])

    def both(f):
        full = _np(f(ds, grid, b))[crop:-crop, crop:-crop]
        part = _np(f(g_c, grid_c, b_c))
        with np.errstate(invalid='ignore'):
            bad = ~np.isclose(full, part, rtol=1e-9, atol=0, equal_nan=True)
        n, m = bad.shape[0], 8
        # rows/cols (offset from each edge) holding a differing cell, judged
        # away from the perpendicular edges so one rim does not list the other
        rows = [r for r in range(n) if bad[r, m:-m].any()]
        cols = [c for c in range(n) if bad[m:-m, c].any()]
        fin = np.isfinite(full) | np.isfinite(part)
        # fraction of finite cells changed at offsets 0..5 from each edge
        def frac(take):
            return [float(bad_k.sum() / max(f_k.sum(), 1))
                    for bad_k, f_k in ((take(bad, k), take(fin, k)) for k in range(6))]
        return dict(
            low_j=[r for r in rows if r < n // 2],
            high_j=[n - 1 - r for r in rows if r >= n // 2],
            low_i=[c for c in cols if c < n // 2],
            high_i=[n - 1 - c for c in cols if c >= n // 2],
            n_bad=int(bad.sum()),
            frac_changed={
                'low j': frac(lambda a, k: a[k, m:-m]),
                'high j': frac(lambda a, k: a[n - 1 - k, m:-m]),
                'low i': frac(lambda a, k: a[m:-m, k]),
                'high i': frac(lambda a, k: a[m:-m, n - 1 - k])})

    def G_sq(d, gr, bb):
        return ng.calculate_grad_squared_tracer(bb, d, gr).compute()

    def G_comp(d, gr, bb):
        bx, by = ng.calculate_native_gradient_tracer(bb, d, gr)
        return (bx ** 2 + by ** 2).compute()

    def J_any(d, gr, bb):
        J = ng.calculate_jacobian(d.U, d.V, d, gr)
        return sum(abs(x) for x in J).compute()   # union of the four rims

    return dict(G_squared=both(G_sq), G_components=both(G_comp),
                jacobian=both(J_any))


def fill_value_is_zero(ds, grid, b):
    """xgcm ``padding='fill'`` with ``fill_value=None`` pads with 0: the
    low-edge X diff of b equals b itself."""
    d = _np(grid.diff(b, 'X').compute())
    bb = _np(b)
    ok = np.isfinite(bb[:, 0])
    return float(np.max(np.abs(d[ok, 0] - bb[ok, 0])))
