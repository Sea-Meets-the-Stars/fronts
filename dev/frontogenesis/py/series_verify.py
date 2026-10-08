""" ``verify_series``: is a raw series store the §3.2 product?

Run by M2 task 2 after the 72-hour pull and again by the acceptance audit.
Everything is read lazily and one hour at a time, so 72 hours cost one pass
over the chunk files (~650 x 2 MB) and a few seconds, not a 0.75 GB load.
No physics: the checks are the data contract (coding §3.2) and the M0
task-3 property that land is NaN exactly where the grid's ``hFac`` is 0.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from dbof.llc4320_ingestion.date_iterations import osn_date_to_iteration, DATE_FMT
from dbof.llc4320_ingestion.grid import COMODO_COORD_META

from osn_tiles import DATA_DIR, CORE_VARS, WIND_VARS, open_grid

# §3.2: variable -> dims, and the hFac whose zeros are its land
VAR_DIMS = {'Theta': ('time', 'j', 'i'), 'Salt': ('time', 'j', 'i'),
            'U': ('time', 'j', 'i_g'), 'V': ('time', 'j_g', 'i'),
            'W': ('time', 'j', 'i'), 'Eta': ('time', 'j', 'i'),
            'KPPhbl': ('time', 'j', 'i'), 'oceTAUX': ('time', 'j', 'i_g'),
            'oceTAUY': ('time', 'j_g', 'i')}
# oceTAUX/oceTAUY live on i_g/j_g but arrive masked with the centred mask
# (M0 task 3, §3.2); they are stored as they come, so that is what to check
LAND_MASK = {v: 'hFacC' for v in VAR_DIMS}
LAND_MASK.update(U='hFacW', V='hFacS')
COORDS = ('time', 'XC', 'YC', 'niter', 'face', 'k', 'k_l')
ATTRS = ('iterations', 'endpoint', 'stores', 'git_commit')
TIME_UNITS = 'seconds since 2011-09-10'
ITER_PER_HOUR = 144                                  # 25 s steps


def _as_dt64(ts: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(ts, DATE_FMT), 's')


def _check_time(ds: xr.Dataset, timestamps) -> dict:
    """Gaps and duplicates: the store's ``time`` against the request."""
    have = ds['time'].values.astype('datetime64[s]')
    want = np.array([_as_dt64(t) for t in timestamps], dtype='datetime64[s]')
    uniq, counts = np.unique(have, return_counts=True)
    missing = want[~np.isin(want, have)]
    extra = have[~np.isin(have, want)]
    ordered = bool(np.all(np.diff(have) > np.timedelta64(0, 's'))) if len(have) > 1 else True
    r = dict(n_store=int(len(have)), n_expected=int(len(want)),
             missing=missing.astype(str).tolist(), extra=extra.astype(str).tolist(),
             duplicates=uniq[counts > 1].astype(str).tolist(), ordered=ordered)
    r['ok'] = not (r['missing'] or r['extra'] or r['duplicates']) and ordered
    return r


def _check_schema(ds: xr.Dataset) -> dict:
    """Vars, dims, dtypes, chunks, coords, encodings and attrs of §3.2."""
    problems = []
    nj, ni = ds.sizes.get('j'), ds.sizes.get('i')
    want_vars = set(CORE_VARS + WIND_VARS)
    if set(ds.data_vars) != want_vars:
        problems.append(f'vars {sorted(ds.data_vars)} != {sorted(want_vars)}')
    for v, dims in VAR_DIMS.items():
        if v not in ds:
            continue
        if ds[v].dims != dims:
            problems.append(f'{v} dims {ds[v].dims} != {dims}')
        if ds[v].dtype != np.float32:
            problems.append(f'{v} dtype {ds[v].dtype} != float32')
        chunks = tuple(ds[v].encoding.get('chunks', ()))
        if chunks != (1, nj, ni):
            problems.append(f'{v} chunks {chunks} != {(1, nj, ni)}')
    for c in COORDS:
        if c not in ds.coords:
            problems.append(f'coord {c} missing')
    if 'face' in ds.coords and (ds['face'].ndim != 0 or int(ds['face']) != 10):
        problems.append(f'face is not the scalar 10: {ds["face"].values}')
    if 'niter' in ds.coords and ds['niter'].dims != ('time',):
        problems.append(f'niter dims {ds["niter"].dims} != (time,)')
    for c in ('XC', 'YC'):
        if c in ds.coords and ds[c].dims != ('j', 'i'):
            problems.append(f'{c} dims {ds[c].dims} != (j, i)')
    tenc = ds['time'].encoding
    if tenc.get('units') != TIME_UNITS or np.dtype(tenc.get('dtype', 'f8')) != np.int64:
        problems.append(f'time encoding {tenc.get("units")!r} / {tenc.get("dtype")} != '
                        f'{TIME_UNITS!r} / int64')
    for d in ('j', 'i', 'j_g', 'i_g'):
        if d not in ds.dims:
            problems.append(f'dim {d} missing')
            continue
        a = ds[d].attrs
        if not all(a.get(k) == v for k, v in COMODO_COORD_META[d].items()):
            problems.append(f'comodo attrs on {d}: {a}')
        if ds[d].dtype != np.int64:
            problems.append(f'{d} dtype {ds[d].dtype} != int64')
    for k in ATTRS:
        if k not in ds.attrs:
            problems.append(f'attr {k} missing')
    if ds.attrs.get('stores') not in (['llc_surf', 'llc_wind'], ('llc_surf', 'llc_wind')):
        problems.append(f'attr stores {ds.attrs.get("stores")} != [llc_surf, llc_wind]')
    return dict(ok=not problems, problems=problems)


def _land_masks(grid_ds: xr.Dataset, ds: xr.Dataset) -> dict:
    """``hFac == 0`` as bool arrays on the stored ``(j, i)`` layout."""
    g = grid_ds.squeeze('face') if 'face' in grid_ds.dims else grid_ds
    masks = {}
    for hf in ('hFacC', 'hFacW', 'hFacS'):
        m = np.asarray(g[hf].values) == 0
        shape = tuple(ds.sizes[d] for d in g[hf].dims)
        if m.shape != shape:
            raise ValueError(f'{hf} shape {m.shape} does not match the store {shape}')
        masks[hf] = m
    return masks


def _check_hours(ds: xr.Dataset, grid_ds: xr.Dataset) -> tuple:
    """One pass over the hours: land-NaN pattern per variable, ``KPPhbl``
    finite on ocean.  Returns the ``land_nan`` and ``KPPhbl`` results."""
    kpp = dict(present='KPPhbl' in ds)
    try:
        masks = _land_masks(grid_ds, ds)
    except (KeyError, ValueError) as e:
        return dict(ok=False, error=str(e), hours_checked=0), dict(ok=False, **kpp)
    vars_ = [v for v in VAR_DIMS if v in ds]
    worst = {v: 0 for v in vars_}
    first_bad = {}
    ocean = ~masks['hFacC']
    kpp_bad_hours = []
    for k in range(ds.sizes['time']):
        hour = ds[vars_].isel(time=k).load()        # 9 chunk reads
        ts = str(hour['time'].values.astype('datetime64[s]'))
        for v in vars_:
            n_bad = int((np.isnan(hour[v].values) != masks[LAND_MASK[v]]).sum())
            if n_bad:
                worst[v] = max(worst[v], n_bad)
                first_bad.setdefault(v, ts)
        if kpp['present'] and not np.all(np.isfinite(hour['KPPhbl'].values[ocean])):
            kpp_bad_hours.append(ts)
    land = dict(ok=not any(worst.values()), hours_checked=int(ds.sizes['time']),
                max_mismatch_cells=worst, first_bad_hour=first_bad,
                land_fraction=float(masks['hFacC'].mean()))
    kpp.update(nonfinite_on_ocean_hours=kpp_bad_hours,
               ok=kpp['present'] and not kpp_bad_hours)
    return land, kpp


def _check_niter(ds: xr.Dataset) -> dict:
    """Consecutive hours are 144 iterations apart, and ``niter`` is the
    OSN iteration of each timestamp (``osn_date_to_iteration``)."""
    if 'niter' not in ds.coords:
        return dict(ok=False, error='niter missing')
    niter = np.asarray(ds['niter'].values).astype('int64')
    steps = sorted(set(np.diff(niter).tolist())) if len(niter) > 1 else []
    times = ds['time'].values.astype('datetime64[s]').astype(str)
    expect = np.array([osn_date_to_iteration(t.replace('T', ' ')) for t in times], dtype='int64')
    r = dict(steps=steps, expected_step=ITER_PER_HOUR,
             matches_timestamps=bool(np.array_equal(niter, expect)),
             matches_attr=list(ds.attrs.get('iterations', [])) == niter.tolist())
    r['ok'] = all(s == ITER_PER_HOUR for s in steps) and r['matches_timestamps'] and r['matches_attr']
    return r


def verify_series(out_zarr, timestamps, grid_ds: xr.Dataset = None) -> dict:
    """Check a :func:`osn_tiles.pull_series` store against §3.2.

    Parameters
    ----------
    out_zarr : path-like
    timestamps : sequence of str
        The hours the store should hold, ``dbof`` format.
    grid_ds : xarray.Dataset, optional
        Source of ``hFacC``/``hFacW``/``hFacS``; default M0's
        ``tile330_grid.zarr``.

    Returns
    -------
    dict
        ``ok`` (everything passed) and one entry per check: ``time`` (gaps,
        duplicates, order), ``schema`` (vars, dims, dtypes, chunks, coords,
        time encoding, comodo attrs, root attrs), ``land_nan`` (every hour,
        every variable, NaN exactly where the hFac is 0; ``oceTAU*``
        against ``hFacC``), ``niter`` (steps of 144, equal to the OSN
        iteration of each timestamp and to the ``iterations`` attr) and
        ``KPPhbl`` (present and finite on ocean in every hour).
    """
    out = Path(out_zarr)
    if not out.exists():
        return dict(ok=False, error=f'{out} does not exist')
    ds = xr.open_zarr(out)
    if grid_ds is None:
        grid_ds = open_grid(DATA_DIR / 'tile330_grid.zarr', with_face=False)
    res = dict(time=_check_time(ds, timestamps), schema=_check_schema(ds),
               niter=_check_niter(ds))
    res['land_nan'], res['KPPhbl'] = _check_hours(ds, grid_ds)
    res['ok'] = all(r['ok'] for r in res.values())
    res['path'] = str(out)
    return res


def summarize(res: dict) -> str:
    """One line per check, for logs."""
    lines = [f'{res.get("path", "")}: {"OK" if res.get("ok") else "FAILED"}']
    for k, r in res.items():
        if isinstance(r, dict):
            detail = {kk: vv for kk, vv in r.items() if kk != 'ok'}
            lines.append(f'  {"ok  " if r.get("ok") else "FAIL"} {k}: {detail}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# the §3.3 chunk store (M2 task 5)
# ---------------------------------------------------------------------------
CHUNK_VARS = {'Theta': ('time', 'k', 'j', 'i'), 'Salt': ('time', 'k', 'j', 'i'),
              'W': ('time', 'k_l', 'j', 'i'), 'oceQnet': ('time', 'j', 'i'),
              'oceQsw': ('time', 'j', 'i'), 'oceFWflx': ('time', 'j', 'i'), 'drF': ('k',)}
CHUNK_COORDS = ('time', 'niter', 'mit_iteration', 'face', 'k', 'k_l', 'j', 'i', 'XC', 'YC',
                'Z', 'Zl')
CHUNK_ATTRS = ('source', 'levels', 'git_commit', 'iterations', 'endpoint', 'source_path',
               'flux_sign_convention', 'flux_sign_conversion', 'provenance')
FLUXES = ('oceQnet', 'oceQsw', 'oceFWflx')


def _check_chunk_schema(ds: xr.Dataset, k_max: int) -> dict:
    """Vars, dims, dtypes, chunks, coords, time encoding and attrs of §3.3."""
    problems = []
    nk, nj, ni = k_max + 1, ds.sizes.get('j'), ds.sizes.get('i')
    if set(ds.data_vars) != set(CHUNK_VARS):
        problems.append(f'vars {sorted(ds.data_vars)} != {sorted(CHUNK_VARS)}')
    for v, dims in CHUNK_VARS.items():
        if v not in ds:
            continue
        if ds[v].dims != dims:
            problems.append(f'{v} dims {ds[v].dims} != {dims}')
        if ds[v].dtype != np.float32:
            problems.append(f'{v} dtype {ds[v].dtype} != float32')
        want = tuple({'time': 1, 'k': nk, 'k_l': nk, 'j': nj, 'i': ni}[d] for d in dims)
        chunks = tuple(ds[v].encoding.get('chunks', ()))
        if chunks != want:
            problems.append(f'{v} chunks {chunks} != {want}')
    for d in ('k', 'k_l'):
        if ds.sizes.get(d) != nk:
            problems.append(f'{d} size {ds.sizes.get(d)} != {nk}')
        elif d in ds.coords and ds[d].values.tolist() != list(range(nk)):
            problems.append(f'{d} values {ds[d].values.tolist()}')
    for c in CHUNK_COORDS:
        if c not in ds.coords:
            problems.append(f'coord {c} missing')
    if 'face' in ds.coords and (ds['face'].ndim != 0 or int(ds['face']) != 10):
        problems.append(f'face is not the scalar 10: {ds["face"].values}')
    tenc = ds['time'].encoding
    if tenc.get('units') != TIME_UNITS or np.dtype(tenc.get('dtype', 'f8')) != np.int64:
        problems.append(f'time encoding {tenc.get("units")!r} / {tenc.get("dtype")}')
    for k in CHUNK_ATTRS:
        if k not in ds.attrs:
            problems.append(f'attr {k} missing')
    if ds.attrs.get('flux_sign_convention') != 'positive downward':
        problems.append(f'flux_sign_convention {ds.attrs.get("flux_sign_convention")!r}')
    for v in FLUXES:
        a = ds[v].attrs if v in ds else {}
        if not str(a.get('sign_convention', '')).startswith('positive downward'):
            problems.append(f'{v} sign_convention {a.get("sign_convention")!r}')
        if '+=down' not in str(a.get('source_sign_convention', '')):
            problems.append(f'{v} source_sign_convention {a.get("source_sign_convention")!r}')
        if 'negated' not in str(a.get('sign_conversion', '')):
            problems.append(f'{v} sign_conversion missing')
    if 'W' in ds and ds['W'].attrs.get('source_dim') != 'k_p1':
        problems.append('W source_dim attr != k_p1')
    return dict(ok=not problems, problems=problems)


def _check_chunk_hours(ds: xr.Dataset, land: np.ndarray, sw_tol: float) -> tuple:
    """One pass over the hours: land-NaN per variable and level, and the
    stored shortwave sign.  Returns the ``land_nan`` and ``sign`` results."""
    nk = ds.sizes['k']
    if land.shape != (nk, ds.sizes['j'], ds.sizes['i']):
        return (dict(ok=False, error=f'land shape {land.shape}', hours_checked=0),
                dict(ok=False, error='not checked'))
    vars_ = [v for v in ('Theta', 'Salt', 'W') + FLUXES if v in ds]
    worst, first_bad = {v: 0 for v in vars_}, {}
    sw_min, sw_max, sw_bad, sw_mean_max = np.inf, -np.inf, [], -np.inf
    for t in range(ds.sizes['time']):
        hour = ds[vars_].isel(time=t).load()
        ts = str(hour['time'].values.astype('datetime64[s]'))
        for v in vars_:
            a = hour[v].values
            n_bad = int((np.isnan(a) != (land if a.ndim == 3 else land[0])).sum())
            if n_bad:
                worst[v] = max(worst[v], n_bad)
                first_bad.setdefault(v, ts)
        if 'oceQsw' in hour:
            sw = hour['oceQsw'].values
            lo, hi = float(np.nanmin(sw)), float(np.nanmax(sw))
            sw_min, sw_max = min(sw_min, lo), max(sw_max, hi)
            sw_mean_max = max(sw_mean_max, float(np.nanmean(sw)))
            if lo < -sw_tol:
                sw_bad.append(ts)
    land_r = dict(ok=not any(worst.values()), hours_checked=int(ds.sizes['time']),
                  levels_checked=int(nk), max_mismatch_cells=worst, first_bad_hour=first_bad)
    sign = dict(oceQsw_min=sw_min, oceQsw_max=sw_max, oceQsw_tile_mean_max=sw_mean_max,
                hours_below_minus_tol=sw_bad, tol=sw_tol)
    sign['ok'] = 'oceQsw' in ds and not sw_bad and sw_max > 0
    return land_r, sign


def verify_chunk_series(out_zarr, timestamps, levels: dict = None, k_max: int = 2,
                        sw_tol: float = 1.0) -> dict:
    """Check a ``vertical.load_chunk_levels`` store against §3.3.

    Parameters
    ----------
    out_zarr : path-like
    timestamps : sequence of str
        The hours the store should hold, ``dbof`` format.
    levels : dict, optional
        ``land`` (bool ``(k, j, i)``, ``hFacC == 0`` at ``k = 0..k_max``) and
        ``drF`` (``k = 0..k_max``) -- the reference.  Default: read from the
        chunk store's ``grid.zarr`` (network, ~0.5 MB) with
        ``chunk_store._load_levels``, which also checks ``hFacC[0]``/``XC``/``YC``
        against M0's ``tile330_grid.zarr``.
    k_max : int
    sw_tol : float
        ``oceQsw`` (stored, downward-positive) may not go below ``-sw_tol``
        W m^-2 anywhere; it must be positive somewhere in the window.

    Returns
    -------
    dict
        ``ok`` and one entry per check: ``time`` (gaps, duplicates, order),
        ``schema`` (§3.3 vars/dims/dtypes/chunks/coords/time encoding/attrs,
        the flux sign attrs), ``niter`` (steps of 144, OSN iteration of each
        timestamp, equal to the ``iterations`` attr; ``mit_iteration ==
        niter - 10368``), ``drF`` (equal to the grid, ``drF[0] == 1.0``,
        ``Z[0] == -0.5``), ``land_nan`` (every hour, every level: NaN exactly
        where ``hFacC == 0``) and ``sign`` (stored ``oceQsw >= 0``).
    """
    out = Path(out_zarr)
    if not out.exists():
        return dict(ok=False, error=f'{out} does not exist')
    if levels is None:
        import chunk_store                 # the reader (moved from vertical.py, M3 task 2 / M3-Q7)
        levels = chunk_store._load_levels(chunk_store.make_fs(), chunk_store.CHUNK_PREFIX, k_max,
                                          local_grid=DATA_DIR / 'tile330_grid.zarr')
    ds = xr.open_zarr(out)
    res = dict(time=_check_time(ds, timestamps), schema=_check_chunk_schema(ds, k_max))
    niter = _check_niter(ds)
    if 'mit_iteration' in ds.coords and 'niter' in ds.coords:
        niter['mit_is_osn_minus_10368'] = bool(np.array_equal(
            ds['mit_iteration'].values, ds['niter'].values - 10368))
        niter['ok'] = niter['ok'] and niter['mit_is_osn_minus_10368']
    res['niter'] = niter
    drF = ds['drF'].values if 'drF' in ds else np.array([])
    ref = np.asarray(levels['drF'], dtype='f4')
    res['drF'] = dict(values=drF.tolist(), reference=ref.tolist(),
                      Z=ds['Z'].values.tolist() if 'Z' in ds.coords else None)
    res['drF']['ok'] = (drF.shape == ref.shape and bool(np.array_equal(drF, ref))
                        and drF.size > 0 and float(drF[0]) == 1.0
                        and 'Z' in ds.coords and float(ds['Z'][0]) == -0.5)
    res['land_nan'], res['sign'] = _check_chunk_hours(ds, np.asarray(levels['land']), sw_tol)
    res['ok'] = all(r['ok'] for r in res.values())
    res['path'] = str(out)
    return res
