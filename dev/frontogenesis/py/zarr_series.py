""" Resumable, per-hour atomic appends to a time-dimensioned zarr store.

The machinery under ``osn_tiles.pull_series`` (M2 task 1), written as a
separate module so ``vertical.load_chunk_levels`` (M2 task 5) can reuse it
rather than copy it.  Nothing here knows about OSN or the §3.2 schema; the
unit of work is "one slab along ``time``" of any dataset.

Why this is not just ``to_zarr(append_dim='time')`` in a loop
---------------------------------------------------------------
``to_zarr(append_dim=...)`` is not atomic.  Measured on xarray 2026.7 /
zarr 3.4 (scratch experiment, M2 task 1): the append resizes and writes
``time`` *first*, then each data variable in turn, and re-consolidates the
root metadata last.  A process killed mid-append therefore leaves ``time``
one hour longer than some of the variables, with the trailing chunk missing
(or half-written) for the rest -- exactly the half-hour that a naive
"skip what is in ``time``" resume would then treat as done.  Two other
facts shape the repair: ``zarr.Array.resize`` to a smaller shape does *not*
delete the chunks beyond it, and xarray drops the consolidated metadata
while the append is in flight, so an interrupted store may have no
consolidated view at all.

The design, then:

1. **Stage in memory, write once.**  The caller assembles the whole hour
   (every variable, every store it comes from) before :func:`append_hour`
   is called, so a network failure never touches the store.
2. **Append only the time-dimensioned variables.**  Everything without a
   ``time`` dim (``XC``, ``YC``, the index coords, scalars) is written once
   when the store is created and dropped from later appends; xarray would
   otherwise rewrite those chunks on every hour (measured), which is both
   wasted I/O and another thing a crash can corrupt.
3. **Repair on resume, before trusting ``time``.**  :func:`repair_trailing`
   opens the *array* metadata directly (not the possibly stale or absent
   consolidated copy), truncates every time-dimensioned array to the
   shortest one, then walks back from the trailing hour while any array's
   trailing slab is unwritten (chunk key missing), unreadable (half-written
   chunk) or entirely fill value -- deleting the orphan chunks that
   ``resize`` leaves behind -- and re-consolidates.  After it returns, the
   store is a clean prefix of complete hours and ``time`` can be believed.
4. **"Present" is read from the store's own ``time`` coord**, never from a
   side file (:func:`present_times`).

Transient failures are wrapped by :func:`with_retries`, with the sleep
injectable so tests run in milliseconds.
"""

import logging
import os
import time as _time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

logger = logging.getLogger('frontogenesis.zarr_series')

# a few attempts with growing waits: OSN hiccups are seconds to a minute
RETRY_BACKOFF_S = (5.0, 20.0, 60.0)


def _say(msg: str, log=None):
    """Progress goes to the module logger and, if given, a callback (the
    detached pull script writes a progress file through it)."""
    logger.info(msg)
    if log is not None:
        log(msg)


def with_retries(fn, *, attempts: int = 3, backoff=RETRY_BACKOFF_S, sleep=_time.sleep,
                 log=None, what: str = ''):
    """Call ``fn()`` up to ``attempts`` times, sleeping ``backoff[k]`` between
    tries; re-raises the last error.  Any ``Exception`` is treated as
    transient -- a genuine bug costs ``attempts`` tries, which is bounded,
    whereas classifying OSN's errors (botocore, aiohttp, OSError, kerchunk
    KeyErrors ...) would be fragile.  ``BaseException`` (KeyboardInterrupt,
    SystemExit) is not caught, so an operator can still stop a pull."""
    last = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001 -- see docstring
            last = e
            if k + 1 < attempts:
                wait = backoff[min(k, len(backoff) - 1)]
                _say(f'{what}: attempt {k + 1}/{attempts} failed ({type(e).__name__}: '
                     f'{e}); retrying in {wait:.0f} s', log)
                sleep(wait)
    _say(f'{what}: giving up after {attempts} attempts ({type(last).__name__}: {last})', log)
    raise last


def consolidate(path):
    """``zarr.consolidate_metadata`` without the v3 "not part of the spec"
    warning (harmless; M0 task 4).  xarray reads the consolidated copy by
    default, so it must be refreshed after any direct zarr edit."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*[Cc]onsolidated metadata.*')
        zarr.consolidate_metadata(path)


def _time_arrays(group) -> dict:
    """``{name: zarr.Array}`` for every array whose first dim is ``time``."""
    out = {}
    for name, arr in group.arrays():
        dims = arr.metadata.dimension_names or ()
        if dims and dims[0] == 'time':
            out[name] = arr
    return out


def _chunk_paths(root: Path, arr, t: int):
    """Filesystem paths of every chunk of ``arr`` that holds time index ``t``."""
    cs, shape = arr.chunks, arr.shape
    n_chunks = [int(np.ceil(s / c)) for s, c in zip(shape[1:], cs[1:])]
    for rest in np.ndindex(*n_chunks) if n_chunks else [()]:
        key = arr.metadata.encode_chunk_key((t // cs[0],) + tuple(int(r) for r in rest))
        yield root / arr.path / key


def _slab_complete(root: Path, arr, t: int) -> bool:
    """Is time index ``t`` of ``arr`` fully on disk?  Missing chunk, a chunk
    that fails to decode (killed mid-write), or an all-fill-value slab all
    count as incomplete.  The last test assumes a real slab is never
    entirely fill (land is ~31% of the tile, and ``time``/``niter`` are
    never the int fill 0)."""
    if not all(p.exists() for p in _chunk_paths(root, arr, t)):
        return False
    try:
        slab = np.asarray(arr[t])
    except Exception:                                # noqa: BLE001 -- corrupt chunk
        return False
    fill = arr.fill_value
    if fill is None:
        return True
    if np.issubdtype(slab.dtype, np.floating) and np.isnan(fill):
        return not np.all(np.isnan(slab))
    return not np.all(slab == fill)


def _truncate(root: Path, arrays: dict, n_keep: int):
    """Resize every time-dimensioned array to ``n_keep`` hours and delete
    the chunks ``resize`` leaves behind."""
    for arr in arrays.values():
        n_old = arr.shape[0]
        if n_old > n_keep:
            # chunks are deleted before the resize so the key encoding sees
            # the old shape; the files are the same either way
            for t in range(n_keep, n_old):
                for p in _chunk_paths(root, arr, t):
                    if p.exists():
                        os.remove(p)
            arr.resize((n_keep,) + tuple(arr.shape[1:]))


def repair_trailing(path, log=None) -> int:
    """Make ``path`` a clean prefix of complete hours (see module docstring).

    Returns the number of hours removed (0 when the store was already
    consistent, which is the common case).  Requires the store to be local
    (chunk files are checked and removed through the filesystem).
    """
    path = Path(path)
    if not path.exists():
        return 0
    g = zarr.open_group(path, mode='r+', use_consolidated=False)
    root = Path(g.store.root)
    arrays = _time_arrays(g)
    if not arrays:
        return 0
    lengths = {n: a.shape[0] for n, a in arrays.items()}
    n_before = max(lengths.values())
    n = min(lengths.values())
    if n < n_before:
        _say(f'repair: time-dim arrays disagree on length {lengths}; truncating to {n}', log)
        _truncate(root, arrays, n)
    # walk back while the trailing hour is incomplete in any array; normally
    # the first check passes and the loop exits at once
    while n > 0:
        bad = [name for name, arr in arrays.items() if not _slab_complete(root, arr, n - 1)]
        if not bad:
            break
        _say(f'repair: hour index {n - 1} incomplete in {bad}; truncating to {n - 1}', log)
        n -= 1
        _truncate(root, arrays, n)
    removed = n_before - n
    if removed:
        # xarray reads the consolidated copy by default; bring it up to date
        # (this also happens to restore it if the crash removed it)
        consolidate(path)
        _say(f'repair: store now holds {n} complete hours', log)
    return removed


def present_times(path) -> np.ndarray:
    """The store's own ``time`` coord as ``datetime64[s]`` (empty if the
    store does not exist).  Call :func:`repair_trailing` first."""
    path = Path(path)
    if not path.exists():
        return np.array([], dtype='datetime64[s]')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # the array metadata, not the consolidated copy: it is the ground
        # truth and the copy may be stale after an interrupted append
        ds = xr.open_zarr(path, consolidated=False)
    return ds['time'].values.astype('datetime64[s]')


def append_hour(path, ds_hour: xr.Dataset, encoding: dict = None, attrs: dict = None):
    """Append one slab along ``time`` (create the store if absent).

    Parameters
    ----------
    path : path-like
    ds_hour : xarray.Dataset
        Length-1 ``time`` dim, all variables.  On a first write everything
        is stored (coords, scalars, attrs); on an append only the
        time-dimensioned variables are written (design point 2).
    encoding : dict, optional
        For the first write only (xarray refuses encodings for existing
        variables).  ``time`` and the other 1-D ``time`` coords get a chunk
        of 1 so each hour is one chunk file in every array.
    attrs : dict, optional
        Replaces the store's root attrs.  xarray overwrites the group attrs
        with the appended dataset's on every append (measured), so the
        caller passes the *full* attrs for the state after this hour.
    """
    path = Path(path)
    ds_hour = ds_hour.copy()
    if attrs is not None:
        ds_hour.attrs = dict(attrs)
    with warnings.catch_warnings():
        # "consolidated metadata is not part of the zarr v3 spec": harmless, M0 task 4
        warnings.filterwarnings('ignore', message='.*[Cc]onsolidated metadata.*')
        if not path.exists():
            enc = dict(encoding or {})
            for name, var in ds_hour.variables.items():
                if var.dims == ('time',):
                    enc.setdefault(name, {})
                    enc[name].setdefault('chunks', (1,))
            path.parent.mkdir(parents=True, exist_ok=True)
            ds_hour.to_zarr(path, mode='w', encoding=enc)
        else:
            static = [v for v in ds_hour.variables if 'time' not in ds_hour[v].dims]
            ds_hour.drop_vars(static).to_zarr(path, append_dim='time')
    return str(path)
