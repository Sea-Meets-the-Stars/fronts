"""Front objects for M4 (coding doc §4.10): finding per hour, then flow-informed tracking.

This module grows task by task through execution prompt 5.  **Task 1 (this file as first
written, 2026-10-26): `find_fronts_series`** -- finding config D applied to the derived
store's midpoint ``G`` at every pair, one integer label array per timestep.  The tracking,
mask advection, scoring, quality and strength functions of §4.10 are later tasks and are not
here yet.

The finding chain
-----------------
Prompt 5 names the path ``build_v5.py`` step 1 -> ``build.tile_find`` ->
``preproc/gradb2.generate_tile_gradb2``.  **Neither exists at this checkout** (M1 task 7a,
coding §2.5 note): the entry point is ``fronts.finding.algorithms.fronts_from_gradb2`` called
the way ``fronts/finding/run.py`` calls it, with config D's ``binary`` block read from
``fronts/finding/configs/finding_config_D.yaml`` -- **read, not assumed** (window 64, 85th
percentile, ``thresh_mode='pool'``, sharpen, despur ``Lspur=10``, ``min_size=7``,
connectivity 2; five of these differ from the function's defaults).

Three caller-side rules from M1 task 7a are applied, because the three ``fronts`` defects they
work around are still open (``tests/test_nan_finding.py`` keeps them as strict xfails):

1. ``n_workers`` is passed explicitly -- config D's ``'pool'`` mode raises ``TypeError``
   without it;
2. ``despur`` is run **here**, after the rest of the chain, and only if any front pixel
   survived -- ``skan`` raises on an empty skeleton;
3. ``fronts &= isfinite(G)`` afterwards -- ``remove_small_holes`` fills a NaN island that a
   front wraps around.  That removes exactly the island fill and changes nothing else.

The mask is applied as **NaN, never zero**: ``front_thresh`` builds its local 64 x 64
percentile from the window's *finite* cells, so a NaN cell can never be a front and never
moves a threshold; a zero would drag the percentile down and manufacture fronts along every
mask boundary (prompt 5, task 1).  Filling NaN before finding is unsafe for the same reason
(task 7a measured it).

Labels are ``int32``, 0 = background, ``skimage.measure.label`` at config D's connectivity,
**never interpolated** (decision Q14 -- only the boolean mask is ever advected, in task 2).

Timestamps
----------
Two formats are live in this project: ``front_tracking.parse_time`` wants
``'%Y-%m-%dT%H_%M_%S'`` (underscores) and ``dbof`` uses ``'%Y-%m-%d %H:%M:%S'``.  The dict
``find_fronts_series`` returns is keyed by the **underscore form of the pair's midpoint
time** (``'2012-07-02T00_30_00'``), which is the format ``follow()`` will consume; the
converters below are the one place the translation lives.
"""
from __future__ import annotations

import datetime as _dt
import json
import time as _time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from skimage import measure

from fronts.finding.algorithms import fronts_from_gradb2
from fronts.finding.despur import prune_short_spurs

import osn_tiles as ot

DATA_DIR = ot.DATA_DIR
CONFIG_DIR = Path(ot.__file__).resolve().parents[3] / 'fronts' / 'finding' / 'configs'

TRACK_FMT = '%Y-%m-%dT%H_%M_%S'      # fronts.front_tracking.parse_time (underscores)
DBOF_FMT = '%Y-%m-%d %H:%M:%S'       # dbof's DATE_FMT

N_WORKERS = 8                        # config D's 'pool' mode needs one; 9.6 s per tile at 8

# the two deprecation notices the fronts chain emits on every call (skimage 0.26 / numpy 2.5);
# silenced here so a 71-pair run does not print them 142 times
_CHAIN_NOISE = ('Parameter `min_size` is deprecated', 'Setting the shape on a NumPy array')


# ---------------------------------------------------------------------------
# timestamps: the one place the two formats meet
# ---------------------------------------------------------------------------
def _as_datetime(t) -> _dt.datetime:
    if isinstance(t, _dt.datetime):
        return t
    if isinstance(t, str):
        for fmt in (TRACK_FMT, DBOF_FMT, '%Y-%m-%dT%H:%M:%S'):
            try:
                return _dt.datetime.strptime(t[:19], fmt)
            except ValueError:
                pass
        raise ValueError(f'{t!r} is neither {TRACK_FMT!r} nor {DBOF_FMT!r}')
    return np.datetime64(t, 's').astype(object)          # datetime64[s] -> datetime.datetime


def to_track_time(t) -> str:
    """Any of ``np.datetime64`` / ``datetime`` / a dbof or ISO string ->
    ``'%Y-%m-%dT%H_%M_%S'``, the format ``front_tracking.parse_time`` wants."""
    return _as_datetime(t).strftime(TRACK_FMT)


def to_dbof_time(t) -> str:
    """... -> dbof's ``'%Y-%m-%d %H:%M:%S'``."""
    return _as_datetime(t).strftime(DBOF_FMT)


def from_track_time(s: str) -> np.datetime64:
    """``'%Y-%m-%dT%H_%M_%S'`` -> ``np.datetime64[s]`` (the inverse of :func:`to_track_time`)."""
    return np.datetime64(_dt.datetime.strptime(s, TRACK_FMT), 's')


# ---------------------------------------------------------------------------
# finding config D, read from the YAML
# ---------------------------------------------------------------------------
def config_path(label: str = 'D') -> Path:
    return CONFIG_DIR / f'finding_config_{label}.yaml'


def load_finding_config(label: str = 'D') -> dict:
    """The ``binary`` block of ``finding_config_<label>.yaml`` as
    ``fronts/finding/run.py`` passes it (``**bparam``) -- read every time, never cached
    from a copy of its numbers."""
    with open(config_path(label)) as f:
        cfg = yaml.safe_load(f)
    if cfg.get('label', label) != label:
        raise ValueError(f'{config_path(label)} says label {cfg.get("label")!r}, not {label!r}')
    return dict(cfg['binary'])


# ---------------------------------------------------------------------------
# one field
# ---------------------------------------------------------------------------
def find_fronts(G, mask=None, config: str = 'D', n_workers: int = N_WORKERS):
    """Config ``config`` on one ``G`` field -> ``(labels, fronts)``: ``int32`` labels
    (0 = background, 1..n, connectivity from the config) and the ``bool`` front mask.

    ``mask`` (bool ``(j, i)``, True = retained) is applied as **NaN**; land is already NaN
    in ``G``.  The three task-7a rules apply (module docstring)."""
    G = np.asarray(getattr(G, 'values', G), dtype='float64')
    G = np.squeeze(G)
    if G.ndim != 2:
        raise ValueError(f'G must be 2-D, got shape {G.shape}')
    if mask is not None:
        mask = np.asarray(getattr(mask, 'values', mask), dtype=bool)
        if mask.shape != G.shape:
            raise ValueError(f'mask shape {mask.shape} != G shape {G.shape}')
        G = np.where(mask, G, np.nan)
    finite = np.isfinite(G)
    cfg = load_finding_config(config)
    despur, Lspur = bool(cfg.pop('despur', False)), cfg.pop('Lspur', None)
    with warnings.catch_warnings():
        for msg in _CHAIN_NOISE:
            warnings.filterwarnings('ignore', message=msg)
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice')
        fronts = fronts_from_gradb2(G, n_workers=int(n_workers), despur=False, **cfg)
        fronts = np.asarray(fronts, dtype=bool)
        if despur and fronts.any():                       # rule 2: skan raises on an empty skeleton
            fronts = np.asarray(prune_short_spurs(fronts, Lspur=Lspur), dtype=bool)
    fronts &= finite                                      # rule 3: the island fill, and nothing else
    labels = measure.label(fronts, connectivity=int(cfg.get('connectivity', 2))).astype('int32')
    return labels, fronts


# ---------------------------------------------------------------------------
# the series
# ---------------------------------------------------------------------------
def fronts_path(L, config: str = 'D', out=None) -> Path:
    return DATA_DIR / f'tile330_fronts_{config}_L{int(L)}.zarr' if out is None else Path(out)


def _open(derived_zarr) -> xr.Dataset:
    if isinstance(derived_zarr, xr.Dataset):
        return derived_zarr
    return xr.open_zarr(str(derived_zarr))


def find_fronts_series(derived_zarr, mask, config: str = 'D', *, n_workers: int = N_WORKERS,
                       pairs=None, log=None) -> dict:
    """Config ``config`` on the derived store's midpoint ``G`` at every pair (coding §4.10)
    -> ``{time_mid as '%Y-%m-%dT%H_%M_%S': int32 label array (j, i)}``, in time order.

    ``derived_zarr``: a §3.4 store path or the opened Dataset (any ``L``).
    ``mask``: bool ``(j, i)`` (True = retained) or the §3.5 Dataset, from which
    ``mask_analysis`` is taken -- the M3 pool, so the object selection stays compatible with
    the per-pixel ``G_mid >= p90`` rule (M3 task 8, carried to M4).  Applied as NaN.
    ``pairs``: optional iterable of time indices (tests; a partial run).
    ``log``: optional ``callable(str)`` for one progress line per pair.
    """
    ds = _open(derived_zarr)
    if isinstance(mask, xr.Dataset):
        mask = mask['mask_analysis'].values
    mask = np.asarray(mask, dtype=bool)
    tmid = ds['time_mid'].values if 'time_mid' in ds.coords else ds['time'].values
    idx = range(ds.sizes['time']) if pairs is None else [int(p) for p in pairs]
    out = {}
    for n, t in enumerate(idx):
        t0 = _time.time()
        G = ds['G'].isel(time=t).values
        labels, _ = find_fronts(G, mask, config=config, n_workers=n_workers)
        key = to_track_time(tmid[t])
        if key in out:
            raise ValueError(f'duplicate midpoint time {key} at pair {t}')
        out[key] = labels
        if log is not None:
            log(f'pair {t:3d} {key}: {int(labels.max())} fronts, {int((labels > 0).sum())} px, '
                f'{_time.time() - t0:.1f} s  [{n + 1}/{len(idx)}]')
    return out


def series_stats(labels_by_time: dict, front_by_time=None) -> list:
    """One dict per time: the counts the log quotes.  ``front_by_time`` (optional, the
    store's p90 ``front`` bool per time, same keys or same order) adds the fraction of
    labelled pixels that are also p90 front pixels -- the compatibility number between the
    object selection and M3's per-pixel rule."""
    rows = []
    fb = None if front_by_time is None else list(front_by_time.values()
                                                 if isinstance(front_by_time, dict)
                                                 else front_by_time)
    for n, (key, lab) in enumerate(labels_by_time.items()):
        sizes = np.bincount(lab.ravel())[1:]
        row = dict(time=key, n_fronts=int(sizes.size), n_pixels=int(sizes.sum()),
                   size_median=float(np.median(sizes)) if sizes.size else 0.0,
                   size_max=int(sizes.max()) if sizes.size else 0,
                   n_ge_50=int((sizes >= 50).sum()), n_ge_100=int((sizes >= 100).sum()))
        if fb is not None:
            f = np.asarray(fb[n], bool)
            row['frac_on_p90'] = float(((lab > 0) & f).sum() / max((lab > 0).sum(), 1))
        rows.append(row)
    return rows


def write_fronts_series(labels_by_time: dict, out, *, like: xr.Dataset = None, attrs=None,
                        clobber: bool = False) -> str:
    """``{time_mid -> labels}`` -> one zarr: ``labels(time, j, i)`` int32, one chunk per
    time, ``time_mid`` the key decoded, ``time`` the pair's first hour (``time_mid`` - 30 min,
    the §3.2 convention) and ``XC``/``YC`` copied from ``like`` when given.  Written once,
    whole -- the series is 71 x 720 x 720 int32, small enough not to need the resumable
    per-pair append the derived stores use."""
    out = Path(out)
    if out.exists():
        if not clobber:
            raise FileExistsError(f'{out} exists (pass clobber=True)')
        import shutil
        shutil.rmtree(out)
    keys = list(labels_by_time)
    tm = np.array([from_track_time(k) for k in keys], dtype='datetime64[ns]')
    lab = np.stack([np.asarray(labels_by_time[k], dtype='int32') for k in keys])
    coords = {'time': ('time', tm - np.timedelta64(30, 'm')), 'time_mid': ('time', tm)}
    if like is not None:
        for c in ('j', 'i'):
            if c in like.coords:
                coords[c] = like[c].values
        for c in ('XC', 'YC'):
            if c in like.coords:
                coords[c] = (('j', 'i'), like[c].values)
    ds = xr.Dataset({'labels': (('time', 'j', 'i'), lab)}, coords=coords)
    ds['labels'].attrs.update(long_name='front labels, 0 = background', dtype='int32',
                              never_interpolated='decision Q14: only the bool mask is advected')
    ds.attrs.update(attrs or {})
    ds.attrs.update(n_times=len(keys), times_mid=keys, time_format=TRACK_FMT,
                    created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
                    git_commit=ot._git_commit(Path(__file__).resolve().parent))
    enc = {'labels': {'chunks': (1,) + lab.shape[1:]},
           'time': {'units': 'seconds since 2011-09-10', 'dtype': 'int64'},
           'time_mid': {'units': 'seconds since 2011-09-10', 'dtype': 'int64'}}
    ds.to_zarr(str(out), mode='w', encoding=enc, consolidated=True)
    return str(out)


def open_fronts_series(path) -> dict:
    """The inverse of :func:`write_fronts_series`: ``{time_mid -> int32 labels}``."""
    ds = xr.open_zarr(str(path))
    tm = ds['time_mid'].values
    lab = ds['labels'].values.astype('int32')
    return {to_track_time(t): lab[n] for n, t in enumerate(tm)}


def labels_at_factory(labels_by_time: dict):
    """``(labels_at, times)`` for ``front_tracking.follow``: ``labels_at(step) -> (H, W)``
    int labels and ``times`` the underscore-format keys, in order."""
    times = list(labels_by_time)
    arrays = [labels_by_time[t] for t in times]
    return (lambda step: arrays[int(step)]), times


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
