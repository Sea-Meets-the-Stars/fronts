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


# ===========================================================================
# Task 2 (2026-10-27): flow-informed tracking
# ===========================================================================
# `front_tracking.follow` predicts the next position by extrapolating the CENTROID velocity
# from the last two sightings, and scores a candidate against the reference mask shifted
# rigidly to that prediction.  A buoyancy front is advected by the flow, and M1's departure
# point machinery (`semilag.departure_index`) gives the displacement of every cell over the
# hour -- so the reference mask can be pushed through the flow instead (decision Q14).
#
# Three things change, and nothing in `front_tracking.py` is edited:
#   1. the prediction is the ADVECTED BOOLEAN MASK (`advect_mask`): the last-found mask as a
#      float through the semi-Lagrangian interpolation, thresholded at 0.5.  The integer
#      label field is never interpolated;
#   2. the position term's distance field is built from that advected mask instead of the
#      centroid-shifted one, so "distance to the nearest point of where the front should be"
#      means where the FLUID put it;
#   3. `IoU(advected mask, candidate)` enters the score as one more term, through
#      `score_candidate`'s existing `weights` dict (`flow_weighted_score`), additive.
# `follow_flow` is `follow`'s walk with those three substitutions and, with `flow=None`,
# reproduces `front_tracking.follow` link for link (`test_follow_flow_without_flow_equals_follow`).

import pandas as pd
from scipy import ndimage

import semilag as sl
from fronts import front_tracking as ft

DT = sl.DT
MASK_ORDER = 1          # the mask is 0/1, not b: bilinear is monotone (no ringing) and a
                        # half-cell shift of a 1-px skeleton gives 0.5 on both neighbours,
                        # which `>= 0.5` keeps -- a thin front never vanishes under advection
FLOW_IOU_DILATE = 1     # cells; config D's fronts are 1 px wide, so a raw IoU of two skeletons
                        # one cell apart is 0 -- the IoU is taken on 1-cell dilated masks
                        # (offset 0 -> 1.0, 1 cell -> 0.5, 2 cells -> 0.2, 3+ -> 0)
FLOW_WEIGHTS = {**ft.WEIGHTS, 'flow_iou': 1.0}      # the flow term weighs as much as position
N_TOP = 10


# ---------------------------------------------------------------------------
# advection of a boolean mask
# ---------------------------------------------------------------------------
def advect_mask_by(mask_bool, di, dj, order=MASK_ORDER) -> np.ndarray:
    """The mask at ``t + dt`` given the displacement ``(di, dj)`` (cells) of the parcel
    *arriving* at each cell -- ``semilag.departure_index``'s convention -- i.e.
    ``mask(x - d)`` through ``semilag.interp_to_departure`` as a float, thresholded at 0.5.
    NaN (a NaN displacement or support) is False."""
    m = np.asarray(mask_bool, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f'advect_mask_by: need a 2-D mask, got {m.shape}')
    f = sl.interp_to_departure(m.astype('float64'), np.squeeze(np.asarray(di, 'float64')),
                               np.squeeze(np.asarray(dj, 'float64')), order,
                               allow_low_order=True)
    return np.where(np.isfinite(f), f, 0.0) >= 0.5


def advect_mask(mask_bool, u_c, v_c, grid_ds, dt=DT) -> np.ndarray:
    """Coding §4.10: the flow-predicted mask at ``t + dt`` from the centred model-basis
    velocities (``semilag.centre_velocities``), through ``semilag.departure_index`` (the
    midpoint-rule trajectory, cubic velocity interpolation) and :func:`advect_mask_by`.
    ``dt < 0`` predicts backwards.  **Never pass the label field** -- only a bool mask."""
    if np.asarray(mask_bool).dtype.kind not in 'b':
        raise TypeError('advect_mask: a BOOL mask, never the integer label field (Q14)')
    di, dj = sl.departure_index(u_c, v_c, grid_ds, dt=dt)
    return advect_mask_by(mask_bool, di.values, dj.values)


class Flow:
    """The model flow between consecutive pair midpoints, as displacement fields.

    Transition ``k`` joins pair ``k`` (midpoint ``k + 0.5`` h) to pair ``k + 1`` and uses the
    raw store's ``U``, ``V`` at integer hour ``k + 1``, the instant between them -- the case
    study's convention (M3 task 10 (iii)) and the budget's own.  ``displacement(k, +1)`` is
    ``departure_index`` with ``dt = +3600`` (pair ``k -> k + 1``), ``displacement(k, -1)`` the
    same with ``dt = -3600`` (pair ``k + 1 -> k``).  Fields are cached in memory and,
    when ``store`` is given, in ``data/tile330_flow.zarr`` (float32, one chunk per
    transition) so the 10 tracks and tasks 3-5 share one computation.
    """

    def __init__(self, raw_ds=None, grid_ds=None, grid=None, store=None, dt=DT):
        self.ds, self.grid_ds, self.grid, self.dt = raw_ds, grid_ds, grid, float(dt)
        self.store = None if store is None else Path(store)
        self._cache = {}
        self._zarr = xr.open_zarr(str(self.store)) if self.store is not None and \
            self.store.exists() else None
        if self._zarr is None and (raw_ds is None or grid_ds is None or grid is None):
            raise ValueError('Flow: pass (raw_ds, grid_ds, grid) or an existing store')
        self.n_transitions = (self._zarr.sizes['transition'] if self._zarr is not None
                              else raw_ds.sizes['time'] - 2)

    @property
    def km_per_px(self) -> float:
        """Mean cell size [km] from the grid (median ``dxC``, ``dyC``) -- what
        ``follow(km_per_px=)`` wants (coding §2.5: the 2.3 default is 25% too large here)."""
        if self.grid_ds is not None:
            dx, dy = sl._spacing_at_centres(self.grid_ds)
            return float(0.5 * (np.nanmedian(dx) + np.nanmedian(dy)) / 1000.0)
        return float(self._zarr.attrs['km_per_px'])

    def _compute(self, k, direction):
        h = self.ds.isel(time=int(k) + 1).load()
        u_c, v_c = sl.centre_velocities(h['U'], h['V'], self.grid_ds, self.grid)
        di, dj = sl.departure_index(u_c, v_c, self.grid_ds, dt=direction * self.dt)
        return (np.squeeze(np.asarray(di.values, 'float64')),
                np.squeeze(np.asarray(dj.values, 'float64')))

    def displacement(self, k, direction=+1):
        """``(di, dj)`` float64 ``(j, i)`` for transition ``k`` in ``direction`` (+1 / -1)."""
        k, direction = int(k), int(np.sign(direction))
        if not 0 <= k < self.n_transitions:
            raise IndexError(f'transition {k} not in 0..{self.n_transitions - 1}')
        key = (k, direction)
        if key not in self._cache:
            if self._zarr is not None:
                s = 'fwd' if direction > 0 else 'bwd'
                self._cache[key] = (self._zarr[f'di_{s}'].isel(transition=k).values.astype('f8'),
                                    self._zarr[f'dj_{s}'].isel(transition=k).values.astype('f8'))
            else:
                self._cache[key] = self._compute(k, direction)
        return self._cache[key]

    def advect(self, mask_bool, step_from, step_to) -> np.ndarray:
        """The flow-predicted mask at pair ``step_to`` from the mask at pair ``step_from``
        (adjacent pairs only)."""
        d = int(step_to) - int(step_from)
        if d == 1:
            di, dj = self.displacement(step_from, +1)
        elif d == -1:
            di, dj = self.displacement(step_to, -1)
        else:
            raise ValueError(f'advect: steps {step_from} -> {step_to} are not adjacent')
        return advect_mask_by(mask_bool, di, dj)

    def precompute(self, out=None, log=None, clobber=False) -> str:
        """Every transition, both directions -> ``out`` (default ``self.store``)."""
        out = Path(self.store if out is None else out)
        if out.exists() and not clobber:
            raise FileExistsError(f'{out} exists')
        n = self.n_transitions
        nj, ni = self.ds.sizes['j'], self.ds.sizes['i']
        arrs = {k: np.full((n, nj, ni), np.nan, 'float32')
                for k in ('di_fwd', 'dj_fwd', 'di_bwd', 'dj_bwd')}
        for k in range(n):
            t0 = _time.time()
            for direction, s in ((+1, 'fwd'), (-1, 'bwd')):
                di, dj = self._compute(k, direction)
                arrs[f'di_{s}'][k], arrs[f'dj_{s}'][k] = di, dj
                self._cache[(k, direction)] = (di, dj)
            if log is not None:
                f = np.isfinite(arrs['di_fwd'][k])
                mag = np.hypot(arrs['di_fwd'][k][f], arrs['dj_fwd'][k][f])
                log(f'flow transition {k:2d} (hour {k + 1}): finite {int(f.sum())}, |d| median '
                    f'{np.median(mag):.3f} max {mag.max():.2f} cells, {_time.time() - t0:.1f} s')
        t = self.ds['time'].values[1:n + 1]
        ds = xr.Dataset({k: (('transition', 'j', 'i'), v) for k, v in arrs.items()},
                        coords={'transition': np.arange(n), 'hour': ('transition', t),
                                'j': self.ds['j'].values, 'i': self.ds['i'].values})
        ds.attrs.update(
            convention='displacement (cells) of the parcel ARRIVING at each cell over dt, '
                       'semilag.departure_index; departure = arrival - d',
            transition='k joins pair k (midpoint k+0.5 h) to pair k+1, with U, V at hour k+1',
            fwd='dt = +3600 s: pair k -> k+1', bwd='dt = -3600 s: pair k+1 -> k',
            dt=self.dt, km_per_px=self.km_per_px, vel_order=3, n_iter=3,
            created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
            git_commit=ot._git_commit(Path(__file__).resolve().parent))
        enc = {k: {'chunks': (1, nj, ni)} for k in arrs}
        enc['hour'] = {'units': 'seconds since 2011-09-10', 'dtype': 'int64'}
        if out.exists():
            import shutil
            shutil.rmtree(out)
        ds.to_zarr(str(out), mode='w', encoding=enc, consolidated=True)
        self.store, self._zarr = out, xr.open_zarr(str(out))
        return str(out)


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------
def iou_dilated(a, b, r=FLOW_IOU_DILATE) -> float:
    """``IoU`` of two bool masks after dilating each by ``r`` cells (8-connected), computed
    on their joint bounding box.  ``r = 0`` is the plain IoU."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    r = int(r)
    ja, ia = np.nonzero(a)
    jb, ib = np.nonzero(b)
    if ja.size == 0 or jb.size == 0:
        return 0.0
    # the dilated masks cannot meet if the boxes are more than 2r apart: no dilation needed
    if (ja.min() > jb.max() + 2 * r or jb.min() > ja.max() + 2 * r
            or ia.min() > ib.max() + 2 * r or ib.min() > ia.max() + 2 * r):
        return 0.0
    box = (slice(max(min(ja.min(), jb.min()) - r, 0), max(ja.max(), jb.max()) + r + 1),
           slice(max(min(ia.min(), ib.min()) - r, 0), max(ia.max(), ib.max()) + r + 1))
    a, b = a[box], b[box]
    if r > 0:
        st = np.ones((3, 3), bool)
        a = ndimage.binary_dilation(a, st, iterations=int(r))
        b = ndimage.binary_dilation(b, st, iterations=int(r))
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def flow_weighted_score(candidate, reference, predicted, radius, predicted_mask=None,
                        weights=None, **kw):
    """Coding §4.10: ``front_tracking.score_candidate`` plus one term,
    ``flow_iou = 1 - IoU(predicted_mask, cand_mask)`` (dilated, :func:`iou_dilated`), weighted
    by ``weights['flow_iou']`` (default :data:`FLOW_WEIGHTS`) and folded into the same
    weighted mean the other terms use.  With ``predicted_mask=None`` (or no ``cand_mask``,
    or a zero weight) it *is* ``score_candidate``.  The position veto (``inf``) is kept."""
    weights = dict(FLOW_WEIGHTS if weights is None else weights)
    w_flow = float(weights.pop('flow_iou', 0.0))
    score, terms = ft.score_candidate(candidate, reference, predicted, radius,
                                      weights=weights, **kw)
    cand_mask = kw.get('cand_mask')
    if predicted_mask is not None and cand_mask is not None and w_flow > 0:
        terms['flow_iou'] = 1.0 - iou_dilated(predicted_mask, cand_mask)
        if np.isfinite(score):
            used = sum(weights[k] for k in terms if k in weights)
            score = (score * used + w_flow * terms['flow_iou']) / (used + w_flow)
    return float(score), terms


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------
def follow_flow(labels_at, times, anchor, flow=None, *, km_per_px=ft.DEFAULT_KM_PER_PX,
                max_score=ft.MAX_SCORE, weights=None, min_pixels=5):
    """``front_tracking.follow`` with the flow-predicted mask in place of the rigid
    centroid-velocity prediction (module note above).  ``flow`` is a :class:`Flow` (or
    anything with ``advect(mask, step_from, step_to)``); ``flow=None`` reproduces ``follow``
    exactly.  Across a gap the prediction keeps being advected, so the front can be
    re-acquired where the fluid carried it.  Each link's ``terms`` carry, under underscore
    keys, the flow-predicted centroid (``_pred_centre``), the chosen centroid
    (``_centre``), the number of candidate labels the predicted mask overlaps
    (``_n_overlap``, >= 2 flags a split / merge for task 4) and the predicted pixel count
    (``_n_pred``)."""
    stamps = [ft.parse_time(t) for t in times]
    n = len(times)
    anchor_labels = labels_at(anchor.step)
    anchor_shape = ft.describe(anchor_labels, anchor.label)
    track = ft.Track(anchor=anchor, labels={anchor.step: anchor.label},
                     centres={anchor.step: anchor.centre},
                     links={anchor.step: ft.Link(anchor.step, anchor.label, 0.0, {'anchor': 0.0})})
    for direction in (1, -1):
        reference, ref_mask = anchor_shape, anchor_labels == anchor.label
        pred_mask = ref_mask
        history = [(anchor.step, anchor.centre, stamps[anchor.step])]
        step = anchor.step + direction
        while 0 <= step < n:
            labels = labels_at(step)
            radius = ft._radius_px((stamps[step] - history[-1][2]).total_seconds(), km_per_px)
            extra = {}
            if flow is None:
                predicted = ft._predict(history, stamps[step])
                moved = ft._shift(ref_mask, predicted[0] - reference.centre[0],
                                  predicted[1] - reference.centre[1])
                pm = None
            else:
                pred_mask = flow.advect(pred_mask, step - direction, step)
                if pred_mask.any():
                    predicted, moved, pm = ft.centroid(pred_mask), pred_mask, pred_mask
                else:                                   # advected off the tile: plain fallback
                    predicted = ft._predict(history, stamps[step])
                    moved = ft._shift(ref_mask, predicted[0] - reference.centre[0],
                                      predicted[1] - reference.centre[1])
                    pm = None
                hit = np.unique(labels[pred_mask])
                extra = {'_pred_centre': tuple(map(float, predicted)),
                         '_n_overlap': int((hit > 0).sum()), '_n_pred': int(pred_mask.sum()),
                         '_overlap_labels': [int(v) for v in hit if v > 0]}
            ref_distance = ft._distance_field(moved)
            best, best_score, best_terms, best_shape = None, np.inf, {}, None
            for value in ft.fronts_present(labels, min_pixels=min_pixels):
                shape = ft.describe(labels, value)
                if not shape.valid:
                    continue
                mask = labels == value
                score, terms = flow_weighted_score(shape, reference, predicted, radius,
                                                   predicted_mask=pm, weights=weights,
                                                   ref_mask=ref_mask, cand_mask=mask,
                                                   ref_distance=ref_distance)
                if score < best_score:
                    best, best_score, best_terms, best_shape = value, score, terms, shape
            if best is not None and np.isfinite(best_score) and best_score <= max_score:
                track.labels[step] = int(best)
                track.centres[step] = best_shape.centre
                track.links[step] = ft.Link(step, int(best), float(best_score),
                                            {**best_terms, **extra,
                                             '_centre': tuple(map(float, best_shape.centre))})
                reference, ref_mask, pred_mask = best_shape, labels == best, labels == best
                history.append((step, best_shape.centre, stamps[step]))
            step += direction
    return track


def largest_labels(labels, n=N_TOP, min_pixels=5):
    """The ``n`` largest labels of one label array, largest first, with their sizes."""
    sizes = np.bincount(np.asarray(labels).ravel())
    sizes[0] = 0
    order = np.argsort(-sizes, kind='stable')
    return [(int(v), int(sizes[v])) for v in order[:n] if sizes[v] >= min_pixels]


def track_top_n(labels_by_time, times=None, n=N_TOP, flow=None, *, anchor_step=None,
                km_per_px=None, max_score=ft.MAX_SCORE, weights=None, min_pixels=5, log=None):
    """Coding §4.10: the ``n`` largest fronts at ``anchor_step`` (default the middle pair,
    so a track can run both ways), each followed through every step by :func:`follow_flow`
    -- flow-informed when ``flow`` is given, plain ``follow`` otherwise.  ``times`` default
    to the dict's keys (already the underscore format).  ``km_per_px`` defaults to the flow's
    grid value, else ``front_tracking``'s 2.3.  Returns the tracks, largest anchor first."""
    labels_at, keys = labels_at_factory(labels_by_time)
    times = keys if times is None else [to_track_time(t) for t in times]
    n_steps = len(times)
    anchor_step = n_steps // 2 if anchor_step is None else int(anchor_step)
    if km_per_px is None:
        km_per_px = flow.km_per_px if (flow is not None and hasattr(flow, 'km_per_px')) \
            else ft.DEFAULT_KM_PER_PX
    tracks = []
    for label, size in largest_labels(labels_at(anchor_step), n, min_pixels):
        t0 = _time.time()
        anchor = ft.anchor_at(labels_at(anchor_step), anchor_step, label)
        tr = follow_flow(labels_at, times, anchor, flow, km_per_px=km_per_px,
                         max_score=max_score, weights=weights, min_pixels=min_pixels)
        tr.anchor_size = size
        tracks.append(tr)
        if log is not None:
            st = tr.steps()
            log(f'track label {label} ({size} px) at step {anchor_step}: found {len(st)}/{n_steps} '
                f'steps, {st[0]}..{st[-1]}, {_time.time() - t0:.1f} s'
                + (' [flow]' if flow is not None else ' [plain]'))
    return tracks


# ---------------------------------------------------------------------------
# summaries, the comparison, persistence
# ---------------------------------------------------------------------------
def track_table(tracks, times) -> pd.DataFrame:
    """One row per track: anchor, steps found, the span in hours (first to last found
    step), gaps inside the span, worst link score, and how many links had the flow term."""
    stamps = [ft.parse_time(t) for t in times]
    rows = []
    for k, tr in enumerate(tracks):
        st = tr.steps()
        span_h = (stamps[st[-1]] - stamps[st[0]]).total_seconds() / 3600.0
        links = [l for s, l in tr.links.items() if s != tr.anchor.step]
        rows.append(dict(
            track=k, anchor_step=tr.anchor.step, anchor_label=tr.anchor.label,
            anchor_size=int(getattr(tr, 'anchor_size', 0)), n_found=len(st),
            first=st[0], last=st[-1], span_h=span_h,
            n_gaps_in_span=int(st[-1] - st[0] + 1 - len(st)),
            worst_score=max((l.score for l in links), default=0.0),
            median_score=float(np.median([l.score for l in links])) if links else 0.0,
            n_flow_links=sum('flow_iou' in l.terms for l in links),
            n_split_merge=sum(l.terms.get('_n_overlap', 0) >= 2 for l in links)))
    return pd.DataFrame(rows)


def compare_tracks(a, b) -> pd.DataFrame:
    """Step-by-step comparison of two track lists with the same anchors (plain vs flow):
    per track, the steps where both found the same label, where they found different
    labels, and where only one found anything.  ``different`` is the answer to "did the flow
    change which fronts were linked"."""
    rows = []
    for k, (ta, tb) in enumerate(zip(a, b)):
        sa, sb = set(ta.labels), set(tb.labels)
        both = sa & sb
        same = sum(ta.labels[s] == tb.labels[s] for s in both)
        rows.append(dict(track=k, anchor_label=ta.anchor.label, same=same,
                         different=len(both) - same, only_a=len(sa - sb), only_b=len(sb - sa),
                         steps_different=sorted(s for s in both if ta.labels[s] != tb.labels[s])))
    return pd.DataFrame(rows)


def tracks_to_json(tracks, times) -> list:
    out = []
    for tr in tracks:
        out.append(dict(anchor=dict(step=tr.anchor.step, label=tr.anchor.label,
                                    centre=list(tr.anchor.centre),
                                    size=int(getattr(tr, 'anchor_size', 0))),
                        times=list(times),
                        labels={str(s): int(v) for s, v in sorted(tr.labels.items())},
                        centres={str(s): list(map(float, c)) for s, c in sorted(tr.centres.items())},
                        links={str(s): dict(label=l.label, score=l.score, terms=l.terms)
                               for s, l in sorted(tr.links.items())}))
    return out


def tracks_from_json(obj) -> list:
    tracks = []
    for d in obj:
        a = d['anchor']
        anchor = ft.Anchor(step=int(a['step']), label=int(a['label']),
                           window=(slice(None), slice(None)), centre=tuple(a['centre']))
        tr = ft.Track(anchor=anchor,
                      labels={int(s): int(v) for s, v in d['labels'].items()},
                      centres={int(s): tuple(c) for s, c in d['centres'].items()},
                      links={int(s): ft.Link(int(s), int(l['label']), float(l['score']), l['terms'])
                             for s, l in d['links'].items()})
        tr.anchor_size = int(a.get('size', 0))
        tracks.append(tr)
    return tracks


# ===========================================================================
# Task 3 (2026-10-28): the reconciliation on Lagrangian-matched pixels
# ===========================================================================
# Front-mean G is a mean over a CHANGING pixel set -- fronts lengthen, split and merge -- so
# d/dt of "mean G over the label at t+1" minus "mean G over the label at t" carries a term
# from the set's own evolution and is not a material derivative (prompt 5 task 3; Q14).
#
# The matched quantity: every front pixel of the label at step t is a PARCEL at its cell
# centre; it is carried to step t+1 by the same flow the tracker uses (`Flow`, the budget's
# own midpoint-rule trajectory), and G at step t+1 is sampled at the parcel's arrival point
# (cubic).  The per-front measured rate is then
#       r_matched = [ mean_p G_{t+1}(x'_p) - mean_p G_t(x_p) ] / dt
# over the parcels that stay finite, and it is compared with the Phase-2 per-pixel rate
# averaged over the SAME parcels,
#       r_store   = mean_p 0.5 [ DGDt_t(x_p) + DGDt_{t+1}(x'_p) ],
# the two straddling pairs' `DGDt_semilag` (each centred on its own pair's midpoint; their
# mean is centred on the integer hour between the two steps, where r_matched is centred).
# This is the M3 case study's construction (§5.1 of the interim report) applied to every
# link of every track, and its known limit applies: the two are estimates of one Lagrangian
# rate on time bases half an hour apart (the store differences hour endpoints, the matched
# rate differences consecutive midpoint fields), which gave corr 0.901 and an r.m.s. gap of
# 0.443 of the stored rate on 450 parcels.  The tolerance is pre-declared from that:
RECON_RMS_TOL = 0.5      # rms(r_matched - r_store) / rms(r_store), pooled over every link
RECON_CORR_MIN = 0.8     # corr(r_matched, r_store), pooled
# Two other rates are computed for CONTRAST, never for the criterion: `r_mask`, the same
# difference on the thresholded advected MASK (task 2's object; 5% of pixels lost to
# rounding per hour), and `r_label`, the endpoint-label difference the prompt's "Do not"
# forbids as a criterion -- its gap to r_matched IS the set-evolution term.
BUDGET_TERMS = ('two_F', 'subfilter', 'vertical', 'surface_flux', 'residual')

from scipy.ndimage import map_coordinates


def sample_at(field, pos, order=3):
    """``field`` (2-D ``(j, i)``) at fractional positions ``pos`` ``(n, 2) = (j, i)``,
    cubic, NaN outside the array or where the support holds a NaN (the case study's
    ``m3_case.sample``, copied so this module does not import a script)."""
    field = np.asarray(field, 'float64')
    j, i = pos[:, 0], pos[:, 1]
    out = map_coordinates(np.nan_to_num(field, nan=0.0), [j, i], order=order,
                          mode='constant', cval=np.nan)
    bad = map_coordinates((~np.isfinite(field)).astype('f8'), [j, i], order=1,
                          mode='constant', cval=1.0)
    out[bad > 1e-6] = np.nan
    inside = ((j >= 0) & (j <= field.shape[0] - 1) & (i >= 0) & (i <= field.shape[1] - 1))
    out[~inside] = np.nan
    return out


def forward_positions(pos, di, dj, n_iter=8, order=3):
    """Arrival positions ``x`` of parcels departing from ``pos`` ``(n, 2) = (j, i)``, from the
    ARRIVAL displacement fields ``(di, dj)`` of :class:`Flow` (``departure = arrival - d``):
    the fixed point of ``x = pos + d(x)``, started at ``pos + d(pos)`` and iterated with
    ``d`` sampled at ``x`` (cubic); the contraction rate is ``|grad d|`` (~0.1 per
    iteration on the tile), so 8 iterations reach round-off.  NaN where the displacement is.  The forward form of the
    budget's backward map, checked against the case study's own forward integrator in
    ``m4_recon.py``."""
    pos = np.asarray(pos, 'float64')
    x = pos.copy()
    for _ in range(int(n_iter) + 1):
        d = np.stack([sample_at(dj, x, order), sample_at(di, x, order)], axis=1)
        x = pos + d
    return x


class DerivedFields:
    """Per-pair fields of a §3.4 store as float64 arrays, loaded on demand and cached:
    ``get('G', t)``."""

    def __init__(self, derived_zarr, names=('G', 'DGDt_semilag') + BUDGET_TERMS):
        self.ds = _open(derived_zarr)
        self.names, self._cache = tuple(names), {}

    def get(self, name, t):
        key = (name, int(t))
        if key not in self._cache:
            self._cache[key] = np.asarray(self.ds[name].isel(time=int(t)).values, 'float64')
        return self._cache[key]


def _mean(a):
    a = np.asarray(a, 'float64')
    return float(np.mean(a)) if a.size else np.nan


# The pre-registered construction is the one-sided forward difference on the label's own
# pixels (`centred=False, band=0`).  Its first run (2026-10-28, task-3 log) failed, and the
# diagnosis is a registration artefact: config D's fronts are 1-px ridges of G, so a parcel
# sampled at its cell at t sits on the cell maximum, and sampled at a fractional position at
# t+1 (cubic) it sits below the new ridge's maximum -- a negative bias of the size of the
# signal.  The store's own `DGDt_semilag` has the MIRROR bias on ridge cells (cell value at
# t+1 minus an interpolated departure value at t: positive).  Two options remove it, both
# kept as arguments and reported as variants, never silently swapped in:
#   centred=True  -- parcels at t sampled at t-1 (backward) and t+1 (forward); both samples
#                    are interpolated, so the artefact cancels to first order, the selection
#                    at t is symmetric in time (the case study's lesson), and the rate is
#                    centred on pair t's midpoint -- the SAME time base as `DGDt_t`, which is
#                    then taken at the parcels' cells (no half-hour averaging);
#   band=r        -- the pixel set is the label dilated by r cells (8-connected): ridge plus
#                    flanks, where the artefact has the opposite sign, so a mean over the
#                    band cancels it to first order in both estimates.  This is what
#                    `colocate_fronts_with_properties(dilation_radius=r)` measures.


def matched_link(step, label_t, label_tp1, labels_by_time, flow, fields, dt=DT,
                 terms=BUDGET_TERMS, *, centred=False, band=0):
    """One link of a track at ``step``: the matched rate, the stored rate on the same
    parcels, the budget terms on them, and the two contrast rates.  One dict (a row), or
    ``None`` when ``centred`` needs a step that does not exist."""
    keys = list(labels_by_time)
    n_steps = len(keys)
    if step + 1 >= n_steps or (centred and step == 0):
        return None
    lab0, lab1 = labels_by_time[keys[step]], labels_by_time[keys[step + 1]]
    S0 = lab0 == int(label_t)
    if band:
        S0 = ndimage.binary_dilation(S0, np.ones((3, 3), bool), iterations=int(band))
    jj, ii = np.nonzero(S0)
    pos = np.stack([jj, ii], axis=1).astype('float64')
    di, dj = flow.displacement(step, +1)
    x1 = forward_positions(pos, di, dj)
    G0, G1 = fields.get('G', step), fields.get('G', step + 1)
    g0, g1 = G0[jj, ii], sample_at(G1, x1)
    D0, D1 = fields.get('DGDt_semilag', step), fields.get('DGDt_semilag', step + 1)
    d0, d1 = D0[jj, ii], sample_at(D1, x1)
    alive = np.isfinite(g0) & np.isfinite(g1) & np.isfinite(d0) & np.isfinite(d1)
    if centred:
        dib, djb = flow.displacement(step - 1, -1)
        xm = forward_positions(pos, dib, djb)
        gm = sample_at(fields.get('G', step - 1), xm)
        alive &= np.isfinite(gm)
    n, n_alive = int(S0.sum()), int(alive.sum())
    row = dict(step=int(step), label_t=int(label_t),
               label_tp1=None if label_tp1 is None else int(label_tp1),
               centred=bool(centred), band=int(band),
               n_parcels=n, n_alive=n_alive, n_lost=n - n_alive,
               disp_median_cells=float(np.nanmedian(np.hypot(*(x1 - pos)[alive].T)))
               if n_alive else np.nan,
               G_mean_t=_mean(g0[alive]), G_mean_tp1_matched=_mean(g1[alive]),
               G_mean_tm1_matched=_mean(gm[alive]) if centred else np.nan,
               r_store_t=_mean(d0[alive]) if n_alive else np.nan,
               r_store_tp1=_mean(d1[alive]) if n_alive else np.nan)
    if not n_alive:
        row.update(r_matched=np.nan, r_store=np.nan, r_forward=np.nan)
    elif centred:
        row.update(r_matched=(_mean(g1[alive]) - _mean(gm[alive])) / (2 * dt),
                   r_store=_mean(d0[alive]),
                   r_forward=(_mean(g1[alive]) - _mean(g0[alive])) / dt)
    else:
        row.update(r_matched=(_mean(g1[alive]) - _mean(g0[alive])) / dt,
                   r_store=_mean(0.5 * (d0[alive] + d1[alive])),
                   r_forward=(_mean(g1[alive]) - _mean(g0[alive])) / dt)
    for name in terms:
        T0 = fields.get(name, step)
        t0 = T0[jj, ii]
        if centred:
            ok = alive & np.isfinite(t0)
            row[name] = _mean(t0[ok]) if ok.any() else np.nan
        else:
            t1 = sample_at(fields.get(name, step + 1), x1)
            ok = alive & np.isfinite(t0) & np.isfinite(t1)
            row[name] = _mean(0.5 * (t0[ok] + t1[ok])) if ok.any() else np.nan
    # contrast 1: the thresholded advected mask (task 2's prediction) as the pixel set
    adv = flow.advect(S0, step, step + 1)
    ga = G1[adv]
    ga = ga[np.isfinite(ga)]
    row.update(n_mask=int(adv.sum()),
               r_mask=(_mean(ga) - _mean(g0[np.isfinite(g0)])) / dt if ga.size else np.nan)
    # contrast 2: the endpoint labels -- NOT material (the prompt's "Do not")
    if label_tp1 is not None:
        S1 = lab1 == int(label_tp1)
        if band:
            S1 = ndimage.binary_dilation(S1, np.ones((3, 3), bool), iterations=int(band))
        gl = G1[S1]
        gl = gl[np.isfinite(gl)]
        row.update(n_label_tp1=int(S1.sum()),
                   r_label=(_mean(gl) - _mean(g0[np.isfinite(g0)])) / dt if gl.size else np.nan)
    else:
        row.update(n_label_tp1=0, r_label=np.nan)
    row['set_evolution'] = row['r_label'] - row['r_matched']
    return row


def reconcile_track(track, labels_by_time, flow, fields, dt=DT, terms=BUDGET_TERMS, *,
                    centred=False, band=0):
    """Every found step of ``track`` that has a successor -> a DataFrame of
    :func:`matched_link` rows, with the split / merge flag of the link copied from the
    track (``_n_overlap >= 2``, task 2).  A gap (step found, step + 1 not) still yields a
    row: the parcels exist and the matched rate is defined; only ``r_label`` is NaN."""
    rows = []
    for s in track.steps():
        nxt = track.labels.get(s + 1)
        row = matched_link(s, track.labels[s], nxt, labels_by_time, flow, fields, dt, terms,
                           centred=centred, band=band)
        if row is None:
            continue
        link = track.links.get(s + 1)
        row['split_merge'] = bool(link is not None and link.terms.get('_n_overlap', 0) >= 2)
        row['anchor_label'] = track.anchor.label
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(x, y):
    """``x`` against ``y`` (the reference): corr, r.m.s. gap over r.m.s. of ``y``, the
    amplitude ratio, n."""
    x, y = np.asarray(x, 'float64'), np.asarray(y, 'float64')
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return dict(n=int(x.size), corr=np.nan, rms_ratio=np.nan, amp_ratio=np.nan)
    return dict(n=int(x.size), corr=float(np.corrcoef(x, y)[0, 1]),
                rms_ratio=float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y ** 2))),
                amp_ratio=float(np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(y ** 2))),
                sign_agree=float(np.mean(np.sign(x) == np.sign(y))))


def reconcile_summary(df, rms_tol=RECON_RMS_TOL, corr_min=RECON_CORR_MIN) -> dict:
    """Criterion 4, pooled over every link of every track and per track: ``r_matched``
    against ``r_store`` with the pre-declared tolerance; the two contrasts reported with the
    same statistics, never gated."""
    out = dict(construction=dict(centred=bool(df['centred'].iloc[0]) if 'centred' in df else False,
                                 band=int(df['band'].iloc[0]) if 'band' in df else 0),
               tolerance=dict(rms_ratio_max=rms_tol, corr_min=corr_min,
                              basis='M3 task 10 (iii): the same two estimates on 450 parcels '
                                    'gave corr 0.901, rms gap 0.443 -- the half-hour time-base '
                                    'offset of the two Lagrangian rates'),
               n_links=int(len(df)), n_tracks=int(df['anchor_label'].nunique()),
               n_split_merge=int(df['split_merge'].sum()),
               parcels=dict(total=int(df['n_parcels'].sum()), lost=int(df['n_lost'].sum()),
                            lost_fraction=float(df['n_lost'].sum() / max(df['n_parcels'].sum(), 1))),
               matched_vs_store=_stats(df['r_matched'], df['r_store']),
               mask_vs_store=_stats(df['r_mask'], df['r_store']),
               label_vs_store=_stats(df['r_label'], df['r_store']),
               label_vs_matched=_stats(df['r_label'], df['r_matched']),
               set_evolution_over_matched=float(
                   np.sqrt(np.nanmean(df['set_evolution'] ** 2))
                   / np.sqrt(np.nanmean(df['r_matched'] ** 2))),
               two_F_vs_matched=_stats(df['two_F'], df['r_matched']),
               per_track={})
    m = out['matched_vs_store']
    out['passes'] = bool(np.isfinite(m['rms_ratio']) and m['rms_ratio'] <= rms_tol
                         and m['corr'] >= corr_min)
    for lab, g in df.groupby('anchor_label', sort=False):
        s = _stats(g['r_matched'], g['r_store'])
        s.update(label_vs_matched=_stats(g['r_label'], g['r_matched']),
                 n_split_merge=int(g['split_merge'].sum()),
                 passes=bool(np.isfinite(s['rms_ratio']) and s['rms_ratio'] <= rms_tol
                             and s['corr'] >= corr_min))
        out['per_track'][int(lab)] = s
    out['n_tracks_passing'] = int(sum(v['passes'] for v in out['per_track'].values()))
    return out
