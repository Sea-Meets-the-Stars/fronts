"""Two-tier cache for statistics results, plus a disk cache for big arrays.

Region statistics are exact and computed on the native grid, so a large
box costs real time.  It should cost it once.  Results live in memory for
the session and on disk between sessions, keyed on everything that affects
the answer.

:func:`array` is the same idea for whole grid-sized planes.  The global
stores are chunked one-chunk-per-channel, so there is no such thing as a
partial read: XC alone is 12960 x 17280 float32 = 0.9 GB, and
``GlobalGridZarrReader.XC`` is a property that re-reads on every access.
Caching those to ``.npy`` and handing back a memmap turns a repeated S3
download into a page fault.
"""

from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np

from fronts.viz.apps import config

_MEM: "OrderedDict[str, object]" = OrderedDict()
_MEM_MAX = 64


def make_key(*parts) -> str:
    """A stable key from any hashable-ish description of a computation."""
    raw = "|".join(repr(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def _disk_path(key: str) -> Path:
    d = config.CACHE_DIR / "stats"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.pkl"


def get(key: str, *, disk: bool = True):
    """Look up a cached result.  Returns ``None`` on a miss."""
    if key in _MEM:
        _MEM.move_to_end(key)
        return _MEM[key]

    if disk:
        path = _disk_path(key)
        if path.exists():
            try:
                with path.open("rb") as fh:
                    value = pickle.load(fh)
            except Exception:
                path.unlink(missing_ok=True)
                return None
            _remember(key, value)
            return value
    return None


def put(key: str, value, *, disk: bool = True) -> None:
    """Store a result in memory, and optionally on disk."""
    _remember(key, value)
    if disk:
        try:
            with _disk_path(key).open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass                      # a cache write failure is never fatal


def _remember(key, value):
    _MEM[key] = value
    _MEM.move_to_end(key)
    while len(_MEM) > _MEM_MAX:
        _MEM.popitem(last=False)


def array(key: str, build) -> np.ndarray:
    """A grid-sized array, built once and thereafter memory-mapped.

    *build* is only called on a miss.  The return is read-only: it is the
    cache, not a copy.
    """
    d = config.CACHE_DIR / "arrays"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.npy"

    if not path.exists():
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(build()))
        tmp.replace(path)
        trim()

    path.touch()                          # newest-used, for trim()
    return np.load(path, mmap_mode="r")


def trim(cap_bytes: int | None = None) -> int:
    """Evict least-recently-used cache files until under the cap.

    Returns the number of files removed.  Covers every tier, since they
    share one directory and one budget.
    """
    cap = config.CACHE_CAP_BYTES if cap_bytes is None else cap_bytes
    files = [p for p in config.CACHE_DIR.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if total <= cap:
        return 0

    removed = 0
    for p in sorted(files, key=lambda p: p.stat().st_mtime):
        if total <= cap:
            break
        size = p.stat().st_size
        try:
            p.unlink()
        except OSError:
            continue
        total -= size
        removed += 1
    return removed


def clear(*, disk: bool = False) -> None:
    """Drop the memory cache, and optionally the disk cache too."""
    _MEM.clear()
    if disk:
        d = config.CACHE_DIR / "stats"
        if d.exists():
            for p in d.glob("*.pkl"):
                p.unlink()
