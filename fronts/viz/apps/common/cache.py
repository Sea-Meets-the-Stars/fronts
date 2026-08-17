"""Two-tier cache for statistics results.

Region statistics are exact and computed on the native grid, so a large
box costs real time.  It should cost it once.  Results live in memory for
the session and on disk between sessions, keyed on everything that affects
the answer.
"""

from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from pathlib import Path

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


def clear(*, disk: bool = False) -> None:
    """Drop the memory cache, and optionally the disk cache too."""
    _MEM.clear()
    if disk:
        d = config.CACHE_DIR / "stats"
        if d.exists():
            for p in d.glob("*.pkl"):
                p.unlink()
