"""Turning a lat/lon box into a selection on the native LLC grid.

The rect grid is **not** a regular lat/lon grid -- the 13 native faces are
stitched and rotated rather than interpolated, so ``XC`` and ``YC`` stay
two-dimensional and latitude spacing is not uniform.  Two things follow:

1. A box cannot be converted to index slices by arithmetic.
2. It should not be converted by nearest-neighbour search either.
   ``fronts.llc.coords`` does have such a routine, but it allocates a full
   ``12960 x 17280`` float64 distance array (~1.8 GB) per query point --
   fine for a one-off CLI call, unusable behind an interactive drag.

Instead we ask about coordinates directly::

    mask = (YC >= lat0) & (YC <= lat1) & (XC >= lon0) & (XC <= lon1)

One vectorised pass, exact on an irregular grid, and dask-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBox:
    """A lat/lon box.

    Longitudes are stored in the -180..180 convention.  A box that crosses
    the antimeridian has ``lon0 > lon1``; :meth:`wraps` reports that and
    :func:`bbox_mask` handles it.
    """

    lon0: float
    lat0: float
    lon1: float
    lat1: float

    @classmethod
    def globe(cls) -> "BBox":
        return cls(-180.0, -90.0, 180.0, 90.0)

    @classmethod
    def from_bounds(cls, bounds) -> "BBox":
        """Build from a HoloViews ``BoundsXY`` tuple ``(x0, y0, x1, y1)``.

        The stream gives corners in drag order, so they are sorted here.
        Longitudes are normalised into -180..180; latitudes are clipped.
        """
        x0, y0, x1, y1 = bounds
        lat0, lat1 = sorted((float(y0), float(y1)))
        lon0, lon1 = float(x0), float(x1)
        if lon1 < lon0:
            lon0, lon1 = lon1, lon0
        return cls(
            wrap180(lon0), max(lat0, -90.0), wrap180(lon1), min(lat1, 90.0)
        )

    @property
    def is_global(self) -> bool:
        return (
            self.lon0 <= -179.99
            and self.lon1 >= 179.99
            and self.lat0 <= -89.99
            and self.lat1 >= 89.99
        )

    def wraps(self) -> bool:
        """True when the box crosses the antimeridian."""
        return self.lon0 > self.lon1

    def label(self) -> str:
        """Short human-readable description, for the status bar."""
        if self.is_global:
            return "global"

        def _lat(v):
            return f"{abs(v):.1f}{'N' if v >= 0 else 'S'}"

        def _lon(v):
            return f"{abs(v):.1f}{'E' if v >= 0 else 'W'}"

        return (
            f"{_lat(self.lat0)}-{_lat(self.lat1)}, "
            f"{_lon(self.lon0)}-{_lon(self.lon1)}"
        )

    def key(self) -> tuple:
        """Hashable form, rounded, for cache keys."""
        return tuple(round(v, 4) for v in (self.lon0, self.lat0, self.lon1, self.lat1))


def wrap180(lon):
    """Normalise longitude(s) into -180..180."""
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def wrap360(lon):
    """Normalise longitude(s) into 0..360 (Pacific-centred display)."""
    return np.asarray(lon) % 360.0


def bbox_mask(XC: np.ndarray, YC: np.ndarray, box: BBox) -> np.ndarray:
    """Boolean mask of the grid cells inside *box*.

    Parameters
    ----------
    XC, YC : numpy.ndarray
        Two-dimensional longitude / latitude arrays on the native grid.
        Longitudes may be in either the -180..180 or 0..360 convention;
        they are normalised internally.
    box : BBox
        The region of interest.

    Returns
    -------
    numpy.ndarray
        Boolean array with the shape of ``XC``.  Works for numpy and dask
        arrays alike -- nothing here forces computation.

    Notes
    -----
    A global box short-circuits to an all-true mask, which avoids touching
    the coordinate arrays at all for the default view.
    """
    if box.is_global:
        return np.ones(XC.shape, dtype=bool)

    lon = wrap180(XC)
    in_lat = (YC >= box.lat0) & (YC <= box.lat1)

    if box.wraps():
        in_lon = (lon >= box.lon0) | (lon <= box.lon1)
    else:
        in_lon = (lon >= box.lon0) & (lon <= box.lon1)

    return in_lat & in_lon


def count_selected(XC: np.ndarray, YC: np.ndarray, box: BBox) -> int:
    """How many grid cells a box selects.

    Used to tell the user what a statistics request is about to cost
    before it runs.
    """
    if box.is_global:
        return int(np.prod(XC.shape))
    return int(bbox_mask(XC, YC, box).sum())
