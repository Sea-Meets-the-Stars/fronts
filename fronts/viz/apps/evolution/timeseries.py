"""Per-step front metrics for the Evolution time series.

Three series, computed once for a front across the whole window:

===  ==========================================================
 a   front length [km], from the main axis
 b   front orientation [deg], 0 = N–S, 90 = E–W
 c   statistics of the selected field over the front
===  ==========================================================

Series (c) is computed over the front's **curtain** — the field sampled
along the main axis at every depth in the clipped range — not over the
surface pixels. That is deliberate: the curtain is what figures (d)–(i)
show, so the line the cursor tracks describes the same thing the movie
does. A surface-only statistic would drift away from the figures as the
front tilts with depth.

Orientation matches ``fronts.properties.geometry.calculate_front_orientation``:
the angle of the mask's major axis from the row (north–south) direction,
taken as an absolute value in 0–90.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dcfield

import numpy as np

from fronts.viz.apps import config
from fronts.viz.apps.common import cache


@dataclass
class FrontSeries:
    """Metrics for one front across a chunk's window.

    Attributes
    ----------
    times : list of str
        Timestamps, one per step.
    steps : numpy.ndarray
        Step indices actually present (a front can vanish mid-window).
    length_km, orientation : numpy.ndarray
        Per-step geometry.  NaN where the front is absent.
    stats : dict of str -> numpy.ndarray
        Per-step field statistics, keyed by statistic name.
    field : str
    label : int
    """

    times: list
    steps: np.ndarray
    length_km: np.ndarray
    orientation: np.ndarray
    stats: dict = _dcfield(default_factory=dict)
    field: str = ""
    label: int = 0

    @property
    def n(self) -> int:
        return len(self.times)

    def present(self) -> np.ndarray:
        """Boolean mask of the steps where the front exists."""
        return np.isfinite(self.length_km)


def orientation_deg(mask: np.ndarray) -> float:
    """Major-axis angle of a boolean mask, in 0–90 degrees.

    Second moments rather than ``skimage.measure.regionprops``, so this
    carries no scikit-image dependency; the convention is the same one
    ``geometry.calculate_front_orientation`` uses.
    """
    jj, ii = np.nonzero(mask)
    if jj.size < 2:
        return float("nan")

    dj = jj - jj.mean()
    di = ii - ii.mean()
    cov_jj = float((dj * dj).mean())
    cov_ii = float((di * di).mean())
    cov_ji = float((dj * di).mean())

    # Measured from axis 0 (rows, north-south), so the denominator is
    # cov_jj - cov_ii and not the other way round.  With the terms
    # swapped a north-south front reads as 90 degrees, which is exactly
    # backwards -- and silently plausible, since the range is still 0-90.
    angle = 0.5 * np.arctan2(2.0 * cov_ji, cov_jj - cov_ii)
    return float(abs(np.degrees(angle)))


def _stats_of(values: np.ndarray) -> dict:
    """The statistic set the page can draw as lines."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {s: float("nan") for s in config.EVOLUTION_STAT_LINES}
    out = {}
    for name in config.EVOLUTION_STAT_LINES:
        if name == "mean":
            out[name] = float(np.mean(v))
        elif name == "median":
            out[name] = float(np.median(v))
        elif name.startswith("p"):
            out[name] = float(np.percentile(v, float(name[1:])))
    return out


def build(provider, chunk: str, label: int, field: str) -> FrontSeries:
    """Compute all three series for one front, cached.

    Each step runs the same ingest pipeline the figures use, so the
    numbers and the pictures cannot disagree.
    """
    key = cache.make_key("evolution-series-v1", provider.mode, chunk, label,
                         field)
    hit = cache.get(key)
    if hit is not None:
        return hit

    from fronts.viz.apps.evolution import pipeline as EP

    times = provider.chunk_timesteps(chunk)
    n = len(times)

    length = np.full(n, np.nan)
    orient = np.full(n, np.nan)
    stats = {s: np.full(n, np.nan) for s in config.EVOLUTION_STAT_LINES}

    for step in range(n):
        try:
            scene = EP.build_step(provider, chunk, step, field, label)
        except Exception:                                   # noqa: BLE001
            continue                                        # front absent here

        km = scene.metrics.get("dist_km")
        if km is not None and len(km):
            length[step] = float(km[-1])
        orient[step] = orientation_deg(scene.front_mask)

        from fronts.viz import curtains
        curtain = curtains.sample_curtain(scene.color, scene.axis_path)
        for name, value in _stats_of(curtain).items():
            stats[name][step] = value

    series = FrontSeries(times=list(times), steps=np.arange(n),
                         length_km=length, orientation=orient,
                         stats=stats, field=field, label=int(label))
    cache.put(key, series)
    return series


def common_labels(provider, chunk: str, *, min_steps: int = 4) -> list[int]:
    """Front labels that survive at least *min_steps* of the window.

    A label that appears in one frame cannot be followed through a movie,
    so the selector only offers the ones that persist.
    """
    times = provider.chunk_timesteps(chunk)
    counts: dict[int, int] = {}
    for step in range(len(times)):
        labels = provider.chunk_labels(chunk, step)
        for value in np.unique(labels):
            if value:
                counts[int(value)] = counts.get(int(value), 0) + 1
    return sorted((l for l, c in counts.items() if c >= min_steps),
                  key=lambda l: -counts[l])
