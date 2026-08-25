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
    """Major-axis angle of a boolean mask, in 0-90 degrees.

    Re-exported from :mod:`fronts.front_tracking`, which needs the same
    measurement to decide whether a candidate front has turned too far to
    be the same one.  Two copies of this would be two conventions waiting
    to disagree.
    """
    from fronts import front_tracking

    return front_tracking.orientation_deg(mask)

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


def build(provider, chunk: str, track, field: str) -> FrontSeries:
    """Compute all three series for the tracked front, cached.

    Each step runs the same ingest pipeline the figures use, so the
    numbers and the pictures cannot disagree.

    *track* rather than a label: the label changes every step, so a series
    built from one label would be one point long.
    """
    label = track.anchor.label
    key = cache.make_key("evolution-series-v2", provider.mode, chunk, label,
                         field, tuple(sorted(track.labels.items())))
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
        step_label = track.label_at(step)
        if step_label is None:
            continue                        # tracking gap: leave it NaN
        try:
            scene = EP.build_step(provider, chunk, step, field, step_label)
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
    """Deprecated: use :func:`tracking.fronts_present` on one step.

    This counted how many steps each label appeared in and offered the
    survivors.  Labels are assigned per date, so it measured label reuse
    rather than front persistence -- and it read a 0.9 GB label plane for
    every step to do it, which is what made *Load chunk* take a quarter of
    an hour.  Kept only so an old caller fails loudly instead of silently
    doing the wrong thing slowly.
    """
    raise NotImplementedError(
        "common_labels followed front labels across steps, which is not "
        "meaningful -- labels are per date.  Pick a front at one step with "
        "tracking.fronts_present, then follow it with tracking.follow."
    )
