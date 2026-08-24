"""Following one front through a chunk window.

Front labels are assigned independently at every timestep, so the same
physical front carries a different label in every frame. A movie that
follows a *label* will jump to an unrelated front. So the front is
identified once, at an anchor step, and re-found at every other step by
**where it is** rather than by what it is called.

Two things make that work, and both matter:

* The window is **frozen** at the anchor. Every frame is cropped to the
  same lat/lon box, so the figures share axes and a colour range instead
  of the frame chasing the front. The window is a *display* crop only --
  it does not gate the search, because a front that advects out of shot
  over a week is still the same front. ``Track.first_escape`` reports
  when that happens rather than the track quietly ending there.

* The search radius scales with **elapsed time, not step count.** A chunk
  is a week of daily snapshots wrapped around one intensive day, so
  consecutive steps can be one hour apart or twenty-four. At one hour a
  front barely moves and mask overlap is a strong signal; across a day it
  can travel 40 km and overlap nothing. One fixed radius would either
  lose the front across the daily gaps or grab a neighbour inside the
  dense day.

Where no candidate is close enough, the step is a **gap**. A movie with a
hole in it is honest; a movie that confidently highlights the wrong front
is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

#: Plausible upper bound on how fast a front's centroid moves, m/s. Only
#: used to size the search radius, so it should be generous -- being too
#: small drops real links across the daily gaps.
MAX_DRIFT_MS = 0.6

#: Floor on the search radius, in pixels. Even at zero elapsed time the
#: centroid of a re-labelled front shifts a little, because the labelling
#: is per-date and the front's extent is not identical between steps.
MIN_RADIUS_PX = 4.0

#: Nominal cell size. The LLC4320 grid is ~1/48 degree, so ~2.3 km at the
#: equator and less towards the poles; callers that care pass the real
#: value measured from the chunk's own coordinates.
DEFAULT_KM_PER_PX = 2.3


def parse_time(stamp: str) -> datetime:
    """``2012-07-03T12_00_00`` -> datetime."""
    return datetime.strptime(stamp, "%Y-%m-%dT%H_%M_%S")


def centroid(mask: np.ndarray) -> tuple[float, float]:
    """``(j, i)`` centre of a boolean mask. NaNs if it is empty."""
    js, iss = np.nonzero(mask)
    if not len(js):
        return float("nan"), float("nan")
    return float(js.mean()), float(iss.mean())


@dataclass
class Anchor:
    """The front we are following, and the frame we are following it in.

    Attributes
    ----------
    step : int
        Step the front was picked on.
    label : int
        Its label *at that step only*.
    window : tuple of slice
        ``(j_slice, i_slice)`` on the chunk, frozen for the whole movie.
    centre : tuple of float
        ``(j, i)`` centroid at the anchor step, in chunk coordinates.
    """

    step: int
    label: int
    window: tuple[slice, slice]
    centre: tuple[float, float]


@dataclass
class Track:
    """Which label is our front, at each step.

    ``centres`` is kept alongside so the page can say when the front
    leaves the frozen window -- which is a display problem, not a
    tracking failure, and worth reporting rather than hiding.
    """

    anchor: Anchor
    labels: dict[int, int] = field(default_factory=dict)
    centres: dict[int, tuple[float, float]] = field(default_factory=dict)

    def label_at(self, step: int) -> int | None:
        """The label at *step*, or ``None`` where the front was lost."""
        return self.labels.get(int(step))

    def steps(self) -> list[int]:
        """Steps where the front was found, in order."""
        return sorted(self.labels)

    def gaps(self, n_steps: int) -> list[int]:
        """Steps with no front, so a caller can say so rather than guess."""
        return [s for s in range(n_steps) if s not in self.labels]

    def first_escape(self, window=None) -> int | None:
        """First step whose front has drifted out of the frozen window.

        The window is the *display* crop, so a front outside it is still
        being followed correctly -- it just is not in shot any more. The
        page uses this to say so.
        """
        js, iss = window or self.anchor.window
        for step in self.steps():
            cj, ci = self.centres[step]
            if not (js.start <= cj < js.stop and iss.start <= ci < iss.stop):
                return step
        return None


def window_for(mask: np.ndarray, pad: float = 0.5,
               shape: tuple[int, int] | None = None) -> tuple[slice, slice]:
    """A box around *mask*, padded, clipped to the array.

    Padded because a box tight to the front at the anchor step would clip
    it as soon as it moved, and the point of freezing the window is to
    keep the front's surroundings in view.
    """
    shape = shape or mask.shape
    js, iss = np.nonzero(mask)
    if not len(js):
        return slice(0, shape[0]), slice(0, shape[1])

    def span(lo, hi, limit):
        margin = max(2.0, (hi - lo + 1) * pad)
        return slice(int(max(0, np.floor(lo - margin))),
                     int(min(limit, np.ceil(hi + margin + 1))))

    return (span(js.min(), js.max(), shape[0]),
            span(iss.min(), iss.max(), shape[1]))


def fronts_present(labels: np.ndarray, *, min_pixels: int = 20) -> list[int]:
    """Labels in this step, in **numerical order**.

    One step, deliberately.  The old selector counted how many steps each
    label appeared in and offered the survivors -- but labels are assigned
    per date, so that was measuring label *reuse*, not front persistence,
    and it cost a 0.9 GB read per step to compute something meaningless.
    Anchoring needs one step: whether the front persists is what
    :func:`follow` answers.

    Numerical rather than by size: the list is a dropdown of five-digit
    numbers that someone has to find a particular value in, and size order
    made that a linear scan.
    """
    values, counts = np.unique(labels, return_counts=True)
    return sorted(int(v) for v, c in zip(values, counts)
                  if v and c >= min_pixels)


#: Rough degrees-to-km, good enough for "which front is nearest".
KM_PER_DEG = 111.0


def nearest_front(labels, lon, lat, point_lon: float, point_lat: float, *,
                  min_pixels: int = 20, max_km: float = 80.0):
    """The label whose pixels come closest to a geographic point.

    This is what makes selection survive re-labelling: a point on the
    ocean means the same thing at every timestep, and a label does not.
    Distance is measured to the *nearest pixel* of each front rather than
    to its centroid -- a long front curving past the point should win over
    a small one whose middle happens to be closer.
    """
    coslat = float(np.cos(np.radians(point_lat)))
    best, best_km = None, np.inf

    for value in fronts_present(labels, min_pixels=min_pixels):
        mask = labels == value
        dlon = ((lon[mask] - point_lon + 180.0) % 360.0) - 180.0
        km = float(np.hypot(dlon * coslat,
                            lat[mask] - point_lat).min()) * KM_PER_DEG
        if km < best_km:
            best, best_km = int(value), km

    if best is None or best_km > max_km:
        return None, best_km
    return best, best_km


def anchor_at_point(labels, lon, lat, point_lon: float, point_lat: float,
                    step: int, *, max_km: float = 80.0, pad: float = 0.5):
    """Anchor on a geographic point rather than on a label."""
    label, km = nearest_front(labels, lon, lat, point_lon, point_lat,
                              max_km=max_km)
    if label is None:
        raise ValueError(
            f"no front within {max_km:.0f} km of "
            f"({point_lat:.3f}, {point_lon:.3f}) at step {step} "
            f"(nearest was {km:.0f} km)")
    return anchor_at(labels, step, label, pad=pad), km


def anchor_at(labels: np.ndarray, step: int, label: int, *,
              pad: float = 0.5) -> Anchor:
    """Pin the front down: its window and its centroid at *step*."""
    mask = labels == int(label)
    if not mask.any():
        raise ValueError(f"label {label} is absent from step {step}")
    return Anchor(step=int(step), label=int(label),
                  window=window_for(mask, pad=pad, shape=labels.shape),
                  centre=centroid(mask))


def _radius_px(dt_seconds: float, km_per_px: float) -> float:
    """How far the centroid could plausibly have moved, in pixels."""
    km = MAX_DRIFT_MS * abs(dt_seconds) / 1000.0
    return max(MIN_RADIUS_PX, km / max(km_per_px, 1e-6))


def _best_candidate(labels: np.ndarray, ref_centre, ref_mask,
                    radius: float):
    """The label nearest *ref_centre* within *radius*, or ``None``.

    Gated on the drift radius alone, deliberately **not** on the display
    window: a front that advects out of frame over a week is still the
    same front, and cutting the track at the frame edge would silently
    turn a display limitation into missing data.

    Distance decides. Overlap with the previous mask breaks ties, which
    only bites inside the dense day -- across a daily gap there is usually
    no overlap to break anything with.
    """
    present = [int(v) for v in np.unique(labels) if v]
    if not present:
        return None

    best, best_key = None, None
    for value in present:
        mask = labels == value
        cj, ci = centroid(mask)
        dist = float(np.hypot(cj - ref_centre[0], ci - ref_centre[1]))
        if dist > radius:
            continue

        overlap = 0.0
        if ref_mask is not None:
            union = np.count_nonzero(mask | ref_mask)
            if union:
                overlap = np.count_nonzero(mask & ref_mask) / union

        key = (dist, -overlap)
        if best_key is None or key < best_key:
            best, best_key = value, key

    return best


def follow(labels_at, times: list[str], anchor: Anchor, *,
           km_per_px: float = DEFAULT_KM_PER_PX) -> Track:
    """Follow the anchored front through every step.

    Parameters
    ----------
    labels_at : callable
        ``step -> (H, W)`` label array for the chunk at that step. A
        callable rather than a provider so this is testable without S3.
    times : list of str
        Timestamps, one per step. These set the search radius, which is
        why the uneven cadence does not break the tracker.
    anchor : Anchor
    km_per_px : float
        Cell size, for turning a drift distance into pixels.

    Notes
    -----
    The walk runs **outward from the anchor** in both directions, each
    step compared with the last one *found* rather than with the anchor.
    Chaining follows advection; comparing everything back to the anchor
    would lose the front as soon as it moved its own width. Where a step
    is a gap the reference is kept, so the track can re-acquire the front
    later instead of ending there.
    """
    track = Track(anchor=anchor, labels={anchor.step: anchor.label},
                  centres={anchor.step: anchor.centre})
    n = len(times)
    stamps = [parse_time(t) for t in times]

    for direction in (1, -1):
        ref_centre = anchor.centre
        ref_mask = labels_at(anchor.step) == anchor.label
        ref_time = stamps[anchor.step]

        step = anchor.step + direction
        while 0 <= step < n:
            labels = labels_at(step)
            radius = _radius_px(
                (stamps[step] - ref_time).total_seconds(), km_per_px)
            found = _best_candidate(labels, ref_centre, ref_mask, radius)
            if found is None and ref_centre != anchor.centre:
                # The chain has drifted or broken.  Retry from the anchor
                # itself, with a radius sized on the elapsed time since
                # the anchor step -- otherwise one bad step ends the
                # track even though the front is still right there.
                anchor_radius = _radius_px(
                    (stamps[step] - stamps[anchor.step]).total_seconds(),
                    km_per_px)
                found = _best_candidate(labels, anchor.centre, None,
                                        anchor_radius)
            if found is not None:
                mask = labels == found
                track.labels[step] = found
                track.centres[step] = centroid(mask)
                ref_centre, ref_mask, ref_time = (
                    track.centres[step], mask, stamps[step])
            step += direction

    return track
