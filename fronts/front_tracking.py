"""Following one front through a sequence of timesteps.

Front labels are assigned independently at every timestep, so the same
physical front carries a different label in every frame. Following a
*label* jumps to an unrelated front; following a *place* is better but not
enough on its own, because the front moves and its neighbours do not.

This module lives outside ``fronts/viz`` on purpose. Tracking is a
statement about the ocean, not about a web page: it takes label maps and
timestamps and returns which label is which front, so analysis code can
use it without importing Panel.

The method
----------
One front is identified at an **anchor** step -- by clicking a point, so
the selection means the same thing at every step -- and then re-found at
every other step by matching against what the front was like last time we
saw it. A candidate is scored on four things:

============  ==========================================================
position      distance from where the front is *predicted* to be
overlap       how much of the candidate's mask coincides with the last
length        how much longer or shorter it has become
orientation   how far it has turned
============  ==========================================================

Position alone was the first version and it drifted onto neighbours: a
front that moves a long way in a day is further from its own last
position than a stationary neighbour is, so distance ranked the wrong one
first. The shape terms are what break that tie, and they are trustworthy
for the same reason the whole approach is: consecutive samples of one
front cannot differ wildly in length or orientation.

What the scores mean
--------------------
Each term is a dimensionless penalty, roughly "how many times more
different than expected". They are summed with weights and compared
against :data:`MAX_SCORE`; nothing wins by default, so where no candidate
is good enough the step is a **gap** rather than a guess. A movie with a
hole in it is honest; one that confidently highlights the wrong front is
not.

Prediction
----------
The search is centred on where the front is *expected* to be, not on
where it was. With two prior positions the velocity is extrapolated;
otherwise the last position is used. This matters because the sampling is
uneven -- a week of daily snapshots wrapped around one intensive day -- so
consecutive steps can be an hour or a day apart, and a front that barely
moves in an hour travels tens of kilometres in a day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

#: Plausible upper bound on how fast a front's centroid moves, m/s. Sizes
#: the search radius, so it should be generous: too small drops real links
#: across the daily gaps.
MAX_DRIFT_MS = 0.6

#: Floor on the search radius, in pixels.
#:
#: This absorbs re-labelling jitter, and it used to be 4 because centroids
#: wander: re-detect the same front and its middle moves.  Measuring
#: mask-to-mask instead leaves very little to absorb -- the two extents
#: nearly coincide -- so the floor can be tight, and a tight floor is what
#: keeps short-timescale links honest.  At 2 cells an hourly link allows
#: ~14 km of travel; at the old 4 it allowed 28, which is faster than a
#: front moves.
MIN_RADIUS_PX = 2.0

#: Nominal cell size. LLC4320 is ~1/48 degree, so ~2.3 km at the equator
#: and less towards the poles.
DEFAULT_KM_PER_PX = 2.3

#: Rough degrees-to-km, good enough for "which front is nearest".
KM_PER_DEG = 111.0

#: How much each term counts. Position still leads -- it is the only term
#: that knows *where* -- but it can no longer win on its own.
WEIGHTS = {
    "position": 1.0,
    "overlap": 0.8,
    "length": 0.6,
    "area": 0.4,
    "orientation": 0.6,
}

#: Total score a candidate must beat. Roughly: one term may be completely
#: wrong, or several may be mildly off, but not both.
MAX_SCORE = 2.5

#: Hard limit on the position term, as a multiple of the search radius.
#:
#: A veto, not a weight.  Motion beyond this is not plausible on any
#: timescale, and the shape terms must not be able to rescue it -- a
#: candidate with *identical* shape a long way off is more likely a
#: different front that happens to look similar than the same front
#: teleporting.  Shape breaks ties between plausible candidates; it does
#: not license implausible ones.
MAX_POSITION_FACTOR = 3.0

#: Scale for the orientation penalty, in degrees. A front that has turned
#: this far scores 1.0 on that term.
ORIENTATION_SCALE_DEG = 25.0

#: Scale for the length and area penalties, as a log ratio. ln(2) means
#: "doubled or halved" scores 1.0.
LENGTH_SCALE_LOG = float(np.log(2.0))


def parse_time(stamp: str) -> datetime:
    """``2012-07-03T12_00_00`` -> datetime."""
    return datetime.strptime(stamp, "%Y-%m-%dT%H_%M_%S")


def centroid(mask: np.ndarray) -> tuple[float, float]:
    """``(j, i)`` centre of a boolean mask. NaNs if it is empty."""
    js, iss = np.nonzero(mask)
    if not len(js):
        return float("nan"), float("nan")
    return float(js.mean()), float(iss.mean())


def orientation_deg(mask: np.ndarray) -> float:
    """Major-axis angle of a boolean mask, in 0-90 degrees.

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


def orientation_signed_deg(mask: np.ndarray) -> float:
    """Major-axis angle in ``[-90, 90)``, keeping the sign.

    :func:`orientation_deg` folds to 0-90, which is the right convention
    for a histogram of front orientations but wrong for comparing two
    fronts: a front tilted +40 and one tilted -40 both report 40, so they
    look identical when they are 80 degrees apart.  Tracking needs to
    tell those apart, so it keeps the sign and wraps at 180 instead.
    """
    jj, ii = np.nonzero(mask)
    if jj.size < 2:
        return float("nan")

    dj = jj - jj.mean()
    di = ii - ii.mean()
    angle = 0.5 * np.arctan2(2.0 * float((dj * di).mean()),
                             float((dj * dj).mean()) - float((di * di).mean()))
    return float(np.degrees(angle))


@dataclass
class FrontShape:
    """What a front looked like at one step.

    Cheap by construction: everything here comes from the label mask,
    which is already in hand, so describing a candidate costs no reads.
    """

    label: int
    centre: tuple[float, float]
    area: float
    length: float
    orientation: float

    @property
    def valid(self) -> bool:
        return np.isfinite(self.centre[0]) and self.area > 0


def describe(labels: np.ndarray, value: int) -> FrontShape:
    """Measure one front in a label map.

    *length* is the major-axis extent from second moments rather than a
    pixel count: a front that thickens should not read as one that grew.
    """
    mask = labels == int(value)
    js, iss = np.nonzero(mask)
    if js.size == 0:
        return FrontShape(int(value), (float("nan"),) * 2, 0.0,
                          float("nan"), float("nan"))

    dj = js - js.mean()
    di = iss - iss.mean()
    cov = np.array([[float((dj * dj).mean()), float((dj * di).mean())],
                    [float((dj * di).mean()), float((di * di).mean())]])
    eigenvalues = np.linalg.eigvalsh(cov)
    length = float(4.0 * np.sqrt(max(eigenvalues[-1], 0.0)))

    return FrontShape(label=int(value),
                      centre=(float(js.mean()), float(iss.mean())),
                      area=float(js.size),
                      length=length,
                      orientation=orientation_signed_deg(mask))


@dataclass
class Anchor:
    """The front we are following, and the frame we follow it in."""

    step: int
    label: int
    window: tuple[slice, slice]
    centre: tuple[float, float]


@dataclass
class Link:
    """One step's match, and how confident it was."""

    step: int
    label: int
    score: float
    terms: dict


@dataclass
class Track:
    """Which label is our front, at each step."""

    anchor: Anchor
    labels: dict[int, int] = field(default_factory=dict)
    centres: dict[int, tuple[float, float]] = field(default_factory=dict)
    links: dict[int, Link] = field(default_factory=dict)

    def label_at(self, step: int) -> int | None:
        """The label at *step*, or ``None`` where the front was lost."""
        return self.labels.get(int(step))

    def steps(self) -> list[int]:
        """Steps where the front was found, in order."""
        return sorted(self.labels)

    def gaps(self, n_steps: int) -> list[int]:
        """Steps with no front, so a caller can say so rather than guess."""
        return [s for s in range(n_steps) if s not in self.labels]

    def weakest(self, n: int = 3) -> list[Link]:
        """The least confident links, worst first.

        Tracking has no ground truth here, so the next best thing is
        saying which joins were the closest calls -- those are where a
        track most likely jumped to a neighbour.
        """
        return sorted(self.links.values(), key=lambda l: -l.score)[:n]

    def first_escape(self, window=None) -> int | None:
        """First step whose front has drifted out of the frozen window.

        The window is the *display* crop, so a front outside it is still
        being followed correctly -- it just is not in shot any more.
        """
        js, iss = window or self.anchor.window
        for step in self.steps():
            cj, ci = self.centres[step]
            if not (js.start <= cj < js.stop and iss.start <= ci < iss.stop):
                return step
        return None


def window_for(mask: np.ndarray, pad: float = 0.5,
               shape: tuple[int, int] | None = None) -> tuple[slice, slice]:
    """A box around *mask*, padded, clipped to the array."""
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
    """Labels in this step, in numerical order.

    Numerical rather than by size: the list is a dropdown of five-digit
    numbers that someone has to find a particular value in.
    """
    values, counts = np.unique(labels, return_counts=True)
    return sorted(int(v) for v, c in zip(values, counts)
                  if v and c >= min_pixels)


def nearest_front(labels, lon, lat, point_lon: float, point_lat: float, *,
                  min_pixels: int = 20, max_km: float = 80.0):
    """The label whose pixels come closest to a geographic point.

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


def anchor_at(labels: np.ndarray, step: int, label: int, *,
              pad: float = 0.5) -> Anchor:
    """Pin the front down: its window and its centroid at *step*."""
    mask = labels == int(label)
    if not mask.any():
        raise ValueError(f"label {label} is absent from step {step}")
    return Anchor(step=int(step), label=int(label),
                  window=window_for(mask, pad=pad, shape=labels.shape),
                  centre=centroid(mask))


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


def _radius_px(dt_seconds: float, km_per_px: float) -> float:
    """How far the centroid could plausibly have moved, in pixels."""
    km = MAX_DRIFT_MS * abs(dt_seconds) / 1000.0
    return max(MIN_RADIUS_PX, km / max(km_per_px, 1e-6))


def _distance_field(mask: np.ndarray) -> np.ndarray:
    """Distance from every cell to the nearest True cell of *mask*.

    Computed once per reference mask, so scoring a candidate is then a
    lookup over its own pixels rather than a comparison against every
    pixel of the reference.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask)
    except ImportError:                                     # pragma: no cover
        # Exact, and fine at tile size; scipy is just faster.
        js, iss = np.nonzero(mask)
        if js.size == 0:
            return np.full(mask.shape, np.inf)
        jj, ii = np.indices(mask.shape)
        out = np.full(mask.shape, np.inf)
        for j, i in zip(js, iss):
            np.minimum(out, np.hypot(jj - j, ii - i), out=out)
        return out


def _shift(mask: np.ndarray, dj: float, di: float) -> np.ndarray:
    """Move a mask by a whole number of cells, without wrapping."""
    dj, di = int(round(dj)), int(round(di))
    if dj == 0 and di == 0:
        return mask
    out = np.zeros_like(mask)
    h, w = mask.shape
    j0, j1 = max(0, dj), min(h, h + dj)
    i0, i1 = max(0, di), min(w, w + di)
    if j0 >= j1 or i0 >= i1:
        return out
    out[j0:j1, i0:i1] = mask[j0 - dj:j1 - dj, i0 - di:i1 - di]
    return out


def _angle_gap(a: float, b: float) -> float:
    """Smallest angle between two axes, in 0-90 degrees.

    An axis has no direction, so it repeats every 180 degrees: +89 and
    -89 describe almost the same line and are two degrees apart, not one
    hundred and seventy-eight.  Inputs are the signed angles from
    :func:`orientation_signed_deg`.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return 0.0
    raw = abs(float(a) - float(b)) % 180.0
    return min(raw, 180.0 - raw)


def score_candidate(candidate: FrontShape, reference: FrontShape,
                    predicted: tuple[float, float], radius: float,
                    ref_mask=None, cand_mask=None,
                    weights: dict | None = None,
                    ref_distance=None) -> tuple[float, dict]:
    """How unlike *reference* is *candidate*?  Lower is better.

    Every term is dimensionless and scaled so that 1.0 means "as different
    as we would expect to tolerate". A term whose inputs are missing is
    skipped and its weight redistributed, so a front too small to have a
    meaningful orientation is not penalised for lacking one.

    *ref_distance* is the distance field of the reference mask, already
    shifted by the predicted motion.  When it is supplied the position
    term is the distance from the candidate to the **nearest point** of
    the reference, which is the only spatial measure that works for an
    extended feature -- see the note in :func:`follow`.
    """
    weights = dict(weights or WEIGHTS)
    terms, used = {}, {}

    if ref_distance is not None and cand_mask is not None:
        dist = float(np.min(ref_distance[cand_mask]))
    else:
        dist = float(np.hypot(candidate.centre[0] - predicted[0],
                              candidate.centre[1] - predicted[1]))
    terms["position"] = dist / max(radius, 1e-6)
    used["position"] = weights["position"]

    if ref_mask is not None and cand_mask is not None:
        union = int(np.count_nonzero(ref_mask | cand_mask))
        inter = int(np.count_nonzero(ref_mask & cand_mask))
        iou = (inter / union) if union else 0.0
        # Overlap is evidence *for*, so its penalty falls as IoU rises.
        # It is deliberately weak across long gaps, where even a correct
        # match may not overlap at all -- hence a penalty that saturates
        # at 1.0 rather than one that grows without bound.
        terms["overlap"] = 1.0 - iou
        used["overlap"] = weights["overlap"]

    if candidate.length > 0 and reference.length > 0:
        ratio = abs(float(np.log(candidate.length / reference.length)))
        terms["length"] = ratio / LENGTH_SCALE_LOG
        used["length"] = weights["length"]

    if candidate.area > 0 and reference.area > 0:
        ratio = abs(float(np.log(candidate.area / reference.area)))
        terms["area"] = ratio / LENGTH_SCALE_LOG
        used["area"] = weights["area"]

    gap = _angle_gap(candidate.orientation, reference.orientation)
    if np.isfinite(candidate.orientation) and np.isfinite(reference.orientation):
        terms["orientation"] = gap / ORIENTATION_SCALE_DEG
        used["orientation"] = weights["orientation"]

    total_weight = sum(used.values()) or 1.0
    score = sum(terms[k] * used[k] for k in terms) / total_weight

    if terms["position"] > MAX_POSITION_FACTOR:
        score = float("inf")            # vetoed: see MAX_POSITION_FACTOR
    return float(score), terms


def _predict(history: list[tuple[int, tuple[float, float], datetime]],
             when: datetime) -> tuple[float, float]:
    """Where the front should be at *when*, from where it has been.

    Constant velocity from the last two sightings; the last position
    alone if that is all there is. Free -- it uses positions already
    measured -- and it is what makes the long daily links work, because
    a front that has been moving steadily keeps moving steadily.
    """
    if not history:
        raise ValueError("cannot predict with no history")

    (_s1, p1, t1) = history[-1]
    if len(history) < 2:
        return p1

    (_s0, p0, t0) = history[-2]
    span = (t1 - t0).total_seconds()
    if abs(span) < 1.0:
        return p1

    rate = ((p1[0] - p0[0]) / span, (p1[1] - p0[1]) / span)
    ahead = (when - t1).total_seconds()
    return (p1[0] + rate[0] * ahead, p1[1] + rate[1] * ahead)


def follow(labels_at, times: list[str], anchor: Anchor, *,
           km_per_px: float = DEFAULT_KM_PER_PX,
           max_score: float = MAX_SCORE,
           weights: dict | None = None,
           min_pixels: int = 5) -> Track:
    """Follow the anchored front through every step.

    Parameters
    ----------
    labels_at : callable
        ``step -> (H, W)`` label array. A callable rather than a provider
        so this is testable without any data source.
    times : list of str
        Timestamps, one per step. These set the search radius and drive
        the prediction, which is why the uneven cadence does not break it.
    anchor : Anchor
    km_per_px : float
        Cell size, for turning a drift distance into pixels.
    max_score : float
        Reject anything worse than this. Raising it fills gaps at the
        cost of confidence; lowering it does the reverse.
    min_pixels : int
        Ignore specks. Much lower than the selector's threshold on
        purpose: a front too small to be worth offering in a list is
        still a real front once you are following it, and excluding it
        would open a gap rather than avoid a mistake.

    Notes
    -----
    The walk runs outward from the anchor in both directions, each step
    compared with the last one *found*. Chaining follows advection;
    comparing everything back to the anchor loses the front as soon as it
    moves its own width. Where a step is a gap the reference is kept, so
    the track can re-acquire the front later instead of ending there.

    **Position is measured mask-to-mask, not centroid-to-centroid.** For
    an extended feature the centroid is a bad proxy for where it is: a
    400-cell front that grows 110 cells at one end has not moved, but its
    centroid has shifted 55 cells -- far enough to be vetoed as
    implausible motion, while a short unrelated front sitting beside it
    scores well.  That is not a tuning problem, it is the wrong measure.
    Distance to the nearest point of the (predicted) previous extent says
    what we actually mean: could this be the same water?
    """
    stamps = [parse_time(t) for t in times]
    n = len(times)

    anchor_labels = labels_at(anchor.step)
    anchor_shape = describe(anchor_labels, anchor.label)

    track = Track(anchor=anchor,
                  labels={anchor.step: anchor.label},
                  centres={anchor.step: anchor.centre},
                  links={anchor.step: Link(anchor.step, anchor.label, 0.0,
                                           {"anchor": 0.0})})

    for direction in (1, -1):
        reference = anchor_shape
        ref_mask = anchor_labels == anchor.label
        history = [(anchor.step, anchor.centre, stamps[anchor.step])]

        step = anchor.step + direction
        while 0 <= step < n:
            labels = labels_at(step)
            radius = _radius_px(
                (stamps[step] - history[-1][2]).total_seconds(), km_per_px)
            predicted = _predict(history, stamps[step])

            # The reference, moved to where we expect it, as a distance
            # field: scoring a candidate is then a lookup over its own
            # pixels.  Built once per step rather than per candidate.
            moved = _shift(ref_mask,
                           predicted[0] - reference.centre[0],
                           predicted[1] - reference.centre[1])
            ref_distance = _distance_field(moved)

            best, best_score, best_terms, best_shape = None, np.inf, {}, None
            for value in fronts_present(labels, min_pixels=min_pixels):
                shape = describe(labels, value)
                if not shape.valid:
                    continue
                mask = labels == value
                score, terms = score_candidate(
                    shape, reference, predicted, radius,
                    ref_mask=ref_mask, cand_mask=mask, weights=weights,
                    ref_distance=ref_distance)
                if score < best_score:
                    best, best_score = value, score
                    best_terms, best_shape = terms, shape

            if best is not None and np.isfinite(best_score) \
                    and best_score <= max_score:
                track.labels[step] = int(best)
                track.centres[step] = best_shape.centre
                track.links[step] = Link(step, int(best), float(best_score),
                                         best_terms)
                reference = best_shape
                ref_mask = labels == best
                history.append((step, best_shape.centre, stamps[step]))

            step += direction

    return track
