"""
Builders for 2-D "curtain" cross-sections of labelled fronts in LLC4320 tiles.

A *curtain* is a vertical cross-section sampled along a path of pixels through
a tile:

* x-axis -- distance along the path (pixels, with an optional km twin),
* y-axis -- depth (the LLC ``Z`` axis, metres, negative downward),
* color  -- a configurable field (default the Richardson number ``Ri``),
* contours -- isopycnals (sigma0 surfaces) overlaid on top.

This module is the 2-D counterpart of ``fronts/viz/fronts_3d.py``.  It reuses
that module's :func:`~fronts.viz.fronts_3d.decompose_front_branches` to split a
thinned front skeleton into branch polylines, then assembles the *main axis*
(the longest end-to-end path, ignoring side branches), throws *offset* paths to
either side of it, and cuts a *perpendicular* transect at a chosen point.

The sampling core (:func:`sample_curtain`) mirrors the inner loop of
``fronts_3d.build_front_curtain`` -- ``scipy.ndimage.map_coordinates`` along a
``(j, i)`` polyline across every depth level -- but returns a plain ``(K, L)``
array for matplotlib rather than a PyVista ribbon.

Rendering is static matplotlib (Agg), matching the repo's 2-D companion figures
in ``fronts/viz/insets.py``.

Design notes
------------
* **Main-axis columns are the real front pixels.**  No resampling onto evenly
  spaced points: every pixel along the diameter path is one curtain column.
  Distance-in-km is integrated great-circle distance between consecutive real
  pixels, so the km axis is just a relabelling of the same columns.
* **Smoothing affects only directions.**  The optional tangential/normal
  smoothing (``smooth_window``) is applied to the unit-direction field used to
  throw offsets and the perpendicular transect -- never to the main-axis
  column positions.  A jagged single-pixel skeleton produces wildly swinging
  raw normals that make adjacent offset points collide immediately; smoothing
  averages the kink out.
* **Offset self-overlap is detected, not hidden.**  Where offset points from
  different axis positions land within < 1 px of each other (the concave side
  of a genuine bend, or residual skeleton noise), the affected columns are
  flagged and shaded on the figure so the curtain still renders but the
  unreliable region is obvious.  The more-correct centre-of-curvature offset is
  a documented follow-up, not implemented here.
"""

# stdlib
from __future__ import annotations
import heapq
import logging
from typing import Sequence

# numerical / plotting
import numpy as np
from scipy import ndimage as scimg
import matplotlib
matplotlib.use("Agg")  # headless-safe; pair to off-screen rendering
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


def _decompose_front_branches(front_mask: np.ndarray):
    """Lazily import the 3-D module's branch decomposition.

    ``fronts/viz/fronts_3d.py`` imports PyVista at module load, but the branch
    decomposition itself is pure NumPy/SciPy.  Importing it lazily here keeps
    the 2-D curtain viewer usable without a 3-D rendering stack installed,
    while still reusing the identical skeleton-handling code.
    """
    from fronts.viz.fronts_3d import decompose_front_branches
    return decompose_front_branches(front_mask)


# ---------------------------------------------------------------------------
# Main-axis extraction (longest end-to-end path through the skeleton)
# ---------------------------------------------------------------------------

def _branch_length(branch: np.ndarray) -> float:
    """Arc length of a ``(L, 2)`` (j, i) polyline in pixels.

    Consecutive 4-connected steps contribute 1, diagonal steps ``sqrt(2)``.

    Parameters
    ----------
    branch : numpy.ndarray
        ``(L, 2)`` integer array of ``(j, i)`` pixel coordinates.

    Returns
    -------
    float
        Sum of Euclidean distances between consecutive pixels (0 for a
        single-pixel branch).
    """
    if branch.shape[0] < 2:
        return 0.0
    d = np.diff(branch.astype(np.float64), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def extract_main_axis(front_mask: np.ndarray) -> np.ndarray:
    """Return the longest end-to-end path through a thinned front skeleton.

    The skeleton is decomposed into branches between junctions/endpoints by
    :func:`fronts.viz.fronts_3d.decompose_front_branches`.  Those branches form
    a graph whose nodes are the branch end pixels and whose edges are weighted
    by branch arc length.  The *main axis* is the graph diameter -- the longest
    end-to-end shortest path -- found with two Dijkstra sweeps (a longest-path
    search on a tree; for the rare cyclic skeleton the double-sweep still
    returns a near-diameter path, which is sufficient for a front backbone).

    Side branches are dropped: only the branches lying on the diameter path are
    concatenated into the returned polyline.

    Parameters
    ----------
    front_mask : numpy.ndarray
        2-D boolean mask, True at the selected front's (thinned) pixels.

    Returns
    -------
    numpy.ndarray
        ``(L, 2)`` integer array of ``(j, i)`` pixel coordinates, ordered along
        the main axis.  Falls back to the single longest branch if the graph is
        degenerate, or to the lone pixel for a single-pixel mask.

    Raises
    ------
    ValueError
        If ``front_mask`` has no True pixels.
    """
    if not front_mask.any():
        raise ValueError("front_mask has no True pixels; nothing to trace.")

    branches = _decompose_front_branches(front_mask)
    # Drop zero-length (single-pixel) branches for graph building but remember
    # them in case the whole front is a single pixel.
    poly_branches = [b for b in branches if b.shape[0] >= 2]
    if not poly_branches:
        # Single isolated pixel (or a handful) -- just return the longest.
        return max(branches, key=lambda b: b.shape[0])

    # Build a node index keyed by pixel coordinate of every branch endpoint.
    node_id: dict[tuple[int, int], int] = {}

    def _node(jy: int, ix: int) -> int:
        key = (int(jy), int(ix))
        if key not in node_id:
            node_id[key] = len(node_id)
        return node_id[key]

    # adjacency: node -> list of (neighbour_node, weight, branch_index, reversed)
    adj: dict[int, list[tuple[int, float, int, bool]]] = {}
    for b_idx, branch in enumerate(poly_branches):
        a = _node(branch[0, 0], branch[0, 1])
        b = _node(branch[-1, 0], branch[-1, 1])
        w = _branch_length(branch)
        adj.setdefault(a, []).append((b, w, b_idx, False))
        adj.setdefault(b, []).append((a, w, b_idx, True))

    def _dijkstra(source: int) -> tuple[np.ndarray, dict[int, tuple[int, int, bool]]]:
        """Shortest-path distances + predecessor edges from ``source``."""
        n = len(node_id)
        dist = np.full(n, np.inf)
        dist[source] = 0.0
        prev: dict[int, tuple[int, int, bool]] = {}  # node -> (prev_node, b_idx, reversed)
        pq: list[tuple[float, int]] = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w, b_idx, rev in adj.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = (u, b_idx, rev)
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    # Double sweep: farthest node from an arbitrary start, then farthest from
    # that node -> the two endpoints of the diameter.
    start = next(iter(node_id.values()))
    dist0, _ = _dijkstra(start)
    end_a = int(np.argmax(np.where(np.isfinite(dist0), dist0, -1.0)))
    dist1, prev1 = _dijkstra(end_a)
    end_b = int(np.argmax(np.where(np.isfinite(dist1), dist1, -1.0)))

    # Walk predecessors from end_b back to end_a, collecting branch indices.
    path_edges: list[tuple[int, bool]] = []  # (branch_index, reversed)
    cur = end_b
    while cur != end_a and cur in prev1:
        pnode, b_idx, rev = prev1[cur]
        path_edges.append((b_idx, rev))
        cur = pnode
    path_edges.reverse()

    if not path_edges:
        # Disconnected / degenerate -- fall back to the single longest branch.
        return max(poly_branches, key=_branch_length)

    # Concatenate branch polylines along the path, orienting each so its tail
    # matches the previous branch's head, and de-duplicating shared junction
    # pixels at the joins.
    pieces: list[np.ndarray] = []
    for k, (b_idx, rev) in enumerate(path_edges):
        seg = poly_branches[b_idx]
        if rev:
            seg = seg[::-1]
        if k > 0:
            # Drop the first pixel if it duplicates the previous tail.
            if np.array_equal(seg[0], pieces[-1][-1]):
                seg = seg[1:]
        if seg.shape[0] > 0:
            pieces.append(seg)
    axis = np.concatenate(pieces, axis=0)
    return axis.astype(np.int32)


# ---------------------------------------------------------------------------
# Path metrics: distances + (optionally smoothed) tangents/normals
# ---------------------------------------------------------------------------

def _great_circle_step_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km) between consecutive lon/lat samples (haversine)."""
    R = 6371.0088  # mean Earth radius, km
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def path_metrics(
    path: np.ndarray,
    XC_rect: np.ndarray | None = None,
    YC_rect: np.ndarray | None = None,
    *,
    smooth: bool = False,
    smooth_window: int = 5,
) -> dict:
    """Per-pixel distance + unit tangent/normal for a ``(j, i)`` path.

    The returned arrays are aligned 1:1 with ``path`` -- one entry per real
    pixel; nothing is resampled onto new positions.  ``smooth`` controls *only*
    the direction field (tangents and normals), never the distances or the path
    itself.

    Parameters
    ----------
    path : numpy.ndarray
        ``(L, 2)`` integer/float ``(j, i)`` pixel coordinates.
    XC_rect, YC_rect : numpy.ndarray, optional
        Longitude / latitude on the rect-tile-local frame, shape
        ``(TILE_SIZE, TILE_SIZE)``.  When supplied the cumulative great-circle
        distance in km is computed; otherwise ``dist_km`` is None.
    smooth : bool, optional
        If True, smooth the tangential direction with a moving average of width
        ``smooth_window`` before deriving normals.  Default False (raw, jagged).
    smooth_window : int, optional
        Odd window length in pixels for the tangent smoothing (default 5).
        Ignored when ``smooth`` is False.

    Returns
    -------
    dict
        Keys: ``dist_px`` (L,), ``dist_km`` (L,) or None, ``tangents`` (L, 2)
        unit ``(dj, di)``, ``normals`` (L, 2) unit ``(dj, di)`` rotated +90 deg,
        ``smoothed`` (bool).
    """
    pts = np.asarray(path, dtype=np.float64)
    L = pts.shape[0]

    # Cumulative pixel distance (Euclidean between consecutive pixels).
    if L >= 2:
        step = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
        dist_px = np.concatenate([[0.0], np.cumsum(step)])
    else:
        dist_px = np.zeros(L)

    # Cumulative km distance from lon/lat at the real pixels.
    dist_km = None
    if XC_rect is not None and YC_rect is not None and L >= 1:
        jj = np.clip(np.round(pts[:, 0]).astype(int), 0, XC_rect.shape[0] - 1)
        ii = np.clip(np.round(pts[:, 1]).astype(int), 0, XC_rect.shape[1] - 1)
        lon = XC_rect[jj, ii]
        lat = YC_rect[jj, ii]
        if L >= 2:
            step_km = _great_circle_step_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            dist_km = np.concatenate([[0.0], np.cumsum(step_km)])
        else:
            dist_km = np.zeros(L)

    # Tangents via central differences on the (un-smoothed) pixel positions.
    if L >= 2:
        tang = np.gradient(pts, axis=0)  # (L, 2)
    else:
        tang = np.zeros((L, 2))
        tang[:, 1] = 1.0  # arbitrary unit direction for a single pixel

    if smooth and L >= 3:
        w = max(3, int(smooth_window) | 1)  # force odd, >= 3
        kernel = np.ones(w) / w
        # Smooth each component of the tangent direction, then renormalise.
        tj = np.convolve(tang[:, 0], kernel, mode="same")
        ti = np.convolve(tang[:, 1], kernel, mode="same")
        tang = np.column_stack([tj, ti])

    norm = np.sqrt((tang ** 2).sum(axis=1))
    norm[norm == 0] = 1.0
    tangents = tang / norm[:, None]

    # Normal = tangent rotated +90 deg in (j, i): (dj, di) -> (-di, dj)... but
    # we keep a consistent right-hand convention: n = (di, -dj) gives a vector
    # 90 deg clockwise.  Either sign is fine as long as +offset/-offset are
    # mirror images; we pick n = (-t_i, t_j) so side A / side B are consistent.
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    return {
        "dist_px": dist_px,
        "dist_km": dist_km,
        "tangents": tangents,
        "normals": normals,
        "smoothed": bool(smooth),
    }


# ---------------------------------------------------------------------------
# Offset + perpendicular path construction
# ---------------------------------------------------------------------------

def offset_paths(
    path: np.ndarray,
    normals: np.ndarray,
    n_offsets: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build offset polylines 1..n pixels to either side of a path.

    Parameters
    ----------
    path : numpy.ndarray
        ``(L, 2)`` ``(j, i)`` main-axis pixel coordinates (float ok).
    normals : numpy.ndarray
        ``(L, 2)`` unit normal per pixel (from :func:`path_metrics`).
    n_offsets : int
        Number of offset steps per side.

    Returns
    -------
    side_a, side_b : list of numpy.ndarray
        ``side_a[k]`` is the path shifted ``(k + 1)`` pixels along ``+normal``;
        ``side_b[k]`` along ``-normal``.  Each is an ``(L, 2)`` float array (not
        rounded -- sampling interpolates).
    """
    pts = np.asarray(path, dtype=np.float64)
    side_a: list[np.ndarray] = []
    side_b: list[np.ndarray] = []
    for k in range(1, int(n_offsets) + 1):
        side_a.append(pts + k * normals)
        side_b.append(pts - k * normals)
    return side_a, side_b


def perpendicular_path(
    path: np.ndarray,
    normals: np.ndarray,
    idx: int,
    half_width: int,
) -> np.ndarray:
    """Cross-front transect centered on ``path[idx]`` along its normal.

    Parameters
    ----------
    path : numpy.ndarray
        ``(L, 2)`` ``(j, i)`` main-axis coordinates.
    normals : numpy.ndarray
        ``(L, 2)`` unit normals from :func:`path_metrics`.
    idx : int
        Index along the path at which to cut the transect.
    half_width : int
        Number of pixels on each side of the axis (transect length is
        ``2 * half_width + 1``).

    Returns
    -------
    numpy.ndarray
        ``(2 * half_width + 1, 2)`` float ``(j, i)`` coordinates, ordered from
        ``-half_width`` (side B) through 0 (the axis point) to ``+half_width``
        (side A).
    """
    idx = int(np.clip(idx, 0, path.shape[0] - 1))
    origin = np.asarray(path[idx], dtype=np.float64)
    n = np.asarray(normals[idx], dtype=np.float64)
    steps = np.arange(-int(half_width), int(half_width) + 1)
    return origin[None, :] + steps[:, None] * n[None, :]


def offset_quality_flags(
    offset_path: np.ndarray,
    *,
    min_separation_px: float = 1.0,
) -> np.ndarray:
    """Flag columns of an offset path that self-overlap (option (b)).

    A column is flagged when its offset point lands within
    ``min_separation_px`` of *any other* column's offset point -- the signature
    of the concave side of a bend folding back on itself (or of residual
    skeleton noise when smoothing is off).  The curtain still renders; the
    caller shades flagged columns.

    Parameters
    ----------
    offset_path : numpy.ndarray
        ``(L, 2)`` offset polyline coordinates.
    min_separation_px : float, optional
        Collision threshold in pixels (default 1.0).

    Returns
    -------
    numpy.ndarray
        ``(L,)`` boolean array, True where the offset point collides with a
        non-adjacent column.
    """
    pts = np.asarray(offset_path, dtype=np.float64)
    L = pts.shape[0]
    flags = np.zeros(L, dtype=bool)
    if L < 3:
        return flags
    # Pairwise distance, but only flag collisions between columns that are not
    # immediate neighbours (adjacent columns are *supposed* to be ~1 px apart).
    for a in range(L):
        d = np.sqrt(((pts - pts[a]) ** 2).sum(axis=1))
        d[max(0, a - 1):min(L, a + 2)] = np.inf  # ignore self + neighbours
        if np.any(d < min_separation_px):
            flags[a] = True
    return flags


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed area of triangle (a, b, c); sign gives orientation of c vs a->b."""
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if open segments p1p2 and p3p4 cross (proper intersection)."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def trim_offset_loops(offset_path: np.ndarray) -> np.ndarray:
    """Trim self-crossing loops out of an offset line so it stays continuous.

    An offset line traced along the inside of a sharp bend can fold back and
    cross over itself, forming a small loop.  This function cuts each loop
    out: wherever two segments of the line cross, the points between them are
    dropped and the line is sewn back together at the crossing.  The result
    is a shorter line that never overlaps itself.

    It works iteratively: find the first place the line crosses itself, cut
    out the loop, then start over, until no crossings remain.

    Performance note: each pass compares every segment against every other
    (O(L^2)), and cutting a loop restarts the pass, so the worst case is
    ~O(L^3).  That is fast for typical lines (tens to low hundreds of
    points).  As a safeguard, the number of cuts is capped at L (each cut
    removes at least one point, so more is impossible); if the cap is
    somehow hit, the line is returned with any remaining crossings intact
    and a warning is logged, rather than stalling the render.

    Parameters
    ----------
    offset_path : numpy.ndarray
        ``(L, 2)`` offset polyline coordinates.

    Returns
    -------
    numpy.ndarray
        ``(L,)`` boolean keep-mask, True for vertices retained in the
        crossing-free line.  (Apply as ``offset_path[mask]`` for the line, or
        NaN-out ``~mask`` columns of the sampled curtain.)
    """
    pts = np.asarray(offset_path, dtype=np.float64)
    L = pts.shape[0]
    if L < 4:
        return np.ones(L, dtype=bool)
    idx = list(range(L))
    changed = True
    n_excisions = 0
    while changed and len(idx) >= 4:
        if n_excisions >= L:  # each excision removes >=1 vertex; see docstring
            log.warning(
                "trim_offset_loops: excision cap (%d) reached with crossings "
                "remaining; leaving offset partially untrimmed.", L,
            )
            break
        changed = False
        m = len(idx)
        for a in range(m - 1):
            for b in range(a + 2, m - 1):
                if _segments_intersect(
                    pts[idx[a]], pts[idx[a + 1]],
                    pts[idx[b]], pts[idx[b + 1]],
                ):
                    # Excise the loop: drop vertices a+1 .. b inclusive.
                    del idx[a + 1:b + 1]
                    n_excisions += 1
                    changed = True
                    break
            if changed:
                break
    if n_excisions:
        log.debug("trim_offset_loops: %d loop(s) excised, %d/%d vertices kept.",
                  n_excisions, len(idx), L)
    mask = np.zeros(L, dtype=bool)
    mask[idx] = True
    return mask


def transect_front_crossings(
    axis_path: np.ndarray,
    normals: np.ndarray,
    front_mask: np.ndarray,
    half_width: int,
) -> np.ndarray:
    """Count how many times each column's perpendicular transect hits the front.

    Used to keep the auto-picked perpendicular point on a clean stretch: a
    transect that crosses the front exactly once (the axis itself) shows
    cross-front dissipation cleanly, while one that crosses several times sits
    where the front loops back on itself (the squiggly hook) and is hard to
    interpret.

    Parameters
    ----------
    axis_path : numpy.ndarray
        ``(L, 2)`` main-axis coordinates (cropped frame).
    normals : numpy.ndarray
        ``(L, 2)`` unit normals from :func:`path_metrics`.
    front_mask : numpy.ndarray
        2-D boolean mask of the selected front (cropped frame).
    half_width : int
        Half-width (px) of the transect used at each column.

    Returns
    -------
    numpy.ndarray
        ``(L,)`` int count of distinct front-pixel runs along each transect.
    """
    L = axis_path.shape[0]
    counts = np.zeros(L, dtype=int)
    fmask = front_mask.astype(np.float64)
    for c in range(L):
        perp = perpendicular_path(axis_path, normals, c, half_width)
        hit = scimg.map_coordinates(
            fmask, np.stack([perp[:, 0], perp[:, 1]]),
            order=0, mode="constant", cval=0.0,
        ) > 0.5
        # Count rising edges (0 -> 1) = number of distinct crossings.
        runs = np.diff(np.concatenate([[0], hit.astype(int), [0]]))
        counts[c] = int((runs == 1).sum())
    return counts


# ---------------------------------------------------------------------------
# Curtain sampling
# ---------------------------------------------------------------------------

def sample_curtain(
    field3d: np.ndarray,
    path: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """Sample a 3-D field along a ``(j, i)`` path at every depth level.

    Mirrors the inner sampling of
    :func:`fronts.viz.fronts_3d.build_front_curtain` but returns a plain
    ``(K, L)`` array.  Off-grid samples (paths leaving the cropped window) come
    back as NaN via ``mode='constant', cval=nan`` so the renderer can flag them.

    Parameters
    ----------
    field3d : numpy.ndarray
        ``(K, J, I)`` field already cropped/clipped to the curtain window
        (e.g. cropped + depth-clipped sigma0, or the transformed color field).
    path : numpy.ndarray
        ``(L, 2)`` ``(j, i)`` coordinates *in the cropped frame*.
    order : int, optional
        Spline interpolation order for ``map_coordinates`` (default 1, linear).

    Returns
    -------
    numpy.ndarray
        ``(K, L)`` sampled field; NaN where the path leaves the window.
    """
    K = field3d.shape[0]
    pts = np.asarray(path, dtype=np.float64)
    L = pts.shape[0]
    j_loc = pts[:, 0]
    i_loc = pts[:, 1]
    kk, ss = np.meshgrid(np.arange(K), np.arange(L), indexing="ij")
    coords = np.stack([
        kk.ravel().astype(np.float64),
        np.repeat(j_loc[None, :], K, axis=0).ravel(),
        np.repeat(i_loc[None, :], K, axis=0).ravel(),
    ], axis=0)
    sampled = scimg.map_coordinates(
        field3d, coords, order=order, mode="constant", cval=np.nan,
    ).reshape(K, L)
    return sampled


def pick_extremum_index(
    curtain_field: np.ndarray,
    mode: str = "min",
) -> int:
    """Index of the column holding the field's extremum over the full depth.

    Parameters
    ----------
    curtain_field : numpy.ndarray
        ``(K, L)`` sampled (display-space) color field along the main axis.
    mode : {'min', 'max'}, optional
        Whether the "most extreme" point is the minimum (default; e.g. lowest
        Ri = most shear-unstable) or the maximum.

    Returns
    -------
    int
        Column index ``0..L-1`` of the extreme value, searched over all depths
        (full column).  NaNs are ignored.

    Raises
    ------
    ValueError
        If the curtain is entirely NaN, or ``mode`` is unrecognised.
    """
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
    if not np.isfinite(curtain_field).any():
        raise ValueError("curtain_field is entirely NaN; no extremum.")
    # Collapse depth: best (min or max) value in each column, then argmin/argmax.
    if mode == "min":
        per_col = np.nanmin(curtain_field, axis=0)
        return int(np.nanargmin(per_col))
    per_col = np.nanmax(curtain_field, axis=0)
    return int(np.nanargmax(per_col))


# ---------------------------------------------------------------------------
# Isopycnal-following coordinates
# ---------------------------------------------------------------------------

def trace_isopycnal(
    sigma0_curtain: np.ndarray,
    target: float,
    start_col: int,
) -> np.ndarray:
    """Trace one isopycnal down through a curtain, column by column.

    Fronts slope with depth, so the density surface a front lives on at the
    surface is displaced horizontally at depth.  For each depth row this finds
    the sub-pixel column where the curtain's density crosses ``target``,
    linearly interpolating between the two bracketing columns.  When a row has
    several crossings (folded isopycnal), the one nearest the previous depth's
    position is chosen so the trace stays continuous.  Rows shallower than
    the surface's first appearance are NaN (the isopycnal may sit below the
    shallowest level); once found, the trace stops at the first depth where
    the crossing disappears (isopycnal left the window or vanished) and
    deeper rows are NaN.

    Parameters
    ----------
    sigma0_curtain : numpy.ndarray
        ``(K, L)`` density curtain (NaN where invalid).
    target : float
        The sigma0 value to follow.
    start_col : int
        Column to anchor the trace at the shallowest depth (typically the
        front axis).

    Returns
    -------
    numpy.ndarray
        ``(K,)`` float array of sub-pixel column positions; NaN where the
        isopycnal was not found.
    """
    K, L = sigma0_curtain.shape
    xstar = np.full(K, np.nan)
    prev = float(start_col)
    started = False
    for k in range(K):
        d = sigma0_curtain[k] - target
        ok = np.isfinite(d)
        crossings = []
        for i in range(L - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            if d[i] == 0.0:
                crossings.append(float(i))
            elif d[i] * d[i + 1] < 0.0:
                crossings.append(i + d[i] / (d[i] - d[i + 1]))
        if ok[L - 1] and d[L - 1] == 0.0:
            crossings.append(float(L - 1))
        if not crossings:
            if started:
                break  # lost after being found: stop rather than jump
            continue  # surface sits deeper; keep looking down
        arr = np.asarray(crossings)
        prev = float(arr[np.argmin(np.abs(arr - prev))])
        xstar[k] = prev
        started = True
    return xstar


def recenter_curtain(curtain: np.ndarray, xstar: np.ndarray) -> np.ndarray:
    """Shift each curtain row horizontally so ``xstar`` lands on the centre.

    Puts the curtain in isopycnal-following (front-following) coordinates:
    after recentering, the traced isopycnal is a vertical line through the
    middle column, and every row shows structure *relative to the front*
    rather than at fixed positions.  Rows are resampled by linear
    interpolation; positions shifted outside the original window, and rows
    where ``xstar`` is NaN, come back NaN.

    Parameters
    ----------
    curtain : numpy.ndarray
        ``(K, L)`` curtain to recenter (color field or density).
    xstar : numpy.ndarray
        ``(K,)`` sub-pixel column of the isopycnal per depth, from
        :func:`trace_isopycnal`.

    Returns
    -------
    numpy.ndarray
        ``(K, L)`` recentered curtain (float64).
    """
    K, L = curtain.shape
    cols = np.arange(L, dtype=np.float64)
    center = (L - 1) / 2.0
    out = np.full((K, L), np.nan)
    for k in range(K):
        if not np.isfinite(xstar[k]):
            continue
        row = np.asarray(curtain[k], dtype=np.float64)
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        src = cols - center + xstar[k]  # where each output column samples from
        out[k] = np.interp(src, cols[ok], row[ok],
                           left=np.nan, right=np.nan)
        # np.interp bridges interior NaN gaps; re-mask anything that landed
        # nearer a missing column than a valid one.
        nearest = np.round(src).astype(int)
        inside = (nearest >= 0) & (nearest < L)
        bad = inside & ~ok[np.clip(nearest, 0, L - 1)]
        out[k][bad] = np.nan
    return out


def isopycnal_curtain(
    field3d: np.ndarray,
    sigma0_field3d: np.ndarray,
    axis_path: np.ndarray,
    normals: np.ndarray,
    target: float,
    half_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one isopycnal surface into an along-front curtain.

    This is the 2-D unrolling of "the field painted on an isopycnal surface"
    from the 3-D scene.  For each along-front column, a cross-front transect
    is cast (as in :func:`perpendicular_path`), the ``target`` isopycnal is
    traced down through it with :func:`trace_isopycnal`, and the field is
    interpolated *at the isopycnal's position* at every depth.  Each column
    therefore dives along the sloping density surface rather than straight
    down; the y-axis of the result is the depth of the surface itself.

    Parameters
    ----------
    field3d, sigma0_field3d : numpy.ndarray
        ``(K, J, I)`` field (display space) and density, cropped frame.
    axis_path : numpy.ndarray
        ``(L, 2)`` main-axis ``(j, i)`` coordinates.
    normals : numpy.ndarray
        ``(L, 2)`` unit normals from :func:`path_metrics`.
    target : float
        The sigma0 surface to flatten.
    half_width : int
        Cross-front search half-width in pixels; where the surface is
        displaced further than this from the axis, the column is NaN.

    Returns
    -------
    curtain : numpy.ndarray
        ``(K, L)`` field values on the surface; NaN where the surface is
        absent (outcropped, off-window, or invalid data).
    displacement : numpy.ndarray
        ``(K, L)`` signed cross-front displacement (px) of the surface from
        the main axis at each depth.
    """
    K = field3d.shape[0]
    L = axis_path.shape[0]
    W = 2 * int(half_width) + 1
    cols = np.arange(W, dtype=np.float64)
    curtain = np.full((K, L), np.nan)
    displacement = np.full((K, L), np.nan)
    for l in range(L):
        perp = perpendicular_path(axis_path, normals, l, half_width)
        sig = sample_curtain(sigma0_field3d, perp)
        fld = sample_curtain(field3d, perp)
        xs = trace_isopycnal(sig, target, int(half_width))
        for k in np.flatnonzero(np.isfinite(xs)):
            row = fld[k]
            ok = np.isfinite(row)
            if ok.sum() < 2:
                continue
            curtain[k, l] = np.interp(xs[k], cols[ok], row[ok],
                                      left=np.nan, right=np.nan)
            displacement[k, l] = xs[k] - half_width
    return curtain, displacement


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def plot_curtain_panel(
    ax,
    dist_px: np.ndarray,
    Z: np.ndarray,
    color_curtain: np.ndarray,
    sigma0_curtain: np.ndarray,
    *,
    dist_km: np.ndarray | None = None,
    levels: Sequence[float] | None = None,
    clim: tuple[float, float] | None = None,
    cmap: str = "RdYlBu",
    color_title: str = "",
    nan_color: str = "#9e9e9e",
    overlap_flags: np.ndarray | None = None,
    mld_curtain: np.ndarray | None = None,
    title: str | None = None,
    mark_index: int | None = None,
    add_colorbar: bool = True,
    contour_color: str = "k",
) -> None:
    """Draw a single curtain panel: color field + isopycnal contours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    dist_px : numpy.ndarray
        ``(L,)`` along-path distance in pixels (the x coordinate of columns).
    Z : numpy.ndarray
        ``(K,)`` depth axis in metres (negative downward).
    color_curtain : numpy.ndarray
        ``(K, L)`` display-space color field (e.g. log10 Ri).
    sigma0_curtain : numpy.ndarray
        ``(K, L)`` sigma0 field for the isopycnal contour overlay.
    dist_km : numpy.ndarray, optional
        ``(L,)`` along-path distance in km; when given, a secondary top x-axis
        labelled in km is attached.
    levels : sequence of float, optional
        sigma0 contour levels.  When None no isopycnals are drawn.
    clim : tuple of (float, float), optional
        Color limits for ``color_curtain``.  Default 2/98 percentile.
    cmap : str, optional
        Colormap for the color field (default ``'RdYlBu'``, the Ri style).
    color_title : str, optional
        Colorbar label.
    nan_color : str, optional
        Color for NaN cells of the color field (default neutral gray).
    overlap_flags : numpy.ndarray, optional
        ``(L,)`` boolean; columns flagged True are shaded (offset self-overlap).
    mld_curtain : numpy.ndarray, optional
        ``(L,)`` mixed-layer depth (metres, negative) sampled along the path;
        drawn as a dashed line when supplied.
    title : str, optional
        Panel title.
    mark_index : int, optional
        Column index to mark with a vertical line (e.g. the perpendicular
        point on an along-front panel).
    add_colorbar : bool, optional
        Whether to attach a colorbar to this panel (default True).
    contour_color : str, optional
        Color of the isopycnal contour lines (default black).
    """
    L = dist_px.shape[0]
    if clim is None:
        if np.isfinite(color_curtain).any():
            clim = (
                float(np.nanpercentile(color_curtain, 2)),
                float(np.nanpercentile(color_curtain, 98)),
            )
        else:
            clim = (0.0, 1.0)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)

    # pcolormesh needs cell edges; build them from the column centres (dist_px)
    # and the depth centres (Z).  Use midpoints + endpoint extrapolation.
    def _edges(centres: np.ndarray) -> np.ndarray:
        c = np.asarray(centres, dtype=np.float64)
        if c.size == 1:
            return np.array([c[0] - 0.5, c[0] + 0.5])
        mids = 0.5 * (c[:-1] + c[1:])
        first = c[0] - (mids[0] - c[0])
        last = c[-1] + (c[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]])

    x_edges = _edges(dist_px)
    z_edges = _edges(Z)

    masked = np.ma.masked_invalid(color_curtain)
    mesh = ax.pcolormesh(
        x_edges, z_edges, masked,
        cmap=cmap_obj, vmin=clim[0], vmax=clim[1], shading="flat",
    )

    # Isopycnal contours (drawn at column/depth centres).
    if levels is not None and np.isfinite(sigma0_curtain).any():
        Xc, Zc = np.meshgrid(dist_px, Z)
        try:
            cs = ax.contour(
                Xc, Zc, sigma0_curtain, levels=list(levels),
                colors=contour_color, linewidths=0.8,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        except ValueError:
            pass  # levels outside the data range -> no contours

    # Mixed-layer depth line.
    if mld_curtain is not None:
        ax.plot(dist_px, mld_curtain, color="white", lw=1.8, ls="--",
                label="mixed-layer depth")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # Overlap shading: vertical spans on flagged columns.
    if overlap_flags is not None and overlap_flags.any():
        x_e = x_edges
        for c in np.where(overlap_flags)[0]:
            ax.axvspan(x_e[c], x_e[c + 1], color="magenta", alpha=0.18, lw=0)

    # Mark a chosen column (e.g. the perpendicular point).
    if mark_index is not None:
        ax.axvline(dist_px[int(np.clip(mark_index, 0, L - 1))],
                   color="lime", lw=1.5, ls="-")

    ax.set_xlabel("distance along path [px]")
    ax.set_ylabel("depth [m]")
    if title:
        ax.set_title(title, fontsize=10, pad=24)  # pad leaves room for km axis

    # Colorbar in its own axes carved from the right of the panel, so it sits
    # fully to the RIGHT of the data instead of overlapping it.  Done before
    # the km twin so the twin inherits the panel's final (shrunk) position and
    # the two x-axes stay aligned.
    if add_colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.08)
        cbar = ax.figure.colorbar(mesh, cax=cax)
        cbar.set_label(color_title)

    # Secondary km axis on top (created last so it tracks ax's final position).
    if dist_km is not None:
        ax_km = ax.twiny()
        ax_km.set_xlim(ax.get_xlim())
        xticks = np.linspace(dist_px[0], dist_px[-1], 6)
        km_at = np.interp(xticks, dist_px, dist_km)
        ax_km.set_xticks(xticks)
        ax_km.set_xticklabels([f"{v:.1f}" for v in km_at])
        ax_km.set_xlabel("distance along path [km]")


# ---------------------------------------------------------------------------
# Figure assemblers (one per deliverable)
# ---------------------------------------------------------------------------

def figure_main_axis(
    color_field3d: np.ndarray,
    sigma0_field3d: np.ndarray,
    Z: np.ndarray,
    axis_path: np.ndarray,
    metrics: dict,
    output_path,
    *,
    levels: Sequence[float] | None = None,
    clim: tuple[float, float] | None = None,
    cmap: str = "RdYlBu",
    color_title: str = "",
    mld_curtain: np.ndarray | None = None,
    mark_index: int | None = None,
    title: str | None = None,
):
    """Figure 1 -- the main-axis curtain (single panel).

    Parameters
    ----------
    color_field3d, sigma0_field3d : numpy.ndarray
        ``(K, J, I)`` cropped+clipped color field and sigma0 (cropped frame).
    Z : numpy.ndarray
        ``(K,)`` depth axis.
    axis_path : numpy.ndarray
        ``(L, 2)`` main-axis ``(j, i)`` coordinates in the cropped frame.
    metrics : dict
        Output of :func:`path_metrics` for ``axis_path``.
    output_path : str or pathlib.Path
        PNG output path.
    levels, clim, cmap, color_title, mld_curtain, mark_index, title
        Passed through to :func:`plot_curtain_panel`.

    Returns
    -------
    pathlib.Path
        The output path.
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    color_curtain = sample_curtain(color_field3d, axis_path)
    sigma0_curtain = sample_curtain(sigma0_field3d, axis_path)

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_curtain_panel(
        ax, metrics["dist_px"], Z, color_curtain, sigma0_curtain,
        dist_km=metrics["dist_km"], levels=levels, clim=clim, cmap=cmap,
        color_title=color_title, mld_curtain=mld_curtain,
        mark_index=mark_index,
        title=title or "Main-axis curtain",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def figure_offsets(
    color_field3d: np.ndarray,
    sigma0_field3d: np.ndarray,
    Z: np.ndarray,
    axis_path: np.ndarray,
    metrics: dict,
    n_offsets: int,
    output_path,
    *,
    levels: Sequence[float] | None = None,
    clim: tuple[float, float] | None = None,
    cmap: str = "RdYlBu",
    color_title: str = "",
    mark_index: int | None = None,
    title: str | None = None,
    trim: bool = True,
):
    """Figure 2 -- along-front curtains: summary means + individual offsets.

    Layout is ``n_offsets + 2`` rows x 2 columns:

    * **Row 0** -- main-axis curtain (left) and the **mean over all** ``2N``
      offsets (right).  The all-offset mean is a *dilation* of the front: the
      average field in a band ``1..N`` px to either side.
    * **Row 1** -- mean over the **+** offsets (left) and mean over the **-**
      offsets (right).  These are *directional dilations* (one side of the
      front each).
    * **Rows 2..N+1** -- the individual offsets: offset ``r`` px on the +side
      (left) and -side (right).

    When ``trim`` is True (default), each offset polyline is "sewn" shut with
    :func:`trim_offset_loops` -- the self-intersection loops on the concave
    side of bends are excised and those columns render as neutral-gray gaps.
    The trimmed (NaN) columns are excluded from the row-0/row-1 means via
    ``nanmean``, so each averaged column only pools the offsets whose geometry
    is valid there.  When ``trim`` is False the loops are kept and shaded
    magenta via :func:`offset_quality_flags` (and counted in the means).

    Parameters
    ----------
    color_field3d, sigma0_field3d, Z, axis_path, metrics
        As in :func:`figure_main_axis`.
    n_offsets : int
        Number of individual offset rows per side (``N``).
    output_path : str or pathlib.Path
        PNG output path.
    levels, clim, cmap, color_title, mark_index, title
        Display options.

    Returns
    -------
    pathlib.Path
        The output path.
    """
    from pathlib import Path
    import warnings
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    side_a, side_b = offset_paths(axis_path, metrics["normals"], n_offsets)
    dist_px = metrics["dist_px"]
    dist_km = metrics["dist_km"]

    # Shared clim across all panels for comparability: derive from the axis
    # curtain if not supplied.
    axis_color = sample_curtain(color_field3d, axis_path)
    axis_sigma0 = sample_curtain(sigma0_field3d, axis_path)
    if clim is None and np.isfinite(axis_color).any():
        clim = (
            float(np.nanpercentile(axis_color, 2)),
            float(np.nanpercentile(axis_color, 98)),
        )

    def _sample_offset(pth):
        """Sample one offset path; trim looped columns to NaN (or flag them)."""
        cc = sample_curtain(color_field3d, pth)
        ss = sample_curtain(sigma0_field3d, pth)
        flags = None
        n_drop = 0
        if trim:
            keep = trim_offset_loops(pth)
            n_drop = int((~keep).sum())
            cc[:, ~keep] = np.nan
            ss[:, ~keep] = np.nan
        else:
            flags = offset_quality_flags(pth)
        return cc, ss, flags, n_drop

    # Sample every individual offset once; reuse for both the rows and the
    # mean panels.
    a_samples = [_sample_offset(p) for p in side_a]   # +side, offsets 1..N
    b_samples = [_sample_offset(p) for p in side_b]   # -side, offsets 1..N

    def _nanmean(stack_color, stack_sigma0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mc = np.nanmean(np.stack(stack_color), axis=0)
            ms = np.nanmean(np.stack(stack_sigma0), axis=0)
        return mc, ms

    pos_c = [s[0] for s in a_samples]
    pos_s = [s[1] for s in a_samples]
    neg_c = [s[0] for s in b_samples]
    neg_s = [s[1] for s in b_samples]
    mean_all_c, mean_all_s = _nanmean(pos_c + neg_c, pos_s + neg_s)
    mean_pos_c, mean_pos_s = _nanmean(pos_c, pos_s)
    mean_neg_c, mean_neg_s = _nanmean(neg_c, neg_s)

    n_rows = n_offsets + 2
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(16, 3.2 * n_rows), squeeze=False,
    )

    def _panel(ax, cc, ss, *, title, flags=None, mk=None, cbar=False):
        plot_curtain_panel(
            ax, dist_px, Z, cc, ss,
            dist_km=dist_km, levels=levels, clim=clim, cmap=cmap,
            color_title=color_title, overlap_flags=flags,
            mark_index=mk, title=title, add_colorbar=cbar,
        )

    # ----- Row 0: main axis | mean over ALL offsets (dilation) -----
    _panel(axes[0][0], axis_color, axis_sigma0,
           title="main axis (offset 0)", mk=mark_index)
    _panel(axes[0][1], mean_all_c, mean_all_s,
           title=f"mean of all +/-{n_offsets} offsets (dilation)",
           mk=mark_index, cbar=True)

    # ----- Row 1: mean over + offsets | mean over - offsets -----
    _panel(axes[1][0], mean_pos_c, mean_pos_s,
           title=f"mean of +offsets 1..{n_offsets} (+dilation)", mk=mark_index)
    _panel(axes[1][1], mean_neg_c, mean_neg_s,
           title=f"mean of -offsets 1..{n_offsets} (-dilation)",
           mk=mark_index, cbar=True)

    # ----- Rows 2..N+1: individual offsets (+side left, -side right) -----
    for k in range(n_offsets):
        row = k + 2
        ac, as_, aflags, adrop = a_samples[k]
        bc, bs, bflags, bdrop = b_samples[k]
        note = (lambda n: f"  [{n} looped cols trimmed]" if (trim and n) else
                (f"  [{n} overlap cols shaded]" if (not trim and n) else ""))
        a_n = adrop if trim else (int(aflags.sum()) if aflags is not None else 0)
        b_n = bdrop if trim else (int(bflags.sum()) if bflags is not None else 0)
        _panel(axes[row][0], ac, as_,
               title=f"+side: offset {k + 1} px" + note(a_n), flags=aflags)
        _panel(axes[row][1], bc, bs,
               title=f"-side: offset {k + 1} px" + note(b_n), flags=bflags,
               cbar=True)

    fig.suptitle(title or "Along-front curtains: dilation + offsets",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def figure_perpendicular(
    color_field3d: np.ndarray,
    sigma0_field3d: np.ndarray,
    Z: np.ndarray,
    perp_path: np.ndarray,
    half_width: int,
    output_path,
    *,
    XC_rect: np.ndarray | None = None,
    YC_rect: np.ndarray | None = None,
    levels: Sequence[float] | None = None,
    clim: tuple[float, float] | None = None,
    cmap: str = "RdYlBu",
    color_title: str = "",
    title: str | None = None,
    follow_isopycnal: bool = False,
    target_sigma0: float | None = None,
):
    """Figure 3 -- the perpendicular (cross-front) curtain (single panel).

    The x-axis is signed cross-front distance: 0 at the main axis, negative on
    side B, positive on side A.

    With ``follow_isopycnal=True`` the curtain is drawn in *front-following*
    coordinates: the isopycnal that the front lives on at the surface (or
    ``target_sigma0``) is traced down through the transect, and each depth row
    is shifted horizontally so that isopycnal sits at x=0.  The front is then
    a vertical line and the x-axis reads distance *from the front* at every
    depth, instead of distance from the surface axis point.  Depths where the
    isopycnal leaves the transect window are blank.

    Parameters
    ----------
    color_field3d, sigma0_field3d, Z
        As in :func:`figure_main_axis`.
    perp_path : numpy.ndarray
        ``(2*half_width+1, 2)`` transect coordinates from
        :func:`perpendicular_path`.
    half_width : int
        Half-width used to build ``perp_path`` (sets the signed x range).
    output_path : str or pathlib.Path
        PNG output path.
    XC_rect, YC_rect : numpy.ndarray, optional
        For a km twin axis along the transect.
    levels, clim, cmap, color_title, title
        Display options.
    follow_isopycnal : bool, optional
        Recenter each depth row on the traced isopycnal (see above).
    target_sigma0 : float, optional
        Density surface to follow.  Default: the shallowest finite sigma0 at
        the transect centre (the front's surface density).

    Returns
    -------
    pathlib.Path
        The output path.
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = path_metrics(perp_path, XC_rect, YC_rect, smooth=False)
    color_curtain = sample_curtain(color_field3d, perp_path)
    sigma0_curtain = sample_curtain(sigma0_field3d, perp_path)

    # Signed cross-front distance in pixels: -half_width .. +half_width.
    signed_px = np.arange(-int(half_width), int(half_width) + 1, dtype=float)

    # Signed km for the twin axis (when lon/lat provided): re-center the
    # cumulative along-transect distance so 0 km sits at the front axis.
    # (In isopycnal-following mode this is the surface spacing; row shifts
    # are small relative to the transect so the km scale is unchanged.)
    signed_km = None
    if metrics["dist_km"] is not None:
        signed_km = metrics["dist_km"] - metrics["dist_km"][int(half_width)]

    xlabel = "cross-front distance [px]  (0 = front axis)"
    if follow_isopycnal:
        centre = sigma0_curtain[:, int(half_width)]
        if target_sigma0 is None:
            finite = np.flatnonzero(np.isfinite(centre))
            if finite.size == 0:
                raise ValueError(
                    "Cannot pick a target isopycnal: sigma0 is NaN at every "
                    "depth of the transect centre."
                )
            target_sigma0 = float(centre[finite[0]])
        xstar = trace_isopycnal(sigma0_curtain, target_sigma0,
                                int(half_width))
        n_ok = int(np.isfinite(xstar).sum())
        log.info("Isopycnal sigma0=%.4f traced through %d/%d depth levels.",
                 target_sigma0, n_ok, xstar.size)
        color_curtain = recenter_curtain(color_curtain, xstar)
        sigma0_curtain = recenter_curtain(sigma0_curtain, xstar)
        xlabel = (f"cross-front distance [px]  "
                  f"(0 = sigma0={target_sigma0:.3f} isopycnal)")

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_curtain_panel(
        ax, signed_px, Z, color_curtain, sigma0_curtain,
        dist_km=signed_km,
        levels=levels, clim=clim, cmap=cmap, color_title=color_title,
        mark_index=int(half_width),  # x=0: the axis, or the traced isopycnal
        title=title or ("Cross-front curtain (isopycnal-following)"
                        if follow_isopycnal
                        else "Cross-front (perpendicular) curtain"),
    )
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def figure_isopycnal_surface(
    color_field3d: np.ndarray,
    sigma0_field3d: np.ndarray,
    Z: np.ndarray,
    axis_path: np.ndarray,
    metrics: dict,
    half_width: int,
    output_path,
    *,
    target_sigma0: float | None = None,
    clim: tuple[float, float] | None = None,
    cmap: str = "RdYlBu",
    color_title: str = "",
    mark_index: int | None = None,
    title: str | None = None,
):
    """Figure 4 -- the front's isopycnal surface flattened to 2-D.

    The 3-D scene paints the field on isopycnal surfaces; this takes the one
    surface the front lives on and unrolls it: x is distance along the front,
    y is depth, and the color at ``(x, z)`` is the field *on the surface*
    where it passes through depth ``z`` at that along-front position -- each
    column follows the sloping surface down-and-sideways instead of sampling
    straight below the axis.  Blank cells are where the surface is above/
    below the depth range, displaced more than ``half_width`` px from the
    axis, or over invalid data.

    Parameters
    ----------
    color_field3d, sigma0_field3d, Z, axis_path, metrics
        As in :func:`figure_main_axis`.
    half_width : int
        Cross-front search half-width (px) for the surface at depth.
    output_path : str or pathlib.Path
        PNG output path.
    target_sigma0 : float, optional
        Density surface to flatten.  Default: the median of the shallowest
        finite sigma0 sampled along the main axis (the front's surface
        density).
    clim, cmap, color_title, mark_index, title
        Display options.

    Returns
    -------
    pathlib.Path
        The output path.
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if target_sigma0 is None:
        sig_axis = sample_curtain(sigma0_field3d, axis_path)  # (K, L)
        # Shallowest finite sigma0 per column, median along the front.
        surface = np.full(sig_axis.shape[1], np.nan)
        for l in range(sig_axis.shape[1]):
            finite = np.flatnonzero(np.isfinite(sig_axis[:, l]))
            if finite.size:
                surface[l] = sig_axis[finite[0], l]
        if not np.isfinite(surface).any():
            raise ValueError("Cannot pick a target isopycnal: sigma0 is NaN "
                             "everywhere along the main axis.")
        target_sigma0 = float(np.nanmedian(surface))

    curtain, displacement = isopycnal_curtain(
        color_field3d, sigma0_field3d, axis_path, metrics["normals"],
        target_sigma0, half_width,
    )
    n_ok = int(np.isfinite(curtain).sum())
    log.info("Isopycnal sigma0=%.4f surface: %d/%d curtain cells filled, "
             "max |displacement| %.1f px.",
             target_sigma0, n_ok, curtain.size,
             float(np.nanmax(np.abs(displacement)))
             if np.isfinite(displacement).any() else float("nan"))

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_curtain_panel(
        ax, metrics["dist_px"], Z, curtain,
        np.full_like(curtain, np.nan),  # sigma0 is constant on the surface
        dist_km=metrics["dist_km"], levels=None,
        clim=clim, cmap=cmap, color_title=color_title,
        mark_index=mark_index,
        title=title or (f"Field on the sigma0={target_sigma0:.3f} "
                        "isopycnal (flattened)"),
    )
    ax.set_xlabel("distance along front [px]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
