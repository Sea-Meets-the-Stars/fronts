"""
PyVista builders for 3-D visualisations of labelled fronts in LLC4320 tiles.

This module wraps the generic helpers in ``pv_helpers.py`` with the
front-specific operations needed by
``fronts/scripts/fronts_viz_3d.py``:

* `front_bbox_and_crop`     -- bbox of a labelled front + margin.
* `truncate_depth`          -- clip a 3-D volume to a few levels below MLD.
* `build_pyvista_grid`      -- wrap a cropped+clipped sigma0 array as a
                               PyVista ``RectilinearGrid``.
* `decompose_front_branches` -- split a thinned front mask into its
                                constituent single-branch polylines.
* `build_front_curtain`     -- extrude each branch into a vertical
                                sigma0-coloured ribbon.
* `pick_isopycnals_across_front` -- choose default isopycnal levels
                                    bracketing the front's surface
                                    density contrast.
* `render_3d`               -- orchestrate the scene (isopycnals or
                                volume) + curtain + axes.

Imports follow the patterns in ``.claude/skills/pyvista-scientific-viz/
references/patterns_reference.md`` (in particular the Fortran-order
ravel for RectilinearGrid scalar arrays).
"""

# stdlib
from __future__ import annotations
from typing import Sequence

# numerical / IO
import numpy as np
from scipy import ndimage as scimg

# 3-D rendering
import pyvista as pv

# Generic helpers shipped with the repo.  These provide the scientific
# theme, off-screen rendering, and labelled scalar bar.
from fronts.viz.pv_helpers import (
    ensure_display,
    new_plotter,
    scientific_theme,
    add_scalar_field,
)


# ---------------------------------------------------------------------------
# Geometry: bbox + depth-clip
# ---------------------------------------------------------------------------

def front_bbox_and_crop(
    labels_tile: np.ndarray,
    selected_label: int,
    margin: int = 50,
) -> tuple[slice, slice]:
    """Tile-local rect-frame slices covering the bbox of one labelled front.

    Parameters
    ----------
    labels_tile : numpy.ndarray
        Integer label mask in the rect-grid tile-local frame, shape
        ``(TILE_SIZE, TILE_SIZE)``.  ``0`` marks "no front".
    selected_label : int
        Non-zero label whose bbox should be returned.
    margin : int, optional
        Number of pixels to pad on each side, clipped to the tile bounds.
        Default 50.

    Returns
    -------
    tuple of (slice, slice)
        ``(j_slice, i_slice)`` suitable for indexing into rect-frame
        arrays.

    Raises
    ------
    ValueError
        If ``selected_label`` does not appear in the tile.
    """
    ys, xs = np.where(labels_tile == selected_label)
    if ys.size == 0:
        raise ValueError(
            f"label {selected_label} does not appear in the labels tile."
        )
    j_lo = max(0, int(ys.min()) - margin)
    j_hi = min(labels_tile.shape[0], int(ys.max()) + margin + 1)
    i_lo = max(0, int(xs.min()) - margin)
    i_hi = min(labels_tile.shape[1], int(xs.max()) + margin + 1)
    return slice(j_lo, j_hi), slice(i_lo, i_hi)


def truncate_depth(
    sigma0_cropped: np.ndarray,
    Z: np.ndarray,
    k_mld_cropped: np.ndarray,
    n_below: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Clip a 3-D volume to a few model levels below the deepest MLD.

    Parameters
    ----------
    sigma0_cropped : numpy.ndarray
        Shape ``(K, J', I')``; sigma0 already cropped to the front's
        bbox (+margin).
    Z : numpy.ndarray
        1-D depth array, length ``K``.
    k_mld_cropped : numpy.ndarray
        Shape ``(J', I')``; LLC level index of the MLD per column.  -1
        marks columns where no MLD could be computed.
    n_below : int, optional
        Number of LLC levels below the deepest MLD index to include in
        the clipped volume (default 3).

    Returns
    -------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` with ``K_clip = min(K, max(k_mld) + n_below + 1)``.
    Z_clipped : numpy.ndarray
        Shape ``(K_clip,)``; the corresponding depth axis.
    k_clip : int
        Number of retained levels (``K_clip``).
    """
    K = sigma0_cropped.shape[0]
    valid_k_mld = k_mld_cropped[k_mld_cropped >= 0]
    if valid_k_mld.size == 0:
        # No MLD anywhere in the bbox -- fall back to the full column so
        # we still render something instead of crashing.
        k_clip = K
    else:
        k_clip = min(K, int(valid_k_mld.max()) + n_below + 1)
    return sigma0_cropped[:k_clip], Z[:k_clip], k_clip


# ---------------------------------------------------------------------------
# 3-D grid construction
# ---------------------------------------------------------------------------

def build_pyvista_grid(
    sigma0_clipped: np.ndarray,
    Z_clipped: np.ndarray,
    j_slice: slice,
    i_slice: slice,
    zscale: float = 50.0,
    *,
    mask_2d: np.ndarray | None = None,
) -> pv.RectilinearGrid:
    """Wrap a cropped+clipped sigma0 array as a PyVista ``RectilinearGrid``.

    The grid is regular in ``i`` and ``j`` (rect-tile pixel coordinates,
    matching the user's locator frame) and irregular in ``z`` (the LLC
    depth axis, multiplied by ``zscale`` so the depth direction is
    visible against the much wider horizontal extent).

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- the field to render.
    Z_clipped : numpy.ndarray
        Shape ``(K_clip,)`` -- depths in metres (negative downward).
    j_slice, i_slice : slice
        The rect-frame slices the field was cropped to; used to position
        the grid in absolute tile-local pixel coordinates so the figure's
        x/y axes show meaningful values.
    zscale : float, optional
        Vertical exaggeration factor applied to ``Z_clipped`` (default 50).
    mask_2d : numpy.ndarray, optional
        Optional 2-D boolean mask of shape ``(J', I')``.  When supplied,
        sigma0 is set to ``NaN`` everywhere the mask is False, so
        downstream filters (``grid.contour``, etc.) only return surfaces
        inside the mask.  Use this together with
        :func:`dilate_front_mask` to render iso-surfaces only in a
        dilated region around the selected front.

    Returns
    -------
    pyvista.RectilinearGrid
        Grid with ``grid["sigma0"]`` stamped using Fortran order (the
        documented silent-bug trap in PyVista's RectilinearGrid).
    """
    K_clip, Jp, Ip = sigma0_clipped.shape

    # Coordinate arrays are POINT counts (Jp, Ip, K_clip).  Using the
    # rect-frame slice origin keeps the axes labelled in the same
    # coordinate system as the user's --i/--j input.
    x = np.arange(i_slice.start, i_slice.start + Ip, dtype=np.float64)
    y = np.arange(j_slice.start, j_slice.start + Jp, dtype=np.float64)
    z = Z_clipped.astype(np.float64) * float(zscale)

    grid = pv.RectilinearGrid(x, y, z)

    # VTK expects Fortran-ordered scalar arrays for image/rectilinear grids;
    # the patterns reference flags C-order as the most common silent bug.
    # sigma0_clipped has axes (k, j, i); we want the iteration order to
    # match (x, y, z) = (i, j, k), so transpose first then ravel('F').
    field = sigma0_clipped.astype(np.float32, copy=True)
    if mask_2d is not None:
        if mask_2d.shape != (Jp, Ip):
            raise ValueError(
                f"mask_2d shape {mask_2d.shape} does not match the (J, I) "
                f"axes of sigma0_clipped ({Jp}, {Ip})."
            )
        # Broadcast the 2-D mask along k so the same (j,i) pixels are
        # active at every depth; contour() will then trace surfaces
        # restricted to the columns where the mask is True.
        field[:, ~mask_2d] = np.nan
    field = np.transpose(field, (2, 1, 0))
    grid["sigma0"] = field.ravel(order="F")
    return grid


def dilate_front_mask(
    front_mask: np.ndarray,
    iterations: int = 2,
) -> np.ndarray:
    """Binary-dilate a 2-D front mask by ``iterations`` pixels (8-connectivity).

    Convenience wrapper around :func:`scipy.ndimage.binary_dilation` that
    encodes the project's 8-connectivity convention.  The dilated mask
    captures the front and a halo around it -- useful for restricting
    iso-surface rendering to "near the front" rather than the whole
    bbox.

    Parameters
    ----------
    front_mask : numpy.ndarray
        2-D boolean mask, True at the selected front's pixels.
    iterations : int, optional
        Number of binary-dilation passes.  Each pass adds one pixel of
        halo.  Default 2.

    Returns
    -------
    numpy.ndarray
        2-D boolean mask of the same shape, dilated by ``iterations``
        pixels along the 8-neighbour structuring element.
    """
    struct = scimg.generate_binary_structure(2, 2)  # 8-connectivity
    return scimg.binary_dilation(
        front_mask, structure=struct, iterations=int(iterations),
    )


def front_volume_clim(
    sigma0_clipped: np.ndarray,
    dilated_mask_2d: np.ndarray,
    *,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[float, float]:
    """Percentile-based colour limits computed inside the dilated-front column.

    Unlike :func:`mixed_layer_clim` (which uses every column above its
    own per-column MLD across the full bbox), this helper restricts the
    pool to the dilated-front 2-D region at *every* depth.  That gives
    even tighter contrast for visualising the front's iso-surface tilt
    structure.

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- cropped + depth-clipped sigma0.
    dilated_mask_2d : numpy.ndarray
        2-D boolean mask of shape ``(J', I')`` selecting the front
        columns plus a small halo (typically from
        :func:`dilate_front_mask`).
    percentile_low, percentile_high : float, optional
        Percentile bounds (default 2/98).

    Returns
    -------
    tuple of (float, float)
        ``(lo, hi)`` colour limits.  Falls back to the full clipped
        volume percentiles if the dilated mask is empty (shouldn't
        normally happen).
    """
    if not dilated_mask_2d.any():
        return (
            float(np.nanpercentile(sigma0_clipped, percentile_low)),
            float(np.nanpercentile(sigma0_clipped, percentile_high)),
        )
    pool = sigma0_clipped[:, dilated_mask_2d]
    return (
        float(np.nanpercentile(pool, percentile_low)),
        float(np.nanpercentile(pool, percentile_high)),
    )


# ---------------------------------------------------------------------------
# Front polyline decomposition + curtain
# ---------------------------------------------------------------------------

# 8-neighbour offsets, ordered so that cardinal neighbours come first.
# Used by the branch-decomposition DFS below.
_NBRS = (
    (-1,  0), (1,  0), (0, -1), (0,  1),
    (-1, -1), (-1, 1), (1, -1), (1,  1),
)


def decompose_front_branches(
    front_mask: np.ndarray,
) -> list[np.ndarray]:
    """Split a thinned front mask into a list of single-branch polylines.

    The V4 labelled fronts are morphologically thinned to 1 pixel wide
    skeletons but may still branch (Y-junctions, T-junctions) or form
    short loops.  We decompose the skeleton into simple branches between
    "junctions" (pixels with >=3 neighbours) and "endpoints" (pixels
    with <=1 neighbour) using a small custom DFS over the 8-neighbour
    adjacency -- no extra dependencies required.

    Parameters
    ----------
    front_mask : numpy.ndarray
        2-D boolean array, True at the front's pixels.

    Returns
    -------
    list of numpy.ndarray
        Each branch is a 2-D array of shape ``(L, 2)`` with columns
        ``(j, i)`` ordered along the polyline.  Branches of length 1 are
        included so isolated pixels don't silently disappear.
    """
    H, W = front_mask.shape
    mask = front_mask.astype(bool)

    # 8-neighbour degree.  Junctions = degree >= 3; endpoints = degree <= 1.
    deg = scimg.convolve(
        mask.astype(np.int32),
        np.ones((3, 3), dtype=np.int32),
        mode="constant",
    ) - mask.astype(np.int32)
    deg[~mask] = 0
    junctions = mask & (deg >= 3)
    endpoints = mask & (deg <= 1)

    # Walk every front pixel along edges between neighbours that are not
    # both junctions; each walk produces one branch.  We keep a visited
    # set keyed by *edge* so we never reuse an edge but can revisit a
    # junction pixel multiple times (once per incident branch).
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def _walk(start_jy: int, start_ix: int, first_jy: int, first_ix: int) -> np.ndarray:
        # Walk the polyline starting at (start_jy, start_ix) and going
        # through (first_jy, first_ix); stops at the next junction/endpoint.
        branch = [(start_jy, start_ix), (first_jy, first_ix)]
        edge = tuple(sorted([(start_jy, start_ix), (first_jy, first_ix)]))
        visited_edges.add(edge)
        prev_jy, prev_ix = start_jy, start_ix
        cur_jy, cur_ix = first_jy, first_ix
        while True:
            if junctions[cur_jy, cur_ix] or endpoints[cur_jy, cur_ix]:
                break
            # The "next" neighbour is the unique other front neighbour;
            # if the skeleton is well thinned there should be exactly one,
            # but we defend against malformed input by breaking on tie.
            nxt: tuple[int, int] | None = None
            for dj, di in _NBRS:
                nj, ni = cur_jy + dj, cur_ix + di
                if not (0 <= nj < H and 0 <= ni < W):
                    continue
                if not mask[nj, ni]:
                    continue
                if (nj, ni) == (prev_jy, prev_ix):
                    continue
                nxt = (nj, ni)
                break
            if nxt is None:
                break
            edge = tuple(sorted([(cur_jy, cur_ix), nxt]))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            branch.append(nxt)
            prev_jy, prev_ix = cur_jy, cur_ix
            cur_jy, cur_ix = nxt
        return np.array(branch, dtype=np.int32)

    branches: list[np.ndarray] = []
    seed_points = np.argwhere(endpoints | junctions)
    for jy, ix in seed_points:
        for dj, di in _NBRS:
            nj, ni = jy + dj, ix + di
            if not (0 <= nj < H and 0 <= ni < W):
                continue
            if not mask[nj, ni]:
                continue
            edge = tuple(sorted([(int(jy), int(ix)), (int(nj), int(ni))]))
            if edge in visited_edges:
                continue
            branches.append(_walk(int(jy), int(ix), int(nj), int(ni)))

    # Closed loops contain no junctions or endpoints; the loop above will
    # miss them.  Sweep over any remaining front pixels and walk them as
    # rings (the start pixel will appear twice -- once at each end).
    visited_pixels = np.zeros_like(mask)
    for branch in branches:
        visited_pixels[branch[:, 0], branch[:, 1]] = True
    remaining = mask & ~visited_pixels
    rem_pts = np.argwhere(remaining)
    for jy, ix in rem_pts:
        if visited_pixels[jy, ix]:
            continue
        # Walk in one direction until we come back; treat as a ring.
        # Pick any front neighbour to start the walk.
        for dj, di in _NBRS:
            nj, ni = int(jy) + dj, int(ix) + di
            if 0 <= nj < H and 0 <= ni < W and mask[nj, ni]:
                ring = _walk(int(jy), int(ix), int(nj), int(ni))
                branches.append(ring)
                visited_pixels[ring[:, 0], ring[:, 1]] = True
                break
        else:
            # Isolated single pixel; keep it so it doesn't vanish.
            branches.append(np.array([[int(jy), int(ix)]], dtype=np.int32))
            visited_pixels[jy, ix] = True

    return branches


def build_front_curtain(
    front_mask: np.ndarray,
    sigma0_clipped: np.ndarray,
    Z_clipped: np.ndarray,
    j_slice: slice,
    i_slice: slice,
    zscale: float = 50.0,
) -> pv.MultiBlock:
    """Extrude each branch of a thinned front mask into a sigma0-coloured ribbon.

    Each branch becomes a ``pv.StructuredGrid`` whose horizontal
    cross-section is the branch polyline and whose vertical extent
    spans ``Z_clipped`` (multiplied by ``zscale`` to match the volume).
    sigma0 is sampled along the ribbon via
    :func:`scipy.ndimage.map_coordinates` and attached as a point-data
    scalar named ``"sigma0"`` so the renderer can colour by it.

    Parameters
    ----------
    front_mask : numpy.ndarray
        2-D boolean mask on the *cropped* rect-frame window (same shape
        as the j/i axes of ``sigma0_clipped``).
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')``; the same field passed to
        :func:`build_pyvista_grid`.
    Z_clipped : numpy.ndarray
        1-D depths, length ``K_clip``, in metres (negative downward).
    j_slice, i_slice : slice
        Tile-local rect-frame slices used to position the ribbons in the
        same coordinate system as the grid built by
        :func:`build_pyvista_grid`.
    zscale : float, optional
        Vertical exaggeration applied to ``Z_clipped``.

    Returns
    -------
    pyvista.MultiBlock
        One ``StructuredGrid`` per branch.  Empty if the front mask has
        no True pixels.
    """
    branches = decompose_front_branches(front_mask)
    multiblock = pv.MultiBlock()
    if not branches:
        return multiblock

    # Cache the branch list on the MultiBlock so downstream helpers
    # (e.g. build_front_top_marker) can reuse the decomposition without
    # re-running the DFS.
    multiblock._fronts_branches = branches  # type: ignore[attr-defined]
    K_clip = Z_clipped.shape[0]
    z_axis = Z_clipped.astype(np.float64) * float(zscale)

    for b_idx, branch in enumerate(branches):
        L = branch.shape[0]
        # Branch coordinates in the cropped frame for sampling sigma0;
        # then shift into absolute tile-local pixel coords for the grid.
        j_local = branch[:, 0].astype(np.float64)
        i_local = branch[:, 1].astype(np.float64)

        # Sample sigma0 along the (k, j, i) volume.  We build coords of
        # shape (3, K_clip * L): axis 0 = k, axis 1 = branch step.
        kk, ss = np.meshgrid(np.arange(K_clip), np.arange(L), indexing="ij")
        coords = np.stack([
            kk.ravel().astype(np.float64),
            np.repeat(j_local[None, :], K_clip, axis=0).ravel(),
            np.repeat(i_local[None, :], K_clip, axis=0).ravel(),
        ], axis=0)
        sampled = scimg.map_coordinates(
            sigma0_clipped, coords, order=1, mode="nearest",
        ).reshape(K_clip, L)

        # Build the StructuredGrid.  Dimensions are (i, j, k) = (L, 1, K).
        # We add a small thickness in y so the ribbon is visible as a
        # surface rather than a zero-area collapse.  Using 1 point in the
        # second axis keeps it a 2-D ribbon embedded in 3-D space.
        x_pts = (i_local + i_slice.start)
        y_pts = (j_local + j_slice.start)
        # Replicate the polyline once per depth level to get a structured
        # ribbon: shape (K_clip, L) per coordinate.
        Xg = np.broadcast_to(x_pts[None, :], (K_clip, L)).astype(np.float64)
        Yg = np.broadcast_to(y_pts[None, :], (K_clip, L)).astype(np.float64)
        Zg = np.broadcast_to(z_axis[:, None], (K_clip, L)).astype(np.float64)
        ribbon = pv.StructuredGrid(Xg, Yg, Zg)
        ribbon["sigma0"] = sampled.astype(np.float32).ravel(order="F")
        multiblock.append(ribbon, name=f"branch_{b_idx:03d}")

    return multiblock


# ---------------------------------------------------------------------------
# Isopycnal-level picking
# ---------------------------------------------------------------------------

def pick_isopycnals_across_front(
    sigma0_clipped: np.ndarray,
    front_mask_cropped: np.ndarray,
    Z_clipped: np.ndarray,
    user_levels: Sequence[float] | None = None,
    *,
    n_levels: int = 5,
    reference_depth_m: float = 10.0,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    buffer_radius: int = 1,
) -> np.ndarray:
    """Choose default isopycnal levels bracketing the front's surface contrast.

    If ``user_levels`` is supplied it is returned verbatim.  Otherwise
    the function samples sigma0 on the LLC level closest to
    ``reference_depth_m`` (matching the MLD reference depth used
    elsewhere in this repo), masks out the front itself + a 1-pixel
    buffer, then returns ``n_levels`` values evenly spaced between the
    5th and 95th percentile of the remaining values inside the cropped
    bbox.  This is round-3 option (i) ("near-surface percentile bracket
    excluding the front itself").

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- cropped + depth-clipped sigma0.
    front_mask_cropped : numpy.ndarray
        2-D boolean mask, shape ``(J', I')``, True at the selected
        front's pixels.
    Z_clipped : numpy.ndarray
        1-D depths, length ``K_clip``.
    user_levels : sequence of float, optional
        If provided, returned without modification.
    n_levels : int, optional
        Number of evenly spaced levels (default 5).
    reference_depth_m : float, optional
        Depth in metres at which to sample the cross-front contrast
        (default 10 m, matching the MLD reference depth).
    percentile_low, percentile_high : float, optional
        Percentiles used to define the cross-front sigma0 range.
    buffer_radius : int, optional
        Pixel radius of the dilation applied to the front mask before
        excluding it from the percentile pool (default 1).

    Returns
    -------
    numpy.ndarray
        1-D float array of isopycnal sigma0 values, length ``n_levels``.
    """
    if user_levels is not None:
        return np.asarray(list(user_levels), dtype=np.float64)

    # Reference slice -- closest LLC level to the requested reference depth.
    k_ref = int(np.abs(np.abs(Z_clipped) - float(reference_depth_m)).argmin())
    ref_slice = sigma0_clipped[k_ref]  # (J', I')

    # Dilate the front mask by buffer_radius pixels so we don't include
    # pixels that straddle the front itself.
    if buffer_radius > 0:
        struct = scimg.generate_binary_structure(2, 2)  # 8-connectivity
        front_buffered = scimg.binary_dilation(
            front_mask_cropped, structure=struct, iterations=buffer_radius,
        )
    else:
        front_buffered = front_mask_cropped

    pool = ref_slice.copy()
    pool[front_buffered] = np.nan
    sigma_lo = float(np.nanpercentile(pool, percentile_low))
    sigma_hi = float(np.nanpercentile(pool, percentile_high))
    if not (np.isfinite(sigma_lo) and np.isfinite(sigma_hi)):
        # Degenerate case (e.g. front covers all pixels in the bbox).
        # Fall back to percentiles over the whole reference slice.
        sigma_lo = float(np.nanpercentile(ref_slice, percentile_low))
        sigma_hi = float(np.nanpercentile(ref_slice, percentile_high))
    return np.linspace(sigma_lo, sigma_hi, n_levels)


# ---------------------------------------------------------------------------
# Top-layer marker (front projected onto the surface for at-a-glance location)
# ---------------------------------------------------------------------------

def build_front_top_marker(
    curtain: pv.MultiBlock,
    Z_clipped: np.ndarray,
    j_slice: slice,
    i_slice: slice,
    zscale: float = 50.0,
    *,
    tube_radius: float = 1.5,
    elevate_pixels: float = 0.0,
) -> pv.MultiBlock:
    """Build a bright polyline at the surface so the front's location is visible from above.

    The 3-D scene's volume render or isopycnal stack can hide the front
    when viewed from a low oblique camera.  This helper produces a
    `pv.MultiBlock` of tube-thickened polylines sitting at
    ``Z_clipped[0] * zscale + elevate_pixels`` so the front shows up as a
    clean line on the top face of the scene, regardless of opacity.

    Parameters
    ----------
    curtain : pyvista.MultiBlock
        Output of :func:`build_front_curtain`.  The branch polylines are
        reused so the marker matches the curtain exactly.
    Z_clipped : numpy.ndarray
        1-D depths, length ``K_clip``; only ``Z_clipped[0]`` is used.
    j_slice, i_slice : slice
        Tile-local rect-frame slices used to position the marker in the
        same coordinate system as the grid built by
        :func:`build_pyvista_grid`.  The cached branches are stored in
        cropped coordinates so we must add the slice origins back.
    zscale : float, optional
        Vertical exaggeration applied to ``Z_clipped`` (must match the
        zscale passed to :func:`build_pyvista_grid`).
    tube_radius : float, optional
        Radius of the cylindrical tube wrapped around each polyline, in
        the same units as the i/j axes (rect-frame pixels).  Default 1.5.
    elevate_pixels : float, optional
        Extra Z offset above the top layer so the marker isn't hidden
        behind transparent contour surfaces.  Default 0.

    Returns
    -------
    pyvista.MultiBlock
        One ``PolyData`` (after ``.tube(...)``) per branch.  Empty if the
        curtain has no branches.
    """
    out = pv.MultiBlock()
    branches = getattr(curtain, "_fronts_branches", None)
    if branches is None or not branches:
        return out

    z_top = float(Z_clipped[0]) * float(zscale) + float(elevate_pixels)
    di, dj = float(i_slice.start), float(j_slice.start)
    for b_idx, branch in enumerate(branches):
        if branch.shape[0] < 2:
            # Single-pixel branch: drop a tiny sphere as a placeholder.
            pt = pv.PolyData(np.array([
                [float(branch[0, 1]) + di, float(branch[0, 0]) + dj, z_top],
            ]))
            out.append(pt, name=f"top_branch_{b_idx:03d}")
            continue
        pts = np.column_stack([
            branch[:, 1].astype(np.float64) + di,  # i -> x in tile-local frame
            branch[:, 0].astype(np.float64) + dj,  # j -> y in tile-local frame
            np.full(branch.shape[0], z_top),
        ])
        # Build a single polyline cell so .tube() can wrap it.
        n = pts.shape[0]
        poly = pv.PolyData(pts)
        # VTK polyline cell connectivity: [n, idx0, idx1, ..., idx_{n-1}].
        poly.lines = np.hstack([[n], np.arange(n)]).astype(np.int32)
        tube = poly.tube(radius=tube_radius, n_sides=12)
        out.append(tube, name=f"top_branch_{b_idx:03d}")
    return out


# ---------------------------------------------------------------------------
# Contrast helper -- focus the colour scale on the mixed layer
# ---------------------------------------------------------------------------

def mixed_layer_clim(
    sigma0_clipped: np.ndarray,
    k_mld_cropped: np.ndarray,
    *,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    pad_levels: int = 0,
) -> tuple[float, float]:
    """Percentile-based colour limits computed inside the mixed layer only.

    Stretching ``--clim`` over the full depth-clipped volume crushes the
    cross-front contrast that lives in the upper ~80 m, because the
    deeper levels span a much wider density range.  This helper restricts
    the percentile pool to columns *above* ``k_mld_cropped`` per cell.

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- cropped + depth-clipped sigma0.
    k_mld_cropped : numpy.ndarray
        Shape ``(J', I')`` -- per-column MLD level index (-1 where
        undefined).
    percentile_low, percentile_high : float, optional
        Percentile bounds (default 2/98).
    pad_levels : int, optional
        Number of model levels to include below the per-column MLD when
        building the percentile pool (default 0 -- strict mixed layer).

    Returns
    -------
    tuple of (float, float)
        ``(lo, hi)`` colour limits.  Falls back to the full-volume
        percentile when the mixed-layer pool is empty.
    """
    K, J, I = sigma0_clipped.shape
    # Broadcast (k_indices, k_mld_cropped) to build a per-cell "in ML" mask.
    k_indices = np.arange(K, dtype=np.int32)[:, None, None]
    k_lim = np.where(k_mld_cropped >= 0, k_mld_cropped + pad_levels, -1)
    in_ml = (k_indices <= k_lim[None, :, :]) & (k_lim[None, :, :] >= 0)
    pool = np.where(in_ml, sigma0_clipped, np.nan)
    if not np.isfinite(pool).any():
        # No MLD pixels in the bbox: fall back to the whole clipped volume.
        return (
            float(np.nanpercentile(sigma0_clipped, percentile_low)),
            float(np.nanpercentile(sigma0_clipped, percentile_high)),
        )
    return (
        float(np.nanpercentile(pool, percentile_low)),
        float(np.nanpercentile(pool, percentile_high)),
    )


# ---------------------------------------------------------------------------
# Front iso-surface (single-level, used as a tilt indicator)
# ---------------------------------------------------------------------------

def build_front_isosurface(
    grid: pv.RectilinearGrid,
    level: float,
) -> pv.PolyData:
    """Single sigma0 iso-surface — naturally tilts with depth at a front.

    Picking one representative cross-front sigma0 (e.g. the median of
    the 5 levels returned by :func:`pick_isopycnals_across_front`) and
    contouring the grid at that value isolates the 3-D "front
    surface".  Because density surfaces slope across a front, the
    resulting iso-surface tilts with depth -- with denser water on one
    side and lighter water on the other -- which is exactly the
    structure the user wants to see.

    Parameters
    ----------
    grid : pyvista.RectilinearGrid
        Grid built by :func:`build_pyvista_grid`.
    level : float
        sigma0 value (kg m^-3) to contour.

    Returns
    -------
    pyvista.PolyData
        Triangulated iso-surface.  Possibly empty if ``level`` lies
        outside the data range.
    """
    return grid.contour(isosurfaces=[float(level)], scalars="sigma0")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_3d(
    grid: pv.RectilinearGrid,
    curtain: pv.MultiBlock,
    levels: np.ndarray,
    mode: str = "isopycnals",
    *,
    clim: tuple[float, float] | None = None,
    cmap_volume: str = "viridis",
    cmap_curtain: str = "magma",
    opacity: str = "sigmoid",
    zscale: float = 50.0,
    show: bool = False,
    top_marker: pv.MultiBlock | None = None,
    top_marker_color: str = "red",
    front_iso: pv.PolyData | None = None,
    front_iso_color: str = "orange",
    front_iso_opacity: float = 0.95,
    draw_curtain: bool = False,
    font_size: int = 40,
    title_font_size: int = 44,
    label_font_size: int = 32,
) -> pv.Plotter:
    """Assemble the 3-D scene: isopycnals / volume + curtain + top marker + axes.

    Parameters
    ----------
    grid : pyvista.RectilinearGrid
        Cropped + depth-clipped sigma0 grid from :func:`build_pyvista_grid`.
    curtain : pyvista.MultiBlock
        Per-branch sigma0-coloured ribbons from :func:`build_front_curtain`.
    levels : numpy.ndarray
        Isopycnal sigma0 values; used only in ``mode='isopycnals'``.
    mode : {'isopycnals', 'volume'}, optional
        ``'isopycnals'`` (default) draws ``grid.contour(levels)`` with a
        semi-transparent fill; ``'volume'`` calls ``add_volume`` with the
        chosen opacity transfer function.
    clim : tuple of (float, float), optional
        Colour-scale limits for sigma0; default 2/98 percentile of the
        grid scalars.  For better mixed-layer contrast pass
        :func:`mixed_layer_clim`'s output.
    cmap_volume, cmap_curtain : str, optional
        Colormaps for the background and curtain respectively.
    opacity : str, optional
        Opacity transfer function name for volume mode (default
        ``'sigmoid'``; see PyVista's opacity table).
    zscale : float, optional
        Z-axis exaggeration factor (only used for the axis label).
    show : bool, optional
        If True, opens an interactive window; otherwise renders
        off-screen (the default).
    top_marker : pyvista.MultiBlock, optional
        Optional surface-layer polyline (output of
        :func:`build_front_top_marker`) drawn in a single bright fixed
        colour so the front's lateral position is visible from above.
    top_marker_color : str, optional
        Colour of the top-layer marker (default ``'red'``).
    font_size, title_font_size, label_font_size : int, optional
        Font sizes for axes / scalar-bar titles / scalar-bar tick labels.

    Returns
    -------
    pyvista.Plotter
        Configured plotter; the caller saves and closes it.
    """
    ensure_display()
    # Apply the requested font sizes by building a theme up front.  The
    # base scientific_theme already sets sensible defaults; we override
    # the two font knobs that drive the bounds axis text and any
    # add_text() calls.
    theme = scientific_theme(font_size=font_size, label_size=label_font_size)
    pl = new_plotter(off_screen=not show, theme=theme)

    # Default contrast: 2/98 percentile of the grid scalars.  Callers
    # that want mixed-layer-focused contrast should pass `clim=` from
    # :func:`mixed_layer_clim` instead.
    if clim is None:
        scalars = np.asarray(grid["sigma0"])
        clim = (
            float(np.nanpercentile(scalars, 2)),
            float(np.nanpercentile(scalars, 98)),
        )

    # Shared scalar-bar styling.  Vertical bars positioned on the right
    # side, so multiple bars stack without overlapping their tick labels.
    # `position_x` is the bar's left edge in normalised viewport coords
    # (0 = left, 1 = right); `position_y` is the bottom edge.
    def _sb_args(title, position_y):
        return dict(
            title=title,
            n_labels=5,
            fmt="%.3g",
            title_font_size=title_font_size,
            label_font_size=label_font_size,
            shadow=False,
            vertical=True,
            position_x=0.86,
            position_y=position_y,
            width=0.06,
            height=0.40,
        )

    if mode == "isopycnals":
        iso = grid.contour(isosurfaces=list(levels), scalars="sigma0")
        if iso.n_points > 0:
            pl.add_mesh(
                iso,
                scalars="sigma0",
                cmap=cmap_volume,
                clim=clim,
                opacity=0.6,
                smooth_shading=True,
                scalar_bar_args=_sb_args(
                    "sigma0 [kg/m^3]", position_y=0.55,
                ),
            )
    elif mode == "volume":
        pl.add_volume(
            grid, scalars="sigma0",
            cmap=cmap_volume, clim=clim, opacity=opacity,
            scalar_bar_args=_sb_args(
                "sigma0 [kg/m^3]", position_y=0.55,
            ),
        )
    else:
        raise ValueError(
            f"Unknown render mode {mode!r}; expected 'isopycnals' or 'volume'."
        )

    # Front curtain -- a vertical sheet at the front's surface pixels.
    # Off by default because it competes visually with the iso-surfaces;
    # opt in via draw_curtain=True for cases where the vertical-sheet
    # reference is useful.
    if draw_curtain and curtain is not None:
        first = True
        for name in curtain.keys():
            ribbon = curtain[name]
            if ribbon is None or ribbon.n_points == 0:
                continue
            if first:
                pl.add_mesh(
                    ribbon,
                    scalars="sigma0",
                    cmap=cmap_curtain,
                    clim=clim,
                    smooth_shading=True,
                    scalar_bar_args=_sb_args(
                        "front sigma0 [kg/m^3]", position_y=0.10,
                    ),
                )
                first = False
            else:
                pl.add_mesh(
                    ribbon,
                    scalars="sigma0",
                    cmap=cmap_curtain,
                    clim=clim,
                    smooth_shading=True,
                    show_scalar_bar=False,
                )

    # Front iso-surface -- a single sigma0 iso-surface that naturally
    # tilts with depth across the front.  Drawn opaque in a distinct
    # colour so the tilt geometry (dense waters on one side, less dense
    # on the other) reads at a glance against the semi-transparent
    # context iso-surfaces.
    if front_iso is not None and front_iso.n_points > 0:
        pl.add_mesh(
            front_iso,
            color=front_iso_color,
            opacity=front_iso_opacity,
            smooth_shading=True,
            show_scalar_bar=False,
        )

    # Top-layer marker (optional).  Drawn in a single bright fixed colour
    # so the front's lateral position is unambiguous in plan view.
    if top_marker is not None and len(top_marker.keys()) > 0:
        for name in top_marker.keys():
            mesh = top_marker[name]
            if mesh is None or mesh.n_points == 0:
                continue
            pl.add_mesh(
                mesh,
                color=top_marker_color,
                show_scalar_bar=False,
                smooth_shading=True,
            )

    # Axis bounds.  Bumping the bounds font_size makes the depth + i/j
    # tick labels legible alongside the larger scalar bars on the right.
    pl.show_bounds(
        xtitle="i (rect)",
        ytitle="j (rect)",
        ztitle=f"depth x {zscale:.0f} [m]",
        location="outer",
        ticks="both",
        font_size=font_size,
    )
    return pl
