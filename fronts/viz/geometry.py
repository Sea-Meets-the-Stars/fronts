"""Front geometry helpers that need no 3-D rendering stack.

``front_bbox_and_crop`` and ``truncate_depth`` are pure NumPy, but they
were defined in :mod:`fronts.viz.fronts_3d`, which imports PyVista at
module load.  That made the 2-D curtain figures -- matplotlib only --
unavailable on any machine without a 3-D stack installed, including a
headless cluster node with no OSMesa build.

They live here instead.  ``fronts_3d`` re-exports both names, so every
existing import keeps working unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as scimg


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
