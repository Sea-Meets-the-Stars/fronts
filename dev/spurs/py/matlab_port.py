"""
Spur removal from binary front skeletons (Algorithm S1).

Direct Python port of the MATLAB prune_short_spurs function.
Traces from each endpoint inward along the skeleton to the first
branchpoint; if the traced path is shorter than Lspur, deletes it.
Iterates until stable.

References
----------
- dev/spurs/pmc_spurs.mat  (original MATLAB implementation)
- claude_spurs_plan.tex, Section S1
"""

import numpy as np
from scipy.ndimage import correlate
from skimage.morphology import skeletonize

# 3x3 kernel for counting 8-connected neighbors
_NEIGHBOR_KERNEL = np.array([[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]], dtype=np.uint8)

# 8-neighbor offsets (row, col)
_OFFSETS = np.array([[-1, -1], [-1, 0], [-1, 1],
                     [ 0, -1],          [ 0, 1],
                     [ 1, -1], [ 1, 0], [ 1, 1]], dtype=np.int32)


def _neighbor_count(skeleton):
    """Count 8-connected skeleton neighbors for each pixel."""
    return correlate(skeleton.astype(np.uint8), _NEIGHBOR_KERNEL,
                     mode='constant', cval=0)


def _detect_branchpoints(skeleton):
    """Branchpoints: skeleton pixels with >= 3 skeleton neighbors."""
    ncount = _neighbor_count(skeleton)
    return skeleton & (ncount >= 3)


def _detect_endpoints(skeleton):
    """Endpoints: skeleton pixels with exactly 1 skeleton neighbor."""
    ncount = _neighbor_count(skeleton)
    return skeleton & (ncount == 1)


def _get_skeleton_neighbors(r, c, skeleton, shape):
    """Return (row, col) pairs of 8-connected skeleton neighbors."""
    nbrs = []
    for dr, dc in _OFFSETS:
        nr, nc = r + dr, c + dc
        # bounds check
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            if skeleton[nr, nc]:
                nbrs.append((nr, nc))
    return nbrs


def _trace_leaf_to_junction(skeleton, branchpoints, start_r, start_c, Lspur):
    """
    Starting from an endpoint, walk inward along the skeleton until:
      - a branchpoint is reached  -> short spur candidate (return path)
      - no continuation exists    -> isolated segment end, keep it (return None)
      - length exceeds Lspur      -> too long, keep it (return None)
      - ambiguous branching       -> keep it (return None)

    Returns
    -------
    path : list of (row, col) or None
        The spur path (excluding the branchpoint) if it qualifies
        for deletion, otherwise None.
    """
    shape = skeleton.shape
    path = [(start_r, start_c)]
    prev_r, prev_c = -1, -1
    cur_r, cur_c = start_r, start_c

    for _ in range(Lspur + 1):
        # get skeleton neighbors of current pixel
        nbrs = _get_skeleton_neighbors(cur_r, cur_c, skeleton, shape)

        # remove the previous pixel (don't walk backward)
        if prev_r >= 0:
            nbrs = [(nr, nc) for nr, nc in nbrs
                    if not (nr == prev_r and nc == prev_c)]

        if len(nbrs) == 0:
            # dead end without reaching a junction:
            # isolated segment end, not a spur
            return None

        if len(nbrs) > 1:
            # ambiguous branching before the branchpoint map catches it
            return None

        next_r, next_c = nbrs[0]

        # check if we reached a branchpoint
        if branchpoints[next_r, next_c]:
            # reached first junction — is the path short enough?
            if len(path) <= Lspur:
                return path  # delete path up to but not including junction
            else:
                return None

        # extend the path
        path.append((next_r, next_c))

        # exceeded maximum spur length
        if len(path) > Lspur:
            return None

        prev_r, prev_c = cur_r, cur_c
        cur_r, cur_c = next_r, next_c

    return None


def prune_short_spurs(binary_image, Lspur=5):
    """
    Remove short dangling leaf branches (spurs) from a binary skeleton.

    Faithfully ports the MATLAB prune_short_spurs algorithm:
    detect endpoints, trace each to the nearest branchpoint, and
    delete the path if it is shorter than Lspur pixels.

    Parameters
    ----------
    binary_image : 2D bool ndarray
        Binary front image (will be skeletonized if not already 1-pixel wide).
    Lspur : int
        Maximum spur length in pixels (endpoint to first junction).
        Spurs with length <= Lspur are removed.

    Returns
    -------
    skeleton : 2D bool ndarray
        De-spurred skeleton image.
    """
    # ensure 1-pixel-wide skeleton
    skeleton = skeletonize(binary_image > 0)

    changed = True
    while changed:
        changed = False

        # detect branchpoints and endpoints via convolution
        branchpoints = _detect_branchpoints(skeleton)
        endpoints = _detect_endpoints(skeleton)

        # find all endpoint coordinates
        ep_rows, ep_cols = np.where(endpoints)
        to_delete = np.zeros_like(skeleton, dtype=bool)

        # trace from each endpoint to the nearest junction
        for r, c in zip(ep_rows, ep_cols):
            path = _trace_leaf_to_junction(
                skeleton, branchpoints, r, c, Lspur)
            if path is not None:
                # mark the spur path for deletion
                for pr, pc in path:
                    to_delete[pr, pc] = True

        # apply deletions
        if np.any(to_delete):
            skeleton[to_delete] = False
            changed = True

    return skeleton
