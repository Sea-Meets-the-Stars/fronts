"""
Sharpens front locations to lie on the gradb2 ridge by performing
priority-queue thinning that removes low-gradb2 boundary pixels first,
with a simple-point test to guarantee contiguity at every step.

See claude_sharpen_planning.tex, Section "Approach 4" for the full
algorithm description.

Usage:
    from sharpen_fronts import sharpen_fronts
    sharpened = sharpen_fronts(labeled_fronts, gradb2)
"""
import numpy as np
import heapq

from concurrent.futures import ProcessPoolExecutor

from scipy.ndimage import distance_transform_edt

from skimage.measure import regionprops
from skimage import morphology 
from skimage.measure import label as sklabel

# ---------------------------------------------------------------
# Simple-point lookup table (pre-computed once at module load)
# ---------------------------------------------------------------
def _build_simple_point_lut():
    """Build a 256-entry LUT for the simple-point test.

    For 8-connected foreground / 4-connected background (the standard
    (8,4)-adjacency convention), a foreground pixel p is simple iff:
      (a) removing p leaves exactly 1 foreground 8-connected component
          in the 3x3 neighborhood, AND
      (b) the background pixels in the 3x3 neighborhood (including the
          center, now treated as background) form exactly 1 four-connected
          component.

    The 8 neighbors of the center pixel are encoded as an 8-bit integer:
        bit 0 = NW, bit 1 = N, bit 2 = NE,
        bit 3 = W,            bit 4 = E,
        bit 5 = SW, bit 6 = S, bit 7 = SE
    """
    lut = np.zeros(256, dtype=bool)

    # Neighbor positions in a 3x3 grid (row, col), excluding center (1,1)
    # Ordered to match bit positions 0..7
    positions = [
        (0, 0), (0, 1), (0, 2),  # NW, N, NE
        (1, 0),         (1, 2),  # W, E
        (2, 0), (2, 1), (2, 2),  # SW, S, SE
    ]

    for n in range(256):
        # Reconstruct the 3x3 neighborhood (center = 0, since we're
        # testing whether removing the center pixel is safe)
        patch = np.zeros((3, 3), dtype=np.uint8)
        for bit, (r, c) in enumerate(positions):
            if n & (1 << bit):
                patch[r, c] = 1

        # (a) Foreground 8-connectivity: how many components among
        # the 8 neighbors (center already removed)?
        fg_cc = sklabel(patch, connectivity=2, return_num=True)[1]

        # (b) Background 4-connectivity: set center to background (0)
        # and count 4-connected background components in the 3x3 patch.
        # We include the center as background.
        bg_patch = 1 - patch.copy()
        bg_patch[1, 1] = 1  # center is background after removal
        bg_cc = sklabel(bg_patch, connectivity=1, return_num=True)[1]

        # Simple point: exactly 1 foreground component AND exactly 1
        # background component
        lut[n] = (fg_cc == 1) and (bg_cc == 1)

    return lut


# Pre-compute the LUT at module load time
SIMPLE_POINT_LUT = _build_simple_point_lut()

# Neighbor offsets for 8-connectivity (dr, dc)
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _encode_neighborhood(mask, r, c):
    """Encode the 8-neighborhood of (r,c) as an 8-bit integer.

    Args:
        mask : np.ndarray (bool, 2D)
            Binary mask of a single labeled front, padded by at least 1 pixel
            of background on all sides.
        r : int
            Row index of the pixel to encode.
        c : int
            Column index of the pixel to encode.

    Returns
    -------
    val : int
        The 8-bit integer encoding the neighborhood.

    Bit ordering matches the LUT convention:
        bit 0 = NW(-1,-1), bit 1 = N(-1,0), bit 2 = NE(-1,1),
        bit 3 = W(0,-1),   bit 4 = E(0,1),
        bit 5 = SW(1,-1),  bit 6 = S(1,0),  bit 7 = SE(1,1)
    """
    val = 0
    for bit, (dr, dc) in enumerate(NEIGHBORS_8):
        if mask[r + dr, c + dc]:
            val |= (1 << bit)
    return val


def _is_boundary(mask, r, c):
    """Check if foreground pixel (r,c) has at least one background 8-neighbor.

    Args:
        mask : np.ndarray (bool, 2D)
            Binary mask of a single labeled front, padded by at least 1 pixel
            of background on all sides.
        r : int
            Row index of the pixel to check.
        c : int
            Column index of the pixel to check.

    Returns
    -------
    is_boundary : bool
        True if the pixel (r,c) has at least one background 8-neighbor,
        False otherwise.
    """
    for dr, dc in NEIGHBORS_8:
        if not mask[r + dr, c + dc]:
            return True
    return False


def _count_fg_neighbors(mask, r, c):
    """Count the number of foreground 8-neighbors.

    Args:
        mask : np.ndarray (bool, 2D)
            Binary mask of a single labeled front, padded by at least 1 pixel
            of background on all sides.
        r : int
            Row index of the pixel to count neighbors for.
        c : int
            Column index of the pixel to count neighbors for.

    Returns
    -------
    count : int
        The number of foreground 8-neighbors of the pixel (r,c).
    """
    count = 0
    for dr, dc in NEIGHBORS_8:
        if mask[r + dr, c + dc]:
            count += 1
    return count


# ---------------------------------------------------------------
# Core: sharpen a single front
# ---------------------------------------------------------------
def sharpen_single_front(mask, gradb2, protect_endpoints=True):
    """Sharpen a single front via priority-queue weighted thinning.

    Parameters
    ----------
    mask : np.ndarray (bool, 2D)
        Binary mask of a single labeled front, padded by at least 1 pixel
        of background on all sides.
    gradb2 : np.ndarray (float, 2D)
        The gradb2 field, same shape as mask.
    protect_endpoints : bool
        If True, do not remove endpoint pixels (those with exactly 1
        foreground neighbor), preserving the extent of the front.

    Returns
    -------
    sharpened : np.ndarray (bool, 2D)
        The sharpened 1-pixel-wide front mask.
    """
    mask = mask.copy()
    nrows, ncols = mask.shape

    # Track which pixels are currently in the queue to avoid duplicates
    in_queue = np.zeros_like(mask, dtype=bool)

    # Initialize min-heap with all boundary pixels, keyed by gradb2 value
    # (lowest gradb2 removed first → skeleton tracks the ridge)
    heap = []
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            if mask[r, c] and _is_boundary(mask, r, c):
                heapq.heappush(heap, (gradb2[r, c], r, c))
                in_queue[r, c] = True

    # Iterative removal
    while heap:
        g_val, r, c = heapq.heappop(heap)
        in_queue[r, c] = False

        # Skip if already removed (stale entry)
        if not mask[r, c]:
            continue

        # Skip if this is an endpoint and we're protecting them
        if protect_endpoints and _count_fg_neighbors(mask, r, c) <= 1:
            continue

        # Simple-point test: only remove if topology is preserved
        code = _encode_neighborhood(mask, r, c)
        if not SIMPLE_POINT_LUT[code]:
            continue

        # Remove this pixel
        mask[r, c] = False

        # Check each neighbor: if it's still foreground and just became
        # a boundary pixel, add it to the queue
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            # Skip neighbors on the array edge (no valid 3x3 neighborhood)
            if nr < 1 or nr >= nrows - 1 or nc < 1 or nc >= ncols - 1:
                continue
            if mask[nr, nc] and not in_queue[nr, nc]:
                if _is_boundary(mask, nr, nc):
                    heapq.heappush(heap, (gradb2[nr, nc], nr, nc))
                    in_queue[nr, nc] = True

    return mask


# ---------------------------------------------------------------
# Wrapper: sharpen all labeled fronts
# ---------------------------------------------------------------
def dilate_labeled_fronts(labeled_fronts, radius:int=3):
    """Dilate each labeled front independently by a given pixel radius.

    Uses the distance transform to expand each front's binary mask by
    `radius` pixels, keeping labels separate (no merging of adjacent fronts).

    Parameters
    ----------
    labeled_fronts : np.ndarray (int, 2D)
        Labeled front array (0 = background, >0 = front label).
    radius : int, optional
        Dilation radius in pixels.

    Returns
    -------
    dilated : np.ndarray (int, 2D)
        Dilated labeled front array, same shape as input.
    """

    dilated = np.zeros_like(labeled_fronts)
    regions = regionprops(labeled_fronts)

    for region in regions:
        lbl = region.label
        min_row, min_col, max_row, max_col = region.bbox

        # Pad cutout by radius so dilation doesn't clip at edges
        r0 = max(min_row - radius - 1, 0)
        c0 = max(min_col - radius - 1, 0)
        r1 = min(max_row + radius + 1, labeled_fronts.shape[0])
        c1 = min(max_col + radius + 1, labeled_fronts.shape[1])

        # Binary mask for this front in the cutout
        cutout_mask = (labeled_fronts[r0:r1, c0:c1] == lbl)

        # Distance transform from background to foreground, then threshold
        dist = distance_transform_edt(~cutout_mask)
        dilated_mask = dist <= radius

        # Write back, but don't overwrite pixels already claimed by
        # another front (first-come-first-served)
        target = dilated[r0:r1, c0:c1]
        target[(dilated_mask) & (target == 0)] = lbl

    return dilated


def sharpen_fronts(labeled_fronts, gradb2, protect_endpoints=True,
                   min_size=7, dilate_radius=0, n_workers=1):
    """Sharpen all fronts in a labeled array via Approach 4.

    Parameters
    ----------
    labeled_fronts : np.ndarray (int, 2D)
        Labeled front array (0 = background, >0 = front label).
    gradb2 : np.ndarray (float, 2D)
        The gradb2 field, same shape as labeled_fronts.
    protect_endpoints : bool
        Protect endpoint pixels from removal.
    min_size : int
        Discard sharpened fronts with fewer than this many pixels.
    dilate_radius : int
        If > 0, dilate each front mask by this many pixels before
        sharpening, giving the algorithm room to shift to the ridge.
    n_workers : int
        Number of parallel workers. 1 = serial (useful for debugging).

    Returns
    -------
    sharpened_labeled : np.ndarray (int, 2D)
        Sharpened labeled front array, same shape as input.
    """

    # Optionally dilate fronts before sharpening
    if dilate_radius > 0:
        work_fronts = dilate_labeled_fronts(labeled_fronts, radius=dilate_radius)
    else:
        work_fronts = labeled_fronts

    result = np.zeros_like(labeled_fronts)
    regions = regionprops(work_fronts)

    if not regions:
        return result

    # Extract per-front cutout tasks
    tasks = []
    for region in regions:
        lbl = region.label
        min_row, min_col, max_row, max_col = region.bbox

        # Pad by 2 pixels on each side so that foreground pixels always
        # have a valid 3x3 neighborhood (even after dilation to bbox edge)
        pad = 2
        r0 = max(min_row - pad, 0)
        c0 = max(min_col - pad, 0)
        r1 = min(max_row + pad, work_fronts.shape[0])
        c1 = min(max_col + pad, work_fronts.shape[1])

        # Extract cutout mask and gradb2
        cutout_mask = (work_fronts[r0:r1, c0:c1] == lbl)
        cutout_gradb2 = gradb2[r0:r1, c0:c1]

        tasks.append((lbl, r0, c0, r1, c1, cutout_mask, cutout_gradb2))

    # Process fronts (serial or parallel)
    if n_workers <= 1:
        results = [
            _sharpen_task(t, protect_endpoints)
            for t in tasks
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_sharpen_task, t, protect_endpoints)
                for t in tasks
            ]
            results = [f.result() for f in futures]

    # Reassemble into the global array
    for (lbl, r0, c0, r1, c1, _, _), sharpened_mask in zip(tasks, results):
        if sharpened_mask.sum() >= min_size:
            result[r0:r1, c0:c1][sharpened_mask] = lbl

    # Final skeletonization to guarantee 1-pixel-wide fronts
    from skimage.morphology import skeletonize
    binary_sharp = result > 0
    skeleton = skeletonize(binary_sharp)
    # Zero out any pixel that didn't survive skeletonization
    result[~skeleton] = 0

    return result


def _sharpen_task(task, protect_endpoints):
    """Worker function for a single front sharpening task.

    Args:
        task : tuple
            A tuple containing the following elements:
            - lbl : int
                The label of the front.
            - r0 : int
                The starting row index of the cutout mask.
            - c0 : int
                The starting column index of the cutout mask.
            - r1 : int
                The ending row index of the cutout mask.
            - c1 : int
                The ending column index of the cutout mask.
            - cutout_mask : np.ndarray (bool, 2D)
                The cutout mask of the front.
            - cutout_gradb2 : np.ndarray (float, 2D)
                The gradb2 field, same shape as cutout_mask.
        protect_endpoints : bool
            If True, do not remove endpoint pixels (those with exactly 1
            foreground neighbor), preserving the extent of the front.

    Returns
    -------
    sharpened_mask : np.ndarray (bool, 2D)
        The sharpened 1-pixel-wide front mask.
    """
    lbl, r0, c0, r1, c1, cutout_mask, cutout_gradb2 = task
    return sharpen_single_front(cutout_mask, cutout_gradb2,
                                protect_endpoints=protect_endpoints)




# ---------------------------------------------------------------
# Core: global priority-queue thinning (Approach G1)
# ---------------------------------------------------------------
def global_sharpen_pq(binary_mask, gradb2, protect_endpoints=True):
    """Sharpen fronts globally via priority-queue weighted thinning.

    Operates on the full binary threshold image without per-front
    labeling.  Removes boundary pixels in ascending gradb2 order,
    preserving topology via the simple-point LUT at every step.

    Finishes with a standard thinning step.

    Parameters
    ----------
    binary_mask : np.ndarray (bool, 2D)
        Binary threshold output from front_thresh().  True = front
        candidate pixel.
    gradb2 : np.ndarray (float, 2D)
        The gradb2 field, same shape as binary_mask.
    protect_endpoints : bool
        If True, do not remove endpoint pixels (those with exactly 1
        foreground neighbor), preserving front extent.

    Returns
    -------
    sharpened : np.ndarray (bool, 2D)
        Sharpened 1-pixel-wide front mask.
    """
    # Work on a padded copy so boundary pixels always have valid 3x3
    # neighborhoods (the LUT reads all 8 neighbors)
    mask = np.pad(binary_mask.astype(bool), 1, mode='constant',
                  constant_values=False)
    gradb2_pad = np.pad(gradb2, 1, mode='constant', constant_values=0.0)

    # Track which pixels are queued to avoid duplicate insertions
    in_queue = np.zeros_like(mask, dtype=bool)

    # Initialize min-heap with all boundary pixels (lowest gradb2 first)
    heap = []
    rows_fg, cols_fg = np.where(mask)
    for r, c in zip(rows_fg, cols_fg):
        # Skip edge pixels of the padded array (row/col 0 or max)
        if r < 1 or r >= mask.shape[0] - 1 or c < 1 or c >= mask.shape[1] - 1:
            continue
        if _is_boundary(mask, r, c):
            heapq.heappush(heap, (gradb2_pad[r, c], r, c))
            in_queue[r, c] = True

    n_removed = 0

    # Iterative removal: peel away low-gradb2 boundary pixels
    while heap:
        _g_val, r, c = heapq.heappop(heap)
        in_queue[r, c] = False

        # Skip stale entries (pixel already removed)
        if not mask[r, c]:
            continue

        # Protect endpoints: pixels with exactly 1 foreground neighbor
        if protect_endpoints and _count_fg_neighbors(mask, r, c) <= 1:
            continue

        # Simple-point test: only remove if topology is preserved
        code = _encode_neighborhood(mask, r, c)
        if not SIMPLE_POINT_LUT[code]:
            continue

        # Remove this pixel
        mask[r, c] = False
        n_removed += 1

        # Add newly-exposed boundary neighbors to the queue
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if nr < 1 or nr >= mask.shape[0] - 1:
                continue
            if nc < 1 or nc >= mask.shape[1] - 1:
                continue
            if mask[nr, nc] and not in_queue[nr, nc]:
                if _is_boundary(mask, nr, nc):
                    heapq.heappush(heap, (gradb2_pad[nr, nc], nr, nc))
                    in_queue[nr, nc] = True

    # Remove padding
    sharpened = mask[1:-1, 1:-1]

    # Thin too!
    return morphology.thin(sharpened)

