"""
Spur removal from binary front skeletons (Algorithm S5).

Uses the skan library to analyze skeleton branch structure and
remove endpoint-to-junction branches shorter than Lspur.
Iterates until stable.

References
----------
- claude_spurs_plan.tex, Section S5
- skan docs: https://skeleton-analysis.org/stable/
"""

import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize


def prune_short_spurs(binary_image, Lspur=5):
    """
    Remove short dangling leaf branches (spurs) from a binary skeleton
    using skan skeleton analysis.

    Builds a Skeleton object, identifies endpoint-to-junction branches
    (branch-type == 1) with branch-distance <= Lspur, removes their
    pixels, and iterates until no more spurs are found.

    Parameters
    ----------
    binary_image : 2D bool ndarray
        Binary front image (will be skeletonized if not already 1-pixel wide).
    Lspur : int
        Maximum spur length in pixels (measured as branch-distance).
        Branches with distance <= Lspur are removed.

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

        # build skeleton graph
        sk = Skeleton(skeleton)
        df = summarize(sk, separator='-')

        # find endpoint-to-junction branches (type 1) shorter than Lspur
        spurs = df[(df['branch-type'] == 1) &
                   (df['branch-distance'] <= Lspur)]

        if len(spurs) == 0:
            break

        # collect all pixels belonging to spur branches
        to_delete = np.zeros_like(skeleton, dtype=bool)
        for idx in spurs.index:
            # path_coordinates returns Nx2 array of (row, col)
            coords = sk.path_coordinates(idx).astype(int)
            for r, c in coords:
                # don't delete junction pixels (they connect to other branches)
                # junction pixels are the last pixel in an ep-to-junc branch
                to_delete[r, c] = True

        # protect junction pixels: any pixel with >= 3 skeleton neighbors
        # in the original skeleton should not be removed
        from scipy.ndimage import correlate
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        ncount = correlate(skeleton.astype(np.uint8), kernel,
                           mode='constant', cval=0)
        junctions = skeleton & (ncount >= 3)
        to_delete &= ~junctions

        if np.any(to_delete):
            skeleton[to_delete] = False
            changed = True

    return skeleton


def analyze_branches(binary_image):
    """
    Return a summary DataFrame of all branches in the skeleton.

    Useful for inspection and debugging.

    Parameters
    ----------
    binary_image : 2D bool ndarray
        Binary front image.

    Returns
    -------
    df : pandas.DataFrame
        skan branch summary with columns including branch-type,
        branch-distance, skeleton-id, etc.
    skeleton : skan.Skeleton
        The Skeleton object for further analysis.
    """
    skel = skeletonize(binary_image > 0)
    sk = Skeleton(skel)
    df = summarize(sk, separator='-')
    return df, sk
