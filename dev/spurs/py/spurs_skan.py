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

from skimage.morphology import skeletonize
from skan import Skeleton, summarize

# Main method moved to fronts/finding/despur.py

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
