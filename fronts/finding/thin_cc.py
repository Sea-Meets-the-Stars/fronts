"""
Cornillon-style front thinning algorithm.

Converted from thin_subroutine.f to work with float SST values
in the range 272-305 K (typical ocean temperatures).

The algorithm thins wide front bands to single-pixel-wide lines by:
1. Scanning vertically (J-direction): For each column, find continuous
   front segments and select the pixel with maximum temperature gradient.
2. Scanning horizontally (I-direction): Same process for each row.

Original FORTRAN code by Peter Cornillon, University of Rhode Island.
"""

import numpy as np
from typing import Tuple, Optional


# Minimum valid SST in Kelvin (below this is land/ice/cloud)
MIN_VALID_SST_K = 271.35  # ~ -1.8°C, seawater freezing point

# Minimum temperature gradient to consider (K)
MIN_GRADIENT_K = 0.02


def thin(
    med_sst: np.ndarray,
    merged_fronts: np.ndarray,
    min_valid_sst: float = MIN_VALID_SST_K,
    min_gradient: float = MIN_GRADIENT_K,
    front_value: int = 4,
    min_segment_gap: int = 2,
) -> np.ndarray:
    """
    Thin wide front bands to single-pixel-wide lines.

    This algorithm processes detected fronts (typically multi-pixel wide)
    and reduces them to single-pixel width by selecting the pixel with
    the maximum cross-front temperature gradient within each continuous
    front segment.

    Parameters
    ----------
    med_sst : np.ndarray
        Median-filtered SST field in Kelvin. Shape (LenX, LenY).
        Values should be in range ~272-305 K for typical ocean.
        NaN values are treated as invalid.
    merged_fronts : np.ndarray
        Binary front mask where front_value indicates a front pixel.
        Shape must match med_sst.
    min_valid_sst : float, optional
        Minimum valid SST value in Kelvin. Pixels below this are
        considered invalid (land/ice/cloud). Default is 271.35 K.
    min_gradient : float, optional
        Minimum temperature gradient (K) to consider as a valid front.
        Default is 0.02 K.
    front_value : int, optional
        Value in merged_fronts that indicates a front pixel.
        Default is 4 (matching original FORTRAN convention).
    min_segment_gap : int, optional
        Minimum gap (in pixels) between front segments to consider
        them separate. Default is 2.

    Returns
    -------
    thinned_fronts : np.ndarray
        Thinned front mask with same shape as input. Front pixels
        have value front_value, non-front pixels are 0.

    Examples
    --------
    >>> import numpy as np
    >>> from fronts.finding.thin_cc import thin
    >>>
    >>> # Create sample SST field (K)
    >>> sst = np.full((100, 100), 290.0)
    >>> sst[40:60, :] = 295.0  # Warm band
    >>>
    >>> # Create wide front mask at the edge
    >>> fronts = np.zeros((100, 100), dtype=np.int16)
    >>> fronts[38:42, :] = 4  # Wide front band
    >>>
    >>> # Thin the fronts
    >>> thinned = thin(sst, fronts)
    >>> print(f"Original: {np.sum(fronts == 4)} pixels")
    >>> print(f"Thinned: {np.sum(thinned == 4)} pixels")

    Notes
    -----
    The algorithm performs two passes:

    1. **Vertical pass (J-direction)**: For each column (i), scan through
       rows (j) and identify continuous front segments. Within each segment,
       find the pixel with maximum |SST(j+1) - SST(j-1)| gradient and mark
       only that pixel in the output.

    2. **Horizontal pass (I-direction)**: Same process but scanning rows
       and computing |SST(i+1) - SST(i-1)| gradients.

    A pixel is included in the thinned output if it's selected by either
    pass, creating a union of the two thinning directions.

    The original FORTRAN code worked with integer SST scaled to 0-255.
    This version works directly with float Kelvin values.
    """
    # Validate inputs
    if med_sst.shape != merged_fronts.shape:
        raise ValueError(
            f"Shape mismatch: med_sst {med_sst.shape} vs "
            f"merged_fronts {merged_fronts.shape}"
        )

    len_x, len_y = med_sst.shape
    thinned_fronts = np.zeros_like(merged_fronts, dtype=np.int16)

    # Create valid SST mask (not NaN and above minimum)
    valid_mask = np.isfinite(med_sst) & (med_sst > min_valid_sst)

    # ================================================================
    # FIRST PASS: Thin fronts in the J-direction (vertical/columns)
    # ================================================================
    for i in range(len_x):
        max_gradient = min_gradient
        i_max, j_max = -1, -1
        gap_count = 0

        for j in range(1, len_y - 1):
            if merged_fronts[i, j] == front_value:
                # Reset gap counter when we hit a front pixel
                gap_count = 0

                # Calculate gradient if neighbors are valid
                if valid_mask[i, j - 1] and valid_mask[i, j + 1]:
                    gradient = abs(med_sst[i, j + 1] - med_sst[i, j - 1])

                    if gradient > max_gradient:
                        max_gradient = gradient
                        i_max = i
                        j_max = j
            else:
                # Not a front pixel - check if we should finalize segment
                if max_gradient > min_gradient:
                    if gap_count >= min_segment_gap:
                        # End of segment - mark the max gradient pixel
                        if i_max >= 0 and j_max >= 0:
                            thinned_fronts[i_max, j_max] = front_value
                        max_gradient = min_gradient
                        i_max, j_max = -1, -1
                    gap_count += 1

        # Handle segment at end of column
        if max_gradient > min_gradient and i_max >= 0 and j_max >= 0:
            thinned_fronts[i_max, j_max] = front_value

    # ================================================================
    # SECOND PASS: Thin fronts in the I-direction (horizontal/rows)
    # ================================================================
    for j in range(len_y):
        max_gradient = min_gradient
        i_max, j_max = -1, -1
        gap_count = 0

        for i in range(1, len_x - 1):
            if merged_fronts[i, j] == front_value:
                # Reset gap counter when we hit a front pixel
                gap_count = 0

                # Calculate gradient if neighbors are valid
                if valid_mask[i - 1, j] and valid_mask[i + 1, j]:
                    gradient = abs(med_sst[i + 1, j] - med_sst[i - 1, j])

                    if gradient > max_gradient:
                        max_gradient = gradient
                        i_max = i
                        j_max = j
            else:
                # Not a front pixel - check if we should finalize segment
                if max_gradient > min_gradient:
                    if gap_count >= min_segment_gap:
                        # End of segment - mark the max gradient pixel
                        if i_max >= 0 and j_max >= 0:
                            thinned_fronts[i_max, j_max] = front_value
                        max_gradient = min_gradient
                        i_max, j_max = -1, -1
                    gap_count += 1

        # Handle segment at end of row
        if max_gradient > min_gradient and i_max >= 0 and j_max >= 0:
            thinned_fronts[i_max, j_max] = front_value

    return thinned_fronts


def thin_fronts(
    sst: np.ndarray,
    fronts: np.ndarray,
    apply_median: bool = True,
    median_size: int = 5,
    **kwargs
) -> np.ndarray:
    """
    Convenience wrapper that optionally applies median filtering before thinning.

    Parameters
    ----------
    sst : np.ndarray
        SST field in Kelvin. Shape (LenX, LenY).
    fronts : np.ndarray
        Binary front mask (front pixels = 4, non-front = 0).
    apply_median : bool, optional
        If True, apply median filter to SST before thinning. Default True.
    median_size : int, optional
        Size of median filter window. Default 5.
    **kwargs
        Additional arguments passed to thin().

    Returns
    -------
    thinned_fronts : np.ndarray
        Thinned front mask.

    See Also
    --------
    thin : Core thinning algorithm
    """
    if apply_median:
        from scipy.ndimage import median_filter
        med_sst = median_filter(sst, size=median_size)
        # Preserve NaN locations
        med_sst = np.where(np.isnan(sst), np.nan, med_sst)
    else:
        med_sst = sst

    return thin(med_sst, fronts, **kwargs)


def compute_gradient_magnitude(
    sst: np.ndarray,
    min_valid_sst: float = MIN_VALID_SST_K
) -> np.ndarray:
    """
    Compute temperature gradient magnitude at each pixel.

    Uses central differences: sqrt((dT/dx)^2 + (dT/dy)^2)

    Parameters
    ----------
    sst : np.ndarray
        SST field in Kelvin.
    min_valid_sst : float, optional
        Minimum valid SST. Default 271.35 K.

    Returns
    -------
    gradient : np.ndarray
        Gradient magnitude in K/pixel. Invalid pixels are NaN.
    """
    len_x, len_y = sst.shape
    gradient = np.full_like(sst, np.nan, dtype=np.float32)

    valid_mask = np.isfinite(sst) & (sst > min_valid_sst)

    for i in range(1, len_x - 1):
        for j in range(1, len_y - 1):
            if (valid_mask[i, j] and
                valid_mask[i-1, j] and valid_mask[i+1, j] and
                valid_mask[i, j-1] and valid_mask[i, j+1]):

                dx = (sst[i+1, j] - sst[i-1, j]) / 2.0
                dy = (sst[i, j+1] - sst[i, j-1]) / 2.0
                gradient[i, j] = np.sqrt(dx**2 + dy**2)

    return gradient
