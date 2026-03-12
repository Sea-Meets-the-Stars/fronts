"""Inpainting methods for ocean model edge artifacts.

Provides methods to fill bad pixels (e.g., -999 sentinel values
at LLC4320 face edges) using biharmonic interpolation and
RegularGridInterpolator-based linear interpolation.  A high-level
``inpaint`` function supports a two-stage workflow: first fill
sentinel-value pixels, then optionally fill residual low-magnitude
pixels with a second pass.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from skimage.restoration import inpaint as sk_inpaint


def inpaint_biharmonic(data, mask=None, bad_value=-999.):
    """Inpaint bad pixels using biharmonic interpolation.

    Uses scikit-image's biharmonic inpainting to smoothly fill
    masked pixels.  NaN pixels are temporarily replaced with the
    median for interpolation and restored to NaN afterward.

    Parameters
    ----------
    data : np.ndarray
        2-D array of field values.
    mask : np.ndarray of bool, optional
        Boolean mask where ``True`` marks pixels to inpaint.
        If *None*, pixels equal to *bad_value* are masked.
    bad_value : float, optional
        Sentinel value used to identify bad pixels when *mask*
        is not provided.  Default is -999.

    Returns
    -------
    np.ndarray
        Copy of *data* with masked pixels inpainted.
        Original NaN pixels remain NaN.
    """
    if mask is None:
        mask = data == bad_value

    tmp = data.copy()

    # Temporarily fill NaN so biharmonic interpolation can proceed
    isnan = np.isnan(tmp)
    tmp[isnan] = np.median(tmp[~isnan])

    inpainted = sk_inpaint.inpaint_biharmonic(
        tmp, np.uint8(mask), channel_axis=None)

    # Restore NaN pixels
    inpainted[isnan] = np.nan

    return inpainted


def inpaint_regular(data, mask=None, bad_value=-999.):
    """Inpaint bad pixels using RegularGridInterpolator.

    Fills bad pixels by nearest-neighbor first to provide a
    complete grid, then applies linear interpolation to smooth
    the filled values using surrounding valid data.  Works best
    for thin bands of bad data (e.g., face-edge artifacts) where
    bad pixels border valid ones.

    Parameters
    ----------
    data : np.ndarray
        2-D array of field values.
    mask : np.ndarray of bool, optional
        Boolean mask where ``True`` marks pixels to inpaint.
        If *None*, pixels equal to *bad_value* are masked.
    bad_value : float, optional
        Sentinel value used to identify bad pixels when *mask*
        is not provided.  Default is -999.

    Returns
    -------
    np.ndarray
        Copy of *data* with masked pixels inpainted.
    """
    if mask is None:
        mask = data == bad_value

    result = data.copy()

    # Fill bad pixels with nearest valid neighbor so the
    # interpolator has a complete grid to work with
    full_mask = mask | np.isnan(result)
    nn_indices = distance_transform_edt(
        full_mask, return_distances=False, return_indices=True)
    filled = result.copy()
    filled[full_mask] = result[tuple(nn_indices)][full_mask]

    # Build the interpolator on the filled grid
    rows = np.arange(result.shape[0], dtype=float)
    cols = np.arange(result.shape[1], dtype=float)
    interp = RegularGridInterpolator(
        (rows, cols), filled, method='linear',
        bounds_error=False, fill_value=None)

    # Evaluate at bad pixel locations
    bad_rows, bad_cols = np.where(mask)
    if bad_rows.size > 0:
        pts = np.column_stack([bad_rows.astype(float),
                               bad_cols.astype(float)])
        result[mask] = interp(pts)

    return result


def inpaint(data, bad_value=-999., method='biharmonic',
            second_pass=None, second_threshold=1e-20):
    """Inpaint bad pixels with an optional two-stage workflow.

    Stage 1 fills pixels matching *bad_value*.  An optional stage 2
    identifies residual low-magnitude (but finite) pixels below
    *second_threshold* and fills them with a second method.  The default
    method uses biharmonic inpainting for sentinel values and 
    a second pass cleans up remaining artifacts.

    Parameters
    ----------
    data : np.ndarray
        2-D array of field values.
    bad_value : float, optional
        Sentinel value for the first-pass mask.  Default is -999.
    method : str, optional
        Inpainting method for the first pass.  One of
        ``'biharmonic'`` or ``'regular'``.  Default is
        ``'biharmonic'``.
    second_pass : str or None, optional
        Inpainting method for the second pass.  One of
        ``'biharmonic'``, ``'regular'``, or *None* to skip.
        Default is *None* (single-stage only).
    second_threshold : float, optional
        Pixels with values below this threshold (and finite) are
        masked for the second pass.  Default is 1e-20.

    Returns
    -------
    np.ndarray
        Copy of *data* with bad pixels inpainted.

    Examples
    --------
    Single-pass biharmonic::

        result = inpaint(gradb2)

    Two-stage: biharmonic then RegularGridInterpolator::

        result = inpaint(gradb2, method='biharmonic',
                         second_pass='regular',
                         second_threshold=1e-20)
    """
    methods = {
        'biharmonic': inpaint_biharmonic,
        'regular': inpaint_regular,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from {list(methods.keys())}.")

    # Stage 1: fill sentinel-value pixels
    result = methods[method](data, bad_value=bad_value)

    # Stage 2 (optional): fill residual low-magnitude pixels
    if second_pass is not None:
        if second_pass not in methods:
            raise ValueError(
                f"Unknown second_pass '{second_pass}'. "
                f"Choose from {list(methods.keys())}.")
        mask2 = (result < second_threshold) & np.isfinite(result)
        if np.any(mask2):
            result = methods[second_pass](result, mask=mask2)

    return result
