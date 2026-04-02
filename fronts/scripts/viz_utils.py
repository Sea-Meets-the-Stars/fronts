"""Shared visualization utilities for front viewer scripts."""

import numpy as np
import pyqtgraph as pg


def make_colormap(divergent=False):
    """Return a pyqtgraph ColorMap.

    Parameters
    ----------
    divergent : bool
        If True, return a blue-white-red (seismic) colormap.
        If False, return an inverted grayscale (white->black) colormap.
    """
    if divergent:
        colors = np.array([
            [ 58,  76, 139],   # muted dark blue
            [103, 137, 196],   # muted blue
            [245, 245, 245],   # near-white
            [196, 117, 103],   # muted red
            [139,  58,  58],   # muted dark red
        ], dtype=np.ubyte)
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        colors = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.ubyte)
        pos = np.array([0.0, 1.0])
    return pg.ColorMap(pos=pos, color=colors)


def compute_levels(data, percentile, divergent=False):
    """Compute display (vmin, vmax) levels from valid data.

    Parameters
    ----------
    data : np.ndarray
        Field data (may contain NaNs).
    percentile : float
        Upper percentile (0-100) for the bright end; lower = 100-percentile.
    divergent : bool
        If True, make levels symmetric around zero.

    Returns
    -------
    tuple[float, float]
        (vmin, vmax)
    """
    vmin = np.nanpercentile(data, 100 - percentile)
    vmax = np.nanpercentile(data, percentile)
    if divergent:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
    return vmin, vmax


def make_fronts_rgba(fronts_data, divergent=False):
    """Build an RGBA uint8 array for the fronts overlay.

    Parameters
    ----------
    fronts_data : np.ndarray
        Binary front mask (>0 = front pixel).
    divergent : bool
        If True, use yellow; otherwise use red.

    Returns
    -------
    np.ndarray
        Shape (rows, cols, 4), dtype uint8.
    """
    rgba = np.zeros((*fronts_data.shape, 4), dtype=np.ubyte)
    if divergent:
        rgba[:, :, 0] = 255
        rgba[:, :, 1] = 255
        rgba[:, :, 2] = 0
        alpha = 200
    else:
        rgba[:, :, 0] = 255
        rgba[:, :, 1] = 0
        rgba[:, :, 2] = 0
        alpha = 120
    rgba[:, :, 3] = (fronts_data > 0).astype(np.ubyte) * alpha
    return rgba


def make_nan_rgba(field_data):
    """Build an RGBA uint8 array highlighting NaN pixels in dark green.

    Parameters
    ----------
    field_data : np.ndarray
        Field data array (used only to locate NaNs).

    Returns
    -------
    np.ndarray or None
        Shape (rows, cols, 4), dtype uint8, or None if no NaNs present.
    """
    nan_mask = np.isnan(field_data)
    if not np.any(nan_mask):
        return None
    rgba = np.zeros((*field_data.shape, 4), dtype=np.ubyte)
    rgba[:, :, 1] = 100  # dark green
    rgba[:, :, 3] = nan_mask.astype(np.ubyte) * 255
    return rgba
