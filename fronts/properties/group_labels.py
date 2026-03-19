"""
Front Labeling Module

Provides functions for grouping connected front pixels into individual fronts
and generating unique identifiers for each front.
"""

import re
import numpy as np
from skimage import measure
from typing import Tuple, Dict, Union, Optional


def label_fronts(
    front_binary: np.ndarray,
    connectivity: int = 2,
    return_num: bool = False
) -> Union[Tuple[np.ndarray, int], np.ndarray]:
    """
    Label connected components in a binary front field.

    Parameters
    ----------
    front_binary : np.ndarray
        Binary 2D array; True/1 = front pixel, False/0 = background.
    connectivity : int, optional
        1 = 4-connected, 2 = 8-connected (default).
    return_num : bool, optional
        If True, also return the number of labels.

    Returns
    -------
    labeled_fronts : np.ndarray
        Integer array; each connected front has a unique integer label.
        Background (non-front) pixels are 0.
    num_labels : int (only if return_num=True)
    """
    
    if front_binary.dtype == bool:
        front_binary = front_binary.astype(int)

    labeled_fronts = measure.label(front_binary, connectivity=connectivity,
                                   return_num=False)
    if return_num:
        return labeled_fronts, labeled_fronts.max()

    return labeled_fronts


def get_front_labels(labeled_fronts: np.ndarray) -> np.ndarray:
    """Return sorted 1-D array of unique front labels (excluding background 0)."""
    labels = np.unique(labeled_fronts)
    return labels[labels > 0]


def get_front_properties(
    labeled_fronts: np.ndarray
) -> Dict[int, Dict[str, Union[int, Tuple]]]:
    """
    Calculate basic properties for each front via skimage regionprops.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        2D labeled array from label_fronts().

    Returns
    -------
    properties : dict
        Maps label -> {'npix', 'bbox', 'centroid_indices'}, where
        'bbox' is (min_row, min_col, max_row, max_col) and
        'centroid_indices' is a (row, col) float tuple.
    """
    properties = {}
    for region in measure.regionprops(labeled_fronts):
        properties[region.label] = {
            'npix':             region.area,
            'bbox':             region.bbox,        # (min_row, min_col, max_row, max_col)
            'centroid_indices': region.centroid,    # (row_float, col_float)
        }
    return properties


def generate_front_ids(
    lat: np.ndarray,
    lon: np.ndarray,
    filename: str,
    properties: Dict[int, Dict] = None,
    labeled_fronts: np.ndarray = None,
) -> Dict[int, str]:
    """
    Generate unique string IDs for each front in TIME_LAT_LON format.

    Time is always extracted from the filename. Centroids are taken from
    a properties dict (from get_front_properties()) if provided; otherwise
    they are computed internally from labeled_fronts.

    Parameters
    ----------
    lat, lon : np.ndarray
        2D latitude / longitude grids, shape (H, W).
    filename : str
        Filename containing the timestamp in the pattern
        YYYY-MM-DDTHH_MM_SS (e.g. 'LLC4320_2012-11-09T12_00_00_bin_A.npy').
    properties : dict, optional
        Output of get_front_properties(). If provided, centroids are taken
        from here (avoids recomputing). One of properties or labeled_fronts
        must be supplied.
    labeled_fronts : np.ndarray, optional
        2D labeled array used to compute properties when properties=None.

    Returns
    -------
    front_ids : dict
        Maps integer label -> string ID, e.g. {1: '20121109T120000_35.2N_123.4W'}.
    """
    # --- Extract time from filename ---
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', filename)
    if not match:
        raise ValueError(
            f"Could not extract timestamp from filename: {filename}. "
            "Expected pattern: YYYY-MM-DDTHH_MM_SS"
        )
    time_str = (match.group(1)
                .replace('-', '').replace('_', '').replace(':', ''))
    # e.g. '2012-11-09T12_00_00' -> '20121109T120000'
    time_str = time_str[:8] + 'T' + time_str[8:]

    # --- Get per-front centroids ---
    if properties is None:
        if labeled_fronts is None:
            raise ValueError("Provide either 'properties' or 'labeled_fronts'.")
        properties = get_front_properties(labeled_fronts)

    # --- Build IDs ---
    front_ids = {}
    for label, props in properties.items():
        row, col = props['centroid_indices']
        lat_c = lat[round(row), round(col)]
        lon_c = lon[round(row), round(col)]

        lat_str = f"{abs(lat_c):.1f}{'N' if lat_c >= 0 else 'S'}"
        lon_str = f"{abs(lon_c):.1f}{'E' if lon_c >= 0 else 'W'}"

        front_ids[label] = f"{time_str}_{lat_str}_{lon_str}"

    return front_ids
