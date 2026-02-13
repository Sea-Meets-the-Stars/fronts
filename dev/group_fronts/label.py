"""
Front Labeling Module

Provides functions for grouping connected front pixels into individual fronts
and generating unique identifiers for each front.
"""

import numpy as np
from scipy import ndimage
from skimage import measure
from datetime import datetime
from typing import Tuple, Dict, List, Union, Optional
import warnings
import re


def label_fronts(
    front_binary: np.ndarray,
    connectivity: int = 2,
    return_num: bool = False
) -> Union[Tuple[np.ndarray, int], np.ndarray]:
    """
    Label connected components in a binary front field.

    Uses skimage.measure.label to identify connected regions in a binary
    front detection array. Each group of connected pixels receives a unique
    integer label.

    Parameters
    ----------
    front_binary : np.ndarray
        Binary array where True/1 indicates front pixels and False/0 indicates
        non-front pixels. Can be 2D (single timestep) or 3D (time, lat, lon).
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
        - 1: 4-connected (2D) or 6-connected (3D)
        - 2: 8-connected (2D) or 26-connected (3D)
        Default is 2 (8-connected).
    return_num : bool, optional
        If True, return both labeled array and number of labels.
        Default is False.

    Returns
    -------
    labeled_fronts : np.ndarray
        Integer array with same shape as input. Each connected front has a
        unique positive integer label. Background (non-front) pixels are 0.
    num_labels : int (only if return_num=True)
        Number of distinct fronts found.

    Examples
    --------
    >>> front_binary = np.array([[1, 1, 0, 0],
    ...                          [0, 1, 0, 1],
    ...                          [0, 0, 0, 1]])
    >>> labeled, num = label_fronts(front_binary, connectivity=2, return_num=True)
    >>> print(labeled)
    [[1 1 0 0]
     [0 1 0 2]
     [0 0 0 2]]
    >>> print(num)
    2

    Notes
    -----
    - Uses skimage.measure.label which is more efficient than scipy.ndimage.label
    - Connectivity=2 (8-connected) is recommended for oceanographic fronts to
      capture diagonal connections
    - For 3D arrays, connectivity determines temporal connectivity as well
    """
    # Convert boolean to integer if needed
    if front_binary.dtype == bool:
        front_binary = front_binary.astype(int)

    # Label connected components
    labeled_fronts = measure.label(
        front_binary,
        connectivity=connectivity,
        return_num=False
    )

    if return_num:
        num_labels = labeled_fronts.max()
        return labeled_fronts, num_labels

    return labeled_fronts


def get_front_labels(labeled_fronts: np.ndarray) -> np.ndarray:
    """
    Get sorted array of unique front labels (excluding background).

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array from label_fronts()

    Returns
    -------
    labels : np.ndarray
        1D array of unique front labels, sorted in ascending order,
        excluding 0 (background)
    """
    labels = np.unique(labeled_fronts)
    return labels[labels > 0]  # Exclude background (0)


def generate_front_ids(
    labeled_fronts: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time: Optional[Union[str, datetime, np.datetime64]] = None,
    filename: Optional[str] = None,
    id_format: str = 'TIME_LAT_LON'
) -> Dict[int, str]:
    """
    Generate unique string IDs for each front in TIME_LAT_LON format.

    Each front receives a unique identifier based on the timestamp and the
    location of its centroid. This allows fronts to be tracked and referenced
    across different analyses.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array from label_fronts(). Shape (lat, lon) for 2D
        or (time, lat, lon) for 3D.
    lat : np.ndarray
        Latitude coordinates. For 2D labeled_fronts, shape (lat, lon) or (lat,).
        For 3D, should be broadcastable to (time, lat, lon).
    lon : np.ndarray
        Longitude coordinates. For 2D labeled_fronts, shape (lat, lon) or (lon,).
        For 3D, should be broadcastable to (time, lat, lon).
    time : str, datetime, np.datetime64, or None, optional
        Timestamp for this front field. Format: ISO 8601 (YYYY-MM-DDTHH:MM:SS).
        If None, must provide filename. Default is None.
    filename : str or None, optional
        Filename to extract timestamp from. Expected format:
        'PREFIX_YYYY-MM-DDTHH_MM_SS_SUFFIX.ext' (e.g., 'LLC4320_2012-11-09T12_00_00_fronts.npy').
        Underscores in the time portion will be converted to colons.
        If provided, this takes precedence over the time parameter.
        Default is None.
    id_format : str, optional
        Format for ID generation. Currently only 'TIME_LAT_LON' is supported.
        Default is 'TIME_LAT_LON'.

    Returns
    -------
    front_ids : dict
        Dictionary mapping integer label -> string ID
        Example: {1: '20200101T000000_35.2N_123.4W', 2: '20200101T000000_36.5N_122.1W'}

    Examples
    --------
    >>> labeled = np.array([[1, 1, 0], [0, 2, 2]])
    >>> lat = np.array([[35.0, 35.0, 35.0], [36.0, 36.0, 36.0]])
    >>> lon = np.array([[-123.0, -122.0, -121.0], [-123.0, -122.0, -121.0]])

    # Using explicit time:
    >>> ids = generate_front_ids(labeled, lat, lon, time='2020-01-01T00:00:00')
    >>> print(ids[1])
    '20200101T000000_35.0N_122.5W'

    # Using filename:
    >>> ids = generate_front_ids(labeled, lat, lon, filename='LLC4320_2012-11-09T12_00_00_fronts.npy')
    >>> print(ids[1])
    '20121109T120000_35.0N_122.5W'

    Notes
    -----
    - Centroids are calculated as the mean lat/lon of all pixels in each front
    - Latitude: N for north, S for south
    - Longitude: E for east, W for west (using -180 to 180 convention)
    - Time format: YYYYMMDDTHHMMSS (compact ISO format without separators)
    - If both time and filename are provided, filename takes precedence
    """
    # Extract time from filename if provided
    if filename is not None:
        # Expected pattern: YYYY-MM-DDTHH_MM_SS
        # Example: LLC4320_2012-11-09T12_00_00_fronts.npy
        pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})'
        match = re.search(pattern, filename)

        if match:
            time_str_raw = match.group(1)
            # Convert underscores to colons: 2012-11-09T12_00_00 -> 2012-11-09T12:00:00
            time_str_iso = time_str_raw.replace('_', ':')
            time = np.datetime64(time_str_iso)

            if time is not None:
                warnings.warn(
                    f"Both filename and time provided. Using time from filename: {time_str_iso}",
                    UserWarning
                )
        else:
            raise ValueError(
                f"Could not extract timestamp from filename: {filename}. "
                f"Expected format: PREFIX_YYYY-MM-DDTHH_MM_SS_SUFFIX.ext"
            )

    # Ensure we have a time
    if time is None:
        raise ValueError(
            "Must provide either 'time' parameter or 'filename' parameter to extract time from"
        )

    # Parse time if it's a string or datetime
    if isinstance(time, str):
        time = np.datetime64(time)
    elif isinstance(time, datetime):
        time = np.datetime64(time)

    # Convert to string format: YYYYMMDDTHHMMSS
    time_str = str(time).replace('-', '').replace(':', '').replace(' ', 'T')
    if '.' in time_str:
        time_str = time_str.split('.')[0]  # Remove microseconds

    # Ensure lat/lon are 2D arrays matching labeled_fronts spatial dimensions
    if labeled_fronts.ndim == 3:
        # For 3D, use first time slice for now (single timestamp case)
        labeled_2d = labeled_fronts[0]
        warnings.warn(
            "3D labeled_fronts detected but single timestamp provided. "
            "Using first time slice for ID generation.",
            UserWarning
        )
    else:
        labeled_2d = labeled_fronts

    # Handle 1D coordinate arrays
    if lat.ndim == 1 and lon.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat_grid = lat
        lon_grid = lon
    else:
        raise ValueError(
            f"lat and lon must be 1D or 2D arrays. Got shapes: "
            f"lat={lat.shape}, lon={lon.shape}"
        )

    # Get unique front labels
    labels = get_front_labels(labeled_2d)

    # Generate IDs for each front
    front_ids = {}
    for label_val in labels:
        # Get mask for this front
        mask = labeled_2d == label_val

        # Calculate centroid
        lat_centroid = np.mean(lat_grid[mask])
        lon_centroid = np.mean(lon_grid[mask])

        # Indices of front pixels
        jj, ii = np.where(mask)

        # Candidate coordinates
        lat_cand = lat_grid[jj, ii]
        lon_cand = lon_grid[jj, ii]

        # Longitude difference with dateline-safe wrap to [-180, 180]
        dlon = (lon_cand - lon_centroid + 180.0) % 360.0 - 180.0
        dlat = lat_cand - lat_centroid

        # Distance proxy in "degrees", scaled by cos(lat) for lon shrinking with latitude
        w = np.cos(np.deg2rad(lat_centroid))
        dist2 = dlat**2 + (w * dlon)**2

        k = np.argmin(dist2)
        lat_on_front = lat_cand[k]
        lon_on_front = lon_cand[k]

        # Format lat/lon
        lat_dir = 'N' if lat_on_front >= 0 else 'S'
        lon_dir = 'E' if lon_on_front >= 0 else 'W'

        lat_str = f"{abs(lat_on_front):.1f}{lat_dir}"
        lon_str = f"{abs(lon_on_front):.1f}{lon_dir}"
        
        # Create ID
        front_id = f"{time_str}_{lat_str}_{lon_str}"
        front_ids[label_val] = front_id

    return front_ids


def get_front_masks(
    labeled_fronts: np.ndarray,
    labels: Optional[Union[int, List[int]]] = None
) -> Dict[int, np.ndarray]:
    """
    Extract binary masks for individual fronts.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array from label_fronts()
    labels : int, list of int, or None, optional
        Specific label(s) to extract. If None, extract all fronts.
        Default is None.

    Returns
    -------
    masks : dict
        Dictionary mapping label -> binary mask array
        Each mask is True where that specific front exists

    Examples
    --------
    >>> labeled = np.array([[1, 1, 0], [0, 2, 2]])
    >>> masks = get_front_masks(labeled)
    >>> print(masks[1])
    [[ True  True False]
     [False False False]]
    """
    if labels is None:
        labels = get_front_labels(labeled_fronts)
    elif isinstance(labels, int):
        labels = [labels]

    masks = {}
    for label_val in labels:
        masks[label_val] = labeled_fronts == label_val

    return masks


def get_front_properties_basic(
    labeled_fronts: np.ndarray
) -> Dict[int, Dict[str, Union[int, Tuple]]]:
    """
    Calculate basic properties for each front (pixel count, bounding box).

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array from label_fronts()

    Returns
    -------
    properties : dict
        Dictionary mapping label -> properties dict
        Properties include:
        - 'npix': number of pixels in front
        - 'bbox': bounding box (min_row, min_col, max_row, max_col) for 2D

    Examples
    --------
    >>> labeled = np.array([[1, 1, 0], [0, 2, 2]])
    >>> props = get_front_properties_basic(labeled)
    >>> print(props[1]['npix'])
    2
    >>> print(props[1]['bbox'])
    (0, 0, 0, 1)
    """
    # Get region properties using skimage
    regions = measure.regionprops(labeled_fronts)

    properties = {}
    for region in regions:
        label_val = region.label
        properties[label_val] = {
            'npix': region.area,
            'bbox': region.bbox,
            'centroid_indices': region.centroid
        }

    return properties


def filter_fronts_by_size(
    labeled_fronts: np.ndarray,
    min_size: int = 1,
    max_size: Optional[int] = None
) -> np.ndarray:
    """
    Remove fronts that are too small or too large.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array from label_fronts()
    min_size : int, optional
        Minimum number of pixels for a front to be kept. Default is 1.
    max_size : int or None, optional
        Maximum number of pixels for a front to be kept. If None, no upper limit.
        Default is None.

    Returns
    -------
    filtered_fronts : np.ndarray
        Labeled array with small/large fronts removed and labels re-numbered

    Examples
    --------
    >>> labeled = np.array([[1, 1, 0], [0, 2, 0], [3, 3, 3]])
    >>> # Remove fronts with < 3 pixels
    >>> filtered = filter_fronts_by_size(labeled, min_size=3)
    >>> print(filtered)
    [[0 0 0]
     [0 0 0]
     [1 1 1]]
    """
    # Get properties for all fronts
    props = get_front_properties_basic(labeled_fronts)

    # Filter based on size
    filtered = np.zeros_like(labeled_fronts)
    new_label = 1

    for label_val, prop in props.items():
        npix = prop['npix']

        # Check size constraints
        if npix < min_size:
            continue
        if max_size is not None and npix > max_size:
            continue

        # Keep this front with new label
        mask = labeled_fronts == label_val
        filtered[mask] = new_label
        new_label += 1

    return filtered
