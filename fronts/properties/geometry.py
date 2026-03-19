"""
Geometric Characterization Module

Calculate geometric properties of fronts including length, curvature,
orientation, and spatial extent.
"""

import numpy as np
from scipy import ndimage
from skimage import measure
from typing import Dict, Tuple, Optional, Union
import warnings
from sklearn.metrics.pairwise import haversine_distances


def haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    # haversine_distances expects radians
    p1 = np.radians([[lat1, lon1]])
    p2 = np.radians([[lat2, lon2]])
    return haversine_distances(p1, p2)[0, 0] * radius


def calculate_front_length(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    skeleton: np.ndarray = None
) -> float:
    """
    Calculate the length of a front.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask for a single front (True where front exists)
    lat : np.ndarray
        2D array of latitude coordinates
    lon : np.ndarray
        2D array of longitude coordinates
    skeleton : np.ndarray, optional
        Pre-computed skeleton (for method='skeleton' only). If provided, skips
        skeletonization step. Used to share skeleton between length and curvature
        calculations for efficiency. Default is None.

    Returns
    -------
    length : float
        Front length in kilometers

    Notes
    -----
    The 'skeleton' method (RECOMMENDED):
    - Traces the centerline of the front
    - Samples ~20 skeleton pixels and measures spacing to connected neighbors
    - Returns the "length" of the front along its central axis
    - For 1-pixel wide fronts, this is the most meaningful length measure

    """
 
    # Extract skeleton/centerline and trace it
    from skimage.morphology import skeletonize

    # Use pre-computed skeleton if provided, otherwise compute it
    if skeleton is None:
        skeleton = skeletonize(mask)

    # Count skeleton pixels
    skel_pixels = np.sum(skeleton)

    rows, cols = np.where(skeleton)
    if len(rows) < 2:
        return 0.0

    # Sample a few points to estimate average spacing, multiply by pixel count
    # This is more efficient than trying to sum distances between all skeleton pixels,
    # and is still robust as typically not covering large variations in grid spacing within a front.

    # Sample points to estimate average grid spacing in this region
    # Picks 20 pixels along the skeleton to measure spacing to neighbors, then averages
    sample_size = min(20, len(rows))
    # Use deterministic sampling (every Nth element)
    step = max(1, len(rows) // sample_size)
    idx = np.arange(0, len(rows), step)[:sample_size]

    # Calculate distances to 8-connected neighbors in skeleton
    # Gives distance for one step along the front, which we can multiply by pixel count for total length
    spacings = []
    for i in idx:
        r, c = rows[i], cols[i]
        # Check 8-connected neighbors
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < skeleton.shape[0] and
                0 <= nc < skeleton.shape[1] and
                skeleton[nr, nc]):
                # Found a connected skeleton neighbor
                dist = haversine_distance(
                    lat[r, c], lon[r, c],
                    lat[nr, nc], lon[nr, nc]
                )
                spacings.append(dist)
                break  # Only need one neighbor per sampled point

    if len(spacings) == 0:
        # Fallback: estimate from grid resolution at centroid
        mean_r, mean_c = int(np.mean(rows)), int(np.mean(cols))
        # Try to get spacing from adjacent grid cells
        if mean_r + 1 < lat.shape[0] and mean_c + 1 < lat.shape[1]:
            dx = haversine_distance(lat[mean_r, mean_c], lon[mean_r, mean_c],
                                    lat[mean_r, mean_c+1], lon[mean_r, mean_c+1])
            dy = haversine_distance(lat[mean_r, mean_c], lon[mean_r, mean_c],
                                    lat[mean_r+1, mean_c], lon[mean_r+1, mean_c])
            avg_spacing = np.mean([dx, dy])
        else:
            avg_spacing = 1.0  # Last resort fallback
    else:
        avg_spacing = np.mean(spacings)

    # Approximate total length as: number of skeleton pixels × average spacing
    length = skel_pixels * avg_spacing

    return length


def calculate_front_orientation(mask: np.ndarray) -> float:
    """
    Calculate the primary orientation of a front.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask for a single front.

    Returns
    -------
    orientation : float
        Acute angle from North (0–90°):
        - 0°  = N-S front (meridional)
        - 45° = diagonal (NE-SW or NW-SE)
        - 90° = E-W front (zonal)

    Notes
    -----
    regionprops.orientation is the angle between axis-0 (rows, N-S) and the
    major axis, in [-π/2, π/2]. An E-W front has its major axis perpendicular
    to rows → ~±90°; an N-S front → ~0°. Taking abs() gives the correct
    convention directly without further transformation.
    """
    props = measure.regionprops(mask.astype(int))
    if not props:
        return np.nan
    return float(abs(np.degrees(props[0].orientation)))


def calculate_front_curvature(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    window_size: int = 5,
    skeleton: np.ndarray = None
) -> Tuple[float, float]:
    """
    Calculate mean curvature and curvature direction of a front.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask for a single front
    lat : np.ndarray
        2D array of latitude coordinates
    lon : np.ndarray
        2D array of longitude coordinates
    window_size : int, optional
        Window size for local curvature calculation. Default is 5.
    skeleton : np.ndarray, optional
        Pre-computed skeleton. If provided, skips skeletonization step.
        Used to share skeleton with length calculation for efficiency.
        Default is None.

    Returns
    -------
    mean_curvature : float
        Mean absolute curvature (1/km)
    curvature_direction : float
        Direction of curvature:
        - Positive: clockwise/cyclonic curvature
        - Negative: counterclockwise/anticyclonic curvature

    Notes
    -----
    Curvature is estimated by fitting a circle to local segments of the front.
    This is a simplified implementation; more sophisticated methods may be needed
    for very complex fronts.
    """
    from skimage.morphology import skeletonize

    # Use pre-computed skeleton if provided, otherwise compute it
    if skeleton is None:
        skeleton = skeletonize(mask)

    rows, cols = np.where(skeleton)

    if len(rows) < 3:
        return np.nan, np.nan

    # Sort points to trace the front
    # Simple approach: sort by distance from first point
    # (More sophisticated path tracing could be implemented)
    points = np.column_stack([rows, cols])

    # Get lat/lon coordinates
    lats = lat[rows, cols]
    lons = lon[rows, cols]

    # Calculate local curvature at each point
    curvatures = []

    for i in range(window_size, len(points) - window_size):
        # Get local segment
        idx_before = i - window_size
        idx_after = i + window_size

        lat_before = lats[idx_before]
        lon_before = lons[idx_before]
        lat_center = lats[i]
        lon_center = lons[i]
        lat_after = lats[idx_after]
        lon_after = lons[idx_after]

        # Calculate vectors
        dx1 = haversine_distance(lat_center, lon_center, lat_center, lon_before)
        dy1 = haversine_distance(lat_center, lon_center, lat_before, lon_center)
        if lon_before > lon_center:
            dx1 = -dx1
        if lat_before > lat_center:
            dy1 = -dy1

        dx2 = haversine_distance(lat_center, lon_center, lat_center, lon_after)
        dy2 = haversine_distance(lat_center, lon_center, lat_after, lon_center)
        if lon_after < lon_center:
            dx2 = -dx2
        if lat_after < lat_center:
            dy2 = -dy2

        # Calculate angle change
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        angle_change = angle2 - angle1

        # Normalize to -pi to pi
        while angle_change > np.pi:
            angle_change -= 2 * np.pi
        while angle_change < -np.pi:
            angle_change += 2 * np.pi

        # Approximate curvature as angle change / arc length
        arc_length = haversine_distance(lat_before, lon_before, lat_after, lon_after)
        if arc_length > 0:
            curv = angle_change / arc_length
            curvatures.append(curv)

    if len(curvatures) == 0:
        return np.nan, np.nan

    curvatures = np.array(curvatures)

    # Mean absolute curvature
    mean_abs_curv = np.mean(np.abs(curvatures))

    # Mean signed curvature (indicates direction)
    mean_signed_curv = np.mean(curvatures)

    return mean_abs_curv, mean_signed_curv

def calculate_branch_points(
    mask: np.ndarray,
    skeleton: np.ndarray = None
) -> int:
    """
    Count the number of branch/junction points in a front.

    A branch point is a skeleton pixel with 3 or more skeleton neighbors,
    indicating where the front splits into multiple arms.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask for a single front
    skeleton : np.ndarray, optional
        Pre-computed skeleton. If None, will compute it.
        Default is None.

    Returns
    -------
    num_branches : int
        Number of branch/junction points

    """
    from skimage.morphology import skeletonize

    # Use pre-computed skeleton if provided, otherwise compute it
    if skeleton is None:
        skeleton = skeletonize(mask)

    # Fast method: Use convolution to count 8-connected neighbors
    # Kernel counts neighbors (excluding center pixel)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Count neighbors for each pixel
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')

    # Branch points are skeleton pixels with 3+ neighbors
    # (2 neighbors = line continuation, 3+ = junction)
    branch_points = (skeleton) & (neighbor_count >= 3)

    return int(np.sum(branch_points))


def process_single_front(
    label: int,
    name: str,
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time_str: str,
    y0: Optional[int] = None,
    y1: Optional[int] = None,
    x0: Optional[int] = None,
    x1: Optional[int] = None,
    skip_curvature: bool = False
) -> Optional[dict]:
    """
    Calculate geometric properties for a single front cutout.

    Operates on a cutout (bbox-extracted region) for efficiency. Intended
    for parallel batch processing of large domains — call this via a
    module-level multiprocessing wrapper. 

    Parameters
    ----------
    label : int
        Integer front label.
    name : str
        Unique front ID string (TIME_LAT_LON format from label.generate_front_ids).
    mask : np.ndarray
        Boolean mask, True where this front exists. Should be pre-extracted
        to the cutout region (same shape as lat and lon).
    lat : np.ndarray
        2D latitude array for the cutout region.
    lon : np.ndarray
        2D longitude array for the cutout region.
    time_str : str
        Timestamp string (ISO 8601 format).
    y0, y1, x0, x1 : int, optional
        Bounding box coordinates in the global array. Stored in output for
        traceability (e.g. reloading the cutout later); not used in any
        geometric calculation.
    skip_curvature : bool, optional
        Skip curvature calculation to save ~50%% time. Default False.

    Returns
    -------
    props : dict or None
        Geometric properties, or None if processing failed entirely.
    """
    try:
        from skimage.morphology import skeletonize
        
        props = {
            'label': label,
            'name': name,
            'time': time_str,
            'npix': int(np.sum(mask))
        }

        if all(v is not None for v in [y0, y1, x0, x1]):
            props.update({'y0': int(y0), 'y1': int(y1),
                          'x0': int(x0), 'x1': int(x1)})

        # Centroid and spatial extent — compute lat[mask] once, reuse for both
        front_lats = lat[mask]
        front_lons = lon[mask]
        props['centroid_lat'] = float(front_lats.mean())
        props['centroid_lon'] = float(front_lons.mean())
        props['lat_min'] = float(front_lats.min())
        props['lat_max'] = float(front_lats.max())
        props['lon_min'] = float(front_lons.min())
        props['lon_max'] = float(front_lons.max())

        # Pre-compute skeleton once — shared by length, branches, curvature
        skeleton = skeletonize(mask)

        props['length_km'] = float(calculate_front_length(mask, lat, lon, skeleton=skeleton))
        props['num_branches'] = int(calculate_branch_points(mask, skeleton=skeleton))
        props['orientation'] = float(calculate_front_orientation(mask))

        if not skip_curvature:
            curv, curv_dir = calculate_front_curvature(mask, lat, lon, skeleton=skeleton)
            props['mean_curvature'] = float(curv)
            props['curvature_direction'] = float(curv_dir)
        else:
            props['mean_curvature'] = np.nan
            props['curvature_direction'] = np.nan

        return props

    except Exception as e:
        print(f"  ERROR processing front {label} ({name}): {e}")
        return None
    
   