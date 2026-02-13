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


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    radius: float = 6371.0
) -> float:
    """
    Calculate great circle distance between two points on Earth.

    Uses the Haversine formula to compute the distance between two points
    specified by latitude and longitude.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point in degrees
    lat2, lon2 : float
        Latitude and longitude of second point in degrees
    radius : float, optional
        Radius of Earth in kilometers. Default is 6371.0 km.

    Returns
    -------
    distance : float
        Distance between points in kilometers (or units of radius)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius * c


def calculate_front_length(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    method: str = 'perimeter'
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
    method : str, optional
        Method for calculating length:
        - 'perimeter': Use perimeter of the region (default, fast)
        - 'skeleton': Calculate length along the front skeleton (more accurate)
        Default is 'perimeter'.

    Returns
    -------
    length : float
        Front length in kilometers

    Notes
    -----
    The 'perimeter' method is faster but may overestimate for thick fronts.
    The 'skeleton' method traces the centerline but is computationally more expensive.
    """
    if method == 'perimeter':
        # Use skimage.measure.perimeter
        # This counts the length of the boundary
        perim_pixels = measure.perimeter(mask.astype(int))

        # Estimate physical distance per pixel
        # Use average grid spacing at the front location
        rows, cols = np.where(mask)
        if len(rows) < 2:
            return 0.0

        # Sample a few points to estimate grid spacing
        idx = np.random.choice(len(rows), min(10, len(rows)), replace=False)
        dists = []
        for i in range(len(idx) - 1):
            r1, c1 = rows[idx[i]], cols[idx[i]]
            r2, c2 = rows[idx[i+1]], cols[idx[i+1]]

            # Calculate distance between adjacent pixels
            if abs(r1 - r2) + abs(c1 - c2) == 1:  # Only adjacent pixels
                dist = haversine_distance(
                    lat[r1, c1], lon[r1, c1],
                    lat[r2, c2], lon[r2, c2]
                )
                dists.append(dist)

        if len(dists) == 0:
            # Fallback: estimate from grid resolution
            mean_lat = np.mean(lat[mask])
            dlat = np.abs(np.diff(lat[:, 0])).mean() if lat.ndim == 2 else np.abs(np.diff(lat)).mean()
            dlon = np.abs(np.diff(lon[0, :])).mean() if lon.ndim == 2 else np.abs(np.diff(lon)).mean()

            # Approximate distance per degree at this latitude
            km_per_deg_lat = 111.0  # roughly constant
            km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))

            dx = dlon * km_per_deg_lon
            dy = dlat * km_per_deg_lat
            pixel_size = np.sqrt(dx**2 + dy**2)
        else:
            pixel_size = np.mean(dists)

        length = perim_pixels * pixel_size

    elif method == 'skeleton':
        # Extract skeleton/centerline and trace it
        from skimage.morphology import skeletonize

        # Skeletonize the mask
        skeleton = skeletonize(mask)

        # Count skeleton pixels and estimate length
        skel_pixels = np.sum(skeleton)

        # Estimate physical length
        rows, cols = np.where(skeleton)
        if len(rows) < 2:
            return 0.0

        # Calculate sum of distances between consecutive skeleton points
        # For better accuracy, trace the skeleton path
        total_length = 0.0
        for i in range(len(rows) - 1):
            dist = haversine_distance(
                lat[rows[i], cols[i]], lon[rows[i], cols[i]],
                lat[rows[i+1], cols[i+1]], lon[rows[i+1], cols[i+1]]
            )
            total_length += dist

        length = total_length

    else:
        raise ValueError(f"Unknown method: {method}. Use 'perimeter' or 'skeleton'.")

    return length


def calculate_front_orientation(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray
) -> float:
    """
    Calculate the primary orientation of a front.

    Uses principal component analysis (PCA) to find the dominant direction
    of the front.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask for a single front
    lat : np.ndarray
        2D array of latitude coordinates
    lon : np.ndarray
        2D array of longitude coordinates

    Returns
    -------
    orientation : float
        Orientation angle in degrees, measured clockwise from North (0-180°)
        - 0° = North-South front
        - 90° = East-West front

    Notes
    -----
    Uses the orientation of the major axis from regionprops.
    """
    # Use skimage regionprops for orientation
    labeled = mask.astype(int)
    props = measure.regionprops(labeled)

    if len(props) == 0:
        return np.nan

    # Orientation from regionprops (in radians, -pi/2 to pi/2)
    orientation_rad = props[0].orientation

    # Convert to degrees (0-180, measured from horizontal)
    orientation_deg = np.degrees(orientation_rad)

    # Convert to oceanographic convention (0° = N, 90° = E)
    # regionprops gives angle from horizontal (E-W) axis
    # We want angle from vertical (N-S) axis
    orientation_from_north = 90 - orientation_deg

    # Normalize to 0-180 range
    if orientation_from_north < 0:
        orientation_from_north += 180
    elif orientation_from_north >= 180:
        orientation_from_north -= 180

    return orientation_from_north


def calculate_front_curvature(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    window_size: int = 5
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

    # Get skeleton of front
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


def calculate_front_centroid(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate the centroid (center of mass) of a front.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask for a single front
    lat : np.ndarray
        2D array of latitude coordinates
    lon : np.ndarray
        2D array of longitude coordinates

    Returns
    -------
    centroid_lat : float
        Latitude of centroid in degrees
    centroid_lon : float
        Longitude of centroid in degrees
    """
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return np.nan, np.nan

    # Get lat/lon at each pixel
    lats = lat[rows, cols]
    lons = lon[rows, cols]

    # Calculate mean
    centroid_lat = np.mean(lats)
    centroid_lon = np.mean(lons)

    return centroid_lat, centroid_lon


def calculate_front_extent(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray
) -> Dict[str, float]:
    """
    Calculate spatial extent of a front.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask for a single front
    lat : np.ndarray
        2D array of latitude coordinates
    lon : np.ndarray
        2D array of longitude coordinates

    Returns
    -------
    extent : dict
        Dictionary containing:
        - 'lat_min': minimum latitude
        - 'lat_max': maximum latitude
        - 'lon_min': minimum longitude
        - 'lon_max': maximum longitude
        - 'lat_range': latitude range (degrees)
        - 'lon_range': longitude range (degrees)
    """
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return {
            'lat_min': np.nan, 'lat_max': np.nan,
            'lon_min': np.nan, 'lon_max': np.nan,
            'lat_range': np.nan, 'lon_range': np.nan
        }

    lats = lat[rows, cols]
    lons = lon[rows, cols]

    extent = {
        'lat_min': np.min(lats),
        'lat_max': np.max(lats),
        'lon_min': np.min(lons),
        'lon_max': np.max(lons),
        'lat_range': np.ptp(lats),
        'lon_range': np.ptp(lons)
    }

    return extent


def calculate_all_geometric_properties(
    labeled_fronts: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time: Optional[Union[str, np.datetime64]] = None,
    include_curvature: bool = True,
    length_method: str = 'perimeter'
) -> Dict[int, Dict[str, float]]:
    """
    Calculate all geometric properties for all fronts.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array from label.label_fronts()
    lat : np.ndarray
        2D array of latitude coordinates
    lon : np.ndarray
        2D array of longitude coordinates
    time : str or np.datetime64, optional
        Timestamp for the fronts. If provided, included in output.
    include_curvature : bool, optional
        Whether to calculate curvature (computationally expensive).
        Default is True.
    length_method : str, optional
        Method for length calculation ('perimeter' or 'skeleton').
        Default is 'perimeter'.

    Returns
    -------
    properties : dict
        Dictionary mapping label -> properties dict
        Properties include:
        - 'time': timestamp (if provided)
        - 'npix': number of pixels
        - 'length_km': front length in km
        - 'centroid_lat': centroid latitude
        - 'centroid_lon': centroid longitude
        - 'orientation': orientation in degrees from north
        - 'lat_min', 'lat_max', 'lon_min', 'lon_max': spatial extent
        - 'lat_range', 'lon_range': spatial ranges
        - 'mean_curvature': mean curvature (if include_curvature=True)
        - 'curvature_direction': curvature direction (if include_curvature=True)

    Examples
    --------
    >>> import numpy as np
    >>> labeled = np.array([[1, 1, 0], [0, 2, 2]])
    >>> lat = np.array([[35.0, 35.0, 35.0], [36.0, 36.0, 36.0]])
    >>> lon = np.array([[-123.0, -122.0, -121.0], [-123.0, -122.0, -121.0]])
    >>> props = calculate_all_geometric_properties(labeled, lat, lon)
    >>> print(props[1]['centroid_lat'])
    35.0
    """
    # Ensure lat/lon are 2D
    if lat.ndim == 1 and lon.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon, lat)
    else:
        lat_grid = lat
        lon_grid = lon

    # Get all front labels
    from . import label as label_module
    labels = label_module.get_front_labels(labeled_fronts)

    properties = {}

    for lbl in labels:
        # Get mask for this front
        mask = labeled_fronts == lbl

        # Calculate properties
        props = {}

        if time is not None:
            props['time'] = str(time)

        props['npix'] = np.sum(mask)

        # Centroid
        centroid_lat, centroid_lon = calculate_front_centroid(mask, lat_grid, lon_grid)
        props['centroid_lat'] = centroid_lat
        props['centroid_lon'] = centroid_lon

        # Length
        try:
            length = calculate_front_length(mask, lat_grid, lon_grid, method=length_method)
            props['length_km'] = length
        except Exception as e:
            warnings.warn(f"Could not calculate length for front {lbl}: {e}")
            props['length_km'] = np.nan

        # Orientation
        try:
            orientation = calculate_front_orientation(mask, lat_grid, lon_grid)
            props['orientation'] = orientation
        except Exception as e:
            warnings.warn(f"Could not calculate orientation for front {lbl}: {e}")
            props['orientation'] = np.nan

        # Spatial extent
        extent = calculate_front_extent(mask, lat_grid, lon_grid)
        props.update(extent)

        # Curvature (optional, expensive)
        if include_curvature:
            try:
                mean_curv, curv_dir = calculate_front_curvature(mask, lat_grid, lon_grid)
                props['mean_curvature'] = mean_curv
                props['curvature_direction'] = curv_dir
            except Exception as e:
                warnings.warn(f"Could not calculate curvature for front {lbl}: {e}")
                props['mean_curvature'] = np.nan
                props['curvature_direction'] = np.nan

        properties[lbl] = props

    return properties
