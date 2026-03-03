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
    method: str = 'skeleton',
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
    method : str, optional
        Method for calculating length:
        - 'skeleton': Calculate length along the front skeleton using haversine distances (default, accurate)
        - 'perimeter': Use perimeter of the region (FAST but INACCURATE for non-uniform grids like LLC4320)
        Default is 'skeleton'.
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

    The 'perimeter' method:
    - Traces the boundary/perimeter of the front
    - Samples ~20 perimeter pixels and measures spacing to connected neighbors
    - Returns the total perimeter length (boundary length)
    - For 1-pixel wide fronts, perimeter ≈ 2× skeleton (both sides of the line)
    - Useful for area/width calculations or validation

    Both methods handle LLC4320's non-uniform grid by sampling local haversine distances.
    """
    if method == 'perimeter':
        # Use skimage.measure.perimeter to count perimeter length in pixels
        perim_pixels = measure.perimeter(mask.astype(int))

        # Extract actual perimeter pixels (boundary pixels)
        # Perimeter pixels are those in the mask that have at least one non-mask neighbor
        from scipy import ndimage
        eroded = ndimage.binary_erosion(mask)
        perimeter_mask = mask & ~eroded  # Pixels in mask but not in eroded version

        rows, cols = np.where(perimeter_mask)
        if len(rows) < 2:
            return 0.0

        # OPTIMIZED: Sample perimeter pixels and measure spacing to connected neighbors
        # Same approach as skeleton method, but for perimeter
        sample_size = min(20, len(rows))
        idx = np.random.choice(len(rows), sample_size, replace=False)
        # Use deterministic sampling (every Nth element) to avoid RNG lock in multiprocessing
        step = max(1, len(rows) // sample_size)
        idx = np.arange(0, len(rows), step)[:sample_size]

        # Calculate distances to 8-connected neighbors on perimeter
        spacings = []
        for i in idx:
            r, c = rows[i], cols[i]
            # Check 8-connected neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < perimeter_mask.shape[0] and
                    0 <= nc < perimeter_mask.shape[1] and
                    perimeter_mask[nr, nc]):
                    # Found a connected perimeter neighbor
                    dist = haversine_distance(
                        lat[r, c], lon[r, c],
                        lat[nr, nc], lon[nr, nc]
                    )
                    spacings.append(dist)
                    break  # Only need one neighbor per sampled point

        if len(spacings) == 0:
            # Fallback: estimate from grid resolution at centroid
            mean_r, mean_c = int(np.mean(rows)), int(np.mean(cols))
            if mean_r + 1 < lat.shape[0] and mean_c + 1 < lat.shape[1]:
                dx = haversine_distance(lat[mean_r, mean_c], lon[mean_r, mean_c],
                                       lat[mean_r, mean_c+1], lon[mean_r, mean_c+1])
                dy = haversine_distance(lat[mean_r, mean_c], lon[mean_r, mean_c],
                                       lat[mean_r+1, mean_c], lon[mean_r+1, mean_c])
                pixel_size = np.mean([dx, dy])
            else:
                pixel_size = 1.0  # Last resort fallback
        else:
            pixel_size = np.mean(spacings)

        # Approximate total length as: perimeter pixel count × average spacing
        length = perim_pixels * pixel_size

    elif method == 'skeleton':
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

        # Sample points to estimate average grid spacing in this region
        sample_size = min(20, len(rows))
        # Use deterministic sampling (every Nth element) to avoid RNG lock in multiprocessing
        step = max(1, len(rows) // sample_size)
        idx = np.arange(0, len(rows), step)[:sample_size]

        # Calculate distances to 8-connected neighbors in skeleton
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
        Orientation angle in degrees, measured as acute angle from North (0-90°)
        - 0° = North-South front (meridional)
        - 45° = Diagonal front (NE-SW or NW-SE)
        - 90° = East-West front (zonal)
        Note: Uses acute angle to eliminate ambiguity since fronts are lines, not vectors

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

    # Normalize to 0-180 range first
    if orientation_from_north < 0:
        orientation_from_north += 180
    elif orientation_from_north >= 180:
        orientation_from_north -= 180

    # Fold to 0-90 range (acute angle from North)
    # This removes ambiguity since fronts are lines, not vectors
    if orientation_from_north > 90:
        orientation_from_north = 180 - orientation_from_north

    return orientation_from_north


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

    Notes
    -----
    Branch points indicate structural complexity:
    - 0 branches: Simple line (no branching)
    - 1-2 branches: Y-shape or T-junction
    - 3+ branches: Complex multi-armed structure

    Examples
    --------
    Simple line: #### → 0 branch points
    Y-shape:  #     → 1 branch point (at the junction)
             # #
              #
    Complex:  # # # → Multiple branch points
               ###
              # # #
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

    NOTE: For batch processing many fronts, it's more efficient to:
    1. Extract bounding boxes using label.get_front_bboxes()
    2. Compute extent directly from bbox region:
       front_lats = lat[bbox][mask]
       lat_min, lat_max = front_lats.min(), front_lats.max()

    This function is useful for:
    - Interactive analysis of a single front
    - Cases where you don't have the full labeled array
    - Legacy code compatibility

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
    length_method: str = 'skeleton',
    skip_curvature: bool = False
) -> Optional[dict]:
    """
    Calculate geometric properties for a single front cutout.

    Operates on a cutout (bbox-extracted region) for efficiency. Intended
    for parallel batch processing of large domains — call this via a
    module-level multiprocessing wrapper. For interactive or serial use on
    small domains, prefer calculate_all_geometric_properties().

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
    length_method : str, optional
        'skeleton' (default, recommended for non-uniform grids like LLC4320)
        or 'perimeter' (deprecated).
    skip_curvature : bool, optional
        Skip curvature calculation to save ~50%% time. Default False.

    Returns
    -------
    props : dict or None
        Geometric properties, or None if processing failed entirely.
        Keys: label, name, time, npix, y0, y1, x0, x1,
              centroid_lat, centroid_lon, lat_min, lat_max, lon_min, lon_max,
              length_km, num_branches, orientation,
              mean_curvature, curvature_direction

    See Also
    --------
    calculate_all_geometric_properties : serial version for interactive/notebook use
    """
    try:
        props = {
            'label': label,
            'name': name,
            'time': time_str,
            'npix': int(np.sum(mask))
        }

        if all(v is not None for v in [y0, y1, x0, x1]):
            props.update({'y0': int(y0), 'y1': int(y1),
                          'x0': int(x0), 'x1': int(x1)})

        # Centroid
        try:
            clat, clon = calculate_front_centroid(mask, lat, lon)
            props['centroid_lat'] = float(clat)
            props['centroid_lon'] = float(clon)
        except Exception:
            props['centroid_lat'] = np.nan
            props['centroid_lon'] = np.nan

        # Spatial extent
        try:
            front_lats = lat[mask]
            front_lons = lon[mask]
            props['lat_min'] = float(front_lats.min())
            props['lat_max'] = float(front_lats.max())
            props['lon_min'] = float(front_lons.min())
            props['lon_max'] = float(front_lons.max())
        except Exception:
            props.update({'lat_min': np.nan, 'lat_max': np.nan,
                          'lon_min': np.nan, 'lon_max': np.nan})

        # Pre-compute skeleton once — shared by length, branches, curvature
        skeleton = None
        if length_method == 'skeleton':
            from skimage.morphology import skeletonize
            skeleton = skeletonize(mask)

        # Length
        try:
            props['length_km'] = float(
                calculate_front_length(mask, lat, lon,
                                       method=length_method, skeleton=skeleton)
            )
        except Exception:
            props['length_km'] = np.nan

        # Branch points
        try:
            props['num_branches'] = int(
                calculate_branch_points(mask, skeleton=skeleton)
            )
        except Exception:
            props['num_branches'] = 0

        # Orientation
        try:
            props['orientation'] = float(
                calculate_front_orientation(mask, lat, lon)
            )
        except Exception:
            props['orientation'] = np.nan

        # Curvature (optional)
        if not skip_curvature:
            try:
                curv, curv_dir = calculate_front_curvature(
                    mask, lat, lon, skeleton=skeleton
                )
                props['mean_curvature'] = float(curv)
                props['curvature_direction'] = float(curv_dir)
            except Exception:
                props['mean_curvature'] = np.nan
                props['curvature_direction'] = np.nan
        else:
            props['mean_curvature'] = np.nan
            props['curvature_direction'] = np.nan

        return props

    except Exception as e:
        print(f"  ERROR processing front {label} ({name}): {e}")
        return None
    
    
def calculate_all_geometric_properties(
    labeled_fronts: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time: Optional[Union[str, np.datetime64]] = None,
    include_curvature: bool = True,
    length_method: str = 'skeleton'
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
        Method for length calculation ('skeleton' or 'perimeter').
        Default is 'skeleton' (recommended for non-uniform grids).

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
