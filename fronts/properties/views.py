#TODO: separate this into multiple files: visualization & vis helpers

""" Methods to visualize properties of fronts """

import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import seaborn as sns

from wrangler.plotting import cutout

'''
from fronts.plotting.images import field_defs

def show_field(field:str, data:np.ndarray, ax:plt.Axes=None,
               grid_sep:float=10.): 

    if ax is None:
        ax = plt.gca()

    cmap = 'viridis'
    clbl = field

    if field in field_defs:
        cmap = field_defs[field]['cmap']
        clbl = field_defs[field]['label']
    else:
        raise IOError(f"Field {field} not in field_defs")

    # Center if needed
    vmin, vmax = None, None
    if ('vcenter' in field_defs[field]) and (field_defs[field]['vcenter'] is not None):
        vcenter = field_defs[field]['vcenter']
        vmax = np.nanmax(np.abs(data - vcenter))
        vmin = vcenter - vmax
        vmax = vcenter + vmax
    elif 'vmin' in field_defs[field]:
        vmin = field_defs[field]['vmin']

    cutout.show_image(data, clbl=clbl, cm=cmap, ax=ax, 
                        vmnx=(vmin, vmax),
                        cb_kws=dict(pad=0.01, fraction=0.04))

    #ax.set_title(fname)
    ax.xaxis.set_major_locator(MultipleLocator(grid_sep))
    ax.yaxis.set_major_locator(MultipleLocator(grid_sep))
    ax.grid()

    return ax

def show_fields(field_dict:dict, outfile:str, grid_sep:float=10.,
                title:str=None):

    # Number of fields
    nfields = len(field_dict)

    # Set figure size and gridspec from nfields
    #  One row for evrery 3 fields
    nrows = nfields // 3 + (nfields % 3 > 0)
    ncols = (nfields + 1) // nrows

    fig = plt.figure(figsize=(5*ncols, 4*nrows))
    plt.clf()
    gs = gridspec.GridSpec(nrows, ncols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Loop on the fields
    for i, (fname, fdata) in enumerate(field_dict.items()):
        row = i // ncols
        col = i % ncols

        ax = plt.subplot(gs[row, col])
        show_field(fname, fdata, ax=ax, grid_sep=grid_sep)

    # Title?
    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def plot_velocities(U, V, ax:plt.Axes=None):

    npix = U.shape[0]

    x = np.linspace(0, npix-1, npix)
    y = np.linspace(0, npix-1, npix)
    X, Y = np.meshgrid(x, y)

    # Calculate horizontal velocity magnitude
    velocity_magnitude = np.sqrt(U**2 + V**2)


    # Normalize U and V by their magnitude, then scale by magnitude
    # This makes arrow length proportional to velocity_magnitude
    U_normalized = np.where(velocity_magnitude > 0, U / velocity_magnitude, 0)
    V_normalized = np.where(velocity_magnitude > 0, V / velocity_magnitude, 0)

    # Scale factor to control overall arrow size (adjust as needed)
    arrow_scale = 5.0

    # Create the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Create quiver plot with arrows proportional to velocity magnitude
    quiver = ax.quiver(X, Y, 
                    U_normalized * velocity_magnitude * arrow_scale, 
                    V_normalized * velocity_magnitude * arrow_scale,
                    velocity_magnitude, 
                    cmap='jet',
                    scale=1,
                    scale_units='xy',
                    angles='xy',
                    width=0.003/2,
                    alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, label='Velocity Magnitude (m/s)',
                        pad=0.01, fraction=0.04)

    # Set labels and title
    ax.set_xlabel('X (grid points)', fontsize=12)
    ax.set_ylabel('Y (grid points)', fontsize=12)
    ax.set_title('Horizontal Velocity Field', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()

    # Optional: Print statistics
    print(f"Velocity magnitude range: {velocity_magnitude.min():.3f} to {velocity_magnitude.max():.3f}")
    print(f"Mean velocity magnitude: {velocity_magnitude.mean():.3f}")

    return ax
'''

# ---------------------------------------------------------------------------------------------
# Group Fronts - Visualization aids
# ---------------------------------------------------------------------------------------------

from typing import Dict, Tuple, Optional, Union, List
from fronts.properties.geometry import calculate_front_centroid, calculate_front_length, calculate_front_orientation, calculate_front_curvature
import warnings
from fronts.properties.group_labels import get_front_labels

def get_front_masks(
    labeled_fronts: np.ndarray,
    labels: Optional[Union[int, List[int]]] = None
) -> Dict[int, np.ndarray]:
    """
    Extract binary masks for individual fronts.
    This creates a mask over the entire region of the labeled array, therefore it is not optimal to use for global arrays
    and should be used only in cases with bounding boxes or small cutouts. 
    
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
) -> Dict[int, Dict[str, float]]:
    """
    Calculate all geometric properties for all fronts.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array from group_labels.label_fronts()
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
    from . import group_labels as label_module
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
            length = calculate_front_length(mask, lat_grid, lon_grid)
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
