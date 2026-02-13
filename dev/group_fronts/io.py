"""
Input/Output Module

Functions for saving and loading front data in various formats (NetCDF, CSV, Parquet).
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Union
import warnings


def to_netcdf(
    labeled_fronts: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time: Union[str, np.datetime64],
    output_path: Union[str, Path],
    front_ids: Optional[Dict[int, str]] = None,
    attrs: Optional[Dict] = None
) -> None:
    """
    Save labeled fronts to NetCDF file.

    Creates an xarray Dataset with the labeled front field and coordinates,
    then saves to NetCDF format.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array. Shape (lat, lon) or (time, lat, lon).
    lat : np.ndarray
        Latitude coordinates. 1D or 2D array.
    lon : np.ndarray
        Longitude coordinates. 1D or 2D array.
    time : str or np.datetime64
        Timestamp(s) for the data.
    output_path : str or Path
        Output file path for NetCDF file.
    front_ids : dict, optional
        Dictionary mapping integer labels to string IDs.
        If provided, saved as a lookup table in the file.
    attrs : dict, optional
        Additional global attributes to include in NetCDF file.

    Examples
    --------
    >>> import numpy as np
    >>> labeled = np.array([[1, 1, 0], [0, 2, 2]])
    >>> lat = np.array([35.0, 36.0])
    >>> lon = np.array([-123.0, -122.0, -121.0])
    >>> to_netcdf(labeled, lat, lon, '2020-01-01', 'fronts_labeled.nc')
    """
    # Convert time to datetime64
    if isinstance(time, str):
        time = np.datetime64(time)

    # Create coordinates
    if lat.ndim == 1 and lon.ndim == 1:
        coords = {
            'lat': ('lat', lat),
            'lon': ('lon', lon)
        }
        dims = ('lat', 'lon')
    elif lat.ndim == 2 and lon.ndim == 2:
        # For 2D coordinate arrays
        coords = {
            'lat': (['y', 'x'], lat),
            'lon': (['y', 'x'], lon)
        }
        dims = ('y', 'x')
    else:
        raise ValueError("lat and lon must both be 1D or both be 2D")

    # Create Dataset
    ds = xr.Dataset(
        data_vars={
            'front_labels': (dims, labeled_fronts, {
                'long_name': 'Front labels',
                'description': 'Integer labels for grouped fronts. 0 = background, >0 = front ID',
                'units': 'dimensionless'
            })
        },
        coords=coords,
        attrs={
            'title': 'Labeled Ocean Fronts',
            'institution': 'UC Santa Cruz',
            'source': 'Front detection and grouping',
            'time': str(time),
            'Conventions': 'CF-1.8'
        }
    )

    # Add front ID lookup table if provided
    if front_ids is not None:
        # Create arrays for lookup table
        labels_arr = np.array(list(front_ids.keys()))
        ids_arr = np.array(list(front_ids.values()), dtype='U64')

        ds['front_id_labels'] = ('front', labels_arr)
        ds['front_id_strings'] = ('front', ids_arr, {
            'long_name': 'Front unique identifiers',
            'description': 'String IDs in TIME_LAT_LON format'
        })

    # Add custom attributes
    if attrs is not None:
        ds.attrs.update(attrs)

    # Save to NetCDF
    ds.to_netcdf(output_path, mode='w')
    print(f"Saved labeled fronts to {output_path}")


def from_netcdf(
    input_path: Union[str, Path]
) -> tuple:
    """
    Load labeled fronts from NetCDF file.

    Parameters
    ----------
    input_path : str or Path
        Path to NetCDF file created by to_netcdf()

    Returns
    -------
    labeled_fronts : np.ndarray
        Labeled front array
    lat : np.ndarray
        Latitude coordinates
    lon : np.ndarray
        Longitude coordinates
    time : str
        Timestamp string
    front_ids : dict or None
        Dictionary mapping labels to IDs (if available)
    """
    ds = xr.open_dataset(input_path)

    labeled_fronts = ds['front_labels'].values
    lat = ds['lat'].values
    lon = ds['lon'].values
    time = ds.attrs.get('time', None)

    # Load front IDs if available
    front_ids = None
    if 'front_id_labels' in ds and 'front_id_strings' in ds:
        labels = ds['front_id_labels'].values
        ids = ds['front_id_strings'].values
        front_ids = dict(zip(labels, ids))

    ds.close()

    return labeled_fronts, lat, lon, time, front_ids


def properties_to_dataframe(
    properties: Dict[int, Dict],
    front_ids: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Convert front properties dictionary to pandas DataFrame.

    Parameters
    ----------
    properties : dict
        Dictionary mapping label -> properties dict
        (e.g., from geometry.calculate_all_geometric_properties)
    front_ids : dict, optional
        Dictionary mapping label -> front ID string

    Returns
    -------
    df : pd.DataFrame
        DataFrame with one row per front, columns for each property

    Examples
    --------
    >>> props = {1: {'npix': 100, 'length_km': 50.3, 'centroid_lat': 35.0}}
    >>> ids = {1: '20200101T000000_35.0N_123.0W'}
    >>> df = properties_to_dataframe(props, ids)
    >>> print(df.columns)
    Index(['front_id', 'npix', 'length_km', 'centroid_lat'], dtype='object')
    """
    # Convert properties to list of dicts
    rows = []
    for label, props in properties.items():
        row = {'label': label}

        # Add front ID if available
        if front_ids is not None and label in front_ids:
            row['front_id'] = front_ids[label]

        row.update(props)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns to put label and front_id first
    cols = ['label']
    if 'front_id' in df.columns:
        cols.append('front_id')
    cols.extend([c for c in df.columns if c not in cols])
    df = df[cols]

    return df


def to_csv(
    properties: Dict[int, Dict],
    output_path: Union[str, Path],
    front_ids: Optional[Dict[int, str]] = None
) -> None:
    """
    Save front properties to CSV file.

    Parameters
    ----------
    properties : dict
        Dictionary mapping label -> properties dict
    output_path : str or Path
        Output file path for CSV file
    front_ids : dict, optional
        Dictionary mapping label -> front ID string

    Examples
    --------
    >>> props = {1: {'npix': 100, 'length_km': 50.3}}
    >>> to_csv(props, 'front_properties.csv')
    """
    df = properties_to_dataframe(properties, front_ids)
    df.to_csv(output_path, index=False)
    print(f"Saved front properties to {output_path}")


def to_parquet(
    properties: Dict[int, Dict],
    output_path: Union[str, Path],
    front_ids: Optional[Dict[int, str]] = None
) -> None:
    """
    Save front properties to Parquet file.

    Parquet is a columnar storage format that is efficient for large datasets
    and integrates well with big data tools.

    Parameters
    ----------
    properties : dict
        Dictionary mapping label -> properties dict
    output_path : str or Path
        Output file path for Parquet file
    front_ids : dict, optional
        Dictionary mapping label -> front ID string

    Examples
    --------
    >>> props = {1: {'npix': 100, 'length_km': 50.3}}
    >>> to_parquet(props, 'front_properties.parquet')
    """
    df = properties_to_dataframe(properties, front_ids)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Saved front properties to {output_path}")


def from_csv(
    input_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load front properties from CSV file.

    Parameters
    ----------
    input_path : str or Path
        Path to CSV file

    Returns
    -------
    df : pd.DataFrame
        DataFrame with front properties
    """
    return pd.read_csv(input_path)


def from_parquet(
    input_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load front properties from Parquet file.

    Parameters
    ----------
    input_path : str or Path
        Path to Parquet file

    Returns
    -------
    df : pd.DataFrame
        DataFrame with front properties
    """
    return pd.read_parquet(input_path, engine='pyarrow')


def save_all(
    labeled_fronts: np.ndarray,
    properties: Dict[int, Dict],
    lat: np.ndarray,
    lon: np.ndarray,
    time: Union[str, np.datetime64],
    front_ids: Dict[int, str],
    output_dir: Union[str, Path],
    base_name: str = 'fronts',
    formats: list = ['netcdf', 'csv', 'parquet']
) -> Dict[str, Path]:
    """
    Save front data in all requested formats.

    Convenience function to save labeled fronts and properties in multiple
    formats with consistent naming.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Labeled front array
    properties : dict
        Front properties dictionary
    lat : np.ndarray
        Latitude coordinates
    lon : np.ndarray
        Longitude coordinates
    time : str or np.datetime64
        Timestamp
    front_ids : dict
        Front ID dictionary
    output_dir : str or Path
        Output directory
    base_name : str, optional
        Base name for output files. Default is 'fronts'.
    formats : list, optional
        List of formats to save. Options: 'netcdf', 'csv', 'parquet'.
        Default is all three.

    Returns
    -------
    output_files : dict
        Dictionary mapping format -> output file path

    Examples
    --------
    >>> output_files = save_all(
    ...     labeled, props, lat, lon, time, ids,
    ...     output_dir='output/',
    ...     base_name='fronts_2020_01_01'
    ... )
    >>> print(output_files['netcdf'])
    output/fronts_2020_01_01.nc
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    if 'netcdf' in formats:
        nc_path = output_dir / f"{base_name}.nc"
        to_netcdf(labeled_fronts, lat, lon, time, nc_path, front_ids)
        output_files['netcdf'] = nc_path

    if 'csv' in formats:
        csv_path = output_dir / f"{base_name}.csv"
        to_csv(properties, csv_path, front_ids)
        output_files['csv'] = csv_path

    if 'parquet' in formats:
        parquet_path = output_dir / f"{base_name}.parquet"
        to_parquet(properties, parquet_path, front_ids)
        output_files['parquet'] = parquet_path

    print(f"\nAll files saved to {output_dir}/")
    return output_files
