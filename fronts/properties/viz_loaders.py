"""Convenience loaders for downstream analysis and visualisation.

These functions load, merge, and align the outputs of the global front
pipeline so that notebooks and scripts can get to plotting quickly.

Atomic loaders (global front results):
  load_metadata            — JSON metadata for a global run
  load_labeled_array       — labeled-fronts .npy file
  load_geometry_table      — geometry parquet (one row per front)
  load_colocation_table    — colocation/properties parquet
  merge_geometry_colocation — inner-merge geometry + colocation

Coordinate handling:
  load_llc_coords          — lat/lon from LLC coordinate NetCDF
  compute_longitude_shift  — shift needed to make lon run -180 … +180
  roll_to_pm180            — roll 2-D arrays by a column shift

Property-file loading:
  property_file_path       — standard NetCDF path for a derived field
  load_single_property     — load one property array
  load_property_arrays     — batch-load multiple properties

Orchestrator:
  load_global_front_results — one-call loader for everything
"""

import json as _json
import os
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from fronts.properties.colocation import _load_property_file
from fronts.properties.io import get_global_front_output_path, fronts_dir


# ---------------------------------------------------------------------------
# Default path helpers
# ---------------------------------------------------------------------------

def _default_ogcm_path(*parts: str) -> Path:
    """Build a path under $OS_OGCM, raising if the env var is unset."""
    root = os.getenv('OS_OGCM')
    if root is None:
        raise EnvironmentError(
            "OS_OGCM environment variable is not set.  "
            "Set it to the OGCM data root (e.g. /home/user/data/OGCM)."
        )
    return Path(root).joinpath(*parts)


# ---------------------------------------------------------------------------
# Atomic loaders — front results
# ---------------------------------------------------------------------------

def load_metadata(
    results_dir: Union[str, Path],
    time_str: str,
    run_tag: str = '',
    verbose: bool = True,
) -> dict:
    """Load the JSON metadata file for a global front run.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing the run outputs.
    time_str : str
        ISO 8601 timestamp (e.g. '2012-11-09T12:00:00').
    run_tag : str, optional
        Run identifier suffix (e.g. 'v1_bin_A').

    Returns
    -------
    dict
        Parsed metadata (keys typically include ``shape``, ``num_fronts``,
        ``downsample_factor``).
    """
    path = get_global_front_output_path(results_dir, time_str, 'metadata', run_tag)
    if verbose:
        print(f"Loading metadata from {path}")
    with open(path) as f:
        return _json.load(f)


def load_labeled_array(
    results_dir: Union[str, Path],
    time_str: str,
    run_tag: str = '',
) -> np.ndarray:
    """Load the labeled-fronts ``.npy`` file.

    Returns
    -------
    np.ndarray
        2-D integer array where each pixel's value is its front label
        (0 = background).
    """
    path = get_global_front_output_path(results_dir, time_str, 'label_map', run_tag)
    return np.load(path)


def load_geometry_table(
    results_dir: Union[str, Path],
    time_str: str,
    run_tag: str = '',
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the geometry parquet for a global front run.

    Returns
    -------
    pd.DataFrame
        One row per front with columns like ``label``, ``name``,
        ``centroid_lat``, ``centroid_lon``, ``length_km``, ``orientation``,
        bounding-box indices, etc.
    """
    path = get_global_front_output_path(results_dir, time_str, 'geometry', run_tag)
    if verbose:
        print(f"Loading geometry table from {path}")
    return pd.read_parquet(path, engine='pyarrow')


def load_colocation_table(
    results_dir: Union[str, Path],
    time_str: str,
    run_tag: str = '',
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the colocation/properties parquet for a global front run.

    Returns
    -------
    pd.DataFrame
        One row per front with columns like ``flabel``, ``npix``,
        and per-property statistics (e.g. ``gradb2_median``).
    """
    path = get_global_front_output_path(results_dir, time_str, 'properties', run_tag)
    if verbose:
        print(f"Loading colocation table from {path}")
    return pd.read_parquet(path, engine='pyarrow')


def merge_geometry_colocation(
    df_geometry: pd.DataFrame,
    df_colocation: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-merge geometry and colocation tables on ``label == flabel``.

    Parameters
    ----------
    df_geometry : pd.DataFrame
        From :func:`load_geometry_table`.
    df_colocation : pd.DataFrame
        From :func:`load_colocation_table`.

    Returns
    -------
    pd.DataFrame
        Enriched table with columns from both inputs.
    """
    return df_geometry.merge(
        df_colocation, left_on='label', right_on='flabel', how='inner',
    )


# ---------------------------------------------------------------------------
# Coordinate handling
# ---------------------------------------------------------------------------

def load_llc_coords(
    coords_file: Optional[Union[str, Path]] = None,
    downsample_factor: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load lat/lon from the LLC coordinate NetCDF.

    Handles both ``lat``/``lon`` and ``YC``/``XC`` variable names.

    Parameters
    ----------
    coords_file : str or Path, optional
        Path to the coordinate NetCDF.  Defaults to
        ``$OS_OGCM/LLC/Fronts/coords/LLC_coords_lat_lon.nc``.
    downsample_factor : int, optional
        If provided, subsample both arrays by this factor along each axis.

    Returns
    -------
    lat, lon : np.ndarray
        2-D coordinate arrays.
    """

    if coords_file is None:
        coords_file = _default_ogcm_path('LLC', 'Fronts', 'coords',
                                          'LLC_coords_lat_lon.nc')

    with xr.open_dataset(coords_file) as ds:
        lat = ds['lat'].values if 'lat' in ds else ds['YC'].values
        lon = ds['lon'].values if 'lon' in ds else ds['XC'].values

    if downsample_factor and downsample_factor > 1:
        lat = lat[::downsample_factor, ::downsample_factor]
        lon = lon[::downsample_factor, ::downsample_factor]

    return lat, lon


def compute_longitude_shift(lon: np.ndarray) -> int:
    """Compute the column roll so longitude runs -180 to +180.

    Examines the middle row of *lon* to find the column with the minimum
    longitude value, then returns ``-min_col`` as the shift to pass to
    ``np.roll(..., shift, axis=1)``.

    Parameters
    ----------
    lon : np.ndarray
        2-D longitude array.

    Returns
    -------
    int
        Column shift (0 if no roll needed).
    """
    sample_row = lon.shape[0] // 2
    min_lon_col = int(np.argmin(lon[sample_row, :]))
    return -min_lon_col if min_lon_col != 0 else 0


def roll_to_pm180(*arrays: np.ndarray, shift: int) -> Tuple[np.ndarray, ...]:
    """Roll 2-D arrays by *shift* columns along axis 1.

    Convenience wrapper for the common pattern of rolling lat, lon,
    labeled, and property arrays together after :func:`compute_longitude_shift`.

    Parameters
    ----------
    *arrays : np.ndarray
        One or more 2-D arrays to roll.
    shift : int
        Column shift (from :func:`compute_longitude_shift`).

    Returns
    -------
    tuple of np.ndarray
        Rolled arrays in the same order as the inputs.
    """
    if shift == 0:
        return arrays if len(arrays) > 1 else arrays[0]
    rolled = tuple(np.roll(a, shift, axis=1) for a in arrays)
    return rolled if len(rolled) > 1 else rolled[0]


# ---------------------------------------------------------------------------
# Property-file loading
# ---------------------------------------------------------------------------

def property_file_path(
    property_name: str,
    timestamp: str,
    version: str = '1',
    properties_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Construct the standard NetCDF path for a derived property field.

    The naming convention is:
    ``{properties_dir}/LLC4320_{timestamp}_{property_name}_v{version}.nc``

    Parameters
    ----------
    property_name : str
        Field name (e.g. ``'gradb2'``, ``'relative_vorticity'``).
    timestamp : str
        Filename-safe timestamp (e.g. ``'2012-11-09T12_00_00'``).
    version : str
        Version tag (e.g. ``'1'``).
    properties_dir : str or Path, optional
        Directory containing the NetCDF files.
        Defaults to ``PATH/V{version}/YYYYMMDD_HHMMSS/``.

    Returns
    -------
    Path
    """
    if properties_dir is None:
        properties_dir = fronts_dir(version, timestamp)
    return Path(properties_dir) / f'LLC4320_{timestamp}_{property_name}_v{version}.nc'


def load_single_property(
    property_name: str,
    timestamp: str,
    version: str = '1',
    properties_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Load one property array from its NetCDF file.

    Uses :func:`fronts.properties.colocation._load_property_file` internally
    and squeezes singleton dimensions.

    Parameters
    ----------
    property_name : str
        Field name (e.g. ``'gradb2'``).
    timestamp : str
        Filename-safe timestamp (e.g. ``'2012-11-09T12_00_00'``).
    version : str
        Version tag.
    properties_dir : str or Path, optional
        Defaults to ``PATH/V{version}/YYYYMMDD_HHMMSS/``.

    Returns
    -------
    np.ndarray
        2-D property array.
    """

    fpath = property_file_path(property_name, timestamp, version, properties_dir)
    # _load_property_file expects a (filepath, varname) tuple
    arr = _load_property_file((str(fpath), property_name))
    return arr.squeeze()


def load_property_arrays(
    property_names: Sequence[str],
    timestamp: str,
    version: str = '1',
    properties_dir: Optional[Union[str, Path]] = None,
    downsample_factor: Optional[int] = None,
    shift: int = 0,
) -> Dict[str, np.ndarray]:
    """Batch-load multiple property arrays into a dict.

    Calls :func:`load_single_property` for each name, then optionally
    downsamples and rolls to match a previously loaded coordinate grid.

    Parameters
    ----------
    property_names : sequence of str
        Field names to load.
    timestamp : str
        Filename-safe timestamp.
    version : str
        Version tag.
    properties_dir : str or Path, optional
        Defaults to ``PATH/V{version}/YYYYMMDD_HHMMSS/``.
    downsample_factor : int, optional
        Spatial subsampling factor.
    shift : int, optional
        Longitude column roll (from :func:`compute_longitude_shift`).

    Returns
    -------
    dict
        ``{property_name: np.ndarray}`` mapping.
    """
    result = {}
    for name in property_names:
        arr = load_single_property(name, timestamp, version, properties_dir)
        if downsample_factor and downsample_factor > 1:
            arr = arr[::downsample_factor, ::downsample_factor]
        if shift != 0:
            arr = np.roll(arr, shift, axis=1)
        result[name] = arr
    return result


# ---------------------------------------------------------------------------
# Orchestrator — one-call convenience loader
# ---------------------------------------------------------------------------

def load_global_front_results(
    results_dir: Union[str, Path],
    time_str: str,
    run_tag: str = '',
    coords_file: Optional[Union[str, Path]] = None,
) -> dict:
    """Load everything needed to visualize a global front run.

    Calls the atomic loaders, merges geometry + colocation, loads and
    aligns coordinates, and rolls arrays so longitude runs -180 to +180.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing the run outputs.
    time_str : str
        ISO 8601 timestamp (e.g. ``'2012-11-09T12:00:00'``).
    run_tag : str, optional
        Run identifier suffix (e.g. ``'v1_bin_A'``).
    coords_file : str or Path, optional
        Path to LLC coordinate NetCDF.  Defaults to
        ``$OS_OGCM/LLC/Fronts/coords/LLC_coords_lat_lon.nc``.

    Returns
    -------
    dict
        Keys:

        - ``metadata`` : dict
        - ``labeled_global`` : np.ndarray (2-D, rolled)
        - ``df_enriched`` : pd.DataFrame
        - ``lat_global`` : np.ndarray (2-D, rolled)
        - ``lon_global`` : np.ndarray (2-D, rolled)
        - ``shift`` : int — longitude roll applied
    """
    # 1. Metadata
    metadata = load_metadata(results_dir, time_str, run_tag)
    ds_factor = metadata.get('downsample_factor', None)

    # 2. Labeled array
    labeled = load_labeled_array(results_dir, time_str, run_tag)

    # 3. Enriched DataFrame
    df_geom = load_geometry_table(results_dir, time_str, run_tag)
    df_coloc = load_colocation_table(results_dir, time_str, run_tag)
    df_enriched = merge_geometry_colocation(df_geom, df_coloc)

    # 4. Coordinates (downsample if metadata says so)
    lat, lon = load_llc_coords(coords_file, downsample_factor=ds_factor)

    # 5. Longitude alignment
    shift = compute_longitude_shift(lon)
    lon, lat, labeled = roll_to_pm180(lon, lat, labeled, shift=shift)

    return {
        'metadata': metadata,
        'labeled_global': labeled,
        'df_enriched': df_enriched,
        'lat_global': lat,
        'lon_global': lon,
        'shift': shift,
    }
