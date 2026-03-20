"""
Front grouping algorithm.

Core computation for labeling connected front components, computing geometric
properties in parallel, and saving results. Parallel structure to
fronts.finding.algorithms — file I/O and path setup live in the caller
(build_v1.py); this module handles the pure processing.
"""

import re
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path

from fronts.properties import group_labels, io, geometry, colocation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_fronts_filename(fronts_file: str):
    """
    Extract time_str and run_tag from a fronts filename.

    Expected pattern: LLC4320_YYYY-MM-DDTHH_MM_SS_{run_tag}.npy
    e.g. 'LLC4320_2012-11-09T12_00_00_v1_bin_A.npy'
      -> time_str = '2012-11-09T12:00:00'
      -> run_tag  = 'v1_bin_A'

    Returns
    -------
    time_str : str
        ISO 8601 timestamp with colons restored.
    run_tag : str
        Everything after the timestamp and before .npy, e.g. 'v1_bin_A'.
    timestamp_raw : str
        Timestamp as it appears in the filename (underscores, not colons),
        e.g. '2012-11-09T12_00_00'. Used to build property filenames.
    """
    fname = str(fronts_file)
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_(.+?)\.npy', fname)
    if not match:
        raise ValueError(
            f"Could not parse timestamp and run_tag from fronts_file: {fronts_file}\n"
            "Expected pattern: *_YYYY-MM-DDTHH_MM_SS_{run_tag}.npy"
        )
    timestamp_raw = match.group(1)                      # '2012-11-09T12_00_00'
    time_str      = timestamp_raw.replace('_', ':')     # '2012-11-09T12:00:00'
    run_tag       = match.group(2)                      # 'v1_bin_A'
    return time_str, run_tag, timestamp_raw


# ---------------------------------------------------------------------------
# Module-level globals for copy-on-write sharing across forked workers.
# Must be at module level to be picklable by multiprocessing.
# Set in group_fronts() before Pool creation; inherited by workers via fork.
# ---------------------------------------------------------------------------
_GLOBAL_LABELED = None
_GLOBAL_LAT     = None
_GLOBAL_LON     = None


def _process_cutout_wrapper(args_tuple):
    """Multiprocessing worker — extracts cutout and calls process_single_front."""
    label, name, y0, y1, x0, x1, time_str, skip_curvature = args_tuple
    labeled_cutout = _GLOBAL_LABELED[y0:y1, x0:x1]
    mask = labeled_cutout == label
    return geometry.process_single_front(
        label=label, name=name,
        mask=mask,
        lat=_GLOBAL_LAT[y0:y1, x0:x1],
        lon=_GLOBAL_LON[y0:y1, x0:x1],
        time_str=time_str,
        y0=y0, y1=y1, x0=x0, x1=x1,
        skip_curvature=skip_curvature,
    )


def group_fronts(
    fronts_binary: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    fronts_file: str,
    output_dir: str,
    n_workers: int = None,
    skip_curvature: bool = False,
) -> pd.DataFrame:
    """
    Label connected front components and compute geometric properties in parallel.

    Parameters
    ----------
    fronts_binary : np.ndarray
        2D binary front field (True/1 = front pixel).
    lat, lon : np.ndarray
        2D coordinate grids, same shape as fronts_binary.
    fronts_file : str
        Path to the source .npy file. Used to extract the timestamp for
        output filenames and front IDs (expects YYYY-MM-DDTHH_MM_SS pattern).
    output_dir : str
        Directory where labeled array, group table, properties parquet,
        and metadata JSON are saved.
    n_workers : int, optional
        Parallel workers. Defaults to CPU count.
    skip_curvature : bool, optional
        Skip curvature calculation (~50% faster). Default False.

    Returns
    -------
    df : pd.DataFrame
        Per-front geometric properties table (also saved as parquet).
    """
    global _GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or cpu_count()

    time_str, run_tag, _ = _parse_fronts_filename(fronts_file)

    # --- Label ---
    labeled, n = group_labels.label_fronts(fronts_binary, connectivity=2, return_num=True)
    print(f"Labeled {n:,} fronts")
    np.save(io.get_global_front_output_path(output_dir, time_str, 'labeled', run_tag), labeled)

    # --- Properties (bbox + centroid indices) and front IDs ---
    properties = group_labels.get_front_properties(labeled)
    front_ids  = group_labels.generate_front_ids(lat, lon, str(fronts_file),
                                                 properties=properties)

    # --- Group table ---
    group_table_file = io.get_global_front_output_path(output_dir, time_str, 'group_table', run_tag)
    group_df = io.write_front_group_table(front_ids, properties, group_table_file)

    # --- Parallel geometric properties ---
    _GLOBAL_LABELED = labeled
    _GLOBAL_LAT     = lat
    _GLOBAL_LON     = lon

    front_args = [
        (row.label, row.name, row.y0, row.y1, row.x0, row.x1, time_str, skip_curvature)
        for row in group_df.itertuples()
    ]
    chunksize = max(100, len(front_args) // (n_workers * 10))

    with Pool(processes=n_workers) as pool:
        results = [
            r for r in pool.imap_unordered(
                _process_cutout_wrapper, front_args, chunksize=chunksize)
            if r is not None
        ]
    print(f"Processed {len(results):,} fronts")

    # --- Save parquet ---
    df = pd.DataFrame(results)
    col_order = ['label', 'name', 'time', 'npix',
                 'y0', 'y1', 'x0', 'x1',
                 'centroid_lat', 'centroid_lon',
                 'length_km', 'orientation', 'num_branches',
                 'lat_min', 'lat_max', 'lon_min', 'lon_max',
                 'mean_curvature', 'curvature_direction']
    df = df[[c for c in col_order if c in df.columns]]

    parquet_file = io.get_global_front_output_path(output_dir, time_str, 'properties', run_tag)
    df.to_parquet(parquet_file, index=False)
    print(f"Wrote: {parquet_file}")

    # --- Save metadata JSON ---
    from datetime import datetime
    metadata = {
        'fronts_file':              str(fronts_file),
        'run_tag':                  run_tag,
        'time':                     time_str,
        'shape':                    list(fronts_binary.shape),
        'num_fronts':               len(df),
        'n_workers':                n_workers,
        'skip_curvature':           skip_curvature,
        'lat_range':                [float(lat.min()), float(lat.max())],
        'lon_range':                [float(lon.min()), float(lon.max())],
        'timestamp':                datetime.now().isoformat(),
    }
    metadata_file = io.get_global_front_output_path(output_dir, time_str, 'metadata', run_tag)
    io.write_json(metadata, metadata_file)

    return df


def colocate_fronts(
    labeled: np.ndarray,
    property_names: list,
    property_dir: str,
    fronts_file: str,
    output_dir: str,
    stats: list = None,
    percentiles: list = None,
    min_npix: int = 1,
    nan_policy: str = 'omit',
    dilation_radius: int = 0,
) -> pd.DataFrame:
    """
    Co-locate labeled fronts with property fields and save per-front statistics.

    Wraps colocation.colocate_fronts_with_properties() with file I/O,
    parallel to how group_fronts() wraps group_labels and geometry.

    Property files are located automatically using the timestamp and version
    extracted from fronts_file. Expected filename pattern:

        LLC4320_{timestamp}_{property_name}_{version}.nc

    e.g. for fronts_file='LLC4320_2012-11-09T12_00_00_v1_bin_A.npy' and
    property_name='relative_vorticity':
        property_dir/LLC4320_2012-11-09T12_00_00_relative_vorticity_v1.nc

    Parameters
    ----------
    labeled : np.ndarray
        Integer label array from group_fronts() (0 = background).
    property_names : list of str
        Names of property fields to co-locate, e.g.
        ['relative_vorticity', 'strain_n', 'frontogenesis_tendency'].
        Each name must match both the variable name inside its .nc file
        and the {property_name} component of the filename.
    property_dir : str
        Directory containing property .nc files.
    fronts_file : str
        Path to the source binary fronts .npy file. Timestamp and version
        (e.g. 'v1') are extracted from this filename and used to build
        property file paths and output filenames.
    output_dir : str
        Directory where the colocation parquet is saved.
    stats : list, optional
        Statistics to compute per property. Any of 'mean', 'std', 'median',
        'min', 'max', 'count'. Defaults to ['mean', 'std', 'median'].
    percentiles : list, optional
        Percentiles to compute per property, e.g. [10, 25, 75, 90].
    min_npix : int, optional
        Minimum front size (pixels) to include. Default 1.
    nan_policy : {'omit', 'propagate'}, optional
        How to handle NaN values (e.g. land pixels). Default 'omit'.
    dilation_radius : int, optional
        Dilate each front by this many pixels before computing stats.
        Useful for capturing the near-front environment. Default 0.

    Returns
    -------
    df : pd.DataFrame
        One row per front. Columns: flabel, npix, {prop}_{stat},
        {prop}_p{pct}. Also saved as parquet.
    """
    import xarray as xr

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_str, run_tag, timestamp_raw = _parse_fronts_filename(fronts_file)
    version = run_tag.split('_')[0]   # e.g. 'v1' from 'v1_bin_A'

    # --- Load property arrays from standardised filenames ---
    property_arrays = {}
    for prop_name in property_names:
        prop_file = Path(property_dir) / f'LLC4320_{timestamp_raw}_{prop_name}_{version}.nc'
        print(f"Loading {prop_name} from {prop_file.name}...")
        with xr.open_dataset(prop_file) as ds:
            property_arrays[prop_name] = ds[prop_name].values.squeeze()

    print(f"Co-locating {len(property_arrays)} properties with "
          f"{(labeled > 0).sum():,} front pixels "
          f"(dilation_radius={dilation_radius})...")

    df = colocation.colocate_fronts_with_properties(
        labeled_fronts=labeled,
        properties=property_arrays,
        stats=stats,
        percentiles=percentiles,
        min_npix=min_npix,
        nan_policy=nan_policy,
        dilation_radius=dilation_radius,
    )
    print(f"Co-located {len(df):,} fronts")

    parquet_file = io.get_global_front_output_path(output_dir, time_str, 'colocation', run_tag)
    df.to_parquet(parquet_file, index=False)
    print(f"Wrote: {parquet_file}")

    return df
