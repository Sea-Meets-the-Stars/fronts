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

from fronts.properties import group_labels, io, geometry


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
        Directory where labeled array, group table, and properties parquet
        are saved.
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

    # Extract timestamp from filename for io path generation
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', str(fronts_file))
    if not match:
        raise ValueError(f"Could not extract timestamp from fronts_file: {fronts_file}")
    time_str = match.group(1).replace('_', ':')   # e.g. '2012-11-09T12:00:00'

    # --- Label ---
    labeled, n = group_labels.label_fronts(fronts_binary, connectivity=2, return_num=True)
    print(f"Labeled {n:,} fronts")
    np.save(io.get_global_front_output_path(output_dir, time_str, 'labeled'), labeled)

    # --- Properties (bbox + centroid indices) and front IDs ---
    properties = group_labels.get_front_properties(labeled)
    front_ids  = group_labels.generate_front_ids(lat, lon, str(fronts_file),
                                                 properties=properties)

    # --- Group table ---
    group_table_file = io.get_global_front_output_path(output_dir, time_str, 'group_table')
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

    parquet_file = io.get_global_front_output_path(output_dir, time_str, 'properties')
    df.to_parquet(parquet_file, index=False)
    print(f"Wrote: {parquet_file}")

    return df
