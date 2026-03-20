#!/usr/bin/env python3
"""
Global Front Processing — Step Methods
=======================================

Exposes each step of the global front processing pipeline as a standalone
callable function. This allows individual steps to be run, skipped, or
rerun from within Python code (e.g. notebooks, other scripts) without
going through the command line.

Pipeline
--------
1. load_fronts_data        — load fronts array + coordinates from disk
2. label_and_filter        — connected component labeling + size filter
3. save_labeled_array      — write labeled array to disk
4. extract_bboxes_and_ids  — bounding boxes + unique front IDs
5. build_group_table       — write group table (label, name, bbox) to disk
6. run_parallel_geometry   — pool workers on cutouts, calculate properties
7. save_results            — write parquet + metadata JSON to disk

Each function returns its outputs explicitly so the caller can pass them
into the next step, or substitute their own data at any point.

Example (full pipeline from Python)
------------------------------------
    from group_fronts_global_methods import (
        load_fronts_data, label_and_filter, save_labeled_array,
        extract_bboxes_and_ids, build_group_table,
        run_parallel_geometry, save_results
    )

    fronts, lat, lon, time_str = load_fronts_data(
        fronts_file='/data/fronts.npy',
        coords_file='/data/coords.nc',
    )
    labeled, n = label_and_filter(fronts, min_size=10)
    labeled_file = save_labeled_array(labeled, '/data/output', time_str)
    properties, front_ids = extract_bboxes_and_ids(
        labeled, lat, lon, fronts_file='/data/fronts.npy'
    )
    group_df, group_table_file = build_group_table(
        front_ids, properties, '/data/output', time_str
    )
    results, total_time = run_parallel_geometry(
        group_df, labeled, lat, lon, time_str, n_workers=8
    )
    df, parquet_file = save_results(
        results=results,
        fronts_file='/data/fronts.npy',
        coords_file='/data/coords.nc',
        fronts_global=fronts,
        lat_global=lat,
        lon_global=lon,
        time_str=time_str,
        output_dir='/data/output',
        min_size=10,
        n_workers=8,
        total_time=total_time,
    )

    python /home/lhoffma2/git/fronts/dev/group_fronts_global_methods.py \
    --fronts_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/outputs/LLC4320_2012-11-09T12_00_00_bin_A.npy' \
    --coords_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/LLC_coords_lat_lon.nc' \
    --output_dir  '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/testing/pr2_1/' \
    --n_workers 2 \
    --skip_curvature
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import time as time_module
import re
from multiprocessing import Pool, cpu_count

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from fronts.properties import group_labels, io, geometry

# ---------------------------------------------------------------------------
# Module-level globals for copy-on-write sharing across forked workers.
# Must be at module level to be picklable by multiprocessing.
# ---------------------------------------------------------------------------
_GLOBAL_LABELED = None
_GLOBAL_LAT = None
_GLOBAL_LON = None


def _process_cutout_wrapper(args_tuple):
    """
    Module-level multiprocessing wrapper for geometry.process_single_front.

    Must be at module level to be picklable by multiprocessing. Extracts
    the cutout from shared globals using bbox coords, then delegates to
    geometry.process_single_front.

    Parameters
    ----------
    args_tuple : tuple
        (label, name, y0, y1, x0, x1, time_str, length_method, skip_curvature)
    """
    label, name, y0, y1, x0, x1, time_str, length_method, skip_curvature = args_tuple

    labeled_cutout = _GLOBAL_LABELED[y0:y1, x0:x1]
    lat_cutout     = _GLOBAL_LAT[y0:y1, x0:x1]
    lon_cutout     = _GLOBAL_LON[y0:y1, x0:x1]
    mask           = labeled_cutout == label

    return geometry.process_single_front(
        label=label, name=name,
        mask=mask, lat=lat_cutout, lon=lon_cutout,
        time_str=time_str,
        y0=y0, y1=y1, x0=x0, x1=x1,
        length_method=length_method,
        skip_curvature=skip_curvature
    )


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------

def load_fronts_data(
    fronts_file,
    coords_file,
    time_input=None,
    downsample=None,
):
    """
    Load global fronts array and coordinates from disk.

    Parameters
    ----------
    fronts_file : str
        Path to binary fronts .npy file
    coords_file : str
        Path to coordinates .nc file containing lat/lon
    time_input : str or None, optional
        ISO 8601 timestamp (e.g. '2012-11-09T12:00:00'). If None,
        timestamp is extracted from the fronts filename.
    downsample : int or None, optional
        Downsample factor for testing (e.g. 2 = half resolution).
        Applied to both fronts and coordinates.

    Returns
    -------
    fronts_global : np.ndarray
        Binary fronts array
    lat_global : np.ndarray
        2D latitude array
    lon_global : np.ndarray
        2D longitude array
    time_str : str
        ISO 8601 timestamp string used for this run
    """
    print("Loading global fronts...")
    t0 = time_module.time()
    fronts_global = np.load(fronts_file)

    if downsample:
        fronts_global = fronts_global[::downsample, ::downsample]

    print(f"  Shape: {fronts_global.shape}")
    print(f"  Front pixels: {np.sum(fronts_global):,}")
    print(f"  Loaded in {time_module.time() - t0:.1f}s")

    print("\nLoading coordinates...")
    t0 = time_module.time()
    ds_coords = xr.open_dataset(coords_file)
    lat_global = (ds_coords['lat'].values if 'lat' in ds_coords
                  else ds_coords['YC'].values)
    lon_global = (ds_coords['lon'].values if 'lon' in ds_coords
                  else ds_coords['XC'].values)

    if downsample:
        lat_global = lat_global[::downsample, ::downsample]
        lon_global = lon_global[::downsample, ::downsample]

    ds_coords.close()
    print(f"  Lat range: [{lat_global.min():.1f}, {lat_global.max():.1f}]")
    print(f"  Lon range: [{lon_global.min():.1f}, {lon_global.max():.1f}]")
    print(f"  Loaded in {time_module.time() - t0:.1f}s")

    if time_input is None:
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', fronts_file)
        if match:
            time_str = match.group(1).replace('_', ':')
        else:
            time_str = datetime.now().isoformat()
        print(f"\nExtracted time: {time_str}")
    else:
        time_str = time_input

    return fronts_global, lat_global, lon_global, time_str


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------

def label_and_filter(
    fronts_global,
    min_size=10,
):
    """
    Label connected front components and filter by minimum size.

    Parameters
    ----------
    fronts_global : np.ndarray
        Binary fronts array from load_fronts_data()
    min_size : int, optional
        Minimum number of pixels for a front to be kept. Default is 10.

    Returns
    -------
    labeled : np.ndarray
        Labeled and filtered front array
    num_fronts : int
        Number of fronts remaining after filtering
    """
    print("\n" + "=" * 70)
    print("LABELING FRONTS")
    print("=" * 70)

    t0 = time_module.time()
    labeled, num_initial = group_labels.label_fronts(
        fronts_global, connectivity=2, return_num=True
    )
    print(f"  Initial fronts: {num_initial:,}")
    print(f"  Labeled in {time_module.time() - t0:.1f}s")

    print(f"\nFiltering fronts (min_size={min_size})...")
    t0 = time_module.time()
    front_labels = group_labels.get_front_labels(labeled)
    num_fronts = len(front_labels)
    print(f"  After filtering: {num_fronts:,} fronts")
    print(f"  Filtered in {time_module.time() - t0:.1f}s")

    return labeled, num_fronts


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------

def save_labeled_array(
    labeled,
    output_dir,
    time_str,
):
    """
    Save labeled front array to disk as .npy file.

    Parameters
    ----------
    labeled : np.ndarray
        Labeled front array from label_and_filter()
    output_dir : str or Path
        Output directory
    time_str : str
        ISO 8601 timestamp string used for filename

    Returns
    -------
    labeled_file : Path
        Path to the saved .npy file
    """
    labeled_file = io.get_global_front_output_path(output_dir, time_str, 'labeled')
    np.save(labeled_file, labeled)
    print(f"\nSaved labeled array: {labeled_file}")
    return labeled_file


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------

def extract_bboxes_and_ids(
    labeled,
    lat_global,
    lon_global,
    fronts_file,
):
    """
    Extract front properties (incl. bounding boxes) and generate unique string IDs.

    Parameters
    ----------
    labeled : np.ndarray
        Labeled front array from label_and_filter()
    lat_global : np.ndarray
        2D latitude array
    lon_global : np.ndarray
        2D longitude array
    fronts_file : str
        Path to the binary fronts .npy file; timestamp is extracted from filename.

    Returns
    -------
    properties : dict
        Mapping label -> {'npix', 'bbox': (min_row, min_col, max_row, max_col),
        'centroid_indices': (row, col)}
    front_ids : dict
        Mapping label -> unique front ID string (TIME_LAT_LON format)
    """
    print("\nExtracting front properties...")
    t0 = time_module.time()
    properties = group_labels.get_front_properties(labeled)
    print(f"  Valid fronts: {len(properties):,}  ({time_module.time() - t0:.1f}s)")

    print("Generating front IDs...")
    t0 = time_module.time()
    front_ids = group_labels.generate_front_ids(
        lat_global, lon_global, fronts_file, properties=properties
    )
    print(f"  Generated {len(front_ids):,} IDs  ({time_module.time() - t0:.1f}s)")

    return properties, front_ids


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------

def build_group_table(
    front_ids,
    properties,
    output_dir,
    time_str,
):
    """
    Write the group table (label, name, x0, y0, x1, y1) to disk.

    Parameters
    ----------
    front_ids : dict
        Mapping label -> front ID string from extract_bboxes_and_ids()
    properties : dict
        Mapping label -> properties dict from extract_bboxes_and_ids()
    output_dir : str or Path
        Output directory
    time_str : str
        ISO 8601 timestamp string used for filename

    Returns
    -------
    group_df : pd.DataFrame
        Group table DataFrame
    group_table_file : Path
        Path to the saved group table file
    """
    print("\n" + "=" * 70)
    print("WRITING GROUP TABLE")
    print("=" * 70)

    group_table_file = io.get_global_front_output_path(output_dir, time_str, 'group_table')
    group_df = io.write_front_group_table(front_ids, properties, group_table_file)

    return group_df, group_table_file


# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------

def run_parallel_geometry(
    group_df,
    labeled,
    lat_global,
    lon_global,
    time_str,
    n_workers=None,
    length_method='skeleton',
    skip_curvature=False,
):
    """
    Calculate geometric properties for all fronts in parallel.

    Sets module-level globals so forked workers inherit the large arrays
    via copy-on-write (no serialization overhead). Each worker extracts
    its own small cutout using the bounding box from group_df.

    Parameters
    ----------
    group_df : pd.DataFrame
        Group table from build_group_table()
    labeled : np.ndarray
        Labeled front array
    lat_global : np.ndarray
        2D latitude array
    lon_global : np.ndarray
        2D longitude array
    time_str : str
        ISO 8601 timestamp string
    n_workers : int or None, optional
        Number of parallel workers. Defaults to cpu_count().
    length_method : str, optional
        'skeleton' (default) or 'perimeter'
    skip_curvature : bool, optional
        Skip curvature calculation to save ~50% time. Default False.

    Returns
    -------
    results : list of dict
        List of property dicts, one per front
    total_time : float
        Wall time for this step in seconds
    """
    global _GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON

    n_workers = n_workers or cpu_count()

    print("\n" + "=" * 70)
    print("CALCULATING GEOMETRIC PROPERTIES (PARALLEL)")
    print("=" * 70)
    print(f"Processing {len(group_df):,} fronts using {n_workers} workers...")
    print(f"Length method : {length_method}")
    print(f"Curvature     : {'SKIPPED' if skip_curvature else 'calculated'}")

    print("\nSetting up shared arrays for workers...")
    _GLOBAL_LABELED = labeled
    _GLOBAL_LAT     = lat_global
    _GLOBAL_LON     = lon_global
    print(f"  Arrays ready (labeled: {labeled.nbytes / 1e9:.1f} GB)")

    front_args = [
        (row.label, row.name, row.y0, row.y1, row.x0, row.x1,
         time_str, length_method, skip_curvature)
        for row in group_df.itertuples()
    ]

    t0 = time_module.time()
    num_fronts = len(front_args)
    optimal_chunksize = max(100, num_fronts // (n_workers * 10))
    print(f"\nStarting Pool (chunksize={optimal_chunksize})...")

    with Pool(processes=n_workers) as pool:
        results = []
        completed = 0

        for result in pool.imap_unordered(
            _process_cutout_wrapper,
            front_args,
            chunksize=optimal_chunksize
        ):
            if result is not None:
                results.append(result)

            completed += 1
            if completed % 1000 == 0:
                elapsed = time_module.time() - t0
                rate = completed / elapsed
                remaining = (num_fronts - completed) / rate / 60
                print(f"  {completed:,}/{num_fronts:,} "
                      f"({100 * completed / num_fronts:.1f}%) — "
                      f"{rate:.1f} fronts/s — "
                      f"ETA: {remaining:.1f} min")

    total_time = time_module.time() - t0
    print(f"\n✓ Processed {len(results):,} fronts in {total_time / 60:.1f} min")
    if results:
        print(f"  Average: {total_time / len(results):.3f}s per front")

    return results, total_time


# ---------------------------------------------------------------------------
# Step 7
# ---------------------------------------------------------------------------

def save_results(
    results,
    fronts_file,
    coords_file,
    fronts_global,
    lat_global,
    lon_global,
    time_str,
    output_dir,
    min_size,
    n_workers,
    total_time,
    downsample=None,
):
    """
    Save geometric properties to Parquet and write processing metadata JSON.

    Parameters
    ----------
    results : list of dict
        Property dicts from run_parallel_geometry()
    fronts_file : str
        Path to original fronts file (recorded in metadata)
    coords_file : str
        Path to original coordinates file (recorded in metadata)
    fronts_global : np.ndarray
        Global fronts array (shape recorded in metadata)
    lat_global : np.ndarray
        2D latitude array (range recorded in metadata)
    lon_global : np.ndarray
        2D longitude array (range recorded in metadata)
    time_str : str
        ISO 8601 timestamp string
    output_dir : str or Path
        Output directory
    min_size : int
        Minimum pixel size filter applied (recorded in metadata)
    n_workers : int
        Number of workers used (recorded in metadata)
    total_time : float
        Processing wall time in seconds from run_parallel_geometry()
    downsample : int or None, optional
        Downsample factor used (recorded in metadata). Default None.

    Returns
    -------
    df : pd.DataFrame
        Properties DataFrame
    parquet_file : Path
        Path to saved Parquet file
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    df = pd.DataFrame(results)

    cols = ['label', 'name', 'time', 'npix',
            'y0', 'y1', 'x0', 'x1',
            'centroid_lat', 'centroid_lon',
            'length_km', 'orientation', 'num_branches',
            'lat_min', 'lat_max', 'lon_min', 'lon_max',
            'mean_curvature', 'curvature_direction']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    parquet_file = io.get_global_front_output_path(output_dir, time_str, 'properties')
    df.to_parquet(parquet_file, index=False)
    print(f"✓ Parquet : {parquet_file}  ({parquet_file.stat().st_size / 1e6:.1f} MB)")

    return df, parquet_file


# ---------------------------------------------------------------------------
# main — chains all steps together, mirrors group_fronts_global.py behaviour
# ---------------------------------------------------------------------------

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Process global fronts with parallelization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--fronts_file', required=True)
    parser.add_argument('--coords_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--time', default=None)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--min_size', type=int, default=10)
    parser.add_argument('--downsample', type=int, default=None)
    parser.add_argument('--length_method', type=str, default='skeleton',
                        choices=['skeleton', 'perimeter'])
    parser.add_argument('--skip_curvature', action='store_true')
    return parser


def main(
    fronts_file,
    coords_file,
    output_dir,
    time_input=None,
    n_workers=None,
    min_size=10,
    downsample=None,
    length_method='skeleton',
    skip_curvature=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or cpu_count()

    print("=" * 70)
    print("GLOBAL FRONT PROCESSING")
    print("=" * 70)
    print(f"Fronts file : {fronts_file}")
    print(f"Coords file : {coords_file}")
    print(f"Output dir  : {output_dir}")
    print(f"Workers     : {n_workers}")
    print(f"Min size    : {min_size} pixels")
    if downsample:
        print(f"Downsample  : {downsample}x (testing mode)")
    print()

    fronts_global, lat_global, lon_global, time_str = load_fronts_data(
        fronts_file, coords_file, time_input=time_input, downsample=downsample
    )
    labeled, _ = label_and_filter(fronts_global, min_size=min_size)
    labeled_file = save_labeled_array(labeled, output_dir, time_str)
    properties, front_ids = extract_bboxes_and_ids(
        labeled, lat_global, lon_global, fronts_file
    )
    group_df, group_table_file = build_group_table(
        front_ids, properties, output_dir, time_str
    )
    results, total_time = run_parallel_geometry(
        group_df, labeled, lat_global, lon_global, time_str,
        n_workers=n_workers, length_method=length_method,
        skip_curvature=skip_curvature
    )
    df, parquet_file = save_results(
        results=results,
        fronts_file=fronts_file,
        coords_file=coords_file,
        fronts_global=fronts_global,
        lat_global=lat_global,
        lon_global=lon_global,
        time_str=time_str,
        output_dir=output_dir,
        min_size=min_size,
        n_workers=n_workers,
        total_time=total_time,
        downsample=downsample,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total fronts   : {len(df):,}")
    print(f"Processing time: {total_time / 60:.1f} minutes")
    print(f"Output files:")
    print(f"  Labeled array : {labeled_file}")
    print(f"  Group table   : {group_table_file}")
    print(f"  Properties PQ : {parquet_file}")
    print()
    print("✓ Ready for visualization in visualize_global_results.ipynb!")
    print("=" * 70)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(
        fronts_file=args.fronts_file,
        coords_file=args.coords_file,
        output_dir=args.output_dir,
        time_input=args.time,
        n_workers=args.n_workers,
        min_size=args.min_size,
        downsample=args.downsample,
        length_method=args.length_method,
        skip_curvature=args.skip_curvature,
    )