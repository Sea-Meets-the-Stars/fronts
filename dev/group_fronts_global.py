#!/usr/bin/env python3
"""
Global Front Processing Script
===============================

Process entire globe of fronts with parallelization for speed.

Pipeline
--------
1. Load global fronts and coordinates
2. Label connected components and filter by size
3. Extract bounding boxes (x0, y0, x1, y1) for each front
4. Write group table (label, name, x0, y0, x1, y1) to disk via io.to_group_table
5. Pool workers on cutouts: each worker extracts its own cutout using
   the bounding box coords and calls geometry.process_single_front
6. Save results (CSV, Parquet) and metadata

Designed to run offline (e.g., on a server) and take 30min-2hr.
Results can then be visualized in visualize_global_results.ipynb.

Usage
-----
    python group_fronts_global.py --fronts_file /path/to/fronts.npy \\
                                    --coords_file /path/to/coords.nc \\
                                    --output_dir /path/to/output \\
                                    --n_workers 8

    python group_fronts_global.py --fronts_file /path/to/fronts.npy \\
                                    --coords_file /path/to/coords.nc \\
                                    --output_dir /path/to/output \\
                                    --n_workers 2 \\
                                    --skip_curvature
    
    python group_fronts_global.py --fronts_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/outputs/LLC4320_2012-11-09T12_00_00_bin_A.npy' \
                                    --coords_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/LLC_coords_lat_lon.nc' \
                                    --output_dir  '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/testing/pr/' \
                                    --n_workers 2 \
                                    --skip_curvature
"""

import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import sys
import time
import re
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from fronts.properties import group_labels, io, geometry

# ---------------------------------------------------------------------------
# Module-level globals for copy-on-write sharing across forked workers.
# Set in main() before Pool creation; inherited by workers via fork.
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


def main():
    parser = argparse.ArgumentParser(
        description="Process global fronts with parallelization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--fronts_file', required=True,
                        help='Path to fronts .npy file')
    parser.add_argument('--coords_file', required=True,
                        help='Path to coordinates .nc file')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--time', default=None,
                        help='Time string (YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--min_size', type=int, default=10,
                        help='Minimum front size in pixels (default: 10)')
    parser.add_argument('--downsample', type=int, default=None,
                        help='Downsample factor for testing (e.g. 2 = half resolution)')
    parser.add_argument('--length_method', type=str, default='skeleton',
                        choices=['skeleton', 'perimeter'],
                        help='Method for length calculation (default: skeleton)')
    parser.add_argument('--skip_curvature', action='store_true',
                        help='Skip curvature calculation (saves ~50%% time)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers or cpu_count()

    print("=" * 70)
    print("GLOBAL FRONT PROCESSING")
    print("=" * 70)
    print(f"Fronts file : {args.fronts_file}")
    print(f"Coords file : {args.coords_file}")
    print(f"Output dir  : {output_dir}")
    print(f"Workers     : {n_workers}")
    print(f"Min size    : {args.min_size} pixels")
    if args.downsample:
        print(f"Downsample  : {args.downsample}x (testing mode)")
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading global fronts...")
    t0 = time.time()
    fronts_global = np.load(args.fronts_file)

    if args.downsample:
        fronts_global = fronts_global[::args.downsample, ::args.downsample]

    print(f"  Shape: {fronts_global.shape}")
    print(f"  Front pixels: {np.sum(fronts_global):,}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\nLoading coordinates...")
    t0 = time.time()
    ds_coords = xr.open_dataset(args.coords_file)
    lat_global = (ds_coords['lat'].values if 'lat' in ds_coords
                  else ds_coords['YC'].values)
    lon_global = (ds_coords['lon'].values if 'lon' in ds_coords
                  else ds_coords['XC'].values)

    if args.downsample:
        lat_global = lat_global[::args.downsample, ::args.downsample]
        lon_global = lon_global[::args.downsample, ::args.downsample]

    ds_coords.close()
    print(f"  Lat range: [{lat_global.min():.1f}, {lat_global.max():.1f}]")
    print(f"  Lon range: [{lon_global.min():.1f}, {lon_global.max():.1f}]")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Extract time from filename if not provided
    if args.time is None:
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})',
                          args.fronts_file)
        if match:
            time_str = match.group(1).replace('_', ':')
        else:
            time_str = datetime.now().isoformat()
        print(f"\nExtracted time: {time_str}")
    else:
        time_str = args.time

    # Filename-safe timestamp for output files
    time_str_safe = time_str.replace(':', '_').replace('-', '')
    print(f"Using timestamp for filenames: {time_str_safe}")

    # ------------------------------------------------------------------
    # 2. Label fronts
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LABELING FRONTS")
    print("=" * 70)
    t0 = time.time()
    labeled, num_fronts = group_labels.label_fronts(
        fronts_global, connectivity=2, return_num=True
    )
    print(f"  Initial fronts: {num_fronts:,}")
    print(f"  Labeled in {time.time() - t0:.1f}s")

    print(f"\nFiltering fronts (min_size={args.min_size})...")
    t0 = time.time()
    labeled = group_labels.filter_fronts_by_size(labeled, min_size=args.min_size)
    front_labels = group_labels.get_front_labels(labeled)
    print(f"  After filtering: {len(front_labels):,} fronts")
    print(f"  Filtered in {time.time() - t0:.1f}s")

    # Save labeled array
    labeled_file = output_dir / f'labeled_fronts_global_{time_str_safe}.npy'
    np.save(labeled_file, labeled)
    print(f"\nSaved labeled array: {labeled_file}")

    # ------------------------------------------------------------------
    # 3. Extract bounding boxes and generate front IDs
    # ------------------------------------------------------------------
    print("\nExtracting bounding boxes...")
    t0 = time.time()
    bbox_coords = group_labels.get_front_bboxes_as_coords(labeled)
    print(f"  Valid bboxes: {len(bbox_coords):,}  ({time.time() - t0:.1f}s)")

    print("Generating front IDs...")
    t0 = time.time()
    front_ids = group_labels.generate_front_ids(labeled, lat_global, lon_global, time_str)
    print(f"  Generated {len(front_ids):,} IDs  ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Write group table (label, name, x0, y0, x1, y1) to disk
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WRITING GROUP TABLE")
    print("=" * 70)
    group_table_file = output_dir / f'group_table_{time_str_safe}.parquet'
    group_df = io.to_group_table(front_ids, bbox_coords, group_table_file)

    # ------------------------------------------------------------------
    # 5. Pool on cutouts
    #
    # Set module-level globals in characterize before forking so workers
    # inherit the arrays via copy-on-write (no serialization overhead).
    # Each worker uses its (y0, y1, x0, x1) from the group table to
    # extract its own small cutout, then calls process_single_front on it.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CALCULATING GEOMETRIC PROPERTIES (PARALLEL)")
    print("=" * 70)
    print(f"Processing {len(group_df):,} fronts using {n_workers} workers...")
    print(f"Length method : {args.length_method}")
    print(f"Curvature     : {'SKIPPED' if args.skip_curvature else 'calculated'}")

    print("\nSetting up shared arrays for workers...")
    global _GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON
    _GLOBAL_LABELED = labeled
    _GLOBAL_LAT     = lat_global
    _GLOBAL_LON     = lon_global
    print(f"  Arrays ready (labeled: {labeled.nbytes / 1e9:.1f} GB)")

    # Build args from group table — only small scalars, no large arrays
    front_args = [
        (row.label, row.name, row.y0, row.y1, row.x0, row.x1,
         time_str, args.length_method, args.skip_curvature)
        for row in group_df.itertuples()
    ]

    t0 = time.time()
    num_filtered = len(front_args)
    optimal_chunksize = max(100, num_filtered // (n_workers * 10))
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
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = (num_filtered - completed) / rate / 60
                print(f"  {completed:,}/{num_filtered:,} "
                      f"({100 * completed / num_filtered:.1f}%) — "
                      f"{rate:.1f} fronts/s — "
                      f"ETA: {remaining:.1f} min")

    total_time = time.time() - t0
    print(f"\n✓ Processed {len(results):,} fronts in {total_time / 60:.1f} min")
    if results:
        print(f"  Average: {total_time / len(results):.3f}s per front")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['label', 'name', 'time', 'npix',
            'y0', 'y1', 'x0', 'x1',
            'centroid_lat', 'centroid_lon',
            'length_km', 'orientation', 'num_branches',
            'lat_min', 'lat_max', 'lon_min', 'lon_max',
            'mean_curvature', 'curvature_direction']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    csv_file = output_dir / f'global_front_properties_{time_str_safe}.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ CSV     : {csv_file}  ({csv_file.stat().st_size / 1e6:.1f} MB)")

    parquet_file = output_dir / f'global_front_properties_{time_str_safe}.parquet'
    df.to_parquet(parquet_file, index=False)
    print(f"✓ Parquet : {parquet_file}  ({parquet_file.stat().st_size / 1e6:.1f} MB)")

    import json
    metadata = {
        'fronts_file': args.fronts_file,
        'coords_file': args.coords_file,
        'time': time_str,
        'shape': list(fronts_global.shape),
        'num_fronts': len(df),
        'min_size': args.min_size,
        'processing_time_minutes': total_time / 60,
        'n_workers': n_workers,
        'downsample_factor': args.downsample,
        'lat_range': [float(lat_global.min()), float(lat_global.max())],
        'lon_range': [float(lon_global.min()), float(lon_global.max())],
        'timestamp': datetime.now().isoformat()
    }
    metadata_file = output_dir / f'metadata_{time_str_safe}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata: {metadata_file}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total fronts   : {len(df):,}")
    print(f"Processing time: {total_time / 60:.1f} minutes")
    print(f"Output files:")
    print(f"  Labeled array : {labeled_file}")
    print(f"  Group table   : {group_table_file}")
    print(f"  Properties CSV: {csv_file}")
    print(f"  Properties PQ : {parquet_file}")
    print(f"  Metadata      : {metadata_file}")
    print()
    print("✓ Ready for visualization in visualize_global_results.ipynb!")
    print("=" * 70)


if __name__ == '__main__':
    main()
