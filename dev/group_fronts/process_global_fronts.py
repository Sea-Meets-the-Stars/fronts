#!/usr/bin/env python3
"""
Global Front Processing Script
===============================

Process entire globe of fronts with parallelization for speed.
This script:
1. Loads global fronts and coordinates
2. Labels connected components
3. Calculates geometric properties in parallel
4. Saves results for later visualization

This is designed to run offline (e.g., on a server) and take 30min-2hr.
Results can then be visualized quickly in visualize_global_results.ipynb

Usage:
------
    python process_global_fronts.py --fronts_file /path/to/fronts.npy \
                                    --coords_file /path/to/coords.nc \
                                    --output_dir /path/to/output \
                                    --n_workers 8
                                    --skip_curvature
    
    python process_global_fronts.py --fronts_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/outputs/LLC4320_2012-11-09T12_00_00_bin_A.npy' \
                                    --coords_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/LLC_coords_lat_lon.nc' \
                                    --output_dir  '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/testing/skip_curvature/' \
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
from datetime import datetime
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from group_fronts import label, geometry, io

# Global variables for sharing large arrays across workers (fork mode)
# These are set before Pool creation and inherited by all workers via copy-on-write
# This avoids serializing 8GB+ arrays for every front!
_GLOBAL_LABELED = None
_GLOBAL_LAT = None
_GLOBAL_LON = None


def _process_with_bbox_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing to unpack (label, bbox, fixed_args) tuples.

    This must be at module level to be picklable for multiprocessing.

    CRITICAL: Uses global variables (_GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON)
    to access shared arrays without serialization overhead.

    Parameters
    ----------
    args_tuple : tuple
        (front_label, bbox, time_str, length_method, skip_curvature)
        Note: Large arrays are accessed from globals, not passed in tuple!

    Returns
    -------
    dict or None
        Front properties or None if processing failed
    """
    front_label, bbox, time_str, length_method, skip_curvature = args_tuple
    return process_single_front(
        front_id=front_label,
        labeled_array=_GLOBAL_LABELED,
        lat=_GLOBAL_LAT,
        lon=_GLOBAL_LON,
        time_str=time_str,
        bbox=bbox,
        length_method=length_method,
        skip_curvature=skip_curvature
    )


def process_single_front(front_id, labeled_array, lat, lon, time_str, bbox, length_method='skeleton', skip_curvature=False):
    """
    Process a single front to calculate geometric properties.

    This function is designed to be called in parallel.

    Parameters
    ----------
    front_id : int
        Front label ID
    labeled_array : np.ndarray
        Full labeled array (shared across all workers)
    lat : np.ndarray
        Latitude array
    lon : np.ndarray
        Longitude array
    time_str : str
        Time string
    bbox : tuple of slices
        Bounding box slices for this front (to optimize geometry calculations)
    length_method : str
        Method for length calculation: 'skeleton' (accurate, uses haversine)
        or 'perimeter' (DEPRECATED, inaccurate for non-uniform grids). Default: 'skeleton'
    skip_curvature : bool
        If True, skip curvature calculation to save time (only applies if length_method='skeleton

    Returns
    -------
    dict
        Geometric properties for this front
    """
    try:
        # Extract only bounding box region for this front
        labeled_bbox = labeled_array[bbox]
        lat_bbox = lat[bbox]
        lon_bbox = lon[bbox]

        # Create mask within bounding box
        mask = labeled_bbox == front_id

        props = {
            'label': front_id,
            'time': time_str,
            'npix': int(np.sum(mask))
        }

        # Bounding box coordinates (array indices i,j)
        # These allow fast extraction of this front's region later
        i_slice, j_slice = bbox
        props['bbox_i_min'] = int(i_slice.start)
        props['bbox_i_max'] = int(i_slice.stop)
        props['bbox_j_min'] = int(j_slice.start)
        props['bbox_j_max'] = int(j_slice.stop)

        # Centroid
        try:
            centroid_lat, centroid_lon = geometry.calculate_front_centroid(mask, lat_bbox, lon_bbox)
            props['centroid_lat'] = float(centroid_lat)
            props['centroid_lon'] = float(centroid_lon)
        except:
            props['centroid_lat'] = np.nan
            props['centroid_lon'] = np.nan

        # Extent (bounding box)
        try:
            # Extract lat/lon only where front exists (within bbox region)
            front_lats = lat_bbox[mask]
            front_lons = lon_bbox[mask]

            props['lat_min'] = float(front_lats.min())
            props['lat_max'] = float(front_lats.max())
            props['lon_min'] = float(front_lons.min())
            props['lon_max'] = float(front_lons.max())
        except:
            props['lat_min'] = np.nan
            props['lat_max'] = np.nan
            props['lon_min'] = np.nan
            props['lon_max'] = np.nan

        # Optimize: If using skeleton for both length AND curvature, compute skeleton once
        skeleton = None
        if length_method == 'skeleton':
            from skimage.morphology import skeletonize
            skeleton = skeletonize(mask)

        # Length
        try:
            # Use skeleton method (recommended for LLC4320 non-uniform grid)
            # Skeleton uses true haversine distances between points
            # If skeleton was pre-computed above, pass it in to avoid re-computing
            # Using bbox-local arrays instead of full global arrays
            props['length_km'] = float(geometry.calculate_front_length(
                mask, lat_bbox, lon_bbox, method=length_method, skeleton=skeleton
            ))
        except:
            props['length_km'] = np.nan

        # Branch points (junction count) - indicates structural complexity
        try:
            # If skeleton already computed above, reuse it; otherwise will compute internally
            props['num_branches'] = int(geometry.calculate_branch_points(mask, skeleton=skeleton))
        except:
            props['num_branches'] = 0

        # Orientation
        try:
            props['orientation'] = float(geometry.calculate_front_orientation(mask, lat_bbox, lon_bbox))
        except:
            props['orientation'] = np.nan

        # Curvature (optional - can be skipped to save time)
        if not skip_curvature:
            try:
                # If skeleton was pre-computed above, pass it in to avoid re-computing
                curv, curv_dir = geometry.calculate_front_curvature(
                    mask, lat_bbox, lon_bbox, skeleton=skeleton
                )
                props['mean_curvature'] = float(curv)
                props['curvature_direction'] = float(curv_dir)
            except:
                props['mean_curvature'] = np.nan
                props['curvature_direction'] = np.nan
        else:
            props['mean_curvature'] = np.nan
            props['curvature_direction'] = np.nan

        return props

    except Exception as e:
        print(f"  ERROR processing front {front_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process global fronts with parallelization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--fronts_file', required=True, help='Path to fronts .npy file')
    parser.add_argument('--coords_file', required=True, help='Path to coordinates .nc file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--time', default=None, help='Time string (YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--min_size', type=int, default=10,
                       help='Minimum front size in pixels (default: 10)')
    parser.add_argument('--downsample', type=int, default=None,
                       help='Downsample factor for testing (e.g., 2 = half resolution)')
    parser.add_argument('--length_method', type=str, default='skeleton',
                       choices=['skeleton', 'perimeter'],
                       help='Method for length calculation: skeleton (accurate, uses haversine) or perimeter (DEPRECATED, inaccurate for non-uniform grids). Default: skeleton')
    parser.add_argument('--skip_curvature', action='store_true',
                       help='Skip curvature calculation (saves ~50%% time when using skeleton method)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.n_workers or cpu_count()

    print("="*70)
    print("GLOBAL FRONT PROCESSING")
    print("="*70)
    print(f"Fronts file: {args.fronts_file}")
    print(f"Coords file: {args.coords_file}")
    print(f"Output dir: {output_dir}")
    print(f"Workers: {n_workers}")
    print(f"Min size: {args.min_size} pixels")
    if args.downsample:
        print(f"DOWNSAMPLING by factor {args.downsample} (for testing)")
    print()

    # Load data
    print("Loading global fronts...")
    t0 = time.time()
    fronts_global = np.load(args.fronts_file)

    if args.downsample:
        print(f"  Downsampling by {args.downsample}x...")
        fronts_global = fronts_global[::args.downsample, ::args.downsample]

    print(f"  Shape: {fronts_global.shape}")
    print(f"  Front pixels: {np.sum(fronts_global):,}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\nLoading coordinates...")
    t0 = time.time()
    ds_coords = xr.open_dataset(args.coords_file)
    lat_global = ds_coords['lat'].values if 'lat' in ds_coords else ds_coords['YC'].values
    lon_global = ds_coords['lon'].values if 'lon' in ds_coords else ds_coords['XC'].values

    if args.downsample:
        lat_global = lat_global[::args.downsample, ::args.downsample]
        lon_global = lon_global[::args.downsample, ::args.downsample]

    ds_coords.close()
    print(f"  Lat range: [{lat_global.min():.1f}, {lat_global.max():.1f}]")
    print(f"  Lon range: [{lon_global.min():.1f}, {lon_global.max():.1f}]")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Extract time from filename if not provided
    if args.time is None:
        import re
        pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})'
        match = re.search(pattern, args.fronts_file)
        if match:
            time_str_raw = match.group(1)
            time_str = time_str_raw.replace('_', ':')
        else:
            time_str = datetime.now().isoformat()
        print(f"\nExtracted time: {time_str}")
    else:
        time_str = args.time

    # Create filename-safe version of timestamp for output files
    time_str_safe = time_str.replace(':', '_').replace('-', '')
    print(f"Using timestamp for filenames: {time_str_safe}")

    # Label fronts
    print("\n" + "="*70)
    print("LABELING FRONTS")
    print("="*70)
    t0 = time.time()
    print("Running connected component labeling...")
    labeled, num_fronts = label.label_fronts(fronts_global, connectivity=2, return_num=True)
    print(f"  Initial fronts: {num_fronts:,}")
    print(f"  Labeled in {time.time()-t0:.1f}s")

    # Filter small fronts
    print(f"\nFiltering fronts (min_size={args.min_size})...")
    t0 = time.time()
    labeled = label.filter_fronts_by_size(labeled, min_size=args.min_size)
    front_labels = label.get_front_labels(labeled)
    num_filtered = len(front_labels)
    print(f"  After filtering: {num_filtered:,} fronts")
    print(f"  Filtered in {time.time()-t0:.1f}s")

    # Save labeled array
    print("\nSaving labeled array...")
    labeled_file = output_dir / f'labeled_fronts_global_{time_str_safe}.npy'
    np.save(labeled_file, labeled)
    print(f"  Saved to {labeled_file}")

    # Generate front IDs
    print("\nGenerating front IDs...")
    t0 = time.time()
    front_ids = label.generate_front_ids(labeled, lat_global, lon_global, time_str)
    print(f"  Generated {len(front_ids):,} IDs in {time.time()-t0:.1f}s")

    # Calculate geometric properties in parallel
    print("\n" + "="*70)
    print("CALCULATING GEOMETRIC PROPERTIES (PARALLEL)")
    print("="*70)
    print(f"Processing {num_filtered:,} fronts using {n_workers} workers...")
    print(f"Length method: {args.length_method}")
    print(f"Curvature: {'SKIPPED (faster)' if args.skip_curvature else 'CALCULATED'}")

    # Extract bounding boxes for all fronts
    # This allows us to work with small regions instead of full arrays!
    print("\nExtracting bounding boxes for all fronts...")
    t_bbox = time.time()
    from scipy import ndimage

    # Use label.py function to get bboxes as slices
    bbox_dict = label.get_front_bboxes(labeled)

    # Also get as explicit coordinates for saving to file
    bbox_coords = label.get_front_bboxes_as_coords(labeled)

    print(f"  Valid bounding boxes: {len(bbox_dict):,}")

    # CRITICAL OPTIMIZATION: Set global variables for fork mode sharing
    # Workers inherit these via copy-on-write (no serialization!)
    # Without this, we'd serialize 8GB+ arrays 113k times = massive bottleneck!
    print("\nSetting up shared arrays for workers...")
    global _GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON
    _GLOBAL_LABELED = labeled
    _GLOBAL_LAT = lat_global
    _GLOBAL_LON = lon_global
    print(f"  ✓ Arrays ready for sharing (labeled: {labeled.nbytes/1e9:.1f} GB)")

    t0 = time.time()

    # Create list of argument tuples for parallel processing
    # IMPORTANT: Only pass small args (label, bbox, etc.) - NOT the large arrays!
    # Large arrays accessed via globals to avoid serialization overhead
    # Each tuple: (front_label, bbox, time_str, length_method, skip_curvature)
    front_args = [
        (label, bbox_dict[label], time_str, args.length_method, args.skip_curvature)
        for label in front_labels if label in bbox_dict
    ]
    print(f"  Prepared {len(front_args):,} fronts for parallel processing")

    # Process in parallel
    print(f"\nStarting multiprocessing Pool with {n_workers} workers...")
    print(f"  PID of main process: {os.getpid()}")
    print(f"  Starting parallel processing now...")

    # Process in parallel
    with Pool(processes=n_workers) as pool:
        results = []
        completed = 0

        # Use imap_unordered for progress tracking with module-level wrapper function
        # IMPORTANT: Large chunksize reduces multiprocessing overhead when tasks are very fast
        # With 113k fronts and 20 workers: chunksize=500 means only ~226 work distributions
        optimal_chunksize = max(100, num_filtered // (n_workers * 10))
        print(f"  Using chunksize={optimal_chunksize} for optimal throughput")

        # Use imap_unordered for progress tracking
        for result in pool.imap_unordered(_process_with_bbox_wrapper, front_args, chunksize=optimal_chunksize):
            if result is not None:
                results.append(result)

            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = (num_filtered - completed) / rate / 60
                print(f"  Progress: {completed:,}/{num_filtered:,} "
                      f"({100*completed/num_filtered:.1f}%) - "
                      f"{rate:.1f} fronts/s - "
                      f"ETA: {remaining:.1f} min")

    total_time = time.time() - t0
    print(f"\n✓ Processed {len(results):,} fronts in {total_time/60:.1f} minutes")
    print(f"  Average: {total_time/len(results):.3f}s per front")

    # Convert to DataFrame
    print("\nCreating DataFrame...")
    df = pd.DataFrame(results)

    # Add front IDs
    df['front_id'] = df['label'].map(front_ids)

    # Reorder columns
    cols = ['label', 'front_id', 'time', 'npix',
            'bbox_i_min', 'bbox_i_max', 'bbox_j_min', 'bbox_j_max',
            'centroid_lat', 'centroid_lon',
            'length_km', 'orientation', 'num_branches',
            'lat_min', 'lat_max', 'lon_min', 'lon_max',
            'mean_curvature', 'curvature_direction']
    df = df[cols]

    print(f"  DataFrame: {len(df):,} rows × {len(df.columns)} columns")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Save as CSV
    csv_file = output_dir / f'global_front_properties_{time_str_safe}.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file}")
    print(f"  Size: {csv_file.stat().st_size / 1e6:.1f} MB")

    # Save as Parquet (more efficient for large datasets)
    parquet_file = output_dir / f'global_front_properties_{time_str_safe}.parquet'
    df.to_parquet(parquet_file, index=False)
    print(f"✓ Saved Parquet: {parquet_file}")
    print(f"  Size: {parquet_file.stat().st_size / 1e6:.1f} MB")

    # Save metadata
    metadata = {
        'fronts_file': args.fronts_file,
        'coords_file': args.coords_file,
        'time': time_str,
        'shape': fronts_global.shape,
        'num_fronts': len(df),
        'min_size': args.min_size,
        'processing_time_minutes': total_time / 60,
        'n_workers': n_workers,
        'downsample_factor': args.downsample,
        'lat_range': [float(lat_global.min()), float(lat_global.max())],
        'lon_range': [float(lon_global.min()), float(lon_global.max())],
        'timestamp': datetime.now().isoformat()
    }

    import json
    metadata_file = output_dir / f'metadata_{time_str_safe}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total fronts processed: {len(df):,}")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Output files:")
    print(f"  - Labeled array: {labeled_file}")
    print(f"  - Properties CSV: {csv_file}")
    print(f"  - Properties Parquet: {parquet_file}")
    print(f"  - Metadata: {metadata_file}")
    print()
    print("✓ Ready for visualization in visualize_global_results.ipynb!")
    print("="*70)


if __name__ == '__main__':
    main()
