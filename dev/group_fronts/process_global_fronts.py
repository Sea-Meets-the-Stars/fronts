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
    
    python process_global_fronts.py --fronts_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/global/LLC4320_2012-11-09T12_00_00_fronts.npy' \
                                    --coords_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/LLC_coords_lat_lon.nc' \
                                    --output_dir  '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/group_fronts/' \
                                    --n_workers 8

Author: Generated for Lauren's front characterization project
Date: 2026-02-13
"""

import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from group_fronts import label, geometry, io


def process_single_front(front_id, labeled_array, lat, lon, time_str):
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

    Returns
    -------
    dict
        Geometric properties for this front
    """
    try:
        # Create mask for this front
        mask = labeled_array == front_id

        props = {
            'label': front_id,
            'time': time_str,
            'npix': int(np.sum(mask))
        }

        # Centroid
        try:
            centroid_lat, centroid_lon = geometry.calculate_front_centroid(mask, lat, lon)
            props['centroid_lat'] = float(centroid_lat)
            props['centroid_lon'] = float(centroid_lon)
        except:
            props['centroid_lat'] = np.nan
            props['centroid_lon'] = np.nan

        # Extent (bounding box)
        try:
            extent = geometry.calculate_front_extent(mask, lat, lon)
            props['lat_min'] = float(extent['lat_min'])
            props['lat_max'] = float(extent['lat_max'])
            props['lon_min'] = float(extent['lon_min'])
            props['lon_max'] = float(extent['lon_max'])
        except:
            props['lat_min'] = np.nan
            props['lat_max'] = np.nan
            props['lon_min'] = np.nan
            props['lon_max'] = np.nan

        # Length
        try:
            props['length_km'] = float(geometry.calculate_front_length(mask, lat, lon, method='skeleton'))
        except:
            props['length_km'] = np.nan

        # Orientation
        try:
            props['orientation'] = float(geometry.calculate_front_orientation(mask, lat, lon))
        except:
            props['orientation'] = np.nan

        # Curvature
        try:
            curv, curv_dir = geometry.calculate_front_curvature(mask, lat, lon)
            props['mean_curvature'] = float(curv)
            props['curvature_direction'] = float(curv_dir)
        except:
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
    labeled_file = output_dir / 'labeled_fronts_global.npy'
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
    print("This may take 30min-2hr depending on data size and CPU...")

    t0 = time.time()

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_front,
        labeled_array=labeled,
        lat=lat_global,
        lon=lon_global,
        time_str=time_str
    )

    # Process in parallel
    with Pool(processes=n_workers) as pool:
        results = []
        completed = 0

        # Use imap_unordered for progress tracking
        for result in pool.imap_unordered(process_func, front_labels, chunksize=10):
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
    cols = ['label', 'front_id', 'time', 'npix', 'centroid_lat', 'centroid_lon',
            'length_km', 'orientation', 'lat_min', 'lat_max', 'lon_min', 'lon_max',
            'mean_curvature', 'curvature_direction']
    df = df[cols]

    print(f"  DataFrame: {len(df):,} rows × {len(df.columns)} columns")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Save as CSV
    csv_file = output_dir / 'global_front_properties.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file}")
    print(f"  Size: {csv_file.stat().st_size / 1e6:.1f} MB")

    # Save as Parquet (more efficient for large datasets)
    parquet_file = output_dir / 'global_front_properties.parquet'
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
    metadata_file = output_dir / 'metadata.json'
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
