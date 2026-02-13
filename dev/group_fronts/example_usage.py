#!/usr/bin/env python
"""
Example Usage Script

Demonstrates how to use the group_fronts module to label and characterize fronts.
This script shows a complete workflow from binary front detection to saving results.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import group_fronts
sys.path.insert(0, str(Path(__file__).parent.parent))

from group_fronts import label, geometry, io


def create_synthetic_fronts():
    """
    Create synthetic binary front data for testing.

    Returns
    -------
    front_binary : np.ndarray
        Binary array with synthetic fronts
    lat : np.ndarray
        Latitude coordinates (1D)
    lon : np.ndarray
        Longitude coordinates (1D)
    time : str
        Timestamp
    """
    # Create coordinate arrays
    lat = np.linspace(30.0, 40.0, 100)  # 30°N to 40°N
    lon = np.linspace(-130.0, -120.0, 100)  # 130°W to 120°W
    time = '2020-01-15T00:00:00'

    # Create 2D grids
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Create synthetic fronts
    front_binary = np.zeros((100, 100), dtype=bool)

    # Front 1: Diagonal line in upper left
    for i in range(20):
        front_binary[i, i:i+3] = True

    # Front 2: Curved line in middle
    for i in range(40, 60):
        j = int(50 + 10 * np.sin((i - 40) * 0.3))
        front_binary[i, max(0, j-1):min(100, j+2)] = True

    # Front 3: Small cluster in lower right
    front_binary[80:85, 85:90] = True

    # Front 4: Very small (will test filtering)
    front_binary[10, 80] = True

    return front_binary, lat, lon, time


def main():
    """
    Main example workflow.
    """
    print("=" * 70)
    print("Front Grouping and Characterization Example")
    print("=" * 70)

    # Step 1: Create or load binary front data
    print("\n[Step 1] Creating synthetic front data...")
    front_binary, lat, lon, time = create_synthetic_fronts()
    print(f"  - Data shape: {front_binary.shape}")
    print(f"  - Number of front pixels: {np.sum(front_binary)}")
    print(f"  - Lat range: {lat[0]:.1f}° to {lat[-1]:.1f}°")
    print(f"  - Lon range: {lon[0]:.1f}° to {lon[-1]:.1f}°")
    print(f"  - Time: {time}")

    # Step 2: Label connected fronts
    print("\n[Step 2] Labeling connected fronts...")
    labeled_fronts, num_fronts = label.label_fronts(
        front_binary,
        connectivity=2,  # 8-connected
        return_num=True
    )
    print(f"  - Number of fronts detected: {num_fronts}")
    print(f"  - Labels: {label.get_front_labels(labeled_fronts)}")

    # Step 3: Filter small fronts
    print("\n[Step 3] Filtering small fronts (min_size=5)...")
    labeled_fronts_filtered = label.filter_fronts_by_size(
        labeled_fronts,
        min_size=5
    )
    num_fronts_filtered = len(label.get_front_labels(labeled_fronts_filtered))
    print(f"  - Fronts after filtering: {num_fronts_filtered}")
    print(f"  - Removed {num_fronts - num_fronts_filtered} small fronts")

    # Step 4: Generate unique front IDs
    print("\n[Step 4] Generating unique front IDs...")
    front_ids = label.generate_front_ids(
        labeled_fronts_filtered,
        lat,
        lon,
        time
    )
    print("  - Front IDs:")
    for lbl, fid in front_ids.items():
        print(f"    Label {lbl}: {fid}")

    # Step 5: Calculate geometric properties
    print("\n[Step 5] Calculating geometric properties...")
    properties = geometry.calculate_all_geometric_properties(
        labeled_fronts_filtered,
        lat,
        lon,
        time,
        include_curvature=True,
        length_method='perimeter'
    )

    print("\n  - Front Properties Summary:")
    print(f"  {'Label':<8} {'ID':<30} {'Pixels':<8} {'Length(km)':<12} {'Orientation':<12}")
    print("  " + "-" * 75)
    for lbl, props in properties.items():
        fid = front_ids.get(lbl, 'N/A')
        npix = props['npix']
        length = props['length_km']
        orient = props['orientation']
        print(f"  {lbl:<8} {fid:<30} {npix:<8} {length:<12.1f} {orient:<12.1f}")

    # Step 6: Save results in multiple formats
    print("\n[Step 6] Saving results...")
    output_dir = Path(__file__).parent / 'example_output'
    output_files = io.save_all(
        labeled_fronts_filtered,
        properties,
        lat,
        lon,
        time,
        front_ids,
        output_dir,
        base_name='example_fronts',
        formats=['netcdf', 'csv', 'parquet']
    )

    print("\n  - Output files:")
    for fmt, path in output_files.items():
        print(f"    {fmt}: {path}")

    # Step 7: Demonstrate loading data back
    print("\n[Step 7] Testing data loading...")

    # Load from NetCDF
    loaded_labeled, loaded_lat, loaded_lon, loaded_time, loaded_ids = io.from_netcdf(
        output_files['netcdf']
    )
    print(f"  - Loaded from NetCDF: shape={loaded_labeled.shape}, {len(loaded_ids)} fronts")

    # Load from CSV
    df_csv = io.from_csv(output_files['csv'])
    print(f"  - Loaded from CSV: {len(df_csv)} rows, {len(df_csv.columns)} columns")

    # Load from Parquet
    df_parquet = io.from_parquet(output_files['parquet'])
    print(f"  - Loaded from Parquet: {len(df_parquet)} rows, {len(df_parquet.columns)} columns")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Examine output files to verify results")
    print("  2. Integrate with your actual front detection workflow")
    print("  3. Add additional characterization modules (fields, dynamics, etc.)")
    print("=" * 70)


if __name__ == '__main__':
    main()
