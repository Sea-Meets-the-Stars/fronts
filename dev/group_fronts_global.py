#!/usr/bin/env python3
"""
CLI wrapper for fronts.properties.algorithms.group_fronts.

Usage
-----
    python dev/group_fronts_global.py \
        --fronts_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/outputs/LLC4320_2012-11-09T12_00_00_v1_bin_A.npy' \
        --coords_file '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/coords/LLC_coords_lat_lon.nc' \
        --output_dir  '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/group_fronts/v1/' \
        --n_workers 8 \
        --skip_curvature
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from fronts.properties import algorithms


def get_parser():
    parser = argparse.ArgumentParser(description="Group and characterize global fronts")
    parser.add_argument('--fronts_file', required=True, help='Binary front .npy file')
    parser.add_argument('--coords_file', required=True, help='Coordinates .nc file')
    parser.add_argument('--output_dir',  required=True, help='Output directory')
    parser.add_argument('--n_workers',   type=int, default=None,
                        help='Parallel workers (default: CPU count)')
    parser.add_argument('--skip_curvature', action='store_true',
                        help='Skip curvature calculation (~50%% faster)')
    return parser


def main(fronts_file, coords_file, output_dir,
         n_workers=None, skip_curvature=False):
    fronts_binary = np.load(fronts_file)
    ds = xr.open_dataset(coords_file)
    lat = ds['lat'].values if 'lat' in ds else ds['YC'].values
    lon = ds['lon'].values if 'lon' in ds else ds['XC'].values
    ds.close()

    algorithms.group_fronts(
        fronts_binary, lat, lon,
        fronts_file=fronts_file,
        output_dir=output_dir,
        n_workers=n_workers,
        skip_curvature=skip_curvature,
    )


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(
        fronts_file=args.fronts_file,
        coords_file=args.coords_file,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        skip_curvature=args.skip_curvature,
    )
