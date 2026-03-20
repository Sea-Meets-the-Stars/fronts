#!/usr/bin/env python3
"""
Global Front Co-location
========================

Co-locate a labeled front array with one or more mapped property fields and
save per-front statistics as a parquet file.

This is a thin CLI wrapper around fronts.properties.algorithms.colocate_fronts().
For step-by-step control from Python, call that function directly.

Usage
-----
python dev/colocate_fronts_global.py \
    --labeled_file  '/path/to/labeled_fronts_global_20121109T12_00_00.npy' \
    --fronts_file   '/path/to/LLC4320_2012-11-09T12_00_00_bin_A.npy' \
    --property relative_vorticity '/path/to/relative_vorticity.npy' \
    --property strain_n            '/path/to/strain_n.npy' \
    --output_dir    '/path/to/output/' \
    --dilation_radius 5 \
    --stats mean std median \
    --percentiles 10 90

Notes
-----
- --property can be repeated for as many fields as needed.
- --fronts_file is used only to extract the timestamp for the output filename;
  it does not need to be loaded again.
- Property arrays must be .npy files with the same shape as the labeled array.
"""

import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from fronts.properties import algorithms


def get_parser():
    parser = argparse.ArgumentParser(
        description="Co-locate labeled fronts with property fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--labeled_file', required=True,
        help="Path to labeled fronts .npy file (output of group_fronts step)"
    )
    parser.add_argument(
        '--fronts_file', required=True,
        help="Path to original binary fronts .npy file (used for timestamp only)"
    )
    parser.add_argument(
        '--property', nargs=2, metavar=('NAME', 'PATH'),
        action='append', required=True,
        help="Property name and path to .npy file. Repeat for each field."
    )
    parser.add_argument(
        '--output_dir', required=True,
        help="Directory to save colocation parquet"
    )
    parser.add_argument(
        '--stats', nargs='+', default=None,
        metavar='STAT',
        help="Statistics to compute: mean std median min max count (default: mean std median)"
    )
    parser.add_argument(
        '--percentiles', nargs='+', type=float, default=None,
        metavar='PCT',
        help="Percentiles to compute, e.g. 10 25 75 90"
    )
    parser.add_argument(
        '--min_npix', type=int, default=1,
        help="Minimum front size in pixels (default: 1)"
    )
    parser.add_argument(
        '--nan_policy', choices=['omit', 'propagate'], default='omit',
        help="How to handle NaN values, e.g. land pixels (default: omit)"
    )
    parser.add_argument(
        '--dilation_radius', type=int, default=0,
        help="Pixels to dilate each front before computing stats (default: 0)"
    )
    return parser


def main(
    labeled_file,
    fronts_file,
    property_files,
    output_dir,
    stats=None,
    percentiles=None,
    min_npix=1,
    nan_policy='omit',
    dilation_radius=0,
):
    labeled = np.load(labeled_file)
    property_arrays = {
        name: np.load(path) for name, path in property_files.items()
    }

    algorithms.colocate_fronts(
        labeled=labeled,
        property_arrays=property_arrays,
        fronts_file=fronts_file,
        output_dir=output_dir,
        stats=stats,
        percentiles=percentiles,
        min_npix=min_npix,
        nan_policy=nan_policy,
        dilation_radius=dilation_radius,
    )


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(
        labeled_file=args.labeled_file,
        fronts_file=args.fronts_file,
        property_files=dict(args.property),   # [['name', 'path'], ...] -> dict
        output_dir=args.output_dir,
        stats=args.stats,
        percentiles=args.percentiles,
        min_npix=args.min_npix,
        nan_policy=args.nan_policy,
        dilation_radius=args.dilation_radius,
    )
