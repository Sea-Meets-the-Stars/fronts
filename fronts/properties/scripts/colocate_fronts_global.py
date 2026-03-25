#!/usr/bin/env python3
"""
Global Front Co-location
========================

Co-locate a labeled front array with one or more mapped property fields and
save per-front statistics as a parquet file.

Property .nc files are located automatically from --property_dir using the
timestamp and version extracted from --fronts_file:

    LLC4320_{timestamp}_{property_name}_{version}.nc

This is a thin CLI wrapper around fronts.properties.algorithms.colocate_fronts().
For step-by-step control from Python, call that function directly.

Usage
-----
python dev/colocate_fronts_global.py \
    --labeled_file  '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/group_fronts/v1/labeled_fronts_global_20121109T12_00_00_v1_bin_A.npy' \
    --fronts_file   '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/outputs/LLC4320_2012-11-09T12_00_00_v1_bin_A.npy' \
    --property_dir  '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/derived/' \
    --properties    coriolis_f divergence Eta frontogenesis_tendency gradb2 gradeta2 gradrho2 gradsalt2 gradtheta2 okubo_weiss relative_vorticity rossby_number Salt strain_mag strain_n strain_s Theta ug vg U V W \
    --output_dir    '/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/group_fronts/v1/' \
    --dilation_radius 5 \
    --stats mean std median \
    --percentiles 10 25 75 90
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
        help="Path to binary fronts .npy file; timestamp and version extracted from filename"
    )
    parser.add_argument(
        '--property_dir', required=True,
        help="Directory containing property .nc files"
    )
    parser.add_argument(
        '--properties', nargs='+', required=True,
        metavar='NAME',
        help="Property names to co-locate, e.g. relative_vorticity strain_n"
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
    property_dir,
    properties,
    output_dir,
    stats=None,
    percentiles=None,
    min_npix=1,
    nan_policy='omit',
    dilation_radius=0,
):
    labeled = np.load(labeled_file)

    algorithms.colocate_fronts(
        labeled=labeled,
        property_names=properties,
        property_dir=property_dir,
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
        property_dir=args.property_dir,
        properties=args.properties,
        output_dir=args.output_dir,
        stats=args.stats,
        percentiles=args.percentiles,
        min_npix=args.min_npix,
        nan_policy=args.nan_policy,
        dilation_radius=args.dilation_radius,
    )
