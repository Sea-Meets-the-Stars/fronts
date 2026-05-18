#!/usr/bin/env python
"""
Convert LLC pixel (i, j) indices to (lat, lon) degrees.

Usage
-----
    python llc_ij_latlon.py <i> <j>

Examples
--------
    python llc_ij_latlon.py 1000 2000
"""

import sys
import argparse

from fronts.llc.coords import ij_to_latlon


def parser(options=None):
    # Mirrors the argparse style used in front_property_viewer.py
    p = argparse.ArgumentParser(
        description='Convert LLC pixel (i, j) indices to (lat, lon) degrees.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('i', type=int, help='Column index (x) into the LLC grid')
    p.add_argument('j', type=int, help='Row index (y) into the LLC grid')

    if options is None:
        return p.parse_args()
    return p.parse_args(options)


def main(pargs):
    # Look up the (lat, lon) for the single pixel
    lat, lon = ij_to_latlon(pargs.i, pargs.j)

    # Print a compact, parsable summary plus a human-readable line
    print(f'i={pargs.i}  j={pargs.j}  ->  lat={lat:.6f}  lon={lon:.6f}')


if __name__ == '__main__':
    main(parser())
