#!/usr/bin/env python
"""
Convert (lat, lon) in degrees to nearest LLC pixel (i, j) indices.

Usage
-----
    python llc_latlon_ij.py <lat> <lon>

Examples
--------
    python llc_latlon_ij.py 36.0 -122.0
"""

import sys
import argparse

from fronts.llc.coords import latlon_to_ij


def parser(options=None):
    # Mirrors the argparse style used in llc_ij_latlon.py
    p = argparse.ArgumentParser(
        description='Convert (lat, lon) in degrees to nearest LLC pixel (i, j).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('lat', type=float, help='Latitude in degrees')
    p.add_argument('lon', type=float, help='Longitude in degrees')

    if options is None:
        return p.parse_args()
    return p.parse_args(options)


def main(pargs):
    # Look up the nearest LLC pixel for the single query
    i, j = latlon_to_ij(pargs.lat, pargs.lon)

    # Human-readable, parsable summary
    print(f'lat={pargs.lat:.6f}  lon={pargs.lon:.6f}  ->  i={i}  j={j}')


if __name__ == '__main__':
    main(parser())
