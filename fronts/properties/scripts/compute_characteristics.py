"""Compute secondary / derived front properties from pre-saved gradient fields.

Reads NetCDF files produced by the upstream LLC4320 pipeline,
computes derived quantities defined in fronts.properties.characteristics, and
saves each result as a new NetCDF in the same directory using the same naming
convention:

    LLC4320_{timestamp}_{field}_v{version}.nc

Usage:
    python compute_characteristics.py <timestamp> [--version 1] [--input-dir ...] [--clobber]

Example:
    python compute_characteristics.py 2012-11-09T12_00_00
    python compute_characteristics.py 2012-11-09T12_00_00 --version 1 --input-dir /my/testing/dir
"""

import argparse
import os

import numpy as np
import xarray as xr

from fronts.llc import io as llc_io
from fronts.properties.characteristics import turner_angle


# Fields required as input for each derived quantity
REQUIRED_INPUTS = {
    'turner_angle': ['gradtheta2', 'gradsalt2', 'gradrho2'],
}


def load_field(timestamp: str, field: str, version: str, input_dir: str) -> np.ndarray:
    fpath = llc_io.derived_filename(timestamp, field, version=version, path=input_dir)
    print(f"  Loading {field} from {fpath}")
    with xr.open_dataset(fpath) as ds:
        return ds[field].values.squeeze()


def save_field(arr: np.ndarray, timestamp: str, field: str,
               version: str, output_dir: str) -> str:
    fpath = llc_io.derived_filename(timestamp, field, version=version, path=output_dir)
    ds = xr.Dataset({field: (('y', 'x'), arr)})
    ds.to_netcdf(fpath)
    print(f"  Saved {field} to {fpath}")
    return fpath


def run(timestamp: str, version: str = '1', input_dir: str = None, clobber: bool = False):
    # input_dir defaults to the standard derived path via llc_io if None
    if input_dir is None:
        input_dir = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'derived')
    output_dir = input_dir   # save alongside inputs

    # --- Turner angle ---
    out_file = llc_io.derived_filename(timestamp, 'turner_angle', version=version,
                                       path=output_dir)
    if os.path.isfile(out_file) and not clobber:
        print(f"turner_angle already exists and clobber is False. Skipping.")
    else:
        print("Computing turner_angle...")
        gradtheta2 = load_field(timestamp, 'gradtheta2', version, input_dir)
        gradsalt2  = load_field(timestamp, 'gradsalt2',  version, input_dir)
        gradrho2   = load_field(timestamp, 'gradrho2',   version, input_dir)
        tu_h = turner_angle(gradtheta2, gradsalt2, gradrho2)
        save_field(tu_h, timestamp, 'turner_angle', version, output_dir)

    # Add additional characteristics here as the module grows, e.g.:
    # out_file = llc_io.derived_filename(timestamp, 'next_property', ...)
    # if not os.path.isfile(out_file) or clobber:
    #     ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('timestamp', help="Snapshot timestamp, e.g. 2012-11-09T12_00_00")
    parser.add_argument('--version',   default='1',  help="Data version (default: 1)")
    parser.add_argument('--input-dir', default=None, help="Override input/output directory")
    parser.add_argument('--clobber',   action='store_true', help="Overwrite existing files")
    args = parser.parse_args()

    run(args.timestamp, version=args.version,
        input_dir=args.input_dir, clobber=args.clobber)
