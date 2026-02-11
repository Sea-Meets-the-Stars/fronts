#!/usr/bin/env python
"""
Compute the squared divergence of buoyancy (Divb2) from LLC4320 data.

This script loads LLC4320 model data, computes the squared magnitude of the
buoyancy gradient field, and saves the result to a NetCDF file.

Usage:
    python global_divb2.py <input_nc_file> [--output <output_file>] [--dx <grid_spacing>]

Example:
    python global_divb2.py LLC4320_2011-09-13T00_00_00.nc --output divb2_output.nc --dx 2.0
"""

import argparse
import sys
import os
import numpy as np
import xarray as xr

# Import from wrangler repository
try:
    from wrangler.preproc.pp_ogcm import calc_gradb2
    from wrangler.ogcm import llc as wr_llc
except ImportError as e:
    print(f"ERROR: Failed to import from wrangler package: {e}")
    print("Make sure the wrangler repository is in your Python path.")
    sys.exit(1)

try:
    from gsw import density
except ImportError:
    print("ERROR: gsw package not found. Please install it: pip install gsw")
    sys.exit(1)


def compute_divb2(theta: np.ndarray, salt: np.ndarray, dx: float = 2.0,
                  ref_rho: float = 1025., g: float = 0.0098,
                  norm_by_b: bool = False):
    """
    Compute the squared divergence of buoyancy (|grad b|^2).

    Uses the calc_gradb2 function from wrangler.preproc.pp_ogcm.

    Parameters:
        theta (np.ndarray): Sea surface temperature field (degC)
        salt (np.ndarray): Sea surface salinity field (PSU)
        dx (float): Grid spacing in km (default: 2.0)
        ref_rho (float): Reference density in kg/m^3 (default: 1025.)
        g (float): Acceleration due to gravity in km/s^2 (default: 0.0098)
        norm_by_b (bool): Normalize by median buoyancy (default: False)

    Returns:
        np.ndarray: Squared magnitude of buoyancy gradient
    """
    # Use wrangler's calc_gradb2 function
    divb2 = calc_gradb2(theta, salt, ref_rho=ref_rho, g=g, dx=dx,
                        norm_by_b=norm_by_b)

    return divb2


def main():
    """Main function to process LLC4320 data and compute Divb2."""

    parser = argparse.ArgumentParser(
        description='Compute squared divergence of buoyancy from LLC4320 data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file', type=str,
                        help='Path to input LLC4320 NetCDF file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output NetCDF file (default: input_divb2.nc)')
    parser.add_argument('--dx', type=float, default=2.0,
                        help='Grid spacing in km')
    parser.add_argument('--ref-rho', type=float, default=1025.,
                        help='Reference density in kg/m^3')
    parser.add_argument('--g', type=float, default=0.0098,
                        help='Acceleration due to gravity in km/s^2')
    parser.add_argument('--norm-by-b', action='store_true',
                        help='Normalize by median buoyancy')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    # Set output filename if not provided
    if args.output is None:
        base = os.path.splitext(args.input_file)[0]
        args.output = f"{base}_divb2.nc"

    print(f"Loading LLC4320 data from: {args.input_file}")

    # Load the dataset
    try:
        ds = xr.open_dataset(args.input_file)
    except Exception as e:
        print(f"ERROR: Failed to load input file: {e}")
        sys.exit(1)

    # Check for required fields
    if 'Theta' not in ds:
        print("ERROR: 'Theta' field not found in dataset")
        sys.exit(1)
    if 'Salt' not in ds:
        print("ERROR: 'Salt' field not found in dataset")
        sys.exit(1)

    print(f"Dataset dimensions: {dict(ds.dims)}")
    print(f"Computing Divb2 with dx={args.dx} km, ref_rho={args.ref_rho} kg/m^3, g={args.g} km/s^2")
    if args.norm_by_b:
        print("Normalizing by median buoyancy")

    # Extract the fields
    theta = ds.Theta.values
    salt = ds.Salt.values

    print(f"Theta shape: {theta.shape}, range: [{np.nanmin(theta):.2f}, {np.nanmax(theta):.2f}] degC")
    print(f"Salt shape: {salt.shape}, range: [{np.nanmin(salt):.2f}, {np.nanmax(salt):.2f}] PSU")

    # Compute Divb2 using wrangler's calc_gradb2
    print("Computing squared divergence of buoyancy using wrangler.preproc.pp_ogcm.calc_gradb2...")
    divb2 = compute_divb2(theta, salt, dx=args.dx,
                          ref_rho=args.ref_rho, g=args.g,
                          norm_by_b=args.norm_by_b)

    print(f"Divb2 computed. Shape: {divb2.shape}, range: [{np.nanmin(divb2):.2e}, {np.nanmax(divb2):.2e}]")

    # Create output dataset
    print(f"Creating output dataset...")

    # Create a new dataset with the same dimensions as input
    out_ds = xr.Dataset()

    # Add Divb2 field with the same dimensions as Theta
    out_ds['Divb2'] = xr.DataArray(
        divb2.astype(np.float32),
        dims=ds.Theta.dims,
        coords={dim: ds.Theta.coords[dim] for dim in ds.Theta.dims if dim in ds.Theta.coords},
        attrs={
            'long_name': 'Squared magnitude of buoyancy gradient',
            'units': '(km/s^2)^2 / km^2',
            'description': 'Computed as |grad b|^2 where b = g*rho/rho_ref',
            'dx': args.dx,
            'ref_rho': args.ref_rho,
            'g': args.g
        }
    )

    # Copy over coordinate variables if they exist
    for coord in ['XC', 'YC', 'XG', 'YG', 'time', 'lat', 'lon']:
        if coord in ds:
            out_ds[coord] = ds[coord]

    # Add global attributes
    out_ds.attrs['title'] = 'Squared divergence of buoyancy (Divb2)'
    out_ds.attrs['source'] = f'Computed from {os.path.basename(args.input_file)}'
    out_ds.attrs['grid_spacing_km'] = args.dx
    out_ds.attrs['reference_density_kgm3'] = args.ref_rho
    out_ds.attrs['gravity_kms2'] = args.g

    # Save to NetCDF
    print(f"Saving to: {args.output}")
    try:
        out_ds.to_netcdf(args.output)
        print(f"Successfully saved Divb2 to {args.output}")
    except Exception as e:
        print(f"ERROR: Failed to save output file: {e}")
        sys.exit(1)

    # Clean up
    ds.close()
    out_ds.close()

    print("Done!")


if __name__ == '__main__':
    main()

    # First run
    # python global_divb2.py /home/xavier/Oceanography/data/OGCM/LLC/data/ThetaUVWSaltEta/LLC4320_2012-11-09T12_00_00.nc -o /home/xavier/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_divb2.nc