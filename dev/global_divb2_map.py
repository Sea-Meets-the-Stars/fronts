#!/usr/bin/env python
"""
Create a map visualization of the Divb2 output data.

This script loads the Divb2 NetCDF file and creates a map/plot visualization
with various customization options.

Usage:
    python map_divb2.py <divb2_file> [options]

Example:
    python map_divb2.py /mnt/tank/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_divb2.nc --output LLC4320_2012-11-09T12_00_00_divb2_map.png --log-scale --downsample 5

"""

import argparse
import sys
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker

# Try to import cartopy for geographic plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available. Will create simple 2D plots without geographic projection.")


def load_data(filepath, downsample=1):
    """
    Load Divb2 data from NetCDF file.

    Parameters:
        filepath (str): Path to NetCDF file
        downsample (int): Downsampling factor (default: 1, no downsampling)

    Returns:
        tuple: (data, dataset) where data is the Divb2 array and dataset is the xarray Dataset
    """
    print(f"Loading data from: {filepath}")
    ds = xr.open_dataset(filepath)

    if 'Divb2' not in ds:
        print("ERROR: 'Divb2' variable not found in dataset")
        sys.exit(1)

    # Extract Divb2 data
    data = ds.Divb2.values

    # Apply downsampling if requested
    if downsample > 1:
        print(f"Downsampling by factor of {downsample}")
        data = data[::downsample, ::downsample]

    print(f"Data shape: {data.shape}")
    print(f"Data range: [{np.nanmin(data):.2e}, {np.nanmax(data):.2e}]")
    print(f"Number of NaN values: {np.sum(np.isnan(data))}")

    return data, ds


def create_map(data, ds, args):
    """
    Create a map visualization of the Divb2 data.

    Parameters:
        data (np.ndarray): Divb2 data array
        ds (xr.Dataset): Original dataset
        args: Command-line arguments
    """
    # Set up figure size
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]))

    # Check if we have lat/lon coordinates for geographic plotting
    has_coords = ('lat' in ds or 'YC' in ds) and ('lon' in ds or 'XC' in ds)

    if has_coords and HAS_CARTOPY and not args.no_geo:
        # Create geographic plot with cartopy
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)

        # Get coordinates
        if 'lon' in ds and 'lat' in ds:
            lon = ds.lon.values
            lat = ds.lat.values
        else:
            lon = ds.XC.values
            lat = ds.YC.values

        # Downsample coordinates if needed
        if args.downsample > 1:
            lon = lon[::args.downsample, ::args.downsample]
            lat = lat[::args.downsample, ::args.downsample]

        # Plot data
        im = plot_data(ax, data, lon, lat, args, pcolormesh=True)
    else:
        # Create simple 2D plot without geographic projection
        ax = plt.axes()
        im = plot_data(ax, data, None, None, args, pcolormesh=False)
        ax.set_xlabel('i (grid index)')
        ax.set_ylabel('j (grid index)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.05, aspect=40, shrink=0.8)

    # Set colorbar label
    units = ds.Divb2.attrs.get('units', '')
    if args.log_scale:
        cbar.set_label(f'log₁₀(Divb2) [{units}]', fontsize=12)
    else:
        cbar.set_label(f'Divb2 [{units}]', fontsize=12)

    # Set title
    title = args.title if args.title else 'Squared Divergence of Buoyancy (Divb2)'
    plt.title(title, fontsize=14, pad=20)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_data(ax, data, lon, lat, args, pcolormesh=True):
    """
    Plot the data on the given axes.

    Parameters:
        ax: Matplotlib axes
        data: Data array to plot
        lon: Longitude coordinates (None for simple plot)
        lat: Latitude coordinates (None for simple plot)
        args: Command-line arguments
        pcolormesh: Whether to use pcolormesh (True) or imshow (False)

    Returns:
        Image object for colorbar
    """
    # Apply log scale if requested
    plot_data = data.copy()
    if args.log_scale:
        plot_data = np.log10(np.abs(plot_data))
        plot_data[np.isinf(plot_data)] = np.nan

    # Set up colormap
    cmap = plt.get_cmap(args.cmap)
    if args.mask_land:
        cmap.set_bad('lightgray')

    # Set up normalization
    if args.vmin is not None and args.vmax is not None:
        norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax)
    else:
        # Use percentiles to avoid outliers
        vmin = np.nanpercentile(plot_data, args.percentile_min)
        vmax = np.nanpercentile(plot_data, args.percentile_max)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot the data
    if pcolormesh and lon is not None and lat is not None:
        # Geographic plot with pcolormesh
        im = ax.pcolormesh(lon, lat, plot_data,
                          cmap=cmap, norm=norm,
                          shading='auto', rasterized=True)
    else:
        # Simple 2D plot with imshow
        im = ax.imshow(plot_data, origin='lower',
                      cmap=cmap, norm=norm,
                      aspect='auto', interpolation='nearest')

    return im


def main():
    """Main function to create map visualization."""

    parser = argparse.ArgumentParser(
        description='Create a map visualization of Divb2 data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file', type=str,
                        help='Path to input Divb2 NetCDF file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output figure path (default: input_map.png)')
    parser.add_argument('--downsample', '-d', type=int, default=1,
                        help='Downsample factor for faster plotting (1 = no downsampling)')
    parser.add_argument('--log-scale', '-l', action='store_true',
                        help='Plot data on log10 scale')
    parser.add_argument('--cmap', '-c', type=str, default='viridis',
                        help='Matplotlib colormap name')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for colorbar')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for colorbar')
    parser.add_argument('--percentile-min', type=float, default=1,
                        help='Lower percentile for colorbar if vmin not specified')
    parser.add_argument('--percentile-max', type=float, default=99,
                        help='Upper percentile for colorbar if vmax not specified')
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 10],
                        help='Figure size in inches (width height)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI for saved figure')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the plot')
    parser.add_argument('--mask-land', action='store_true',
                        help='Mask NaN values (land) with gray color')
    parser.add_argument('--no-geo', action='store_true',
                        help='Disable geographic projection even if coordinates available')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Display the plot interactively (in addition to saving)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    # Set output filename if not provided
    if args.output is None:
        base = os.path.splitext(args.input_file)[0]
        args.output = f"{base}_map.png"

    # Load the data
    data, ds = load_data(args.input_file, downsample=args.downsample)

    # Create the map
    print("Creating map visualization...")
    fig = create_map(data, ds, args)

    # Save the figure
    print(f"Saving figure to: {args.output}")
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Successfully saved map to {args.output}")

    # Show interactively if requested
    if args.show:
        print("Displaying figure...")
        plt.show()

    # Clean up
    plt.close(fig)
    ds.close()

    print("Done!")


if __name__ == '__main__':
    main()
