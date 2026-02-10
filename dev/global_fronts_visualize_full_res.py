#!/usr/bin/env python3
"""
Interactive Bokeh visualization for ocean fronts detection output.

This script loads the detected fronts from .npy files and creates an interactive
map with pan/zoom, hover tooltips, and front overlays to verify masking behavior.
"""

import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, LogColorMapper
from bokeh.layouts import column, row
from bokeh.models.widgets import Select, CheckboxGroup
from bokeh.io import curdoc
from bokeh.palettes import Viridis256, Plasma256
import os

# ==============================================================================
# Configuration
# ==============================================================================

# Path to the fronts detection output
FRONTS_FILE = "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/global/LLC4320_2012-11-09T12_00_00_fronts.npy"
DIVB2_FILE = "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_divb2.nc"  # Adjust if needed

# ==============================================================================
# Load Data
# ==============================================================================

print(f"Loading fronts data from: {FRONTS_FILE}")
fronts_data = np.load(FRONTS_FILE, allow_pickle=True)
print(f"Fronts data shape: {fronts_data.shape}")
print(f"Fronts data range: [{np.nanmin(fronts_data):.6f}, {np.nanmax(fronts_data):.6f}]")
print(f"Number of NaN values: {np.sum(np.isnan(fronts_data))}")

# Load divb2 data if available
divb2_data = None
lon = None
lat = None

if DIVB2_FILE and os.path.exists(DIVB2_FILE):
    print(f"\nLoading divb2 data from: {DIVB2_FILE}")

    # Check file extension and load accordingly
    if DIVB2_FILE.endswith('.nc'):
        try:
            import xarray as xr
            ds = xr.open_dataset(DIVB2_FILE)
            print(f"NetCDF variables: {list(ds.data_vars.keys())}")
            print(f"NetCDF coordinates: {list(ds.coords.keys())}")

            # Try to find the main data variable (divb2 or similar)
            possible_vars = ['divb2', 'divB2', 'div_b2', 'data', 'field']
            divb2_var = None
            for var in possible_vars:
                if var in ds.data_vars:
                    divb2_var = var
                    break

            if divb2_var is None:
                # Take the first data variable
                divb2_var = list(ds.data_vars.keys())[0]
                print(f"Using first data variable: {divb2_var}")

            divb2_data = ds[divb2_var].values

            # Extract coordinates if available
            if 'lon' in ds.coords or 'longitude' in ds.coords:
                lon = ds['lon'].values if 'lon' in ds.coords else ds['longitude'].values
            if 'lat' in ds.coords or 'latitude' in ds.coords:
                lat = ds['lat'].values if 'lat' in ds.coords else ds['latitude'].values

            ds.close()

        except ImportError:
            print("xarray not available, trying netCDF4...")
            try:
                from netCDF4 import Dataset
                nc = Dataset(DIVB2_FILE, 'r')
                print(f"NetCDF variables: {list(nc.variables.keys())}")

                # Try to find the main data variable
                possible_vars = ['divb2', 'divB2', 'div_b2', 'data', 'field']
                divb2_var = None
                for var in possible_vars:
                    if var in nc.variables:
                        divb2_var = var
                        break

                if divb2_var is None:
                    # Take the first non-coordinate variable
                    for var in nc.variables.keys():
                        if var not in ['lon', 'lat', 'longitude', 'latitude', 'time']:
                            divb2_var = var
                            break

                if divb2_var:
                    divb2_data = nc.variables[divb2_var][:]

                # Extract coordinates
                if 'lon' in nc.variables:
                    lon = nc.variables['lon'][:]
                elif 'longitude' in nc.variables:
                    lon = nc.variables['longitude'][:]

                if 'lat' in nc.variables:
                    lat = nc.variables['lat'][:]
                elif 'latitude' in nc.variables:
                    lat = nc.variables['latitude'][:]

                nc.close()

            except ImportError:
                print("Neither xarray nor netCDF4 available. Cannot read .nc file.")
                print("Install with: pip install xarray netcdf4")

    elif DIVB2_FILE.endswith('.npy'):
        divb2_data = np.load(DIVB2_FILE, allow_pickle=True)

    if divb2_data is not None:
        print(f"Divb2 data shape: {divb2_data.shape}")
        print(f"Divb2 data range: [{np.nanmin(divb2_data):.6f}, {np.nanmax(divb2_data):.6f}]")
        print(f"Number of NaN values: {np.sum(np.isnan(divb2_data))}")
else:
    print(f"\nDivb2 file not found. Searched for:")
    print("  - *.npy and *.nc files near fronts file")
    print("Proceeding with fronts data only.")

# ==============================================================================
# Prepare visualization data
# ==============================================================================

ny, nx = fronts_data.shape

# Create coordinate arrays if not loaded from NetCDF
if lon is None or lat is None:
    # Assuming global coverage: longitude [-180, 180], latitude [-90, 90]
    # Adjust these if your LLC4320 grid is different
    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90, 90, ny)
    print("\nUsing default global coordinates (lon: -180 to 180, lat: -90 to 90)")
    print("Adjust if your grid uses different conventions.")
else:
    print(f"\nUsing coordinates from NetCDF file")
    print(f"Lon range: [{lon.min():.2f}, {lon.max():.2f}]")
    print(f"Lat range: [{lat.min():.2f}, {lat.max():.2f}]")

    # Handle 2D coordinate arrays (common in LLC grids)
    if lon.ndim == 2:
        print("Note: 2D coordinates detected. Using 1D approximation for visualization.")
        lon = lon[0, :]  # Take first row
    if lat.ndim == 2:
        lat = lat[:, 0]  # Take first column

# Create meshgrid for image coordinates
dlon = lon[1] - lon[0] if len(lon) > 1 else 1
dlat = lat[1] - lat[0] if len(lat) > 1 else 1

# Extent: [left, right, bottom, top]
extent_lon = [lon[0] - dlon/2, lon[-1] + dlon/2]
extent_lat = [lat[0] - dlat/2, lat[-1] + dlat/2]

# ==============================================================================
# Create Bokeh figure
# ==============================================================================

# Determine value ranges for color mapping
fronts_valid = fronts_data[~np.isnan(fronts_data)]
if len(fronts_valid) > 0:
    fronts_min, fronts_max = np.percentile(fronts_valid, [1, 99])
else:
    fronts_min, fronts_max = 0, 1

# Color mapper for fronts
fronts_mapper = LinearColorMapper(
    palette=Plasma256,
    low=fronts_min,
    high=fronts_max,
    nan_color='gray'
)

# Create the main plot
p = figure(
    width=1200,
    height=600,
    title="Ocean Fronts Detection - Interactive Viewer",
    x_axis_label="Longitude",
    y_axis_label="Latitude",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_scroll="wheel_zoom",
    x_range=(extent_lon[0], extent_lon[1]),
    y_range=(extent_lat[0], extent_lat[1])
)

# Display fronts as image
fronts_image = p.image(
    image=[fronts_data],
    x=extent_lon[0],
    y=extent_lat[0],
    dw=extent_lon[1] - extent_lon[0],
    dh=extent_lat[1] - extent_lat[0],
    color_mapper=fronts_mapper,
    level="image",
    name="fronts"
)

# Add color bar for fronts
fronts_color_bar = ColorBar(
    color_mapper=fronts_mapper,
    width=8,
    location=(0, 0),
    title="Front Strength"
)
p.add_layout(fronts_color_bar, 'right')

# ==============================================================================
# Add divb2 overlay if available
# ==============================================================================

if divb2_data is not None:
    divb2_valid = divb2_data[~np.isnan(divb2_data)]
    if len(divb2_valid) > 0:
        divb2_min, divb2_max = np.percentile(divb2_valid, [1, 99])
    else:
        divb2_min, divb2_max = 0, 1

    divb2_mapper = LinearColorMapper(
        palette=Viridis256,
        low=divb2_min,
        high=divb2_max,
        nan_color='gray'
    )

    # Add divb2 as a separate image (initially hidden)
    divb2_image = p.image(
        image=[divb2_data],
        x=extent_lon[0],
        y=extent_lat[0],
        dw=extent_lon[1] - extent_lon[0],
        dh=extent_lat[1] - extent_lat[0],
        color_mapper=divb2_mapper,
        level="image",
        visible=False,
        name="divb2"
    )

# ==============================================================================
# Create front detection overlay (threshold-based)
# ==============================================================================

# Create a binary mask for strong fronts (adjust threshold as needed)
threshold_percentile = 90
if len(fronts_valid) > 0:
    threshold = np.percentile(fronts_valid, threshold_percentile)
    front_mask = np.where(fronts_data > threshold, fronts_data, np.nan)
else:
    front_mask = np.full_like(fronts_data, np.nan)

# Overlay front detections with high contrast
front_overlay_mapper = LinearColorMapper(
    palette=['red', 'yellow'],
    low=threshold if len(fronts_valid) > 0 else 0,
    high=fronts_max,
    nan_color=(0, 0, 0, 0)  # Transparent for NaN
)

front_overlay = p.image(
    image=[front_mask],
    x=extent_lon[0],
    y=extent_lat[0],
    dw=extent_lon[1] - extent_lon[0],
    dh=extent_lat[1] - extent_lat[0],
    color_mapper=front_overlay_mapper,
    level="overlay",
    alpha=0.6,
    name="front_overlay"
)

# ==============================================================================
# Add hover tool for value interrogation
# ==============================================================================

# For precise hover, we need to add a second representation with actual data points
# Create sample points (decimated for performance)
sample_factor = max(1, min(nx, ny) // 500)  # Sample every N points
lon_sample = lon[::sample_factor]
lat_sample = lat[::sample_factor]
fronts_sample = fronts_data[::sample_factor, ::sample_factor]

# Flatten for scatter plot
lon_flat = np.repeat(lon_sample, len(lat_sample))
lat_flat = np.tile(lat_sample, len(lon_sample))
fronts_flat = fronts_sample.T.flatten()

# Filter out NaN values for hover
valid_mask = ~np.isnan(fronts_flat)
lon_hover = lon_flat[valid_mask]
lat_hover = lat_flat[valid_mask]
fronts_hover = fronts_flat[valid_mask]

# Create hover data source
hover_source = {
    'lon': lon_hover,
    'lat': lat_hover,
    'front_value': fronts_hover,
}

if divb2_data is not None:
    divb2_sample = divb2_data[::sample_factor, ::sample_factor]
    divb2_flat = divb2_sample.T.flatten()[valid_mask]
    hover_source['divb2_value'] = divb2_flat

# Add invisible scatter for hover (smaller sample for performance)
hover_sample_factor = max(1, len(lon_hover) // 10000)
hover_scatter = p.scatter(
    lon_hover[::hover_sample_factor],
    lat_hover[::hover_sample_factor],
    size=1,
    alpha=0,
    name="hover_points"
)

# Configure hover tool
if divb2_data is not None:
    hover_tooltips = [
        ("Lon, Lat", "(@x{0.00}, @y{0.00})"),
        ("Front", "@front_value{0.0000}"),
        ("Divb2", "@divb2_value{0.0000}")
    ]
else:
    hover_tooltips = [
        ("Lon, Lat", "(@x{0.00}, @y{0.00})"),
        ("Front", "@front_value{0.0000}")
    ]

hover = HoverTool(
    tooltips=hover_tooltips,
    mode='mouse',
    point_policy='snap_to_data',
    renderers=[hover_scatter]
)
p.add_tools(hover)

# ==============================================================================
# Add statistics panel
# ==============================================================================

stats_text = f"""
<div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
<h3>Data Statistics</h3>
<p><b>Fronts Data:</b></p>
<ul>
  <li>Shape: {fronts_data.shape}</li>
  <li>Valid values: {np.sum(~np.isnan(fronts_data)):,}</li>
  <li>NaN values: {np.sum(np.isnan(fronts_data)):,}</li>
  <li>Min: {np.nanmin(fronts_data):.6f}</li>
  <li>Max: {np.nanmax(fronts_data):.6f}</li>
  <li>Mean: {np.nanmean(fronts_data):.6f}</li>
  <li>Threshold ({threshold_percentile}%): {threshold:.6f}</li>
</ul>
"""

if divb2_data is not None:
    stats_text += f"""
<p><b>Divb2 Data:</b></p>
<ul>
  <li>Shape: {divb2_data.shape}</li>
  <li>Valid values: {np.sum(~np.isnan(divb2_data)):,}</li>
  <li>NaN values: {np.sum(np.isnan(divb2_data)):,}</li>
  <li>Min: {np.nanmin(divb2_data):.6f}</li>
  <li>Max: {np.nanmax(divb2_data):.6f}</li>
  <li>Mean: {np.nanmean(divb2_data):.6f}</li>
</ul>
"""

stats_text += """
<p><b>Land Mask Check:</b></p>
<p>Use the red/yellow overlay to check if fronts appear along coastlines or NaN edges.
Strong fronts near land boundaries may indicate masking issues.</p>
<p><b>Note:</b> With 0 NaN values, the entire domain is valid. Check if this is expected
or if land masking should be applied.</p>
</div>
"""

from bokeh.models import Div
stats_div = Div(text=stats_text, width=350, height=450)

# ==============================================================================
# Layout and output
# ==============================================================================

layout = row(p, stats_div)

output_file("/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/front_finding/global/ocean_fronts_viewer.html")
show(layout)

print("\n" + "="*80)
print("Visualization complete!")
print("="*80)
print("\nUsage tips:")
print("  - Use mouse wheel to zoom in/out")
print("  - Click and drag to pan")
print("  - Hover over the map to see values at cursor location")
print("  - Red/yellow overlay shows strong fronts (top 10% by default)")
print("  - Gray areas represent NaN values (likely land or invalid data)")
print("\nLand Mask Verification:")
print("  - Check if the red/yellow overlay appears along coastlines")
print("  - If fronts light up at land boundaries, masking may need adjustment")
print("  - Compare front patterns with known ocean features")
print(f"\nNote: Found {np.sum(np.isnan(fronts_data)):,} NaN values in fronts data")
print("      If this is 0, verify that land masking was applied correctly.")
print("="*80)
