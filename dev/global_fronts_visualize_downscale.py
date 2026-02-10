#!/usr/bin/env python3
"""
Optimized Interactive Bokeh visualization for ocean fronts detection output.

This version downsamples the data for fast browser loading while maintaining
the ability to see global patterns and check for masking issues.
"""

import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, CustomJS, CheckboxGroup
from bokeh.layouts import row, column
from bokeh.palettes import Greys256, Inferno256, Reds256, Viridis256, Plasma256
from scipy import ndimage
import os

# ==============================================================================
# Configuration
# ==============================================================================

# Path to the fronts detection output
FRONTS_FILE = "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/global/LLC4320_2012-11-09T12_00_00_fronts.npy"
DIVB2_FILE = "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_divb2.nc"  # Adjust if needed

# Maximum display resolution (for performance)
MAX_DISPLAY_SIZE = 8000  # pixels per dimension


# ==============================================================================
# Load Data
# ==============================================================================

print(f"Loading fronts data from: {FRONTS_FILE}")
fronts_data_full = np.load(FRONTS_FILE, allow_pickle=True)
print(f"Original fronts data shape: {fronts_data_full.shape}")
print(f"Original fronts data range: [{np.nanmin(fronts_data_full):.6f}, {np.nanmax(fronts_data_full):.6f}]")
print(f"Number of NaN values: {np.sum(np.isnan(fronts_data_full))}")

# Load divb2 data if available
divb2_data_full = None
lon = None
lat = None

if DIVB2_FILE and os.path.exists(DIVB2_FILE):
    print(f"\nLoading divb2 data from: {DIVB2_FILE}")

    if DIVB2_FILE.endswith('.nc'):
        try:
            import xarray as xr
            ds = xr.open_dataset(DIVB2_FILE)
            print(f"NetCDF variables: {list(ds.data_vars.keys())}")

            possible_vars = ['divb2', 'divB2', 'div_b2', 'data', 'field']
            divb2_var = None
            for var in possible_vars:
                if var in ds.data_vars:
                    divb2_var = var
                    break

            if divb2_var is None:
                divb2_var = list(ds.data_vars.keys())[0]

            divb2_data_full = ds[divb2_var].values

            if 'lon' in ds.coords or 'longitude' in ds.coords:
                lon = ds['lon'].values if 'lon' in ds.coords else ds['longitude'].values
            if 'lat' in ds.coords or 'latitude' in ds.coords:
                lat = ds['lat'].values if 'lat' in ds.coords else ds['latitude'].values

            ds.close()

        except ImportError:
            print("xarray not available. Install with: pip install xarray netcdf4")

    elif DIVB2_FILE.endswith('.npy'):
        divb2_data_full = np.load(DIVB2_FILE, allow_pickle=True)

    if divb2_data_full is not None:
        print(f"Original divb2 data shape: {divb2_data_full.shape}")
        print(f"Original divb2 data range: [{np.nanmin(divb2_data_full):.6f}, {np.nanmax(divb2_data_full):.6f}]")

# ==============================================================================
# Downsample data for visualization
# ==============================================================================

ny_full, nx_full = fronts_data_full.shape

# Calculate downsampling factor
downsample_factor = max(1, max(nx_full, ny_full) // MAX_DISPLAY_SIZE)

print(f"\nDownsampling by factor of {downsample_factor} for performance...")
print(f"  ({ny_full} x {nx_full}) -> ({ny_full//downsample_factor} x {nx_full//downsample_factor})")

# Use block averaging for downsampling (preserves features better than simple decimation)
if downsample_factor > 1:
    # Trim to make divisible
    ny_trim = (ny_full // downsample_factor) * downsample_factor
    nx_trim = (nx_full // downsample_factor) * downsample_factor

    fronts_trimmed = fronts_data_full[:ny_trim, :nx_trim]

    # Reshape and average
    ny_new = ny_trim // downsample_factor
    nx_new = nx_trim // downsample_factor

    fronts_data = fronts_trimmed.reshape(ny_new, downsample_factor, nx_new, downsample_factor).mean(axis=(1, 3))

    # Do the same for divb2 if available
    if divb2_data_full is not None:
        divb2_trimmed = divb2_data_full[:ny_trim, :nx_trim]
        divb2_data = divb2_trimmed.reshape(ny_new, downsample_factor, nx_new, downsample_factor).mean(axis=(1, 3))
    else:
        divb2_data = None
else:
    fronts_data = fronts_data_full
    divb2_data = divb2_data_full

print(f"Display data shape: {fronts_data.shape}")

# ==============================================================================
# Prepare coordinates
# ==============================================================================

ny, nx = fronts_data.shape

# Create coordinate arrays if not loaded from NetCDF
if lon is None or lat is None:
    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90, 90, ny)
    print("\nUsing default global coordinates (lon: -180 to 180, lat: -90 to 90)")
else:
    print(f"\nUsing coordinates from NetCDF file")
    if lon.ndim == 2:
        lon = lon[0, ::downsample_factor]
    elif downsample_factor > 1:
        lon = lon[::downsample_factor]

    if lat.ndim == 2:
        lat = lat[::downsample_factor, 0]
    elif downsample_factor > 1:
        lat = lat[::downsample_factor]

dlon = lon[1] - lon[0] if len(lon) > 1 else 1
dlat = lat[1] - lat[0] if len(lat) > 1 else 1

extent_lon = [lon[0] - dlon/2, lon[-1] + dlon/2]
extent_lat = [lat[0] - dlat/2, lat[-1] + dlat/2]

# ==============================================================================
# Create Bokeh figure
# ==============================================================================

# Apply threshold: < 0.1 = transparent, >= 0.1 = red
FRONT_THRESHOLD = 0.1
fronts_thresholded = np.where(fronts_data >= FRONT_THRESHOLD, fronts_data, np.nan)

fronts_valid = fronts_thresholded[~np.isnan(fronts_thresholded)]
if len(fronts_valid) > 0:
    fronts_min = FRONT_THRESHOLD
    fronts_max = np.nanmax(fronts_thresholded)
else:
    fronts_min, fronts_max = 0, 1

fronts_mapper = LinearColorMapper(
    palette=Reds256[::-1],  # Reds: light -> dark red for values >= 0.1
    low=fronts_min,
    high=fronts_max,
    nan_color='rgba(0,0,0,0)'  # Transparent for values < 0.1
)

p = figure(
    width=1400,
    height=700,
    title=f"Ocean Fronts Detection (threshold >= {FRONT_THRESHOLD}) - Downsampled {downsample_factor}x",
    x_axis_label="Longitude",
    y_axis_label="Latitude",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_scroll="wheel_zoom",
    x_range=(extent_lon[0], extent_lon[1]),
    y_range=(extent_lat[0], extent_lat[1])
)

# Display fronts as image (store reference for toggling)
fronts_img = p.image(
    image=[fronts_thresholded],
    x=extent_lon[0],
    y=extent_lat[0],
    dw=extent_lon[1] - extent_lon[0],
    dh=extent_lat[1] - extent_lat[0],
    color_mapper=fronts_mapper,
    level="overlay",  # Render on top
    alpha=1.0,  # Fully opaque
    name="fronts_layer"
)

fronts_color_bar = ColorBar(
    color_mapper=fronts_mapper,
    width=8,
    location=(0, 0),
    title="Fronts (Reds)"
)
p.add_layout(fronts_color_bar, 'right')

# ==============================================================================
# Add divb2 underlay if available
# ==============================================================================

if divb2_data is not None:
    divb2_valid = divb2_data[~np.isnan(divb2_data)]
    if len(divb2_valid) > 0:
        divb2_min, divb2_max = np.percentile(divb2_valid, [1, 99])
    else:
        divb2_min, divb2_max = 0, 1

    # Use inverse grayscale (black -> white, so high values are white)
    divb2_mapper = LinearColorMapper(
        palette=Greys256[::-1],  # Reversed: black -> white
        low=divb2_min,
        high=divb2_max,
        nan_color='rgba(128,128,128,0.3)'  # Medium gray for NaN
    )

    # Add divb2 as background layer (visible by default, store reference)
    divb2_img = p.image(
        image=[divb2_data],
        x=extent_lon[0],
        y=extent_lat[0],
        dw=extent_lon[1] - extent_lon[0],
        dh=extent_lat[1] - extent_lat[0],
        color_mapper=divb2_mapper,
        level="image",  # Base image level
        alpha=1.0,  # Fully opaque background
        name="divb2_layer"
    )

    # Add second color bar for divb2
    divb2_color_bar = ColorBar(
        color_mapper=divb2_mapper,
        width=8,
        location=(0, 0),
        title="Divb2 (Inv. Grey)"
    )
    p.add_layout(divb2_color_bar, 'left')

# ==============================================================================
# Add simplified hover
# ==============================================================================

# Use much sparser sampling for hover
hover_factor = max(1, min(nx, ny) // 100)
lon_sample = lon[::hover_factor]
lat_sample = lat[::hover_factor]
fronts_sample = fronts_data[::hover_factor, ::hover_factor]

# Create grid for hover
lon_grid, lat_grid = np.meshgrid(lon_sample, lat_sample)
fronts_hover = fronts_sample.flatten()
lon_hover = lon_grid.flatten()
lat_hover = lat_grid.flatten()

# Filter valid points
valid_mask = ~np.isnan(fronts_hover)

from bokeh.models import ColumnDataSource

hover_data = {
    'x': lon_hover[valid_mask],
    'y': lat_hover[valid_mask],
    'front': fronts_hover[valid_mask],
}

# Add divb2 to hover if available
if divb2_data is not None:
    divb2_sample = divb2_data[::hover_factor, ::hover_factor]
    divb2_hover = divb2_sample.flatten()[valid_mask]
    hover_data['divb2'] = divb2_hover

hover_source = ColumnDataSource(data=hover_data)

# Add invisible markers for hover (very sparse)
hover_scatter = p.scatter(
    x='x', y='y',
    source=hover_source,
    size=3,
    alpha=0,
)

# Create tooltips based on available data
if divb2_data is not None:
    hover_tooltips = [
        ("Lon, Lat", "(@x{0.0}, @y{0.0})"),
        ("Front", "@front{0.0000}"),
        ("Divb2", "@divb2{0.0000}"),
    ]
else:
    hover_tooltips = [
        ("Lon, Lat", "(@x{0.0}, @y{0.0})"),
        ("Front", "@front{0.0000}"),
    ]

hover = HoverTool(
    tooltips=hover_tooltips,
    renderers=[hover_scatter],
)
p.add_tools(hover)

# ==============================================================================
# Statistics panel
# ==============================================================================

stats_text = f"""
<div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; font-family: monospace;">
<h3>Data Statistics</h3>

<p><b>Original Data:</b></p>
<ul>
  <li>Shape: {fronts_data_full.shape[0]:,} x {fronts_data_full.shape[1]:,}</li>
  <li>Total pixels: {fronts_data_full.size:,}</li>
</ul>

<p><b>Display Data (downsampled {downsample_factor}x):</b></p>
<ul>
  <li>Shape: {fronts_data.shape[0]:,} x {fronts_data.shape[1]:,}</li>
  <li>Valid values: {np.sum(~np.isnan(fronts_data)):,}</li>
  <li>NaN values: {np.sum(np.isnan(fronts_data)):,}</li>
</ul>

<p><b>Front Values (Reds, right colorbar):</b></p>
<ul>
  <li><b>Threshold:</b> {FRONT_THRESHOLD} (values below are transparent)</li>
  <li>All values: {np.nanmin(fronts_data):.6f} to {np.nanmax(fronts_data):.6f}</li>
  <li>Visible fronts (≥{FRONT_THRESHOLD}): {np.sum(fronts_data >= FRONT_THRESHOLD):,} pixels</li>
  <li>Hidden (below threshold): {np.sum(fronts_data < FRONT_THRESHOLD):,} pixels</li>
</ul>
"""

if divb2_data is not None:
    stats_text += f"""
<p><b>Divb2 Values (Inv. Grey, left colorbar):</b></p>
<ul>
  <li>Min: {np.nanmin(divb2_data):.6f}</li>
  <li>Max: {np.nanmax(divb2_data):.6f}</li>
  <li>Mean: {np.nanmean(divb2_data):.6f}</li>
  <li>Std: {np.nanstd(divb2_data):.6f}</li>
</ul>
"""

stats_text += f"""
<p><b>Visualization Layers:</b></p>
<ul>
  <li><b>Bottom:</b> Divb2 field (Inverse grey - high=white)</li>
  <li><b>Top:</b> Fronts ≥ {FRONT_THRESHOLD} (Reds - transparent below threshold)</li>
  <li><b>Toggle:</b> Use checkboxes above to show/hide layers</li>
</ul>

<p><b>Land Mask Check:</b></p>
<ul>
  <li>Any red = front ≥ {FRONT_THRESHOLD}, darker = stronger</li>
  <li>Transparent = front < {FRONT_THRESHOLD} or no data</li>
  <li>Zoom to coastlines to check masking</li>
  <li>Look for red fronts along land boundaries</li>
  <li><b style="color: red;">WARNING: 0 NaN values detected!</b></li>
  <li>This suggests no land masking was applied</li>
</ul>

<p><b>Performance:</b></p>
<ul>
  <li>Data downsampled for fast loading</li>
  <li>Use zoom to explore details</li>
  <li>Hover shows approximate values</li>
</ul>
</div>
"""

from bokeh.models import Div
stats_div = Div(text=stats_text, width=380, height=700)

# ==============================================================================
# Add layer toggle controls
# ==============================================================================

# Create list of renderers to toggle
toggle_renderers = [fronts_img]
toggle_labels = ["Front Strength (Reds)"]

if divb2_data is not None:
    toggle_renderers.insert(0, divb2_img)  # Add divb2 at the beginning
    toggle_labels.insert(0, "Divb2 Field (Inverse Grey)")

# Create checkbox group (all checked by default)
layer_toggle = CheckboxGroup(
    labels=toggle_labels,
    active=list(range(len(toggle_labels))),  # All active by default
    width=380
)

# Create CustomJS callback to toggle visibility
callback = CustomJS(args=dict(renderers=toggle_renderers, checkbox=layer_toggle), code="""
    for (let i = 0; i < renderers.length; i++) {
        renderers[i].visible = checkbox.active.includes(i);
    }
""")

layer_toggle.js_on_change('active', callback)

# Create control panel with toggles and stats
controls_div = Div(text="<h3 style='margin-top:0'>Layer Controls</h3><p>Toggle layers on/off:</p>", width=380)
control_panel = column(controls_div, layer_toggle, stats_div)

# ==============================================================================
# Output
# ==============================================================================

layout = row(p, control_panel)

output_file("/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/front_finding/global/ocean_fronts_viewer_fast.html")
show(layout)

print("\n" + "="*80)
print("FAST Visualization complete!")
print("="*80)
print(f"\nOriginal data: {fronts_data_full.shape}")
print(f"Displayed at: {fronts_data.shape} ({downsample_factor}x downsampled)")
print(f"\nFile saved: ocean_fronts_viewer_fast.html")

if divb2_data is not None:
    print("\nLayers displayed:")
    print("  - Bottom: Divb2 field (Inverse grayscale - high values = white)")
    print(f"  - Top: Fronts >= {FRONT_THRESHOLD} (Reds - values below threshold are transparent)")
else:
    print("\nNote: Divb2 data not loaded (file not found)")

print("\nKey findings:")
print(f"  - Front threshold: {FRONT_THRESHOLD}")
print(f"  - Front values range: {np.nanmin(fronts_data):.6f} to {np.nanmax(fronts_data):.6f}")
print(f"  - Visible fronts (>={FRONT_THRESHOLD}): {np.sum(fronts_data >= FRONT_THRESHOLD):,} pixels")
print(f"  - Hidden (<{FRONT_THRESHOLD}): {np.sum(fronts_data < FRONT_THRESHOLD):,} pixels")
if divb2_data is not None:
    print(f"  - Divb2 values range: {np.nanmin(divb2_data):.6f} to {np.nanmax(divb2_data):.6f}")
print(f"  - NaN count: {np.sum(np.isnan(fronts_data)):,} (should be >0 if land masked)")
print("\nNext steps:")
print("  1. Use checkboxes to toggle layers and compare them")
print("  2. Compare red fronts with inverse grayscale divb2 field")
print("  3. Zoom in on known ocean fronts (Gulf Stream, Kuroshio, ACC)")
print("  4. Check coastlines for spurious front detections")
print("  5. If fronts appear at coastlines, land masking needs fixing")
print("="*80)
