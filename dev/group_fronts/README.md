# group_fronts

**Label and characterize the geometric properties of ocean fronts**

This module groups connected front pixels into individual fronts and characterizes their geometric properties. For advanced characterization (field properties, dimensionless numbers, dynamical properties), see the companion `characterize_fronts` module.

## What This Module Does

Starting with binary front arrays (from front detection algorithms), this module:
1. **Labels** connected pixels into individual fronts with unique IDs
2. **Calculates** geometric properties (length, orientation, curvature, centroid)
3. **Saves** results to NetCDF, CSV, or Parquet formats
4. **Visualizes** labeled fronts interactively

## Module Structure

### Core Modules

- **`label.py`**: Connected component labeling and ID generation
  - Groups connected front pixels (4-connected or 8-connected)
  - Generates unique front IDs in `TIME_LAT_LON` format
  - Filters fronts by size or other criteria

- **`geometry.py`**: Geometric property calculation
  - Length and perimeter (in km)
  - Orientation angle (0-180°)
  - Mean curvature and curvature direction
  - Centroid location (lat, lon)
  - Bounding box (lat/lon ranges)

- **`io.py`**: Input/output operations
  - Export labeled arrays to NetCDF
  - Export properties to CSV/Parquet
  - Convert between formats

## Quick Start

### Basic Usage

```python
from group_fronts import label, geometry, io
import numpy as np
import xarray as xr

# Load binary front data and coordinates
front_binary = np.load('my_fronts.npy')
ds_coords = xr.open_dataset('coords.nc')
time = '2012-11-09T12:00:00'

lat = ds_coords['YC'].values
lon = ds_coords['XC'].values

# 1. Label connected fronts
labeled = label.label_fronts(front_binary, connectivity=2)
labeled = label.filter_fronts_by_size(labeled, min_size=10)

# 2. Generate unique IDs
front_ids = label.generate_front_ids(labeled, lat, lon, time)

# 3. Calculate geometric properties
geom_props = geometry.calculate_all_geometric_properties(
    labeled, lat, lon, time, include_curvature=True
)

# 4. Export to DataFrame
df = io.properties_to_dataframe(geom_props, front_ids)
df.to_csv('front_geometric_properties.csv')
print(f"Found {len(df)} fronts")
print(df.head())
```

### Example Output

The resulting DataFrame contains these geometric properties for each front:

| Property | Description | Units |
|----------|-------------|-------|
| `front_id` | Unique ID (TIME_LAT_LON format) | - |
| `time` | Timestamp | ISO 8601 |
| `npix` | Number of pixels | count |
| `centroid_lat` | Centroid latitude | degrees |
| `centroid_lon` | Centroid longitude | degrees |
| `length_km` | Front length | km |
| `orientation` | Front angle (0° = E-W, 90° = N-S) | degrees |
| `lat_min`, `lat_max` | Latitude bounds | degrees |
| `lon_min`, `lon_max` | Longitude bounds | degrees |
| `lat_range`, `lon_range` | Spatial extent | degrees |
| `mean_curvature` | Mean curvature magnitude | km⁻¹ |
| `curvature_direction` | Curvature direction (+ or -) | - |

## Notebooks

- **`visual_inspection.ipynb`**: Quick visualization of labeled fronts with log₁₀(divB²) background
- **`interactive_viewer_bokeh.py`**: Interactive Bokeh app with:
  - Global map with color-coded fronts (random or by property)
  - Dynamic PDFs that update based on map zoom/selection
  - Scatter plots of properties vs. latitude/longitude

## Advanced Characterization

For field-based properties (SST gradients, dominance), dimensionless numbers (Rossby, Richardson, Burger), and dynamical properties (frontogenesis, PV, vertical velocity), see the `characterize_fronts` module.

## API Reference

### label.py

```python
# Label connected fronts
labeled = label.label_fronts(binary_fronts, connectivity=2, return_num=False)

# Filter by size
labeled = label.filter_fronts_by_size(labeled, min_size=10)

# Generate IDs
front_ids = label.generate_front_ids(labeled, lat, lon, time)

# Get list of front labels
labels = label.get_front_labels(labeled)
```

### geometry.py

```python
# Calculate all geometric properties
props = geometry.calculate_all_geometric_properties(
    labeled, lat, lon, time, include_curvature=True
)

# Calculate individual properties
length = geometry.calculate_length(labeled, lat, lon)
orientation = geometry.calculate_orientation(labeled, lat, lon)
curvature = geometry.calculate_curvature(labeled, lat, lon)
```

### io.py

```python
# Convert properties to DataFrame
df = io.properties_to_dataframe(properties_dict, front_ids)

# Save labeled array
io.save_labeled_fronts_netcdf(labeled, lat, lon, time, 'labeled_fronts.nc')

# Export DataFrame
df.to_csv('fronts.csv')
df.to_parquet('fronts.parquet')
```

## Dependencies

```
numpy
scipy
pandas
xarray
netCDF4
```

For interactive visualization:
```
bokeh
matplotlib
```

## Installation

This module is part of the `fronts` package. To use:

```python
import sys
sys.path.insert(0, '/path/to/fronts/dev')
from group_fronts import label, geometry, io
```
