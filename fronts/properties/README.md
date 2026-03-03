# properties

**Label and characterize the properties of ocean fronts**

This module groups connected front pixels into individual fronts and characterizes their properties. 

## What This Module Does

Starting with binary front arrays (from front detection algorithms), this module:
1. **Labels** connected pixels into individual fronts with unique IDs
2. **Calculates** geometric properties (length, orientation, curvature, centroid)
3. **Saves** results to NetCDF, CSV, or Parquet formats
4. **Visualizes** labeled fronts interactively

## Module Structure

### Core Modules

- **`group_labels.py`**: Connected component labeling and ID generation
  - Groups connected front pixels (4-connected or 8-connected; skimage.measure.label)
  - Generates unique front IDs in `TIME_LAT_LON` format
  - Filters fronts by size (i.e. if number of pixels < X, don't include)

- **`geometry.py`**: Geometric property calculation
  - Length and perimeter (in km) 
      - skimage.measure.skeleton or skimage.measure.perimeter
      - skeleton just looks at centerline ; it is more accurate, but takes longer
  - Orientation angle (0-90°)
      - skimage.measure.regionprops.orientation
      - Uses PCA to find dominant direction of front
      - 0 = N/S ; 90 = E/W
  - Mean curvature and curvature direction
      - Skeletonize; calculate angle change between incoming and outgoing (before and after points); curvature = d(theta)/dS
      - Mean curvature: mean(abs) = how curvy
      - Curvature direction: mean(curvature); positive = clockwise, negative = counterclockwise
  - Centroid location (lat, lon)
  - Bounding box (lat/lon ranges)

- **`io.py`**: Input/output operations
  - Export labeled arrays to NetCDF
  - Export properties to CSV/Parquet
  - Convert between formats

## Quick Start

### Basic Usage

```python
from fronts.properties import label, geometry, io
import numpy as np
import xarray as xr

# Load binary front data and coordinates
front_binary = np.load('my_fronts.npy')
ds_coords = xr.open_dataset('coords.nc')
lat, lon = ds_coords['YC'].values, ds_coords['XC'].values
time = '2012-11-09T12:00:00'

# 1. Label connected fronts
labeled = label.label_fronts(front_binary, connectivity=2)
labeled = label.filter_fronts_by_size(labeled, min_size=10)

# 2. Generate unique IDs
front_ids = label.generate_front_ids(labeled, lat, lon, time)

# 3. Calculate geometric properties
geom_props = geometry.calculate_all_geometric_properties(
    labeled, lat, lon, time, include_curvature=True
)

# 4. Export to DataFrame and save
df = io.properties_to_dataframe(geom_props, front_ids)
df.to_csv('front_properties.csv', index=False)
print(f"Found {len(df)} fronts")
print(df.head())

# 5. Load results
df = io.from_csv('front_properties.csv')
labeled, lat, lon, time, front_ids = io.from_netcdf('front_properties.nc')
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
| `orientation` | Front angle (90° = E-W, 0° = N-S) | degrees |
| `lat_min`, `lat_max` | Latitude bounds | degrees |
| `lon_min`, `lon_max` | Longitude bounds | degrees |
| `lat_range`, `lon_range` | Spatial extent | degrees |
| `mean_curvature` | Mean curvature magnitude | km⁻¹ |
| `curvature_direction` | Curvature direction (+ or -) | - |

## Notebooks

- **`visualize.ipynb`**: Quick visualization of labeled fronts with log₁₀(divB²) background; processes fronts on the fly; must be over small region
- **`visualize_global.ipynb`**: Visualization of globally labeled fronts; processes fronts on preexisitng file for grouped global fronts 
  - run after process_global_fronts.py


## API Reference

### group_labels.py

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

### characterize.py

```python
# Calculate all properties
work in progress

# Calculate individual properties
work in progress
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
matplotlib
```

## Installation

This module is part of the `fronts` package. To use:

```python
import sys
sys.path.insert(0, '/path/to/fronts/properties')
from properties import group_labels, geometry, io
```
