# properties

**Label and characterize the properties of ocean fronts**

Starting with binary front arrays (from front detection), this module labels connected front pixels into individual fronts, computes geometric properties, and co-locates fronts with oceanographic property fields.

## Pipeline

```
binary fronts → group_fronts() → label map + geometry parquet
                                        ↓
              colocate_fronts() → properties parquet + metadata JSON
```

## Module Structure

- **`algorithms.py`** — top-level pipeline functions
  - `group_fronts()`: labels connected components, computes geometry, writes output files
  - `colocate_fronts()`: co-locates labeled fronts with property fields, writes output files

- **`group_labels.py`** — connected component labeling
  - Groups connected front pixels (4- or 8-connected via `skimage.measure.label`)
  - Filters fronts by minimum pixel size

- **`geometry.py`** — geometric property calculation per front
  - Length (km), orientation (0–90°), centroid (lat/lon), bounding box
  - Mean curvature and curvature direction (skeletonize → angle change along front)

- **`colocation.py`** — co-location of fronts with mapped property fields
  - Computes per-front statistics (mean, std, median, percentiles) for any property array
  - Supports dilation (expand front mask N pixels before computing stats)

- **`io.py`** — standardised file I/O
  - `write_front_index` / `load_front_index` — save/load front label index (Parquet/CSV)
  - `get_global_front_output_path` — canonical output paths for all file types (see below)
  - `write_json` — write metadata sidecar files

## Output Files

All paths are resolved via `get_global_front_output_path(output_dir, time_str, file_type, run_tag)`.

| `file_type` | Filename pattern | Written by |
|---|---|---|
| `label_map` | `labeled_fronts_global_{time}_{tag}.npy` | `group_fronts` |
| `front_index` | `front_index_{time}_{tag}.parquet` | `group_fronts` |
| `geometry` | `global_front_geometry_{time}_{tag}.parquet` | `group_fronts` |
| `metadata` | `metadata_{time}_{tag}.json` | `group_fronts` |
| `properties` | `front_properties_{time}_{tag}.parquet` | `colocate_fronts` |
| `metadata_properties` | `metadata_properties_{time}_{tag}.json` | `colocate_fronts` |

## Quick Start

```python
from fronts.properties import algorithms
import numpy as np
import xarray as xr

# 1. Label fronts and compute geometry
algorithms.group_fronts(
    fronts_binary,          # 2-D bool array
    lat, lon,
    fronts_file=fronts_file,
    output_dir=output_dir,
)

# 2. Co-locate with oceanographic property fields
algorithms.colocate_fronts(
    labeled=labeled,                    # integer label array from group_fronts
    property_names=['relative_vorticity', 'divergence'],
    property_dir='/path/to/nc_files',   # LLC4320_{timestamp}_{prop}_v{version}.nc
    fronts_file=fronts_file,
    output_dir=output_dir,
    stats=['mean', 'std', 'median'],
    percentiles=[10, 90],
    dilation_radius=2,
)
```

## Geometry Output Columns

| Column | Description |
|---|---|
| `flabel` | Integer front label |
| `front_id` | Unique ID (`TIME_LAT_LON` format) |
| `npix` | Pixel count |
| `centroid_lat/lon` | Centroid position |
| `length_km` | Front length |
| `orientation` | Angle (0° = N/S, 90° = E/W) |
| `mean_curvature` | Mean absolute curvature |
| `curvature_direction` | Sign: + clockwise, − counterclockwise |
| `lat/lon_min/max` | Bounding box |


## Oceanographic Property Fields & Subsets

Fields available via `generate_properties()`, grouped by subset as defined in `testing_global_v1.yaml`.

| Field | Subset |
|---|---|
| `Theta` | `native_fields` |
| `Salt` | `native_fields` |
| `Eta` | `native_fields` |
| `U` | `native_fields` |
| `V` | `native_fields` |
| `W` | `native_fields` |
| `gradb2` | `frontal_structure` |
| `gradsalt2` | `frontal_structure` |
| `gradtheta2` | `frontal_structure` |
| `gradeta2` | `frontal_structure` |
| `relative_vorticity` | `kinematic` |
| `strain_n` | `kinematic` |
| `strain_s` | `kinematic` |
| `strain_mag` | `kinematic` |
| `divergence` | `kinematic` |
| `coriolis_f` | `kinematic` |
| `rossby_number` | `kinematic` |
| `okubo_weiss` | `kinematic` |
| `frontogenesis_tendency` | `frontogenesis` |
| `ug` | `frontogenesis` |
| `vg` | `frontogenesis` |
| `frontogenesis_geo` | `frontogenesis` |
| `frontogenesis_ageo` | `frontogenesis` |

## Notebooks

**`GroupFronts_Viz.ipynb`** 
**`GroupFronts_Global_Viz.ipynb`** 
**`ColocateFrontProperties_Global_Viz.ipynb`** 
**`PCA_Global_Viz.ipynb`** 
**`JPDF_Global_Viz.ipynb`** 
**`Okubo_Weiss_Global_Viz.ipynb`** 
**`Turner_Angle_Global_Viz.ipynb`** 



## Dependencies

```
numpy, scipy, pandas, xarray, scikit-image
```
