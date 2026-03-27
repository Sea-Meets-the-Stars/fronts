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

| Field | Subset | Units | Equation | Description |
|---|---|---|---|---|
| `Theta` | `native_fields` | °C | — | Potential temperature (LLC4320 native field) |
| `Salt` | `native_fields` | PSU | — | Salinity (LLC4320 native field) |
| `Eta` | `native_fields` | m | — | Sea surface height (LLC4320 native field) |
| `U` | `native_fields` | m/s | — | Zonal velocity (LLC4320 native field) |
| `V` | `native_fields` | m/s | — | Meridional velocity (LLC4320 native field) |
| `W` | `native_fields` | m/s | — | Vertical velocity (LLC4320 native field) |
| `gradb2` | `frontal_structure` | s⁻⁴ | \|∇b\|² = (∂b/∂x)² + (∂b/∂y)² | Squared surface buoyancy gradient magnitude |
| `gradsalt2` | `frontal_structure` | (PSU/m)² | \|∇S\|² = (∂S/∂x)² + (∂S/∂y)² | Squared salinity gradient magnitude |
| `gradtheta2` | `frontal_structure` | (K/m)² | \|∇θ\|² = (∂θ/∂x)² + (∂θ/∂y)² | Squared temperature gradient magnitude |
| `gradeta2` | `frontal_structure` | (m/m)² | \|∇η\|² = (∂η/∂x)² + (∂η/∂y)² | Squared SSH gradient magnitude |
| `relative_vorticity` | `kinematic` | s⁻¹ | ω = ∂v/∂x − ∂u/∂y | Relative vorticity |
| `strain_n` | `kinematic` | s⁻¹ | σ_n = ∂u/∂x − ∂v/∂y | Normal (stretching) strain |
| `strain_s` | `kinematic` | s⁻¹ | σ_s = ∂u/∂y + ∂v/∂x | Shear strain |
| `strain_mag` | `kinematic` | s⁻¹ | \|σ\| = √(σ_n² + σ_s²) | Strain magnitude |
| `divergence` | `kinematic` | s⁻¹ | δ = ∂u/∂x + ∂v/∂y | Horizontal velocity divergence |
| `coriolis_f` | `kinematic` | s⁻¹ | f = 2Ω sin(φ) | Coriolis parameter |
| `rossby_number` | `kinematic` | dimensionless | Ro = ω/f | Rossby number |
| `okubo_weiss` | `kinematic` | s⁻² | OW = σ_n² + σ_s² − ω² | Okubo-Weiss parameter |
| `frontogenesis_tendency` | `frontogenesis` | s⁻⁵ | F = −(∂u/∂x · b_x² + (∂u/∂y + ∂v/∂x) · b_x b_y + ∂v/∂y · b_y²) | Kinematic frontogenesis tendency |
| `ug` | `frontogenesis` | m/s | u_g = −(g/f) ∂η/∂y | Geostrophic zonal velocity |
| `vg` | `frontogenesis` | m/s | v_g = (g/f) ∂η/∂x | Geostrophic meridional velocity |
| `frontogenesis_geo` | `frontogenesis` | s⁻⁵ | F(u_g, v_g) | Geostrophic frontogenesis tendency |
| `frontogenesis_ageo` | `frontogenesis` | s⁻⁵ | F − F_geo | Ageostrophic frontogenesis tendency |

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
