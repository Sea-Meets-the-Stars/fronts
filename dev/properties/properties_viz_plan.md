# Plan: Global Visualization of Front-Weighted Properties

## Objective

Build reusable methods that produce global visualizations of any front-weighted
property (relative vorticity, divergence, strain, Turner angle, etc.) for one
or more LLC4320 timestamps.  The implementation follows the pattern established
in Section 5 of `fronts/properties/nb/Turner_Angle_Global_Viz.ipynb`.

---

## Existing Code to Reuse

| Module | What it provides |
|---|---|
| `fronts.properties.io` | `get_global_front_output_path()` for standardized file paths; `load_front_index()` for parquet I/O |
| `fronts.properties.colocation` | `_load_property_file()` to load NetCDF/npy property arrays; `colocate_fronts_with_properties()` for per-front statistics |
| `fronts.properties.characteristics` | `turner_angle()` for derived fields |
| `fronts.properties.defs` | `ocean_field_defs` for units, equations, descriptions |
| `fronts.properties.analysis.jpdf` | `compute_jpdf()`, `conditional_mean()`, `JPDFAccumulator` |

---

## Architecture

Two new files:

1. **`fronts/properties/io.py`** (extend existing) -- add data-loading helpers
2. **`fronts/viz/properties.py`** (new) -- visualization methods

All methods operate on plain numpy arrays / pandas DataFrames and use
matplotlib + cartopy.  No classes -- methods only.

---

## Module 1: I/O Additions (`fronts/properties/io.py`)

The loading logic is split into small, independently useful methods so that
callers can load exactly what they need (e.g. just coordinates, or just
one property array) without pulling in the full pipeline.  Two orchestrator
methods are provided for convenience.

---

### Atomic loaders -- front results

Each of these uses `get_global_front_output_path()` internally to resolve
the standardized filename from `(results_dir, time_str, run_tag)`.

#### `load_metadata(results_dir, time_str, run_tag) -> dict`
Load the JSON metadata file.  Returns the parsed dict (keys include
`shape`, `num_fronts`, `downsample_factor`, etc.).

#### `load_labeled_array(results_dir, time_str, run_tag) -> np.ndarray`
Load the labeled-fronts `.npy` file.  Returns a 2-D integer array.

#### `load_geometry_table(results_dir, time_str, run_tag) -> pd.DataFrame`
Load the geometry parquet (one row per front: label, name, bbox, centroid,
length, orientation, curvature, etc.).

#### `load_colocation_table(results_dir, time_str, run_tag) -> pd.DataFrame`
Load the colocation/properties parquet (one row per front: flabel, npix,
per-property statistics like `gradb2_median`, `relative_vorticity_mean`, ...).

#### `merge_geometry_colocation(df_geometry, df_colocation) -> pd.DataFrame`
Inner-merge on `label == flabel`.  Returns the enriched DataFrame.

---

### Coordinate handling

#### `load_llc_coords(coords_file=None, downsample_factor=None) -> (lat, lon)`
Load lat/lon from the LLC coordinate NetCDF.  Handles both `lat`/`lon`
and `YC`/`XC` variable names.  Optionally downsamples by
`downsample_factor`.  Defaults `coords_file` to
`$OS_OGCM/LLC/Fronts/coords/LLC_coords_lat_lon.nc`.

#### `compute_longitude_shift(lon) -> int`
Compute the column shift needed so that the longitude axis runs -180 to
+180.  Finds the column of the minimum longitude in the middle row and
returns `-min_col` (pass to `np.roll(..., shift, axis=1)`).  Returns 0
if no roll is needed.

#### `roll_to_pm180(*arrays, shift) -> tuple`
Roll one or more 2-D arrays by `shift` columns along axis 1.  Convenience
wrapper around `np.roll` that handles the common pattern of rolling lat,
lon, labeled, and property arrays together.  Returns a tuple of rolled
arrays in the same order.

---

### Property-file loading

#### `property_file_path(property_name, timestamp, version, properties_dir=None) -> Path`
Construct the standard NetCDF path for a derived property field:
`{properties_dir}/LLC4320_{timestamp}_{property_name}_v{version}.nc`.
Defaults `properties_dir` to `$OS_OGCM/LLC/Fronts/derived`.

#### `load_single_property(property_name, timestamp, version, properties_dir=None) -> np.ndarray`
Load one property array from its NetCDF file.  Uses `_load_property_file()`
internally, squeezes singleton dims.

#### `load_property_arrays(property_names, timestamp, version, properties_dir=None, downsample_factor=None, shift=0) -> dict`
Batch-load multiple properties into a `{name: np.ndarray}` dict.  Calls
`load_single_property()` for each, then optionally downsamples and rolls.

---

### Orchestrator (convenience)

#### `load_global_front_results(results_dir, time_str, run_tag, coords_file=None) -> dict`
One-call loader that calls all the atomic methods above and returns a dict:
```python
{
    'metadata':       dict,
    'labeled_global': np.ndarray,
    'df_enriched':    pd.DataFrame,
    'lat_global':     np.ndarray,
    'lon_global':     np.ndarray,
    'shift':          int,           # longitude roll applied
}
```
Steps:
1. `load_metadata()`
2. `load_labeled_array()`
3. `load_geometry_table()` + `load_colocation_table()` + `merge_geometry_colocation()`
4. `load_llc_coords()` with `downsample_factor` from metadata
5. `compute_longitude_shift()`
6. `roll_to_pm180()` on lat, lon, labeled

---

## Module 2: Visualization Methods (`fronts/viz/properties.py`)

All plotting methods accept pre-loaded data (arrays, DataFrames) so they
remain decoupled from I/O.  Each returns the matplotlib Figure for further
customization or saving.

### `plot_global_property_map(lon, lat, data, ...)`

Cartopy Robinson projection map of a gridded 2-D property field.

Parameters:
- `lon, lat` -- 2-D coordinate arrays
- `data` -- 2-D property array (same shape)
- `cmap` -- colormap (default `'RdBu_r'`)
- `symmetric` -- force symmetric colorbar around zero (default `True`)
- `clip_pct` -- percentile clipping for colorbar range (default 2)
- `downsample` -- spatial downsample factor for rendering (default 4)
- `mask_seams` -- blank LLC4320 tile-boundary seams (default `True`)
- `title`, `clabel` -- plot title and colorbar label
- `ax` -- optional existing GeoAxes

Logic (from notebook Section 5, Figure 1):
1. Downsample lat/lon/data
2. Mask seam rows where adjacent lat jumps >5 deg or lon jumps >90 deg
3. Clip colorbar to `clip_pct` percentile
4. `pcolormesh` on Robinson projection with land, coastlines, gridlines

### `plot_binned_front_map(df, property_col, ...)`

Cartopy map of a per-front property spatially binned by centroid lat/lon.

Parameters:
- `df` -- DataFrame with `centroid_lat`, `centroid_lon`, and property column
- `property_col` -- column name to visualize
- `n_lat_bins`, `n_lon_bins` -- spatial bin counts (default 90, 180 = 2 deg)
- `statistic` -- binning statistic (`'mean'`, `'median'`, `'count'`, etc.)
- `min_count` -- minimum fronts per bin to display (default 2)
- `cmap`, `symmetric`, `clip_pct`, `title`, `clabel`, `ax`

Logic (from notebook Section 5, spatial binning):
1. `binned_statistic_2d` on centroid lat/lon
2. Mask bins with count < `min_count`
3. Render on Robinson projection

### `plot_property_jpdf(x_values, y_values, ...)`

JPDF of two per-front or per-pixel properties (e.g. Turner angle vs |nabla b|^2).

Parameters:
- `x_values, y_values` -- 1-D arrays of the two variables
- `n_x_bins, n_y_bins` -- bin counts
- `x_range, y_range` -- explicit ranges (auto from percentiles if None)
- `y_log` -- log-scale y-axis (default `True`)
- `cmap` -- colormap (default `'Reds'`)
- `xlabel, ylabel, title`
- `annotations` -- list of dicts for reference lines/text labels
- `ax`

Logic (from notebook Section 5, Figure 2):
1. Compute 2-D histogram with uniform x-bins and log-spaced y-bins (if `y_log`)
2. Normalize to PDF
3. Display with `LogNorm` colorscale
4. Overlay annotation lines and region labels

### `plot_multi_timestamp(plot_fn, data_per_time, ...)`

Convenience wrapper that calls any of the above plot methods for multiple
timestamps, arranging results in a row of subplots.

Parameters:
- `plot_fn` -- one of the plotting functions above
- `data_per_time` -- dict mapping timestamp string to kwargs for `plot_fn`
- `ncols` -- columns per row (default 3)
- `figsize_per_panel` -- (width, height) per subplot

---

## Data Flow

```
  Atomic loaders (pick what you need)        Orchestrator (loads everything)
  ──────────────────────────────────         ────────────────────────────────
  load_metadata()                            load_global_front_results()
  load_labeled_array()                         calls all atomic loaders
  load_geometry_table()                        returns dict with all results
  load_colocation_table()
  merge_geometry_colocation()
  load_llc_coords()
  compute_longitude_shift()
  roll_to_pm180()

  Property loaders
  ────────────────
  property_file_path()
  load_single_property()      ─┐
  load_property_arrays()       ◄── batch wrapper
                           │
                           v
            df_enriched, lat/lon, labeled_global,
            property_arrays (dict of 2-D fields)
                           │
          ┌────────────────┼────────────────┐
          v                v                v
  plot_global_        plot_binned_     plot_property_
  property_map()      front_map()        jpdf()
          │                │                │
          └───────┬────────┴────────┬───────┘
                  v                 v
          plot_multi_timestamp()  (direct use)
```

---

## Implementation Order

1. **I/O helpers** in `fronts/properties/io.py`
   - Atomic front-result loaders: `load_metadata`, `load_labeled_array`,
     `load_geometry_table`, `load_colocation_table`, `merge_geometry_colocation`
   - Coordinate handling: `load_llc_coords`, `compute_longitude_shift`,
     `roll_to_pm180`
   - Property loading: `property_file_path`, `load_single_property`,
     `load_property_arrays`
   - Orchestrator: `load_global_front_results`

2. **Core viz methods** in `fronts/viz/properties.py`
   - `plot_global_property_map()` -- gridded field on map
   - `plot_binned_front_map()` -- per-front binned map
   - `plot_property_jpdf()` -- 2-D histogram / JPDF

3. **Multi-timestamp wrapper**
   - `plot_multi_timestamp()`

4. **Update `fronts/viz/__init__.py`** to import the new module.

5. **Smoke test** with a notebook or script in `dev/properties/` that
   reproduces the Turner Angle Section 5 plots using the new methods.

---

## Design Decisions

- **Methods, not classes** (per project guidelines).
- **No derived-field computation inside viz** -- Turner angle, Rossby number
  etc. are computed upstream (via `characteristics.py` or the JPDF module)
  and passed in as data.  Viz methods are field-agnostic.
- **`ocean_field_defs`** can optionally supply default units/labels when the
  caller doesn't provide them, but the viz methods work without it.
- **`$OS_OGCM`** environment variable for all data paths (per project
  convention), never hardcoded `/mnt/tank`.
- **Cartopy Robinson projection** for global maps (matches the notebook).
- **Return `Figure`** from every plot method so callers can save or further
  customize.
