# Visualisation scripts in `fronts`

Index of every script under [fronts/scripts/](../../fronts/scripts/) that produces a figure or interactive view, plus the shared modules they pull from.

These scripts cover four different visual stories and use four different rendering backends. Pick by what you want to look at:

| Script | What it shows | Backend | Output |
|---|---|---|---|
| [fronts_viz_3d.py](fronts_viz_3d.md) | One labelled front extruded through a 3-D sigma0 volume down to a few levels below the MLD. | PyVista (VTK) | PNG + interactive HTML + 2-D matplotlib inset |
| [front_property_viewer.py](front_property_viewer.md) | Four linked 2-D panels in a regional bbox: gradb2 + binary fronts + three derived fields you choose. | PyQt6 + pyqtgraph | Interactive GUI window |
| [global_field_viewer.py](global_field_viewer.md) | The full global LLC4320 field at one timestep with one or two front masks overlaid. | PyQt6 + pyqtgraph | Interactive GUI window |
| [front_viz_groups_bokeh.py](front_viz_groups_bokeh.md) | Labelled-front background with per-front properties exposed as hover tooltips. | Bokeh | Standalone HTML |
| [grabllc.py](grabllc.md) | Static cutout panels for one entry in a preproc table. Used to extract or inspect LLC data on a UID basis. | matplotlib | Window / PNG |

## Shared modules

The 3-D pipeline introduced a couple of re-usable modules; the rest of the viz scripts share a small pyqtgraph helper module.

| Module | Used by | Description |
|---|---|---|
| [fronts/viz/pv_helpers.py](../../fronts/viz/pv_helpers.py) | fronts_viz_3d | Generic PyVista wrappers: `ensure_display` (Xvfb), `new_plotter`, `scientific_theme`, `add_scalar_field`, `save_with_rst`. |
| [fronts/viz/fronts_3d.py](../../fronts/viz/fronts_3d.py) | fronts_viz_3d | Front-specific PyVista builders: `front_bbox_and_crop`, `truncate_depth`, `build_pyvista_grid`, `decompose_front_branches`, `build_front_curtain`, `pick_isopycnals_across_front`, `render_3d`. |
| [fronts/viz/insets.py](../../fronts/viz/insets.py) | fronts_viz_3d | Matplotlib 2-D companion inset (`plot_bbox_inset`). Kept separate from `viz_utils.py` which is pyqtgraph-based. |
| [fronts/viz/viz_utils.py](../../fronts/viz/viz_utils.py) | front_property_viewer, global_field_viewer | Pyqtgraph helpers: `make_colormap`, `compute_levels`, `make_fronts_rgba`, `make_nan_rgba`. |
| [fronts/llc/analysis.py](../../fronts/llc/analysis.py) | fronts_viz_3d (and `dev/mld/plot_top_N_density_profiles.py` after the v1 refactor) | Stratification diagnostics: scalar `mixed_layer_depth` and vectorised `mixed_layer_depth_field`. The threshold is a parameter, so the same helper covers the 0.03 ("mixed layer") and 0.125 ("isopycnal MLD") conventions. |

## Running on a headless machine

`fronts_viz_3d.py` and `grabllc.py` both render off-screen and write files, so they work in a non-graphical environment.

The pyqtgraph viewers (`front_property_viewer.py`, `global_field_viewer.py`) and the Bokeh script need a display server or, at minimum, a browser; they open interactive windows and are not meant to be batched.

For PyVista off-screen on Linux, `pv_helpers.ensure_display()` auto-starts Xvfb when `$DISPLAY` is empty. Install Xvfb via the system package manager (`apt-get install xvfb` or similar) once; the script handles the rest.

## Environment

All five scripts are exercised on the `ocean14` conda env. The `pyvista[jupyter]` extras (Trame) are required for the interactive-HTML export from `fronts_viz_3d.py`; the rest of the scripts depend only on packages that were already in the env.
