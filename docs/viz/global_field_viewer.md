# `global_field_viewer` — full-globe field viewer with one or two front masks

[fronts/scripts/global_field_viewer.py](../../fronts/scripts/global_field_viewer.py) opens a PyQt6 + pyqtgraph GUI that displays a single 2-D field across the full LLC4320 global grid (12960 × 17280), with one or two front-mask overlays.

## What it shows

A single panel covering the whole global rect grid, with up to three layers:

1. **Field underlay** — gradb2 (default) or any other field readable from a NetCDF file the user supplies.
2. **Fronts overlay (red)** — primary front mask, expected as a binary `.npy` file (1 = front, 0 = no front).
3. **Fronts overlay (blue, optional)** — second mask via `--fronts2`, useful when comparing two detection runs.

NaN pixels in the field are also overlaid in dark green.

## What it's for

The "is the front-detection coverage what I expect at global scale?" question. Pan/zoom around continents, current systems, and gyres to spot regions where the algorithm is under- or over-firing, and compare two configurations side-by-side without rerunning anything.

## Interactive controls

| Control | Effect |
|---|---|
| Pan / scroll-wheel zoom | Standard pyqtgraph. |
| `Show Fronts (red)` checkbox | Toggle the primary mask. |
| `Show Fronts 2 (blue)` checkbox | Toggle the secondary mask (only if `--fronts2` was supplied). |
| `Divergent cmap` checkbox | Switch the underlay to a blue-white-red diverging map centred at zero. |
| `Log₁₀ Scale` checkbox | Replace the underlay with `log10(|x|)`. |
| `Reset View` button | Snap back to the full-globe extent. |
| `Adjust Limits to View` button | Recompute display levels from the currently visible viewport. |
| Contrast slider (1–99) | Percentile cutoff for display levels. Default 95. |

## CLI

```bash
python -m fronts.scripts.global_field_viewer \
    LLC4320_2012-11-09T12_00_00_gradb2_v3.nc \
    fronts.npy \
    --fronts2 fronts2.npy \
    --downsample 5
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `data_file` (positional) | path | — | NetCDF carrying the 2-D field. |
| `fronts_file` (positional) | path | — | Primary binary fronts `.npy` (1 = front, 0 = no front). |
| `--fronts2` | path | — | Optional second binary fronts file (rendered in blue). |
| `--field` | str | `gradb2` | Name of the variable to read from `data_file`. |
| `--divergent` | flag | off | Start with the diverging colormap (matches the GUI checkbox). |
| `--downsample / -d` | int | `1` | Stride applied to both the field and the fronts before display. `--downsample 5` makes the global view much more responsive on modest hardware. |

## How the data is loaded

- The field is opened with xarray and indexed by `--field`.
- Fronts files are `np.load`-ed; first arg is the primary mask, `--fronts2` is the secondary.
- Colormap / level / overlay logic comes from [fronts/viz/viz_utils.py](../../fronts/viz/viz_utils.py): `make_colormap`, `compute_levels`, `make_fronts_rgba`, `make_nan_rgba`.

## When to use this vs. the other viewers

- Use [global_field_viewer](global_field_viewer.md) for **whole-globe** browsing of one field at a time and side-by-side comparison of two mask runs.
- Use [front_property_viewer](front_property_viewer.md) for a **regional bbox** with four fields linked.
- Use [front_viz_groups_bokeh](front_viz_groups_bokeh.md) for **per-front property tooltips** in a browser.
