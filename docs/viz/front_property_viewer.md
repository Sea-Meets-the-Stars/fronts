# `front_property_viewer` — four-panel regional viewer of fronts + derived fields

[fronts/scripts/front_property_viewer.py](../../fronts/scripts/front_property_viewer.py) opens a PyQt6 + pyqtgraph GUI that shows four 2-D panels of LLC4320 data inside a user-supplied bounding box, with the binary fronts mask overlaid on the gradb2 panel and three derived fields you pick at launch.

## What it shows

Four linked panels arranged in a 2×2 grid:

| Panel | Field |
|---|---|
| 0 (top-left) | `gradb2` with the binary fronts overlay (red, semi-transparent). |
| 1 (top-right) | First derived field from `--fields`. |
| 2 (bottom-left) | Second derived field. |
| 3 (bottom-right) | Third derived field. |

All four panels share their pan/zoom view, so dragging in any panel updates the others.

## What it's for

A fast way to ask "given this regional bbox, how does the binary front mask line up with vorticity / strain rate / Okubo–Weiss / divergence / Fs?" without writing per-figure matplotlib code. Especially useful for sanity-checking new derived fields against the existing front detections.

## Interactive controls

| Control | Effect |
|---|---|
| Pan / scroll-wheel zoom inside any panel | Linked across all four. |
| `Show Fronts (red)` checkbox | Toggles the front overlay on panel 0. |
| `Divergent cmap` checkbox | Switches to blue-white-red centred at zero (recommended for vorticity, divergence, OW, …). The script already auto-selects divergent for a known set of fields (vorticity, OW, divergence, Fs, Turner, Ertel PV, …); this checkbox forces it for everything else. |
| `Log₁₀ Scale` checkbox | Replaces each panel's data with `log10(|x|)`. |
| Contrast slider (1–99) | Sets the percentile cutoff used to compute display levels. 95 (the default) clips the top/bottom 5 %. |
| `Reset View` | Resets pan/zoom to the full bbox. |
| `Adjust Limits to View` | Recomputes display levels from the data currently inside the visible viewport. |

## CLI

```bash
python -m fronts.scripts.front_property_viewer \
    2012-11-09T12_00_00 \
    --fields vorticity strain_rate OW \
    --bbox 100 200 500 600
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `timestamp` (positional) | str | required | LLC timestamp, e.g. `2012-11-09T12_00_00`. |
| `--fields F1 F2 F3` | three strings | required | Field names for panels 1–3. Must match the field-file naming convention used by `fronts.finding.io`. |
| `--version` | str | `4` | Data version string (which V-set to read). |
| `--bbox X0 Y0 X1 Y1` | four ints | — | Pixel bbox in the global LLC rect grid. |
| `--latlon_bbox LAT0 LON0 LAT1 LON1` | four floats | — | Lat/lon bbox; converted to pixels via `fronts.llc.coords.latlon_to_pixel_bbox`. Mutually exclusive with `--bbox`. |
| `--vl{0,1,2,3} VMIN,VMAX` | string | auto | Per-panel display levels, overriding the contrast slider for that panel. |
| `--cl{0,1,2,3} {blue,green,red}` | enum | auto | Per-panel single-hue colormap override. |

## How the data is loaded

- `gradb2`, fronts mask, and each derived field are pulled via `fronts.finding.io` and `fronts.llc.io` using the supplied `timestamp` + `--version` + `--config_lbl`.
- Colour maps and display levels come from [fronts/viz/viz_utils.py](../../fronts/viz/viz_utils.py): `make_colormap` (single-hue or divergent), `compute_levels` (percentile-based), `make_fronts_rgba` (red overlay), and `make_nan_rgba` (dark-green NaN overlay).

## When to use this vs. the other viewers

- Use [front_property_viewer](front_property_viewer.md) when you want **four fields side-by-side in one bbox** and you're already comfortable with the pyqtgraph idioms.
- Use [global_field_viewer](global_field_viewer.md) when you want to **scan the whole globe** for one field at a time and toggle one or two fronts masks.
- Use [front_viz_groups_bokeh](front_viz_groups_bokeh.md) when you want **per-front property tooltips** on hover (Bokeh, browser-based).
