# `front_viz_groups_bokeh` — Bokeh viewer with per-front property tooltips

[fronts/scripts/front_viz_groups_bokeh.py](../../fronts/scripts/front_viz_groups_bokeh.py) builds a standalone Bokeh HTML page that pairs a background field (e.g. `gradb2`) with the labelled-fronts mask from the *groups* table, so you can hover over any front pixel and see the corresponding per-front property values in a side panel.

## What it shows

One Bokeh figure with two raster layers and a controls/tooltip column:

1. **Background field** — `gradb2` by default, rendered as either a grayscale or diverging RGBA image.
2. **Labelled fronts overlay** — every connected front coloured by its integer label, drawn semi-transparent in red (or yellow when the diverging cmap is active).
3. **Controls** — a `MultiSelect` to pick which per-front property columns appear in the tooltip; a `Select` for grey vs. diverging cmap; a tooltip `Div` that updates on hover.

## What it's for

Quickly answering "what are the measured properties of *this particular* front?" — gradient magnitude, strain rate, normalised vorticity, length, age, etc. — without leaving the regional context. The hover/select interaction makes it ergonomic to walk along a feature and read off properties pixel by pixel.

## Interactive controls

| Control | Effect |
|---|---|
| Pan / wheel zoom / box zoom | Standard Bokeh tools (the script wires `active_scroll="wheel_zoom"`). |
| `Reset` | Returns to the full bbox view. |
| `Save` | Bokeh's built-in PNG snapshot. |
| **MultiSelect** of properties | Pick which columns from the groups-table parquet show in the tooltip. |
| **Hover** anywhere | Bokeh CustomJS reads the labelled-fronts array, looks up the front label, and renders the selected property values in the side `Div`. Hovering over a 0-label pixel clears the tooltip. |
| **cmap Select** | Swap the background between grey and diverging RGBA. |

## CLI

```bash
python -m fronts.scripts.front_viz_groups_bokeh \
    2012-11-09T12_00_00 \
    --field gradb2 \
    --bbox 100 200 500 600
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `timestamp` (positional) | str | required | LLC timestamp, e.g. `2012-11-09T12_00_00`. |
| `--field` | str | `gradb2` | Variable to read as the background field. |
| `--config_lbl` | str | `A` | Config label for the labels + properties files (e.g. `A`, `B`). |
| `--version` | str | `1` | Data version string. |
| `--bbox X0 Y0 X1 Y1` | four ints | — | Pixel bbox; required if `--latlon_bbox` is not given. |
| `--latlon_bbox LAT0 LON0 LAT1 LON1` | four floats | — | Lat/lon bbox; converted to pixels via `latlon_to_pixel_bbox` defined inside this script. Mutually exclusive with `--bbox`. |

## Where the files come from

The script reads three artefacts from the **groups** pipeline at `$OS_OGCM/LLC/Fronts/group_fronts/v{version}/`:

- **`labeled_fronts_global_{YYYYMMDDTHH_MM_SS}_v{version}_bin_{config_lbl}.npy`** — int label mask (one int per front, 0 = no front).
- **`front_properties_{YYYYMMDDTHH_MM_SS}_v{version}_bin_{config_lbl}.parquet`** — one row per labelled front, with property columns (`gradb2_median`, `strain_mag_median_over_f`, etc.).
- **Background field NetCDF** — same source as [global_field_viewer](global_field_viewer.md), pulled via `fronts.llc.io`.

The `_safe_timestamp` helper at the top of the script normalises `2012-11-09T12_00_00` into the directory's `20121109T12_00_00` convention.

## Output

A single self-contained HTML file written by `bokeh.io.output_file` then opened with `bokeh.plotting.show`. The interaction is implemented client-side via Bokeh `CustomJS`, so the file works without a server once written.

## When to use this vs. the other viewers

- Use [front_viz_groups_bokeh](front_viz_groups_bokeh.md) when you want to **inspect per-front properties** in a browser-friendly format and the **groups parquet** already exists.
- Use [front_property_viewer](front_property_viewer.md) for **four linked fields in one bbox** without per-front property tooltips.
- Use [global_field_viewer](global_field_viewer.md) for **full-globe** browsing with one or two mask overlays.
- Use [fronts_viz_3d](fronts_viz_3d.md) for a **3-D extruded view** of one labelled front down to the MLD.
