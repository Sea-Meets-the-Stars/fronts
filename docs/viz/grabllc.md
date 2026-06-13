# `grabllc` — extract cutouts and (optionally) plot panels for one preproc table entry

[fronts/scripts/grabllc.py](../../fronts/scripts/grabllc.py) loads one row from a preprocessed parquet table (keyed by `UID`), grabs the corresponding LLC4320 cutout at a fixed physical size, and emits the result as a matplotlib panel of fields and/or a saved data file. This script is more of a data-extraction helper than a pure visualisation tool; the figure is one of several outputs.

## What it does

1. **Look up the row.** Opens the parquet at `table_file`, locates the row with `df.UID == UID`, reads `idf.datetime`, `idf.row`, `idf.col`.
2. **Load the LLC dataset** for that datetime (`llc_io.grab_llc_datafile` + `llc_io.load_llc_ds`).
3. **Convert a physical size to pixel size.** Uses the LLC `lat` grid at `(row, col)` to compute a per-pixel km, then chooses `dr = dc = round(fixed_km / per-pixel km)`.
4. **For every requested field:** extract via `llc_extract.field_from_ds`, slice `[row:row+dr, col:col+dc]`, then resize to a square `(field_size, field_size)` with `skimage.transform.resize_local_mean`.
5. **Plot** (if `--show` or `--fig_file` is set) one matplotlib panel per field via `fronts.plotting.images.show_image`, side-by-side in a single row.
6. **Save** the raw cutouts to `--data_file` (if supplied).

## What it's for

A single-call way to inspect or persist a fixed-size physical cutout (e.g. 144 km on a side) for one entry in a preproc evaluation table. Common workflow: train a model, score it, pick a UID, and re-extract the underlying cutout to look at what the model was given.

## CLI

```bash
python -m fronts.scripts.grabllc table.parquet 12345 \
    --fields SST,SSS,Divb2 \
    --field_size 64 \
    --fixed_km 144 \
    --fig_file /tmp/uid12345.png
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `table_file` (positional) | path | required | Parquet preproc table; must contain `UID`, `datetime`, `row`, `col`. |
| `UID` (positional) | int | required | The `UID` value of the row to extract. |
| `--fields` | comma-separated str | `SST,SSS,Divb2` | Fields to extract; one panel per field. |
| `--fig_file` | path | — | If set, save the assembled matplotlib panel here. |
| `--data_file` | path | — | If set, persist the raw cutouts (the script's `idict` dictionary) here. |
| `--field_size` | int | `64` | Output size (pixels) per cutout after `resize_local_mean`. |
| `--fixed_km` | float | `144.0` | Physical side length of the cutout in km, used to pick the LLC pixel window. |
| `-s / --show` | flag | off | Open the matplotlib window interactively. |

## Outputs

- Optional matplotlib PNG (`--fig_file`) with one panel per field.
- Optional data file (`--data_file`) holding the raw cutouts in a dictionary.

## When to use this vs. the other viewers

- Use [grabllc](grabllc.md) when you want **the cutout for one UID** at a fixed physical size — for spot-checking preproc rows or saving inputs for downstream analysis.
- Use [front_property_viewer](front_property_viewer.md) for **interactive bbox exploration** of multiple fields at full resolution.
- Use [global_field_viewer](global_field_viewer.md) for **whole-globe** scanning.
- Use [front_viz_groups_bokeh](front_viz_groups_bokeh.md) for **per-front property tooltips**.
- Use [fronts_viz_3d](fronts_viz_3d.md) for a **3-D view** of one labelled front through a density volume.
