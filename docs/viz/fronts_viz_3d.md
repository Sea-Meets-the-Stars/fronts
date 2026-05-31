# `fronts_viz_3d` — 3-D rendering of a single labelled front through a density volume

[fronts/scripts/fronts_viz_3d.py](../../fronts/scripts/fronts_viz_3d.py) renders one labelled front from the LLC4320 V4 catalogue as a sigma0-coloured "curtain" extruded through a 3-D potential-density volume, clipped to a few model levels below the local mixed-layer depth. A 2-D matplotlib inset (surface σ₀ + the front + lon/lat ticks) is written alongside the 3-D PNG as a context map.

## What it produces

Three files per run:

- **3-D PNG** (at `--output`) — the PyVista scene; isopycnal surfaces *or* a volume render of σ₀ in the cropped bbox, with the selected front overlaid as a sigma0-coloured vertical ribbon.
- **Interactive HTML** (at `--interactive-html`; default `{stem}.html`) — the same scene exported via PyVista's Trame back-end. Open in a browser to rotate, zoom, and toggle layers.
- **2-D inset PNG** (default `{stem}_inset.png`; opt out with `--no-inset`) — a matplotlib pcolormesh of surface σ₀ in the bbox with the selected front overlaid in red and lon/lat secondary axes attached.

## Inputs

1. **3-D density tile** — NetCDF written by [llc4320-native-grid-preprocessing/src/dbof/tiles/generate_tile.py](../../../llc4320-native-grid-preprocessing/src/dbof/tiles/generate_tile.py). Carries `sigma0(k, j, i)` on **face-local** axes (k=51, j=720, i=720), 2-D `XC`/`YC`, 1-D `Z`, plus the `rect_i_start`/`rect_j_start`/`face_index` provenance attrs.
2. **Labelled-fronts mask** — global `.npy` mask on the **rect grid** (12960 × 17280, integer labels, `0 = no front`). For V4 this is the `labeled_fronts_global_*_V4.npy` file — not the boolean `*_bfronts.npy`.
3. **Locator** — either `(--i, --j)` global rect indices, or `(--lat, --lon)` degrees.

## Frame handling

The density tile is on face-local axes; the labels mask is on the rect grid. The script reuses [dev/mld/density_utils.py](../../dev/mld/density_utils.py) `build_tile_lookup` to get the per-pixel `(j_face, i_face)` for every rect-tile-local pixel, then fancy-indexes σ₀, XC, and YC onto the rect frame. After this everything downstream — bbox, MLD, isopycnal pick, curtain construction — operates in one coordinate system that matches the user's `--i / --j` input.

## Algorithm flow

1. **Load** — `load_density_tile` + `load_labels_tile` (both from `dev/mld/density_utils.py`) read the tile NetCDF and the cropped labels window.
2. **Remap** — `remap_to_rect` fancy-indexes any face-local 2-D or 3-D array onto the rect-tile-local frame.
3. **Pick label** — `pick_front_label`:
    - `(--i, --j)` looks up `labels_tile[j_local, i_local]` and errors clearly if it's 0.
    - `(--lat, --lon)` finds the nearest pixel via squared distance in `(XC_rect, YC_rect)`, then snaps to the nearest non-zero label inside the tile using `scipy.ndimage.distance_transform_edt(..., return_indices=True)`.
4. **Crop** — `front_bbox_and_crop` returns `(j_slice, i_slice)` covering the front's bounding box plus `--margin` (default 50 px).
5. **MLD** — `mixed_layer_depth_field` (vectorised) returns `(z_mld(j,i), k_mld(j,i))` for the cropped column with the 0.125 kg m⁻³ threshold and 10 m reference depth.
6. **Depth clip** — `truncate_depth` keeps levels `0 .. max(k_mld) + n_below` (default `--n-below=3`), so the rendered volume reaches a handful of model levels below the deepest MLD in the bbox.
7. **Isopycnal pick** — `pick_isopycnals_across_front` either uses `--isopycnals` if supplied, or auto-picks 5 levels evenly spaced between the 5th and 95th percentile of σ₀ at the LLC level closest to 10 m, with the front + 1-pixel buffer masked out. The chosen values are printed to stdout so they can be pinned via `--isopycnals` on the next run.
8. **Grid** — `build_pyvista_grid` wraps the cropped+clipped σ₀ as a `pyvista.RectilinearGrid` (regular in i,j, irregular in z) with the field stamped in Fortran order to satisfy VTK.
9. **Curtain** — `decompose_front_branches` (custom 8-neighbour DFS, no `skan` dependency) splits the front mask into single-branch polylines; `build_front_curtain` extrudes each branch into a `pyvista.StructuredGrid` ribbon and samples σ₀ along it via `scipy.ndimage.map_coordinates`. Returns a `pyvista.MultiBlock` (one entry per branch).
10. **Render** — `render_3d` adds either `grid.contour(levels)` ("isopycnals" mode) or `pl.add_volume(grid, opacity=...)` ("volume" mode) as the background, then iterates the MultiBlock to draw every branch ribbon. Only the first branch contributes a scalar bar to avoid legend duplication.
11. **Save** — `save_with_rst` from `pv_helpers` writes the 3-D PNG and (unless `--no-inset`) the interactive HTML in one call; the script then calls `plot_bbox_inset` from [fronts/viz/insets.py](../../fronts/viz/insets.py) for the 2-D companion.

## CLI

```text
python -m fronts.scripts.fronts_viz_3d \
    --density-tile density_tile330_20121109T12.nc \
    --labels       labeled_fronts_global_20121109T12_00_00_V4.npy \
    --i 13310 --j 9628 \
    --output /tmp/fronts_viz_3d_calcurrent.png
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--density-tile` | path | required | NetCDF density tile from `generate_tile.py`. |
| `--labels` | path | required | Global integer-labels mask (`.npy` or `.nc`). |
| `--i / --j` | int | — | Rect-grid pixel coordinates (locator option A). |
| `--lat / --lon` | float | — | Degrees (locator option B); snapped to the nearest labelled pixel. |
| `--output` | path | required | 3-D PNG output path. |
| `--interactive-html` | path | `{stem}.html` | Trame-exported interactive HTML scene. |
| `--no-inset` | flag | off | Skip the 2-D companion figure. |
| `--inset-output` | path | `{stem}_inset.png` | Override the inset PNG path. |
| `--mode` | `isopycnals` / `volume` | `isopycnals` | Background rendering style. |
| `--isopycnals` | floats | auto-picked | Explicit σ₀ contour values for `--mode isopycnals`. |
| `--opacity` | `sigmoid` / `linear` / `geom` | `sigmoid` | Opacity transfer for `--mode volume`. |
| `--clim` | LO HI | 2/98 percentile | σ₀ colour limits. |
| `--cmap-volume` | str | `viridis` | Colormap for the volume/isopycnal background. |
| `--cmap-curtain` | str | `magma` | Colormap for the front curtain. |
| `--zscale` | float | `50.0` | Vertical exaggeration of the depth axis. See "tuning" below. |
| `--margin` | int | `50` | Pixel margin around the front's bbox. |
| `--n-below` | int | `3` | LLC levels below the deepest MLD to include. |
| `--cpos` | JSON | — | Camera-position triple to lock framing across runs. |
| `--show` | flag | off | Open an interactive window instead of off-screen rendering. Hard-errors if `$DISPLAY` is empty. |

## Tuning notes

- **`--zscale`**: the default `50.0` was chosen when the typical bbox was expected to span ~720 pixels (full tile). For a tightly cropped front (~100–150 pixels) clipped at the MLD (~80 m), `50×` produces a 30:1 needle that's hard to read. Try `--zscale 1.0` or `--zscale 2.0` for cropped scenes; an adaptive default is on the TODO list.
- **`--clim`**: the default 2/98 percentile is computed over the depth-clipped *volume*, which spans the full vertical density gradient. The cross-front contrast at the surface is usually much narrower (e.g. 24.04–24.29 vs. a 24.03–25.03 volume range), so the curtain ends up close to one end of the colour ramp. Pass `--clim` explicitly with the printed isopycnal values if you want the curtain to span the full ramp.
- **`--cpos`**: `save_with_rst` returns the camera position used; capture it from the first run and pass it back via `--cpos '[[...], [...], [...]]'` (JSON) to get pixel-stable figures on subsequent runs.
- **Branch decomposition**: V4 fronts are thinned skeletons but can branch at Y-junctions and form closed loops. The custom DFS handles all cases, but 8-connectivity treats every corner of an orthogonal loop as a junction, so a square ring decomposes into many short branches. Edge coverage is correct (every edge appears once across the MultiBlock).
- **Hard errors**: bad locators (`--i/--j` lands on label 0, `--lat/--lon` outside the tile, `--show` with no `$DISPLAY`) exit non-zero with a clear diagnostic — no fallback figure is produced.

## Module map

| File | Role |
|---|---|
| [fronts/scripts/fronts_viz_3d.py](../../fronts/scripts/fronts_viz_3d.py) | CLI thin wrapper; orchestrates everything. |
| [fronts/viz/fronts_3d.py](../../fronts/viz/fronts_3d.py) | Front-specific PyVista builders. Re-usable by future 3-D scripts. |
| [fronts/viz/pv_helpers.py](../../fronts/viz/pv_helpers.py) | Generic PyVista helpers (scientific theme, off-screen Xvfb, scalar bar, PNG + HTML export). |
| [fronts/viz/insets.py](../../fronts/viz/insets.py) | Matplotlib 2-D companion inset. |
| [fronts/llc/analysis.py](../../fronts/llc/analysis.py) | `mixed_layer_depth` (scalar) and `mixed_layer_depth_field` (vectorised). |
| [dev/mld/density_utils.py](../../dev/mld/density_utils.py) | Reused for `load_density_tile`, `load_labels_tile`, `build_tile_lookup`, `attach_lonlat_twins`. |

## Smoke test recipe

```bash
conda activate ocean14
python -m fronts.scripts.fronts_viz_3d \
    --density-tile "$OS_OGCM/LLC/Fronts/V4/20121109_120000/tiles/density_tile330_20121109T12.nc" \
    --labels       "$OS_OGCM/LLC/Fronts/V4/20121109_120000/labeled_fronts_global_20121109T12_00_00_V4.npy" \
    --i 13310 --j 9628 \
    --zscale 1.0 \
    --output /tmp/fronts_viz_3d_calcurrent.png
```

That command selects label 107558 at lon=−120.70, lat=31.15 (California Current). The MLD inside the bbox sits around z=−60 m, clipped to z=−88 m; auto isopycnals are picked around 24.04–24.29 kg m⁻³.

## Known follow-ups

- Adaptive default for `--zscale` (`max(j_extent, i_extent) / |Z_clipped[-1]|`).
- The cross-front isopycnal-bracket method (round-3 option (i)) uses surface percentiles; option (ii) (local normals + per-pixel sampling) was deferred.
- The 8-connectivity branch decomposition could be reduced to a 4-connectivity walk to avoid spurious junctions on orthogonal loops; not pursued because branches that share endpoints render correctly as a union of ribbons.
