# `fronts_viz_3d` — 3-D rendering of a single labelled front through a density volume

[fronts/scripts/fronts_viz_3d.py](../../fronts/scripts/fronts_viz_3d.py) renders one labelled front from the LLC4320 V4 catalogue against a 3-D potential-density volume. The front itself appears as a single **opaque, tilted σ₀ iso-surface** (the median of the cross-front contrast levels) so the cross-front density discontinuity reads at a glance — denser water on one side, lighter water on the other, with the surface tilting with depth. The broader **"waters near the front" context** is drawn as semi-transparent iso-surfaces over the cropped bbox, and the front's surface trace is marked with a small, semi-transparent red tube at the top of the volume.

The script also writes a **2-D matplotlib inset** alongside the 3-D PNG with surface σ₀, the front overlaid in red, and lon/lat secondary axes.

Optionally, the σ₀ iso-surfaces can be **colored by a second field** (e.g. Richardson number) supplied via `--field-tile` — the geometry stays density-driven while the colors carry the second field interpolated onto each density surface. See [Coloring isopycnals by a second field](#coloring-isopycnals-by-a-second-field-eg-richardson-number).

## What it produces

Three files per run:

- **3-D PNG** (at `--output`) — the PyVista scene.
- **Interactive HTML** (at `--interactive-html`; default `{stem}.html`) — Trame-exported, browser-rotatable.
- **2-D inset PNG** (default `{stem}_inset.png`; opt out with `--no-inset`) — surface σ₀ pcolormesh + front overlay + lon/lat twin axes.

## Inputs

1. **3-D density tile** — NetCDF written by [llc4320-native-grid-preprocessing/src/dbof/tiles/generate_tile.py](../../../llc4320-native-grid-preprocessing/src/dbof/tiles/generate_tile.py). `sigma0(k, j, i)` on **face-local** axes (k=51, j=720, i=720), 2-D `XC`/`YC`, 1-D `Z`, plus `rect_i_start`/`rect_j_start`/`face_index` provenance. **Always drives the geometry.**
2. **Field tile** *(optional, `--field-tile`)* — a second NetCDF from the same generator, same tile window + timestamp, holding a different property (e.g. `--property richardson` → `Ri(k, j, i)`). Its variable **colors** the density iso-surfaces. See [Coloring isopycnals by a second field](#coloring-isopycnals-by-a-second-field-eg-richardson-number).
3. **Labelled-fronts mask** — global `.npy` on the **rect grid** (12960 × 17280, integer labels, `0 = no front`). For V4 the integer-label file is `labeled_fronts_global_*_V4.npy` (the `*_bfronts.npy` neighbour is boolean only).
4. **Locator** — either `(--i, --j)` global rect indices, or `(--lat, --lon)` degrees.

## Frame handling

The density tile is on face-local axes; the labels mask is on the rect grid. The script reuses [dev/mld/density_utils.py](../../dev/mld/density_utils.py) `build_tile_lookup` to get the per-pixel `(j_face, i_face)` for every rect-tile-local pixel, then fancy-indexes σ₀, XC, YC onto the rect frame. After this everything downstream operates in one coordinate system that matches the user's `--i / --j` input.

## Algorithm flow

1. **Load** — `load_density_tile` + `load_labels_tile` (from `dev/mld/density_utils.py`) read the tile NetCDF and the cropped labels window.
2. **Remap** — `remap_to_rect` fancy-indexes face-local arrays onto the rect-tile-local frame.
3. **Pick label** — `pick_front_label`:
    - `(--i, --j)` looks up `labels_tile[j_local, i_local]` and hard-errors if 0.
    - `(--lat, --lon)` finds the nearest pixel via squared distance in `(XC_rect, YC_rect)`, then snaps to the nearest non-zero label inside the tile via `scipy.ndimage.distance_transform_edt(..., return_indices=True)`.
4. **Crop** — `front_bbox_and_crop` returns `(j_slice, i_slice)` over the front's bbox plus `--margin` (default 50 px).
5. **MLD** — `mixed_layer_depth_field` (vectorised, from [fronts/llc/analysis.py](../../fronts/llc/analysis.py)) returns `(z_mld(j,i), k_mld(j,i))` with the 0.125 kg m⁻³ Bodner-style threshold and 10 m reference depth.
6. **Depth clip** — `truncate_depth` keeps levels `0 .. max(k_mld) + n_below` (default `--n-below=3`).
7. **Isopycnal pick** — `pick_isopycnals_across_front` either uses `--isopycnals` if supplied, or auto-picks 5 levels evenly spaced between the 5th and 95th percentile of σ₀ at the LLC level closest to 10 m, masking out the front + 1-pixel buffer. The chosen values are printed to stdout for pinning on the next run.
8. **Grid** — `build_pyvista_grid` wraps the cropped+clipped σ₀ as a `pyvista.RectilinearGrid` (regular in i,j, irregular in z, Fortran-order ravel per the VTK convention). Accepts an optional `mask_2d=` to set σ₀ to NaN outside a 2-D region (not used in the default path).
9. **Front iso-surface (tilt indicator)** — `build_front_isosurface` contours the grid at the **median** of the cross-front σ₀ levels. Because density surfaces slope across a front, this single iso-surface naturally tilts with depth — denser water on one side, lighter on the other. It is rendered opaque in distinct orange so the tilt geometry pops against the semi-transparent background iso-surfaces.
10. **Background context** — `render_3d` adds either `grid.contour(levels)` (mode `isopycnals`, default) or `pl.add_volume(grid, ...)` (mode `volume`) as a semi-transparent backdrop showing the σ₀ field across the whole cropped bbox.
11. **Surface trace marker** — `build_front_top_marker` extrudes each branch polyline at `z = Z[0] * zscale` as a small tube (radius 0.8, opacity 0.6 by default) — a low-key red overlay on top of the volume identifying the surface footprint of the front. The vertical-sheet "curtain" that previous versions used is still computed (the marker reuses its branch decomposition) but is **off by default** since v1.4; pass `draw_curtain=True` to opt in.
12. **Camera** — `render_3d` sets a **south-east elevated camera** so that i increases left → right and j increases bottom → top on screen. The script captures `pl.camera_position` after `render_3d` and explicitly passes it to `save_with_rst` so the default isometric override inside `save_figure` doesn't reset it. Override per-run with `--cpos JSON`.
13. **Render & save** — `save_with_rst` from `pv_helpers` writes the 3-D PNG and the interactive HTML in one call; then the script calls `plot_bbox_inset` from [fronts/viz/insets.py](../../fronts/viz/insets.py) for the 2-D companion.

## Coloring isopycnals by a second field (e.g. Richardson number)

The scene **geometry is always density-driven**: the MLD depth clip, the
isopycnal contour levels, and the tilted front iso-surface are all computed
from σ₀. Passing `--field-tile` adds a *second* field that is used purely for
**color**.

How it works: both fields are stamped as point-data arrays on the same
`RectilinearGrid` (`build_pyvista_grid(extra_fields=...)`). The iso-surfaces
are still contoured on σ₀ — but VTK's `contour` filter interpolates every
point array onto the extracted surfaces, so the second field "rides along" and
can be used as the `add_mesh` color scalar with no resampling code. Each
isopycnal therefore shows the second field's value *on that density surface*,
and the tilted front iso-surface is likewise colored by it (so e.g. Ri along
the front sheet shows directly where the front is shear-unstable).

**Transforms and styles** live in [fronts/viz/field_styles.py](../../fronts/viz/field_styles.py), keyed by the tile variable name (`Ri`, `vorticity`, …). Each style sets a transform (`log10` / `symlog` / `linear`), a raw-value clip, a colormap, a scalar-bar title, and an optional `center` (diverging fields get symmetric color limits). For `Ri` the default is `log10(clip(Ri, 1e-2, 1e4))` with `Ri ≤ 0 → NaN` and a pinned clim of `(-1, 2)`. Unknown fields fall back to a linear transform with percentile limits and a warning.

**NaN handling**: cells that are NaN in the field (land, the tile edge rim, clipped/undefined values such as `Ri ≤ 0`) render in **neutral gray** (`#9e9e9e`) so the iso-surfaces stay hole-free while flagging missing data.

**Interpretation (Ri)**: the gradient Richardson number measures the competition between stratification (stabilising) and vertical shear (destabilising). `Ri < 0.25` (≈ `log10(Ri) < -0.6`) flags shear instability; the default `RdYlBu` colormap with clim `(-1, 2)` keeps that unstable range in the warm half of the bar.

Two-tile example (California Current front, rect tile 330):

```text
python -m fronts.scripts.fronts_viz_3d \
    --density-tile density_tile330_20121109T12.nc \
    --field-tile   Ri_tile330_20121109T12.nc \
    --labels       labeled_fronts_global_20121109T12_00_00_V4.npy \
    --i 13142 --j 9956 \
    --zscale 1.0 \
    --output /tmp/fronts_viz_3d_calcurrent_Ri.png
```

`--cmap-volume`, `--clim`, `--field-transform`, and `--field-clip` override the
registered style if you want to retune. `--mode volume` ignores the field tile
(volume mode renders σ₀ itself; coloring a σ₀ volume by a second scalar is
ill-defined).

## CLI

```text
python -m fronts.scripts.fronts_viz_3d \
    --density-tile density_tile330_20121109T12.nc \
    --labels       labeled_fronts_global_20121109T12_00_00_V4.npy \
    --i 13142 --j 9956 \
    --zscale 1.0 \
    --clim 24.4 25.0 \
    --cmap-volume dense \
    --output /tmp/fronts_viz_3d_calcurrent.png
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--density-tile` | path | required | NetCDF density tile from `generate_tile.py`. Drives geometry. |
| `--field-tile` | path | none | Optional second tile (same window + timestamp) whose variable colors the iso-surfaces. |
| `--field-name` | str | auto | Variable inside `--field-tile` (default: the single 3-D variable). |
| `--field-transform` | `log10`/`symlog`/`linear` | style | Override the field's registered display transform. |
| `--field-clip` | LO HI | style | Override the field's raw-value clip range (applied before the transform). |
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
| `--clim` | LO HI | style clim, else mixed-layer 2/98 percentile | Colour limits for the color scalar (σ₀, or the **transformed** field in `--field-tile` mode). |
| `--cmap-volume` | str | `viridis` (σ₀) / field-style cmap | Colormap for the volume/isopycnal background. See [colormap suggestions](#colormap-suggestions-denser--darker). |
| `--cmap-curtain` | str | `magma` | Colormap for the front curtain (only used when `draw_curtain=True`). |
| `--font-size` | int | 56 | Bounds-axis font size. |
| `--title-font-size` | int | 60 | Scalar-bar title font size. |
| `--label-font-size` | int | 44 | Scalar-bar tick label font size. |
| `--zscale` | float | `50.0` | Vertical exaggeration of the depth axis. See "tuning". |
| `--margin` | int | `50` | Pixel margin around the front's bbox. |
| `--n-below` | int | `3` | LLC levels below the deepest MLD to include. |
| `--cpos` | JSON | — | Camera-position triple to lock framing across runs. Overrides the SE-up default. |
| `--show` | flag | off | Open an interactive window instead of off-screen rendering. Hard-errors if `$DISPLAY` is empty. |

## Colormap suggestions (denser = darker)

`viridis` (the default) maps higher σ₀ to *lighter* (yellow), which is backwards if you want "dense = dark." Pass any of these via `--cmap-volume`:

| Colormap | Family | Notes |
|---|---|---|
| `dense` | cmocean | Purpose-built for ocean density. Cream → dark purple. Perceptually uniform. **Recommended for σ₀.** |
| `deep` | cmocean | Light yellow → deep blue. Perceptually uniform. |
| `gray` | cmocean | Perceptually-uniform monochrome. White → black. |
| `Blues` | matplotlib | Light blue → dark blue. |
| `bone` | matplotlib | Bluish-white → black. |
| `viridis_r` / `cividis_r` | matplotlib | Reversed perceptually-uniform; colorblind-safe. |
| `magma_r` / `plasma_r` / `inferno_r` | matplotlib | Reversed sequential, dark = high. |
| `Greys` / `gray_r` | matplotlib | Pure grayscale, light → dark. |

PyVista's colormap lookup accepts the bare cmocean names (`dense`, `deep`, `gray`) — not the `cmo.` prefix. So `--cmap-volume dense` works, `--cmap-volume cmo.dense` does not.

## Tuning notes

- **`--zscale`**: the default `50.0` matches a full-tile (~720 px) horizontal extent; for a tightly cropped front (~100–150 px) clipped at the MLD (~80 m), `50×` produces a 30:1 needle. Try `--zscale 1.0` or `--zscale 2.0` for cropped scenes. (Adaptive default is on the TODO list — see follow-ups.)
- **`--clim`**: the default is the 2/98 percentile **inside the mixed layer** (computed by `mixed_layer_clim`). This gives readable contrast where most cross-front structure lives and avoids the bottom half of the volume dragging the colour scale wider than it needs to be. Override with `--clim LO HI` if you want to pin the bar to specific values across runs.
- **`--cpos`**: `render_3d` sets a south-east elevated camera so i and j both increase rightward/upward. Capture the camera triple from a first run (printed in the RST block) and pass it back as JSON via `--cpos` to lock framing pixel-perfect across runs.
- **`--font-size` / `--title-font-size` / `--label-font-size`**: PyVista doesn't auto-scale fonts with image size, so these are point sizes in the 1600×1200 window (supersampled to 3200×2400 in the PNG). Defaults of 56/60/44 are tuned for readability at the default window size; lower them if you render at a smaller window or upscale the output.
- **Branch decomposition**: V4 fronts are thinned skeletons but can branch at Y-junctions and form closed loops. The custom 8-neighbour DFS handles all cases; 8-connectivity treats every corner of an orthogonal loop as a junction, so a square ring decomposes into many short branches. Edge coverage is correct (every edge appears once across the MultiBlock).
- **Hard errors**: bad locators (`--i/--j` lands on label 0, `--lat/--lon` outside the tile with no labelled pixels in the tile, `--show` with no `$DISPLAY`) exit non-zero with a clear diagnostic — no fallback figure is produced.

## Module map

| File | Role |
|---|---|
| [fronts/scripts/fronts_viz_3d.py](../../fronts/scripts/fronts_viz_3d.py) | CLI thin wrapper; orchestrates everything. |
| [fronts/viz/fronts_3d.py](../../fronts/viz/fronts_3d.py) | Front-specific PyVista builders. Re-usable by future 3-D scripts. |
| [fronts/viz/pv_helpers.py](../../fronts/viz/pv_helpers.py) | Generic PyVista helpers (scientific theme, off-screen Xvfb, scalar bar, PNG + HTML export). |
| [fronts/viz/field_styles.py](../../fronts/viz/field_styles.py) | Display registry for the color field: per-variable transform / clip / cmap / title / center, plus `apply_transform` and `default_clim`. |
| [fronts/viz/insets.py](../../fronts/viz/insets.py) | Matplotlib 2-D companion inset. |
| [fronts/llc/analysis.py](../../fronts/llc/analysis.py) | `mixed_layer_depth` (scalar) and `mixed_layer_depth_field` (vectorised). |
| [dev/mld/density_utils.py](../../dev/mld/density_utils.py) | Reused for `load_density_tile`, `load_labels_tile`, `build_tile_lookup`, `attach_lonlat_twins`. |

### Public helpers in `fronts/viz/fronts_3d.py`

| Function | What it does |
|---|---|
| `front_bbox_and_crop(labels, label, margin)` | `(j_slice, i_slice)` covering the front's bbox + margin. |
| `truncate_depth(sigma0, Z, k_mld, n_below)` | Clip the volume to `max(k_mld) + n_below + 1` levels. |
| `build_pyvista_grid(sigma0, Z, j_slice, i_slice, zscale, mask_2d=None, extra_fields=None)` | `pv.RectilinearGrid` with σ₀ (and any `extra_fields`) stamped Fortran-ordered; optional NaN-mask outside `mask_2d`. σ₀ stays the active scalar so geometry is always density-driven. |
| `dilate_front_mask(mask, iterations)` | 8-connectivity binary dilation. |
| `decompose_front_branches(front_mask)` | Custom 8-neighbour DFS; returns a list of per-branch (j,i) polylines. |
| `build_front_curtain(...)` | `pv.MultiBlock` of vertical sigma0-coloured ribbons per branch. Computed even when off; used by the surface marker. |
| `build_front_top_marker(curtain, Z, j_slice, i_slice, zscale, tube_radius)` | Small red tube at `z = Z[0] * zscale` tracing the surface footprint. |
| `build_front_isosurface(grid, level)` | Single-level σ₀ iso-surface — the tilted "front" indicator. |
| `pick_isopycnals_across_front(sigma0, mask, Z, ...)` | Auto-pick 5 σ₀ levels bracketing the cross-front contrast. |
| `mixed_layer_clim(sigma0, k_mld, ...)` | 2/98 percentile inside the mixed layer (the default for `--clim`). |
| `front_volume_clim(sigma0, dilated_mask, ...)` | 2/98 percentile inside a dilated-front 2-D mask (kept for future use; unused in the default path). |
| `render_3d(grid, curtain, levels, ..., color_scalar="sigma0", color_title=None, nan_color="#9e9e9e")` | Assembles the scene: background, front iso-surface, top marker, axes, SE-up camera, scalar bar. Geometry contours σ₀; `color_scalar` selects the coloring array (any stamped field); NaNs render neutral gray. |

## Smoke-test examples

The two examples worked through during development:

```bash
# 1. California Current (off Big Sur)
conda activate ocean14
python -m fronts.scripts.fronts_viz_3d \
    --density-tile "$OS_OGCM/LLC/Fronts/V4/20121109_120000/tiles/density_tile330_20121109T12.nc" \
    --labels       "$OS_OGCM/LLC/Fronts/V4/20121109_120000/labeled_fronts_global_20121109T12_00_00_V4.npy" \
    --i 13142 --j 9956 \
    --zscale 1.0 --clim 24.4 25.0 \
    --cmap-volume dense \
    --output /tmp/fronts_viz_3d_calcurrent.png
# Selects label 111425 at lon=-124.20, lat=36.38.

# 2. Gulf Stream central
python -m fronts.scripts.fronts_viz_3d \
    --density-tile "$OS_OGCM/LLC/Fronts/V4/20121109_120000/tiles/density_tile334_20121109T12.nc" \
    --labels       "$OS_OGCM/LLC/Fronts/V4/20121109_120000/labeled_fronts_global_20121109T12_00_00_V4.npy" \
    --i 16347 --j 9998 \
    --zscale 1.0 \
    --output /tmp/fronts_viz_3d_gs_central.png
# Selects label 111387 in the Gulf Stream.
```

## Version history (highlights)

- **v1.0** — initial script: bbox iso-surfaces + vertical curtain at front pixels + red top marker.
- **v1.1** — fixed an aspect-ratio crash in the matplotlib inset (`aspect="auto"` for compatibility with the twin-axis overlay).
- **v1.2** — added top-layer marker (`build_front_top_marker`); switched clim to `mixed_layer_clim`; vertical scalar bars on the right; larger fonts.
- **v1.3** — added `dilate_front_mask` + optional `mask_2d` for `build_pyvista_grid` + `front_volume_clim`; isopycnals could be restricted to a dilated-front band. (Reverted by default in v1.4.)
- **v1.4** — dropped the dilation mask from the default path (broader-context iso-surfaces back); replaced the vertical-sheet curtain with a single **tilted σ₀ iso-surface** via `build_front_isosurface`; inset figsize bumped (7×6 → 10×8).
- **v1.5** — explicit **south-east elevated camera** so i goes left→right and j goes bottom→top; fonts bumped (40/44/32 → 56/60/44); top marker tube radius reduced (1.5 → 0.8) and opacity dropped (1.0 → 0.6).
- **v1.6** — `--cmap-volume` suggestions for "dark = dense" (recommend `dense`); CLI font knobs (`--font-size`, `--title-font-size`, `--label-font-size`); enlarged scalar bar (width 0.05 → 0.08, height 0.42 → 0.50) and shifted left of the viewport edge so the title doesn't clip.
- **v1.7** — **dual-field coloring**: geometry stays σ₀-driven while an optional `--field-tile` (e.g. a Richardson-number tile) colors the iso-surfaces via VTK contour pass-through interpolation. New `field_styles.py` display registry (per-variable transform / clip / cmap / title / center); `build_pyvista_grid(extra_fields=...)` and `render_3d(color_scalar=, color_title=, nan_color=)`; NaN cells (land / edge rim / clipped) render neutral gray `#9e9e9e`. New flags `--field-tile`, `--field-name`, `--field-transform`, `--field-clip`; `--clim`/`--cmap-volume` now apply to the color scalar. Fully backward compatible — without `--field-tile` the scene is unchanged.

## Known follow-ups

- Adaptive default for `--zscale` based on the cropped bbox aspect ratio (`max(j_extent, i_extent) / |Z_clipped[-1]|`).
- Curtain-on case (`draw_curtain=True`) currently overlaps with the iso/volume scalar bar at the new bar height; trivial fix via per-mode height adjustment.
- CLI knobs for `--top-marker-opacity`, `--top-marker-radius`, and `--front-iso-opacity` are not yet exposed.
- The cross-front isopycnal-bracket method uses surface percentiles (round-3 option (i)); the alternative of local-normal sampling at each pixel was deferred.
- The 8-connectivity branch decomposition could be reduced to 4-connectivity to avoid spurious junctions on orthogonal loops; not pursued because the union of branch ribbons renders correctly already.
