# `fronts_viz_curtain` — 2-D "curtain" cross-sections of a single labelled front

[fronts/scripts/fronts_viz_curtain.py](../../fronts/scripts/fronts_viz_curtain.py) is the 2-D companion to [`fronts_viz_3d`](fronts_viz_3d.md). Instead of a 3-D PyVista scene it produces flat **curtain** plots — vertical cross-sections sampled along a path through the tile, with **distance along the path** on the x-axis, **depth** on the y-axis, a configurable field (default the Richardson number `Ri`) as **color**, and **isopycnals** (σ₀ surfaces) overlaid as contour lines.

It reuses the 3-D script's tile-loading, frame-remapping, and front-picking pipeline verbatim, so the inputs and locators are identical.

## What it produces

Four PNGs per run, all sharing the `--output-prefix`. The filenames embed the field name (`{field}` = the tile variable, e.g. `Ri`) and, for the offsets figure, the offset count (`{N}` = `--n-offsets`):

- **`{prefix}_{field}_mainaxis.png`** — the main-axis curtain (single panel).
- **`{prefix}_{field}_offsets_n{N}.png`** — along-front curtains: two summary (dilation) rows on top + the `N` individual offset rows (two columns × `N+2` rows).
- **`{prefix}_{field}_perp.png`** — the cross-front (perpendicular) curtain (single panel).
- **`{prefix}_{field}_inset.png`** — a plan-view map of the bbox: the **color field** near-surface slice (same colormap/colorbar as the curtains) with the main axis, the offset envelope, and the marked perpendicular point overlaid (opt out with `--no-inset`).

All figures are static matplotlib (Agg backend, dpi 150), matching the repo's existing 2-D companion in [fronts/viz/insets.py](../../fronts/viz/insets.py).

## The three figures

### 1. Main-axis curtain

Plots only the front's **main axis** — the longest end-to-end path through the thinned skeleton, with side branches dropped (see [Main-axis extraction](#main-axis-extraction)). Every real front pixel along that path is one curtain column; nothing is resampled onto evenly-spaced points. x = distance along the front (pixels, with a km twin axis on top), y = depth, color = the selected field, σ₀ isopycnals overlaid, the mixed-layer depth drawn as a dashed white line, and the chosen perpendicular point marked with a green vertical line.

### 2. Along-front curtains with offsets

Two columns, `N+2` rows, all sharing one color scale:

- **Row 0** — the main-axis curtain (left) and the **mean over all `2N` offsets** (right). The all-offset mean is a *dilation* of the front: the average field in a band 1..N px to either side.
- **Row 1** — the mean over the **`+`** offsets (left) and the mean over the **`−`** offsets (right): *directional dilations*, one side of the front each.
- **Rows 2..N+1** — the individual offsets: offset *r* px on the `+normal` side (left) and `−normal` side (right).

The means are computed on the sampled curtains, column-aligned, with trimmed/looped columns NaN'd *before* averaging, so each averaged column only pools the offsets whose geometry is valid there.

By default each offset line is **trimmed** of self-intersection loops so it contains no crossings — the looped columns on the concave side of a bend are excised ("sewn shut") and render as neutral-gray gaps in the curtain (and as a continuous, crossing-free line on the inset). Pass `--no-trim-offsets` to keep the loops instead and shade them magenta. Off-window/NaN cells always render gray. See [Offsets, trimming, and self-overlap](#offsets-trimming-and-self-overlap).

### 3. Cross-front (perpendicular) curtain

A perpendicular transect is cut at a chosen point along the main axis. By default that point is the **field extremum over the full curtain depth** (e.g. the column whose deepest-searched `Ri` is lowest — most shear-unstable), **restricted to columns whose transect crosses the front at most `--perp-max-crossings` times (default 1)** so the transect lands on a clean stretch rather than the front's squiggly self-overlapping parts. Override with `--extremum max`, lift the crossing filter with `--perp-allow-crossings`, or pin a specific column with `--perp-point` (use `--list-perp-candidates` to print each column's `(i, j)`, along-path km, and crossing count). The transect spans `2·--perp-half-width + 1` pixels — which is exactly the length of the green line drawn on the inset — and the curtain's x-axis is **signed cross-front distance** with 0 at the front axis (negative on side B, positive on side A).

## Inputs

Same as `fronts_viz_3d`, with one difference: the field tile is **required** here (it supplies both the curtain color and the perpendicular-point extremum).

1. **3-D density tile** — NetCDF from `generate_tile.py`. `sigma0(k, j, i)` on face-local axes, plus `XC`/`YC`/`Z` and `rect_i_start`/`rect_j_start`/`face_index` provenance. Drives the isopycnal contours.
2. **Field tile** (`--field-tile`, **required**) — a second NetCDF (same tile window + timestamp) holding the color field, e.g. a `Ri` tile from `--property richardson`. Auto-detects the single 3-D variable unless `--field-name` is given.
3. **Labelled-fronts mask** — global `.npy`/`.nc` on the rect grid (integer labels, `0 = no front`).
4. **Locator** — `(--i, --j)` rect indices or `(--lat, --lon)` degrees (snapped to the nearest labelled pixel).

## Frame handling

Identical to the 3-D script: `build_tile_lookup` gives the per-pixel `(j_face, i_face)` for every rect-tile-local pixel; σ₀, the field, `XC`, `YC` are fancy-indexed onto the rect frame; everything downstream operates in the one coordinate system that matches `--i / --j`.

## Algorithm flow

1. **Load** — `load_density_tile` + `load_tile` (field) + `load_labels_tile`; `check_tiles_consistent` enforces matching provenance.
2. **Remap** — `remap_to_rect` (reused from `fronts_viz_3d`) for σ₀, the field, `XC`, `YC`.
3. **Pick label** — `pick_front_label` (reused) via `(--i,--j)` or snapped `(--lat,--lon)`.
4. **Crop** — `front_bbox_and_crop` → `(j_slice, i_slice)` over the bbox + `--margin`.
5. **MLD + depth clip** — `mixed_layer_depth_field` then `truncate_depth` keep levels `0 .. max(k_mld) + --n-below`; this sets the curtain's depth extent.
6. **Color transform** — `apply_transform` from [field_styles.py](../../fronts/viz/field_styles.py) maps raw field → display values (for `Ri`: `log10(clip(Ri, 1e-2, 1e4))`, `Ri ≤ 0 → NaN`).
7. **Main axis** — `curtains.extract_main_axis` (skeleton diameter; see below).
8. **Path metrics** — `curtains.path_metrics` returns per-pixel pixel-distance, great-circle km distance, and unit tangents/normals. `--smooth-normals` smooths *only* the direction field.
9. **Isopycnal levels** — `pick_isopycnal_levels` brackets the 2/98 percentile of the whole clipped volume (the curtain spans full depth), or uses `--isopycnals`.
10. **Perpendicular point** — `--perp-point`, else `curtains.pick_extremum_index` over the full-depth axis curtain, with columns whose transect re-crosses the front (`curtains.transect_front_crossings > --perp-max-crossings`) excluded unless `--perp-allow-crossings`.
11. **Render** — `curtains.figure_main_axis`, `figure_offsets` (with the dilation summary rows), `figure_perpendicular`, and the script's `plot_map_inset` (field background + colorbar).

## Main-axis extraction

`extract_main_axis` builds on the 3-D module's `decompose_front_branches` (a custom 8-neighbour DFS that splits the thinned skeleton into branch polylines between junctions and endpoints). Those branches form a graph whose nodes are branch end pixels and whose edges are weighted by branch arc length (4-connected steps = 1, diagonal = √2). The **main axis is the graph diameter** — the longest end-to-end path — found by a double Dijkstra sweep (farthest node from an arbitrary start, then farthest from that node). Only the branches on that path are concatenated, so **side branches are dropped by construction**. Closed loops still yield a finite, well-defined diameter. Single-pixel and degenerate masks fall back to the longest available branch.

> The `decompose_front_branches` import is lazy, so importing `fronts.viz.curtains` does **not** require PyVista — the 2-D viewer runs without the 3-D rendering stack.

## Path metrics, smoothing, and km distance

`path_metrics` returns arrays aligned 1:1 with the main-axis pixels — it never resamples or moves columns:

- **`dist_px`** — cumulative Euclidean pixel distance between consecutive real pixels (so diagonal hops add √2).
- **`dist_km`** — cumulative haversine great-circle distance from `XC`/`YC` at the real pixels. This is just a relabelling of the same columns; the km axis on the figures is interpolated onto the pixel ticks.
- **`tangents` / `normals`** — unit direction vectors used to throw offsets and the perpendicular.

**Smoothing (`--smooth-normals`, off by default)** applies a moving average of width `--smooth-window` (odd, default 5) to the tangent direction before deriving normals. It affects **only the direction field** — never the main-axis column positions or the distances. A raw thinned skeleton zig-zags between 4- and 8-connected steps, so per-pixel normals can swing ~45–90° between neighbours; that throws adjacent offset points in inconsistent directions and makes them collide immediately. Smoothing averages those kinks out so offsets stay roughly parallel. Render the same front both ways to compare.

## Offsets, trimming, and self-overlap

`offset_paths` shifts the main axis ±k pixels along the per-pixel normal for k = 1..N, giving two mirror-image families (side A = `+normal`, side B = `−normal`). Each offset path is sampled exactly like the main axis.

On the concave side of a bend an offset folds back and crosses itself. Two handling modes:

- **Trim (default).** `trim_offset_loops` finds each pair of crossing segments and excises the looped vertices between them, leaving a shorter but crossing-free polyline — the "sew the line shut" behaviour. The dropped columns are NaN'd in the curtain (gray gaps) and skipped on the inset line, so neither the curtain nor the map ever shows a self-crossing offset. Trimming runs whether or not `--smooth-normals` is set, but smoothing first gives cleaner results because it removes the pixel-scale kinks that create spurious tiny loops.
- **Shade (`--no-trim-offsets`).** Keep the loops and instead flag them: `offset_quality_flags` marks a column when its offset point lands within `< 1 px` of any non-adjacent column's offset point, and those columns are shaded magenta. The curtain still renders; the unreliable region is just made obvious.

Either way the two failure modes stay visually distinct: **gray** = trimmed loop or off-window/NaN sample; **magenta** (shade mode only) = kept self-overlap.

Two causes of overlap, two complementary tools: pixel-scale jaggedness is an artifact best removed by `--smooth-normals`; genuine tight curvature is real geometry, removed cleanly by trimming. A more-correct centre-of-curvature offset (move outward from the local centre of curvature instead of straight along the normal) would *preserve* rather than drop the concave-side material, but it is fragile to pixel noise and is left as a follow-up.

## Curtain sampling

`sample_curtain(field3d, path)` samples a `(K, J, I)` field along a `(j, i)` path at every depth via `scipy.ndimage.map_coordinates(order=1, mode="constant", cval=nan)`, returning a `(K, L)` array. This mirrors the inner loop of `fronts_3d.build_front_curtain` but returns a plain array for matplotlib. Paths leaving the cropped window come back as NaN, which the renderer shades neutral gray.

## CLI

```text
python -m fronts.scripts.fronts_viz_curtain \
    --density-tile density_tile330_20121109T12.nc \
    --field-tile   Ri_tile330_20121109T12.nc \
    --labels       labeled_fronts_global_20121109T12_00_00_V4.npy \
    --i 13142 --j 9956 \
    --n-offsets 3 --perp-half-width 30 \
    --output-prefix /tmp/calcurrent_curtain
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--density-tile` | path | required | NetCDF density tile (σ₀); drives isopycnals. |
| `--field-tile` | path | **required** | Field tile (same window+timestamp) coloring the curtains, e.g. a `Ri` tile. |
| `--field-name` | str | auto | Variable inside `--field-tile` (default: the single 3-D variable). |
| `--field-transform` | `log10`/`symlog`/`linear` | style | Override the field's registered transform. |
| `--field-clip` | LO HI | style | Override the field's raw-value clip range. |
| `--labels` | path | required | Global integer-labels mask (`.npy` or `.nc`). |
| `--i / --j` | int | — | Rect-grid pixel coordinates (locator option A). |
| `--lat / --lon` | float | — | Degrees (locator option B); snapped to the nearest labelled pixel. |
| `--margin` | int | `50` | Pixel margin around the front bbox. |
| `--n-below` | int | `3` | LLC levels below the deepest MLD to include (sets depth extent). |
| `--n-offsets` | int | `3` | Number of offset rows per side. |
| `--perp-half-width` | int | `30` | Half-width (px) of the perpendicular transect (length `2N+1`); also the length of the green line on the inset. |
| `--perp-point` | int | extremum | Main-axis column index for the perpendicular transect. |
| `--extremum` | `min`/`max` | `min` | Default perpendicular point = field min (e.g. lowest `Ri`) or max. |
| `--perp-max-crossings` | int | `1` | Auto-pick only considers columns whose transect crosses the front ≤ this many times (keeps it off the squiggly hook). |
| `--perp-allow-crossings` | flag | off | Disable the crossing filter; auto-pick the extremum anywhere. |
| `--list-perp-candidates` | flag | off | Log each column's `(i, j)`, along-path km, and crossing count, then continue. Helps choose `--perp-point`. |
| `--smooth-normals` | flag | off | Smooth the tangent/normal direction field before offsets/perpendicular. Never moves the main-axis columns. |
| `--smooth-window` | int | `5` | Odd pixel window for `--smooth-normals`. |
| `--no-trim-offsets` | flag | off (trim on) | Keep offset self-intersection loops (shaded magenta) instead of trimming them to crossing-free lines. |
| `--isopycnals` | floats | auto | Explicit σ₀ contour levels. |
| `--n-isopycnals` | int | `8` | Number of auto-picked isopycnal levels. |
| `--clim` | LO HI | style clim, else 2/98 pct | Color limits for the (transformed) color field. |
| `--cmap` | str | field-style cmap | Colormap for the color field. |
| `--output-prefix` | path | required | Prefix; appends `_{field}_mainaxis.png` / `_{field}_offsets_n{N}.png` / `_{field}_perp.png` / `_{field}_inset.png` (`{field}` = tile variable name, `{N}` = `--n-offsets`). |
| `--no-inset` | flag | off | Skip the plan-view map inset. |

## Module map

| File | Role |
|---|---|
| [fronts/scripts/fronts_viz_curtain.py](../../fronts/scripts/fronts_viz_curtain.py) | CLI orchestrator; reuses the `fronts_viz_3d` ingest pipeline + the map-view inset. |
| [fronts/viz/curtains.py](../../fronts/viz/curtains.py) | Reusable builders: `extract_main_axis`, `path_metrics`, `offset_paths`, `perpendicular_path`, `offset_quality_flags`, `sample_curtain`, `pick_extremum_index`, and the matplotlib panel/figure renderers. |
| [fronts/viz/fronts_3d.py](../../fronts/viz/fronts_3d.py) | Reused for `decompose_front_branches` (lazy import), `front_bbox_and_crop`, `truncate_depth`. |
| [fronts/viz/field_styles.py](../../fronts/viz/field_styles.py) | Per-variable transform / clip / cmap / title for the color field. |
| [fronts/llc/analysis.py](../../fronts/llc/analysis.py) | `mixed_layer_depth_field` for the depth clip + MLD overlay. |
| [dev/mld/density_utils.py](../../dev/mld/density_utils.py) | `load_density_tile`, `load_tile`, `load_labels_tile`, `build_tile_lookup`, `tile_scalar`, `attach_lonlat_twins`. |

### Public helpers in `fronts/viz/curtains.py`

| Function | What it does |
|---|---|
| `extract_main_axis(front_mask)` | Longest end-to-end path through the skeleton (diameter); side branches dropped. |
| `path_metrics(path, XC, YC, smooth=, smooth_window=)` | Per-pixel `dist_px` / `dist_km` / unit tangents / normals. Smoothing affects directions only. |
| `offset_paths(path, normals, n)` | Two mirror families of offset polylines (sides A/B). |
| `perpendicular_path(path, normals, idx, half_width)` | Cross-front transect centered on `path[idx]`. |
| `offset_quality_flags(offset_path)` | Boolean mask of self-overlapping columns (< 1 px collision); used in `--no-trim-offsets` mode. |
| `trim_offset_loops(offset_path)` | Keep-mask that excises self-intersection loops so the offset line has no crossings ("sew shut"). |
| `transect_front_crossings(axis, normals, mask, half_width)` | Per-column count of how many times the perpendicular transect hits the front; drives the perpendicular auto-pick filter. |
| `sample_curtain(field3d, path)` | `(K, L)` sample of a field along a path at every depth; off-window → NaN. |
| `pick_extremum_index(curtain_field, mode)` | Column of the full-depth min/max of the field. |
| `plot_curtain_panel(ax, ...)` | One curtain panel: color mesh + isopycnal contours + km twin axis + MLD + overlap shading. |
| `figure_main_axis / figure_offsets / figure_perpendicular(...)` | The three deliverable figures. |

## Testing

[fronts/tests/test_curtains.py](../../fronts/tests/test_curtains.py) covers the geometry and sampling with synthetic skeletons: straight / branched / curved / single-pixel main-axis extraction, normals and km distances, smoothing-keeps-columns invariance, offset mirror symmetry, self-overlap detection on tight curves, NaN handling off-window, perpendicular geometry, and extremum picking.

```bash
pytest fronts/tests/test_curtains.py
```

## Known follow-ups

- **Centre-of-curvature offsets** — offset outward from the local centre of curvature instead of straight along the normal, for a more correct concave-side spacing that *preserves* rather than trims the inner material. Deferred (fragile to pixel-scale noise; needs smoothing first).
- **km twin axis on the perpendicular figure** — currently the perpendicular x-axis is signed pixels only; a signed-km twin could be added.
- **Adaptive `--perp-half-width`** from the front's local cross-stream density scale.
- **Shared interactive HTML** (plotly) export, if hover/zoom is wanted — not currently a repo pattern.
