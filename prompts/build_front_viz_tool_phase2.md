# Phase 2 — five pages, real data plan

Extends [build_front_viz_tool.md](build_front_viz_tool.md), which covered the
first three pages and still describes the stack, the grid problem and the
prerequisites. This adds two pages, revises two, and settles where the data
comes from.

Data plan lives in [docs/viz/apps/DATA.md](../docs/viz/apps/DATA.md) — read
that first; this document is the build order.

## The five pages

| # | Page | Route | Status |
|---|---|---|---|
| 1 | Field Characteristics **at the Surface** | `/surface` | exists, being revised |
| 2 | Field Characteristics **at Depth** | `/depth` | new — mirrors page 1 |
| 3 | **Bivariate** | `/bivariate` | new |
| 4 | **Tiles** | `/tiles` | exists, being revised |
| 5 | **Evolution** | `/evolution` | still a stub |

## Decisions taken

| Decision | Choice |
|---|---|
| Where the app runs | profx, profx filesystem first, S3 fallback |
| Tiles | density pre-generated; colour fields lazy, cached, kept forever |
| Evolution chunks | pre-generated |
| Bivariate | its own page, with a SURFACE / DEPTH mode selector |
| Serving | loopback + SSH tunnel |

---

## 1. Field Characteristics at the Surface

Four changes.

### 1.1 Rename
`characteristics` → `surface` throughout: route, title, module directory,
docs. Page 2 becomes `depth`.

### 1.2 Fix the map zoom

Today the box-select drives the statistics but the map stays global, so
there is nothing to tell you which region the PDFs describe.

Selecting a box should **zoom the map to it**. Implementation: after
`BoundsXY` fires, set the plot's `xlim`/`ylim` to the box (padded ~10%), and
draw the box outline so the selection stays visible after the zoom. **Reset
region** returns to the full globe. Zoom and selection stay coupled — one
concept, not two.

### 1.3 New section — front properties

Below the map. Uses the **geometry** and **colocation** parquet from
`build_v5` steps 3 and 4, filtered to the fronts whose centroid falls in the
selected region.

Six panels:

| | Panel |
|---|---|
| a | PDF of front length (`length_km`) |
| b | PDF of front orientation (`orientation`, degrees) |
| c | JPDF latitude × length |
| d | JPDF latitude × orientation |
| e | JPDF {selected field} × length |
| f | JPDF {selected field} × orientation |

Panels (e) and (f) need a per-front statistic of the field, so the section
carries its own **statistic** selector — `mean`, `median`, `p25`, `p75`,
`p90` — resolving to the colocation column `{field}_{stat}`.

> **`p95` does not exist.** `run_v5_100_timesteps.yaml` has
> `percentiles: [25, 75, 90]`, so the available set is p25 / p75 / p90 plus
> mean / median (median = p50). To get p95, add it to the config and re-run
> step 4. The selector is built from the columns actually present, so it
> will pick up p95 automatically if you do.

New module: **`fronts/viz/apps/surface/front_props.py`** — filtering and the
six builders.

### 1.4 Removed from this page

The bivariate section moves to its own page (§3).

---

## 2. Field Characteristics at Depth

Identical to page 1 — same six distribution panels, same six front-property
panels, same map behaviour — with one extra control.

**DEPTH LEVEL**, mapping onto the suffixes the channels already carry:

| Label | Suffix |
|---|---|
| Surface | `sfc` |
| 25 m | `z25m` |
| Mixed layer depth | `mld` |
| Mean over mixed layer | `mld_mean` |

Restricted to the three 3-D dates. The date selector shows only those, with
a note saying why.

**Built by sharing, not copying.** Pages 1 and 2 are the same page with a
different channel resolver, so the layout and the panel builders live in one
module and the two pages differ only in how a `(field, depth)` pair becomes
a channel name:

```
surface:  field                 -> "relative_vorticity"
depth:    field + depth level   -> "relative_vorticity_mld"
```

Concretely: `surface/page.py` holds the shared assembly; `surface/app.py`
and `depth/app.py` are thin, each supplying a resolver and a date list.

---

## 3. Bivariate (new page)

Full-width map of fronts coloured by a **two-field bivariate scheme**,
generalised from `fronts/properties/nb/Bivariate_TurnerAngle.ipynb`.

### Controls

| Control | Notes |
|---|---|
| **Mode** | SURFACE or DEPTH — drives which dates, depth levels and fields are offered |
| Date | 100 dates in surface mode, 3 in depth |
| Depth level | depth mode only |
| Field A / Field B | any colocated channel |
| Statistic | mean / median / p25 / p75 / p90 |
| **Sections** | colour divisions per field, default 2 |
| Spatial binning | on/off, bin size in degrees |

### New module — `fronts/viz/bivariate.py`

The notebook is hardcoded to 2×2 and to Turner angle. Generalising it:

| Function | Role |
|---|---|
| `bivariate_colormap(n, hue_a, hue_b)` | `(n, n, 3)` colour grid; lightness carries field A, hue carries field B |
| `assign_bins(values_a, values_b, n, edges=...)` | bin index per front; quantile edges by default, or explicit edges |
| `plot_bivariate_map(df, ...)` | the map — scatter or spatially binned |
| `plot_bivariate_legend(...)` | the `n × n` legend square with axis edge labels |

Two details worth preserving from the notebook:

- **A physically meaningful split beats a quantile split** when one exists.
  Turner angle splits at 0°, not at its median. So `assign_bins` takes
  optional explicit edges, and the registry names a natural split for the
  fields that have one (Turner angle at 0, vorticity at 0).
- **Quantile edges, not equal-width**, as the default — front properties are
  heavy-tailed and equal-width bins put almost every front in one cell.

The notebook keeps working; it can import these instead of defining them.

---

## 4. Tiles

Two changes.

### 4.1 Three field columns

Move figures (a)–(f) from beside the map to **a column below it**, then
allow **up to three columns side by side, one per selected field**, so the
same front can be compared across fields.

```
+---------------------------------------------------+
|  overview map          |  tile map                |
+---------------------------------------------------+
|   field: Ri     |   field: N2     |  field: wB    |
|   (a) 3-D       |   (a) 3-D       |  (a) 3-D      |
|   (b) inset     |   (b) inset     |  (b) inset    |
|   (c) isopycnal |   (c) isopycnal |  (c) ...      |
|   (d) mainaxis  |   (d) mainaxis  |               |
|   (e) offsets   |   (e) offsets   |               |
|   (f) perp      |   (f) perp      |               |
+---------------------------------------------------+
```

Field selection becomes a **multi-select capped at three**. The geometry —
front, crop, main axis, perpendicular point — is computed **once** and
shared across the columns, so only the colour field differs. That is both
faster and the only way the columns are actually comparable.

### 4.2 Regenerate button

Changing region / date / field / front no longer rebuilds anything. It marks
the figures stale — dimmed, with a "settings changed" note — and
**Regenerate** does the work.

This is what makes the lazy-tile plan usable: a new field costs ~25 s, and
the user chooses when to spend it. The button reports what it is about to
do ("2 tiles to generate, ~30 s") before starting.

Restricted to the three 3-D dates.

---

## 5. Data layer changes

### `LocalProvider` (new, primary)

Reads profx paths; falls back to `S3Provider` per file. Selected by
`FRONTS_APP_DATA=profx`.

### Provider interface additions

| Method | For |
|---|---|
| `depth_levels(date)` | Depth page selector |
| `channel(field, depth=None)` | resolve field + depth → channel name |
| `geometry(date)` / `colocation(date)` | already present; now actually used |
| `ensure_tile(date, tile, field)` | generate-if-missing, returns a path |

`ensure_tile` is where the lazy tile policy lives: look on disk, else run
`generate_tile`, write into the standard tree, return the path. Everything
else treats tiles as if they had always been there.

---

## 6. Build order

| # | Step | Why first |
|---|---|---|
| 1 | `LocalProvider` + `ensure_tile` | everything else reads through it |
| 2 | Rename to `surface`, fix the map zoom | small, visible, unblocks page 2 |
| 3 | Front-properties section | new panels, still one page |
| 4 | Share page 1 → build page 2 (Depth) | proves the shared assembly |
| 5 | `fronts/viz/bivariate.py` + Bivariate page | independent of 2–4 |
| 6 | Tiles: 3 columns + Regenerate | independent of 2–5 |
| 7 | Evolution | needs the chunks to exist |

Steps 5 and 6 do not depend on 2–4, so they can be done in any order.

---

## 7. To confirm

- **p95**: use p90, or re-run step 4 with 95 added?
- **Evolution regions**: which one or two, and which 24-hour window?
- **Tile netcdf-base on profx**: the exact root for the standard tile tree.
- **Bivariate default pair**: `gradb2` × `turner_angle`, as in the notebook?
- Whether a fourth 3-D date is coming, since it only changes a list.
