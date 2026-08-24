# Build plan — front visualisation web tool

Four browser pages for exploring LLC4320 fronts, served by one
`panel serve` process out of `fronts/viz/apps/`. Every page has the same
shape: **global map → pick something → get figures for it.**

Three of the four are built and in use. Evolution is part-built; §8 is
the outstanding work and says what is already done.

| Page | Route | What you look at | Status |
|---|---|---|---|
| **Field Characteristics** | `/surface` | statistics of a field over a lat/lon box you draw | built |
| **Bivariate** | `/bivariate` | fronts coloured by a two-field scheme | built |
| **Tiles** | `/tiles` | one front inside a 720×720 tile, in section and in 3-D | built |
| **Evolution** | `/evolution` | one front over a week of snapshots, as a movie | **in progress** |

A Depth twin of Field Characteristics, and a Depth mode for Bivariate,
are complete and tested but not served — they are hidden by
`config.ENABLED_PAGES` and `config.BIVARIATE_MODES` rather than deleted,
so putting the names back brings them straight back.

---

## 1. The grid problem — read this before touching a map

**The LLC4320 rect grid is not a regular lat/lon grid.** This is the
single biggest constraint in the whole tool and it is easy to get wrong.

The grid is 12960 × 17280, stitched from 13 native faces — some rotated
to tile correctly. `XC` and `YC` survive as **2-D** arrays. So:

- latitude spacing is not uniform (the `j` axis is Mercator-like);
- longitude is only *approximately* column-aligned;
- `hv.Image` assumes a regular 1-D axis per dimension, so it silently
  misplaces data. Only `hv.QuadMesh` with 2-D coordinates is strictly
  correct, and datashading a curvilinear QuadMesh at 224M points is far
  too slow to drive an interactive map.

The resolution is to **split display from statistics**:

| Path | Grid | Correctness |
|---|---|---|
| the map | regridded offline onto a regular lat/lon pyramid | good enough to look at; never feeds a number |
| the numbers | native rect grid, read directly | exact, full resolution |

`common/pyramid.py` bins the native field into regular lat/lon rasters at
`PYRAMID_WIDTHS = (1440, 2880, 5760, 11520)`. Once that exists,
`hv.Image` is both correct and fast, and Pacific-centring is a column
roll — on the regridded raster, where the grid really is regular.

**Selecting a box never converts to indices.** `common/selection.py`
masks on the coordinate arrays directly:

```python
mask = (YC >= lat0) & (YC <= lat1) & (XC >= lon0) & (XC <= lon1)
```

One vectorised pass, exact, and correct on the irregular grid by
construction — because it asks about coordinates instead of assuming
positions. (The brute-force nearest-neighbour in `fronts/llc/coords.py`
allocates ~1.8 GB per query point; fine for a CLI, unusable behind a
box-select.)

---

## 2. Data

Everything the pages read sits under one prefix, for exactly four dates:

```
s3://dbof/globals_for_chunks/V5/{YYYYMMDD_HHMMSS}/
    {subset}.zarr        fields, surface and depth
    Fronts/              binary, labels, geometry
```

| date | prefix |
|---|---|
| 2012-02-29 18:00 | `20120229_180000` |
| 2012-05-16 06:00 | `20120516_060000` |
| 2012-09-18 11:00 | `20120918_110000` |
| 2012-11-09 12:00 | `20121109_120000` |

| What | Where |
|---|---|
| 3-D source for tiles | `s3://dbof/LLC4320_RAW/DEPTH/` |
| generated tiles | `s3://dbof/tiles/{date}/{region}/{field}.zarr` |
| Evolution chunks | `s3://dbof/LLC4320_RAW/CHUNKS/{chunk}/{YYYYMMDDTHH}.zarr` |
| endpoint | `https://s3-west.nrp-nautilus.io` |

Chunks exist for `monterey_bay` and `southern_ocean_scotia_sea`.

Full detail in [docs/viz/apps/DATA.md](../docs/viz/apps/DATA.md).

### What it costs, and why the design looks like this

The global stores are chunked `(1, H, W)` — **one chunk per channel**, so
there is no such thing as a partial read. Every field plane is 0.83 GB,
and measured throughput to the bucket is ~28 MB/s. So a cold field is
~30 s and a cached one is ~1 s, and the whole app is bound by S3 traffic
rather than by CPU.

Everything follows from that:

- field planes, the ice mask, and `XC`/`YC` are cached to disk as `.npy`
  and read back **memmapped** (`common/cache.py`);
- `XC` on the reader is a *property* that re-reads 0.9 GB on every
  access, so it is touched once and kept;
- the land mask comes from the field's own NaNs rather than a separate
  0.9 GB `hFacC` read;
- the disk cache is LRU-trimmed to `CACHE_CAP_BYTES` (10 GB default);
- tiles are generated once, written back to `s3://dbof/tiles/`, and read
  from there forever after.

**Nothing is computed until you ask for it.** Every expensive stage is
behind an explicit button, and the button says what it is about to do.

---

## 3. Architecture

```mermaid
flowchart TB
    subgraph s3["S3"]
        G["globals_for_chunks/V5/{date}/<br/>fields · fronts"]
        R["LLC4320_RAW/DEPTH<br/>LLC4320_RAW/CHUNKS"]
        T["tiles/{date}/{region}/{field}.zarr"]
    end

    R -->|"generate on miss,<br/>write back"| T

    subgraph app["fronts/viz/apps/"]
        C["common/<br/>s3source · cache · state · basemap<br/>pyramid · selection · regions · tilestore"]
        P1["characteristics/"]
        P2["bivariate/"]
        P3["tiles/"]
        P4["evolution/"]
        C --> P1 & P2 & P3 & P4
    end

    G -->|"exact statistics"| C
    G -->|"display pyramid"| C
    T --> C
```

```
fronts/viz/apps/
  serve.py                 routes, filtered by config.ENABLED_PAGES
  config.py                dates, paths, field lists, caps — one place
  build_tiles.py           batch tile generation CLI
  check_s3.py  check_tiles.py  warm.py      diagnostics / prefetch
  common/
    s3source.py   the provider: fields, masks, fronts, tiles
    tilestore.py  read/write the generated-tile store
    cache.py      disk cache + LRU trim
    state.py      one param.Parameterized per page
    basemap.py  pyramid.py  selection.py  regions.py  widgets.py
  characteristics/  page.py (shared) + surface.py / depth.py + panels + stats
  bivariate/  tiles/  evolution/
```

Two things worth knowing about the shape:

**The figure builders are the repo's existing code.** `curtains.figure_*`,
`fronts_3d.render_3d`, `plot_map_inset`, `dbof.plotting.pdfs` / `jpdfs`
are called unchanged. The app is a shell around them.

**State is `param.Parameterized`, so the whole thing is testable
headlessly.** Set date/field/region/front, assert the derived values. Only
the Panel layout goes untested, which is the right thing to leave
untested.

**We do not edit the `llc4320-native-grid-preprocessing` repo** from
here. It is imported (`dbof.tiles.field_registry`, `dbof.tiles.tile_utils`,
`dbof.plotting.*`) and its tile-composition steps are reproduced in
`s3source._compose_tile`, so the app does not depend on which branch is
checked out.

---

## 4. Field Characteristics (`/surface`)

```
+---------------------------------+------------------+------------------+
|                                 |   ALL GRID PTS   |   FRONTS ONLY    |
|      GLOBAL MAP                 | (a) PDF of field | (a) PDF of field |
|      Pacific-centred            | (b) JPDF strain  | (b) JPDF strain  |
|      land gray, coastlines      |     vs vorticity |     vs vorticity |
|      [drag a box to select]     | (c) conditional  | (c) conditional  |
|                                 |     JPDF         |     JPDF         |
+---------------------------------+------------------+------------------+
|  front properties: (a)-(f)  length · orientation · JPDFs vs latitude  |
+-----------------------------------------------------------------------+
```

**Navigation is not computation.** Box-select and *Reset region* move the
map immediately; only the panels below wait for *Rebuild*. This is the
rule the page is built around, and getting it wrong was the source of a
whole family of bugs (see §7).

The six front-property panels come from the geometry parquet, filtered to
fronts whose centroid falls in the box. Panels (e) and (f) additionally
need a per-front field statistic from the colocation table, and carry
their own **statistic** selector (`mean`, `median`, `p25`, `p75`, `p90`
— built from the columns actually present, so adding `p95` upstream makes
it appear). Colocation has not been run for V5, so (e) and (f) are
degraded and the panels fall back to geometry alone rather than failing.

---

## 5. Bivariate (`/bivariate`)

A full-width map of fronts coloured by a **two-field** scheme, generalised
out of `Bivariate_TurnerAngle.ipynb` into `fronts/viz/bivariate.py`:

| Function | Role |
|---|---|
| `bivariate_colormap(n, hue_a, hue_b)` | `(n, n, 3)` grid; lightness carries A, hue carries B |
| `assign_bins(values_a, values_b, n, edges=...)` | bin index per front |
| `plot_bivariate_map(df, ...)` | scatter or spatially binned |
| `plot_bivariate_legend(...)` | the `n × n` legend square |

Two details from the notebook are worth preserving:

- **Quantile edges, not equal-width**, as the default — front properties
  are heavy-tailed and equal-width bins put nearly every front in one
  cell.
- **A physically meaningful split beats a quantile split when one
  exists.** Turner angle splits at 0°, not at its median. So
  `assign_bins` accepts explicit edges and the registry names a natural
  split for the fields that have one.

The notebook still works and can import these instead of defining them.

---

## 6. Tiles (`/tiles`)

The page is **three stages**, cheapest first, each behind its own button
on its own row. Stage 1 invalidates stage 2; stage 2 never invalidates
stage 1. One arrow, never a cycle — which is what keeps three buttons
comprehensible.

```mermaid
flowchart TB
    A["date · region · field"] -->|"Load tile  (~30 s)"| B["(a) global map + tile map"]
    B --> C["pick a front · pick sigma · pick the map field"]
    C -->|"Build plan view  (~15 s)"| D["(b) isopycnal-depth map<br/>region field map<br/>interactive plan view"]
    D --> E["click along the axis · click profile points"]
    E -->|"Build sections  (~10 s per field)"| F["(c) curtains · inset · profiles · 3-D"]
```

**(a) Overview.** Global map with the region boxed in red — derived from
the tile's *own* `XC`/`YC`, not from a nominal centre, so the box lands
where the tile actually is. Below it, the tile itself, on **lat/lon**
axes.

**(b) Plan view and maps.** The isopycnal-depth map shows the whole tile:
colour is the depth at which `sigma0 == sigma`, **gray** where the
isopycnal does not exist in the column (the surface has outcropped —
a physical statement, not missing data), all fronts in cyan, the selected
one in **red**. Beside it, the same treatment for any chosen field. The
plan view is the interactive one: click to set the cross-front transect,
switch mode and click to drop up to 5 profile points.

**(c) Sections**, one column per selected field (up to three):

| # | Figure |
|---|---|
| a | 3-D field on the front's isopycnals (`pn.pane.VTK`) |
| b | inset — plan view, two rows: surface and a chosen depth |
| c | isopycnal surface |
| d | main-axis curtain |
| e | along-front offsets |
| f | cross-front transect |
| — | vertical profiles, one line per picked point |

The geometry — front, crop, main axis, transect point — is computed
**once** and shared across the columns, so only the colour field differs.
That is both faster and the only reason the columns are comparable at all.

Axis tick distances are drawn on both the plan view and the section
x-axes, so a feature in a section can be found on the map.

Offsets in front of and behind the front are colour-coded, with a
two-entry legend saying which is + and which is −.

### Field list

`config.TILE_FIELDS_3D`. Surface-only quantities are deliberately absent:
`ug`, `vg`, `frontogenesis_geo` and `frontogenesis_ageo` need the
SSH-derived geostrophic velocity, and `KE` is built from the mixed-layer
depth — all exist at one level, so there is nothing to section through.

---

## 7. Conventions learned the hard way

Short list, but each one cost a day.

**A HoloViews stream must be attached to what is actually rendered.** A
`Tap` or `BoundsXY` bound to an element that is then composed into an
Overlay never reaches the renderer. Use a **sourceless** stream as a
`DynamicMap` stream instead.

**Never redraw inside a stream callback.** It destroys the plot the tool
belongs to — which resets the zoom and kills the stream. The pattern that
works everywhere is *static base + dynamic markers*: build the image and
the axis once, put only the markers in a `DynamicMap`.

Those two together are the root cause of the entire "clicking does
nothing / the zoom jumps back / the selection is lost" family, on both
the plan view and the Field Characteristics map.

**`shared_axes=False`**, or HoloViews links the axes of any two plots
sharing a dimension name — which is why zooming the tile map used to zoom
the plan view.

**Do not trust the dim order of a tile variable.** xarray puts the first
operand's dims first, so `mld(j,i) * grad(k,j,i)` comes back `(j,i,k)`
and gets written to the store that way. Read 3-D tile variables through
`pipeline.field_values`, which transposes by dim *name*.

**Other small traps.** `sizing_mode` is not a valid Overlay option
(`responsive=True` is). matplotlib rejects cmocean names, so colour maps
go through `field_styles.resolve_cmap`. `Widget.name` is deprecated and
silently breaks two-way param links — use `.label`.

---

## 8. Evolution — the same figures, as a movie

*(Carried over unchanged: this is the outstanding work.)*

Evolution gets the Tiles figures, with **one field at a time** rather
than a column per field — a movie of three fields is three movies, which
is not a page. The chunk window supplies the frames.

Layout:

* a global map with the chunks boxed, and the chunk map at the current
  step;
* three time series — **(a)** length, **(b)** orientation, **(c)** the
  field's statistics — with a cursor that moves with playback, so you can
  always see where in the window the figures below are;
* **(d)–(i)** the six figures, one frame per step.

Playback swaps pre-rendered images. Rendering costs roughly ten seconds a
frame, so building the movie is an explicit, progress-tracked step and the
result is cached — after which scrubbing is instant.

Decisions already taken:

| Question | Choice |
|---|---|
| Isopycnal `SIGMA` | free number input, defaulting to the tile's median density |
| Profile locations | up to 5, cleared when the front changes |
| Isopycnal map on Evolution | animates with the rest of the movie |

Profiles are cleared per front because the points are picked in the
front's *crop* frame: on a different front the same pixel is a different
place, so keeping them would silently plot the wrong column.

### The window is a week, not a day

Neither chunk is 24 hourly steps. Each is a **week of daily snapshots
wrapped around one intensive day**:

| chunk | steps | shape |
|---|---|---|
| `monterey_bay` | 17 | daily 06-29 → 07-07, **3-hourly** through 07-03 |
| `southern_ocean_scotia_sea` | 20 | daily 10-31 → 11-07, **hourly** through 11-03 |

All of them play, so the **datestamp is drawn at the top of every frame**
— without it there is no way to tell that the interval between frames
just changed by a factor of eight. For the same reason the time series
are plotted against **real time, not step index**: on a step axis the
intensive day is squeezed into the width of one daily gap, which hides
the only part with any time resolution in it.

`EVOLUTION_N_STEPS` and `EVOLUTION_START` are synthetic-mode only. They
name a date neither chunk uses, so they are labelled as such rather than
read as real config.

### Fronts come from the global products

There is no chunk-specific front detection, and none is needed. A chunk
is floored onto the 720-cell tile lattice, so it **is** a rect tile, and
its fronts are the global ones for that timestamp sliced to that window —
the same thing `pipeline.tile_labels` does for a tile.

`chunk_timesteps` intersects the chunk store listing with the timestamps
that actually have a `Fronts/` store, so a snapshot without fronts
(2012-07-09 for Monterey) disappears from the selector instead of failing
in the middle of a movie build.

### Following a front: location, not label

Labels are assigned per date, so the same physical front is called
something different in every frame. Following a *label* jumps to an
unrelated front. `evolution/tracking.py` follows position instead:

```mermaid
flowchart LR
    A["pick a front<br/>at the anchor step"] --> B["Anchor:<br/>centroid + frozen window"]
    B --> C["walk outward,<br/>each step compared<br/>with the last one found"]
    C --> D["nearest candidate<br/>within the drift radius"]
    D --> E["Track: {step: label}<br/>gaps left as gaps"]
```

Three decisions carry the design:

**The search radius scales with elapsed time, not step count.** This is
what the uneven cadence forces. Across one hour a front moves ~2 km
against ~2 km cells; across a day it moves 40 km and overlaps nothing. A
single fixed radius would either drop every daily link or grab a
neighbouring front inside the intensive day. So the radius is
`MAX_DRIFT_MS × Δt`, from the timestamps.

**Chain step to step, not back to the anchor.** Comparing every step with
the anchor loses the front as soon as it advects its own width. Chaining
follows the advection. Mask overlap breaks ties, which only matters
inside the dense day — across a daily gap there is usually no overlap to
break anything with.

**The frozen window is a display crop, not a search gate.** A front that
drifts out of shot over a week is still the same front, so gating the
search on the window would turn a display limitation into missing data.
`Track.first_escape` reports the step where the front leaves the frame,
so the page can say so.

Where nothing is close enough the step is a **gap** — the window is shown
with no front highlighted. A movie with a hole in it is honest; one that
confidently highlights the wrong front is not. A gap does not end the
track: the reference is kept, so the front can be re-acquired later.

### What the frozen window buys elsewhere

`build_step` currently re-crops per front per step, which is why
`shared_settings` exists — a hack to pin `perp_index` and `clim` so the
movie neither jumps nor pulses. A window frozen at the anchor gives all of
that for free: constant crop, constant axes, one colour range, and most of
`shared_settings` goes away. `perp_index` becomes a **fraction along the
axis** rather than an absolute column, so it means the same place as the
axis length changes.

### Page structure — three stages, like Tiles

Evolution mirrors the Tiles flow. Each button builds only its own
figures, which is what keeps the cost honest: nothing renders a curtain
until you are in the transect section.

```mermaid
flowchart TB
    A["chunk · timestep"] -->|"Load chunk"| B["(a) region at one step:<br/>density + all fronts labelled"]
    B --> C["front · isopycnal sigma · region-map field"]
    C -->|"Build region movie"| D["(b) isopycnal-depth movie<br/>+ region-field movie<br/>whole region, 2 renders/frame"]
    D --> E["plan view: transect fraction<br/>+ profile points"]
    E -->|"Build sections"| F["(c) section movie + time series<br/>uses the field chosen in (b)"]
```

**(a)** is one step, not a movie: the region with density behind it and
every front labelled. The *global* overview keeps `gradb2` — that is the
field the fronts were detected on. Anything else in the region figure
comes from (b)'s field selector.

**(b)** is the whole region, not a front crop, so it needs no curtains and
no 3-D: two renders a frame. The selected front is drawn **red in every
frame** via the tracker while the others stay cyan, which doubles as a
visual check that the tracking works before any sections get built. A
tracking gap draws no red front rather than guessing one.

**(c)** has no "fields (max 3)" selector — Evolution is one field at a
time, the one chosen in (b). Two things differ from Tiles here, and both
follow from the frame being frozen:

* **Profile points are fixed in lat/lon**, not in crop pixels. They are
  currently picked in the front's crop frame, which moves with the front;
  anchoring them geographically is what makes "the same location
  throughout" actually true.
* **The transect is a fraction along the front, not a column index.** You
  set "40% along" once; because the front's length and shape change every
  step, the transect walks with it. This is the `perp_index`-as-a-fraction
  change — for Evolution it stops being an optimisation and becomes the
  feature.

The time series are built here too, not on load: they walk every step, so
they belong with the only other thing that does.

**No 3-D frame.** A fixed-camera still cost about as much as the other
five figures together and read worst of all of them as a movie. The 3-D
scene stays on Tiles, where it is interactive and built once.

### Order of work

| # | Step | State |
|---|---|---|
| 1 | `chunk_labels` + front-bearing `chunk_timesteps` | **done** |
| 2 | `evolution/tracking.py`, tested headlessly | **done** |
| 3 | drop the 3-D frame | **done** |
| 4 | `load_chunk` off the server thread, per-stage timings | **done** |
| 5 | three-stage layout: (a) region figure, (b) region movie, (c) sections | to do |
| 6 | freeze the crop in `build_step`; transect as a fraction | to do |
| 7 | profile points anchored in lat/lon | to do |
| 8 | real-time series axis; datestamp on every frame | to do |

Already in place from earlier work: the **Timestep** selector, *Load
chunk* with its progress bar and automatic map redraw, and `gradb2`
behind the global overview with both chunks boxed in red.

### Open question: face-local vs rect alignment

The chunk transfer records **face-local** provenance —
`resolved_face`, `j_start`, `i_start` — while `chunk_labels` slices the
**global** label map with a rect window. On a rotated face those frames do
not correspond, so the labels could come back rotated relative to the
chunk data, and nothing downstream would notice.

Tiles already solves this: remap the tile into the rect frame first
(`tile_lookup` + `remap_to_rect`), then slice global labels by the rect
window. Chunks should go through the same path. `tile_mapping` has only
`rect_ij_to_tile` — no inverse — which is why the current code resolves
the location by searching the global coordinates instead.

Cheapest check: on the region figure, do the fronts sit on the `gradb2`
ridges or beside them? Monterey is on a lat-lon face and may look right
either way; Scotia is the one that would expose a rotation.

## 9. Open

- **Colocation (build_v5 step 4) has not been run for V5.** Until it is,
  Field Characteristics panels (e)/(f) and the Bivariate map stay
  degraded.
- **Tiles are missing for `20120918_110000` and `20121109_120000`** —
  `build_tiles --all-fields` fills them.
- **Chunk fronts are not wired up.** Needs build_v5 steps 2–3 over the
  chunk windows.
- Optionally `hv.QuadMesh` in place of the linearised `hv.Image` axes, if
  the curvilinear error ever starts to matter.
