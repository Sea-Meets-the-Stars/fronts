# Phase 3 — the V5 dates, and five new Tiles figures

Follows [phase 2](build_front_viz_tool_phase2.md). Phase 2 built the five
pages; this narrows them to the V5 dates and adds the figures that were
missing from the Tiles and Evolution pages.

## What changed in the data

Everything the four pages read now sits under **one** prefix:

```
s3://dbof/globals_for_chunks/V5/{YYYYMMDD_HHMMSS}/
    {subset}.zarr          fields, surface and depth
    Fronts/                binary, labels, geometry (colocation pending)
```

for exactly four dates:

| date | prefix |
|---|---|
| 2012-02-29 18:00 | `20120229_180000` |
| 2012-05-16 06:00 | `20120516_060000` |
| 2012-09-18 11:00 | `20120918_110000` |
| 2012-11-09 12:00 | `20121109_120000` |

3-D tiles come from `s3://dbof/LLC4320_RAW/DEPTH/`, cached as zarr in
`s3://dbof/tiles/{date}/{region}/{field}.zarr`.

Evolution chunks: `s3://dbof/LLC4320_RAW/CHUNKS/{chunk}/{YYYYMMDDTHH}.zarr`,
now `monterey_bay` and `southern_ocean_scotia_sea`.

## Done in this phase

- Dates restricted to the four above, on every page.
- Surface and Depth read `globals_for_chunks/V5` for both fields and fronts.
- Box-select and *Reset region* move the map immediately; only the figures
  below wait for *Rebuild*. Navigation is not computation.
- Taller global maps on Tiles and Evolution.
- `southern_ocean_scotia_sea` added to the Evolution chunk allow-list.
- 3-D figure: depth ticks in metres. The geometry is stretched by
  `zscale` so the volume is readable, which made the raw z numbers
  meaningless; `show_bounds(axes_ranges=...)` relabels the axis with the
  true range and leaves the geometry alone.

---

## Still to build

Five figures and one interaction. The interaction is the hard part, so it
comes first.

### A. Front-axis selection (Tiles)

Today `pick_perp_index` chooses where to cut the cross-front transect —
the field extremum along a clean stretch. The user should choose instead.

**The two-button problem.** The page already has *Regenerate figures*.
Adding "pick a point, then generate" naively gives two buttons with
unclear scope. The resolution is that they operate on **different
stages of the same pipeline**, and the cheap stage should come first:

```
        date / region / field / front            ── stage 1 ──▶  Regenerate tile + plan view
                     │                                            (~15 s per field)
                     ▼
        plan view: click along the front axis    ── stage 2 ──▶  Build sections
                     │                                            (~10 s per field)
                     ▼
        curtains, isopycnal, offsets, perpendicular, profiles
```

So:

- **Regenerate** (stage 1) builds the tile, the front geometry and the
  **plan view** — everything that does not depend on where along the axis
  you cut. Enabled when the date/region/field/front selection changes.
- **Build sections** (stage 2) builds the figures that *do* depend on the
  cut point. Enabled when the axis position or the profile points change.

Stage 1 implies stage 2 is stale; stage 2 never invalidates stage 1. One
arrow, never a cycle — which is what keeps two buttons comprehensible.

The plan view (panel **b** today) becomes interactive: an `hv.Image` of
the front's crop with the axis drawn, `hv.Points` for the tick positions,
and a `Tap` stream that snaps the click to the nearest axis vertex.

**Axis ticks.** The plan view labels the axis `start` and `end` and marks
evenly-spaced ticks along it. The same tick distances are drawn on the
x-axis of the isopycnal and curtain figures, so a feature on the section
can be located on the map. This needs an optional `ticks=` argument
threaded through `curtains.figure_isopycnal_surface` and
`figure_mainaxis` — additive, so the CLI scripts keep working.

### B. Isopycnal depth map (Tiles)

A 2-D map of *the depth at which* `sigma0 == SIGMA`, `SIGMA` configurable.

- colour = depth, one shared colour bar
- **gray** where the isopycnal does not exist in the column — the surface
  has outcropped, which is a physical statement and not missing data
- sits **above** the per-field columns: it is a property of density
  alone, so it does not belong in the one-column-per-field structure

Implementation: for each `(j, i)`, find the first `k` where `sigma0`
crosses `SIGMA` and interpolate `Z` linearly between the bracketing
levels. Vectorised over the tile; no loop.

### C. Vertical profiles (Tiles)

y-axis depth (0 at the top, increasing downward), x-axis the field. One
line per selected location, one panel per field — so this **does** follow
the column-per-field structure.

Locations are picked on the same plan view as the axis point, in a second
click mode: N points, cleared by a button. That keeps every spatial
choice in one figure instead of scattering pickers across the page.

### D. Evolution: the same figures, as a movie

Evolution gets A–C, with one field at a time rather than a column per
field — a movie of three fields is three movies, which is not a page.
The chunk window supplies the frames.

### E. Decisions

| Question | Choice |
|---|---|
| Isopycnal `SIGMA` | free number input, defaulting to the tile's median density |
| Profile locations | up to 5, cleared when the front changes |
| Isopycnal map on Evolution | animates with the rest of the movie |

Profiles are cleared per front because the points are picked in the
front's *crop* frame: on a different front the same pixel is a different
place, so keeping them would silently plot the wrong column.
