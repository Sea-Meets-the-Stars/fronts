# `characteristics` — field statistics over a region you draw

Route `/characteristics`. A Pacific-centred global map of one field, with six
statistics panels describing whatever lat/lon box you drag on it.

> **Status: prototype, running on synthetic data.** Complete and working;
> see [WIRING.md](WIRING.md) to switch to the real stores.

## What it shows

```
+---------------------------------+------------------+------------------+
|                                 |   ALL GRID PTS   |   FRONTS ONLY    |
|                                 +------------------+------------------+
|      GLOBAL MAP                 | (a) PDF of field | (a) PDF of field |
|      Pacific-centred            +------------------+------------------+
|      land gray, coastlines      | (b) Joint PDF    | (b) Joint PDF    |
|      lat/lon labels             |     strain vs    |     strain vs    |
|                                 |     vorticity    |     vorticity    |
|      [drag a box to select]     +------------------+------------------+
|                                 | (c) Conditional  | (c) Conditional  |
|                                 |     JPDF on      |     JPDF on      |
|                                 |     {field}      |     {field}      |
+---------------------------------+------------------+------------------+
  [date v] [field v] [x] show fronts        region: global | 30N-45N, 130W-115W
```

**Left column** uses every grid point in the box. **Right column** uses only
front pixels, taken from the colocated output.

## What it's for

The question "does this field behave differently on fronts than off them, and
does that change by region?" — answered without writing a notebook cell per
region.

## Controls

| Control | Effect |
|---|---|
| **date** dropdown | Which timestamp to load. One entry in the prototype. |
| **field** dropdown | Any channel present in that date's subsets. Redraws the map and all six panels. |
| **show fronts** toggle | Overlays the binary front mask on top of the field. |
| **drag a box** on the map | Sets the region. All six panels recompute. |
| **reset region** | Back to the whole globe. |
| Pan / wheel zoom | Standard; the map re-aggregates server-side at each zoom level. |

## The six panels

| Row | Panel | Built by |
|---|---|---|
| a | PDF of the selected field | `dbof.plotting.pdfs.pdf_panel`, with bins from `shared_bins` so both columns share an axis |
| b | Joint PDF: σ/\|f₀\| vs ζ/f₀ | `dbof.plotting.jpdfs.plot_jpdf_occurrence` |
| c | Conditional joint PDF: mean of the selected field per (ζ/f₀, σ/\|f₀\|) bin | `dbof.plotting.jpdfs.plot_jpdf_conditional` |

Row (c) uses `plot_jpdf_conditional_log` for positive-definite fields such as
`gradb2`, and `plot_jpdf_conditional` (symmetric-log, diverging) otherwise.

All three take a matplotlib axis, so they go into `pn.pane.Matplotlib`
unchanged — these are the one part of the stack that needs no adaptation.

## Where the data comes from

`s3://dbof/globals_for_cutouts/v2_2_01/20120516_060000/`

| Artefact | Used for |
|---|---|
| 2-D field per channel | The map, and the left column |
| Binary fronts mask | The **show fronts** overlay |
| Grouped / labelled fronts | Identifying front pixels |
| Colocated front properties (parquet) | The right column |

Loading goes through `fronts.properties.viz_loaders`
(`load_labeled_array`, `load_geometry_table`, `load_colocation_table`,
`merge_geometry_colocation`) with an S3-aware path resolver.

**Channel names must be confirmed against the store, not the config.** Panels
(b) and (c) need vorticity, strain and Coriolis. On the SURFACE pipeline those
are `relative_vorticity`, `strain_mag`, `coriolis_f`. On the DEPTH pipeline the
first two carry depth suffixes (`relative_vorticity_sfc`, `strain_mag_mld`, …)
and `coriolis_f` moves to the extra channels. List the store to find out which.

## The map

The rect grid is 12960 × 17280 — about 224 million pixels — and it is **not a
regular lat/lon grid**. The faces are stitched and rotated rather than
interpolated, `XC`/`YC` remain 2-D, and latitude spacing is non-uniform.
Plotting it as an `hv.Image` would silently misplace data.

So the map is drawn from a **display pyramid**: the field is regridded once,
offline, onto a regular lat/lon multiscale raster stored as zarr. On that
raster `hv.Image` + datashader is correct and fast, and Pacific-centring really
is just a column roll. Cartopy draws coastlines and gray land on top; it never
warps the field.

The pyramid is for looking at. It never feeds a number.

## How the statistics are computed

**Exactly, at full resolution, on the native grid.** No decimation enters the
numbers, and the pyramid is not involved.

A dragged box becomes a selection by masking the 2-D coordinate arrays:

```python
mask = (YC >= lat0) & (YC <= lat1) & (XC >= lon0) & (XC <= lon1)
```

One vectorised pass, exact on an irregular grid, and no nearest-neighbour
search. (`fronts/llc/coords.py` does have a lat/lon → index routine, but it
allocates a ~1.8 GB distance array per query point — fine for a CLI call,
unusable behind an interactive drag.)

Four things keep the exact path usable:

- the read is **chunk-aligned dask**, touching only the masked region;
- it runs in a **background thread** with a progress bar, so the page stays
  live and the panels repopulate when it finishes;
- results are **cached on `(date, field, bbox, fronts_only)`**, so returning to
  a box is instant;
- the **whole-globe default is precomputed offline** into that cache, so the
  page opens on an exact result rather than a cold multi-minute compute.

A large box still takes real time on first request. The progress bar says so.

## When to use this vs. the other viewers

- Use **characteristics** for statistics of a field over an arbitrary region.
- Use [tiles](tiles.md) to look at one specific front in depth.
- Use [global_field_viewer](../global_field_viewer.md) for the desktop
  PyQt equivalent of the map alone, with no statistics.
