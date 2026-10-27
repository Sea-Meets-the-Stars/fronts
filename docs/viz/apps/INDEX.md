# Web viewers in `fronts`

Three browser pages under [fronts/viz/apps/](../../../fronts/viz/apps/). All
three follow the same shape: **global map → pick a region → get plots.**

> **Status: prototype, running on synthetic data.** All three routes work;
> pages 1 and 2 are complete against a fabricated ocean, page 3 is a stub.
> To point them at the real stores see [WIRING.md](WIRING.md); the build
> plan is [prompts/build_front_viz_tool.md](../../../prompts/build_front_viz_tool.md).

| Page | Route | What it shows | Data |
|---|---|---|---|
| [characteristics](characteristics.md) | `/characteristics` | Statistics of one field over a lat/lon box you draw | surface globals |
| [tiles](tiles.md) | `/tiles` | One front inside a 720×720 tile: a 3-D scene, a plan-view inset, and four curtains | 3-D raw tiles |
| evolution | `/evolution` | *specified later* | *later* |

Wiring the pages to real data: [WIRING.md](WIRING.md).

## How this relates to the existing viewers

The scripts in [docs/viz/INDEX.md](../INDEX.md) are single-purpose: one CLI
invocation, one set of figures. These pages wrap the *same* builders in a
browser so you can pick date / field / region / front without editing a command
line.

| Existing script | Superseded by |
|---|---|
| `global_field_viewer.py` (PyQt) | **characteristics** — same global browsing, plus statistics |
| `front_viz_groups_bokeh.py` | **tiles** — same hover-for-front-label, declarative instead of CustomJS |
| `fronts_viz_curtain.py` | **tiles** — same figures, selected by clicking |
| `fronts_viz_3d.py` | **tiles** — same scene, live in the browser |

The scripts stay. They are the right tool for batch rendering and for headless
cluster runs; the pages are the right tool for exploring.

## Stack

| Layer | Library | Role |
|---|---|---|
| App | **Panel** | Serving, layout, widgets, routing |
| Plots | **HoloViews** | Declarative plots; `BoundsXY` / `Tap` streams for selection |
| Big rasters | **Datashader** | Server-side re-aggregation on every zoom |
| Maps | **GeoViews** + cartopy | Coastlines, gray land, lat/lon labels |
| State | **Param** | `date / field / region / front` as typed parameters |
| Existing figures | `pn.pane.Matplotlib`, `pn.pane.VTK` | Embed our curtains and our PyVista scene |

`fronts_3d.render_3d` already returns a `pyvista.Plotter`, which Panel takes
directly. The `curtains.figure_*` builders currently write a PNG and close the
figure, so they need a small change first — see R4 in the build plan.

## The grid, and why the map needs a pyramid

The LLC4320 rect grid is 12960 × 17280 and is **not** a regular lat/lon grid:
the faces are stitched and rotated, not interpolated, and `XC`/`YC` stay 2-D.
Latitude spacing is definitely non-uniform; longitude is only approximately
column-aligned.

So the pages split two things that look like one:

| Path | Grid | Used for |
|---|---|---|
| **Display** | Regridded offline onto a regular lat/lon multiscale pyramid | The map you look at |
| **Statistics** | Native rect grid, read directly | Every number reported |

A lat/lon box is turned into a selection by masking the 2-D coordinate arrays
directly — `(YC >= lat0) & … & (XC <= lon1)` — never by index arithmetic and
never by nearest-neighbour search. That is exact on an irregular grid and cheap
enough to sit behind an interactive drag.

## Shared layer

`fronts/viz/apps/common/` — used by all three pages.

| Module | Role |
|---|---|
| `sources.py` | Open and cache every S3 artefact for a date. Wraps `fronts.properties.viz_loaders`. |
| `state.py` | The `param.Parameterized` classes holding date / field / region / front. |
| `pyramid.py` | Build and read the regular-grid display pyramid. |
| `basemap.py` | Pacific-centred map from the pyramid, gray land, coastlines. |
| `selection.py` | Lat/lon box → boolean mask on the native grid. |
| `regions.py` | The six named page-2 regions, resolved to LLC tile indices. |
| `cache.py` | Memory + disk cache keyed on `(date, field, bbox)`. |
| `widgets.py` | Shared dropdowns, toggles, status bar. |

Putting the state in Param classes means the pages are testable without a
browser: set the parameters, assert the derived values.

## Running

```bash
python -m fronts.viz.apps.serve --show
```

One process, four routes: `/`, `/characteristics`, `/tiles`, `/evolution`.
With no configuration it runs on synthetic data and needs no store, no
credentials and no network. `--data s3` (or `FRONTS_APP_DATA=s3`) switches
the provider; [WIRING.md](WIRING.md) lists what has to be in place first.

The 3-D pane on the **tiles** page needs the OSMesa VTK build and
`DISPLAY=dummy` from step 0 of
[fronts_viz_3d_runbook.md](../fronts_viz_3d_runbook.md). The other two pages
have no rendering requirements beyond a browser.

`llc4320-native-grid-preprocessing` must be importable — the pages use its
`jpdfs`, `pdfs`, `field_registry` and `tile_mapping` modules. `pip install -e`
it, or set `LLC4320_PREPROC_SRC`.

## Dependencies

```
panel  holoviews  datashader  geoviews  hvplot  cmocean
```

All conda-forge. `bokeh` and `pyvista[jupyter]` are already in
`requirements.txt`. **`cartopy` is already imported** by
`fronts/viz/properties.py` and `fronts/plotting/spatial.py` but is declared in
neither `requirements.txt` nor `setup.py` — worth adding while we are here.
