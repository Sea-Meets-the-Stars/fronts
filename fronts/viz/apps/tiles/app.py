"""Tiles — one front in 3-D and cross-section, across up to three fields.

Global map with the six regions boxed in red. Click one (or use the
dropdown), pick a front on the tile map, choose up to three fields, and
press **Regenerate**: the six figures build in one column per field, so the
same front can be compared across fields.

Two things shape this page.

**The geometry is computed once.** Front, crop, main axis, perpendicular
point — all derived from density, which every column shares. Only the
colour field differs per column. That is both faster and the only way the
columns are actually comparable; if each column picked its own
perpendicular point they would be different transects.

**Nothing rebuilds on its own.** A new field costs a tile generation
(~15 s) plus figures (~10 s), so changing a setting marks the figures
stale and the user chooses when to spend it.
"""

from __future__ import annotations

import asyncio

import holoviews as hv
import numpy as np
import panel as pn

from fronts.viz import field_styles
from fronts.viz.geometry import front_bbox_and_crop
from fronts.viz.apps import config
from fronts.viz.apps.common import basemap, regions as regions_mod, widgets
from fronts.viz.apps.common.state import TilesState
from fronts.viz.apps.tiles import panels as F
from fronts.viz.apps.tiles import pipeline

hv.extension("bokeh")

#: Half-width of the red region boxes on the overview map, in degrees.
BOX_HALF = (7.5, 5.0)

#: Cycled so adjacent fronts differ in colour.  Not a scale -- the label
#: itself comes from the hover and the click.
FRONT_PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
)

#: Padding around the selected front when the tile map zooms to it.
TILE_ZOOM_MARGIN = 60

#: Figures that depend on where along the axis the cut is made -- these
#: are what *Build sections* rebuilds, without re-fetching a tile.
SECTION_KEYS = ("isopycnal", "mainaxis", "offsets", "perpendicular",
                "profiles")

#: Which figures a given control actually invalidates.  Adding a profile
#: location used to rebuild all five, so the one cheap figure waited
#: behind the offsets figure -- which is why the profiles looked slow.
SECTION_DEPENDS = {
    "inset_depth": ("inset",),
    # inset is in here because it draws the transect marker: leaving it
    # out meant the plan-view pick moved everywhere except the panel that
    # shows where the cut is.
    "perp_index": ("inset", "isopycnal", "mainaxis", "perpendicular"),
    "perp_half_width": ("perpendicular",),
    "n_offsets": ("offsets",),
    "profile_points": ("profiles",),
}

ROW_KEYS = ("vtk",) + F.FIGURE_ORDER + ("profiles",)
ROW_LABELS = {
    "vtk": "(a) 3-D field on the front's isopycnals",
    **F.FIGURE_TITLES,
    "profiles": "(g) vertical profiles at the chosen locations",
}


class TilesPage:
    """Assembles the tiles page."""

    def __init__(self, provider=None):
        self.state = TilesState(provider=provider)

        self._overview = pn.pane.HoloViews(min_height=540,
                                           sizing_mode="stretch_width")
        self._tilemap = pn.pane.HoloViews(min_height=620,
                                          sizing_mode="stretch_width")
        self._status = widgets.status()
        self._hover = widgets.status("hover a front to read its label")
        self._build_status = widgets.status()

        #: One column of panes per field, built on demand.
        self._columns = pn.Row(sizing_mode="stretch_width")
        self._panes: dict[tuple[str, str], pn.viewable.Viewable] = {}
        #: Fields the current column skeleton was built for, so Regenerate
        #: can tell whether the layout is stale.
        self._column_fields: list[str] = []

        #: Isopycnal-depth map.  Density decides it, so it is one figure
        #: for the whole page rather than one per field column.
        self._isodepth = pn.pane.PNG(sizing_mode="stretch_width",
                                     min_height=380)
        self._iso_status = widgets.status()

        #: The same frame as the isopycnal map, coloured by a chosen field
        #: rather than by depth -- so the two read together.
        self._regionmap = pn.pane.PNG(sizing_mode="stretch_width",
                                      min_height=380)
        self._region_status = widgets.status()

        #: The clickable plan view, and the scenes it is drawn from.
        self._planview = pn.pane.HoloViews(sizing_mode="stretch_width",
                                           min_height=580)
        self._plan_status = widgets.status(
            "press *Regenerate figures* to build the plan view")
        self._scenes: dict = {}
        #: The density scene behind figure (c).  Kept separately from the
        #: per-field scenes: the plan view exists before any field has
        #: been chosen, and the geometry it shows is density's.
        self._density_scene = None
        #: Section figures known to be out of date.  Empty means "all of
        #: them" -- the first build, or a new front.
        self._stale_keys: set[str] = set()
        #: Fields with a complete set of figures for the current scenes.
        #: Anything not in here is built in full rather than filtered.
        self._built_fields: set[str] = set()
        #: Drives the marker layer of the plan view.  Firing this redraws
        #: the markers without touching the plot, so the zoom survives.
        self._plan_marks = hv.streams.Stream.define("PlanMarks", tick=0)()
        #: Click coordinates.  Deliberately *sourceless*: a Tap bound to an
        #: element only reaches the renderer if that element is the thing
        #: displayed, and here the displayed object is an Overlay built
        #: from it.  A sourceless Tap used as a DynamicMap stream is linked
        #: to whatever plot the DynamicMap is rendered in, which is the
        #: composite -- so the clicks actually arrive.
        self._tap = hv.streams.Tap(x=None, y=None)
        #: The last click acted on, so re-renders do not re-apply it.
        self._last_tap = (None, None)

        self._labels_tile = None
        #: Last tile drawn, so re-selecting a front does not re-fetch it.
        #: Redrawing is now driven by front_label too, and every redraw
        #: hitting S3 would make picking a front feel broken.
        self._tile_cache: tuple[tuple, object] | None = None
        self._token = 0

        self._build_controls()
        # Only the 2-D overview at load.  Drawing the tile map here meant
        # every page view fetched -- or generated from LLC4320_RAW/DEPTH --
        # a 3-D tile for whichever region happened to be selected, before
        # the user had configured anything.
        self.draw_overview()
        self._render_columns()
        self._status.object = (
            "pick a region and a field, then press *Load tile* — "
            "the 3-D tile is only fetched when you ask for it")

    # -- controls --------------------------------------------------------

    def _build_controls(self):
        s = self.state
        self.w_date = pn.widgets.Select.from_param(s.param.date, width=175)
        self.w_region = pn.widgets.Select.from_param(s.param.region, width=225)
        self.w_fields = pn.widgets.MultiChoice.from_param(
            s.param.fields, name=f"Fields (max {s.MAX_FIELDS})",
            max_items=s.MAX_FIELDS, width=340,
        )
        self.w_fronts = pn.widgets.Checkbox.from_param(s.param.show_fronts,
                                                       name="Show fronts")
        self.w_label = pn.widgets.IntInput.from_param(
            s.param.front_label, name="Front label", width=110, step=1)
        self.w_offsets = pn.widgets.IntSlider.from_param(
            s.param.n_offsets, name="Offsets per side", width=165)
        self.w_perp = pn.widgets.IntSlider.from_param(
            s.param.perp_half_width, name="Transect half-width", width=175)
        self.w_avail = pn.widgets.Select(name="Fronts in this tile",
                                         options=[], width=160)
        self.w_avail.param.watch(self._pick_from_list, "value")

        self.w_sigma = pn.widgets.FloatInput.from_param(
            s.param.sigma, name="Isopycnal sigma [kg/m^3]", step=0.05,
            width=185)
        self.w_regionfield = pn.widgets.Select.from_param(
            s.param.region_field, name="Region map field", width=185)
        self.w_insetdepth = pn.widgets.FloatInput.from_param(
            s.param.inset_depth, name="Inset depth [m]", step=10.0,
            width=155)
        self.w_loadtile = pn.widgets.Button(
            name="Load tile", button_type="primary", width=140)
        self.w_loadtile.on_click(lambda _: self._load_tile_clicked())

        self.w_plan = pn.widgets.Button(
            name="Build plan view", button_type="primary", width=170)
        self.w_plan.on_click(lambda _: self.build_plan())

        self.w_pickmode = pn.widgets.RadioButtonGroup.from_param(
            s.param.pick_mode, button_type="default", width=250)
        # A slider as well as the click.  Clicking a HoloViews overlay
        # depends on the Tap stream surviving a redraw, and the redraw is
        # what draws the marker -- so the click alone is a fragile way to
        # set something this important.  The slider always works.
        self.w_axis = pn.widgets.IntSlider(
            name="Transect at axis vertex", start=0, end=1, value=0,
            width=300)
        self.w_axis.param.watch(
            lambda e: setattr(s, "perp_index", int(e.new)), "value")
        self.w_clearpts = pn.widgets.Button(name="Clear points", width=130)
        self.w_clearpts.on_click(lambda _: self._clear_points())
        self.w_sections = pn.widgets.Button(
            name="Build sections", button_type="primary", width=170)
        self.w_sections.on_click(lambda _: self.build_sections())

        # Front label and sigma feed figure (c), so they mark that stale
        # rather than the sections.
        s.param.watch(lambda *_: self._stale_plan(),
                      ["front_label", "sigma", "region_field"])
        s.param.watch(self._on_section_control,
                      list(SECTION_DEPENDS))

        # Only front_label redraws by itself: by then the tile is in the
        # page's memo, so it is a repaint, not a fetch.  Changing region,
        # date or field needs a different tile, so it waits for
        # *Load tile*.
        s.param.watch(lambda *_: self._on_tile_selection(),
                      ["region", "field", "date"])
        s.param.watch(lambda *_: self.draw_tile(), ["show_fronts",
                                                    "front_label"])
        s.param.watch(lambda *_: self._render_columns(), ["fields"])
        s.param.watch(lambda *_: self._reflect_dirty(), ["dirty"])
        self._reflect_dirty()

    def _load_tile_clicked(self):
        """Fetch the tile, with the button showing that it is working."""
        self.w_loadtile.loading = True
        self.w_loadtile.name = "Loading…"
        self._status.object = f"reading tile for **{self.state.region}** …"
        try:
            self.draw_tile(force=True)
        finally:
            self.w_loadtile.loading = False
            self.w_loadtile.name = "Load tile"

    def _on_tile_selection(self):
        """A different tile is wanted.  Say so; do not fetch it."""
        self._tile_cache = None
        self._labels_tile = None
        self._scenes.clear()
        self._density_scene = None
        self._stale_keys.clear()
        self._built_fields.clear()
        self.w_loadtile.button_type = "primary"
        self._status.object = (
            f"**{self.state.region}** · {self.state.date} · "
            f"{self.state.field} — press *Load tile*")

    def _pick_from_list(self, event):
        if event.new:
            self.state.select_front(int(event.new))

    def _on_section_control(self, event):
        """Record which figures this control invalidated, and only those."""
        self._stale_keys.update(SECTION_DEPENDS.get(event.name, SECTION_KEYS))

    def _stale_plan(self):
        # A new front is new geometry, so the per-field scenes are wrong
        # too -- not just the plan view.
        self._scenes.clear()
        self._density_scene = None
        self._stale_keys.clear()
        self._built_fields.clear()
        self.w_plan.button_type = "primary"
        self._plan_status.object = (
            f"front **{self.state.front_label or '—'}** · sigma "
            f"**{self.state.sigma or 'auto'}** — press *Build plan view*")

    def _reflect_dirty(self):
        """Show whether the section figures match the current settings."""
        if self.state.dirty:
            self.w_sections.button_type = "primary"
            self._build_status.object = (
                "⟳ **settings changed** — press *Build sections*"
                f"{self._cost_note()}")
        else:
            self.w_sections.button_type = "default"

    def _cost_note(self) -> str:
        n = len(self.state.fields)
        return (f" (up to {n} tile{'s' if n != 1 else ''} to generate, "
                f"~{15 * n + 10}s)")

    # -- overview map ----------------------------------------------------

    def draw_overview(self):
        s = self.state
        try:
            base = basemap.global_map(
                s.provider, s.date, "gradb2",
                height=520, title="Click a region",
                tools=("tap",), active_tools=("tap",),
            )
        except Exception as exc:                            # noqa: BLE001
            self._overview.object = None
            self._status.object = f"**Overview unavailable:** {exc}"
            return

        # Boxes drawn at the tile's *actual* extent.  Using the region's
        # nominal centre put the box wherever the config said, while the
        # tile is the 720-cell block the centre happens to fall in -- so
        # the outline and the data it labelled could be far apart.
        boxes, labels, selected = [], [], []
        sel_name = s.region
        for r in regions_mod.REGIONS:
            try:
                idx = (regions_mod.synthetic_tile_idx(r) if s.synthetic
                       else s.tile_index_of(r))
                lon0, lat0, lon1, lat1 = regions_mod.tile_extent(
                    s.provider, s.date, idx)
            except Exception:                               # noqa: BLE001
                continue
            (selected if r.name == sel_name else boxes).append(
                (lon0, lat0, lon1, lat1))
            labels.append((0.5 * (lon0 + lon1), lat1 + 2.0, r.name))

        overlay = base
        if boxes:
            overlay = overlay * hv.Rectangles(boxes).opts(
                fill_alpha=0.0, line_color="red", line_width=2)
        if selected:
            overlay = overlay * hv.Rectangles(selected).opts(
                fill_alpha=0.12, fill_color="red", line_color="red",
                line_width=3)
        overlay = overlay * hv.Labels(labels, vdims="text").opts(
            text_color="red", text_font_size="7pt", text_align="center")
        tap = hv.streams.Tap(source=overlay)
        tap.add_subscriber(self._on_region_tap)
        self._overview.object = overlay

    def _on_region_tap(self, x, y):
        if x is None or y is None:
            return
        hit = regions_mod.nearest(float(y), ((float(x) + 180) % 360) - 180)
        if hit is not None:
            self.state.region = hit.name

    # -- tile map --------------------------------------------------------

    def draw_tile(self, force: bool = False):
        """Draw the tile map.  Only fetches when asked to.

        Called from *Load tile* (``force``) and from the front / fronts
        toggles, which are repaints of a tile already in the memo.  With
        no tile loaded and no force, this does nothing -- that is what
        keeps a page view off LLC4320_RAW/DEPTH.
        """
        s = self.state
        if not force and self._tile_cache is None:
            return
        try:
            tile_idx = s.tile_index()
            ds = self._cached_tile(s.date, tile_idx, s.field, s.region)
            var = ds.attrs.get("tile_var_name") or pipeline._sole_3d(ds)
            # Remap to the rect frame first, so the surface and the labels
            # share one orientation -- the convention fronts_viz_curtain
            # and fronts_viz_3d already use.
            lookup = pipeline.tile_lookup(ds, synthetic=s.provider.synthetic)
            surface = pipeline.remap_to_rect(
                pipeline.field_values(ds, var), lookup)[0]
            tile_lon = pipeline.remap_to_rect(
                pipeline.field_values(ds, "XC"), lookup) % 360.0
            tile_lat = pipeline.remap_to_rect(
                pipeline.field_values(ds, "YC"), lookup)

            # Row/column centres of the tile's own coordinates, so every
            # layer below is in degrees rather than pixel counts.
            nj, ni = surface.shape
            xs = np.linspace(float(np.nanmin(tile_lon)),
                             float(np.nanmax(tile_lon)), ni)
            ys = np.linspace(float(np.nanmin(tile_lat)),
                             float(np.nanmax(tile_lat)), nj)
        except Exception as exc:                            # noqa: BLE001
            self._tilemap.object = None
            self._status.object = (
                f"**Tile unavailable** — {type(exc).__name__}: {exc}")
            self.w_avail.options = []
            return

        # A tile without fronts is still worth drawing -- the label map is
        # a separate product and may not have been built for this date.
        try:
            labels = pipeline.tile_labels(s.provider, s.date, tile_idx,
                                          surface.shape, ds=ds,
                                          region=s.region)
        except Exception as exc:                            # noqa: BLE001
            labels = np.zeros(surface.shape, dtype=np.int32)
            self._status.object = f"*Fronts not overlaid:* {exc}"

        self._labels_tile = labels
        self._tile_coords = (xs, ys)
        self.w_loadtile.button_type = "default"
        available = pipeline.available_fronts(labels)
        self.w_avail.options = [str(l) for l in available]

        # Same style the curtains and the 3-D scene use, so a field looks
        # the same wherever it appears.
        style = field_styles.get_style(
            ds.attrs.get("tile_var_name") or var)
        shown_surface = field_styles.apply_transform(surface, style)
        clim = field_styles.default_clim(shown_surface, style)

        img = hv.Image(
            (xs, ys, shown_surface), kdims=["lon", "lat"], vdims=[s.field],
        ).opts(cmap=basemap.bokeh_cmap(style.cmap), clim=clim,
               colorbar=True, tools=["hover"])

        layers = [img]
        if s.show_fronts:
            # Two value dimensions: the first colours the overlay, the
            # second carries the real label so hover reports the number
            # the dropdown uses.  Colouring by label directly would give
            # 500 near-identical shades.
            palette_idx = np.where(labels > 0,
                                   (labels - 1) % len(FRONT_PALETTE),
                                   np.nan).astype(float)
            true_label = np.where(labels > 0, labels, np.nan).astype(float)
            layers.append(hv.Image(
                (xs, ys, palette_idx, true_label),
                kdims=["lon", "lat"], vdims=["front", "label"],
            ).opts(cmap=list(FRONT_PALETTE), colorbar=False, tools=["hover"],
                   clim=(0, len(FRONT_PALETTE) - 1)))

            layers.append(self._label_markers(labels, available,
                                              coords=(xs, ys)))

            if s.front_label:
                picked = np.where(labels == s.front_label, 1.0, np.nan)
                layers.append(hv.Image(
                    (xs, ys, picked), kdims=["lon", "lat"], vdims=["selected"],
                ).opts(cmap=["#00e5ff"], colorbar=False, clim=(0, 1)))

        xlim, ylim = self._tile_zoom(labels, surface.shape, xs, ys)
        overlay = hv.Overlay(layers).opts(hv.opts.Overlay(
            responsive=True, height=600, active_tools=["tap"],
            # Both this and the plan view carry dimensions called i/j, and
            # HoloViews links axes across plots that share dimension names
            # -- which is why zooming one moved the other.
            shared_axes=False,
            title=f"{s.region}  —  tile {tile_idx}  —  {s.field}",
            xlabel="i (tile pixels)", ylabel="j (tile pixels)",
            xlim=xlim, ylim=ylim))

        hv.streams.Tap(source=overlay).add_subscriber(self._on_tile_tap)
        hv.streams.PointerXY(source=overlay).add_subscriber(self._on_pointer)

        self._tilemap.object = overlay
        self._status.object = (
            f"tile **{tile_idx}** · {len(available)} fronts with 25+ pixels"
            if available else
            f"tile **{tile_idx}** · no fronts with 25+ pixels here")

    def _cached_tile(self, date, tile_idx, field, region):
        key = (date, tile_idx, field, region)
        if self._tile_cache is not None and self._tile_cache[0] == key:
            return self._tile_cache[1]
        ds = self.state.provider.tile(date, tile_idx, field, region)
        self._tile_cache = (key, ds)
        return ds

    def _tile_zoom(self, labels, shape, lon=None, lat=None):
        """Axis limits for the tile map: the selected front, or the tile.

        A front is a filament in a 720 x 720 window, so showing the whole
        tile means the thing that was selected is a few pixels wide.  The
        window is the same crop the figures use, so what is on the map and
        what is in the panels below are the same piece of ocean.
        """
        nj, ni = shape

        def to_deg(i0, i1, j0, j1):
            """Pixel window -> the axis units the plot is actually in.

            The map is in degrees now; returning pixel indices here is
            what made *Load tile* jump to a nonsense window.
            """
            if lon is None or lat is None:
                return ((i0, i1), (j0, j1))
            i0, i1 = int(np.clip(i0, 0, ni - 1)), int(np.clip(i1, 0, ni - 1))
            j0, j1 = int(np.clip(j0, 0, nj - 1)), int(np.clip(j1, 0, nj - 1))
            return ((float(lon[i0]), float(lon[i1])),
                    (float(lat[j0]), float(lat[j1])))

        full = to_deg(0, ni - 1, 0, nj - 1)

        label = int(self.state.front_label or 0)
        if not label or labels is None or not np.any(labels == label):
            return full

        try:
            j_slice, i_slice = front_bbox_and_crop(
                labels, label, margin=TILE_ZOOM_MARGIN)
        except Exception:                                   # noqa: BLE001
            return full
        return to_deg(i_slice.start, i_slice.stop - 1,
                      j_slice.start, j_slice.stop - 1)

    def _label_markers(self, labels, available, *, top: int = 15,
                       coords=None):
        """The front number printed on the map, at each front's centroid.

        Hover gives one label at a time; this makes the biggest fronts
        readable at a glance, so the dropdown and the picture can be
        matched up without pointing at anything.
        """
        rows = []
        lon, lat = coords if coords is not None else (None, None)

        # `available` is in numerical order now, for the dropdown's sake --
        # so rank by size here instead of trusting the incoming order.
        # Taking the first N of a numerical list would annotate whichever
        # fronts happen to have small numbers.
        sized = sorted(
            ((int((labels == label).sum()), int(label)) for label in available),
            reverse=True)[:top]

        for _, label in sized:
            js, iss = np.nonzero(labels == label)
            if not js.size:
                continue
            jm, im = float(js.mean()), float(iss.mean())
            if lon is not None:
                # Rounded only to index the coordinate arrays; the pixel
                # path keeps the fractional centroid.
                rows.append((float(lon[min(int(round(im)), len(lon) - 1)]),
                             float(lat[min(int(round(jm)), len(lat) - 1)]),
                             str(label)))
            else:
                rows.append((im, jm, str(label)))

        if not rows:
            return hv.Labels([], kdims=["lon", "lat"], vdims=["text"])

        # A dark chip behind the text, not a bare colour.  A label sits at
        # its front's centroid, which lands on the bright end of the
        # greyscale ramp or on a saturated overlay colour about as often
        # as it lands on the background -- so no single text colour is
        # readable everywhere, and the background is what fixes it.
        return hv.Labels(rows, kdims=["lon", "lat"], vdims=["text"]).opts(
            text_color="white", text_font_size="8pt", text_align="center",
            text_baseline="middle", text_font_style="bold",
            background_fill_color="#101010", background_fill_alpha=0.72,
            padding=2, border_radius=2)

    def _lookup_label(self, x, y):
        if self._labels_tile is None or x is None or y is None:
            return 0
        j, i = int(round(float(y))), int(round(float(x)))
        nj, ni = self._labels_tile.shape
        if not (0 <= j < nj and 0 <= i < ni):
            return 0
        return int(self._labels_tile[j, i])

    def _on_pointer(self, x, y):
        label = self._lookup_label(x, y)
        self._hover.object = (f"front **{label}**" if label
                              else "hover a front to read its label")

    def _on_tile_tap(self, x, y):
        label = self._lookup_label(x, y)
        if label:
            self.state.select_front(label)

    # -- figure columns --------------------------------------------------

    def _render_columns(self):
        """Rebuild the column skeleton for the selected fields."""
        fields = list(self.state.fields)
        self._panes = {}
        self._column_fields = fields
        columns = []

        for field in fields:
            items = [pn.pane.Markdown(f"#### {field}", margin=(4, 5, 0, 5))]
            for key in ROW_KEYS:
                if key == "vtk":
                    pane = pn.Column(min_height=320,
                                     sizing_mode="stretch_width")
                else:
                    pane = pn.pane.PNG(sizing_mode="stretch_width",
                                       min_height=160)
                self._panes[(field, key)] = pane
                items.append(pn.pane.Markdown(
                    f"<small>{ROW_LABELS[key]}</small>", margin=(4, 5, 0, 5)))
                items.append(pane)
            columns.append(pn.Column(*items, sizing_mode="stretch_width"))

        self._columns.objects = columns
        self._reflect_dirty()

    def schedule_figures(self):
        """Build every column.  Explicitly triggered by Regenerate."""
        # Reconcile the skeleton with the selection first.  The columns are
        # normally rebuilt by the ``fields`` watcher, but Regenerate is the
        # point at which what is on screen must match what was asked for,
        # so it is rebuilt here rather than trusted to be current.
        if self._column_fields != list(self.state.fields):
            self._render_columns()

        self._token += 1
        token = self._token

        if not self.state.front_label:
            self._build_status.object = (
                "**Pick a front first** — click one on the tile map, or "
                "choose a label from the dropdown.")
            return

        for pane in self._panes.values():
            pane.loading = True
        self._build_status.object = "building…"

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._build(token)
        else:
            pn.state.execute(
                lambda: asyncio.ensure_future(self._build_async(token)))

    async def _build_async(self, token):
        await asyncio.to_thread(self._build, token)

    def _build(self, token):
        s = self.state
        fields = list(s.fields)
        done, failed = [], []
        # Scenes are kept between builds, so changing a pick and pressing
        # the button again is figures only -- no tile fetch.  They are
        # dropped when the front or the tile changes, which is where they
        # would actually be wrong.

        # The geometry is a property of the front, not of the colour
        # field, so it is resolved from the first column and the crop /
        # axis / perpendicular point are reused by the rest.
        shared_index = None

        # Only what a control actually invalidated -- but a field with no
        # figures yet gets all of them.  Filtering by the stale set alone
        # meant that picking anything before the first build left the
        # panels nothing marked stale (inset, offsets) permanently empty.
        all_keys = {"inset", *SECTION_KEYS}
        #: Accumulated across columns rather than read off the loop
        #: variable at the end: if every field fails, the loop `continue`s
        #: before `wanted` is ever bound and the status line -- the one
        #: place that would have said *why* -- raises instead.
        built_keys: set[str] = set()

        for field in fields:
            if token != self._token:
                return
            try:
                cached = self._scenes.get(field)
                scene = cached if cached is not None else pipeline.build_scene(
                    s.provider, s.date, s.tile_index(), field,
                    s.front_label, region=s.region)
            except Exception as exc:                        # noqa: BLE001
                failed.append((field, exc))
                for key in ROW_KEYS:
                    pane = self._panes.get((field, key))
                    if pane is None:
                        continue
                    if key == "vtk":
                        pane.objects = [widgets.error_card(exc, field)]
                    else:
                        pane.object = None
                        pane.alt_text = f"{field} failed: {exc}"
                    pane.loading = False
                continue

            self._scenes[field] = scene
            wanted = (all_keys if field not in self._built_fields
                      else (set(self._stale_keys) or all_keys))
            built_keys |= wanted

            if shared_index is None:
                # The axis comes from density, so the plan view's pick
                # applies unchanged to every colour field.
                shared_index = self._resolved_perp_index(scene)
                # One isopycnal map for the page, from the first column's
                # scene -- density is shared, so any column would give the
                # same answer.
                self._build_isodepth(scene, token)
            idx = shared_index

            builders = {
                "inset": lambda sc=scene, i=idx: F.figure_inset(
                    sc, perp_index=i, half_width=s.perp_half_width,
                    depth=s.inset_depth),
                "isopycnal": lambda sc=scene, i=idx: F.figure_isopycnal(
                    sc, perp_index=i),
                "mainaxis": lambda sc=scene, i=idx: F.figure_mainaxis(
                    sc, perp_index=i),
                "offsets": lambda sc=scene: F.figure_offsets(
                    sc, n_offsets=s.n_offsets),
                "perpendicular": lambda sc=scene, i=idx: F.figure_perpendicular(
                    sc, index=i, half_width=s.perp_half_width),
                # Was missing entirely, which is why the profiles pane
                # never filled: the only builder that knew about them was
                # the one nothing called.
                "profiles": lambda sc=scene: F.figure_profiles(
                    sc, list(s.profile_points)),
            }
            for key, build in builders.items():
                if key not in wanted:
                    continue
                pane = self._panes.get((field, key))
                if pane is None:
                    continue
                try:
                    pane.object = str(build())
                except Exception as exc:                    # noqa: BLE001
                    pane.object = None
                    pane.alt_text = f"{key} failed: {exc}"
                pane.loading = False

            self._build_vtk(field, scene)
            self._built_fields.add(field)
            done.append(field)

        if token != self._token:
            return

        s.dirty = False
        parts = [f"built **{len(done)}** column{'s' if len(done) != 1 else ''} "
                 f"for front **{s.front_label}**"]
        if shared_index is not None:
            parts.append(f"shared transect at column {shared_index}")
        if failed:
            # Name them.  A bare count sent you to the terminal to find
            # out which column broke and why.
            parts.append("failed: " + "; ".join(
                f"**{f}** ({type(e).__name__}: {e})" for f, e in failed))
        if built_keys:
            parts.append("figures: " + ", ".join(sorted(built_keys)))
        self._build_status.object = " · ".join(parts)
        self._stale_keys.clear()
        self.w_sections.button_type = "default"

    # -- interactive plan view -------------------------------------------

    def draw_planview(self, scene):
        """The clickable plan view: axis, offsets, transect, profile points.

        The base -- image, axis, offsets, ticks -- is built **once** and
        kept.  Only the markers live in a ``DynamicMap``, so a click
        redraws the markers and nothing else.  Redrawing the whole overlay
        (what this did before) both reset the zoom and replaced the
        element the Tap stream was attached to, which is why clicking
        appeared to do nothing but jump back to the start.
        """
        surface = np.asarray(scene.color[0])
        nj, ni = surface.shape
        # Real coordinates, not pixel indices: the crop's own XC/YC are
        # in the scene, and a map with i/j axes cannot be compared with
        # anything else on the page.
        xs, ys, path = self._plan_coords(scene)

        img = hv.Image((xs, ys, surface), kdims=["lon", "lat"],
                       vdims=[scene.field_name]).opts(
            cmap=basemap.bokeh_cmap(scene.style.cmap), clim=scene.clim,
            colorbar=True, tools=["tap", "hover"])

        layers = [img,
                  hv.Path([path[:, ::-1]]).opts(color="white", line_width=3),
                  hv.Path([path[:, ::-1]]).opts(color="black",
                                                line_width=1.2)]
        px_lon = float(xs[1] - xs[0]) if len(xs) > 1 else 0.05
        px_lat = float(ys[1] - ys[0]) if len(ys) > 1 else 0.05

        # Offset rows either side of the axis, coloured by sign so the
        # along-front offsets panel can be read against a side.
        normals = scene.metrics.get("normals")
        if normals is not None:
            for k in range(1, int(self.state.n_offsets) + 1):
                for sign, colour in ((+1, F.OFFSET_PLUS),
                                     (-1, F.OFFSET_MINUS)):
                    step = np.asarray(normals) * np.array([px_lat, px_lon])
                    off = path + sign * k * step
                    layers.append(hv.Path([off[:, ::-1]]).opts(
                        color=colour, line_width=1.1,
                        line_dash="dashed" if k > 1 else "solid",
                        alpha=0.9 - 0.12 * (k - 1)))

        ticks = F.axis_ticks(scene)
        if ticks:
            marks = [(float(path[k][1]), float(path[k][0])) for k, _ in ticks]
            layers.append(hv.Points(marks).opts(color="black", size=6,
                                                fill_alpha=0.6))
            layers.append(hv.Labels(
                [(m[0], m[1], f"{km:.0f} km") for m, (_, km)
                 in zip(marks, ticks)], kdims=["lon", "lat"], vdims=["text"]
            ).opts(text_color="white", text_font_size="7pt",
                   background_fill_color="#101010",
                   background_fill_alpha=0.7, padding=2))

        layers.append(hv.Labels(
            [(float(path[0][1]), float(path[0][0]), "start"),
             (float(path[-1][1]), float(path[-1][0]), "end")],
            kdims=["lon", "lat"], vdims=["text"],
        ).opts(text_color="#ffd400", text_font_size="9pt",
               text_font_style="bold",
               background_fill_color="#101010",
               background_fill_alpha=0.7, padding=2))

        base = hv.Overlay(layers)
        markers = hv.DynamicMap(self._plan_markers,
                                streams=[self._plan_marks, self._tap])

        overlay = (base * markers).opts(hv.opts.Overlay(
            responsive=True, height=680, active_tools=["tap"],
            shared_axes=False,
            # Explicit limits: without them an Image of an elongated crop
            # is letterboxed into a fraction of the pane.
            xlim=(float(xs[0]), float(xs[-1])),
            ylim=(float(ys[0]), float(ys[-1])),
            title=f"Plan view — front {scene.label} — "
                  f"click to set the {self.state.pick_mode}",
            xlabel="longitude", ylabel="latitude"))

        self._planview.object = overlay

        n = max(len(path) - 1, 1)
        self.w_axis.end = n
        self.w_axis.value = min(int(self._resolved_perp_index(scene)), n)
        self._plan_marks.event(tick=self._plan_marks.contents.get("tick", 0) + 1)
        self._plan_status.object = self._plan_note(scene)

    def _build_regionmap(self, scene, token):
        """The region map of the chosen field, in the isopycnal's frame."""
        s = self.state
        try:
            ctx = self._tile_context()
            idx = s.tile_index()
            ds = self._cached_tile(s.date, idx, s.region_field, s.region)
            lookup = pipeline.tile_lookup(ds, synthetic=s.synthetic)
            var = ds.attrs.get("tile_var_name") or pipeline._sole_3d(ds)
            surface = pipeline.remap_to_rect(
                pipeline.field_values(ds, var), lookup)[0]
            path = F.figure_region_field(
                scene, surface, ctx.get("tile_labels"),
                field_name=var,
                lon=ctx.get("tile_lon"), lat=ctx.get("tile_lat"))
        except Exception as exc:                            # noqa: BLE001
            if token == self._token:
                self._regionmap.object = None
                self._region_status.object = (
                    f"**Unavailable** — {type(exc).__name__}: {exc}")
            return
        if token != self._token:
            return
        self._regionmap.object = str(path)
        self._region_status.object = (
            f"**{s.region_field}** at the surface over the whole tile")

    def _tile_context(self):
        """The whole density tile, for the isopycnal map's wider view.

        Empty when the tile is not in hand, in which case the figure
        falls back to the front's crop.
        """
        s = self.state
        try:
            idx = s.tile_index()
            ds = self._cached_tile(s.date, idx, config.TILE_GEOMETRY_FIELD,
                                   s.region)
            lookup = pipeline.tile_lookup(ds, synthetic=s.synthetic)
            var = ds.attrs.get("tile_var_name") or pipeline._sole_3d(ds)
            sigma0 = pipeline.remap_to_rect(
                pipeline.field_values(ds, var), lookup)
            XC = pipeline.remap_to_rect(
                pipeline.field_values(ds, "XC"), lookup)
            YC = pipeline.remap_to_rect(
                pipeline.field_values(ds, "YC"), lookup)
            labels = pipeline.tile_labels(s.provider, s.date, idx,
                                          sigma0.shape[1:], ds=ds,
                                          region=s.region)
            return {
                "tile_sigma0": sigma0,
                "tile_Z": np.asarray(ds["Z"].values),
                "tile_labels": labels,
                "tile_lon": np.asarray(XC) % 360.0,
                "tile_lat": np.asarray(YC),
            }
        except Exception as exc:                            # noqa: BLE001
            print(f"[isodepth] falling back to the crop: "
                  f"{type(exc).__name__}: {exc}")
            return {}

    def _plan_coords(self, scene):
        """``(lon, lat, axis_path_in_degrees)`` for the crop.

        The axis path is in crop pixels, so it is converted with the same
        coordinate arrays -- otherwise the line and the image would be in
        different spaces and the click would resolve against the wrong one.
        """
        # scene.XC / scene.YC are already the crop -- slicing them again
        # would crop the crop.
        XC = np.asarray(scene.XC)
        YC = np.asarray(scene.YC)
        # Evenly spaced across the true extent: hv.Image assumes a regular
        # raster, and the curvilinear centres are not evenly sampled -- it
        # would place cells slightly wrong and says so.  Linearising over
        # one crop is sub-cell, and the same approximation the pyramid
        # makes for the global maps.
        lon_all = np.asarray(XC, dtype=float) % 360.0
        lat_all = np.asarray(YC, dtype=float)
        lon = np.linspace(float(np.nanmin(lon_all)),
                          float(np.nanmax(lon_all)), XC.shape[1])
        lat = np.linspace(float(np.nanmin(lat_all)),
                          float(np.nanmax(lat_all)), YC.shape[0])

        path = np.asarray(scene.axis_path)
        deg = np.stack([lat[np.clip(path[:, 0], 0, len(lat) - 1)],
                        lon[np.clip(path[:, 1], 0, len(lon) - 1)]], axis=1)
        return lon, lat, deg

    def _plan_markers(self, tick=0, x=None, y=None):
        """The markers, and the place a click is turned into a selection.

        Doing the pick here rather than in a separate subscriber is what
        makes it reliable: this callback is only ever run by the renderer
        that drew the plot, so if the user can see the markers the click
        path is live.
        """
        scene = self._first_scene()
        if scene is None:
            return hv.Points([]).opts(alpha=0)

        if x is not None and y is not None and (x, y) != self._last_tap:
            self._last_tap = (x, y)
            self._apply_pick(scene, float(x), float(y))

        lon, lat, deg = self._plan_coords(scene)
        items = []

        idx = self._resolved_perp_index(scene)
        if 0 <= idx < len(deg):
            items.append(hv.Points([(float(deg[idx][1]),
                                     float(deg[idx][0]))]).opts(
                color="#00e5ff", size=17, marker="x", line_width=4))

        dlat = float(lat[1] - lat[0]) if len(lat) > 1 else 0.05
        for n, (j, i) in enumerate(self.state.profile_points):
            colour = F.PROFILE_COLORS[n % len(F.PROFILE_COLORS)]
            px = float(lon[min(int(i), len(lon) - 1)])
            py = float(lat[min(int(j), len(lat) - 1)])
            items.append(hv.Points([(px, py)]).opts(
                color=colour, size=13, marker="triangle",
                line_color="white", line_width=1.5))
            items.append(hv.Labels(
                [(px, py + 4 * dlat, str(n + 1))],
                kdims=["lon", "lat"], vdims=["text"],
            ).opts(text_color=colour, text_font_size="9pt",
                   text_font_style="bold"))

        return hv.Overlay(items) if items else hv.Points([]).opts(alpha=0)

    def _apply_pick(self, scene, x: float, y: float):
        """Turn one click into either a transect point or a profile.

        The plan view is in degrees now, so a click is a lon/lat: it is
        converted back to the crop's pixel frame, which is the frame the
        axis path and the profile locations live in.
        """
        lon, lat, _ = self._plan_coords(scene)
        i = int(np.argmin(np.abs(lon - x)))
        j = int(np.argmin(np.abs(lat - y)))

        if self.state.pick_mode == "perpendicular":
            path = np.asarray(scene.axis_path)
            d2 = (path[:, 1] - i) ** 2 + (path[:, 0] - j) ** 2
            idx = int(np.argmin(d2))
            self.state.perp_index = idx
            # The slider is the readout as well as a control, so it has to
            # follow a click -- not updating was the visible symptom that
            # the click was not arriving at all.
            self.w_axis.value = min(idx, self.w_axis.end)
        elif not self.state.add_profile_point(j, i):
            self._plan_status.object = (
                f"**{self.state.MAX_PROFILES} locations already** — "
                "press *Clear points* to start over")
            return
        self._plan_status.object = self._plan_note(scene)

    def _refresh_plan_markers(self):
        """Redraw the markers only -- the plot, and its zoom, stay put."""
        self._plan_marks.event(
            tick=self._plan_marks.contents.get("tick", 0) + 1)
        scene = self._first_scene()
        if scene is not None:
            self._plan_status.object = self._plan_note(scene)

    def build_plan(self):
        """Figure (c): isopycnal depth and the plan view, from density.

        No colour field is involved -- at this point in the flow none has
        been chosen.  The front geometry comes from density anyway, so the
        axis this draws is the same axis every section will use.
        """
        if not self.state.front_label:
            self._plan_status.object = "choose a front label first"
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._build_plan()
            return
        self._plan_status.object = "building the plan view…"
        pn.state.execute(
            lambda: asyncio.ensure_future(self._build_plan_async()))

    async def _build_plan_async(self):
        await asyncio.to_thread(self._build_plan)

    def _build_plan(self):
        s = self.state
        self._token += 1
        token = self._token
        try:
            scene = pipeline.build_scene(
                s.provider, s.date, s.tile_index(),
                config.TILE_GEOMETRY_FIELD, s.front_label, region=s.region)
        except Exception as exc:                            # noqa: BLE001
            self._planview.object = None
            self._plan_status.object = (
                f"**Plan view unavailable** — {type(exc).__name__}: {exc}")
            return

        self._density_scene = scene
        self._build_isodepth(scene, token)
        self._build_regionmap(scene, token)
        self.draw_planview(scene)
        self.w_plan.button_type = "default"

    def build_sections(self):
        """Figure (d) onward: one column per selected field.

        Everything here depends on the axis point and the profile
        locations picked on the plan view, which is why it is a separate
        button from the one that builds the plan view itself.
        """
        self.schedule_figures()

    def _plan_note(self, scene) -> str:
        s = self.state
        idx = self._resolved_perp_index(scene)
        how = "chosen" if s.perp_index >= 0 else "auto (field extremum)"
        return (f"transect at axis vertex **{idx}** of "
                f"{len(scene.axis_path) - 1} — {how}  ·  "
                f"**{len(s.profile_points)}**/{s.MAX_PROFILES} profile "
                f"locations  ·  offsets: "
                f"<span style='color:{F.OFFSET_PLUS}'>+ side</span> / "
                f"<span style='color:{F.OFFSET_MINUS}'>− side</span>")

    def _first_scene(self):
        """The scene the plan view is drawn from."""
        if self._density_scene is not None:
            return self._density_scene
        for field in self.state.fields:
            if field in self._scenes:
                return self._scenes[field]
        return next(iter(self._scenes.values()), None)

    def _resolved_perp_index(self, scene):
        """The axis vertex to cut at: the user's, or the auto pick.

        ``perp_index`` is -1 until the plan view is clicked, which keeps
        the old behaviour as the default rather than making the page
        useless before anything has been picked.
        """
        n = len(scene.axis_path)
        chosen = int(self.state.perp_index)
        if 0 <= chosen < n:
            return chosen
        return F.pick_perp_index(scene,
                                 half_width=self.state.perp_half_width)

    def _build_isodepth(self, scene, token):
        """The isopycnal-depth map, once per build, over the whole tile."""
        s = self.state
        sigma = s.sigma or F.default_sigma(scene)
        try:
            path = F.figure_isopycnal_depth(scene, sigma,
                                            **self._tile_context())
        except Exception as exc:                            # noqa: BLE001
            if token == self._token:
                self._isodepth.object = None
                self._iso_status.object = f"**Unavailable:** {exc}"
            return
        if token != self._token:
            return

        self._isodepth.object = str(path)
        note = ("" if s.sigma else
                "  ·  sigma defaulted to the volume's median; type a value "
                "to change it")
        self._iso_status.object = f"sigma = **{sigma:.2f}** kg/m^3{note}"

    def _build_vtk(self, field, scene):
        pane = self._panes.get((field, "vtk"))
        if pane is None:
            return
        try:
            plotter = F.build_3d(scene)
            pane.objects = [pn.pane.VTK(plotter.ren_win,
                                        sizing_mode="stretch_width",
                                        min_height=320,
                                        enable_keybindings=False)]
        except Exception as exc:                            # noqa: BLE001
            pane.objects = [widgets.error_card(
                exc, "3-D unavailable (2-D figures unaffected)")]
        finally:
            pane.loading = False

    # -- layout ----------------------------------------------------------

    def view(self):
        # One row per stage, each ending in the button that runs it, so
        # which control belongs to which step is visible rather than
        # remembered.
        # (a) Overview -> Build tile
        row_a = pn.Row(self.w_date, self.w_region, self.w_fronts,
                       sizing_mode="stretch_width", margin=(0, 10))
        row_a_go = pn.Row(pn.pane.Markdown("**a · overview → tile**",
                                           margin=(8, 5, 0, 10)),
                          self.w_loadtile,
                          sizing_mode="stretch_width", margin=(0, 10))

        # (b) Tile -> front label + sigma -> figure (c)
        row_b = pn.Row(self.w_avail, self.w_label, self.w_sigma,
                       self.w_regionfield,
                       sizing_mode="stretch_width", margin=(0, 10))
        row_b_go = pn.Row(pn.pane.Markdown("**b · front + sigma → plan view**",
                                           margin=(8, 5, 0, 10)),
                          self.w_plan,
                          sizing_mode="stretch_width", margin=(0, 10))

        # (c) Plan view -> picks + fields -> the sections
        row_c = pn.Row(self.w_pickmode, self.w_axis, self.w_clearpts,
                       self.w_insetdepth,
                       sizing_mode="stretch_width", margin=(0, 10))
        row_c2 = pn.Row(self.w_fields, self.w_offsets, self.w_perp,
                        sizing_mode="stretch_width", margin=(0, 10))
        row_c_go = pn.Row(pn.pane.Markdown("**c · plan view → sections**",
                                           margin=(8, 5, 0, 10)),
                          self.w_sections,
                          sizing_mode="stretch_width", margin=(0, 10))

        maps = pn.Column(
            pn.pane.Markdown("**Overview**", margin=(0, 10)),
            self._overview,
            pn.pane.Markdown("**Tile**", margin=(8, 10, 0, 10)),
            self._tilemap,
            self._hover,
            sizing_mode="stretch_width",
        )

        planview = pn.Column(
            pn.layout.Divider(),
            pn.pane.Markdown("### Plan view — pick the transect and profiles",
                             margin=(6, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>The front's crop with its main axis, ticked in "
                "kilometres from <b>start</b>. Clicking sets whichever of "
                "the two the mode says: the <b>perpendicular</b> transect "
                "(snapped onto the axis) or a <b>profile</b> location. Then "
                "press <i>Build sections</i> — it reuses the tiles already "
                "loaded, so it costs figures only.</small>",
                margin=(0, 10)),
            self._plan_status,
            self._planview,
            sizing_mode="stretch_width",
        )

        isodepth = pn.Column(
            pn.layout.Divider(),
            pn.pane.Markdown("### Isopycnal depth", margin=(6, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>Depth of one density surface over the front's crop. "
                "Gray is where it does not exist in the column — it has "
                "outcropped, or lies below the model floor. Density alone "
                "decides this, so there is one map however many fields "
                "are selected.</small>", margin=(0, 10)),
            self._iso_status,
            self._isodepth,
            pn.pane.Markdown("### Region map of a chosen field",
                             margin=(6, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>The same frame as the isopycnal map, coloured by a "
                "field instead of by depth. Fronts in cyan, the selected "
                "one in red.</small>", margin=(0, 10)),
            self._region_status,
            self._regionmap,
            sizing_mode="stretch_width",
        )

        figures = pn.Column(
            pn.layout.Divider(),
            pn.pane.Markdown("### Figures for the selected front",
                             margin=(6, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>One column per field. The front geometry — crop, "
                "main axis, transect point — is computed once and shared, "
                "so the columns differ only in the colour field.</small>",
                margin=(0, 10)),
            self._build_status,
            self._columns,
            sizing_mode="stretch_width",
        )

        body = pn.Column(
            pn.pane.Markdown("### Tiles — one front in 3-D and cross-section",
                             margin=(4, 10, 0, 10)),
            row_a, row_a_go, self._status, maps,
            row_b, row_b_go, isodepth, planview,
            row_c, row_c2, row_c_go, figures,
            sizing_mode="stretch_width",
        )

        notes = [n for n in (widgets.banner(self.state.provider),
                             widgets.degraded_notice()) if n]
        notes.append(pn.pane.Alert(
            "Only the timestamps with full 3-D raw data are offered here.",
            alert_type="secondary", margin=(0, 10, 8, 10)))
        return pn.Column(*notes, body, sizing_mode="stretch_width")


def page(provider=None):
    """Entry point used by ``serve.py``."""
    return TilesPage(provider=provider).view()
