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

ROW_KEYS = ("vtk",) + F.FIGURE_ORDER
ROW_LABELS = {
    "vtk": "(a) 3-D field on the front's isopycnals",
    **F.FIGURE_TITLES,
}


class TilesPage:
    """Assembles the tiles page."""

    def __init__(self, provider=None):
        self.state = TilesState(provider=provider)

        self._overview = pn.pane.HoloViews(min_height=330,
                                           sizing_mode="stretch_width")
        self._tilemap = pn.pane.HoloViews(min_height=430,
                                          sizing_mode="stretch_width")
        self._status = widgets.status()
        self._hover = widgets.status("hover a front to read its label")
        self._build_status = widgets.status()

        #: One column of panes per field, built on demand.
        self._columns = pn.Row(sizing_mode="stretch_width")
        self._panes: dict[tuple[str, str], pn.viewable.Viewable] = {}

        self._labels_tile = None
        self._token = 0

        self._build_controls()
        self.draw_overview()
        self.draw_tile()
        self._render_columns()

    # -- controls --------------------------------------------------------

    def _build_controls(self):
        s = self.state
        self.w_date = pn.widgets.Select.from_param(s.param.date, width=175)
        self.w_region = pn.widgets.Select.from_param(s.param.region, width=225)
        self.w_fields = pn.widgets.MultiChoice.from_param(
            s.param.fields, name=f"Fields (max {s.MAX_FIELDS})",
            max_items=s.MAX_FIELDS, width=340,
        )
        self.w_mapfield = pn.widgets.Select.from_param(
            s.param.field, name="Field on map", width=160)
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

        self.w_regen = pn.widgets.Button(
            name="Regenerate figures", button_type="primary", width=185)
        self.w_regen.on_click(lambda _: self.schedule_figures())

        s.param.watch(lambda *_: self.draw_tile(),
                      ["region", "field", "show_fronts", "date"])
        s.param.watch(lambda *_: self._render_columns(), ["fields"])
        s.param.watch(lambda *_: self._reflect_dirty(), ["dirty"])
        self._reflect_dirty()

    def _pick_from_list(self, event):
        if event.new:
            self.state.select_front(int(event.new))

    def _reflect_dirty(self):
        """Show whether what is on screen matches the current settings."""
        if self.state.dirty:
            self.w_regen.button_type = "primary"
            self._build_status.object = (
                "⟳ **settings changed** — press *Regenerate figures*"
                f"{self._cost_note()}")
        else:
            self.w_regen.button_type = "default"

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
                width=760, height=300, title="Click a region",
                tools=("tap",), active_tools=("tap",),
            )
        except Exception as exc:                            # noqa: BLE001
            self._overview.object = None
            self._status.object = f"**Overview unavailable:** {exc}"
            return

        boxes, labels = [], []
        for r in regions_mod.REGIONS:
            lon = r.lon % 360.0
            boxes.append((lon - BOX_HALF[0], r.lat - BOX_HALF[1],
                          lon + BOX_HALF[0], r.lat + BOX_HALF[1]))
            labels.append((lon, r.lat + BOX_HALF[1] + 3.0, r.name))

        sel = s.region_obj
        overlay = (
            base
            * hv.Rectangles(boxes).opts(fill_alpha=0.0, line_color="red",
                                        line_width=2)
            * hv.Labels(labels, vdims="text").opts(
                text_color="red", text_font_size="7pt", text_align="center")
            * hv.Points([(sel.lon % 360.0, sel.lat)]).opts(
                color="red", size=11, marker="s", fill_alpha=0.25)
        )
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

    def draw_tile(self):
        s = self.state
        try:
            tile_idx = s.tile_index()
            ds = s.provider.tile(s.date, tile_idx, s.field)
            var = ds.attrs.get("tile_var_name") or pipeline._sole_3d(ds)
            # Remap to the rect frame first, so the surface and the labels
            # share one orientation -- the convention fronts_viz_curtain
            # and fronts_viz_3d already use.
            lookup = pipeline.tile_lookup(ds, synthetic=s.provider.synthetic)
            surface = pipeline.remap_to_rect(
                np.asarray(ds[var].values), lookup)[0]
        except Exception as exc:                            # noqa: BLE001
            self._tilemap.object = None
            self._status.object = f"**Tile unavailable:** {exc}"
            self.w_avail.options = []
            return

        # A tile without fronts is still worth drawing -- the label map is
        # a separate product and may not have been built for this date.
        try:
            labels = pipeline.tile_labels(s.provider, s.date, tile_idx,
                                          surface.shape, ds=ds)
        except Exception as exc:                            # noqa: BLE001
            labels = np.zeros(surface.shape, dtype=np.int32)
            self._status.object = f"*Fronts not overlaid:* {exc}"

        self._labels_tile = labels
        available = pipeline.available_fronts(labels)
        self.w_avail.options = [str(l) for l in available]

        img = hv.Image(
            (np.arange(surface.shape[1]), np.arange(surface.shape[0]), surface),
            kdims=["i", "j"], vdims=[s.field],
        ).opts(cmap="viridis", colorbar=True, tools=["hover"])

        layers = [img]
        if s.show_fronts:
            shown = np.where(labels > 0, (labels - 1) % len(FRONT_PALETTE),
                             np.nan).astype(float)
            layers.append(hv.Image(
                (np.arange(labels.shape[1]), np.arange(labels.shape[0]), shown),
                kdims=["i", "j"], vdims=["front"],
            ).opts(cmap=list(FRONT_PALETTE), colorbar=False, tools=["hover"],
                   clim=(0, len(FRONT_PALETTE) - 1)))

        overlay = hv.Overlay(layers).opts(hv.opts.Overlay(
            width=560, height=420, active_tools=["tap"],
            title=f"{s.region}  —  tile {tile_idx}  —  {s.field}",
            xlabel="i (tile pixels)", ylabel="j (tile pixels)"))

        hv.streams.Tap(source=overlay).add_subscriber(self._on_tile_tap)
        hv.streams.PointerXY(source=overlay).add_subscriber(self._on_pointer)

        self._tilemap.object = overlay
        self._status.object = (
            f"tile **{tile_idx}** · {len(available)} fronts with 25+ pixels"
            if available else
            f"tile **{tile_idx}** · no fronts with 25+ pixels here")

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

        # The geometry is a property of the front, not of the colour
        # field, so it is resolved from the first column and the crop /
        # axis / perpendicular point are reused by the rest.
        shared_index = None

        for field in fields:
            if token != self._token:
                return
            try:
                scene = pipeline.build_scene(s.provider, s.date,
                                             s.tile_index(), field,
                                             s.front_label)
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
                    pane.loading = False
                continue

            if shared_index is None:
                shared_index = F.pick_perp_index(
                    scene, half_width=s.perp_half_width)
            idx = shared_index

            builders = {
                "inset": lambda sc=scene, i=idx: F.figure_inset(
                    sc, perp_index=i, half_width=s.perp_half_width),
                "isopycnal": lambda sc=scene, i=idx: F.figure_isopycnal(
                    sc, perp_index=i),
                "mainaxis": lambda sc=scene, i=idx: F.figure_mainaxis(
                    sc, perp_index=i),
                "offsets": lambda sc=scene: F.figure_offsets(
                    sc, n_offsets=s.n_offsets),
                "perpendicular": lambda sc=scene, i=idx: F.figure_perpendicular(
                    sc, index=i, half_width=s.perp_half_width),
            }
            for key, build in builders.items():
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
            done.append(field)

        if token != self._token:
            return

        s.dirty = False
        parts = [f"built **{len(done)}** column{'s' if len(done) != 1 else ''} "
                 f"for front **{s.front_label}**"]
        if shared_index is not None:
            parts.append(f"shared transect at column {shared_index}")
        if failed:
            parts.append(f"{len(failed)} failed")
        self._build_status.object = " · ".join(parts)
        self.w_regen.button_type = "default"

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
        row1 = pn.Row(self.w_date, self.w_region, self.w_mapfield,
                      self.w_fronts, sizing_mode="stretch_width",
                      margin=(0, 10))
        row2 = pn.Row(self.w_fields, self.w_avail, self.w_label,
                      self.w_offsets, self.w_perp, self.w_regen,
                      sizing_mode="stretch_width", margin=(0, 10))

        maps = pn.Row(
            pn.Column(pn.pane.Markdown("**Overview**", margin=(0, 10)),
                      self._overview, width=790),
            pn.Column(pn.pane.Markdown("**Tile**", margin=(0, 10)),
                      self._tilemap, self._hover),
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
            row1, row2, self._status, maps, figures,
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
