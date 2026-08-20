"""Field characteristics — the shared assembly for the Surface and Depth pages.

The two pages are the same page. They differ only in how a field name
becomes a channel name in the store:

    surface:  field                -> "relative_vorticity"
    depth:    field + depth level  -> "relative_vorticity_mld"

So the layout, the panels and the interactions live here once, and
``surface.py`` / ``depth.py`` supply a mode.

Three sections, top to bottom:

1. **map + distributions** — a Pacific-centred map with box-select, and six
   panels describing the grid cells in the box (all points / fronts only);
2. **front properties** — six panels describing the *fronts* in the box,
   from the geometry and colocation tables.

Statistics are exact and computed on the native grid, so a large box costs
real time. It runs off the event loop with the panels marked loading, and
every result is cached.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import holoviews as hv
import panel as pn

from fronts.viz.apps import config
from fronts.viz.apps.characteristics import front_props as FP
from fronts.viz.apps.characteristics import panels as P
from fronts.viz.apps.characteristics import stats
from fronts.viz.apps.common import basemap, widgets
from fronts.viz.apps.common.selection import BBox
from fronts.viz.apps.common.state import CharacteristicsState

hv.extension("bokeh")

ROW_TITLES = ("PDF", "Joint PDF", "Conditional JPDF")

FP_TITLES = (
    "(a) front length",
    "(b) front orientation",
    "(c) latitude × length",
    "(d) latitude × orientation",
    "(e) field × length",
    "(f) field × orientation",
)

#: Fraction of the box added as padding when the map zooms to a selection.
ZOOM_PAD = 0.12


@dataclass(frozen=True)
class Mode:
    """What distinguishes the Surface page from the Depth page."""

    key: str
    title: str
    #: Whether the page offers a depth-level selector.
    has_depth: bool
    #: Explanation shown when the date list is restricted.
    date_note: str = ""


SURFACE = Mode(
    key="surface",
    title="Field Characteristics at the Surface",
    has_depth=False,
)

DEPTH = Mode(
    key="depth",
    title="Field Characteristics at Depth",
    has_depth=True,
    date_note=(
        "Only the timestamps with full 3-D data carry depth-resolved "
        "fields, so this page offers those."
    ),
)


class CharacteristicsPage:
    """Assembles a characteristics page and keeps its panes in sync."""

    def __init__(self, mode: Mode = SURFACE, provider=None):
        self.mode = mode
        self.state = CharacteristicsState(provider=provider,
                                          depth_mode=mode.has_depth)

        self._map = pn.pane.HoloViews(sizing_mode="stretch_width",
                                      min_height=560)
        self._bounds = hv.streams.BoundsXY(bounds=None)

        self._panes = {
            (col, row): pn.pane.Matplotlib(
                tight=True, format="png", dpi=120,
                width=640, height=470, margin=(0, 6),
            )
            for col in ("all", "fronts")
            for row in range(3)
        }
        self._fp_panes = [
            pn.pane.Matplotlib(tight=True, format="png", dpi=120,
                               width=470, height=350, margin=(0, 6))
            for _ in range(6)
        ]

        self._status = widgets.status()
        self._fp_status = widgets.status()
        self._token = 0
        self._fp_token = 0

        self._build_controls()
        # The map is cheap and orients the user, so it is drawn up front;
        # the panels wait for Rebuild.
        self.redraw_map()

    # -- channel resolution ----------------------------------------------

    def resolve(self, field: str) -> str:
        """Base field name -> channel name in the store."""
        if not self.mode.has_depth:
            return field
        return self.state.provider.channel(field, self.state.depth_level)

    def _tag(self) -> str:
        return self.state.depth_level if self.mode.has_depth else ""

    # -- controls --------------------------------------------------------

    def _build_controls(self):
        s = self.state
        self.w_date = pn.widgets.Select.from_param(s.param.date, width=185)
        self.w_field = pn.widgets.Select.from_param(s.param.field, width=185)
        self.w_fronts = pn.widgets.Checkbox.from_param(
            s.param.show_fronts, name="Show fronts")
        self.w_reset = pn.widgets.Button(name="Reset region",
                                         button_type="default", width=140)
        self.w_reset.on_click(lambda _: self._reset_region())

        self.w_depth = (
            pn.widgets.Select.from_param(s.param.depth_level, width=195)
            if self.mode.has_depth else None
        )
        self.w_stat = pn.widgets.Select.from_param(s.param.front_stat,
                                                   width=140)

        self.w_build = pn.widgets.Button(name="Rebuild", width=150,
                                         button_type="primary")
        self.w_build.on_click(lambda _: self.rebuild())

        # Nothing rebuilds on its own.
        s.param.watch(lambda *_: self._reflect_dirty(), ["dirty"])
        self._reflect_dirty()

    def _reset_region(self):
        self.state.reset_region()

    def rebuild(self):
        """Build everything for the current selection.  The only entry."""
        self.redraw_map()
        self.schedule_stats()
        self.schedule_front_props()
        self.state.dirty = False

    def _reflect_dirty(self):
        if self.state.dirty:
            self.w_build.button_type = "primary"
            self._status.object = (
                "⟳ **settings changed** — press *Rebuild*"
                f"  ·  region: {self.state.box.label()}")
        else:
            self.w_build.button_type = "default"

    def _on_bounds(self, bounds):
        """Record the box.  The map and panels follow on *Rebuild*."""
        if not bounds:
            return
        self.state.set_bounds(bounds)

    # -- map -------------------------------------------------------------

    def _zoom_limits(self):
        """``(xlim, ylim)`` for the current selection, in 0..360 map coords.

        Returns the full globe when nothing is selected.
        """
        box = self.state.box
        if box.is_global:
            return (0, 360), config.PYRAMID_LAT_RANGE

        lon0, lon1 = box.lon0 % 360.0, box.lon1 % 360.0
        if lon1 <= lon0:                      # selection crosses the seam
            lon1 += 360.0

        dx = max((lon1 - lon0) * ZOOM_PAD, 1.0)
        dy = max((box.lat1 - box.lat0) * ZOOM_PAD, 1.0)
        return ((lon0 - dx, lon1 + dx),
                (max(box.lat0 - dy, -90.0), min(box.lat1 + dy, 90.0)))

    def _selection_outline(self):
        """The selected box drawn on the map, so it survives the zoom."""
        box = self.state.box
        if box.is_global:
            return None
        lon0, lon1 = box.lon0 % 360.0, box.lon1 % 360.0
        if lon1 <= lon0:
            lon1 += 360.0
        return hv.Rectangles([(lon0, box.lat0, lon1, box.lat1)]).opts(
            fill_alpha=0.0, line_color="#00e5ff", line_width=2,
            line_dash="dashed",
        )

    def redraw_map(self):
        s = self.state
        extent = self._zoom_limits()
        try:
            channel = self.resolve(s.field)
            overlay = basemap.global_map(
                s.provider, s.date, channel,
                show_fronts=s.show_fronts,
                title=f"{channel}  —  {s.date}",
                extent=extent,
            )
        except Exception as exc:                        # noqa: BLE001
            self._map.object = None
            self._status.object = f"**Map unavailable:** {exc}"
            return

        outline = self._selection_outline()
        if outline is not None:
            overlay = overlay * outline

        xlim, ylim = extent
        overlay = overlay.opts(hv.opts.Overlay(xlim=xlim, ylim=ylim))

        # The stream must be attached before the pane renders, or the
        # box-select tool has nothing to report to.
        box_stream = hv.streams.BoundsXY(source=overlay, bounds=None)
        box_stream.add_subscriber(self._on_bounds)
        self._bounds = box_stream
        self._map.object = overlay

    # -- distributions ---------------------------------------------------

    def schedule_stats(self):
        self._token += 1
        token = self._token
        for pane in self._panes.values():
            pane.loading = True
        self._status.object = (
            f"region: **{self.state.box.label()}** — computing…")
        _run(lambda: self._compute(token))

    def _compute(self, token):
        s = self.state
        try:
            columns = stats.extract_both(s.provider, s.date, s.field, s.box,
                                         resolve=self.resolve, tag=self._tag())
            bins = P.pdf_bins(columns, s.field)
        except Exception as exc:                        # noqa: BLE001
            if token == self._token:
                self._fail(str(exc))
            return
        if token != self._token:
            return

        for col, samples in columns.items():
            head = "all points" if col == "all" else "fronts only"
            if samples.unavailable:
                figs = (P._blank(samples.unavailable), P._blank("—"),
                        P._blank("—"))
            else:
                figs = (
                    P.figure_pdf(samples, s.field, bins, title=head),
                    P.figure_jpdf(samples, title=head),
                    P.figure_jpdf_conditional(samples, s.field, title=head),
                )
            for row, fig in enumerate(figs):
                pane = self._panes[(col, row)]
                pane.object = fig
                pane.loading = False

        cells = columns["all"].n_cells
        fronts = columns["fronts"]
        on_fronts = ("fronts pending" if fronts.unavailable
                     else f"{fronts.n:,} on fronts")
        self._status.object = (
            f"region: **{s.box.label()}** — {cells:,} grid cells, "
            f"{columns['all'].n:,} samples, {on_fronts}"
            "  ·  exact, full resolution"
        )

    def _fail(self, message):
        for (col, row), pane in self._panes.items():
            pane.object = P._blank(message if row == 0 else "—")
            pane.loading = False
        self._status.object = f"**Statistics unavailable:** {message}"

    # -- front properties ------------------------------------------------

    def schedule_front_props(self):
        self._fp_token += 1
        token = self._fp_token
        for pane in self._fp_panes:
            pane.loading = True
        self._fp_status.object = "front properties — computing…"
        _run(lambda: self._compute_front_props(token))

    def _compute_front_props(self, token):
        s = self.state
        try:
            table = FP.merged_table(s.provider, s.date)
            region = FP.in_region(table, s.box)
            self.state.refresh_front_stats(table)
        except Exception as exc:                        # noqa: BLE001
            if token == self._fp_token:
                for pane in self._fp_panes:
                    pane.object = FP._blank(str(exc))
                    pane.loading = False
                self._fp_status.object = (
                    f"**Front properties unavailable:** {exc}")
            return
        if token != self._fp_token:
            return

        stat = s.front_stat
        figs = (
            FP.figure_length_pdf(region, title=""),
            FP.figure_orientation_pdf(region, title=""),
            FP.figure_lat_vs(region, "length"),
            FP.figure_lat_vs(region, "orientation"),
            FP.figure_field_vs(region, s.field, stat, "length"),
            FP.figure_field_vs(region, s.field, stat, "orientation"),
        )
        for pane, fig in zip(self._fp_panes, figs):
            pane.object = fig
            pane.loading = False

        self._fp_status.object = (
            f"**{len(region):,}** fronts with centroid in "
            f"{s.box.label()} (of {len(table):,} global) · "
            f"panels (e)/(f) use `{s.field}_{stat}`"
        )

    # -- layout ----------------------------------------------------------

    def _stats_grid(self):
        cells = [
            pn.pane.Markdown("**All grid points**", margin=(0, 5)),
            pn.pane.Markdown("**Fronts only**", margin=(0, 5)),
        ]
        for row in range(3):
            cells.append(pn.Column(
                pn.pane.Markdown(
                    f"<small>({'abc'[row]}) {ROW_TITLES[row]}</small>",
                    margin=(2, 5, 0, 5)),
                self._panes[("all", row)], margin=0))
            cells.append(pn.Column(
                pn.pane.Markdown("<small>&nbsp;</small>", margin=(2, 5, 0, 5)),
                self._panes[("fronts", row)], margin=0))
        return pn.GridBox(*cells, ncols=2)

    def _front_props_section(self):
        cells = []
        for title, pane in zip(FP_TITLES, self._fp_panes):
            cells.append(pn.Column(
                pn.pane.Markdown(f"<small>{title}</small>",
                                 margin=(2, 5, 0, 5)),
                pane, margin=0))
        return pn.Column(
            pn.layout.Divider(),
            pn.pane.Markdown("### Front properties in this region",
                             margin=(6, 10, 0, 10)),
            pn.Row(
                pn.pane.Markdown(
                    "<small>One row per labelled front, from the geometry "
                    "and colocation tables. A front belongs to the region "
                    "its centroid falls in.</small>",
                    margin=(0, 10), width=560),
                self.w_stat,
                margin=(0, 0),
            ),
            self._fp_status,
            pn.GridBox(*cells, ncols=3, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def view(self):
        controls = [self.w_date]
        if self.w_depth is not None:
            controls.append(self.w_depth)
        controls += [self.w_field, self.w_fronts, self.w_reset,
                     self.w_build]

        top = pn.Column(
            pn.pane.Markdown(f"### {self.mode.title}", margin=(4, 10, 0, 10)),
            pn.Row(*controls, sizing_mode="stretch_width", margin=(0, 10)),
            self._status,
            self._map,
            pn.pane.Markdown(
                "<small>Drag a box on the map to choose a region — the map "
                "zooms to it and the selection stays outlined. Statistics "
                "are exact, at full resolution, on the native grid; the map "
                "is drawn from a regridded display pyramid.</small>",
                margin=(0, 10)),
            sizing_mode="stretch_width",
        )

        # Map above, distributions below, each across the full width:
        # side by side, both were squeezed into half a page for no reason
        # -- they are read one after the other, not compared.
        body = pn.Column(top, pn.layout.Divider(), self._stats_grid(),
                         self._front_props_section(),
                         sizing_mode="stretch_width")

        notes = [n for n in (widgets.banner(self.state.provider),
                             widgets.degraded_notice()) if n]
        if self.mode.date_note:
            notes.append(pn.pane.Alert(self.mode.date_note,
                                       alert_type="secondary",
                                       margin=(0, 10, 8, 10)))
        return pn.Column(*notes, body, sizing_mode="stretch_width")


def _run(fn):
    """Run off the event loop when there is one, inline when there is not."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        fn()
        return

    async def _go():
        await asyncio.to_thread(fn)

    pn.state.execute(lambda: asyncio.ensure_future(_go()))


def page(mode: Mode = SURFACE, provider=None):
    return CharacteristicsPage(mode=mode, provider=provider).view()
