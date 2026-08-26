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
    #: Whether the map waits for a button instead of following the
    #: controls.  Surface draws straight away -- its pyramids are built
    #: and a redraw is a cached read.  A depth channel may have no pyramid
    #: yet, and building one regrids a 0.9 GB plane, so on Depth changing
    #: the field is a request, not a glance.
    manual_map: bool = False


SURFACE = Mode(
    key="surface",
    title="Field Characteristics at the Surface",
    has_depth=False,
)

DEPTH = Mode(
    key="depth",
    title="Field Characteristics at Depth",
    has_depth=True,
    manual_map=True,
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
                                      min_height=760)
        #: The region the panels below actually describe.  A static
        #: figure, built on *Rebuild* -- unlike the map above, which
        #: follows navigation -- so what you are looking at and what the
        #: numbers describe cannot drift apart, and nothing here is paid
        #: for on the interactive path.
        self._regionmap = pn.pane.Matplotlib(
            tight=False, format="png", dpi=110,
            sizing_mode="stretch_width", min_height=620)
        #: One box-select stream for the life of the page.  Recreating it
        #: per redraw is what lost the selection -- see ``redraw_map``.
        #: ``(overlay, extent)`` of the last successfully drawn map, so a
        #: box-select can recompose without re-fetching and a failed
        #: redraw does not replace a working map with an empty one.
        self._map_base = None
        #: Set by ``redraw_map``: the next frame must refetch.  Separate
        #: from the base so a failed refetch still has something to show.
        self._map_stale = True
        self._bounds = hv.streams.BoundsXY(bounds=None)
        self._last_bounds = None
        # Record the box here, not inside the frame callback.  Recording it
        # while drawing tied "the region changed" to "the map re-rendered":
        # true in the browser, but it made the state depend on a render
        # that may be skipped or deferred.  A subscriber fires as soon as
        # the tool reports, and runs before the plot's own refresh, so the
        # frame that follows already sees the new box.
        self._bounds.add_subscriber(self._record_bounds)
        # Zooming needs a *new* plot (see ``redraw_map``), so it must not
        # happen while the current one is mid-refresh.  Plot refreshes are
        # subscribed at precedence 1.1; sitting above that means the frame
        # already being drawn finishes first and is then replaced.
        self._bounds.add_subscriber(self._zoom_to_bounds, precedence=2.0)

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
        if self.mode.manual_map:
            # Nothing read until asked.  A depth channel may have no
            # pyramid yet, and building one regrids a 0.9 GB plane -- not
            # something to do because a page was opened.
            self._status.object = (
                "pick a date, depth level and field, then press "
                "*Rebuild map*")
        else:
            # Cheap and orienting, so it is drawn up front; the panels
            # wait for Rebuild.
            self.redraw_map()

    # -- channel resolution ----------------------------------------------

    def resolve(self, field: str) -> str:
        """Base field name -> channel name in the store.

        Depth mode asks the store rather than assuming the suffix: some
        channels are bare at every level (the wind, the mixed-layer
        quantities), and a depth-resolved one may not have been built at
        every suffix.  Surface mode returns the name unchanged, exactly as
        before.
        """
        if not self.mode.has_depth:
            return field
        return self.state.provider.channel_in(
            self.state.date, field, self.state.depth_level)

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

        self.w_build = pn.widgets.Button(
            label="Rebuild map" if self.mode.manual_map else "Rebuild",
            width=170, button_type="primary")
        self.w_build.on_click(lambda _: self.rebuild())

        #: Only where the map is manual: there the two halves are separate
        #: requests, so they need separate buttons.  With one button the
        #: label would have to mean "draw the map" before a region is
        #: chosen and "compute the figures" after it.
        self.w_stats = pn.widgets.Button(
            label="Run statistics for region", width=230,
            button_type="primary")
        self.w_stats.on_click(lambda _: self.run_statistics())

        map_triggers = ["date", "field", "show_fronts"]
        if self.mode.has_depth:
            map_triggers.append("depth_level")
        if self.mode.manual_map:
            # Mark the map stale rather than redrawing it.
            s.param.watch(lambda *_: self._reflect_dirty(), map_triggers)
        else:
            # The map keeps up with the selection -- a cached pyramid
            # level, so redrawing is cheap.  The panels are what wait.
            s.param.watch(lambda *_: self.redraw_map(), map_triggers)

        s.param.watch(lambda *_: self._reflect_dirty(), ["dirty"])
        self._reflect_dirty()

    def _reset_region(self):
        """Clear the box and un-zoom the map.  The panels stay stale.

        Choosing a region is navigation, not computation: the map has to
        keep up with the box or you cannot see what you selected.  Only
        the figures below it cost real time, so only they wait.
        """
        self.state.reset_region()
        self._last_bounds = None
        self._bounds.event(bounds=None)
        self.redraw_map()

    def rebuild(self):
        """What the primary button does, which differs by mode.

        Where the map follows the controls it is already current, so this
        is the panels -- the expensive half.  Where the map is manual this
        draws the map and stops: the region has not been chosen yet, so
        there is nothing to compute statistics over.
        """
        # Look again.  The channel listing is cached per date, so a
        # subset written *after* the page first read it stays invisible
        # for the life of the process -- which is exactly what happens
        # while a build is still running.  Rebuild means "go and look".
        self.state.provider.refresh()
        self.state.refresh_fields()
        self.state.refresh_depth_levels()

        self.redraw_map()
        if self.mode.manual_map:
            self._status.object = (
                f"**{self.resolve(self.state.field)}** at "
                f"**{self.state.date}** — draw a box, then press "
                "*Run statistics for region*")
            return

        self.run_statistics()

    def run_statistics(self):
        """The figures below the map, for the box that is selected."""
        self.schedule_stats()
        self.schedule_front_props()
        self.state.dirty = False

    def _reflect_dirty(self):
        if self.state.dirty:
            self.w_build.button_type = "primary"
            self._status.object = (
                f"region: **{self.state.box.label()}** — "
                "⟳ press *Rebuild* to compute the figures below")
        else:
            self.w_build.button_type = "default"

    def _record_bounds(self, bounds=None):
        """A box was drawn: make it the selection.

        Called by the stream, so it happens whether or not the map goes on
        to re-render.
        """
        if not bounds or bounds == self._last_bounds:
            return
        self._last_bounds = bounds
        self.state.set_bounds(bounds)
        self._reflect_dirty()

    def _zoom_to_bounds(self, bounds=None):
        """Put the axes on the box.  Cheap: no data is read."""
        if not bounds:
            return
        self.redraw_map(refetch=False)

    def _on_bounds(self, bounds):
        """Apply a box as if it had been drawn on the map.

        The programmatic entry point; the tool itself goes through the
        stream, which reaches the same place.
        """
        if not bounds:
            return
        self._bounds.event(bounds=bounds)

    # -- map -------------------------------------------------------------

    def _zoom_limits(self, pad: float = ZOOM_PAD):
        """``(xlim, ylim)`` for the current selection, in 0..360 map coords.

        Returns the full globe when nothing is selected.  *pad* of 0 gives
        the box exactly, which is what the region map below wants -- the
        padding above exists so the selection outline is not flush against
        the frame while you are still navigating.
        """
        box = self.state.box
        if box.is_global:
            return (0, 360), config.PYRAMID_LAT_RANGE

        lon0, lon1 = box.lon0 % 360.0, box.lon1 % 360.0
        if lon1 <= lon0:                      # selection crosses the seam
            lon1 += 360.0

        dx = max((lon1 - lon0) * pad, 1.0) if pad else 0.0
        dy = max((box.lat1 - box.lat0) * pad, 1.0) if pad else 0.0
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

    def redraw_map(self, refetch: bool = True):
        """Draw the map again, as a new plot the box-select still reaches.

        Two constraints pull against each other here, and getting one
        without the other is what made the region selection work sometimes
        and not others.

        The plot has to be **replaced**: axis limits are fixed when a
        Bokeh plot is built, so a frame returned by an existing DynamicMap
        cannot change them.  Zooming to a box, and un-zooming on *Reset
        region*, both need a new plot -- returning a differently-limited
        frame into the old one silently keeps the old limits.

        The stream has to be **kept**: a fresh ``BoundsXY`` per redraw
        loses the selection outright.

        Keeping the stream across a new DynamicMap is not enough on its
        own, though, and that was the bug.  HoloViews links the box-select
        tool by looking up ``Stream.registry[stream.source]``, and
        ``source`` stays pinned to the *first* DynamicMap the stream was
        given to.  So the replacement plot was built with no
        BoundsCallback: the box drew and reported nothing, the region
        silently stayed as it was, and everything downstream -- the zoom,
        the region figure, the JPDFs -- described the old box.  Re-pointing
        ``source`` at the new map restores the link.

        *refetch* false reuses the field already in hand and changes only
        the limits and the outline -- what drawing a box needs.
        """
        if refetch:
            # Mark the base stale rather than dropping it.  Dropping it
            # first made *Rebuild map* refetch -- and defeated the
            # fallback in ``_map_for_bounds`` at exactly the moment it was
            # needed: if the refetch failed there was then nothing to fall
            # back to, so a failed Rebuild replaced a working map with an
            # empty frame.  Staleness says "refetch"; the old base stays
            # available until a new one succeeds.
            self._map_stale = True
        dmap = hv.DynamicMap(self._map_for_bounds, streams=[self._bounds])
        self._bounds.source = dmap          # or the tool goes unwired
        self._map.object = dmap

    #: Frame options that must be re-applied after every composition.
    #:
    #: ``base * outline`` builds a *new* Overlay, and Overlay-level options
    #: set on ``base`` do not come with it -- so the composed map lost its
    #: height and fell back to Bokeh's default, which is the squashed
    #: frame.  Neither does ``.opts(xlim=...)`` restore them.  Keeping them
    #: here and applying them last is the only way the two paths cannot
    #: disagree.
    MAP_HEIGHT = 720

    def _frame_opts(self, extent):
        s = self.state
        xlim, ylim = extent
        return hv.opts.Overlay(
            height=self.MAP_HEIGHT, responsive=True,
            title=f"{self.resolve(s.field)}  —  {s.date}",
            xlabel="longitude", ylabel="latitude",
            show_grid=True, active_tools=["box_select"],
            xlim=tuple(xlim), ylim=tuple(ylim), shared_axes=False,
        )

    def _base_overlay(self, extent):
        """The field, land, coastline and fronts at *extent*."""
        s = self.state
        channel = self.resolve(s.field)
        return basemap.global_map(
            s.provider, s.date, channel,
            show_fronts=s.show_fronts,
            title=f"{channel}  —  {s.date}",
            extent=extent, height=self.MAP_HEIGHT,
        )

    def _map_for_bounds(self, bounds=None):
        """One frame of the map, for whatever region is selected."""
        # The box itself was already recorded by ``_record_bounds``; by the
        # time a frame is drawn, ``state.box`` is the truth.
        extent = self._zoom_limits()

        if self._map_base is not None and not self._map_stale:
            # Static base, moving frame.  Drawing a box must not re-fetch
            # the field: the zoomed extent asks for a finer pyramid level
            # than the global view did, and for a depth channel that level
            # may not exist -- so the redraw failed, the handler returned
            # an empty overlay, and the map collapsed.  Cropping what is
            # already in hand costs nothing, so selection stays instant;
            # *Rebuild* is what re-reads the field at the finer level.
            base, _ = self._map_base
            outline = self._selection_outline()
            overlay = base * outline if outline is not None else base
            return overlay.opts(self._frame_opts(extent))

        try:
            overlay = self._base_overlay(extent)
        except Exception as exc:                        # noqa: BLE001
            self._status.object = f"**Map unavailable:** {exc}"
            # Keep whatever is already on screen.  Returning an empty
            # overlay replaces a working map with a collapsed frame, which
            # loses the box-select tool along with the picture.
            if self._map_base is not None:
                base, base_extent = self._map_base
                return base.opts(self._frame_opts(base_extent))
            return hv.Overlay([hv.Curve([])])

        self._map_base = (overlay, extent)
        self._map_stale = False

        outline = self._selection_outline()
        if outline is not None:
            overlay = overlay * outline

        return overlay.opts(self._frame_opts(extent))

    # -- distributions ---------------------------------------------------

    def draw_regionmap(self):
        """The selected region as a static figure, at native resolution.

        Static on purpose.  The map above is interactive and draws from
        the display pyramid, which is budgeted to stay responsive while
        you navigate; this one is built once behind *Rebuild* and reads
        the same native arrays the statistics do.  Mixing the two -- an
        interactive map demanding the finest pyramid level -- makes every
        pan and zoom heavier for a picture nobody is panning.
        """
        s = self.state
        box = s.box
        if box is None or box.is_global:
            self._regionmap.object = None
            return

        self._regionmap.loading = True
        try:
            self._regionmap.object = P.figure_region_map(
                s.provider, s.date, self.resolve(s.field), box,
                show_fronts=True, field_name=s.field)
        except Exception as exc:                            # noqa: BLE001
            self._regionmap.object = P._blank(f"region map: {exc}")
            print(f"[characteristics] region map unavailable: {exc}")
        finally:
            self._regionmap.loading = False

    def schedule_stats(self):
        self._token += 1
        token = self._token
        self.draw_regionmap()
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
        controls += [self.w_field, self.w_fronts, self.w_reset]

        top = pn.Column(
            pn.pane.Markdown(f"### {self.mode.title}", margin=(4, 10, 0, 10)),
            pn.Row(*controls, sizing_mode="stretch_width", margin=(0, 10)),
            # The button on its own row: the controls above it change the
            # selection, it is the thing that spends time on it.
            pn.Row(pn.pane.Markdown(
                       "**map**" if self.mode.manual_map else "**figures**",
                       margin=(8, 5, 0, 10)),
                   self.w_build, sizing_mode="stretch_width",
                   margin=(0, 10)),
            self._status,
            self._map,
            pn.pane.Markdown(
                "<small>Drag a box on the map to choose a region — the map "
                "zooms to it and the selection stays outlined. Statistics "
                "are exact, at full resolution, on the native grid; the map "
                "is drawn from a regridded display pyramid."
                + (" Fronts are always the <b>surface</b> fronts: a "
                   "front-only panel at depth is that field at depth, "
                   "sampled where the surface front is."
                   if self.mode.has_depth else "")
                + "</small>",
                margin=(0, 10)),
            # Where the map is manual the statistics are their own
            # request, so the button sits with the region it acts on.
            *([pn.Row(pn.pane.Markdown("**figures**",
                                       margin=(8, 5, 0, 10)),
                      self.w_stats, sizing_mode="stretch_width",
                      margin=(0, 10))] if self.mode.manual_map else []),
            pn.pane.Markdown(
                "#### The selected region, with fronts",
                margin=(10, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>A static figure, built on <i>Rebuild</i>, at the "
                "grid's own resolution — the same native arrays the "
                "statistics read, so it costs no extra data. The map above "
                "is the interactive one and draws from the display "
                "pyramid.</small>", margin=(0, 10)),
            self._regionmap,
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
