"""Evolution — one front, played through 24 consecutive hours.

Same shape as the Tiles page — map, pick a region, pick a front — but the
region is a **chunk** (one box saved at many timesteps) and the result is a
movie rather than a still.

Layout:

* a global map with the chunks boxed, and the chunk map at the current step;
* three time series — **(a)** length, **(b)** orientation, **(c)** the
  field's statistics — with a cursor that moves with playback, so you can
  always see where in the window the figures below are;
* **(d)–(i)** the six figures, one frame per step.

Playback swaps pre-rendered images. Rendering costs roughly ten seconds a
frame, so building the movie is an explicit, progress-tracked step and the
result is cached — after which scrubbing is instant.
"""

from __future__ import annotations

import asyncio

import holoviews as hv
import numpy as np
import panel as pn
import param

from fronts.viz import field_styles
from fronts.viz.apps import config
from fronts.viz.apps.common import basemap, widgets
from fronts.viz.apps.common.state import PageState
from fronts.viz.apps.evolution import pipeline as EP
from fronts.viz.apps.tiles import pipeline as TP
from fronts.viz.apps.evolution import timeseries as TS

hv.extension("bokeh")

FRONT_PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
)

BOX_HALF = (7.5, 5.0)

#: Background for the regional chunk map.  The fronts were detected on
#: gradb2, so it is the field they sit on most legibly.
CHUNK_MAP_FIELD = "gradb2"


def _label_rgba(labels, selected: int):
    """Labelled fronts as explicit RGBA channels.

    A colormapped ``hv.Image`` cannot express "draw nothing here": with a
    single-colour cmap every value maps to that colour, NaN included, so
    an overlay of "everything except this front" paints the whole map.
    Building the alpha channel by hand is unambiguous -- background is
    alpha 0, other fronts are muted, the selected front is opaque cyan.
    """
    import matplotlib.colors as mcolors

    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=float)

    present = labels > 0
    idx = (labels[present].astype(int) - 1) % len(FRONT_PALETTE)
    colours = np.array([mcolors.to_rgb(c) for c in FRONT_PALETTE])
    rgba[present, :3] = colours[idx]
    rgba[present, 3] = 0.75

    if selected:
        hit = labels == selected
        rgba[hit, :3] = mcolors.to_rgb("#00e5ff")
        rgba[hit, 3] = 1.0

    return rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]


class EvolutionState(PageState):
    """A chunk, a field, a front, and where we are in the window."""

    chunk = param.Selector(objects=list(config.EVOLUTION_CHUNKS),
                           default=config.EVOLUTION_CHUNKS[0],
                           doc="Which saved chunk to play.")
    field = param.Selector(objects=list(config.TILE_FIELDS_3D), default="Ri",
                           doc="3-D field colouring the figures.")
    front_label = param.Integer(default=0, bounds=(0, None),
                                doc="Selected front, 0 = none.")
    step = param.Integer(default=0, bounds=(0, config.EVOLUTION_N_STEPS - 1),
                         doc="Current timestep.")
    # Two offsets rather than three: the offsets figure is the most
    # expensive of the six and its cost scales with this, which matters
    # when it runs once per frame instead of once.
    n_offsets = param.Integer(default=2, bounds=(1, 5),
                              doc="Offset rows per side.")
    perp_half_width = param.Integer(default=30, bounds=(5, 120),
                                    doc="Half-width of the cross-front transect.")
    include_3d = param.Boolean(True, doc="Render the 3-D still per frame.")
    stat_lines = param.ListSelector(
        objects=list(config.EVOLUTION_STAT_LINES),
        default=list(config.DEFAULT_EVOLUTION_STAT_LINES),
        doc="Statistics drawn on the field time series.")
    built = param.Boolean(False, doc="Frames exist for the current settings.")

    def __init__(self, provider=None, **params):
        super().__init__(provider=provider, **params)
        chunks = self.usable_chunks()
        self.param.chunk.objects = chunks
        if self.chunk not in chunks:
            self.chunk = chunks[0]
        self._set_step_bounds()

    def usable_chunks(self) -> list[str]:
        """Chunks on S3 that are also complete enough to play.

        ``config.EVOLUTION_CHUNKS`` is the allow-list; the intersection
        keeps a name from appearing before its transfer has finished, and
        keeps a configured name from appearing before it exists at all.
        """
        found = self.provider.chunks()
        allowed = config.EVOLUTION_CHUNKS
        if not allowed:
            return found
        usable = [c for c in found if c in allowed]
        return usable or found[:1]

    @param.depends("chunk", watch=True)
    def _set_step_bounds(self):
        """A chunk holds however many timesteps were transferred for it."""
        n = max(1, len(self.times()))
        self.param.step.bounds = (0, n - 1)
        if self.step > n - 1:
            self.step = n - 1

    @param.depends("chunk", "field", "front_label", "n_offsets",
                   "perp_half_width", "include_3d", watch=True)
    def _invalidate(self):
        self.built = False

    def times(self):
        return self.provider.chunk_timesteps(self.chunk)


class EvolutionPage:
    """Assembles the evolution page."""

    def __init__(self, provider=None):
        self.state = EvolutionState(provider=provider)

        self._overview = pn.pane.HoloViews(min_height=540,
                                           sizing_mode="stretch_width")
        self._chunkmap = pn.pane.HoloViews(min_height=620,
                                           sizing_mode="stretch_width")
        self._series = pn.pane.HoloViews(min_height=260,
                                         sizing_mode="stretch_width")
        self._status = widgets.status()
        self._build_status = widgets.status()
        self._progress = pn.indicators.Progress(
            value=0, max=config.EVOLUTION_N_STEPS, width=320, visible=False)
        #: Separate from the movie's progress: loading a chunk is three
        #: reads, and "nothing is happening" was the previous experience.
        self._chunk_progress = pn.indicators.Progress(
            value=0, max=3, width=240, visible=False)

        self._frames: list[dict] = []
        self._series_data: TS.FrontSeries | None = None
        self._labels_step = None
        self._token = 0

        self._panes = {
            k: pn.pane.PNG(sizing_mode="stretch_width", min_height=200)
            for k in EP.FRAME_ORDER
        }

        self._build_controls()
        # Only the 2-D overview at load.  draw_chunkmap reads a chunk
        # store, and refresh_labels/draw_series walk the whole window --
        # none of which should happen before a chunk has been chosen.
        self.draw_overview()
        self._status.object = (
            "pick a chunk, then press *Load chunk* — the regional map and "
            "the front list are read only when you ask")

    # -- controls --------------------------------------------------------

    def _build_controls(self):
        s = self.state
        self.w_chunk = pn.widgets.Select.from_param(s.param.chunk, width=195)
        # An explicit timestep, so the regional map is a known date rather
        # than whatever `step` happened to be.  Front labels are per
        # timestep, so which one you are looking at matters.
        self.w_when = pn.widgets.Select(name="Timestep", options=[],
                                        width=210)
        self.w_when.param.watch(self._on_when, "value")
        self._refresh_timesteps()
        self.w_field = pn.widgets.Select.from_param(s.param.field, width=155)
        self.w_label = pn.widgets.IntInput.from_param(
            s.param.front_label, name="Front label", width=110)
        self.w_avail = pn.widgets.Select(name="Persistent fronts",
                                         options=[], width=150)
        self.w_avail.param.watch(
            lambda e: e.new and setattr(s, "front_label", int(e.new)), "value")
        self.w_offsets = pn.widgets.IntSlider.from_param(
            s.param.n_offsets, name="Offsets per side", width=150)
        self.w_perp = pn.widgets.IntSlider.from_param(
            s.param.perp_half_width, name="Transect half-width", width=165)
        self.w_3d = pn.widgets.Checkbox.from_param(s.param.include_3d)
        self.w_3d.label = "Include 3-D (doubles build time)"
        self.w_stats = pn.widgets.MultiChoice.from_param(
            s.param.stat_lines, name="Statistic lines", width=260)

        self.w_loadchunk = pn.widgets.Button(
            name="Load chunk", button_type="primary", width=140)
        self.w_loadchunk.on_click(lambda _: self.load_chunk())

        self.w_build = pn.widgets.Button(name="Build movie",
                                         button_type="primary", width=170)
        self.w_build.on_click(lambda _: self.schedule_build())

        self.w_player = pn.widgets.Player(
            name="Timestep", start=0,
            end=max(len(s.times()) - 1, 1), value=0,
            interval=400, loop_policy="loop", width=520,
        )
        self.w_player.param.watch(self._on_step, "value")

        s.param.watch(lambda *_: self._on_chunk_change(), ["chunk"])
        s.param.watch(lambda *_: self.draw_chunkmap(), ["step"])
        s.param.watch(lambda *_: self.draw_series(),
                      ["front_label", "field", "stat_lines"])
        s.param.watch(lambda *_: self._reflect_built(), ["built"])
        self._reflect_built()

    def _refresh_timesteps(self):
        """Fill the timestep list from the chunk's own store listing."""
        try:
            times = self.state.times()
        except Exception:                                   # noqa: BLE001
            times = []
        self.w_when.options = list(times)
        if times:
            current = min(int(self.state.step), len(times) - 1)
            self.w_when.value = times[current]

    def _on_when(self, event):
        if not event.new:
            return
        try:
            self.state.step = list(self.w_when.options).index(event.new)
        except ValueError:
            pass

    def _on_chunk_change(self):
        """A different chunk is wanted.  Redraw the cheap map only."""
        self._labels_step = None
        self._chunkmap.object = None
        self.w_loadchunk.button_type = "primary"
        self._refresh_timesteps()
        self.draw_overview()
        self._status.object = (
            f"**{self.state.chunk}** — press *Load chunk*")

    def load_chunk(self):
        """Read the chosen chunk: regional map, front list, time series.

        Reading a chunk timestep takes seconds, so the button says so --
        without that, a slow load and a silent failure look identical.
        """
        self.w_loadchunk.loading = True
        self._chunk_progress.visible = True
        steps = (("regional map", self.draw_chunkmap),
                 ("front list", self.refresh_labels),
                 ("time series", self.draw_series))
        self._chunk_progress.max = len(steps)
        try:
            for n, (what, run) in enumerate(steps, start=1):
                self._status.object = (
                    f"reading **{self.state.chunk}** — {what} "
                    f"({n}/{len(steps)}) …")
                run()
                self._chunk_progress.value = n
        finally:
            self.w_loadchunk.loading = False
            self.w_loadchunk.button_type = "default"
            self._chunk_progress.visible = False
            self._chunk_progress.value = 0

    def _on_step(self, event):
        self.state.step = int(event.new)
        self.show_frame(int(event.new))

    def _reflect_built(self):
        if self.state.built:
            self.w_build.button_type = "default"
        else:
            self.w_build.button_type = "primary"
            n = len(self.state.times())
            # Measured on the synthetic chunk: ~14 s a frame with the 3-D
            # still, ~8 s without.  The offsets figure and the 3-D render
            # are roughly half each.
            lo = (n * (8 if not self.state.include_3d else 14)) // 60
            hi = (n * (14 if not self.state.include_3d else 25)) // 60
            self._build_status.object = (
                f"⟳ **not built for these settings** — *Build movie* renders "
                f"{n} frames, roughly {max(lo,1)}–{max(hi,2)} min. "
                "Cached on disk afterwards, so scrubbing is instant.")

    # -- maps ------------------------------------------------------------

    def draw_overview(self):
        s = self.state
        try:
            base = basemap.global_map(
                s.provider, s.provider.dates_3d()[0], "gradb2",
                height=520, title="Saved chunks",
                tools=("tap",), active_tools=("tap",))
        except Exception as exc:                            # noqa: BLE001
            self._overview.object = None
            self._status.object = f"**Overview unavailable:** {exc}"
            return

        boxes, labels = [], []
        for name in s.usable_chunks():
            try:
                lat, lon = s.provider.chunk_location(name)
            except Exception:                               # noqa: BLE001
                continue                    # a chunk with no grid of its own
            x = lon % 360.0
            boxes.append((x - BOX_HALF[0], lat - BOX_HALF[1],
                          x + BOX_HALF[0], lat + BOX_HALF[1]))
            labels.append((x, lat + BOX_HALF[1] + 3.0, name))

        overlay = (
            base
            * hv.Rectangles(boxes).opts(fill_alpha=0.0, line_color="red",
                                        line_width=2)
            * hv.Labels(labels, vdims="text").opts(
                text_color="red", text_font_size="7pt", text_align="center")
        )
        self._overview.object = overlay

    def draw_chunkmap(self):
        """The chunk at the current step, with its fronts.

        This is the regional map: 720 x 720 native cells over the chunk's
        own box, so it is already at full resolution -- there is no
        pyramid between it and the data.
        """
        s = self.state
        step = int(s.step)
        try:
            # gradb2 in the background, not the movie's colour field: this
            # map is for finding a front, and gradb2 is what the fronts
            # were detected on.
            ds = s.provider.chunk_tile(s.chunk, step, CHUNK_MAP_FIELD)
            var = ds.attrs.get("tile_var_name") or list(ds.data_vars)[0]
            surface = TP.field_values(ds, var)[0]
        except Exception as exc:                            # noqa: BLE001
            self._chunkmap.object = None
            self._status.object = (
                f"**Chunk unavailable** — {type(exc).__name__}: {exc}")
            return

        # The field comes from the chunk store; the labels come from the
        # front detection, which may not have been run over this window.
        # Missing labels cost the overlay, not the map.
        try:
            labels = s.provider.chunk_labels(s.chunk, step)
        except Exception as exc:                            # noqa: BLE001
            labels = None
            self._status.object = f"*Fronts not overlaid:* {exc}"

        self._labels_step = labels

        style = field_styles.get_style(var)
        shown = field_styles.apply_transform(surface, style)
        overlay = hv.Image(
            (np.arange(surface.shape[1]), np.arange(surface.shape[0]),
             shown), kdims=["i", "j"], vdims=[CHUNK_MAP_FIELD],
        ).opts(cmap=basemap.bokeh_cmap(style.cmap),
               clim=field_styles.default_clim(shown, style), colorbar=True)

        if labels is not None:
            overlay = overlay * hv.RGB(
                (np.arange(labels.shape[1]), np.arange(labels.shape[0]),
                 *_label_rgba(labels, s.front_label)),
                kdims=["i", "j"], vdims=["R", "G", "B", "A"])

        frame_opts = dict(
            responsive=True, height=600, active_tools=["tap"],
            title=f"{s.chunk} — step {step} — {s.times()[step]}",
            xlabel="i", ylabel="j")
        overlay = overlay.opts(
            hv.opts.Overlay(**frame_opts) if labels is not None
            else hv.opts.Image(**frame_opts))

        hv.streams.Tap(source=overlay).add_subscriber(self._on_tap)
        self._chunkmap.object = overlay

    def _on_tap(self, x, y):
        if self._labels_step is None or x is None or y is None:
            return
        j, i = int(round(float(y))), int(round(float(x)))
        nj, ni = self._labels_step.shape
        if 0 <= j < nj and 0 <= i < ni:
            label = int(self._labels_step[j, i])
            if label:
                self.state.front_label = label

    def refresh_labels(self):
        s = self.state
        try:
            labels = TS.common_labels(s.provider, s.chunk)
        except Exception:                                   # noqa: BLE001
            labels = []
        self.w_avail.options = [str(l) for l in labels]
        if labels and s.front_label not in labels:
            s.front_label = labels[0]
        if labels:
            # Keep the dropdown showing the selection rather than blank.
            self.w_avail.value = str(s.front_label)
        self._status.object = (
            f"**{len(labels)}** fronts persist across the window "
            f"({len(s.times())} hourly steps)")

    # -- time series -----------------------------------------------------

    def draw_series(self):
        """Three panels, with a cursor that follows playback."""
        s = self.state
        if not s.front_label:
            self._series.object = None
            return

        try:
            series = TS.build(s.provider, s.chunk, s.front_label, s.field)
        except Exception as exc:                            # noqa: BLE001
            self._series.object = None
            self._status.object = f"**Time series unavailable:** {exc}"
            return

        self._series_data = series
        steps = series.steps

        def panels(step):
            cursor = hv.VLine(float(step)).opts(
                color="#e6194b", line_width=2, line_dash="dashed")

            a = (hv.Curve((steps, series.length_km), "step", "length [km]"
                          ).opts(color="#1f4e5f", line_width=2)
                 * hv.Scatter((steps, series.length_km)).opts(
                     size=4, color="#1f4e5f") * cursor
                 ).opts(title="(a) front length", width=430, height=230)

            b = (hv.Curve((steps, series.orientation), "step",
                          "orientation [deg]"
                          ).opts(color="#8a5a00", line_width=2)
                 * hv.Scatter((steps, series.orientation)).opts(
                     size=4, color="#8a5a00") * cursor
                 ).opts(title="(b) orientation (0 = N–S)", width=430,
                        height=230, ylim=(0, 90))

            lines = []
            for name in s.stat_lines:
                values = series.stats.get(name)
                if values is None:
                    continue
                lines.append(hv.Curve((steps, values), "step", s.field,
                                      label=name).opts(line_width=2))
            c = (hv.Overlay(lines) * cursor if lines
                 else hv.Curve(([], []), "step", s.field) * cursor)
            c = c.opts(title=f"(c) {s.field} over the front", width=470,
                       height=230, legend_position="right", show_legend=True)

            return (a + b + c).cols(3)

        self._series.object = pn.bind(panels, self.w_player.param.value)

    # -- building --------------------------------------------------------

    def schedule_build(self):
        self._token += 1
        token = self._token

        if not self.state.front_label:
            self._build_status.object = "**Pick a front first.**"
            return

        self._progress.visible = True
        self._progress.value = 0
        self._progress.max = len(self.state.times())
        self._build_status.object = "building frames…"

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

        def progress(done, total):
            if token == self._token:
                self._progress.value = done
                self._build_status.object = f"rendering frame {done}/{total}…"

        try:
            frames = EP.prerender(
                s.provider, s.chunk, s.field, s.front_label,
                n_offsets=s.n_offsets, perp_half_width=s.perp_half_width,
                progress=progress)
        except Exception as exc:                            # noqa: BLE001
            if token == self._token:
                self._progress.visible = False
                self._build_status.object = f"**Build failed:** {exc}"
            return

        if token != self._token:
            return

        self._frames = frames
        self._progress.visible = False
        s.built = True

        ok = sum(1 for f in frames if f.get("mainaxis"))
        self._build_status.object = (
            f"**{ok}/{len(frames)}** frames ready for front "
            f"**{s.front_label}** · press play")
        self.show_frame(int(self.w_player.value))

    def show_frame(self, step: int):
        if not self._frames or step >= len(self._frames):
            return
        frame = self._frames[step]
        for kind, pane in self._panes.items():
            pane.object = frame.get(kind)

    # -- layout ----------------------------------------------------------

    def view(self):
        # One row per stage, each ending in its own button.
        # (a) chunk -> Load chunk
        row_a = pn.Row(self.w_chunk, self.w_when,
                       sizing_mode="stretch_width", margin=(0, 10))
        row_a_go = pn.Row(pn.pane.Markdown("**a · chunk → regional map**",
                                           margin=(8, 5, 0, 10)),
                          self.w_loadchunk, self._chunk_progress,
                          sizing_mode="stretch_width", margin=(0, 10))

        # (b) everything the movie needs, then build it
        row_b = pn.Row(self.w_field, self.w_avail, self.w_label,
                       sizing_mode="stretch_width", margin=(0, 10))
        row_b2 = pn.Row(self.w_offsets, self.w_perp, self.w_3d,
                        self.w_stats,
                        sizing_mode="stretch_width", margin=(0, 10))
        row_b_go = pn.Row(pn.pane.Markdown("**b · field + front → movie**",
                                           margin=(8, 5, 0, 10)),
                          self.w_build, self._progress,
                          sizing_mode="stretch_width", margin=(0, 10))

        # Stacked so each map gets the full page width.
        maps = pn.Column(
            pn.pane.Markdown("**Chunks**", margin=(0, 10)),
            self._overview,
            pn.pane.Markdown("**Chunk at this step**", margin=(8, 10, 0, 10)),
            self._chunkmap,
            sizing_mode="stretch_width",
        )

        frames = pn.GridBox(*[
            pn.Column(pn.pane.Markdown(f"<small>{EP.FRAME_TITLES[k]}</small>",
                                       margin=(4, 5, 0, 5)),
                      self._panes[k])
            for k in EP.FRAME_ORDER
        ], ncols=3, sizing_mode="stretch_width")

        body = pn.Column(
            pn.pane.Markdown("### Evolution — one front over 24 hours",
                             margin=(4, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>A chunk is one box saved at many consecutive "
                "timesteps. Pick a front that persists across the window, "
                "build the frames once, then play. The 3-D view is a fixed "
                "camera on purpose — with the data already moving, a "
                "moveable camera makes it impossible to tell what "
                "changed.</small>",
                margin=(0, 10)),
            row_a, row_a_go, self._status, maps,
            # The movie controls sit under the regional map, which is where
            # the front is chosen -- not scattered above it.
            row_b, row_b2, row_b_go,
            pn.layout.Divider(),
            pn.pane.Markdown("### Time series", margin=(6, 10, 0, 10)),
            self._series,
            pn.layout.Divider(),
            pn.Row(pn.pane.Markdown("### Frames", margin=(6, 10, 0, 10)),
                   self.w_player, margin=(0, 0)),
            self._build_status,
            frames,
            sizing_mode="stretch_width",
        )

        notes = [n for n in (widgets.banner(self.state.provider),
                             widgets.degraded_notice()) if n]
        return pn.Column(*notes, body, sizing_mode="stretch_width")


def page(provider=None):
    """Entry point used by ``serve.py``."""
    return EvolutionPage(provider=provider).view()
