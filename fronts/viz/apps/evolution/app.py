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
import logging
import pathlib
import time

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
from fronts.viz.apps.tiles import panels as F
from fronts.viz.apps.evolution import tracking as TR

hv.extension("bokeh")

log = logging.getLogger(__name__)

#: One colour for every front.  The numbers on the map identify them, so
#: the colour does not have to.
FRONT_COLOR = "#00e5ff"
SELECTED_COLOR = "#ff1744"

BOX_HALF = (7.5, 5.0)

#: Background for the regional chunk map.  The fronts were detected on
#: gradb2, so it is the field they sit on most legibly.
CHUNK_MAP_FIELD = "gradb2"


def _label_rgba(labels, selected: int, *, width: int = 1):
    """Labelled fronts as explicit RGBA channels: cyan, thickened.

    A colormapped ``hv.Image`` cannot express "draw nothing here": with a
    single-colour cmap every value maps to that colour, NaN included, so
    an overlay of "everything except this front" paints the whole map.
    Building the alpha channel by hand is unambiguous -- background is
    alpha 0, fronts are opaque.

    One colour, not a palette.  A per-front palette had to be matched
    against a dropdown by eye; the numbers printed on the map do that job
    properly, so the colour is free to be uniform and legible.

    *width* dilates the mask by one cell: a front is often a single cell
    wide, which at 720 cells across a 600-pixel figure is a sub-pixel
    line that does not survive rasterising.  One cell is enough to make
    it visible; two turned the fronts into blobs.
    """
    import matplotlib.colors as mcolors

    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=float)

    present = _thicken(labels > 0, width)
    rgba[present, :3] = mcolors.to_rgb(FRONT_COLOR)
    rgba[present, 3] = 1.0

    if selected:
        hit = _thicken(labels == selected, width)
        rgba[hit, :3] = mcolors.to_rgb(SELECTED_COLOR)
        rgba[hit, 3] = 1.0

    return rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]


def _read_bytes(path):
    """A frame's PNG bytes, or ``None`` if it was never produced."""
    if not path:
        return None
    try:
        return pathlib.Path(path).read_bytes()
    except OSError:
        return None


def _thicken(mask, width: int):
    """Grow a boolean mask by *width* cells, with numpy alone.

    ``scipy.ndimage.binary_dilation`` would do this, but rolling a 1-cell
    front by a few offsets is the whole operation and needs no dependency.
    """
    if width <= 0:
        return mask
    out = mask.copy()
    for dj in range(-width, width + 1):
        for di in range(-width, width + 1):
            out |= np.roll(np.roll(mask, dj, axis=0), di, axis=1)
    return out


def _front_number_labels(labels, *, top: int = 60):
    """The front number at each front's centroid, biggest fronts first.

    This is the figure a front is chosen from, so the numbers have to be
    on it -- a dropdown of five-digit integers is not a way to find a
    feature you can see.  *top* is generous for the same reason: a cap
    that only labels the few biggest hides most of what you can see.
    """
    present = [int(v) for v in np.unique(labels) if v]
    sized = sorted(((int((labels == v).sum()), v) for v in present),
                   reverse=True)[:top]

    rows = []
    for _, value in sized:
        js, iss = np.nonzero(labels == value)
        if js.size:
            rows.append((float(iss.mean()), float(js.mean()), str(value)))

    if not rows:
        return hv.Labels([], kdims=["i", "j"], vdims=["text"])

    return hv.Labels(rows, kdims=["i", "j"], vdims=["text"]).opts(
        text_color="white", text_font_size="8pt", text_align="center",
        text_baseline="middle", text_font_style="bold",
        background_fill_color="#101010", background_fill_alpha=0.72,
        padding=2, border_radius=2)


class EvolutionState(PageState):
    """A chunk, a field, a front, and where we are in the window."""

    chunk = param.Selector(objects=list(config.EVOLUTION_CHUNKS),
                           default=config.EVOLUTION_CHUNKS[0],
                           doc="Which saved chunk to play.")
    field = param.Selector(
        objects=list(config.TILE_FIELDS_3D) + list(config.CHUNK_SURFACE_FIELDS),
        default="Ri",
        doc="Field colouring the figures.  The surface-only ones can "
            "colour the region movie but have no depth to section.")
    # The selection is a *place*, not a label.  Labels are assigned per
    # timestep, so a label identifies a front in one frame and nothing in
    # the next; a point on the ocean means the same thing in all of them.
    # front_label is now derived -- it says which label the point resolved
    # to at the anchor step, for display only.
    anchor_lon = param.Number(default=0.0,
                              doc="Longitude of the selected point, 0..360.")
    anchor_lat = param.Number(default=0.0,
                              doc="Latitude of the selected point.")
    front_label = param.Integer(default=0, bounds=(0, None),
                                doc="Label the point resolved to, 0 = none.")

    # 0 means "unset" -- fall back to the front point.  A sentinel rather
    # than None so the widgets stay plain numeric inputs.
    perp_lat = param.Number(default=0.0, doc="Transect latitude, 0 = follow.")
    perp_lon = param.Number(default=0.0, doc="Transect longitude, 0 = follow.")

    step = param.Integer(default=0, bounds=(0, config.EVOLUTION_N_STEPS - 1),
                         doc="Current timestep.")
    # Two offsets rather than three: the offsets figure is the most
    # expensive of the six and its cost scales with this, which matters
    # when it runs once per frame instead of once.
    n_offsets = param.Integer(default=2, bounds=(1, 5),
                              doc="Offset rows per side.")
    perp_half_width = param.Integer(default=30, bounds=(5, 120),
                                    doc="Half-width of the cross-front transect.")
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

    @param.depends("chunk", "field", "anchor_lon", "anchor_lat", "n_offsets",
                   "perp_half_width", "perp_lat", "perp_lon", watch=True)
    def _invalidate(self):
        self.built = False

    def times(self):
        return self.provider.chunk_timesteps(self.chunk)

    @param.depends("anchor_lat", "anchor_lon", watch=True)
    def _follow_anchor(self):
        """Move the transect to the newly chosen front point.

        Choosing a front and then having to retype almost the same
        coordinates for the transect was busywork -- the front point is
        the right default.  Editing the transect afterwards still sticks,
        because only a change to the *front* point moves it.
        """
        self.perp_lat = float(self.anchor_lat)
        self.perp_lon = float(self.anchor_lon)

    def perp_point(self):
        """Where the transect sits, or the front point if it is unset."""
        if self.perp_lat or self.perp_lon:
            return (float(self.perp_lon), float(self.perp_lat))
        if self.anchor_lat or self.anchor_lon:
            return (float(self.anchor_lon), float(self.anchor_lat))
        return None

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

        # False until the page is assembled.  _refresh_timesteps sets the
        # Timestep widget during construction, which fires _on_when -- and
        # that now redraws the chunk map, which reads a tile.  Nothing must
        # touch S3 before the user presses Load chunk.
        self._ready = False
        self._frames: list[dict] = []
        self._region_frames: list[str | None] = []
        #: The same frames as bytes, so playback never waits on a fetch.
        self._region_bytes: list[bytes | None] = []
        # scale_width: the full page width with the aspect ratio kept.
        # Safe for playback because every frame is rendered at the same
        # figsize and dpi, so the height never changes between them and
        # there is no reflow to see.
        self._region_pane = pn.pane.PNG(sizing_mode="scale_width")
        self._region_progress = pn.indicators.Progress(
            value=0, max=1, width=320, visible=False)
        self._region_status = widgets.status()
        self._preview = pn.pane.HoloViews(sizing_mode="stretch_width",
                                          min_height=560)
        self._preview_scene = None
        self._preview_status = widgets.status()
        #: The stacked profile figure, one variant per highlighted step.
        self._profile_bytes: list[bytes | None] = []
        # Fixed width: this figure is tall and narrow, and scale_width
        # across the whole page would make it enormous.
        self._profile_pane = pn.pane.PNG(width=460)
        self._series_data: TS.FrontSeries | None = None
        self._track = None
        self._labels_step = None
        self._coords_step = None
        self._token = 0

        self._panes = {
            k: pn.pane.PNG(sizing_mode="scale_width")
            for k in EP.FRAME_ORDER
        }
        self._downloads = {
            k: pn.widgets.FileDownload(
                label="Download GIF", filename=f"{k}.gif", width=150,
                callback=(lambda kind=k: self._figure_gif(kind)),
                disabled=True)
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
        self._ready = True

    # -- controls --------------------------------------------------------

    def _build_controls(self):
        s = self.state
        self.w_chunk = pn.widgets.Select.from_param(s.param.chunk, width=195)
        # An explicit timestep, so the regional map is a known date rather
        # than whatever `step` happened to be.  Front labels are per
        # timestep, so which one you are looking at matters.
        self.w_when = pn.widgets.Select(label="Timestep", options=[],
                                        width=210)
        self.w_when.param.watch(self._on_when, "value")
        self._refresh_timesteps()
        self.w_field = pn.widgets.Select.from_param(s.param.field, width=155)
        # The selection.  Typed or clicked -- either way it is a place.
        self.w_lat = pn.widgets.FloatInput.from_param(
            s.param.anchor_lat, label="Latitude", width=120, step=0.01)
        self.w_lon = pn.widgets.FloatInput.from_param(
            s.param.anchor_lon, label="Longitude (0-360)", width=140,
            step=0.01)
        self.w_resolve = pn.widgets.Button(label="Find front here", width=150)
        self.w_resolve.on_click(lambda _: self._resolve_anchor())

        # Where the cross-front transect sits, geographically.  Blank means
        # "wherever the front was selected", which is the useful default.
        self.w_perp_lat = pn.widgets.FloatInput.from_param(
            s.param.perp_lat, label="Transect lat", width=120, step=0.01)
        self.w_perp_lon = pn.widgets.FloatInput.from_param(
            s.param.perp_lon, label="Transect lon", width=140, step=0.01)



        # "at this step", not "persistent": the label is only valid
        # here, and tracking is what carries it across steps.
        self.w_offsets = pn.widgets.IntSlider.from_param(
            s.param.n_offsets, name="Offsets per side", width=150)
        self.w_perp = pn.widgets.IntSlider.from_param(
            s.param.perp_half_width, name="Transect half-width", width=165)
        self.w_stats = pn.widgets.MultiChoice.from_param(
            s.param.stat_lines, name="Statistic lines", width=260)

        self.w_loadchunk = pn.widgets.Button(
            name="Load chunk", button_type="primary", width=140)
        self.w_loadchunk.on_click(lambda _: self.load_chunk())

        # Stage (b): the region movie.  Separate from the section movie
        # because it needs no front chosen -- it is how you choose one.
        self.w_region = pn.widgets.Button(
            label="Build region movie", button_type="primary", width=190)
        self.w_region.on_click(lambda _: self.schedule_region())

        self.w_download = pn.widgets.FileDownload(
            label="Download movie (GIF)", filename="region_movie.gif",
            callback=self._movie_gif, button_type="default", width=200,
            disabled=True)

        self.w_region_player = pn.widgets.Player(
            label="Region step", start=0,
            end=max(len(s.times()) - 1, 1), value=0,
            interval=600, loop_policy="loop", width=520)
        self.w_region_player.param.watch(self._on_region_step, "value")

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
        # NOT watching `step`.  The region player sets it on every frame,
        # and draw_chunkmap reads a tile -- so playback queued a network
        # read per frame, fell behind the player, and showed a frame that
        # did not match the marker.  The Timestep selector redraws the map
        # explicitly instead.
        # Deliberately NOT watching front_label or field.  The series walk
        # every step, so recomputing them when a selection changes made
        # picking a front a multi-minute operation -- the same "navigation
        # is not computation" rule the Field Characteristics map follows.
        # They are built by *Build movie*, which already walks the window.
        s.param.watch(lambda *_: self._restyle_series(), ["stat_lines"])
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
        """The Timestep selector: move the step *and* redraw the map."""
        if not event.new:
            return
        try:
            self.state.step = list(self.w_when.options).index(event.new)
        except ValueError:
            return
        if self._ready:
            self.draw_chunkmap()

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
        """Read the chosen chunk, off the server thread.

        Run inline, the status updates below never reach the browser: Panel
        flushes property changes when the callback yields, so a long
        synchronous load shows the *previous* message for its whole
        duration -- indistinguishable from a hang.  ``_build`` already went
        to a thread for exactly this reason.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._load_chunk()
        else:
            pn.state.execute(
                lambda: asyncio.ensure_future(self._load_chunk_async()))

    async def _load_chunk_async(self):
        await asyncio.to_thread(self._load_chunk)

    def _load_chunk(self):
        self.w_loadchunk.loading = True
        self._chunk_progress.visible = True
        # Two stages, both one timestep.  The time series used to be a
        # third -- but they need every step, which is ~0.9 GB of labels
        # each, so loading a chunk took a quarter of an hour.  They belong
        # with the movie build, which is already explicit and cached.
        steps = (("regional map", self.draw_chunkmap),
                 ("front list", self.refresh_labels))
        self._chunk_progress.max = len(steps)
        # Timed per stage.  Guessing which stage was slow cost more than
        # one round of debugging, and the numbers are two lines of code.
        timings = []
        try:
            for n, (what, run) in enumerate(steps, start=1):
                self._status.object = (
                    f"reading **{self.state.chunk}** — {what} "
                    f"({n}/{len(steps)}) …")
                started = time.perf_counter()
                run()
                elapsed = time.perf_counter() - started
                timings.append(f"{what} {elapsed:.1f}s")
                log.info("load_chunk: %s took %.1fs", what, elapsed)
                self._chunk_progress.value = n
            self._status.object = (
                f"**{self.state.chunk}** loaded — " + " · ".join(timings)
                + f"\n\n{self._status.object or ''}")
        finally:
            self.w_loadchunk.loading = False
            self.w_loadchunk.button_type = "default"
            self._chunk_progress.visible = False
            self._chunk_progress.value = 0

    def _on_step(self, event):
        self.state.step = int(event.new)
        self.show_frame(int(event.new))
        # The profile stack tracks the same cursor the time series do, so
        # the lit-up profile is always the step being looked at.
        self._show_profile(int(event.new))

    def _reflect_built(self):
        if self.state.built:
            self.w_build.button_type = "default"
        else:
            self.w_build.button_type = "primary"
            n = len(self.state.times())
            # Measured on the synthetic chunk: ~8 s a frame for the five
            # figures, of which the offsets figure is about half.
            lo = (n * 8) // 60
            hi = (n * 14) // 60
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
            # Through chunk_plane, so this is in the rect frame like the
            # labels are.  Reading the tile directly here is what left the
            # chunk unrotated under rotated fronts.
            surface, _lon, _lat, _lbl, var = EP.chunk_plane(
                s.provider, s.chunk, step, CHUNK_MAP_FIELD)
            # Kept so a click can be turned into a place.  Both are 2-D:
            # XC/YC are 2-D on this grid and stay that way.
            self._coords_step = (_lon, _lat)
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
            # Numbered here, and deliberately *not* on the movie frames:
            # this is the still you pick from, those are the ones you watch.
            overlay = overlay * _front_number_labels(labels)

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
        """Clicking the map sets the anchor **point**, not a label.

        The tapped pixel used to be resolved straight to a label, which
        threw away the only part of the click that survives re-labelling:
        where it was.  You no longer have to hit a front exactly -- the
        nearest one within the search radius is taken.
        """
        if self._coords_step is None or x is None or y is None:
            return
        lon, lat = self._coords_step
        j, i = int(round(float(y))), int(round(float(x)))
        nj, ni = lat.shape
        if not (0 <= j < nj and 0 <= i < ni):
            return

        self.state.anchor_lon = float(lon[j, i])
        self.state.anchor_lat = float(lat[j, i])
        self._resolve_anchor()

    def _point_from_label(self, label: int):
        """Set the anchor point to a chosen front's centroid."""
        if self._coords_step is None:
            self.state.front_label = int(label)
            return
        try:
            labels = self.state.provider.chunk_labels(
                self.state.chunk, self.state.step)
        except Exception:                                   # noqa: BLE001
            return
        mask = labels == int(label)
        if not mask.any():
            return
        lon, lat = self._coords_step
        self.state.anchor_lon = float(lon[mask].mean())
        self.state.anchor_lat = float(lat[mask].mean())
        self.state.front_label = int(label)

    def _resolve_anchor(self):
        """Which front is at the anchor point, at the current step."""
        s = self.state
        try:
            labels = s.provider.chunk_labels(s.chunk, s.step)
            lon, lat = self._coords_step
            label, km = TR.nearest_front(labels, lon, lat,
                                         s.anchor_lon, s.anchor_lat)
        except Exception as exc:                            # noqa: BLE001
            self._status.object = f"**Could not resolve the point:** {exc}"
            return

        if label is None:
            s.front_label = 0
            self._status.object = (
                f"no front within reach of ({s.anchor_lat:.3f}, "
                f"{s.anchor_lon:.3f}) — nearest is {km:.0f} km away")
            return

        s.front_label = int(label)
        self._status.object = (
            f"point ({s.anchor_lat:.3f}, {s.anchor_lon:.3f}) → front "
            f"**{label}** at this step, {km:.1f} km away — drawing it…")
        self._draw_preview()

    def _draw_preview(self):
        """The selected front alone, at the current step, and clickable.

        A HoloViews plot rather than a rendered PNG: the transect point is
        chosen from this figure, and a PNG can be looked at but not
        clicked.  Everything else on the page is a still because it gets
        played back; this one is interactive because it is an input.
        """
        s = self.state
        if not s.front_label:
            self._preview.object = None
            return

        self._preview_status.object = "drawing the selected front…"
        try:
            scene = EP.build_step(s.provider, s.chunk, s.step, s.field,
                                  int(s.front_label))
        except Exception as exc:                            # noqa: BLE001
            self._preview.object = None
            self._preview_status.object = (
                f"**Could not draw front {s.front_label}:** {exc}")
            return

        self._preview_scene = scene
        surface = np.asarray(scene.color[0])
        lon = np.asarray(scene.XC) % 360.0
        lat = np.asarray(scene.YC)

        # hv.Image needs a regular axis per dimension; over a crop this
        # small an even span across the true extent is accurate to well
        # under a cell.  Same approximation the Tiles plan view makes.
        xs = np.linspace(float(np.nanmin(lon)), float(np.nanmax(lon)),
                         surface.shape[1])
        ys = np.linspace(float(np.nanmin(lat)), float(np.nanmax(lat)),
                         surface.shape[0])

        # scene.color is ALREADY through the display transform -- step 6
        # of build_scene does it, so the curtains and this share one
        # convention.  Transforming again is log of a log, which for a
        # positive-definite field goes NaN and leaves the map blank.  The
        # scene's own clim goes with it, for the same reason.
        style = field_styles.get_style(s.field)
        img = hv.Image((xs, ys, surface), kdims=["lon", "lat"],
                       vdims=[s.field]).opts(
            cmap=basemap.bokeh_cmap(style.cmap),
            clim=tuple(scene.clim), colorbar=True)

        front = np.where(np.asarray(scene.front_mask), 1.0, np.nan)
        overlay = img * hv.Image((xs, ys, front), kdims=["lon", "lat"],
                                 vdims=["front"]).opts(
            cmap=["#00e5ff"], colorbar=False)

        marker = hv.Points([(s.perp_lon % 360.0, s.perp_lat)]).opts(
            size=13, color="#ff1744", marker="x", line_width=3)

        overlay = (overlay * marker).opts(
            responsive=True, height=560, active_tools=["tap"],
            title=f"front {s.front_label} — {s.times()[s.step]} "
                  "(click to place the transect)",
            xlabel="longitude [deg]", ylabel="latitude [deg]",
            shared_axes=False)

        hv.streams.Tap(source=overlay).add_subscriber(self._on_preview_tap)
        self._preview.object = overlay
        self._preview_status.object = (
            f"front **{s.front_label}** at {s.times()[s.step]} — click to "
            "set the transect point, then *Build movie*")

    def refresh_labels(self):
        s = self.state
        try:
            labels = TR.fronts_present(
                s.provider.chunk_labels(s.chunk, s.step))
        except Exception:                                   # noqa: BLE001
            labels = []
        # No dropdown any more: the selection is a place, so the list of
        # labels was a second way to say the same thing -- and the worse
        # one, since a label only means anything in the step it came from.
        self._status.object = (
            f"**{len(labels)}** fronts at this step "
            f"({self.w_when.value or s.step}) — click one on the map, or "
            "type a lat/lon below")

    def _on_preview_tap(self, x, y):
        """Clicking the plan view places the transect."""
        if x is None or y is None:
            return
        self.state.perp_lon = float(x) % 360.0
        self.state.perp_lat = float(y)
        # Redraw so the marker lands where it was clicked.  Cheap: the
        # scene is already in hand, so this rebuilds a plot, not the data.
        self._draw_preview()

    def draw_series(self):
        """Three panels, with a cursor that follows playback."""
        s = self.state
        if not s.front_label or self._track is None:
            self._series.object = None
            return

        try:
            series = TS.build(s.provider, s.chunk, self._track, s.field)
        except Exception as exc:                            # noqa: BLE001
            self._series.object = None
            self._status.object = f"**Time series unavailable:** {exc}"
            return

        self._series_data = series

        # Real time on the x-axis, not step index.  A chunk is daily
        # snapshots wrapped around one intensive day, so on a step axis
        # the interesting hours are stretched to the same width as a
        # week of gaps -- which hides the only part with time resolution
        # in it.
        stamps = [TR.parse_time(t) for t in series.times]
        xs = np.array(stamps, dtype="datetime64[s]")
        xdim = "time"

        def panels(step):
            at = xs[int(np.clip(step, 0, len(xs) - 1))]
            cursor = hv.VLine(at).opts(
                color="#e6194b", line_width=2, line_dash="dashed")

            a = (hv.Curve((xs, series.length_km), xdim, "length [km]"
                          ).opts(color="#1f4e5f", line_width=2)
                 * hv.Scatter((xs, series.length_km)).opts(
                     size=4, color="#1f4e5f") * cursor
                 ).opts(title="(a) front length", width=430, height=230)

            b = (hv.Curve((xs, series.orientation), xdim,
                          "orientation [deg]"
                          ).opts(color="#8a5a00", line_width=2)
                 * hv.Scatter((xs, series.orientation)).opts(
                     size=4, color="#8a5a00") * cursor
                 ).opts(title="(b) orientation (0 = N–S)", width=430,
                        height=230, ylim=(0, 90))

            lines = []
            for name in s.stat_lines:
                values = series.stats.get(name)
                if values is None:
                    continue
                lines.append(hv.Curve((xs, values), xdim, s.field,
                                      label=name).opts(line_width=2))
            c = (hv.Overlay(lines) * cursor if lines
                 else hv.Curve(([], []), xdim, s.field) * cursor)
            c = c.opts(title=f"(c) {s.field} over the front", width=470,
                       height=230, legend_position="right", show_legend=True)

            return (a + b + c).cols(3)

        self._series.object = pn.bind(panels, self.w_player.param.value)

    # -- building --------------------------------------------------------

    def schedule_build(self):
        self._token += 1
        token = self._token

        if config.is_surface_only(self.state.field):
            self._build_status.object = (
                f"**{self.state.field} is surface-only** — it has no depth "
                "to section. It works in the region movie above; pick a "
                "depth-resolved field here.")
            return

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

    def _gif_from(self, images):
        """A list of PNG bytes as one animated GIF."""
        import io

        from PIL import Image

        buf = io.BytesIO()
        frames = [Image.open(io.BytesIO(b)).convert("P", palette=Image.ADAPTIVE)
                  for b in images if b]
        if not frames:
            return buf
        frames[0].save(buf, format="GIF", save_all=True,
                       append_images=frames[1:], duration=600, loop=0)
        buf.seek(0)
        return buf

    def _figure_gif(self, kind: str):
        """One movie figure across all steps, as a GIF.

        Each panel gets its own download: the whole point of a per-figure
        movie is to look at one thing evolving, so exporting the set as a
        single sheet would be the wrong artefact.
        """
        return self._gif_from([_read_bytes(f.get(kind)) if f else None
                               for f in self._frames])

    def _movie_gif(self):
        """The region frames as one animated GIF, built on demand.

        On demand rather than with the movie: most builds are looked at
        and not kept, and assembling the GIF costs a second or two that
        nobody should pay unless they want the file.
        """
        return self._gif_from(self._region_bytes)

    def _on_region_step(self, event):
        """Swap the image.  Nothing else -- this runs once per frame.

        Anything expensive here shows up directly as the movie lagging the
        player marker, because Panel cannot deliver the next frame until
        this returns.
        """
        step = int(event.new)
        if self._region_bytes and step < len(self._region_bytes):
            # Bytes, not a path.  A path makes the browser fetch the image
            # on every frame, so the pane is briefly empty each time --
            # which is the flicker.  Bytes are already in the document.
            self._region_pane.object = self._region_bytes[step]
        # The front list should describe the frame on screen, so the step
        # does follow playback -- but nothing watches it any more, so this
        # costs nothing.
        self.state.step = step

    def schedule_region(self):
        self._token += 1
        token = self._token
        self._region_progress.visible = True
        self._region_progress.value = 0
        self._region_progress.max = len(self.state.times())
        self._region_status.object = "building region frames…"
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._build_region(token)
        else:
            pn.state.execute(
                lambda: asyncio.ensure_future(self._build_region_async(token)))

    async def _build_region_async(self, token):
        await asyncio.to_thread(self._build_region, token)

    def _build_region(self, token):
        """The region movie.  No front needed -- this is how you pick one."""
        s = self.state

        def progress(done, total, what=""):
            if token == self._token:
                self._region_progress.max = total
                self._region_progress.value = done
                self._region_status.object = f"{what} ({done}/{total})…"

        try:
            frames = EP.prerender_region(
                s.provider, s.chunk, s.field, track=self._track,
                progress=progress)
        except Exception as exc:                            # noqa: BLE001
            self._region_progress.visible = False
            self._region_status.object = f"**Region movie failed:** {exc}"
            return

        if token != self._token:
            return

        self._region_frames = frames
        self._region_bytes = [_read_bytes(f) for f in frames]
        self.w_download.disabled = not any(self._region_bytes)
        self._region_progress.visible = False
        ok = sum(1 for f in frames if f)
        self._region_status.object = (
            f"**{ok}/{len(frames)}** region frames ready — play to watch the "
            "fronts move, then pick one by its number")
        self.w_region_player.end = max(len(frames) - 1, 1)
        self._on_region_step(type("E", (), {"new": 0})())

    def _build_profile_stack(self, frames):
        """One figure per highlighted step, from the collected columns.

        Pre-rendered rather than redrawn on each step, for the same reason
        the movie is: a matplotlib render inside the player's callback
        shows up directly as the playback stuttering.
        """
        s = self.state
        columns = [f.get("_profile") if f else None for f in frames]
        depths = next((f.get("_Z") for f in frames
                       if f and f.get("_Z") is not None), None)

        if depths is None or not any(c is not None for c in columns):
            self._profile_bytes = []
            self._profile_pane.object = None
            return

        times = s.times()
        out = []
        for step in range(len(columns)):
            try:
                path = F.figure_profile_stack(
                    columns, depths, times, field_name=s.field,
                    highlight=step)
                out.append(pathlib.Path(path).read_bytes())
            except Exception as exc:                        # noqa: BLE001
                log.warning("profile stack step %s: %s", step, exc)
                out.append(None)
        self._profile_bytes = out
        self._show_profile(int(self.w_player.value))

    def _show_profile(self, step: int):
        if self._profile_bytes and step < len(self._profile_bytes):
            self._profile_pane.object = self._profile_bytes[step]

    def _restyle_series(self):
        """Redraw the series only if they have already been computed.

        Toggling which statistic lines are shown must not trigger the
        window walk that computes them.
        """
        if self._series_data is not None:
            self.draw_series()

    async def _build_async(self, token):
        await asyncio.to_thread(self._build, token)

    def _build(self, token):
        s = self.state

        # Follow the picked front first.  Its label is only valid at the
        # step it was picked on, so every other step needs it found by
        # position -- otherwise the build asks for one label at all 17
        # steps and produces nothing.
        try:
            self._track = EP.build_track_at_point(
                s.provider, s.chunk, s.step, s.anchor_lon, s.anchor_lat)
        except Exception as exc:                            # noqa: BLE001
            self._progress.visible = False
            self._build_status.object = (
                f"**Could not follow front {s.front_label}:** {exc}")
            return

        n_steps = len(s.times())
        found = len(self._track.steps())
        gaps = self._track.gaps(n_steps)
        escape = self._track.first_escape()
        note = [f"front **{s.front_label}** followed through "
                f"**{found}/{n_steps}** steps"]
        if gaps:
            note.append(f"{len(gaps)} gap{'s' if len(gaps) > 1 else ''}")
        if escape is not None:
            note.append(f"leaves the window at step {escape}")

        # The weakest joins are where a track most likely jumped to a
        # neighbour.  There is no ground truth to check against, so
        # saying which calls were closest is the honest substitute.
        weak = self._track.weakest(2)
        if weak:
            note.append("weakest links: " + ", ".join(
                f"step {l.step} ({l.score:.2f})" for l in weak))
        print(f"[evolution] {' · '.join(note)}", flush=True)
        self._build_status.object = " · ".join(note)

        def progress(done, total, what=""):
            if token == self._token:
                self._progress.max = total
                self._progress.value = done
                self._build_status.object = f"{what} ({done}/{total})…"

        try:
            frames = EP.prerender(
                s.provider, s.chunk, s.field, self._track,
                n_offsets=s.n_offsets, perp_half_width=s.perp_half_width,
                perp_point=s.perp_point(), progress=progress)
        except Exception as exc:                            # noqa: BLE001
            if token == self._token:
                self._progress.visible = False
                self._build_status.object = f"**Build failed:** {exc}"
            return

        if token != self._token:
            return

        self._frames = frames

        # Series here, not in schedule_build.  schedule_build runs on the
        # server thread, so walking the window there froze the whole app
        # before a single frame could be drawn -- and prerender above has
        # just walked the same window, so doing it there paid for it twice.
        for kind, widget in self._downloads.items():
            widget.disabled = not any(f and f.get(kind) for f in frames)

        self._build_status.object = "building the profile stack…"
        self._build_profile_stack(frames)

        self._build_status.object = "building time series…"
        try:
            self.draw_series()
        except Exception as exc:                            # noqa: BLE001
            log.warning("series failed: %s", exc)

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
        # Only the figure keys.  A frame also carries the profile column
        # and the depth axis under underscored keys -- data, not pictures.
        for kind, pane in self._panes.items():
            if not kind.startswith("_"):
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
        row_b2 = pn.Row(self.w_offsets, self.w_perp,
                        self.w_stats,
                        sizing_mode="stretch_width", margin=(0, 10))
        row_b_go = pn.Row(
            pn.pane.Markdown("**b · field → region movie**",
                             margin=(8, 5, 0, 10)),
            self.w_region, self._region_progress,
            sizing_mode="stretch_width", margin=(0, 10))

        row_c_go = pn.Row(
            pn.pane.Markdown("**c · front → sections + time series**",
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

        # One column, each figure the full width of the page.  Three
        # across made every curtain a thumbnail.
        frames = pn.Column(*[
            pn.Column(
                pn.Row(pn.pane.Markdown(f"**{EP.FRAME_TITLES[k]}**",
                                        margin=(10, 5, 0, 5)),
                       self._downloads[k],
                       sizing_mode="stretch_width"),
                self._panes[k],
                sizing_mode="stretch_width")
            for k in EP.FRAME_ORDER
        ], sizing_mode="stretch_width")

        body = pn.Column(
            pn.pane.Markdown("### Evolution — one region over a week",
                             margin=(4, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>Three stages. <b>(a)</b> load a chunk. <b>(b)</b> "
                "build the region movie: the whole surface with every front "
                "drawn and numbered, so you can watch how the fronts move "
                "and how their numbers change before choosing one — labels "
                "are assigned per timestep, so the same front is called "
                "something different in every frame. <b>(c)</b> pick a front "
                "by its number and build the sections; it is then followed "
                "by position, not by label. The cadence is not uniform — "
                "daily snapshots around one intensive day — which is why "
                "every frame carries its datestamp.</small>",
                margin=(0, 10)),
            row_a, row_a_go, self._status, maps,
            pn.layout.Divider(),
            # (b) the whole region, every step, fronts numbered.  This is
            # the picture you choose a front from, so it comes before every
            # control that needs one.
            pn.pane.Markdown("### Region movie — every front, numbered",
                             margin=(6, 10, 0, 10)),
            pn.Row(self.w_field, sizing_mode="stretch_width",
                   margin=(0, 10)),
            row_b_go, self._region_status,
            pn.Row(self.w_region_player, self.w_download, margin=(0, 10)),
            self._region_pane,
            pn.layout.Divider(),
            # (c) now that a front can be seen and named, choose one.
            pn.pane.Markdown("### Front → sections", margin=(6, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>Click the map above to set the point, or type it. "
                "The front nearest that point is followed — labels change "
                "every timestep, a place does not.</small>",
                margin=(0, 10)),
            pn.Row(self.w_lat, self.w_lon, self.w_resolve,
                   sizing_mode="stretch_width", margin=(0, 10)),
            pn.pane.Markdown(
                "<small>The cross-front transect sits at a fixed place too, "
                "so the front is seen moving through one point rather than "
                "the transect sliding along it. Left blank it follows the "
                "front point above. Profile locations are fixed the same "
                "way — one line per point, at every step.</small>",
                margin=(6, 10, 0, 10)),
            self._preview_status,
            self._preview,
            pn.Row(self.w_perp_lat, self.w_perp_lon,
                   sizing_mode="stretch_width", margin=(0, 10)),
            row_b2, row_c_go,
            pn.layout.Divider(),
            pn.pane.Markdown("### Time series", margin=(6, 10, 0, 10)),
            self._series,
            pn.pane.Markdown("**Vertical profile at the transect point — "
                             "every timestep**", margin=(12, 10, 0, 10)),
            self._profile_pane,
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
