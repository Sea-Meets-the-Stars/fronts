"""Frame building for the Evolution page.

One chunk step is one movie frame. A step is shaped exactly like a tile, so
this reuses the Tiles ingest pipeline verbatim — the only difference is
where the data comes from and that the result is rendered to a **fixed
camera**.

Frames are pre-rendered to PNG and cached on disk. Playback then costs
nothing but swapping an image, which is what makes a 24-step movie
watchable; rendering on demand would stutter at roughly one frame per
second.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from fronts.viz.apps import config
from fronts.viz.apps.tiles import panels as F
from fronts.viz.apps.tiles import pipeline as TP

#: Figure keys in the order the page lays them out.
#:
#: No 3-D frame.  A fixed-camera still was the slowest render on the page
#: by a wide margin -- roughly as expensive as the other five together --
#: and a rotating-free volume is the least readable of them as a movie.
#: The 3-D scene stays on the Tiles page, where it is interactive and
#: built once rather than per step.
FRAME_ORDER = F.FIGURE_ORDER

FRAME_TITLES = {
    "inset": "(d) inset — plan view",
    "isopycnal": "(e) isopycnal surface",
    "mainaxis": "(f) main-axis curtain",
    "offsets": "(g) along-front offsets",
    "perpendicular": "(h) cross-front transect",
}


def frame_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "fronts-viz-frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(chunk, label, field, step, kind, extra="") -> Path:
    raw = f"{chunk}|{label}|{field}|{step}|{kind}|{extra}"
    stem = hashlib.sha1(raw.encode()).hexdigest()[:18]
    return frame_dir() / f"{kind}_{stem}.png"


class ChunkStep:
    """Adapts a chunk step so the tile pipeline can consume it.

    ``pipeline.build_scene`` asks a provider for ``tile(date, tile, prop)``
    and ``tile_labels(...)``.  A chunk step answers both, so the scene
    builder needs no changes at all.
    """

    def __init__(self, provider, chunk: str, step: int):
        self.provider = provider
        self.chunk = chunk
        self.step = int(step)
        self.mode = provider.mode
        self.synthetic = provider.synthetic

    def tile(self, date, tile_idx, prop, region=None):
        # A chunk is its own store, so there is no tile-store slot to
        # consult -- region is accepted and ignored.
        return self.provider.chunk_tile(self.chunk, self.step, prop)

    def labels_for_step(self):
        return self.provider.chunk_labels(self.chunk, self.step)


def build_track(provider, chunk: str, step: int, label: int):
    """Follow the front picked at *step* through the whole window.

    The label is only valid at the step it was picked on -- labels are
    assigned per date -- so every other step needs the front found by
    position.  Without this the movie asks for one label at all 17 steps
    and gets nothing: "label 96277 is absent from step 16".
    """
    from fronts.viz.apps.evolution import tracking

    times = provider.chunk_timesteps(chunk)
    labels = provider.chunk_labels(chunk, step)
    anchor = tracking.anchor_at(labels, step, int(label))
    return tracking.follow(lambda s: provider.chunk_labels(chunk, s),
                           times, anchor)


def build_track_at_point(provider, chunk: str, step: int,
                         point_lon: float, point_lat: float):
    """Follow whichever front is at a geographic point.

    The selection that survives re-labelling.  ``build_track`` needs a
    label, which only identifies a front in the step it came from; a point
    on the ocean is the same point in every step, so it is resolved to a
    label per step instead of being one.
    """
    from fronts.viz.apps.evolution import tracking

    times = provider.chunk_timesteps(chunk)
    _surface, lon, lat, labels, _var = chunk_plane(
        provider, chunk, step, config.TILE_GEOMETRY_FIELD)

    anchor, km = tracking.anchor_at_point(
        labels, lon, lat, point_lon, point_lat, step)
    print(f"[evolution] point ({point_lat:.3f}, {point_lon:.3f}) -> front "
          f"{anchor.label} at step {step} ({km:.1f} km away)", flush=True)

    return tracking.follow(lambda s: provider.chunk_labels(chunk, s),
                           times, anchor)


def build_step(provider, chunk: str, step: int, field: str, label: int):
    """Run the tile ingest pipeline on one chunk step.

    Returns the same :class:`~fronts.viz.apps.tiles.pipeline.FrontScene`
    the Tiles page uses, so every downstream figure builder applies.
    """
    adapter = ChunkStep(provider, chunk, step)
    labels = adapter.labels_for_step()

    if label <= 0 or not np.any(labels == label):
        raise TP.NoSuchFront(
            f"label {label} is absent from {chunk} step {step}")

    # Patch the label lookup for this call only.  ``build_scene`` reads
    # labels through ``tile_labels``, which is global-store shaped; a chunk
    # is already the window, so it is returned as-is.
    original = TP.tile_labels
    try:
        TP.tile_labels = lambda *a, **k: labels
        return TP.build_scene(adapter, "", 0, field, label)
    finally:
        TP.tile_labels = original


def render_frame(provider, chunk: str, step: int, field: str, label: int, *,
                 n_offsets: int = 3, perp_half_width: int = 30,
                 perp_index: int | None = None,
                 clim: tuple[float, float] | None = None,
                 xmax: float | None = None,
                 use_cache: bool = True) -> dict:
    """Render every figure for one step.  Returns ``{kind: path}``.

    Parameters
    ----------
    perp_index : int, optional
        Force the cross-front transect to a fixed column.  Passing the
        same value for every step keeps the transect in one place as the
        movie runs; letting each step pick its own extremum makes figure
        (i) jump around and the movie unreadable.
    clim : tuple, optional
        Shared colour limits.  Without this every frame rescales to its
        own range and the movie appears to pulse even where the field is
        steady.
    """
    extra = f"{n_offsets}|{perp_half_width}|{perp_index}|{clim}|{xmax}"
    paths = {k: _key(chunk, label, field, step, k, extra)
             for k in FRAME_ORDER}

    if use_cache and all(p.exists() for p in paths.values()):
        return {k: str(p) for k, p in paths.items()}

    scene = build_step(provider, chunk, step, field, label)
    if clim is not None:
        scene.clim = clim

    idx = perp_index
    if idx is None:
        idx = F.pick_perp_index(scene, half_width=perp_half_width)
    idx = int(np.clip(idx, 0, len(scene.axis_path) - 1))

    builders = {
        "inset": lambda: F.figure_inset(scene, perp_index=idx,
                                        half_width=perp_half_width),
        # The three along-front figures share one x extent, so the front
        # grows and shrinks inside a fixed axis instead of the axis
        # rescaling under it every frame.
        "isopycnal": lambda: F.figure_isopycnal(scene, perp_index=idx,
                                                xmax=xmax),
        "mainaxis": lambda: F.figure_mainaxis(scene, perp_index=idx,
                                              xmax=xmax),
        "offsets": lambda: F.figure_offsets(scene, n_offsets=n_offsets,
                                            xmax=xmax),
        "perpendicular": lambda: F.figure_perpendicular(
            scene, index=idx, half_width=perp_half_width),
    }

    out = {}
    for kind, build in builders.items():
        try:
            produced = Path(build())
            produced.replace(paths[kind])
            out[kind] = str(paths[kind])
        except Exception:                                   # noqa: BLE001
            out[kind] = None

    return out


def shared_settings(provider, chunk: str, field: str, track, *,
                    perp_half_width: int = 30, progress=None) -> dict:
    """Choices that must be the same in every frame.

    A fixed transect column and shared colour limits, so that what changes
    between frames is the data and nothing else.

    Every scene here costs a tile composition from raw fields -- the most
    expensive thing on the page -- so this samples **three** steps, not
    seven, and reuses the scene it already built for the transect instead
    of rebuilding it.  It also reports progress: this used to run before
    ``prerender``'s loop with no feedback at all, so the first eight
    compositions of a build were indistinguishable from a hang.
    """
    found = track.steps()
    if not found:
        return {"perp_index": None, "clim": None, "xmax": None}

    # Sample steps the front is actually present in -- picked from the
    # track, not from the window, so a step where it was never found is
    # never asked for.
    mid = found[len(found) // 2]
    wanted = [s for s in dict.fromkeys((mid, found[0], found[-1]))]

    scenes = {}
    for k, step in enumerate(wanted, start=1):
        if progress is not None:
            progress(k, len(wanted), f"sampling step {step}")
        try:
            scenes[step] = build_step(provider, chunk, step, field,
                                      track.label_at(step))
        except Exception as exc:                            # noqa: BLE001
            # Say so.  Swallowing these hid a failure that killed every
            # step of every chunk: the sampling looked fine and only the
            # frame loop reported anything.
            print(f"[evolution]   sampling step {step} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue

    if not scenes:
        return {"perp_index": None, "clim": None, "xmax": None}

    # The transect comes from the middle of the window where the front is
    # most fully developed, falling back to whatever else we have.
    anchor = scenes.get(mid) or next(iter(scenes.values()))
    idx = F.pick_perp_index(anchor, half_width=perp_half_width)

    # A shared x extent as well as shared colour limits.  Without it every
    # frame's along-front axis rescales to its own front length, so the
    # front appears the same size in every frame and only the axis numbers
    # change -- which reads as the figure jumping rather than the front
    # growing.  Fixing the axis puts the change where it belongs.
    #
    # The margin covers steps that were not sampled; plot_curtain_panel
    # only ever *extends* from this value, so an underestimate costs
    # consistency on one frame rather than clipping it.
    xmax = None
    lengths = [float(sc.metrics["dist_px"][-1]) for sc in scenes.values()
               if len(sc.metrics.get("dist_px", []))]
    if lengths:
        xmax = max(lengths) * 1.35

    samples = []
    for scene in scenes.values():
        finite = scene.color[np.isfinite(scene.color)]
        if finite.size:
            samples.append(np.percentile(finite, [2, 98]))

    clim = None
    if samples:
        arr = np.array(samples)
        clim = (float(arr[:, 0].min()), float(arr[:, 1].max()))
        if clim[0] >= clim[1]:
            clim = None

    return {"perp_index": int(idx), "clim": clim, "xmax": xmax}


def prerender(provider, chunk: str, field: str, track, *,
              n_offsets: int = 3, perp_half_width: int = 30,
              progress=None) -> list[dict]:
    """Render every frame of the movie.  Returns one dict per step.

    *progress* is called as ``(done, total, what)`` and covers the whole
    job -- the shared-settings sampling included.  Leaving that prelude
    out of the count is what made a build look hung for its first minute.
    """
    times = provider.chunk_timesteps(chunk)
    n = len(times)
    n_prep = min(3, max(len(track.steps()), 1))
    total = n_prep + n

    def report(done, what):
        print(f"[evolution] {done}/{total} {what}", flush=True)
        if progress is not None:
            progress(done, total, what)

    shared = shared_settings(
        provider, chunk, field, track, perp_half_width=perp_half_width,
        progress=lambda k, _tot, what: report(k, what))

    frames = []
    for step in range(n):
        report(n_prep + step + 1, f"rendering frame {step + 1}/{n}")

        label = track.label_at(step)
        if label is None:
            # A gap: the front was not found here.  A blank frame is the
            # honest answer -- better than rendering a different front.
            print(f"[evolution]   step {step}: no front (tracking gap)",
                  flush=True)
            frames.append({k: None for k in FRAME_ORDER})
            continue

        try:
            frames.append(render_frame(
                provider, chunk, step, field, label,
                n_offsets=n_offsets, perp_half_width=perp_half_width,
                perp_index=shared["perp_index"], clim=shared["clim"],
                xmax=shared.get("xmax")))
        except Exception as exc:                            # noqa: BLE001
            print(f"[evolution]   step {step} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            frames.append({k: None for k in FRAME_ORDER})
    return frames


# --------------------------------------------------------------------------
# Stage (b): the whole region, every step, before any front is chosen
# --------------------------------------------------------------------------

def chunk_plane(provider, chunk: str, step: int, field: str):
    """One chunk step in the **rect** frame: surface, lon, lat, labels.

    A chunk is stored face-local.  On LLC faces 7-12 -- which includes
    face 10, the California Current -- the face axes are rotated ~90
    degrees from east/north, so face-local data drawn under rect-frame
    front labels is visibly wrong: the fronts sit across the features
    instead of along them.

    Two different rotations are involved and only one of them is our job:

    * **Vector components** are already handled upstream.  The tile
      registry computes ``U``/``V`` through ``CF.geographic_velocity``,
      documented as "tracer points, CS/SN rotated", so the numbers are
      already eastward/northward before they reach us.
    * **Positions** are what remain, and that is this function.  It is a
      pure index shuffle -- the same one for every field, scalar or
      vector -- so nothing here needs to know which kind it has.

    Everything that draws a chunk goes through this, because the bug it
    fixes was two callers doing the remap differently: one did it, one
    did not, and the mismatch only showed up as fronts lying beside the
    gradients rather than on them.
    """
    ds = provider.chunk_tile(chunk, step, field)
    var = ds.attrs.get("tile_var_name") or TP._sole_3d(ds)

    try:
        lookup = TP.tile_lookup(ds, synthetic=provider.synthetic)
    except Exception as exc:                                # noqa: BLE001
        print(f"[evolution]   step {step}: NO FACE REMAP ({exc}) -- the "
              "fronts will not line up with the field", flush=True)
        lookup = None

    plane = TP.remap_to_rect(TP.field_values(ds, var), lookup)
    surface = plane[0] if plane.ndim == 3 else plane
    lon = TP.remap_to_rect(TP.field_values(ds, "XC"), lookup) % 360.0
    lat = TP.remap_to_rect(TP.field_values(ds, "YC"), lookup)
    labels = provider.chunk_labels(chunk, step)
    return surface, lon, lat, labels, var


def region_frame(provider, chunk: str, step: int, field: str, *,
                 selected: int = 0, use_cache: bool = True) -> str | None:
    """One frame of the region movie: the field with every front numbered.

    Deliberately **not** built on ``build_scene``.  A scene needs a chosen
    front, a crop and a mixed-layer clip, and costs two tile compositions;
    this needs one, and no choice at all.  That is the whole point -- it is
    the picture you look at in order to choose.
    """
    times = provider.chunk_timesteps(chunk)
    date = times[int(step)]
    # "noann" is part of the key on purpose: frames built before the
    # numbers were dropped are still on disk, and a key that ignored the
    # difference would keep serving them.
    path = _key(chunk, selected, field, step, "region", extra="noann")
    if use_cache and path.exists():
        return str(path)

    surface, lon, lat, labels, _ = chunk_plane(provider, chunk, step, field)

    n = len(times)
    produced = F.figure_region_fronts(
        surface, labels, lon=lon, lat=lat, field_name=field,
        selected=selected, out=path,
        # Drawn, not numbered.  Seventeen frames of number chips is
        # unreadable as a movie, and the numbers are in the ordered
        # dropdown where they can actually be found.
        annotate=False,
        # The datestamp goes on every frame: the cadence is not uniform --
        # daily snapshots around one intensive day -- so without it there
        # is no way to see that the interval just changed.
        title=f"{chunk}  ·  {date}  ·  step {step + 1}/{n}")
    return str(produced)


def prerender_region(provider, chunk: str, field: str, *, track=None,
                     progress=None) -> list[str | None]:
    """The region movie: one frame per step, no front selection needed.

    *track* only decides which front is drawn red.  Without one every front
    is cyan, which is the honest picture before anything is chosen.
    """
    times = provider.chunk_timesteps(chunk)
    total = len(times)
    frames = []
    for step in range(total):
        what = f"region frame {step + 1}/{total}"
        print(f"[evolution] {step + 1}/{total} {what}", flush=True)
        if progress is not None:
            progress(step + 1, total, what)
        selected = track.label_at(step) if track is not None else 0
        try:
            frames.append(region_frame(provider, chunk, step, field,
                                       selected=int(selected or 0)))
        except Exception as exc:                            # noqa: BLE001
            print(f"[evolution]   step {step} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            frames.append(None)
    return frames
