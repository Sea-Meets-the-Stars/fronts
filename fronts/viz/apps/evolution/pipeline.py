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

#: Figure keys in the order the page lays them out, after the three time
#: series.  The 3-D scene is a still image here, not an interactive pane.
FRAME_ORDER = ("scene3d",) + F.FIGURE_ORDER

FRAME_TITLES = {
    "scene3d": "(d) 3-D field on the front's isopycnals",
    "inset": "(e) inset — plan view",
    "isopycnal": "(f) isopycnal surface",
    "mainaxis": "(g) main-axis curtain",
    "offsets": "(h) along-front offsets",
    "perpendicular": "(i) cross-front transect",
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

    def tile(self, date, tile_idx, prop):
        return self.provider.chunk_tile(self.chunk, self.step, prop)

    def labels_for_step(self):
        return self.provider.chunk_labels(self.chunk, self.step)


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
    extra = f"{n_offsets}|{perp_half_width}|{perp_index}|{clim}"
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
        "isopycnal": lambda: F.figure_isopycnal(scene, perp_index=idx),
        "mainaxis": lambda: F.figure_mainaxis(scene, perp_index=idx),
        "offsets": lambda: F.figure_offsets(scene, n_offsets=n_offsets),
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

    out["scene3d"] = _render_3d_still(scene, paths["scene3d"])
    return out


def _render_3d_still(scene, path: Path):
    """The 3-D scene as a still, from a fixed camera.

    The camera is pinned deliberately.  During playback the data is
    already moving; a camera the user can also move makes it impossible to
    tell whether a change on screen is the ocean or the viewpoint.  Every
    frame therefore uses ``config.EVOLUTION_CAMERA``.
    """
    try:
        plotter = F.build_3d(scene)
    except Exception:                                       # noqa: BLE001
        return None

    try:
        cam = config.EVOLUTION_CAMERA
        plotter.camera_position = "iso"
        plotter.camera.azimuth = cam["azimuth"]
        plotter.camera.elevation = cam["elevation"]
        plotter.camera.zoom(cam["zoom"])
        plotter.window_size = (760, 560)
        plotter.screenshot(str(path))
        return str(path)
    except Exception:                                       # noqa: BLE001
        return None
    finally:
        try:
            plotter.close()
        except Exception:                                   # noqa: BLE001
            pass


def shared_settings(provider, chunk: str, field: str, label: int, *,
                    perp_half_width: int = 30) -> dict:
    """Choices that must be the same in every frame.

    Picked from the middle of the window, where the front is most fully
    developed, then applied to all steps: a fixed transect column and
    shared colour limits.  Both exist so that what changes between frames
    is the data and nothing else.
    """
    times = provider.chunk_timesteps(chunk)
    mid = len(times) // 2

    scene = None
    for step in (mid, 0, len(times) - 1):
        try:
            scene = build_step(provider, chunk, step, field, label)
            break
        except Exception:                                   # noqa: BLE001
            continue
    if scene is None:
        return {"perp_index": None, "clim": None}

    idx = F.pick_perp_index(scene, half_width=perp_half_width)

    # Colour limits pooled across a few steps, so no frame clips.
    samples = []
    for step in range(0, len(times), max(len(times) // 6, 1)):
        try:
            s = build_step(provider, chunk, step, field, label)
        except Exception:                                   # noqa: BLE001
            continue
        finite = s.color[np.isfinite(s.color)]
        if finite.size:
            samples.append(np.percentile(finite, [2, 98]))

    clim = None
    if samples:
        arr = np.array(samples)
        clim = (float(arr[:, 0].min()), float(arr[:, 1].max()))
        if clim[0] >= clim[1]:
            clim = None

    return {"perp_index": int(idx), "clim": clim}


def prerender(provider, chunk: str, field: str, label: int, *,
              n_offsets: int = 3, perp_half_width: int = 30,
              progress=None) -> list[dict]:
    """Render every frame of the movie.  Returns one dict per step."""
    shared = shared_settings(provider, chunk, field, label,
                             perp_half_width=perp_half_width)
    times = provider.chunk_timesteps(chunk)

    frames = []
    for step in range(len(times)):
        try:
            frames.append(render_frame(
                provider, chunk, step, field, label,
                n_offsets=n_offsets, perp_half_width=perp_half_width,
                perp_index=shared["perp_index"], clim=shared["clim"]))
        except Exception:                                   # noqa: BLE001
            frames.append({k: None for k in FRAME_ORDER})
        if progress is not None:
            progress(step + 1, len(times))
    return frames
