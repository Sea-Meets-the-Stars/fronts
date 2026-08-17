"""The six figures on page 2.

Five of them come from ``fronts.viz.curtains``; the sixth is the 3-D scene
from ``fronts.viz.fronts_3d``.  Both are the repo's existing code, called
with no modifications.

The curtain builders write a PNG and close the figure, so they cannot hand
back something a ``Matplotlib`` pane could take.  Here they render into a
temporary directory and the page shows the PNG.  Prerequisite **R4** in
``prompts/build_front_viz_tool.md`` -- making ``output_path`` optional so
the builders can return the ``Figure`` -- would remove that round-trip;
until then this keeps the prototype free of changes to core modules.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from fronts.viz import curtains
from fronts.viz.apps.common.render import MPL_LOCK
from fronts.viz.apps.tiles.pipeline import FrontScene

#: Figure keys, in the order the page lays them out.
FIGURE_ORDER = ("inset", "isopycnal", "mainaxis", "offsets", "perpendicular")

FIGURE_TITLES = {
    "inset": "(b) inset — plan view",
    "isopycnal": "(c) isopycnal surface",
    "mainaxis": "(d) main-axis curtain",
    "offsets": "(e) along-front offsets",
    "perpendicular": "(f) cross-front transect",
}


class Missing3DStack(RuntimeError):
    """PyVista is not installed, so figure (a) cannot be built.

    Raised instead of letting ``ImportError`` escape, so the page can say
    what to install and carry on rendering figures (b)-(f).
    """


def _outdir() -> Path:
    d = Path(tempfile.gettempdir()) / "fronts-viz-panels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(scene: FrontScene, kind: str) -> Path:
    return _outdir() / f"{kind}_{scene.field_name}_lbl{scene.label}.png"


# --------------------------------------------------------------------------
# Curtains
# --------------------------------------------------------------------------

def figure_mainaxis(scene: FrontScene, *, perp_index=None) -> Path:
    with MPL_LOCK:
        out = _stem(scene, "mainaxis")
        return curtains.figure_main_axis(
            scene.color, scene.sigma0, scene.Z, scene.axis_path, scene.metrics,
            out,
            levels=scene.levels, clim=scene.clim, cmap=scene.style.cmap,
            color_title=scene.style.title, mld_curtain=scene.mld_curtain,
            mark_index=perp_index,
            title=f"Main axis — front {scene.label}",
        )


def figure_offsets(scene: FrontScene, *, n_offsets=3) -> Path:
    with MPL_LOCK:
        out = _stem(scene, "offsets")
        return curtains.figure_offsets(
            scene.color, scene.sigma0, scene.Z, scene.axis_path, scene.metrics,
            n_offsets, out,
            levels=scene.levels, clim=scene.clim, cmap=scene.style.cmap,
            color_title=scene.style.title,
            title=f"Offsets — front {scene.label}",
        )


def figure_perpendicular(scene: FrontScene, *, index, half_width=30) -> Path:
    with MPL_LOCK:
        out = _stem(scene, "perpendicular")
        perp_path = curtains.perpendicular_path(
            scene.axis_path, scene.metrics["normals"], index, half_width
        )
        return curtains.figure_perpendicular(
            scene.color, scene.sigma0, scene.Z, perp_path, half_width, out,
            XC_rect=scene.XC, YC_rect=scene.YC,
            levels=scene.levels, clim=scene.clim, cmap=scene.style.cmap,
            color_title=scene.style.title,
            title=f"Perpendicular — front {scene.label}",
        )


def figure_isopycnal(scene: FrontScene, *, perp_index=None) -> Path:
    with MPL_LOCK:
        out = _stem(scene, "isopycnal")
        return curtains.figure_isopycnal_surface(
            scene.color, scene.sigma0, scene.Z, scene.axis_path, scene.metrics,
            out,
            clim=scene.clim, cmap=scene.style.cmap,
            color_title=scene.style.title, mark_index=perp_index,
            title=f"Isopycnal surface — front {scene.label}",
        )


def figure_inset(scene: FrontScene, *, perp_index=None, half_width=30) -> Path:
    """Plan view of the crop, with the axis and the transect marked.

    ``plot_map_inset`` lives inside ``fronts/scripts/fronts_viz_curtain.py``
    today, and importing that module runs a ``sys.path`` hack and calls
    ``matplotlib.use('Agg')`` at import time -- neither of which belongs in
    a server process.  Prerequisite **R2/R3** moves it into
    ``fronts/viz/map_inset.py``; until then the page draws the equivalent
    view here.
    """
    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        out = _stem(scene, "inset")
        surf = scene.color[0]

        fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=130)
        im = ax.pcolormesh(surf, cmap=scene.style.cmap,
                           vmin=scene.clim[0], vmax=scene.clim[1],
                           shading="nearest")
        fig.colorbar(im, ax=ax, label=scene.style.title, shrink=0.85)

        path = scene.axis_path
        ax.plot(path[:, 1], path[:, 0], "-", color="white", lw=1.8)
        ax.plot(path[:, 1], path[:, 0], "-", color="black", lw=0.8)

        normals = scene.metrics.get("normals")
        if normals is not None:
            for k in (1, 2, 3):
                for sgn in (+1, -1):
                    off = path + sgn * k * normals
                    ax.plot(off[:, 1], off[:, 0], "-", color="0.85", lw=0.5,
                            alpha=0.8)

        if perp_index is not None and normals is not None:
            j0, i0 = path[perp_index]
            nj, ni = normals[perp_index]
            t = np.linspace(-half_width, half_width, 2)
            ax.plot(i0 + t * ni, j0 + t * nj, "-", color="lime", lw=2.0)
            ax.plot([i0], [j0], "o", color="lime", ms=4)

        ax.set_xlabel("i (tile pixels)")
        ax.set_ylabel("j (tile pixels)")
        ax.set_title(f"Plan view — front {scene.label}", fontsize=10)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out


# --------------------------------------------------------------------------
# Perpendicular point
# --------------------------------------------------------------------------

def pick_perp_index(scene: FrontScene, *, half_width=30, mode="min") -> int:
    """Where to cut the cross-front transect.

    The field extremum over the full curtain depth, restricted to columns
    whose transect crosses the front at most once -- so the cut lands on a
    clean stretch rather than a self-overlapping hook.  Falls back to the
    unrestricted extremum, then to the midpoint.
    """
    axis_curtain = curtains.sample_curtain(scene.color, scene.axis_path)
    try:
        crossings = curtains.transect_front_crossings(
            scene.axis_path, scene.metrics["normals"], scene.front_mask,
            half_width,
        )
        allowed = np.nonzero(np.asarray(crossings) <= 1)[0]
    except Exception:                                   # noqa: BLE001
        allowed = np.array([])

    if allowed.size:
        sub = axis_curtain[:, allowed]
        with np.errstate(invalid="ignore"):
            per_col = np.nanmin(sub, axis=0) if mode == "min" \
                else np.nanmax(sub, axis=0)
        if np.any(np.isfinite(per_col)):
            pick = np.nanargmin(per_col) if mode == "min" \
                else np.nanargmax(per_col)
            return int(allowed[int(pick)])

    try:
        return int(curtains.pick_extremum_index(axis_curtain, mode))
    except Exception:                                   # noqa: BLE001
        return int(len(scene.axis_path) // 2)


# --------------------------------------------------------------------------
# 3-D
# --------------------------------------------------------------------------

def auto_zscale(scene: FrontScene, *, aspect: float = 0.55) -> float:
    """Vertical exaggeration that makes the scene readable.

    ``render_3d`` defaults to 50, which suits a deep volume in a wide
    tile.  A front clipped a few levels below a shallow mixed layer can be
    under 100 m deep in a window hundreds of pixels across, and at 50x the
    scene renders as an unreadable spike.  This picks the factor that puts
    the depth extent at *aspect* times the horizontal extent.
    """
    span_j = scene.j_slice.stop - scene.j_slice.start
    span_i = scene.i_slice.stop - scene.i_slice.start
    horizontal = max(span_j, span_i)

    z = np.asarray(scene.Z, dtype=float)
    depth = float(np.nanmax(z) - np.nanmin(z))
    if not np.isfinite(depth) or depth <= 0:
        return 1.0
    return float(np.clip(aspect * horizontal / depth, 0.05, 200.0))


def build_3d(scene: FrontScene, *, zscale: float | None = None,
             n_levels: int = 5):
    """The 3-D scene: the colour field painted on the front's isopycnals.

    Returns a configured ``pyvista.Plotter``, which is what
    ``pn.pane.VTK`` wants.  Raises on a missing GL backend, which the page
    turns into a message rather than a broken layout -- the five 2-D
    figures are matplotlib only and stay available either way.
    """
    try:
        from fronts.viz import fronts_3d
    except ImportError as exc:                          # pragma: no cover
        raise Missing3DStack(
            "The 3-D scene needs PyVista, which is not installed.  The five "
            "2-D figures below are matplotlib only and are unaffected.\n"
            "  pip install 'pyvista[jupyter]'\n"
            "For off-screen rendering on a headless machine, see step 0 of "
            "docs/viz/fronts_viz_3d_runbook.md."
        ) from exc

    # PyVista >= 0.44 dropped ``start_xvfb``, which ``pv_helpers.ensure_display``
    # still calls when $DISPLAY is empty.  Any non-empty value makes it skip
    # that branch; OSMesa ignores DISPLAY entirely.  Same workaround as step 1
    # of docs/viz/fronts_viz_3d_runbook.md.
    import os
    os.environ.setdefault("DISPLAY", "dummy")
    if not os.environ["DISPLAY"]:
        os.environ["DISPLAY"] = "dummy"

    if zscale is None:
        zscale = auto_zscale(scene)

    grid = fronts_3d.build_pyvista_grid(
        scene.sigma0, scene.Z, scene.j_slice, scene.i_slice, zscale=zscale,
        extra_fields={scene.field_name: scene.color},
    )
    curtain = fronts_3d.build_front_curtain(
        scene.front_mask, scene.sigma0, scene.Z,
        scene.j_slice, scene.i_slice, zscale=zscale,
    )
    levels = fronts_3d.pick_isopycnals_across_front(
        scene.sigma0, scene.front_mask, scene.Z, None, n_levels=n_levels,
    )
    top = fronts_3d.build_front_top_marker(
        curtain, scene.Z, scene.j_slice, scene.i_slice, zscale=zscale,
    )
    iso = None
    if len(levels):
        iso = fronts_3d.build_front_isosurface(grid, float(np.median(levels)))

    return fronts_3d.render_3d(
        grid, curtain, levels,
        mode="isopycnals",
        clim=scene.clim,
        cmap_volume=scene.style.cmap,
        zscale=zscale,
        top_marker=top,
        front_iso=iso,
        color_scalar=scene.field_name,
        color_title=scene.style.title,
        # render_3d's defaults (56/60/44) are sized for a poster-scale
        # screenshot; in a browser pane they overlap the axes.
        font_size=14, title_font_size=15, label_font_size=11,
    )
