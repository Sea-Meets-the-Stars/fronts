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

from matplotlib import colors

from fronts.viz import curtains, field_styles
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

def figure_mainaxis(scene: FrontScene, *, perp_index=None, xmax: float | None = None) -> Path:
    with MPL_LOCK:
        out = _stem(scene, "mainaxis")
        return curtains.figure_main_axis(
            scene.color, scene.sigma0, scene.Z, scene.axis_path, scene.metrics,
            out,
            levels=scene.levels, clim=scene.clim, cmap=field_styles.resolve_cmap(scene.style.cmap),
            color_title=scene.style.title, mld_curtain=scene.mld_curtain,
            mark_index=perp_index, xmax=xmax,
            title=f"Main axis — front {scene.label}",
        )


def figure_offsets(scene: FrontScene, *, n_offsets=3, xmax: float | None = None) -> Path:
    with MPL_LOCK:
        out = _stem(scene, "offsets")
        return curtains.figure_offsets(
            scene.color, scene.sigma0, scene.Z, scene.axis_path, scene.metrics,
            n_offsets, out,
            levels=scene.levels, clim=scene.clim, cmap=field_styles.resolve_cmap(scene.style.cmap),
            color_title=scene.style.title, xmax=xmax,
            title=f"Offsets — front {scene.label}",
        )


def figure_perpendicular(scene: FrontScene, *, index, half_width=30) -> Path:
    with MPL_LOCK:
        out = _stem(scene, "perpendicular")
        perp_path = curtains.perpendicular_path(
            scene.axis_path, scene.metrics["normals"], index, half_width
        )
        # The mixed layer along *this* path, not along the main axis --
        # the perpendicular crosses the axis, so the along-axis curtain
        # cannot answer for it.
        mld = np.array(
            [scene.mld_field[int(np.clip(j, 0, scene.mld_field.shape[0] - 1)),
                             int(np.clip(i, 0, scene.mld_field.shape[1] - 1))]
             for j, i in perp_path], dtype=float)

        return curtains.figure_perpendicular(
            scene.color, scene.sigma0, scene.Z, perp_path, half_width, out,
            XC_rect=scene.XC, YC_rect=scene.YC, mld_curtain=mld,
            levels=scene.levels, clim=scene.clim, cmap=field_styles.resolve_cmap(scene.style.cmap),
            color_title=scene.style.title,
            title=f"Perpendicular — front {scene.label}",
        )


def figure_isopycnal(scene: FrontScene, *, perp_index=None,
                     xmax: float | None = None) -> Path:
    """(c) -- the isopycnal surface, on a depth axis.

    An along-surface-length version was tried and dropped: measuring the
    surface in its own coordinates is defensible, but it breaks the
    one-for-one correspondence with the 3-D scene -- a point at depth z
    there is no longer at depth z here -- and that correspondence turns
    out to be worth more than equal distances on the plot.
    """
    with MPL_LOCK:
        out = _stem(scene, "isopycnal")
        return curtains.figure_isopycnal_surface(
            scene.color, scene.sigma0, scene.Z, scene.axis_path, scene.metrics,
            out,
            clim=scene.clim, cmap=field_styles.resolve_cmap(scene.style.cmap),
            color_title=scene.style.title, mark_index=perp_index, xmax=xmax,
            title=f"Isopycnal surface — front {scene.label}",
        )

def figure_inset(scene: FrontScene, *, perp_index=None, half_width=30,
                 depth=None) -> Path:
    """Plan view of the crop: the surface, and a second row at *depth*.

    Two rows so the front can be read at the surface and at a chosen
    level at once -- the same axis, offsets and transect drawn on both, so
    the comparison is like for like.  ``depth`` of ``None`` (or a depth
    outside the volume) draws the surface row alone.

    Axes are lon/lat, taken from the crop's own coordinates.
    """
    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        out = _stem(scene, "inset")
        path = scene.axis_path
        normals = scene.metrics.get("normals")

        # The crop's own coordinates, linearised across the extent: the
        # curvilinear centres are not evenly spaced, and pcolormesh with
        # 1-D coordinate vectors assumes they are.
        lon = np.linspace(float(np.nanmin(scene.XC)),
                          float(np.nanmax(scene.XC)), scene.XC.shape[1])
        lat = np.linspace(float(np.nanmin(scene.YC)),
                          float(np.nanmax(scene.YC)), scene.YC.shape[0])
        dlon = float(lon[1] - lon[0]) if len(lon) > 1 else 0.01
        dlat = float(lat[1] - lat[0]) if len(lat) > 1 else 0.01

        def px_to_deg(p):
            """(j, i) pixels -> (lat, lon) degrees, for the overlays."""
            p = np.asarray(p, dtype=float)
            j = np.clip(p[..., 0], 0, len(lat) - 1)
            i = np.clip(p[..., 1], 0, len(lon) - 1)
            return (np.interp(j, np.arange(len(lat)), lat),
                    np.interp(i, np.arange(len(lon)), lon))

        k_depth = _level_for_depth(scene.Z, depth)
        rows = [(0, "surface")]
        if k_depth is not None and k_depth != 0:
            rows.append((k_depth, f"{float(scene.Z[k_depth]):.0f} m"))

        fig, axes = plt.subplots(len(rows), 1, figsize=(5.6, 4.2 * len(rows)),
                                 dpi=130, squeeze=False)
        for ax, (k, label) in zip(axes[:, 0], rows):
            surf = scene.color[k]
            im = ax.pcolormesh(lon, lat, surf,
                               cmap=field_styles.resolve_cmap(
                                   scene.style.cmap),
                               vmin=scene.clim[0], vmax=scene.clim[1],
                               shading="nearest")
            fig.colorbar(im, ax=ax, label=scene.style.title, shrink=0.85)

            ay, ax_lon = px_to_deg(path)
            ax.plot(ax_lon, ay, "-", color="white", lw=1.8)
            ax.plot(ax_lon, ay, "-", color="black", lw=0.8)

            if normals is not None:
                for kk in (1, 2, 3):
                    for sgn, colour in ((+1, OFFSET_PLUS),
                                        (-1, OFFSET_MINUS)):
                        off = np.asarray(path, dtype=float) + \
                            sgn * kk * np.asarray(normals)
                        oy, ox = px_to_deg(off)
                        ax.plot(ox, oy, "-", color=colour, lw=0.5, alpha=0.75)

                ax.legend(
                    handles=[Line2D([], [], color=OFFSET_PLUS, lw=1.2,
                                    label="+ offsets"),
                             Line2D([], [], color=OFFSET_MINUS, lw=1.2,
                                    label="\u2212 offsets")],
                    loc="upper right", fontsize=6, framealpha=0.6,
                    handlelength=1.4, borderpad=0.3, labelspacing=0.25,
                )

            if perp_index is not None and normals is not None:
                j0, i0 = path[perp_index]
                nj, ni = normals[perp_index]
                t = np.linspace(-half_width, half_width, 2)
                ty, tx = px_to_deg(np.stack(
                    [j0 + t * nj, i0 + t * ni], axis=1))
                ax.plot(tx, ty, "-", color="lime", lw=2.0)
                py, px_ = px_to_deg(np.array([[j0, i0]]))
                ax.plot(px_, py, "o", color="lime", ms=4)

            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            ax.set_title(f"Plan view, {label} — front {scene.label}",
                         fontsize=9)
            ax.set_aspect("auto")

        fig.tight_layout()
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out


def _level_for_depth(Z, depth):
    """Index of the model level nearest *depth*, or ``None``.

    ``None`` for a depth that is not given or lies outside the clipped
    volume -- the inset then shows the surface row alone rather than
    silently plotting the deepest level it happens to have.
    """
    if depth is None:
        return None
    z = np.asarray(Z, dtype=float)
    if z.size == 0:
        return None
    target = -abs(float(depth))
    if target < float(np.min(z)) - 1e-9:
        return None
    return int(np.argmin(np.abs(z - target)))


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
        cmap_volume=scene.style.cmap,   # PyVista takes the name
        zscale=zscale,
        top_marker=top,
        front_iso=iso,
        color_scalar=scene.field_name,
        color_title=scene.style.title,
        # render_3d's defaults (56/60/44) are sized for a poster-scale
        # screenshot; in a browser pane they overlap the axes.
        font_size=14, title_font_size=15, label_font_size=11,
    )


def _depth_cmap():
    """Light-to-dark ramp for depth.  cmocean's if it is installed."""
    try:
        import cmocean
        return cmocean.cm.deep
    except Exception:                                       # noqa: BLE001
        import matplotlib as mpl
        return mpl.colormaps["YlGnBu"]


def figure_isopycnal_depth(scene: FrontScene, sigma: float, *,
                           tile_sigma0=None, tile_Z=None, tile_labels=None,
                           tile_lon=None, tile_lat=None,
                           figsize=(9.0, 6.0)) -> Path:
    """A map of the depth of one isopycnal over the front's crop.

    Colour is depth; gray is where the surface does not exist in the
    column -- it has outcropped, or lies below the model floor.  That is
    a statement about the ocean, so it gets a legend entry rather than
    being left blank.

    Density alone decides this, so it is not one of the per-field
    columns: one map serves whatever fields are selected.
    """
    from fronts.viz.geometry import isopycnal_depth

    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        # The whole tile when it is supplied, not just this front's crop:
        # the surface is a property of the region, and seeing one front's
        # bounding box says nothing about where it sits.
        whole = tile_sigma0 is not None and tile_Z is not None
        depth = isopycnal_depth(tile_sigma0 if whole else scene.sigma0,
                                tile_Z if whole else scene.Z, sigma)
        labels = tile_labels if whole else None
        out = _stem(scene, f"isodepth_{sigma:.2f}".replace(".", "p"))

        fig, ax = plt.subplots(figsize=figsize, dpi=130)

        # Gray underlay: every cell, so wherever `depth` is NaN the gray
        # shows through.  Painting the NaNs directly would need a second
        # masked array; one flat layer underneath is simpler and exact.
        span = None
        if tile_lon is not None and tile_lat is not None:
            span = (float(np.min(tile_lon)), float(np.max(tile_lon)),
                    float(np.min(tile_lat)), float(np.max(tile_lat)))

        ax.imshow(np.ones_like(depth), cmap=colors.ListedColormap(
            [field_styles.NAN_COLOR]), vmin=0, vmax=1, origin="lower",
            interpolation="nearest", extent=span)

        # Plotted as positive metres below the surface with a ramp that
        # runs light-to-dark, so shallow reads shallow and deep reads
        # deep.  Signed depth on a normal ramp inverts that and the map
        # is read backwards.
        below = -depth
        finite = below[np.isfinite(below)]
        if finite.size:
            im = ax.imshow(below, cmap=_depth_cmap(), origin="lower",
                           interpolation="nearest", extent=span,
                           vmin=float(np.nanpercentile(finite, 2)),
                           vmax=float(np.nanpercentile(finite, 98)))
            fig.colorbar(im, ax=ax, label="depth below surface [m]",
                         shrink=0.85)
            title = (f"Depth of sigma = {sigma:.2f} kg/m^3   "
                     f"— front {scene.label}")
        else:
            title = (f"sigma = {sigma:.2f} kg/m^3 is nowhere in this "
                     f"volume — front {scene.label}")

        # Every front in cyan, the selected one in red -- the point of
        # showing the whole tile is to see the chosen front among the rest.
        if labels is not None:
            lab = np.asarray(labels)
            if span is not None:
                gx = np.linspace(span[0], span[1], lab.shape[1])
                gy = np.linspace(span[2], span[3], lab.shape[0])
                grid = (gx, gy)
            else:
                grid = (np.arange(lab.shape[1]), np.arange(lab.shape[0]))
            ax.contour(*grid, (lab > 0).astype(float), levels=[0.5],
                       colors="#00e5ff", linewidths=0.7, alpha=0.85)
            ax.contour(*grid, (lab == scene.label).astype(float),
                       levels=[0.5], colors="#ff1744", linewidths=1.8)
        else:
            ax.contour(scene.front_mask.astype(float), levels=[0.5],
                       colors="#ff1744", linewidths=1.5)

        outcropped = int(np.sum(~np.isfinite(depth)))
        from matplotlib.lines import Line2D
        handles = [Patch(facecolor=field_styles.NAN_COLOR,
                         label=f"undefined / outcropped "
                               f"({outcropped:,} cells)")]
        if labels is not None:
            handles += [
                Line2D([], [], color="#00e5ff", lw=1.0, label="fronts"),
                Line2D([], [], color="#ff1744", lw=1.8,
                       label=f"front {scene.label}"),
            ]
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  framealpha=0.9)

        if tile_lon is not None and tile_lat is not None:
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        else:
            ax.set_xlabel("i (tile pixels)")
            ax.set_ylabel("j (tile pixels)")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("auto")
        fig.tight_layout()
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out


def figure_region_field(scene: FrontScene, surface, labels, *,
                        field_name: str, lon=None, lat=None,
                        clim: tuple[float, float] | None = None,
                        figsize=(9.0, 6.0)) -> Path:
    """One field over the whole tile, with every front drawn on it.

    The companion to :func:`figure_isopycnal_depth`: same frame, same
    front colours, so a feature in the field can be matched to a feature
    in the isopycnal surface.  Fronts are cyan and the selected one red.
    """
    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        out = _stem(scene, f"region_{field_name}")
        style = field_styles.get_style(field_name)
        shown = field_styles.apply_transform(np.asarray(surface), style)
        if clim is None:
            clim = field_styles.default_clim(shown, style)

        span = None
        if lon is not None and lat is not None:
            span = (float(np.min(lon)), float(np.max(lon)),
                    float(np.min(lat)), float(np.max(lat)))

        fig, ax = plt.subplots(figsize=figsize, dpi=130)
        im = ax.imshow(shown, cmap=field_styles.resolve_cmap(style.cmap),
                       origin="lower", interpolation="nearest",
                       vmin=clim[0], vmax=clim[1], extent=span)
        fig.colorbar(im, ax=ax, label=style.title or field_name, shrink=0.85)

        if labels is not None:
            lab = np.asarray(labels)
            if span is not None:
                grid = (np.linspace(span[0], span[1], lab.shape[1]),
                        np.linspace(span[2], span[3], lab.shape[0]))
            else:
                grid = (np.arange(lab.shape[1]), np.arange(lab.shape[0]))
            ax.contour(*grid, (lab > 0).astype(float), levels=[0.5],
                       colors="#00e5ff", linewidths=0.7, alpha=0.85)
            ax.contour(*grid, (lab == scene.label).astype(float),
                       levels=[0.5], colors="#ff1744", linewidths=1.8)
            ax.legend(handles=[
                Line2D([], [], color="#00e5ff", lw=1.0, label="fronts"),
                Line2D([], [], color="#ff1744", lw=1.8,
                       label=f"front {scene.label}"),
            ], loc="lower right", fontsize=7, framealpha=0.9)

        ax.set_xlabel("longitude" if span else "i (tile pixels)")
        ax.set_ylabel("latitude" if span else "j (tile pixels)")
        ax.set_title(f"{field_name} at the surface — front {scene.label}",
                     fontsize=10)
        ax.set_aspect("auto")
        fig.tight_layout()
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out


def default_sigma(scene: FrontScene) -> float:
    """A sigma that exists in this volume: its median density.

    The map is useless if the chosen surface is nowhere in the tile, so
    the control starts somewhere guaranteed to be present.
    """
    finite = scene.sigma0[np.isfinite(scene.sigma0)]
    if finite.size == 0:
        return 27.0
    return float(np.median(finite))


def figure_profiles(scene: FrontScene, points, *, figsize=(4.4, 5.2)) -> Path:
    """Vertical profiles of the colour field at chosen locations.

    Depth on the y-axis with the surface at the top and depth increasing
    downward -- the convention every oceanographer reads without thinking
    -- and the field on the x-axis.  One line per location, so this does
    belong in the per-field columns.

    *points* are ``(j, i)`` in the crop frame, which is the frame the plan
    view is clicked in.
    """
    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        out = _stem(scene, f"profiles{len(points)}")
        if not points:
            return _blank_png(out, "click the plan view to place profile "
                                   "locations", figsize)

        nk, nj, ni = scene.color.shape
        fig, ax = plt.subplots(figsize=figsize, dpi=130)

        drawn = 0
        for n, (j, i) in enumerate(points):
            j, i = int(j), int(i)
            if not (0 <= j < nj and 0 <= i < ni):
                continue
            values = scene.color[:, j, i]
            if not np.any(np.isfinite(values)):
                continue
            ax.plot(values, scene.Z[:len(values)], "-o", ms=2.5, lw=1.2,
                    color=PROFILE_COLORS[n % len(PROFILE_COLORS)],
                    label=f"{n + 1}: (j={j}, i={i})")
            drawn += 1

        if drawn == 0:
            plt.close(fig)
            return _blank_png(out, "no finite data at those locations",
                              figsize)

        ax.set_xlabel(scene.style.title or scene.field_name, fontsize=8)
        ax.set_ylabel("depth [m]", fontsize=8)
        # Z is negative downward, so the natural ordering already puts the
        # surface at the top; state it rather than relying on the sign.
        ax.set_ylim(float(np.min(scene.Z)), float(np.max(scene.Z)))
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, loc="best")
        ax.set_title(f"Vertical profiles — front {scene.label}", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out


#: Offset sides, shared with the plan view so + and - mean the same thing
#: in the inset, on the plan view, and in the offsets panel.
OFFSET_PLUS = "#ff7043"
OFFSET_MINUS = "#42a5f5"

#: One colour per profile location, shared with the plan-view markers so
#: line 3 on the profile plot is marker 3 on the map.
PROFILE_COLORS = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4")


def _blank_png(out: Path, message: str, figsize) -> Path:
    """A placeholder image carrying an explanation."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8,
            wrap=True, transform=ax.transAxes, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def axis_ticks(scene: FrontScene, n: int = 6):
    """Evenly spaced indices along the front axis, with their distances.

    Returned as ``[(index, km_from_start), ...]``.  The plan view marks
    these and the section figures put the same distances on their x-axis,
    so a feature in a section can be found on the map.
    """
    path = np.asarray(scene.axis_path)
    if len(path) < 2:
        return []

    # path_metrics calls it dist_km, and returns None when it had no
    # lon/lat to work from; dist_px is the always-present fallback.
    dist = scene.metrics.get("dist_km")
    if dist is None:
        dist = scene.metrics.get("dist_px")
    if dist is None or len(dist) != len(path):
        dist = np.arange(len(path), dtype=float)

    idx = np.linspace(0, len(path) - 1, max(int(n), 2)).astype(int)
    return [(int(k), float(dist[k])) for k in idx]


# --------------------------------------------------------------------------
# The whole region, with every front numbered
# --------------------------------------------------------------------------

def figure_region_fronts(surface, labels, *, lon=None, lat=None,
                         field_name: str = "density", selected: int = 0,
                         title: str | None = None, out: Path | None = None,
                         annotate: bool = True, min_pixels: int = 20,
                         clim: tuple[float, float] | None = None,
                         figsize=(9.0, 7.2)) -> Path:
    """Every front in the region, numbered, over the field.

    Takes plain arrays rather than a :class:`FrontScene` on purpose.  A
    scene requires a chosen front, a crop and a mixed-layer clip -- none of
    which you can sensibly choose *before* seeing which fronts are here and
    where they are.  This is the figure you look at to make that choice, so
    it must not depend on having already made it.

    Fronts are cyan with their label drawn at the centroid; the selected
    one is red.  The numbers are the point: a dropdown of five-digit
    integers is not a way to pick a front.
    """
    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        surface = np.asarray(surface)
        labels = np.asarray(labels)
        style = field_styles.get_style(field_name)

        out = Path(out) if out is not None else (
            _outdir() / f"region_fronts_{field_name}_{int(selected)}.png")

        if lon is not None and lat is not None:
            extent = [float(np.nanmin(lon)), float(np.nanmax(lon)),
                      float(np.nanmin(lat)), float(np.nanmax(lat))]
            xlabel, ylabel = "longitude [deg]", "latitude [deg]"
        else:
            extent = [0, surface.shape[1], 0, surface.shape[0]]
            xlabel, ylabel = "i", "j"

        fig, ax = plt.subplots(figsize=figsize)
        # Same transform + limits path as figure_region_field, so the two
        # views of the same field are directly comparable.
        shown = field_styles.apply_transform(surface, style)
        # A shared range when one is given.  Per-frame limits make a movie
        # unreadable: the colours change because the scale moved, not
        # because the ocean did, and the two are indistinguishable on
        # screen.
        if clim is None:
            clim = field_styles.default_clim(shown, style)
        im = ax.imshow(shown, origin="lower", extent=extent, aspect="auto",
                       cmap=field_styles.resolve_cmap(style.cmap),
                       vmin=clim[0], vmax=clim[1])
        fig.colorbar(im, ax=ax, label=getattr(style, "title", field_name))

        def to_xy(j, i):
            """Row/column -> axis coordinates, matching `extent`."""
            fx = (i + 0.5) / surface.shape[1]
            fy = (j + 0.5) / surface.shape[0]
            return (extent[0] + fx * (extent[1] - extent[0]),
                    extent[2] + fy * (extent[3] - extent[2]))

        present = [int(v) for v in np.unique(labels) if v]
        drawn = 0
        for value in present:
            mask = labels == value
            if mask.sum() < min_pixels:
                continue
            drawn += 1
            js, iss = np.nonzero(mask)
            xs, ys = to_xy(js, iss)
            is_sel = int(value) == int(selected)
            ax.scatter(xs, ys, s=2.5,
                       c="#ff1744" if is_sel else "#00e5ff",
                       linewidths=0, zorder=3 if is_sel else 2)
            if annotate:
                cx, cy = to_xy(js.mean(), iss.mean())
                ax.annotate(
                    str(value), (cx, cy), color="white", fontsize=7,
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="#ff1744" if is_sel else "#006064",
                              ec="none", alpha=0.85))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"{drawn} fronts")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out


def figure_profile_stack(columns, Z, times, *, field_name: str,
                         highlight: int | None = None,
                         out: Path | None = None,
                         figsize=(4.6, 6.4)) -> Path:
    """Every timestep's profile at one point, on one axis.

    A profile at a fixed location is one line per timestep, so a *movie*
    of them shows one line at a time and throws away the comparison that
    matters.  Drawn together, coloured by time, the evolution is the
    picture.

    *highlight* draws one step in the foreground, thick and dark, so the
    figure can track the movie and the time-series cursor without being
    rebuilt from data each time.

    Parameters
    ----------
    columns : sequence
        One ``(K,)`` array per step, or ``None`` where the front was
        absent.  Gaps are skipped, not interpolated.
    Z : array
        Depth axis, negative downward.
    times : sequence of str
        Timestamps, for the colour bar's labels.
    """
    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors as mcolors

        out = Path(out) if out is not None else (
            _outdir() / f"profilestack_{field_name}_{highlight}.png")

        usable = [(k, np.asarray(c, dtype=float))
                  for k, c in enumerate(columns) if c is not None]

        fig, ax = plt.subplots(figsize=figsize)
        if not usable:
            ax.text(0.5, 0.5, "no profile at this point",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            return out

        depth = np.asarray(Z, dtype=float)
        n = len(columns)
        cmap = plt.get_cmap("viridis")      # cm.get_cmap is deprecated
        norm = mcolors.Normalize(vmin=0, vmax=max(n - 1, 1))

        for k, col in usable:
            m = min(len(col), len(depth))
            ax.plot(col[:m], depth[:m], color=cmap(norm(k)), lw=1.0,
                    alpha=0.75, zorder=2)

        if highlight is not None:
            for k, col in usable:
                if k != int(highlight):
                    continue
                m = min(len(col), len(depth))
                # Drawn twice: a white casing under a dark line, so the
                # highlight reads against any colour it happens to sit on.
                ax.plot(col[:m], depth[:m], color="white", lw=4.0, zorder=3)
                ax.plot(col[:m], depth[:m], color="#b71c1c", lw=2.2,
                        zorder=4, label=str(times[k]) if k < len(times) else "")
                ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("timestep")
        if times:
            ticks = np.linspace(0, max(n - 1, 1), min(5, n)).astype(int)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([str(times[t])[:16] for t in ticks
                                 if t < len(times)])
            cbar.ax.tick_params(labelsize=6)

        style = field_styles.get_style(field_name)
        ax.set_xlabel(getattr(style, "title", field_name))
        ax.set_ylabel("depth [m]")
        ax.set_title("vertical profile at the transect point", fontsize=10)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
