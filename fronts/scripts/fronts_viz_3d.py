"""
Render a 3-D view of a single labelled front through a sigma0 density tile.

Inputs:
  * a 3-D density tile NetCDF produced by
    ``llc4320-native-grid-preprocessing/src/dbof/tiles/generate_tile.py``
    (sigma0(k, j, i) on **face-local** axes, plus XC, YC, Z and the
    rect_i_start / rect_j_start / face_index / tile_index provenance
    fields).
  * a global labelled-fronts mask (``.npy`` or ``.nc``) on the **rect
    grid** (shape 12960 x 17280, integer labels, 0 = no front).
  * a locator: either ``--i / --j`` (rect-grid pixel coordinates) or
    ``--lat / --lon``.

The script picks the labelled front whose pixel contains (or is closest
to) the locator, remaps the density volume onto the rect-tile-local
frame, crops to the front's bounding box plus a margin, clips depth to a
few LLC levels below the deepest mixed-layer depth in that bbox, and
renders the result in PyVista as either isopycnal surfaces or a volume
render, with the front itself overlaid as a sigma0-coloured "curtain".
A companion 2-D matplotlib inset showing surface sigma0 + the front +
lon/lat ticks is written next to the 3-D PNG by default.

CLI usage
---------
    python -m fronts.scripts.fronts_viz_3d \\
        --density-tile density_tile330_20121109T12.nc \\
        --labels       LLC4320_2012-11-09T12_00_00_V4_bfronts.npy \\
        --i 13170 --j 9950 \\
        --output       fronts_viz_3d_californiacurrent.png
"""

# stdlib
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# numerical / IO
import numpy as np
from scipy import ndimage as scimg

# tile lookup + label loading helpers from the dev/mld utilities
_DEV_MLD = Path(__file__).resolve().parents[2] / "dev" / "mld"
if str(_DEV_MLD) not in sys.path:
    sys.path.insert(0, str(_DEV_MLD))
from density_utils import (  # noqa: E402
    load_density_tile,
    load_labels_tile,
    tile_scalar,
    build_tile_lookup,
    attach_lonlat_twins,
)

# Repo helpers
from fronts.llc.analysis import mixed_layer_depth_field  # noqa: E402
from fronts.viz.fronts_3d import (  # noqa: E402
    front_bbox_and_crop,
    truncate_depth,
    build_pyvista_grid,
    build_front_curtain,
    build_front_top_marker,
    build_front_isosurface,
    pick_isopycnals_across_front,
    mixed_layer_clim,
    dilate_front_mask,
    front_volume_clim,
    render_3d,
)
from fronts.viz.insets import plot_bbox_inset  # noqa: E402
from fronts.viz.pv_helpers import save_with_rst  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Build and parse the CLI arguments for ``fronts_viz_3d.py``.

    Parameters
    ----------
    argv : list of str or None
        Argument list (default ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed namespace.  ``args.locator_kind`` is set to ``'ij'`` or
        ``'latlon'`` after validation.
    """
    p = argparse.ArgumentParser(
        description=(
            "Render a 3-D view of one labelled front through a sigma0 "
            "density tile."
        ),
    )
    p.add_argument("--density-tile", type=Path, required=True,
                   help="NetCDF density tile produced by generate_tile.py.")
    p.add_argument("--labels", type=Path, required=True,
                   help="Global labelled-fronts mask (.npy or .nc).")

    # Locator -- exactly one pair must be supplied.
    p.add_argument("--i", type=int, default=None,
                   help="Rect-grid column (locator option A).")
    p.add_argument("--j", type=int, default=None,
                   help="Rect-grid row (locator option A).")
    p.add_argument("--lat", type=float, default=None,
                   help="Latitude in degrees (locator option B).")
    p.add_argument("--lon", type=float, default=None,
                   help="Longitude in degrees (locator option B).")

    # Outputs.
    p.add_argument("--output", type=Path, required=True,
                   help="Output PNG for the 3-D figure.")
    p.add_argument("--interactive-html", type=Path, default=None,
                   help="Output HTML for the interactive view "
                        "(default: derived from --output).")
    p.add_argument("--no-inset", action="store_true",
                   help="Skip the 2-D companion inset figure.")
    p.add_argument("--inset-output", type=Path, default=None,
                   help="Output PNG for the 2-D inset "
                        "(default: {stem}_inset.png).")

    # Visualisation controls.
    p.add_argument("--mode", choices=["isopycnals", "volume"],
                   default="isopycnals",
                   help="Background rendering mode (default isopycnals).")
    p.add_argument("--isopycnals", type=float, nargs="+", default=None,
                   help="Explicit isopycnal sigma0 values; otherwise "
                        "5 levels are auto-picked from the cross-front "
                        "surface contrast.")
    p.add_argument("--opacity", choices=["sigmoid", "linear", "geom"],
                   default="sigmoid",
                   help="Opacity transfer for --mode volume.")
    p.add_argument("--clim", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="sigma0 colour limits (default 2/98 percentile).")
    p.add_argument(
        "--cmap-volume", type=str, default="viridis",
        help=(
            "Colormap for the volume/isopycnal background.  Suggestions "
            "where DENSER water is DARKER (more natural for sigma0): "
            "dense, deep, gray (cmocean, bare names accepted by PyVista), "
            "Blues, bone, viridis_r, cividis_r, magma_r, plasma_r.  "
            "Default 'viridis' (denser=lighter)."
        ),
    )
    p.add_argument("--cmap-curtain", type=str, default="magma",
                   help="Colormap for the front curtain.")
    p.add_argument(
        "--font-size", type=int, default=None,
        help=(
            "Override the renderer's bounds-axis font size.  When "
            "unset, the scene defaults defined in render_3d are used "
            "(currently 56)."
        ),
    )
    p.add_argument(
        "--title-font-size", type=int, default=None,
        help="Override the scalar-bar title font size (renderer default 60).",
    )
    p.add_argument(
        "--label-font-size", type=int, default=None,
        help="Override the scalar-bar tick label font size (renderer default 44).",
    )
    p.add_argument("--zscale", type=float, default=50.0,
                   help="Vertical exaggeration of the depth axis.")
    p.add_argument("--margin", type=int, default=50,
                   help="Pixel margin around the front bbox.")
    p.add_argument("--n-below", type=int, default=3,
                   help="LLC levels below the deepest MLD to include.")
    p.add_argument("--cpos", type=str, default=None,
                   help="Camera position JSON triple to lock framing.")
    p.add_argument("--show", action="store_true",
                   help="Open an interactive window instead of off-screen.")

    args = p.parse_args(argv)

    # Locator validation: exactly one of (--i,--j) or (--lat,--lon).
    ij = (args.i is not None) and (args.j is not None)
    ll = (args.lat is not None) and (args.lon is not None)
    if ij == ll:
        p.error("Supply exactly one of (--i AND --j) or (--lat AND --lon).")
    args.locator_kind = "ij" if ij else "latlon"
    return args


# ---------------------------------------------------------------------------
# Front-label picking
# ---------------------------------------------------------------------------

def pick_front_label(
    labels_tile: np.ndarray,
    locator_kind: str,
    locator: tuple[int, int] | tuple[float, float],
    rect_i_start: int,
    rect_j_start: int,
    XC_rect: np.ndarray,
    YC_rect: np.ndarray,
) -> tuple[int, int, int]:
    """Return ``(label, j_local, i_local)`` for the front at the locator.

    Parameters
    ----------
    labels_tile : numpy.ndarray
        Integer labelled-fronts mask cropped to the tile window
        (TILE_SIZE x TILE_SIZE), in the rect-grid tile-local frame.
    locator_kind : {'ij', 'latlon'}
        Which locator was supplied on the CLI.
    locator : tuple
        ``(i, j)`` rect-grid global indices when ``locator_kind='ij'``,
        else ``(lat, lon)`` in degrees.
    rect_i_start, rect_j_start : int
        Origin of the tile on the global rect grid (from the density
        tile's provenance attrs).
    XC_rect, YC_rect : numpy.ndarray
        Longitude / latitude on the rect-grid tile-local frame, shape
        ``(TILE_SIZE, TILE_SIZE)``.

    Returns
    -------
    label : int
        Selected non-zero label.
    j_local, i_local : int
        Tile-local rect-frame pixel indices of the chosen front pixel.
    """
    H, W = labels_tile.shape

    if locator_kind == "ij":
        i_global, j_global = locator
        i_local = int(i_global) - int(rect_i_start)
        j_local = int(j_global) - int(rect_j_start)
        if not (0 <= i_local < W and 0 <= j_local < H):
            raise SystemExit(
                f"--i={i_global} --j={j_global} maps to tile-local "
                f"({i_local}, {j_local}), which lies outside the tile "
                f"(0..{W - 1}, 0..{H - 1}).  "
                f"The density tile covers rect i in "
                f"[{rect_i_start}, {rect_i_start + W}) and j in "
                f"[{rect_j_start}, {rect_j_start + H})."
            )
        label = int(labels_tile[j_local, i_local])
        if label == 0:
            raise SystemExit(
                f"--i={i_global} --j={j_global} lands on a 0-label pixel "
                "(no front).  Try a (lat, lon) locator and let the script "
                "snap to the nearest labelled pixel."
            )
        return label, j_local, i_local

    # --lat / --lon: nearest grid pixel + snap to nearest labelled pixel.
    lat, lon = locator
    # Squared distance in rect-frame (degrees); a Euclidean approximation
    # is fine for picking the nearest pixel inside a single ~720x720 tile.
    d2 = (XC_rect - float(lon)) ** 2 + (YC_rect - float(lat)) ** 2
    j_nearest, i_nearest = np.unravel_index(int(np.argmin(d2)), d2.shape)

    label_here = int(labels_tile[j_nearest, i_nearest])
    if label_here != 0:
        return label_here, int(j_nearest), int(i_nearest)

    # Snap to nearest non-zero label pixel inside the tile.  Use a
    # distance transform of the zero-label region: the indices argument
    # gives, for every background pixel, the (j, i) of the nearest
    # foreground (labelled) pixel.
    has_label = labels_tile > 0
    if not has_label.any():
        raise SystemExit(
            f"No labelled front pixels exist inside the tile covering "
            f"rect i in [{rect_i_start}, {rect_i_start + W}) and j in "
            f"[{rect_j_start}, {rect_j_start + H}); cannot snap "
            f"--lat={lat} --lon={lon} to a front."
        )
    _, (jj, ii) = scimg.distance_transform_edt(
        ~has_label, return_indices=True,
    )
    j_snap = int(jj[j_nearest, i_nearest])
    i_snap = int(ii[j_nearest, i_nearest])
    return int(labels_tile[j_snap, i_snap]), j_snap, i_snap


# ---------------------------------------------------------------------------
# Field remapping
# ---------------------------------------------------------------------------

def remap_to_rect(arr_face: np.ndarray,
                  j_tile_lookup: np.ndarray,
                  i_tile_lookup: np.ndarray) -> np.ndarray:
    """Remap a face-local array onto the rect-grid tile-local frame.

    Works for 2-D ``(j_face, i_face)`` and 3-D ``(k, j_face, i_face)``
    arrays alike (k slides through with broadcasting).

    Parameters
    ----------
    arr_face : numpy.ndarray
        Face-local array, last two axes are ``(j_face, i_face)``.
    j_tile_lookup, i_tile_lookup : numpy.ndarray
        Tile-local face-index lookups returned by
        :func:`density_utils.build_tile_lookup`.

    Returns
    -------
    numpy.ndarray
        Array of the same dtype + leading dims, with the last two axes
        now in the rect-grid tile-local frame.
    """
    return arr_face[..., j_tile_lookup, i_tile_lookup]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str or None
        Argument list (default ``sys.argv[1:]``).

    Side effects
    ------------
    Writes the 3-D PNG, the interactive HTML, and (unless ``--no-inset``)
    the 2-D companion PNG.  Prints the chosen isopycnal levels and the
    camera position triple to stdout so subsequent runs can pin both.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    log = logging.getLogger("fronts_viz_3d")

    # Hard-error on --show with no display (round-2 policy: hard error
    # on misuse rather than silent fallback).
    if args.show and not os.environ.get("DISPLAY"):
        raise SystemExit(
            "--show requires an X display, but $DISPLAY is empty.  Drop "
            "--show to render off-screen instead."
        )

    # ---------- Load density tile + labels ----------
    log.info("Loading density tile %s", args.density_tile)
    ds = load_density_tile(args.density_tile)
    rect_i_start = int(tile_scalar(ds, "rect_i_start"))
    rect_j_start = int(tile_scalar(ds, "rect_j_start"))
    face_index   = int(tile_scalar(ds, "face_index"))

    sigma0_face = ds["sigma0"].values  # (K, J_face, I_face)
    Z = ds["Z"].values.astype(np.float64)  # (K,), negative downward
    XC_face = ds["XC"].values  # (J_face, I_face)
    YC_face = ds["YC"].values  # (J_face, I_face)

    log.info("Building tile lookup (face -> rect-tile-local)")
    j_tile_lookup, i_tile_lookup = build_tile_lookup(
        rect_i_start, rect_j_start, face_index,
    )

    log.info("Loading labels for the tile window")
    rect_j_slice = slice(rect_j_start, rect_j_start + j_tile_lookup.shape[0])
    rect_i_slice = slice(rect_i_start, rect_i_start + j_tile_lookup.shape[1])
    labels_tile = load_labels_tile(args.labels, rect_j_slice, rect_i_slice)

    # Remap sigma0, XC, YC onto the rect-grid tile-local frame so all
    # downstream operations share one coordinate system.
    log.info("Remapping sigma0/XC/YC face-local -> rect-tile-local")
    sigma0_rect = remap_to_rect(sigma0_face, j_tile_lookup, i_tile_lookup)
    XC_rect = remap_to_rect(XC_face, j_tile_lookup, i_tile_lookup)
    YC_rect = remap_to_rect(YC_face, j_tile_lookup, i_tile_lookup)

    # ---------- Pick the front ----------
    if args.locator_kind == "ij":
        locator = (args.i, args.j)
    else:
        locator = (args.lat, args.lon)
    label, j_pick, i_pick = pick_front_label(
        labels_tile, args.locator_kind, locator,
        rect_i_start, rect_j_start, XC_rect, YC_rect,
    )
    log.info("Selected front label=%d at tile-local (j=%d, i=%d) lon=%.3f lat=%.3f",
             label, j_pick, i_pick,
             float(XC_rect[j_pick, i_pick]), float(YC_rect[j_pick, i_pick]))

    # ---------- Crop, MLD, depth clip ----------
    j_slice, i_slice = front_bbox_and_crop(
        labels_tile, label, margin=args.margin,
    )
    log.info("Cropped to bbox j=%s i=%s (margin=%d)",
             (j_slice.start, j_slice.stop),
             (i_slice.start, i_slice.stop), args.margin)

    sigma0_cropped = sigma0_rect[:, j_slice, i_slice]
    front_mask_full = (labels_tile == label)
    front_mask_cropped = front_mask_full[j_slice, i_slice]

    # MLD (vectorised) over the cropped column to keep the search small.
    z_mld, k_mld = mixed_layer_depth_field(sigma0_cropped, Z)
    sigma0_clipped, Z_clipped, k_clip = truncate_depth(
        sigma0_cropped, Z, k_mld, n_below=args.n_below,
    )
    log.info(
        "MLD inside bbox: deepest k=%d (z=%.1f m); clipping to k=%d (z=%.1f m)",
        int(np.nanmax(k_mld)) if (k_mld >= 0).any() else -1,
        float(np.nanmin(z_mld)) if (k_mld >= 0).any() else float("nan"),
        k_clip - 1, float(Z_clipped[-1]),
    )

    # ---------- Pick isopycnal levels (mode==isopycnals only) ----------
    levels = pick_isopycnals_across_front(
        sigma0_clipped, front_mask_cropped, Z_clipped,
        user_levels=args.isopycnals, n_levels=5,
    )
    if args.isopycnals is None:
        log.info("Auto-picked isopycnal sigma0 levels (kg m^-3): "
                 "%s", ", ".join(f"{lv:.4f}" for lv in levels))
        print("Auto-picked --isopycnals " + " ".join(f"{lv:.6f}" for lv in levels))

    # ---------- Front region (used for clim only; iso-surfaces span bbox) ---
    # The 2-pixel dilation gives a tight pool for picking a representative
    # front-iso level; the iso-surfaces themselves are rendered over the
    # full cropped bbox so the surrounding "waters near the front" stay
    # visible.
    dilated_front = dilate_front_mask(front_mask_cropped, iterations=2)

    # ---------- Default clim: 2/98 percentile of the mixed layer ----------
    # Restricting to the mixed layer keeps the iso-surfaces colourful
    # (most cross-front contrast lives there) without washing out the
    # broader bbox context.
    if args.clim is None:
        clim = mixed_layer_clim(sigma0_clipped, k_mld)
        log.info(
            "Auto clim = (%.4f, %.4f) kg m^-3 (mixed layer)", *clim,
        )
    else:
        clim = tuple(args.clim)

    # ---------- PyVista scene ----------
    # No mask -> iso-surfaces are rendered over the whole cropped bbox
    # ("waters near the front") -- restoring the v1.2 behaviour the user
    # asked back for.
    grid = build_pyvista_grid(
        sigma0_clipped, Z_clipped, j_slice, i_slice, zscale=args.zscale,
    )

    # Build a single front-iso-surface: picking the median of the cross-
    # front sigma0 levels gives an opaque "front sheet" that tilts with
    # depth -- the visualization the user asked for ("dense on one side,
    # lighter on the other; visualize the tilt").
    front_iso_level = float(np.median(levels))
    log.info("Front iso-surface drawn at sigma0 = %.4f kg m^-3", front_iso_level)
    front_iso = build_front_isosurface(grid, level=front_iso_level)

    # The curtain is still built so the top-layer marker can reuse its
    # cached branch decomposition, but it is no longer rendered into the
    # scene by default (the vertical-sheet appearance was confusing).
    curtain = build_front_curtain(
        front_mask_cropped, sigma0_clipped, Z_clipped,
        j_slice, i_slice, zscale=args.zscale,
    )
    # Smaller tube + lower default opacity in render_3d so the surface
    # marker is identifiable without dominating the iso-surface tilt.
    top_marker = build_front_top_marker(
        curtain, Z_clipped, j_slice, i_slice,
        zscale=args.zscale, tube_radius=0.8,
    )

    # Only pass font overrides when the user supplied them; otherwise let
    # render_3d use its baked-in defaults.
    font_kwargs = {}
    if args.font_size is not None:
        font_kwargs["font_size"] = args.font_size
    if args.title_font_size is not None:
        font_kwargs["title_font_size"] = args.title_font_size
    if args.label_font_size is not None:
        font_kwargs["label_font_size"] = args.label_font_size

    pl = render_3d(
        grid, curtain, levels,
        mode=args.mode, clim=clim,
        cmap_volume=args.cmap_volume, cmap_curtain=args.cmap_curtain,
        opacity=args.opacity, zscale=args.zscale, show=args.show,
        top_marker=top_marker,
        front_iso=front_iso,
        **font_kwargs,
    )

    # ---------- Save PNG + HTML ----------
    # render_3d sets a south-east elevated camera so i increases L->R
    # and j increases bottom->top.  save_with_rst will fall back to the
    # default isometric if `cpos` is None, so capture the current camera
    # explicitly and pass it through unless the user supplied --cpos.
    if args.cpos:
        cpos_in = json.loads(args.cpos)
    else:
        cpos_in = list(pl.camera_position)
    html_path = args.interactive_html
    if html_path is None:
        html_path = args.output.with_suffix(".html")
    rst = save_with_rst(
        pl, args.output,
        caption=f"3-D view of front label={label}",
        alt=f"3D fronts viz (label {label})",
        cpos=cpos_in,
        interactive_html=html_path,
    )
    log.info("Wrote %s and %s", args.output, html_path)
    # save_with_rst closes the plotter; pl.camera_position is no longer
    # safe to read.  Instead, surface the rst block for downstream use.
    print(rst)

    # ---------- 2-D companion inset ----------
    if not args.no_inset:
        # Surface sigma0 = the reference depth (closest LLC level to 10 m).
        k_ref = int(np.abs(np.abs(Z) - 10.0).argmin())
        surf = sigma0_rect[k_ref]
        inset_path = args.inset_output
        if inset_path is None:
            inset_path = args.output.with_name(
                args.output.stem + "_inset" + args.output.suffix,
            )
        # Inset clim is decoupled from the 3-D scene clim: the 3-D
        # contrast favours depth tilt (broad range across the front
        # column), while the surface map should favour cross-front
        # contrast at one depth.  Pass clim=None so plot_bbox_inset
        # computes 2/98 percentile of the surface slice in the bbox.
        plot_bbox_inset(
            surf, front_mask_full, XC_rect, YC_rect,
            j_slice, i_slice, inset_path,
            clim=None,
            title=(
                f"Front {label} at lon={float(XC_rect[j_pick, i_pick]):.2f}, "
                f"lat={float(YC_rect[j_pick, i_pick]):.2f}"
            ),
            attach_lonlat_twins=attach_lonlat_twins,
            j_tile_lookup=j_tile_lookup,
            i_tile_lookup=i_tile_lookup,
            XC_face=XC_face,
            YC_face=YC_face,
        )
        log.info("Wrote inset %s", inset_path)


if __name__ == "__main__":
    main()
