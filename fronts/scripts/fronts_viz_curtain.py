"""
Render 2-D "curtain" cross-sections of a single labelled front through a tile.

This is the 2-D companion to ``fronts/scripts/fronts_viz_3d.py``.  It reuses
that script's tile-loading / remapping / front-picking pipeline, then -- instead
of a 3-D PyVista scene -- produces three static matplotlib curtain figures:

1. **Main-axis curtain** (``{prefix}_{field}_{loc}_mainaxis.png``) -- a vertical
   cross-section along the front's main axis (longest end-to-end path through
   the skeleton, side branches dropped).  x = distance along the front,
   y = depth, color = a configurable field (default ``Ri``), with isopycnal
   (sigma0) contours overlaid.
2. **Along-front curtains with offsets** (``{prefix}_{field}_{loc}_offsets_n{N}.png``)
   -- two columns (the two sides of the front); row 0 of each is the main
   axis, rows below are offsets 1..N pixels away.  Offset columns whose
   geometry self-overlaps (concave bends / skeleton noise) are trimmed.
3. **Cross-front curtain** (``{prefix}_{field}_{loc}_perp.png``) -- a
   perpendicular transect cut at a chosen point along the main axis (default:
   the field extremum over the full depth range), same curtain style.

A 2-D **map-view inset** (``{prefix}_{field}_{loc}_inset.png``) shows the bbox
with the main axis, the offset envelope, and the marked perpendicular point.
``{field}`` is the tile variable name (e.g. ``Ri``), ``{loc}`` is the picked
front's location (e.g. ``lat36.38_lon-124.20``), and ``{N}`` is ``--n-offsets``.

Inputs mirror ``fronts_viz_3d``:
  * a 3-D **density tile** NetCDF (``sigma0(k, j, i)``, face-local) -- drives
    isopycnal contours;
  * a **field tile** (same window+timestamp, e.g. ``--property Ri``)
    whose variable is the curtain color (REQUIRED here -- the curtain color and
    the perpendicular-point extremum both need it; default ``Ri``);
  * a global **labelled-fronts mask** on the rect grid;
  * a locator: ``--i / --j`` or ``--lat / --lon``.

CLI usage
---------
    python -m fronts.scripts.fronts_viz_curtain \
        --density-tile density_tile330_20121109T12.nc \
        --field-tile   ri_tile330_20121109T12.nc \
        --labels       labeled_fronts_global_20121109T12_00_00_V4.npy \
        --i 13142 --j 9956 \
        --n-offsets 3 --perp-half-width 10 \
        --output-prefix /tmp/calcurrent_curtain
"""

# stdlib
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# numerical / plotting
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# tile lookup + label loading helpers from the dev/mld utilities (same as 3-D).
_DEV_MLD = Path(__file__).resolve().parents[2] / "dev" / "mld"
if str(_DEV_MLD) not in sys.path:
    sys.path.insert(0, str(_DEV_MLD))
from density_utils import (  # noqa: E402
    load_density_tile,
    load_tile,
    check_tiles_consistent,
    load_labels_tile,
    tile_scalar,
    build_tile_lookup,
    attach_lonlat_twins,
)

# Repo helpers.
from fronts.llc.analysis import mixed_layer_depth_field  # noqa: E402
from fronts.viz.fronts_3d import (  # noqa: E402
    front_bbox_and_crop,
    truncate_depth,
)
from fronts.viz.field_styles import (  # noqa: E402
    get_style,
    apply_transform,
    default_clim,
    NAN_COLOR,
)
from fronts.viz import curtains  # noqa: E402

# Reuse the 3-D script's front-picking + remap helpers directly (no copy).
from fronts.scripts.fronts_viz_3d import (  # noqa: E402
    pick_front_label,
    remap_to_rect,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Build and parse the CLI arguments for ``fronts_viz_curtain.py``.

    Parameters
    ----------
    argv : list of str or None
        Argument list (default ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed namespace; ``args.locator_kind`` is ``'ij'`` or ``'latlon'``.
    """
    p = argparse.ArgumentParser(
        description="Render 2-D curtain cross-sections of one labelled front.",
    )
    p.add_argument("--density-tile", type=Path, required=True,
                   help="NetCDF density tile (sigma0); drives isopycnals.")
    p.add_argument("--field-tile", type=Path, required=True,
                   help="NetCDF field tile (same window+timestamp) whose "
                        "variable colors the curtains (e.g. a Ri tile).")
    p.add_argument("--field-name", type=str, default=None,
                   help="Variable name inside --field-tile "
                        "(default: auto-detect the single 3-D variable).")
    p.add_argument("--field-transform",
                   choices=["log10", "symlog", "linear"], default=None,
                   help="Override the field's registered display transform.")
    p.add_argument("--field-clip", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Override the field's registered raw-value clip range.")
    p.add_argument("--labels", type=Path, required=True,
                   help="Global labelled-fronts mask (.npy or .nc).")

    # Locator -- exactly one pair.
    p.add_argument("--i", type=int, default=None,
                   help="Rect-grid column (PREFERRED locator). Must land on a "
                        "front pixel; errors on a 0 (no-front) pixel.")
    p.add_argument("--j", type=int, default=None,
                   help="Rect-grid row (PREFERRED locator).")
    p.add_argument("--lat", type=float, default=None,
                   help="Latitude in degrees (advanced locator). Snaps to the "
                        "nearest labelled front IN THIS TILE; guarded against "
                        "out-of-tile values (the run logs the tile's bbox).")
    p.add_argument("--lon", type=float, default=None,
                   help="Longitude in degrees (advanced locator). See --lat.")

    # Geometry controls.
    p.add_argument("--margin", type=int, default=50,
                   help="Pixel margin around the front bbox (default 50).")
    p.add_argument("--n-below", type=int, default=3,
                   help="LLC levels below the deepest MLD to include "
                        "(default 3); sets the curtain depth extent.")
    p.add_argument("--n-offsets", type=int, default=3,
                   help="Number of offset rows per side (default 3).")
    p.add_argument("--perp-half-width", type=int, default=30,
                   help="Half-width (px) of the perpendicular transect "
                        "(default 30; total length 2*N+1).")
    p.add_argument("--perp-point", type=int, default=None,
                   help="Main-axis column index for the perpendicular "
                        "transect.  Default: the field-extremum column.  Use "
                        "--list-perp-candidates to see column -> (i,j) values.")
    p.add_argument("--extremum", choices=["min", "max"], default="min",
                   help="Whether the default perpendicular point is the field "
                        "minimum (default; e.g. lowest Ri) or maximum.")
    p.add_argument("--isopycnal-curtain", action="store_true",
                   help="Also write a flattened-isopycnal figure: the "
                        "density surface the front lives on, unrolled to 2-D "
                        "(x = along-front distance, y = depth, color = the "
                        "field ON that surface as it slopes down-and-"
                        "sideways).")
    p.add_argument("--isopycnal-sigma0", type=float, default=None,
                   help="Density surface for --isopycnal-curtain (default: "
                        "median surface sigma0 along the main axis).")
    p.add_argument("--isopycnal-half-width", type=int, default=None,
                   help="Limit the cross-front search for the surface to "
                        "this many px (default: unlimited -- the surface is "
                        "followed across the whole tile).")
    p.add_argument("--perp-isopycnal", action="store_true",
                   help="Draw the perpendicular curtain in isopycnal-"
                        "following coordinates: each depth row is shifted so "
                        "the front's density surface sits at x=0, so the "
                        "x-axis reads distance from the (sloping) front at "
                        "every depth.")
    p.add_argument("--perp-sigma0", type=float, default=None,
                   help="Density surface to follow with --perp-isopycnal "
                        "(default: sigma0 at the transect centre at the "
                        "shallowest level).")
    p.add_argument("--perp-max-crossings", type=int, default=1,
                   help="When auto-picking the perpendicular point, only "
                        "consider main-axis columns whose transect crosses the "
                        "front at most this many times (default 1 = a clean "
                        "single crossing, away from the front's squiggly "
                        "self-overlapping parts).")
    p.add_argument("--perp-allow-crossings", action="store_true",
                   help="Disable the crossing filter above; auto-pick the "
                        "field extremum anywhere along the axis.")
    p.add_argument("--list-perp-candidates", action="store_true",
                   help="Log the main-axis column index, (i,j), along-path km, "
                        "and transect crossing-count for each column, then "
                        "continue.  Helps choose --perp-point.")

    # Smoothing of the direction field (NOT the main-axis columns).
    p.add_argument("--smooth-normals", action="store_true",
                   help="Smooth the tangent/normal direction field before "
                        "throwing offsets + the perpendicular (off by "
                        "default).  Does NOT move the main-axis columns.")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Odd pixel window for --smooth-normals (default 5).")
    p.add_argument("--no-trim-offsets", dest="trim_offsets",
                   action="store_false",
                   help="Keep self-intersection loops in the offset lines "
                        "(shaded magenta) instead of trimming them.  By "
                        "default each offset polyline is 'sewn' shut so it "
                        "contains no crossings (looped columns become gray "
                        "gaps in the curtain).")
    p.set_defaults(trim_offsets=True)

    # Display.
    p.add_argument("--isopycnals", type=float, nargs="+", default=None,
                   help="Explicit sigma0 contour levels; otherwise auto-picked "
                        "from the cross-front surface contrast.")
    p.add_argument("--n-isopycnals", type=int, default=8,
                   help="Number of auto-picked isopycnal levels (default 8).")
    p.add_argument("--clim", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Color limits for the (transformed) color field; "
                        "default: registered style clim, else 2/98 percentile.")
    p.add_argument("--cmap", type=str, default=None,
                   help="Colormap for the color field (default: the field's "
                        "registered style cmap).")

    # Output.
    p.add_argument("--output-prefix", type=Path, required=True,
                   help="Output path prefix; the script appends "
                        "_{field}_{loc}_mainaxis.png / _..._offsets_n{N}.png / "
                        "_..._perp.png / _..._inset.png, where {field} is the "
                        "tile variable name, {loc} is the front's "
                        "lat/lon (e.g. lat36.38_lon-124.20), and {N} is "
                        "--n-offsets.  The lat/lon keeps different fronts from "
                        "overwriting each other.")
    p.add_argument("--no-inset", action="store_true",
                   help="Skip the map-view inset figure.")

    args = p.parse_args(argv)

    ij = (args.i is not None) and (args.j is not None)
    ll = (args.lat is not None) and (args.lon is not None)
    if ij == ll:
        p.error("Supply exactly one of (--i AND --j) or (--lat AND --lon).")
    args.locator_kind = "ij" if ij else "latlon"
    return args


# ---------------------------------------------------------------------------
# Isopycnal level picking (depth-aware percentile over the whole curtain)
# ---------------------------------------------------------------------------

def pick_isopycnal_levels(
    sigma0_clipped: np.ndarray,
    user_levels,
    n_levels: int,
) -> np.ndarray:
    """Choose sigma0 contour levels for the curtain isopycnals.

    Unlike the 3-D ``pick_isopycnals_across_front`` (which brackets the
    near-surface cross-front contrast), the curtain spans the full depth, so we
    bracket the 2/98 percentile of the whole cropped+clipped volume -- this
    yields contours that read across the entire depth-distance panel.

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        ``(K, J, I)`` cropped + depth-clipped sigma0.
    user_levels : sequence of float or None
        Explicit levels; returned verbatim when supplied.
    n_levels : int
        Number of evenly spaced levels when auto-picking.

    Returns
    -------
    numpy.ndarray
        1-D array of sigma0 contour values.
    """
    if user_levels is not None:
        return np.asarray(list(user_levels), dtype=np.float64)
    lo = float(np.nanpercentile(sigma0_clipped, 2))
    hi = float(np.nanpercentile(sigma0_clipped, 98))
    return np.linspace(lo, hi, int(n_levels))


# ---------------------------------------------------------------------------
# Map-view inset
# ---------------------------------------------------------------------------

def plot_map_inset(
    surface_field: np.ndarray,
    front_mask_full: np.ndarray,
    axis_path_cropped: np.ndarray,
    side_a, side_b,
    perp_path_cropped: np.ndarray,
    mark_jicropped,
    j_slice: slice,
    i_slice: slice,
    output_path: Path,
    *,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    color_title: str = "",
    nan_color: str = "#9e9e9e",
    trim: bool = True,
    title: str = "",
) -> Path:
    """Plan-view map of the bbox: surface field + axis + offsets + perp point.

    The background is the near-surface slice of the **color field** (e.g. Ri),
    in the same display space / colormap as the curtains, with a colorbar.

    Parameters
    ----------
    surface_field : numpy.ndarray
        2-D near-surface display-space color field (rect-tile-local frame,
        full tile) -- e.g. ``apply_transform(field_rect[k_ref], style)``.
    cmap : str, optional
        Colormap for the field background (default the field-style cmap).
    clim : tuple of (float, float), optional
        Color limits for the field; default 2/98 percentile of the bbox crop.
    color_title : str, optional
        Colorbar label (the field's display title).
    nan_color : str, optional
        Color for NaN cells (land / clipped / undefined); default neutral gray.
    front_mask_full : numpy.ndarray
        Full-tile boolean mask of the selected front.
    axis_path_cropped : numpy.ndarray
        ``(L, 2)`` main axis in the *cropped* frame.
    side_a, side_b : list of numpy.ndarray
        Offset polylines in the cropped frame.
    perp_path_cropped : numpy.ndarray
        Perpendicular transect in the cropped frame.
    mark_jicropped : tuple
        ``(j, i)`` of the marked perpendicular point in the cropped frame.
    j_slice, i_slice : slice
        Crop slices (to shift cropped coords back to tile-local for plotting).
    output_path : pathlib.Path
        PNG output.
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    surf_crop = surface_field[j_slice, i_slice]
    mask_crop = front_mask_full[j_slice, i_slice]
    extent = (i_slice.start, i_slice.stop, j_slice.start, j_slice.stop)

    fig, ax = plt.subplots(figsize=(9, 8))
    if clim is not None:
        vlo, vhi = clim
    elif np.isfinite(surf_crop).any():
        vlo = float(np.nanpercentile(surf_crop, 2))
        vhi = float(np.nanpercentile(surf_crop, 98))
    else:
        vlo, vhi = 0.0, 1.0
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)
    im = ax.imshow(np.ma.masked_invalid(surf_crop), origin="lower",
                   extent=extent, cmap=cmap_obj, vmin=vlo, vmax=vhi,
                   interpolation="nearest", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(color_title)

    di, dj = i_slice.start, j_slice.start

    def _plot(p, **kw):
        ax.plot(p[:, 1] + di, p[:, 0] + dj, **kw)

    def _plot_offset(p, **kw):
        # "Sew" the offset shut: drop self-intersection loops so the drawn line
        # has no crossings (matches the trimmed curtain).
        if trim:
            keep = curtains.trim_offset_loops(p)
            p = p[keep]
        _plot(p, **kw)

    # All front pixels (faint), then the main axis (bold).
    yy, xx = np.where(mask_crop)
    ax.scatter(xx + di, yy + dj, s=2, c="k", alpha=0.35,
               label="front pixels")
    _plot(axis_path_cropped, color="red", lw=2.0, label="main axis")
    for k, p in enumerate(side_a):
        _plot_offset(p, color="dodgerblue", lw=0.8, alpha=0.8,
                     label="offset +n" if k == 0 else None)
    for k, p in enumerate(side_b):
        _plot_offset(p, color="orange", lw=0.8, alpha=0.8,
                     label="offset -n" if k == 0 else None)
    _plot(perp_path_cropped, color="lime", lw=2.0, label="perpendicular")
    if mark_jicropped is not None:
        ax.scatter([mark_jicropped[1] + di], [mark_jicropped[0] + dj],
                   s=80, marker="*", c="lime", edgecolors="k", zorder=5,
                   label="perp point")

    ax.set_xlabel("i (rect tile-local)")
    ax.set_ylabel("j (rect tile-local)")
    ax.set_title(title or "Curtain geometry (plan view)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    Writes ``{prefix}_{field}_{loc}_mainaxis.png``,
    ``{prefix}_{field}_{loc}_offsets_n{N}.png``,
    ``{prefix}_{field}_{loc}_perp.png`` and (unless ``--no-inset``)
    ``{prefix}_{field}_{loc}_inset.png`` (``{loc}`` = ``lat{LAT}_lon{LON}``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    log = logging.getLogger("fronts_viz_curtain")

    # ---------- Load density + field tiles ----------
    log.info("Loading density tile %s", args.density_tile)
    ds = load_density_tile(args.density_tile)
    rect_i_start = int(tile_scalar(ds, "rect_i_start"))
    rect_j_start = int(tile_scalar(ds, "rect_j_start"))
    face_index = int(tile_scalar(ds, "face_index"))

    sigma0_face = ds["sigma0"].values  # (K, J_face, I_face)
    Z = ds["Z"].values.astype(np.float64)  # (K,), negative downward
    XC_face = ds["XC"].values
    YC_face = ds["YC"].values

    log.info("Loading field tile %s", args.field_tile)
    ds_field = load_tile(args.field_tile, var_name=args.field_name)
    try:
        check_tiles_consistent(ds, ds_field, "--density-tile", "--field-tile")
    except ValueError as err:
        raise SystemExit(str(err)) from err
    field_name = ds_field.attrs["tile_var_name"]
    field_face = ds_field[field_name].values
    if field_face.shape != sigma0_face.shape:
        raise SystemExit(
            f"--field-tile variable '{field_name}' has shape "
            f"{field_face.shape}, expected {sigma0_face.shape}."
        )
    field_style = get_style(field_name)
    # Human-readable field name (e.g. "Richardson number N2/S2"); fall back to the
    # tile variable name (e.g. "Ri") when no long_name attr.
    field_label = ds_field[field_name].attrs.get("long_name", field_name)
    # Filesystem-safe token built into the output filenames (e.g. "Ri", "wB").
    field_tag = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in str(field_name)
    ).strip("_") or "field"
    log.info("Coloring curtains by %s (%s, transform=%s)",
             field_name, field_label,
             args.field_transform or field_style.transform)

    # ---------- Frame remap ----------
    log.info("Building tile lookup + remapping to rect-tile-local frame")
    j_tile_lookup, i_tile_lookup = build_tile_lookup(
        rect_i_start, rect_j_start, face_index,
    )
    rect_j_slice = slice(rect_j_start, rect_j_start + j_tile_lookup.shape[0])
    rect_i_slice = slice(rect_i_start, rect_i_start + j_tile_lookup.shape[1])
    labels_tile = load_labels_tile(args.labels, rect_j_slice, rect_i_slice)

    sigma0_rect = remap_to_rect(sigma0_face, j_tile_lookup, i_tile_lookup)
    XC_rect = remap_to_rect(XC_face, j_tile_lookup, i_tile_lookup)
    YC_rect = remap_to_rect(YC_face, j_tile_lookup, i_tile_lookup)
    field_rect = remap_to_rect(field_face, j_tile_lookup, i_tile_lookup)

    # ---------- Tile lat/lon bbox + lat/lon-locator guard ----------
    # The tile is the full TILE_SIZE x TILE_SIZE chunk, not just the region
    # around one front.  Log its lat/lon extent every run so the valid
    # --lat/--lon range is always visible.
    lat_min, lat_max = float(YC_rect.min()), float(YC_rect.max())
    lon_min, lon_max = float(XC_rect.min()), float(XC_rect.max())
    log.info("Tile lat/lon bbox: lat [%.3f, %.3f], lon [%.3f, %.3f]",
             lat_min, lat_max, lon_min, lon_max)
    if args.locator_kind == "latlon":
        # --lat/--lon snaps to the nearest in-tile pixel via argmin, so an
        # out-of-tile value would *silently* pick a corner front.  Hard-error
        # (printing the bounds) instead.
        pad = 0.05 * max(lat_max - lat_min, lon_max - lon_min)
        if not (lat_min - pad <= args.lat <= lat_max + pad
                and lon_min - pad <= args.lon <= lon_max + pad):
            raise SystemExit(
                f"--lat {args.lat} --lon {args.lon} is outside this tile's "
                f"lat/lon bbox: lat [{lat_min:.3f}, {lat_max:.3f}], "
                f"lon [{lon_min:.3f}, {lon_max:.3f}] (pad {pad:.3f} deg).  "
                "The lat/lon locator snaps to the nearest in-tile pixel, so an "
                "out-of-tile value would pick the wrong front.  Use --i/--j "
                "(rect-grid indices) or a lat/lon inside the bbox above."
            )

    # ---------- Pick the front ----------
    locator = ((args.i, args.j) if args.locator_kind == "ij"
               else (args.lat, args.lon))
    label, j_pick, i_pick = pick_front_label(
        labels_tile, args.locator_kind, locator,
        rect_i_start, rect_j_start, XC_rect, YC_rect,
    )
    lon_pick = float(XC_rect[j_pick, i_pick])
    lat_pick = float(YC_rect[j_pick, i_pick])
    # Location tag baked into the output filenames so different fronts don't
    # overwrite each other (e.g. "lat36.38_lon-124.20").
    loc_tag = f"lat{lat_pick:.2f}_lon{lon_pick:.2f}"
    log.info("Selected front label=%d at tile-local (j=%d, i=%d) lon=%.3f lat=%.3f",
             label, j_pick, i_pick, lon_pick, lat_pick)

    # ---------- Crop + depth clip ----------
    j_slice, i_slice = front_bbox_and_crop(labels_tile, label, margin=args.margin)
    sigma0_cropped = sigma0_rect[:, j_slice, i_slice]
    field_cropped = field_rect[:, j_slice, i_slice]
    front_mask_full = (labels_tile == label)
    front_mask_cropped = front_mask_full[j_slice, i_slice]

    z_mld, k_mld = mixed_layer_depth_field(sigma0_cropped, Z)
    sigma0_clipped, Z_clipped, k_clip = truncate_depth(
        sigma0_cropped, Z, k_mld, n_below=args.n_below,
    )
    field_clipped = field_cropped[:k_clip]
    log.info("Curtain depth extent: k=%d levels, z=[%.1f, %.1f] m",
             k_clip, float(Z_clipped[0]), float(Z_clipped[-1]))

    # Transform the color field into display space (e.g. log10 Ri).
    color_display = apply_transform(
        field_clipped, field_style,
        clip_override=(tuple(args.field_clip)
                       if args.field_clip is not None else None),
        transform_override=args.field_transform,
    )

    # ---------- Geometry: main axis, metrics, offsets, perpendicular ----------
    axis_path = curtains.extract_main_axis(front_mask_cropped)  # cropped frame
    log.info("Main axis: %d pixels", axis_path.shape[0])

    # path_metrics wants lon/lat in the cropped frame for km distances.
    XC_crop = XC_rect[j_slice, i_slice]
    YC_crop = YC_rect[j_slice, i_slice]
    metrics = curtains.path_metrics(
        axis_path, XC_crop, YC_crop,
        smooth=args.smooth_normals, smooth_window=args.smooth_window,
    )

    # Color limits + colormap from the field style unless overridden.
    if args.clim is not None:
        clim = tuple(args.clim)
    else:
        clim = default_clim(
            color_display, field_style,
            clip_override=(tuple(args.field_clip)
                           if args.field_clip is not None else None),
            transform_override=args.field_transform,
        )
    cmap = args.cmap or field_style.cmap
    color_title = field_style.title or field_name

    # Isopycnal levels span the full curtain depth.
    levels = pick_isopycnal_levels(
        sigma0_clipped, args.isopycnals, args.n_isopycnals,
    )
    log.info("Isopycnal levels (kg m^-3): %s",
             ", ".join(f"{lv:.3f}" for lv in levels))

    # Transect crossing-count per column (used for the auto-pick filter and
    # the optional candidate listing).
    crossings = curtains.transect_front_crossings(
        axis_path, metrics["normals"], front_mask_cropped, args.perp_half_width,
    )

    if args.list_perp_candidates:
        log.info("Perpendicular-point candidates (column: i,j  km  crossings):")
        dist_km = metrics["dist_km"]
        for c in range(axis_path.shape[0]):
            km = "" if dist_km is None else f"{dist_km[c]:7.2f} km"
            log.info("  col %4d: i=%4d j=%4d  %s  crossings=%d",
                     c, int(axis_path[c, 1]) + i_slice.start,
                     int(axis_path[c, 0]) + j_slice.start, km, crossings[c])

    # Perpendicular point: user index or the field extremum along the axis.
    axis_color = curtains.sample_curtain(color_display, axis_path)
    if args.perp_point is not None:
        perp_idx = int(np.clip(args.perp_point, 0, axis_path.shape[0] - 1))
        log.info("Perpendicular point: axis column %d (user --perp-point); "
                 "transect crosses front %d time(s)",
                 perp_idx, crossings[perp_idx])
    else:
        search = axis_color.copy()
        if not args.perp_allow_crossings:
            # Exclude columns whose transect re-crosses the front (the squiggly
            # hook); NaN their color so the extremum search ignores them.
            bad = crossings > args.perp_max_crossings
            search[:, bad] = np.nan
            if not np.isfinite(search).any():
                log.warning("No columns with <= %d crossings; falling back to "
                            "the whole axis for the perpendicular point.",
                            args.perp_max_crossings)
                search = axis_color
        perp_idx = curtains.pick_extremum_index(search, mode=args.extremum)
        log.info("Perpendicular point: axis column %d (%s of %s, "
                 "<= %d crossings); transect crosses front %d time(s)",
                 perp_idx, args.extremum, color_title,
                 args.perp_max_crossings, crossings[perp_idx])

    perp_path = curtains.perpendicular_path(
        axis_path, metrics["normals"], perp_idx, args.perp_half_width,
    )

    # MLD sampled along the axis (depths, negative metres) for the overlay.
    # k_mld is (J', I'); convert to depth then sample along the axis pixels.
    mld_depth_2d = np.where(k_mld >= 0, z_mld, np.nan)
    jj = np.clip(np.round(axis_path[:, 0]).astype(int), 0, mld_depth_2d.shape[0] - 1)
    ii = np.clip(np.round(axis_path[:, 1]).astype(int), 0, mld_depth_2d.shape[1] - 1)
    mld_curtain = mld_depth_2d[jj, ii]

    # ---------- Render the three figures ----------
    # Filenames carry the field tag and the front's lat/lon (so different
    # fronts don't overwrite); the offsets figure also carries the offset
    # count.  e.g. {prefix}_Ri_lat36.38_lon-124.20_mainaxis.png
    prefix = args.output_prefix
    stem = f"{prefix.name}_{field_tag}_{loc_tag}"
    out_main = prefix.with_name(f"{stem}_mainaxis.png")
    out_off = prefix.with_name(f"{stem}_offsets_n{args.n_offsets}.png")
    out_perp = prefix.with_name(f"{stem}_perp.png")

    curtains.figure_main_axis(
        color_display, sigma0_clipped, Z_clipped, axis_path, metrics, out_main,
        levels=levels, clim=clim, cmap=cmap, color_title=color_title,
        mld_curtain=mld_curtain, mark_index=perp_idx,
        title=f"Main-axis curtain — {field_label} (front {label})",
    )
    log.info("Wrote %s", out_main)

    curtains.figure_offsets(
        color_display, sigma0_clipped, Z_clipped, axis_path, metrics,
        args.n_offsets, out_off,
        levels=levels, clim=clim, cmap=cmap, color_title=color_title,
        mark_index=perp_idx, trim=args.trim_offsets,
        title=f"Along-front curtains + offsets — {field_label} (front {label})",
    )
    log.info("Wrote %s", out_off)

    curtains.figure_perpendicular(
        color_display, sigma0_clipped, Z_clipped, perp_path,
        args.perp_half_width, out_perp,
        XC_rect=XC_crop, YC_rect=YC_crop,
        levels=levels, clim=clim, cmap=cmap, color_title=color_title,
        follow_isopycnal=args.perp_isopycnal,
        target_sigma0=args.perp_sigma0,
        title=(f"Cross-front curtain — {field_label} "
               f"at axis col {perp_idx} (front {label})"),
    )
    log.info("Wrote %s", out_perp)

    # ---------- Flattened isopycnal surface ----------
    if args.isopycnal_curtain:
        out_iso = prefix.with_name(f"{stem}_isopycnal.png")
        curtains.figure_isopycnal_surface(
            color_display, sigma0_clipped, Z_clipped, axis_path, metrics,
            out_iso,
            half_width=args.isopycnal_half_width,
            target_sigma0=args.isopycnal_sigma0,
            clim=clim, cmap=cmap, color_title=color_title,
            mark_index=perp_idx,
            title=(f"{field_label} on the front's isopycnal "
                   f"(front {label})"),
        )
        log.info("Wrote %s", out_iso)

    # ---------- Map-view inset ----------
    if not args.no_inset:
        out_inset = prefix.with_name(f"{stem}_inset.png")
        side_a, side_b = curtains.offset_paths(
            axis_path, metrics["normals"], args.n_offsets,
        )
        # Background = near-surface slice of the COLOR FIELD (display space),
        # same colormap/clim as the curtains.
        k_ref = int(np.abs(np.abs(Z) - 10.0).argmin())
        field_surf = apply_transform(
            field_rect[k_ref], field_style,
            clip_override=(tuple(args.field_clip)
                           if args.field_clip is not None else None),
            transform_override=args.field_transform,
        )
        plot_map_inset(
            field_surf, front_mask_full, axis_path, side_a, side_b, perp_path,
            (axis_path[perp_idx, 0], axis_path[perp_idx, 1]),
            j_slice, i_slice, out_inset,
            cmap=cmap, clim=clim, color_title=color_title, nan_color=NAN_COLOR,
            trim=args.trim_offsets,
            title=(f"Front {label} curtain geometry "
                   f"(lon={float(XC_rect[j_pick, i_pick]):.2f}, "
                   f"lat={float(YC_rect[j_pick, i_pick]):.2f})"),
        )
        log.info("Wrote %s", out_inset)


if __name__ == "__main__":
    main()
