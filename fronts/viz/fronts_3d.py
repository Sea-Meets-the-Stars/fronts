"""
PyVista builders for 3-D visualisations of labelled fronts in LLC4320 tiles.

This module wraps the generic helpers in ``pv_helpers.py`` with the
front-specific operations needed by
``fronts/scripts/fronts_viz_3d.py``:

* `front_bbox_and_crop`     -- bbox of a labelled front + margin.
* `truncate_depth`          -- clip a 3-D volume to a few levels below MLD.
* `build_pyvista_grid`      -- wrap a cropped+clipped sigma0 array as a
                               PyVista ``RectilinearGrid``.
* `decompose_front_branches` -- split a thinned front mask into its
                                constituent single-branch polylines.
* `build_front_curtain`     -- extrude each branch into a vertical
                                sigma0-coloured ribbon.
* `pick_isopycnals_across_front` -- choose default isopycnal levels
                                    bracketing the front's surface
                                    density contrast.
* `render_3d`               -- orchestrate the scene (isopycnals or
                                volume) + curtain + axes.

Imports follow the patterns in ``.claude/skills/pyvista-scientific-viz/
references/patterns_reference.md`` (in particular the Fortran-order
ravel for RectilinearGrid scalar arrays).
"""

# stdlib
from __future__ import annotations
from typing import Sequence

# numerical / IO
import numpy as np
from scipy import ndimage as scimg

# 3-D rendering
import pyvista as pv

# Generic helpers shipped with the repo.  These provide the scientific
# theme, off-screen rendering, and labelled scalar bar.
from fronts.viz.pv_helpers import (
    ensure_display,
    new_plotter,
    scientific_theme,
    add_scalar_field,
)


# ---------------------------------------------------------------------------
# Geometry: bbox + depth-clip
# ---------------------------------------------------------------------------

# Moved to fronts/viz/geometry.py so the 2-D figures do not need PyVista.
# Re-exported here so existing imports keep working unchanged.
from fronts.viz.geometry import (      # noqa: E402,F401
    front_bbox_and_crop,
    truncate_depth,
)


# ---------------------------------------------------------------------------
# 3-D grid construction
# ---------------------------------------------------------------------------

def build_pyvista_grid(
    sigma0_clipped: np.ndarray,
    Z_clipped: np.ndarray,
    j_slice: slice,
    i_slice: slice,
    zscale: float = 50.0,
    *,
    mask_2d: np.ndarray | None = None,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> pv.RectilinearGrid:
    """Wrap a cropped+clipped sigma0 array as a PyVista ``RectilinearGrid``.

    The grid is regular in ``i`` and ``j`` (rect-tile pixel coordinates,
    matching the user's locator frame) and irregular in ``z`` (the LLC
    depth axis, multiplied by ``zscale`` so the depth direction is
    visible against the much wider horizontal extent).

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- the field to render.
    Z_clipped : numpy.ndarray
        Shape ``(K_clip,)`` -- depths in metres (negative downward).
    j_slice, i_slice : slice
        The rect-frame slices the field was cropped to; used to position
        the grid in absolute tile-local pixel coordinates so the figure's
        x/y axes show meaningful values.
    zscale : float, optional
        Vertical exaggeration factor applied to ``Z_clipped`` (default 50).
    mask_2d : numpy.ndarray, optional
        Optional 2-D boolean mask of shape ``(J', I')``.  When supplied,
        sigma0 is set to ``NaN`` everywhere the mask is False, so
        downstream filters (``grid.contour``, etc.) only return surfaces
        inside the mask.  Use this together with
        :func:`dilate_front_mask` to render iso-surfaces only in a
        dilated region around the selected front.
    extra_fields : dict of str -> numpy.ndarray, optional
        Additional point-data scalars to stamp alongside ``sigma0``, each
        of the same ``(K_clip, J', I')`` shape (e.g. a transformed
        Richardson-number volume named ``"log10_Ri"``).  VTK's ``contour``
        filter interpolates *all* point arrays onto extracted iso-surfaces,
        so any field stamped here can be used to color sigma0 iso-surfaces
        via ``render_3d(color_scalar=...)``.  The ``mask_2d`` NaN-out (when
        given) is applied to these fields too.

    Returns
    -------
    pyvista.RectilinearGrid
        Grid with ``grid["sigma0"]`` (and each ``extra_fields`` entry)
        stamped using Fortran order (the documented silent-bug trap in
        PyVista's RectilinearGrid).
    """
    K_clip, Jp, Ip = sigma0_clipped.shape

    # Coordinate arrays are POINT counts (Jp, Ip, K_clip).  Using the
    # rect-frame slice origin keeps the axes labelled in the same
    # coordinate system as the user's --i/--j input.
    x = np.arange(i_slice.start, i_slice.start + Ip, dtype=np.float64)
    y = np.arange(j_slice.start, j_slice.start + Jp, dtype=np.float64)
    z = Z_clipped.astype(np.float64) * float(zscale)

    grid = pv.RectilinearGrid(x, y, z)

    # VTK expects Fortran-ordered scalar arrays for image/rectilinear grids;
    # the patterns reference flags C-order as the most common silent bug.
    # The arrays have axes (k, j, i); we want the iteration order to match
    # (x, y, z) = (i, j, k), so transpose first then ravel('F').
    def _stamp(name: str, arr: np.ndarray) -> None:
        if arr.shape != (K_clip, Jp, Ip):
            raise ValueError(
                f"field {name!r} shape {arr.shape} does not match "
                f"sigma0_clipped {(K_clip, Jp, Ip)}."
            )
        field = arr.astype(np.float32, copy=True)
        if mask_2d is not None:
            if mask_2d.shape != (Jp, Ip):
                raise ValueError(
                    f"mask_2d shape {mask_2d.shape} does not match the "
                    f"(J, I) axes of the volume ({Jp}, {Ip})."
                )
            # Broadcast the 2-D mask along k so the same (j,i) pixels are
            # active at every depth; contour() will then trace surfaces
            # restricted to the columns where the mask is True.
            field[:, ~mask_2d] = np.nan
        grid[name] = np.transpose(field, (2, 1, 0)).ravel(order="F")

    _stamp("sigma0", sigma0_clipped)
    if extra_fields:
        for name, arr in extra_fields.items():
            _stamp(name, arr)
    # contour() defaults to the active scalars; keep that sigma0 so the
    # geometry is always density-driven regardless of stamping order.
    grid.set_active_scalars("sigma0")
    return grid


def dilate_front_mask(
    front_mask: np.ndarray,
    iterations: int = 2,
) -> np.ndarray:
    """Binary-dilate a 2-D front mask by ``iterations`` pixels (8-connectivity).

    Convenience wrapper around :func:`scipy.ndimage.binary_dilation` that
    encodes the project's 8-connectivity convention.  The dilated mask
    captures the front and a halo around it -- useful for restricting
    iso-surface rendering to "near the front" rather than the whole
    bbox.

    Parameters
    ----------
    front_mask : numpy.ndarray
        2-D boolean mask, True at the selected front's pixels.
    iterations : int, optional
        Number of binary-dilation passes.  Each pass adds one pixel of
        halo.  Default 2.

    Returns
    -------
    numpy.ndarray
        2-D boolean mask of the same shape, dilated by ``iterations``
        pixels along the 8-neighbour structuring element.
    """
    struct = scimg.generate_binary_structure(2, 2)  # 8-connectivity
    return scimg.binary_dilation(
        front_mask, structure=struct, iterations=int(iterations),
    )


def front_volume_clim(
    sigma0_clipped: np.ndarray,
    dilated_mask_2d: np.ndarray,
    *,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[float, float]:
    """Percentile-based colour limits computed inside the dilated-front column.

    Unlike :func:`mixed_layer_clim` (which uses every column above its
    own per-column MLD across the full bbox), this helper restricts the
    pool to the dilated-front 2-D region at *every* depth.  That gives
    even tighter contrast for visualising the front's iso-surface tilt
    structure.

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- cropped + depth-clipped sigma0.
    dilated_mask_2d : numpy.ndarray
        2-D boolean mask of shape ``(J', I')`` selecting the front
        columns plus a small halo (typically from
        :func:`dilate_front_mask`).
    percentile_low, percentile_high : float, optional
        Percentile bounds (default 2/98).

    Returns
    -------
    tuple of (float, float)
        ``(lo, hi)`` colour limits.  Falls back to the full clipped
        volume percentiles if the dilated mask is empty (shouldn't
        normally happen).
    """
    if not dilated_mask_2d.any():
        return (
            float(np.nanpercentile(sigma0_clipped, percentile_low)),
            float(np.nanpercentile(sigma0_clipped, percentile_high)),
        )
    pool = sigma0_clipped[:, dilated_mask_2d]
    return (
        float(np.nanpercentile(pool, percentile_low)),
        float(np.nanpercentile(pool, percentile_high)),
    )


# ---------------------------------------------------------------------------
# Front polyline decomposition + curtain
# ---------------------------------------------------------------------------

# Moved to fronts/viz/geometry.py (pure NumPy/SciPy); re-exported so
# existing imports keep working unchanged.
from fronts.viz.geometry import decompose_front_branches  # noqa: E402,F401


def build_front_curtain(
    front_mask: np.ndarray,
    sigma0_clipped: np.ndarray,
    Z_clipped: np.ndarray,
    j_slice: slice,
    i_slice: slice,
    zscale: float = 50.0,
) -> pv.MultiBlock:
    """Extrude each branch of a thinned front mask into a sigma0-coloured ribbon.

    Each branch becomes a ``pv.StructuredGrid`` whose horizontal
    cross-section is the branch polyline and whose vertical extent
    spans ``Z_clipped`` (multiplied by ``zscale`` to match the volume).
    sigma0 is sampled along the ribbon via
    :func:`scipy.ndimage.map_coordinates` and attached as a point-data
    scalar named ``"sigma0"`` so the renderer can colour by it.

    Parameters
    ----------
    front_mask : numpy.ndarray
        2-D boolean mask on the *cropped* rect-frame window (same shape
        as the j/i axes of ``sigma0_clipped``).
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')``; the same field passed to
        :func:`build_pyvista_grid`.
    Z_clipped : numpy.ndarray
        1-D depths, length ``K_clip``, in metres (negative downward).
    j_slice, i_slice : slice
        Tile-local rect-frame slices used to position the ribbons in the
        same coordinate system as the grid built by
        :func:`build_pyvista_grid`.
    zscale : float, optional
        Vertical exaggeration applied to ``Z_clipped``.

    Returns
    -------
    pyvista.MultiBlock
        One ``StructuredGrid`` per branch.  Empty if the front mask has
        no True pixels.
    """
    branches = decompose_front_branches(front_mask)
    multiblock = pv.MultiBlock()
    if not branches:
        return multiblock

    # Cache the branch list on the MultiBlock so downstream helpers
    # (e.g. build_front_top_marker) can reuse the decomposition without
    # re-running the DFS.
    multiblock._fronts_branches = branches  # type: ignore[attr-defined]
    K_clip = Z_clipped.shape[0]
    z_axis = Z_clipped.astype(np.float64) * float(zscale)

    for b_idx, branch in enumerate(branches):
        L = branch.shape[0]
        # Branch coordinates in the cropped frame for sampling sigma0;
        # then shift into absolute tile-local pixel coords for the grid.
        j_local = branch[:, 0].astype(np.float64)
        i_local = branch[:, 1].astype(np.float64)

        # Sample sigma0 along the (k, j, i) volume.  We build coords of
        # shape (3, K_clip * L): axis 0 = k, axis 1 = branch step.
        kk, ss = np.meshgrid(np.arange(K_clip), np.arange(L), indexing="ij")
        coords = np.stack([
            kk.ravel().astype(np.float64),
            np.repeat(j_local[None, :], K_clip, axis=0).ravel(),
            np.repeat(i_local[None, :], K_clip, axis=0).ravel(),
        ], axis=0)
        sampled = scimg.map_coordinates(
            sigma0_clipped, coords, order=1, mode="nearest",
        ).reshape(K_clip, L)

        # Build the StructuredGrid.  Dimensions are (i, j, k) = (L, 1, K).
        # We add a small thickness in y so the ribbon is visible as a
        # surface rather than a zero-area collapse.  Using 1 point in the
        # second axis keeps it a 2-D ribbon embedded in 3-D space.
        x_pts = (i_local + i_slice.start)
        y_pts = (j_local + j_slice.start)
        # Replicate the polyline once per depth level to get a structured
        # ribbon: shape (K_clip, L) per coordinate.
        Xg = np.broadcast_to(x_pts[None, :], (K_clip, L)).astype(np.float64)
        Yg = np.broadcast_to(y_pts[None, :], (K_clip, L)).astype(np.float64)
        Zg = np.broadcast_to(z_axis[:, None], (K_clip, L)).astype(np.float64)
        ribbon = pv.StructuredGrid(Xg, Yg, Zg)
        ribbon["sigma0"] = sampled.astype(np.float32).ravel(order="F")
        multiblock.append(ribbon, name=f"branch_{b_idx:03d}")

    return multiblock


# ---------------------------------------------------------------------------
# Isopycnal-level picking
# ---------------------------------------------------------------------------

def pick_isopycnals_across_front(
    sigma0_clipped: np.ndarray,
    front_mask_cropped: np.ndarray,
    Z_clipped: np.ndarray,
    user_levels: Sequence[float] | None = None,
    *,
    n_levels: int = 5,
    reference_depth_m: float = 10.0,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    buffer_radius: int = 1,
) -> np.ndarray:
    """Choose default isopycnal levels bracketing the front's surface contrast.

    If ``user_levels`` is supplied it is returned verbatim.  Otherwise
    the function samples sigma0 on the LLC level closest to
    ``reference_depth_m`` (matching the MLD reference depth used
    elsewhere in this repo), masks out the front itself + a 1-pixel
    buffer, then returns ``n_levels`` values evenly spaced between the
    5th and 95th percentile of the remaining values inside the cropped
    bbox.  This is round-3 option (i) ("near-surface percentile bracket
    excluding the front itself").

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- cropped + depth-clipped sigma0.
    front_mask_cropped : numpy.ndarray
        2-D boolean mask, shape ``(J', I')``, True at the selected
        front's pixels.
    Z_clipped : numpy.ndarray
        1-D depths, length ``K_clip``.
    user_levels : sequence of float, optional
        If provided, returned without modification.
    n_levels : int, optional
        Number of evenly spaced levels (default 5).
    reference_depth_m : float, optional
        Depth in metres at which to sample the cross-front contrast
        (default 10 m, matching the MLD reference depth).
    percentile_low, percentile_high : float, optional
        Percentiles used to define the cross-front sigma0 range.
    buffer_radius : int, optional
        Pixel radius of the dilation applied to the front mask before
        excluding it from the percentile pool (default 1).

    Returns
    -------
    numpy.ndarray
        1-D float array of isopycnal sigma0 values, length ``n_levels``.
    """
    if user_levels is not None:
        return np.asarray(list(user_levels), dtype=np.float64)

    # Reference slice -- closest LLC level to the requested reference depth.
    k_ref = int(np.abs(np.abs(Z_clipped) - float(reference_depth_m)).argmin())
    ref_slice = sigma0_clipped[k_ref]  # (J', I')

    # Dilate the front mask by buffer_radius pixels so we don't include
    # pixels that straddle the front itself.
    if buffer_radius > 0:
        struct = scimg.generate_binary_structure(2, 2)  # 8-connectivity
        front_buffered = scimg.binary_dilation(
            front_mask_cropped, structure=struct, iterations=buffer_radius,
        )
    else:
        front_buffered = front_mask_cropped

    pool = ref_slice.copy()
    pool[front_buffered] = np.nan
    sigma_lo = float(np.nanpercentile(pool, percentile_low))
    sigma_hi = float(np.nanpercentile(pool, percentile_high))
    if not (np.isfinite(sigma_lo) and np.isfinite(sigma_hi)):
        # Degenerate case (e.g. front covers all pixels in the bbox).
        # Fall back to percentiles over the whole reference slice.
        sigma_lo = float(np.nanpercentile(ref_slice, percentile_low))
        sigma_hi = float(np.nanpercentile(ref_slice, percentile_high))
    return np.linspace(sigma_lo, sigma_hi, n_levels)


# ---------------------------------------------------------------------------
# Top-layer marker (front projected onto the surface for at-a-glance location)
# ---------------------------------------------------------------------------

def build_front_top_marker(
    curtain: pv.MultiBlock,
    Z_clipped: np.ndarray,
    j_slice: slice,
    i_slice: slice,
    zscale: float = 50.0,
    *,
    tube_radius: float = 1.5,
    elevate_pixels: float = 0.0,
) -> pv.MultiBlock:
    """Build a bright polyline at the surface so the front's location is visible from above.

    The 3-D scene's volume render or isopycnal stack can hide the front
    when viewed from a low oblique camera.  This helper produces a
    `pv.MultiBlock` of tube-thickened polylines sitting at
    ``Z_clipped[0] * zscale + elevate_pixels`` so the front shows up as a
    clean line on the top face of the scene, regardless of opacity.

    Parameters
    ----------
    curtain : pyvista.MultiBlock
        Output of :func:`build_front_curtain`.  The branch polylines are
        reused so the marker matches the curtain exactly.
    Z_clipped : numpy.ndarray
        1-D depths, length ``K_clip``; only ``Z_clipped[0]`` is used.
    j_slice, i_slice : slice
        Tile-local rect-frame slices used to position the marker in the
        same coordinate system as the grid built by
        :func:`build_pyvista_grid`.  The cached branches are stored in
        cropped coordinates so we must add the slice origins back.
    zscale : float, optional
        Vertical exaggeration applied to ``Z_clipped`` (must match the
        zscale passed to :func:`build_pyvista_grid`).
    tube_radius : float, optional
        Radius of the cylindrical tube wrapped around each polyline, in
        the same units as the i/j axes (rect-frame pixels).  Default 1.5.
    elevate_pixels : float, optional
        Extra Z offset above the top layer so the marker isn't hidden
        behind transparent contour surfaces.  Default 0.

    Returns
    -------
    pyvista.MultiBlock
        One ``PolyData`` (after ``.tube(...)``) per branch.  Empty if the
        curtain has no branches.
    """
    out = pv.MultiBlock()
    branches = getattr(curtain, "_fronts_branches", None)
    if branches is None or not branches:
        return out

    z_top = float(Z_clipped[0]) * float(zscale) + float(elevate_pixels)
    di, dj = float(i_slice.start), float(j_slice.start)
    for b_idx, branch in enumerate(branches):
        if branch.shape[0] < 2:
            # Single-pixel branch: drop a tiny sphere as a placeholder.
            pt = pv.PolyData(np.array([
                [float(branch[0, 1]) + di, float(branch[0, 0]) + dj, z_top],
            ]))
            out.append(pt, name=f"top_branch_{b_idx:03d}")
            continue
        pts = np.column_stack([
            branch[:, 1].astype(np.float64) + di,  # i -> x in tile-local frame
            branch[:, 0].astype(np.float64) + dj,  # j -> y in tile-local frame
            np.full(branch.shape[0], z_top),
        ])
        # Build a single polyline cell so .tube() can wrap it.
        n = pts.shape[0]
        poly = pv.PolyData(pts)
        # VTK polyline cell connectivity: [n, idx0, idx1, ..., idx_{n-1}].
        poly.lines = np.hstack([[n], np.arange(n)]).astype(np.int32)
        tube = poly.tube(radius=tube_radius, n_sides=12)
        out.append(tube, name=f"top_branch_{b_idx:03d}")
    return out


# ---------------------------------------------------------------------------
# Contrast helper -- focus the colour scale on the mixed layer
# ---------------------------------------------------------------------------

def mixed_layer_clim(
    sigma0_clipped: np.ndarray,
    k_mld_cropped: np.ndarray,
    *,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    pad_levels: int = 0,
) -> tuple[float, float]:
    """Percentile-based colour limits computed inside the mixed layer only.

    Stretching ``--clim`` over the full depth-clipped volume crushes the
    cross-front contrast that lives in the upper ~80 m, because the
    deeper levels span a much wider density range.  This helper restricts
    the percentile pool to columns *above* ``k_mld_cropped`` per cell.

    Parameters
    ----------
    sigma0_clipped : numpy.ndarray
        Shape ``(K_clip, J', I')`` -- cropped + depth-clipped sigma0.
    k_mld_cropped : numpy.ndarray
        Shape ``(J', I')`` -- per-column MLD level index (-1 where
        undefined).
    percentile_low, percentile_high : float, optional
        Percentile bounds (default 2/98).
    pad_levels : int, optional
        Number of model levels to include below the per-column MLD when
        building the percentile pool (default 0 -- strict mixed layer).

    Returns
    -------
    tuple of (float, float)
        ``(lo, hi)`` colour limits.  Falls back to the full-volume
        percentile when the mixed-layer pool is empty.
    """
    K, J, I = sigma0_clipped.shape
    # Broadcast (k_indices, k_mld_cropped) to build a per-cell "in ML" mask.
    k_indices = np.arange(K, dtype=np.int32)[:, None, None]
    k_lim = np.where(k_mld_cropped >= 0, k_mld_cropped + pad_levels, -1)
    in_ml = (k_indices <= k_lim[None, :, :]) & (k_lim[None, :, :] >= 0)
    pool = np.where(in_ml, sigma0_clipped, np.nan)
    if not np.isfinite(pool).any():
        # No MLD pixels in the bbox: fall back to the whole clipped volume.
        return (
            float(np.nanpercentile(sigma0_clipped, percentile_low)),
            float(np.nanpercentile(sigma0_clipped, percentile_high)),
        )
    return (
        float(np.nanpercentile(pool, percentile_low)),
        float(np.nanpercentile(pool, percentile_high)),
    )


# ---------------------------------------------------------------------------
# Front iso-surface (single-level, used as a tilt indicator)
# ---------------------------------------------------------------------------

def build_front_isosurface(
    grid: pv.RectilinearGrid,
    level: float,
) -> pv.PolyData:
    """Single sigma0 iso-surface — naturally tilts with depth at a front.

    Picking one representative cross-front sigma0 (e.g. the median of
    the 5 levels returned by :func:`pick_isopycnals_across_front`) and
    contouring the grid at that value isolates the 3-D "front
    surface".  Because density surfaces slope across a front, the
    resulting iso-surface tilts with depth -- with denser water on one
    side and lighter water on the other -- which is exactly the
    structure the user wants to see.

    Parameters
    ----------
    grid : pyvista.RectilinearGrid
        Grid built by :func:`build_pyvista_grid`.
    level : float
        sigma0 value (kg m^-3) to contour.

    Returns
    -------
    pyvista.PolyData
        Triangulated iso-surface.  Possibly empty if ``level`` lies
        outside the data range.
    """
    return grid.contour(isosurfaces=[float(level)], scalars="sigma0")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_3d(
    grid: pv.RectilinearGrid,
    curtain: pv.MultiBlock,
    levels: np.ndarray,
    mode: str = "isopycnals",
    *,
    clim: tuple[float, float] | None = None,
    cmap_volume: str = "viridis",
    cmap_curtain: str = "magma",
    opacity: str = "sigmoid",
    zscale: float = 50.0,
    show: bool = False,
    top_marker: pv.MultiBlock | None = None,
    top_marker_color: str = "red",
    top_marker_opacity: float = 0.6,
    front_iso: pv.PolyData | None = None,
    front_iso_color: str = "orange",
    front_iso_opacity: float = 0.95,
    draw_curtain: bool = False,
    color_scalar: str = "sigma0",
    color_title: str | None = None,
    nan_color: str = "#9e9e9e",
    nan_opacity: float = 1.0,
    front_iso_use_color_scalar: bool = True,
    font_size: int = 56,
    title_font_size: int = 60,
    label_font_size: int = 44,
) -> pv.Plotter:
    """Assemble the 3-D scene: isopycnals / volume + curtain + top marker + axes.

    Geometry is always sigma0-driven (contours extract density surfaces);
    *coloring* follows ``color_scalar``, which may be any point-data array
    stamped on the grid via ``build_pyvista_grid(extra_fields=...)`` --
    VTK interpolates every point array onto the extracted iso-surfaces.
    With the default ``color_scalar='sigma0'`` the scene is identical to
    the classic single-field behaviour.

    Parameters
    ----------
    grid : pyvista.RectilinearGrid
        Cropped + depth-clipped sigma0 grid from :func:`build_pyvista_grid`.
    curtain : pyvista.MultiBlock
        Per-branch sigma0-coloured ribbons from :func:`build_front_curtain`.
    levels : numpy.ndarray
        Isopycnal sigma0 values; used only in ``mode='isopycnals'``.
    mode : {'isopycnals', 'volume'}, optional
        ``'isopycnals'`` (default) draws ``grid.contour(levels)`` with a
        semi-transparent fill; ``'volume'`` calls ``add_volume`` with the
        chosen opacity transfer function.  Volume mode always renders
        sigma0 (a secondary-field volume is not supported).
    clim : tuple of (float, float), optional
        Colour-scale limits for the *color scalar*; default 2/98
        percentile of that scalar on the grid.  For better mixed-layer
        contrast pass :func:`mixed_layer_clim`'s output (computed on the
        same scalar).
    cmap_volume, cmap_curtain : str, optional
        Colormaps for the background and curtain respectively.
    opacity : str, optional
        Opacity transfer function name for volume mode (default
        ``'sigmoid'``; see PyVista's opacity table).
    zscale : float, optional
        Z-axis exaggeration factor (only used for the axis label).
    show : bool, optional
        If True, opens an interactive window; otherwise renders
        off-screen (the default).
    top_marker : pyvista.MultiBlock, optional
        Optional surface-layer polyline (output of
        :func:`build_front_top_marker`) drawn in a single bright fixed
        colour so the front's lateral position is visible from above.
    top_marker_color : str, optional
        Colour of the top-layer marker (default ``'red'``).
    front_iso : pyvista.PolyData, optional
        Single sigma0 iso-surface from :func:`build_front_isosurface`.
    front_iso_color : str, optional
        Solid colour for the front iso-surface when it is NOT being
        coloured by the color scalar (default ``'orange'``).
    color_scalar : str, optional
        Name of the point-data array used for coloring (default
        ``'sigma0'``).  Must be stamped on the grid.
    color_title : str, optional
        Scalar-bar title.  Defaults to ``"sigma0 [kg/m^3]"`` for sigma0,
        else the scalar name.
    nan_color : str, optional
        Colour for NaN values of the color scalar on extracted surfaces
        (default neutral gray ``'#9e9e9e'``) -- keeps surfaces hole-free
        over land / clipped / undefined cells.
    nan_opacity : float, optional
        Opacity of NaN-coloured cells (default 1.0; set 0 to hide them).
    front_iso_use_color_scalar : bool, optional
        When True (default) and ``color_scalar != 'sigma0'``, the front
        iso-surface is coloured by the color scalar too (e.g. Ri along the
        tilted front sheet) instead of the solid ``front_iso_color``.
    font_size, title_font_size, label_font_size : int, optional
        Font sizes for axes / scalar-bar titles / scalar-bar tick labels.

    Returns
    -------
    pyvista.Plotter
        Configured plotter; the caller saves and closes it.
    """
    ensure_display()
    # Apply the requested font sizes by building a theme up front.  The
    # base scientific_theme already sets sensible defaults; we override
    # the two font knobs that drive the bounds axis text and any
    # add_text() calls.
    theme = scientific_theme(font_size=font_size, label_size=label_font_size)
    pl = new_plotter(off_screen=not show, theme=theme)

    # Resolve the coloring scalar + title.  Geometry stays sigma0-driven;
    # only the colors follow color_scalar.
    if color_scalar != "sigma0" and color_scalar not in grid.array_names:
        raise ValueError(
            f"color_scalar {color_scalar!r} is not stamped on the grid "
            f"(available: {list(grid.array_names)}).  Pass it via "
            "build_pyvista_grid(extra_fields=...)."
        )
    if color_title is None:
        color_title = ("sigma0 [kg/m^3]" if color_scalar == "sigma0"
                       else color_scalar)

    # Default contrast: 2/98 percentile of the coloring scalar.  Callers
    # that want mixed-layer-focused contrast should pass `clim=` from
    # :func:`mixed_layer_clim` (computed on the same scalar) instead.
    if clim is None:
        scalars = np.asarray(grid[color_scalar])
        clim = (
            float(np.nanpercentile(scalars, 2)),
            float(np.nanpercentile(scalars, 98)),
        )

    # Shared scalar-bar styling.  Vertical bar positioned on the right
    # side: wider and taller than the v1.5 default so the bar AND its
    # label text are easy to read in a printed/projected figure, with
    # enough margin from the viewport edge that the title text
    # (rendered to the right of the bar) doesn't clip at large font
    # sizes.  `position_x` is the bar's left edge in normalised
    # viewport coords (0 = left, 1 = right); `position_y` is the bottom
    # edge.
    def _sb_args(title, position_y):
        return dict(
            title=title,
            n_labels=5,
            fmt="%.3g",
            title_font_size=title_font_size,
            label_font_size=label_font_size,
            shadow=False,
            vertical=True,
            position_x=0.74,
            position_y=position_y,
            width=0.08,
            # height capped so position_y + height <= ~0.88 leaves
            # ~12% of the viewport above the bar for the title text
            # at title_font_size=60 without clipping it off the top.
            height=0.50,
        )

    if mode == "isopycnals":
        # Contour on sigma0 (geometry); VTK interpolates every other point
        # array onto the surfaces, so coloring by color_scalar is free.
        iso = grid.contour(isosurfaces=list(levels), scalars="sigma0")
        if iso.n_points > 0:
            pl.add_mesh(
                iso,
                scalars=color_scalar,
                cmap=cmap_volume,
                clim=clim,
                opacity=0.6,
                smooth_shading=True,
                nan_color=nan_color,
                nan_opacity=nan_opacity,
                scalar_bar_args=_sb_args(
                    color_title, position_y=0.30,
                ),
            )
    elif mode == "volume":
        # Volume mode renders the sigma0 field itself; coloring a sigma0
        # volume by a second scalar is ill-defined, so color_scalar is
        # ignored here (documented in the docstring).
        pl.add_volume(
            grid, scalars="sigma0",
            cmap=cmap_volume, clim=clim if color_scalar == "sigma0" else None,
            opacity=opacity,
            scalar_bar_args=_sb_args(
                "sigma0 [kg/m^3]", position_y=0.55,
            ),
        )
    else:
        raise ValueError(
            f"Unknown render mode {mode!r}; expected 'isopycnals' or 'volume'."
        )

    # Front curtain -- a vertical sheet at the front's surface pixels.
    # Off by default because it competes visually with the iso-surfaces;
    # opt in via draw_curtain=True for cases where the vertical-sheet
    # reference is useful.
    if draw_curtain and curtain is not None:
        first = True
        for name in curtain.keys():
            ribbon = curtain[name]
            if ribbon is None or ribbon.n_points == 0:
                continue
            if first:
                pl.add_mesh(
                    ribbon,
                    scalars="sigma0",
                    cmap=cmap_curtain,
                    clim=clim,
                    smooth_shading=True,
                    scalar_bar_args=_sb_args(
                        "front sigma0 [kg/m^3]", position_y=0.10,
                    ),
                )
                first = False
            else:
                pl.add_mesh(
                    ribbon,
                    scalars="sigma0",
                    cmap=cmap_curtain,
                    clim=clim,
                    smooth_shading=True,
                    show_scalar_bar=False,
                )

    # Front iso-surface -- a single sigma0 iso-surface that naturally
    # tilts with depth across the front.  In single-field mode it is drawn
    # opaque in a distinct colour so the tilt geometry (dense waters on one
    # side, less dense on the other) reads at a glance.  In dual-field mode
    # (color_scalar != 'sigma0') it is coloured by the color scalar instead
    # -- e.g. Ri along the tilted front sheet shows directly where the
    # front is shear-unstable.
    if front_iso is not None and front_iso.n_points > 0:
        color_front_by_scalar = (
            front_iso_use_color_scalar
            and color_scalar != "sigma0"
            and color_scalar in front_iso.array_names
        )
        if color_front_by_scalar:
            pl.add_mesh(
                front_iso,
                scalars=color_scalar,
                cmap=cmap_volume,
                clim=clim,
                opacity=front_iso_opacity,
                smooth_shading=True,
                nan_color=nan_color,
                nan_opacity=nan_opacity,
                show_scalar_bar=False,
            )
        else:
            pl.add_mesh(
                front_iso,
                color=front_iso_color,
                opacity=front_iso_opacity,
                smooth_shading=True,
                show_scalar_bar=False,
            )

    # Top-layer marker (optional).  Drawn in a single bright fixed colour
    # so the front's lateral position is identifiable in plan view, but
    # toned down (lower opacity) so it doesn't overwhelm the iso-surface
    # tilt structure underneath.
    if top_marker is not None and len(top_marker.keys()) > 0:
        for name in top_marker.keys():
            mesh = top_marker[name]
            if mesh is None or mesh.n_points == 0:
                continue
            pl.add_mesh(
                mesh,
                color=top_marker_color,
                opacity=top_marker_opacity,
                show_scalar_bar=False,
                smooth_shading=True,
            )

    # Axis bounds.  Bumping the bounds font_size makes the depth + i/j
    # tick labels legible alongside the larger scalar bars on the right.
    pl.show_bounds(
        xtitle="i (rect)",
        ytitle="j (rect)",
        ztitle=f"depth x {zscale:.0f} [m]",
        location="outer",
        ticks="both",
        font_size=font_size,
    )

    # Default camera: south-east elevated isometric.
    # VTK's view_isometric() places the camera at (1,1,1)*scale, which
    # puts +x (i) screen-LEFTWARD and +y (j) screen-RIGHTWARD.  We want
    # the more conventional "map" orientation: i increases left -> right
    # and j increases bottom -> top.  Placing the camera at
    # (+dx, -dy, +dz) (south-east elevated) and looking at the bbox
    # centre with up=z gives a view where the "screen right" cross
    # product R = V x U has +x and +y components, i.e. both axes
    # increase rightward AND upward, matching the user's expected
    # convention.
    bounds = grid.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    dx = 0.5 * (bounds[1] - bounds[0])
    dy = 0.5 * (bounds[3] - bounds[2])
    dz = 0.5 * (bounds[5] - bounds[4])
    span = float(max(dx, dy, dz))
    distance = 3.0 * span
    pl.camera_position = [
        (cx + distance, cy - distance, cz + distance),  # SE-up corner
        (cx, cy, cz),                                    # focal point
        (0.0, 0.0, 1.0),                                 # up = +z
    ]
    return pl
