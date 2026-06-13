"""
2-D matplotlib companion figures for 3-D PyVista renderings.

The existing ``viz_utils`` module is pyqtgraph-based; we keep matplotlib
helpers in their own module so neither backend pulls the other into
scripts that don't need it.

Currently only :func:`plot_bbox_inset` lives here.  It writes a 2-D
context map that accompanies the 3-D PNG produced by
``fronts/scripts/fronts_viz_3d.py``: surface sigma0 inside the cropped
bbox, the selected front overlaid in a contrasting colour, and
lon/lat secondary axes via the existing ``attach_lonlat_twins`` helper.
"""

# stdlib
from __future__ import annotations
from pathlib import Path

# numerical / plotting
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; pair to the PyVista off-screen render
import matplotlib.pyplot as plt  # noqa: E402


def plot_bbox_inset(
    surface_sigma0: np.ndarray,
    front_mask: np.ndarray,
    XC_rect: np.ndarray,
    YC_rect: np.ndarray,
    j_slice: slice,
    i_slice: slice,
    output_path: str | Path,
    *,
    clim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    front_color: str = "red",
    title: str | None = None,
    attach_lonlat_twins=None,
    j_tile_lookup: np.ndarray | None = None,
    i_tile_lookup: np.ndarray | None = None,
    XC_face: np.ndarray | None = None,
    YC_face: np.ndarray | None = None,
) -> Path:
    """Write a 2-D pcolormesh inset showing surface sigma0 + front overlay.

    Renders the cropped bbox in the rect-grid tile-local frame so the x/y
    coordinates match the 3-D figure's primary axes, then overlays the
    selected front pixels and (optionally) attaches lon/lat twin axes.

    Parameters
    ----------
    surface_sigma0 : numpy.ndarray
        2-D potential-density slice on the rect-grid tile-local frame,
        shape ``(TILE_SIZE, TILE_SIZE)``, sampled at the near-surface
        reference depth used for the isopycnal-bracket pick.
    front_mask : numpy.ndarray
        2-D boolean mask, same shape as ``surface_sigma0``, with True at
        the selected front's pixels (and False elsewhere).
    XC_rect, YC_rect : numpy.ndarray
        Longitude / latitude on the rect-grid tile-local frame, same
        shape as ``surface_sigma0``.  Only used as defaults when
        ``attach_lonlat_twins`` is not supplied.
    j_slice, i_slice : slice
        Tile-local rect-frame slices used to crop the rendered region
        (matches the crop applied to the 3-D figure).
    output_path : str or pathlib.Path
        Where to write the PNG.
    clim : tuple of (float, float), optional
        Colour-scale limits for sigma0.  Defaults to ``(2nd, 98th)``
        percentile of the cropped data.
    cmap : str, optional
        Matplotlib colormap for sigma0 (default ``'viridis'``).
    front_color : str, optional
        Colour for the front overlay (default ``'red'``).
    title : str, optional
        Figure title.  If None, a sensible default is generated.
    attach_lonlat_twins : callable, optional
        The ``attach_lonlat_twins`` helper from
        ``dev/mld/density_utils.py``.  When supplied alongside the
        ``j_tile_lookup``, ``i_tile_lookup``, ``XC_face``, ``YC_face``
        arguments, lon (top) and lat (right) twin axes are added to the
        plot.  Passed in to avoid hard-coding the dev-tree import path.
    j_tile_lookup, i_tile_lookup : numpy.ndarray, optional
        Tile-local face-index lookups returned by
        ``build_tile_lookup``.  Only used when ``attach_lonlat_twins`` is
        provided.
    XC_face, YC_face : numpy.ndarray, optional
        Face-local lon/lat arrays.  Only used when
        ``attach_lonlat_twins`` is provided.

    Returns
    -------
    pathlib.Path
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Crop everything to the requested window so the inset shows exactly
    # the region the 3-D figure renders.
    surf_crop = surface_sigma0[j_slice, i_slice]
    mask_crop = front_mask[j_slice, i_slice]

    if clim is None:
        # Robust default that suppresses a handful of outlier pixels --
        # mirrors the 3-D figure's default contrast policy.
        clim = (
            float(np.nanpercentile(surf_crop, 2)),
            float(np.nanpercentile(surf_crop, 98)),
        )

    # Pixel-extent axes in the rect-grid tile-local frame so the inset's
    # x,y match the 3-D figure's labelled axes ("i (rect)", "j (rect)").
    i_lo, i_hi = i_slice.start, i_slice.stop
    j_lo, j_hi = j_slice.start, j_slice.stop
    extent = (i_lo, i_hi, j_lo, j_hi)

    # Roomy figsize so the bumped title/axis/colorbar fonts still leave
    # plenty of canvas for the surface field.
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        surf_crop,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        interpolation="nearest",
        aspect="auto",  # 'equal' + twin axes triggers RuntimeError in mpl
    )
    # Overlay the front as a binary mask in a contrasting colour.  Using
    # a masked-array trick + a single-colour colormap keeps the rasterised
    # look honest (one pixel = one labelled pixel).
    front_overlay = np.ma.masked_where(~mask_crop, mask_crop.astype(float))
    ax.imshow(
        front_overlay,
        origin="lower",
        extent=extent,
        cmap=matplotlib.colors.ListedColormap([front_color]),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )

    # Bump label/tick/title font sizes so the inset is readable next to
    # the 3-D figure.
    label_fs = 14
    title_fs = 16
    tick_fs = 12
    ax.set_xlabel("i (rect tile-local)", fontsize=label_fs)
    ax.set_ylabel("j (rect tile-local)", fontsize=label_fs)
    ax.set_title(title or "Surface sigma0 + selected front", fontsize=title_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r"$\sigma_0$ [kg m$^{-3}$]", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    # Attach lon/lat twin axes if the helper + lookups are provided.  The
    # twin-axis helper expects tile-local face-index lookups, so it only
    # makes sense to call it when the caller has them on hand.
    if (
        attach_lonlat_twins is not None
        and j_tile_lookup is not None
        and i_tile_lookup is not None
        and XC_face is not None
        and YC_face is not None
    ):
        attach_lonlat_twins(ax, j_tile_lookup, i_tile_lookup, XC_face, YC_face)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
