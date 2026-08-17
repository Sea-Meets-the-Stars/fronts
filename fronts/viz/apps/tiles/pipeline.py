"""The front pipeline behind page 2.

This is the ingest half of ``fronts_viz_curtain.py`` / ``fronts_viz_3d.py``,
lifted out of the CLI so a server can drive it: load two tiles, remap to
the rect frame, pick a front, crop, clip to the mixed layer, and derive
the axis geometry.  Everything downstream -- the curtains and the 3-D
scene -- is the repo's existing code, untouched.

Two things differ from the scripts:

* Nothing is read from ``sys.argv``, and nothing is written to disk here.
* The face remap is a hook.  Real tiles carry face-local axes that must be
  fancy-indexed onto the rect frame; synthetic tiles are already in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fronts.llc.analysis import mixed_layer_depth_field
from fronts.viz import curtains, field_styles
from fronts.viz.geometry import front_bbox_and_crop, truncate_depth


@dataclass
class FrontScene:
    """Everything the figure builders need for one front.

    Attributes
    ----------
    label : int
        The selected front's label.
    sigma0, color : numpy.ndarray
        ``(K', J', I')`` cropped and depth-clipped density and display
        field.  ``color`` has already been through the field style's
        transform.
    Z : numpy.ndarray
        ``(K',)`` depth axis for the clipped volume.
    axis_path : numpy.ndarray
        ``(L, 2)`` main-axis ``(j, i)`` in the cropped frame.
    metrics : dict
        Output of :func:`fronts.viz.curtains.path_metrics`.
    front_mask : numpy.ndarray
        ``(J', I')`` boolean mask of this front in the cropped frame.
    mld_curtain : numpy.ndarray
        Mixed-layer depth sampled along the main axis.
    clim : tuple of float
        Display colour limits.
    j_slice, i_slice : slice
        The crop, in tile-local coordinates.
    XC, YC : numpy.ndarray
        Cropped coordinates, for the km axis and the inset.
    style : FieldStyle
        The colour field's registered display style.
    field_name : str
    n_levels : numpy.ndarray
        Isopycnal levels chosen for the contours.
    """

    label: int
    sigma0: np.ndarray
    color: np.ndarray
    Z: np.ndarray
    axis_path: np.ndarray
    metrics: dict
    front_mask: np.ndarray
    mld_curtain: np.ndarray
    clim: tuple[float, float]
    j_slice: slice
    i_slice: slice
    XC: np.ndarray
    YC: np.ndarray
    style: object
    field_name: str
    levels: np.ndarray


class NoSuchFront(ValueError):
    """The requested label is absent from this tile."""


def tile_labels(provider, date: str, tile_idx: int, shape) -> np.ndarray:
    """The global label mask, sliced to a tile window.

    In synthetic mode the tile slices come from the fabricated world; with
    real data the tile's ``rect_j_start`` / ``rect_i_start`` attrs give the
    window directly.
    """
    labels = provider.labels(date)
    nj, ni = shape
    if provider.synthetic:
        from fronts.viz.apps.common import synthetic
        js, iss = synthetic.get_world(date).tile_slices(tile_idx)
        return labels[js, iss]
    raise NotImplementedError(
        "Slice the global labels with the tile's rect_j_start / rect_i_start "
        "attrs once the real store is wired up."
    )


def remap_to_rect(arr, lookup=None):
    """Map a face-local array onto the rect frame.

    Synthetic tiles are already in the rect frame, so *lookup* is ``None``
    and this is the identity.  With real tiles, pass the ``(j_face,
    i_face)`` lookup from ``build_tile_lookup`` and this fancy-indexes the
    array, exactly as ``fronts_viz_3d.remap_to_rect`` does.
    """
    if lookup is None:
        return arr
    j_face, i_face = lookup
    if arr.ndim == 2:
        return arr[j_face, i_face]
    return arr[:, j_face, i_face]


def available_fronts(labels_tile: np.ndarray, *, min_pixels: int = 25) -> list[int]:
    """Labels present in this tile, largest first.

    Small pieces are dropped: a front needs a real main axis before a
    curtain along it means anything.
    """
    flat = labels_tile.ravel()
    counts = np.bincount(flat[flat > 0])
    labels = np.nonzero(counts >= min_pixels)[0]
    return sorted(labels.tolist(), key=lambda l: -counts[l])


def build_scene(
    provider,
    date: str,
    tile_idx: int,
    field: str,
    label: int,
    *,
    margin: int = 50,
    n_below: int = 3,
    n_isopycnals: int = 8,
    lookup=None,
) -> FrontScene:
    """Run the ingest pipeline for one front.

    Mirrors steps 1-9 of the ``fronts_viz_curtain`` algorithm flow
    documented in ``docs/viz/fronts_curtain.md``.
    """
    # 1 -- load the two tiles.
    ds_rho = provider.tile(date, tile_idx, "density")
    ds_fld = provider.tile(date, tile_idx, field)

    rho_var = ds_rho.attrs.get("tile_var_name") or _sole_3d(ds_rho)
    fld_var = ds_fld.attrs.get("tile_var_name") or _sole_3d(ds_fld)

    _check_provenance(ds_rho, ds_fld)

    # 2 -- remap onto the rect frame.
    sigma0 = remap_to_rect(np.asarray(ds_rho[rho_var].values), lookup)
    colour = remap_to_rect(np.asarray(ds_fld[fld_var].values), lookup)
    XC = remap_to_rect(np.asarray(ds_rho["XC"].values), lookup)
    YC = remap_to_rect(np.asarray(ds_rho["YC"].values), lookup)
    Z = np.asarray(ds_rho["Z"].values)

    # 3 -- labels for this window.
    labels = tile_labels(provider, date, tile_idx, sigma0.shape[1:])
    if label <= 0 or not np.any(labels == label):
        raise NoSuchFront(f"label {label} is not in tile {tile_idx}")

    # 4 -- crop to the front's bbox.
    j_slice, i_slice = front_bbox_and_crop(labels, label, margin=margin)
    sigma0_c = sigma0[:, j_slice, i_slice]
    colour_c = colour[:, j_slice, i_slice]
    front_mask = labels[j_slice, i_slice] == label

    # 5 -- mixed layer, then clip the depth range.
    mld_depth, k_mld = mixed_layer_depth_field(sigma0_c, Z)
    sigma0_k, Z_k, _ = truncate_depth(sigma0_c, Z, k_mld, n_below=n_below)
    colour_k = colour_c[: sigma0_k.shape[0]]

    # 6 -- display transform for the colour field.
    style = field_styles.get_style(fld_var)
    colour_disp = field_styles.apply_transform(colour_k, style)
    clim = field_styles.default_clim(colour_disp, style)

    # 7 -- main axis and its metrics.
    axis_path = curtains.extract_main_axis(front_mask)
    metrics = curtains.path_metrics(
        axis_path, XC[j_slice, i_slice], YC[j_slice, i_slice]
    )

    # 8 -- MLD along the axis, for the dashed overlay.
    mld_curtain = np.array(
        [mld_depth[int(j), int(i)] for j, i in axis_path], dtype=float
    )

    # 9 -- isopycnal levels bracketing the volume.
    finite = sigma0_k[np.isfinite(sigma0_k)]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [2, 98])
        levels = np.linspace(lo, hi, n_isopycnals)
    else:
        levels = np.array([])

    return FrontScene(
        label=int(label),
        sigma0=sigma0_k,
        color=colour_disp,
        Z=Z_k,
        axis_path=axis_path,
        metrics=metrics,
        front_mask=front_mask,
        mld_curtain=mld_curtain,
        clim=clim,
        j_slice=j_slice,
        i_slice=i_slice,
        XC=XC[j_slice, i_slice],
        YC=YC[j_slice, i_slice],
        style=style,
        field_name=fld_var,
        levels=levels,
    )


def _sole_3d(ds):
    cands = [v for v in ds.data_vars if ds[v].ndim == 3]
    if len(cands) != 1:
        raise KeyError(f"expected exactly one 3-D variable, found {cands}")
    return cands[0]


def _check_provenance(a, b):
    """Refuse to mix tiles from different windows or timestamps."""
    for key in ("tile_index", "rect_i_start", "rect_j_start", "timestamp"):
        va, vb = a.attrs.get(key), b.attrs.get(key)
        if va != vb:
            raise ValueError(
                f"Tile provenance mismatch on {key!r}: {va!r} vs {vb!r}.  "
                "Regenerate both tiles with the same --i/--j/--timestamp."
            )
