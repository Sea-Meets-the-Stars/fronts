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
from fronts.viz.apps import config
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


def _dev_mld():
    """The ``dev/mld`` helpers the 3-D and curtain scripts already use."""
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[4] / "dev" / "mld"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import density_utils
    return density_utils


def _attr(ds, name):
    """One provenance attr as a plain int, or ``None`` if absent."""
    if name not in ds.attrs:
        return None
    return int(np.asarray(ds.attrs[name]).item())


def rect_origin(ds) -> tuple[int, int]:
    """``(i0, j0)`` of the tile on the global rect grid.

    Which attrs carry this depends on the version of ``tile_utils`` that
    made the tile: some write ``rect_i_start`` / ``rect_j_start``
    directly, others only the tile row and column.  Both say the same
    thing, so either is accepted rather than requiring one branch.
    """
    i0, j0 = _attr(ds, "rect_i_start"), _attr(ds, "rect_j_start")
    if i0 is not None and j0 is not None:
        return i0, j0

    ti, tj = _attr(ds, "tile_i_rect"), _attr(ds, "tile_j_rect")
    if ti is not None and tj is not None:
        return ti * config.TILE_SIZE, tj * config.TILE_SIZE

    raise KeyError(
        "tile carries no rect origin: expected rect_i_start/rect_j_start "
        f"or tile_i_rect/tile_j_rect, got {sorted(ds.attrs)}"
    )


def tile_window(ds) -> tuple[slice, slice]:
    """The tile's window on the global rect grid, as ``(j_slice, i_slice)``."""
    i0, j0 = rect_origin(ds)
    n = int(ds.sizes.get("j", config.TILE_SIZE))
    return slice(j0, j0 + n), slice(i0, i0 + n)


def tile_lookup(ds, *, synthetic: bool = False):
    """Face-local index maps for a real tile, or ``None`` for a synthetic one.

    ``fronts_viz_curtain`` and ``fronts_viz_3d`` both remap the *tile* into
    the rect frame and then slice the global labels by the rect window.
    Doing the same here keeps the page and the scripts on one convention.

    The synthetic world carries the same provenance attrs on purpose, so
    the caller says which kind of tile this is rather than guessing.  A
    real tile whose origin cannot be read raises: returning ``None`` would
    skip the remap and misalign the labels on every rotated face, which is
    wrong in a way nothing downstream could notice.
    """
    if synthetic:
        return None

    i0, j0 = rect_origin(ds)
    face = _attr(ds, "face_index")
    if face is None:
        raise KeyError("tile carries no face_index; cannot build the remap")
    return _dev_mld().build_tile_lookup(i0, j0, face)


def tile_labels(provider, date: str, tile_idx: int, shape, ds=None,
                region: str | None = None) -> np.ndarray:
    """The global label mask, sliced to a tile window.

    In synthetic mode the tile slices come from the fabricated world; with
    real data the window comes from the tile's own ``rect_j_start`` /
    ``rect_i_start`` attrs, so it needs the Dataset.
    """
    labels = provider.labels(date)
    if provider.synthetic:
        from fronts.viz.apps.common import synthetic
        js, iss = synthetic.get_world(date).tile_slices(tile_idx)
        return labels[js, iss]

    if ds is None:
        ds = provider.tile(date, tile_idx, "density", region)

    js, iss = tile_window(ds)
    return np.asarray(labels[js, iss])


#: The order every consumer here assumes for a 3-D tile variable.
VERTICAL_DIMS = ("k", "k_l", "k_u", "k_p1", "Z", "Zl")


def field_values(ds, name):
    """A tile variable as numpy, with the vertical axis first.

    Dim *order* is not guaranteed by anything upstream: a compute that
    multiplies a 2-D field by a 3-D one comes back as ``(j, i, k)``,
    because xarray puts the first operand's dims first.  That array then
    reaches ``remap_to_rect``, which indexes positionally, and fails with
    an opaque ``index 51 is out of bounds for axis 2 with size 51``.

    Reading the order off the DataArray instead of assuming it makes the
    page immune to that, including for tiles already written to the store
    in the wrong order.
    """
    da = ds[name] if hasattr(ds, "__getitem__") else ds
    dims = list(da.dims)

    vertical = [d for d in dims if d in VERTICAL_DIMS]
    if not vertical:
        return np.asarray(da.values)

    horizontal = [d for d in dims if d not in vertical]
    return np.asarray(da.transpose(*vertical, *horizontal).values)


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

    # Check the axes being indexed before indexing them, so a wrongly
    # ordered array says so rather than raising an out-of-bounds index
    # that names the vertical size and explains nothing.
    plane = arr.shape[-2:]
    if plane[0] <= int(j_face.max()) or plane[1] <= int(i_face.max()):
        raise ValueError(
            f"remap_to_rect got an array shaped {arr.shape}: its last two "
            f"axes must be (j, i) and large enough for the lookup "
            f"({int(j_face.max()) + 1}, {int(i_face.max()) + 1}).  A 3-D "
            "tile variable must be (k, j, i) -- use field_values() to read "
            "it rather than .values.")

    if arr.ndim == 2:
        return arr[j_face, i_face]
    return arr[:, j_face, i_face]


def available_fronts(labels_tile: np.ndarray, *, min_pixels: int = 25) -> list[int]:
    """Labels present in this tile, in **numerical order**.

    Numerical, not largest-first: these are five-digit label numbers and
    the list is something a person has to find a specific value in.  Size
    order made that a linear scan.  Where size matters -- deciding which
    few fronts to annotate on the map -- the caller ranks them itself.

    Small pieces are dropped: a front needs a real main axis before a
    curtain along it means anything.
    """
    flat = labels_tile.ravel()
    counts = np.bincount(flat[flat > 0])
    labels = np.nonzero(counts >= min_pixels)[0]
    return sorted(int(v) for v in labels)


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
    region: str | None = None,
) -> FrontScene:
    """Run the ingest pipeline for one front.

    Mirrors steps 1-9 of the ``fronts_viz_curtain`` algorithm flow
    documented in ``docs/viz/fronts_curtain.md``.
    """
    # 1 -- load the two tiles.
    ds_rho = provider.tile(date, tile_idx, "density", region)
    ds_fld = provider.tile(date, tile_idx, field, region)

    rho_var = ds_rho.attrs.get("tile_var_name") or _sole_3d(ds_rho)
    fld_var = ds_fld.attrs.get("tile_var_name") or _sole_3d(ds_fld)

    _check_provenance(ds_rho, ds_fld)

    if lookup is None:
        lookup = tile_lookup(ds_rho, synthetic=provider.synthetic)

    # 2 -- remap onto the rect frame.
    sigma0 = remap_to_rect(field_values(ds_rho, rho_var), lookup)
    colour = remap_to_rect(field_values(ds_fld, fld_var), lookup)
    XC = remap_to_rect(field_values(ds_rho, "XC"), lookup)
    YC = remap_to_rect(field_values(ds_rho, "YC"), lookup)
    Z = np.asarray(ds_rho["Z"].values)

    # 3 -- labels for this window.
    labels = tile_labels(provider, date, tile_idx, sigma0.shape[1:],
                         ds=ds_rho, region=region)
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
