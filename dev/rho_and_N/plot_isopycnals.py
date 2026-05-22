"""
Plot the depth of a chosen isopycnal across one LLC4320 tile.

Given:
  * a 3D density tile NetCDF produced by
    ``llc4320-native-grid-preprocessing/dev/pot_density/generate_tile_density.py``
    (sigma0(k, j, i) on face-local axes, plus XC, YC, Z and provenance attrs),
  * the global labeled-fronts integer mask (.npy or .nc),
  * the front-index parquet (label, name, x0, y0, x1, y1),
  * a target potential density (kg m^-3),

this script

  1. computes the 2D field ``z_iso(j, i)`` -- the depth at which sigma0 first
     reaches the target value, linearly interpolated between bracketing model
     levels (NaN where the target is out of range),
  2. optionally writes ``z_iso`` to a NetCDF named
     ``isopycnal_depth_tile{tile_index:03d}_{YYYYMMDDTHH}.nc`` (use
     ``--write-nc``), or reads back such a NetCDF if it is supplied via
     ``--isopycnal-nc`` (the calculation step is then skipped),
  3. renders the depth field as a map with the labeled fronts whose bbox
     overlaps the tile outlined in white (one continuous contour per front).

CLI usage
---------
    python dev/rho_and_N/plot_isopycnals.py \\
        --density-tile  density_tile207_20121109T12.nc \\
        --labels        labeled_fronts_global_20121109T12.npy \\
        --front-index   front_index_20121109T12.parquet \\
        --sigma0        1026.0 \\
        --write-nc
"""

# stdlib
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# numerical / IO
import numpy as np
import xarray as xr

# plotting -- headless-safe backend so this runs on cluster nodes too.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# repo helpers
from fronts.properties.io import load_front_index  # noqa: E402

# Shared helpers live next to this script; make sure that directory is
# importable regardless of how the script is invoked.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from density_utils import (  # noqa: E402
    TILE_SIZE,
    timestamp_to_stamp,
    load_density_tile,
    tile_scalar,
    build_tile_lookup,
    filter_overlapping_fronts,
    load_labels_tile,
    attach_lonlat_twins,
)


# ---------------------------------------------------------------------------
# Isopycnal-depth field computation
# ---------------------------------------------------------------------------

def compute_isopycnal_depth(
    sigma0: np.ndarray, Z: np.ndarray, target: float,
) -> np.ndarray:
    """2D field of depths at which ``sigma0 == target`` (shallowest crossing).

    For each ``(j, i)`` column the algorithm:
      * finds the shallowest depth index ``k`` where ``sigma0[k] >= target``;
      * linearly interpolates between ``Z[k-1]`` and ``Z[k]`` using
        ``(target - sigma0[k-1]) / (sigma0[k] - sigma0[k-1])`` as the weight;
      * assigns ``NaN`` to columns that never cross the target (water too
        light or too dense throughout) and to columns whose first cross is
        already at the surface ``k == 0``.

    The shallowest-crossing rule (Clarification 4) handles density inversions
    cleanly: if a thin pycnocline straddles ``target`` more than once we keep
    the shallower one.  ``np.argmax`` is used to locate ``k``: applied to the
    boolean ``sigma0 >= target`` along the depth axis it returns the index of
    the first ``True`` in every column (or 0 if no ``True`` exists, which we
    filter out with a separate "any-True" mask).

    Parameters
    ----------
    sigma0 : numpy.ndarray
        Potential density, shape ``(K, J, I)`` in kg m^-3 (typically
        ``K=51``, ``J=I=720`` for an LLC4320 tile).
    Z : numpy.ndarray
        1-D depth array, length ``K``, in metres.  Convention: negative
        downward (matches LLC4320), with the surface as ``Z[0]`` (largest).
    target : float
        Target potential density in kg m^-3.

    Returns
    -------
    numpy.ndarray
        ``(J, I)`` array of isopycnal depths in metres (negative downward).
        ``NaN`` marks columns where ``target`` is never crossed or where the
        crossing already lies at the surface.
    """
    # Boolean cube: True where this depth has reached the target density.
    # sigma0 grows with depth in a stably stratified column, so the first
    # True along axis 0 is the shallowest crossing.
    reached = sigma0 >= target  # shape (K, J, I)

    # any_reached[j, i] is True iff this column ever crosses target.  We use
    # this to mask the NaN result; np.argmax always returns 0 when no value
    # is True, which we cannot distinguish from "true at the surface" without
    # this guard.
    any_reached = np.any(reached, axis=0)  # (J, I)

    # First crossing index per column.  np.argmax on a boolean array returns
    # the index of the first True (or 0 if there are none -- handled above).
    k_first = np.argmax(reached, axis=0)  # (J, I), int64

    # Columns where the crossing already lives at the surface (k == 0) have
    # no bracketing level above them, so we can't interpolate.  Treat them
    # as out-of-range and NaN them out alongside the never-cross columns.
    valid = any_reached & (k_first > 0)

    # Build the interpolated depth field.  Use np.take_along_axis to pull
    # sigma0/Z values at k_first and k_first-1 for every (j, i) cell at once.
    # Wherever valid is False we'll overwrite with NaN at the end, so the
    # k_first-1 indexing below using clip(0) won't produce garbage that
    # leaks into the output.
    k_above = np.clip(k_first - 1, 0, None)  # bracket above (shallower)

    # take_along_axis needs an indexer with the same ndim as the source.
    # Insert a length-1 axis at position 0 so each (j, i) takes one depth.
    sig_below = np.take_along_axis(
        sigma0, k_first[None, :, :], axis=0,
    )[0]  # sigma0(z = Z[k_first])
    sig_above = np.take_along_axis(
        sigma0, k_above[None, :, :], axis=0,
    )[0]  # sigma0(z = Z[k_first - 1])
    z_below = Z[k_first]      # depth of the heavier-side bracket
    z_above = Z[k_above]      # depth of the lighter-side bracket

    # Linear interpolation: fraction along the (sigma_above -> sigma_below)
    # segment to reach the target density.  Guard against the (very rare)
    # case where the two sigma0 values are identical to avoid 0/0.
    denom = sig_below - sig_above
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom != 0.0, (target - sig_above) / denom, 0.0)
    z_iso = z_above + frac * (z_below - z_above)

    # NaN out everything that isn't a real, interpolatable crossing.
    z_iso = np.where(valid, z_iso, np.nan)
    return z_iso.astype(np.float32)


# ---------------------------------------------------------------------------
# NetCDF read / write
# ---------------------------------------------------------------------------

def write_isopycnal_nc(
    z_iso: np.ndarray,
    XC: np.ndarray,
    YC: np.ndarray,
    target: float,
    tile_index: int,
    face_index: int,
    rect_i_start: int,
    rect_j_start: int,
    timestamp: str,
    out_path: Path,
) -> None:
    """Persist the 2D isopycnal-depth field to a NetCDF tagged with provenance.

    The file structure mirrors the input density tile -- same ``(j, i)``
    axes, same ``XC``/``YC`` coords, same provenance attrs -- plus the new
    ``target_sigma0`` attr so consumers can verify the file matches the
    density they want.

    Parameters
    ----------
    z_iso : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` isopycnal-depth field in metres
        (negative downward; NaN out-of-range).
    XC, YC : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` longitude/latitude arrays from the source
        density tile.
    target : float
        Target potential density (kg m^-3), stored as ``target_sigma0``.
    tile_index, face_index, rect_i_start, rect_j_start : int
        Provenance scalars copied from the density tile.
    timestamp : str
        Timestamp string (``'YYYY-MM-DD HH:MM:SS'``) copied from the density
        tile.
    out_path : pathlib.Path
        Destination path for the NetCDF (parents created if needed).

    Returns
    -------
    None
        Side-effect only: the NetCDF is written to ``out_path``.
    """
    # Build an xarray.Dataset with the same dims as the source tile so the
    # written file plugs straight back into any downstream xarray consumer.
    ds = xr.Dataset(
        data_vars={
            "z_isopycnal": (
                ("j", "i"), z_iso,
                {
                    "long_name": "isopycnal depth",
                    "units": "m",
                    "description": (
                        "Depth (negative downward) of the shallowest "
                        f"sigma0 = {target} kg/m^3 crossing; NaN where the "
                        "isopycnal is shallower than the surface density "
                        "or deeper than the deepest sample."
                    ),
                },
            ),
        },
        coords={
            "XC": (("j", "i"), XC),
            "YC": (("j", "i"), YC),
        },
        attrs={
            "target_sigma0": float(target),
            "tile_index":    int(tile_index),
            "face_index":    int(face_index),
            "rect_i_start":  int(rect_i_start),
            "rect_j_start":  int(rect_j_start),
            "timestamp":     str(timestamp),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)


def read_isopycnal_nc(
    path: Path,
    tile_index: int,
    rect_i_start: int,
    rect_j_start: int,
    target: float,
) -> np.ndarray:
    """Load a precomputed isopycnal-depth NetCDF and cross-check provenance.

    Parameters
    ----------
    path : pathlib.Path
        Path to the NetCDF written by :func:`write_isopycnal_nc` (or
        equivalent).
    tile_index : int
        Expected ``tile_index`` (from the density tile); mismatch is an error.
    rect_i_start, rect_j_start : int
        Expected rect-grid origin; mismatch is an error.
    target : float
        Expected target potential density (kg m^-3); a mismatch is logged as
        a warning (not an error) since the user may deliberately re-use a
        file with a slightly different density.

    Returns
    -------
    numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` isopycnal-depth field in metres.

    Raises
    ------
    KeyError
        If the file is missing the ``z_isopycnal`` variable.
    RuntimeError
        If the provenance scalars don't match the density tile.
    """
    ds = xr.open_dataset(path)
    if "z_isopycnal" not in ds.data_vars:
        raise KeyError(f"{path} has no 'z_isopycnal' variable.")
    # Validate tile_index + rect origin against the density tile so we don't
    # silently plot a depth field for a different chunk of the globe.
    cached_idx = int(ds.attrs.get("tile_index", -1))
    cached_i0  = int(ds.attrs.get("rect_i_start", -1))
    cached_j0  = int(ds.attrs.get("rect_j_start", -1))
    if (cached_idx, cached_i0, cached_j0) != (tile_index, rect_i_start, rect_j_start):
        raise RuntimeError(
            f"Cached isopycnal NetCDF {path} provenance "
            f"(tile_index={cached_idx}, rect_i={cached_i0}, rect_j={cached_j0}) "
            "does not match the density tile "
            f"(tile_index={tile_index}, rect_i={rect_i_start}, rect_j={rect_j_start})."
        )
    cached_target = float(ds.attrs.get("target_sigma0", float("nan")))
    if not np.isnan(cached_target) and abs(cached_target - target) > 1e-6:
        logging.warning(
            f"Cached NetCDF target_sigma0={cached_target} differs from "
            f"--sigma0={target}; using cached values anyway."
        )
    return ds["z_isopycnal"].values


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _remap_field_to_rect(
    field_face: np.ndarray,
    j_tile_lookup: np.ndarray,
    i_tile_lookup: np.ndarray,
) -> np.ndarray:
    """Gather a face-local 2D field onto the rect-grid tile-local frame.

    The density tile (and thus ``z_iso``, ``XC``, ``YC``) lives on face-local
    ``(j_tile, i_tile)`` axes, but the labels mask and the lon/lat helper
    (:func:`attach_lonlat_twins`) operate in the rect-grid tile-local frame.
    Plotting ``z_iso`` directly therefore renders the tile rotated for the
    LLC4320 faces whose face-local axes are rotated relative to the rect grid
    (faces 7..12).  This helper pulls every face-local value into the matching
    rect-grid tile-local cell so the imshow, labels overlay and twin axes all
    share one orientation.

    Parameters
    ----------
    field_face : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` field on the face-local frame (e.g.
        ``z_iso`` straight out of :func:`compute_isopycnal_depth`).
    j_tile_lookup, i_tile_lookup : numpy.ndarray
        Tile-local face-index lookups from :func:`build_tile_lookup`.

    Returns
    -------
    numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` field on the rect-grid tile-local frame
        (same dtype as the input).
    """
    # Fancy-indexing gather: rect[r, c] = face[j_lookup[r, c], i_lookup[r, c]].
    return field_face[j_tile_lookup, i_tile_lookup]


def plot_isopycnal_depth(
    z_iso_rect: np.ndarray,
    labels_rect: np.ndarray,
    front_labels: np.ndarray,
    XC: np.ndarray,
    YC: np.ndarray,
    j_tile_lookup: np.ndarray,
    i_tile_lookup: np.ndarray,
    target: float,
    tile_index: int,
    timestamp: str,
    out_path: Path,
) -> None:
    """Render the isopycnal-depth map with one white outline per front.

    Both inputs live in the **rect-grid tile-local** frame so the imshow, the
    labels contour overlay and the lon/lat twin axes share one orientation.
    Plotting the face-local field directly would rotate the map for LLC4320
    faces whose face-local axes are rotated relative to the rect grid (see
    Modification 1 in ``prompts/fronts_isopycnals.md``).

    Parameters
    ----------
    z_iso_rect : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` isopycnal-depth field in metres
        (negative downward; NaN where the isopycnal is out of range), already
        remapped to the rect-grid tile-local frame via
        :func:`_remap_field_to_rect`.
    labels_rect : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` integer label mask in the rect-grid
        tile-local frame (the natural frame of the global labels file).
    front_labels : numpy.ndarray
        1-D array of front labels whose bbox overlaps the tile (one outline
        is drawn per element).
    XC, YC : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` longitude/latitude arrays from the
        density tile (face-local frame -- consumed only via the lookups).
    j_tile_lookup, i_tile_lookup : numpy.ndarray
        Tile-local face-index lookups; used to draw the secondary lon/lat
        twin axes (helper expects rect-grid axes).
    target : float
        Target potential density (kg m^-3), used in the title.
    tile_index : int
        Tile index used in the title.
    timestamp : str
        Timestamp used in the title.
    out_path : pathlib.Path
        Path to save the PNG.

    Returns
    -------
    None
        The figure is written to ``out_path`` and closed.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # cmap='viridis_r' so deeper isopycnals render darker -- depth reads
    # intuitively as "more pigment = more water above".  set_bad gives NaN
    # cells a distinct neutral grey so out-of-range pixels are obvious.
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="lightgrey")

    # Mask NaNs so the bad-colour applies.  vmin/vmax left to auto so the
    # colorbar tracks the in-range data.
    z_masked = np.ma.masked_invalid(z_iso_rect)

    im = ax.imshow(
        z_masked,
        origin="lower",
        cmap=cmap,
        extent=(0, TILE_SIZE, 0, TILE_SIZE),
        aspect="auto",  # required so twiny/twinx (shared axes) can coexist
        interpolation="nearest",
    )

    # Overlay one white contour per front.  contour(labels == label, [0.5])
    # produces a single closed outline around each front's pixel set without
    # merging into neighbouring fronts.  linewidths kept thin so dense
    # tiles don't end up dominated by overlay ink.
    for label in front_labels:
        if label == 0:
            continue  # background label -- skip if it sneaks into the list
        mask = (labels_rect == int(label)).astype(np.uint8)
        if not mask.any():
            # bbox overlapped but no actual pixels inside the tile -- rare.
            continue
        ax.contour(
            mask,
            levels=[0.5],
            colors="white",
            linewidths=0.7,
            extent=(0, TILE_SIZE, 0, TILE_SIZE),
            origin="lower",
        )

    ax.set_xlabel("i (rect-grid tile-local)")
    ax.set_ylabel("j (rect-grid tile-local)")
    ax.set_title(
        f"Tile {tile_index}  {timestamp}\n"
        f"isopycnal depth at sigma0 = {target:.3f} kg m$^{{-3}}$"
    )

    # Secondary lon/lat axes (shared helper) -- anchor the colorbar to the
    # rightmost twin axis so it doesn't fight the lat tick labels.
    _, ax_lat = attach_lonlat_twins(
        ax, j_tile_lookup, i_tile_lookup, XC, YC,
    )

    fig.colorbar(
        im, ax=ax_lat,
        label="isopycnal depth [m]",
        pad=0.10, fraction=0.05,
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run(
    density_tile: Path,
    labels_path: Path,
    front_index_path: Path,
    target_sigma0: float,
    outdir: Path,
    isopycnal_nc: Path | None,
    write_nc: bool,
) -> None:
    """End-to-end: load tile -> compute or read z_iso -> write NetCDF -> plot.

    Parameters
    ----------
    density_tile : pathlib.Path
        Path to the 3D density tile NetCDF (``sigma0(k, j, i)``).
    labels_path : pathlib.Path
        Path to the global labeled-fronts integer mask (``.npy`` or ``.nc``).
    front_index_path : pathlib.Path
        Path to the front-index parquet (``label, name, x0, y0, x1, y1``).
    target_sigma0 : float
        Target potential density (kg m^-3).
    outdir : pathlib.Path
        Output directory for the PNG and (optionally) the NetCDF.
    isopycnal_nc : pathlib.Path or None
        Path to a precomputed isopycnal-depth NetCDF.  If supplied and the
        file exists, the calculation step is skipped and ``z_iso`` is read
        from disk; provenance is cross-checked against the density tile.
    write_nc : bool
        When True (and ``isopycnal_nc`` was not used), write ``z_iso`` to
        ``{outdir}/isopycnal_depth_tile{idx:03d}_{stamp}.nc``.

    Returns
    -------
    None
        Side-effects only -- writes the PNG (and possibly the NetCDF) into
        ``outdir``.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load density tile provenance + arrays. -------------------
    ds = load_density_tile(density_tile)
    tile_index   = int(tile_scalar(ds, "tile_index"))
    face_index   = int(tile_scalar(ds, "face_index"))
    rect_i_start = int(tile_scalar(ds, "rect_i_start"))
    rect_j_start = int(tile_scalar(ds, "rect_j_start"))
    timestamp    = str(tile_scalar(ds, "timestamp"))
    stamp        = timestamp_to_stamp(timestamp)
    logging.info(
        f"Density tile: tile_index={tile_index}, face_index={face_index}, "
        f"rect origin=({rect_i_start}, {rect_j_start}), timestamp={timestamp}"
    )

    XC = ds["XC"].values
    YC = ds["YC"].values

    # Canonical NetCDF name -- shared between the read-back and write paths
    # so a previously-written file is auto-discoverable when re-running.
    nc_name = f"isopycnal_depth_tile{tile_index:03d}_{stamp}.nc"
    nc_path = outdir / nc_name

    # ---- Step 2: compute or read the isopycnal depth field. ---------------
    # Precedence: explicit --isopycnal-nc > existing default file > compute.
    cache_path: Path | None = None
    if isopycnal_nc is not None:
        if not isopycnal_nc.exists():
            raise FileNotFoundError(
                f"--isopycnal-nc {isopycnal_nc} does not exist."
            )
        cache_path = isopycnal_nc

    if cache_path is not None:
        logging.info(f"Reading precomputed isopycnal NetCDF: {cache_path}")
        z_iso = read_isopycnal_nc(
            cache_path,
            tile_index=tile_index,
            rect_i_start=rect_i_start,
            rect_j_start=rect_j_start,
            target=target_sigma0,
        )
    else:
        # Load sigma0 + Z eagerly -- sigma0 is ~50 MB at float32 and we'll
        # touch every column once.  No win to be had from dask here.
        logging.info("Loading sigma0 + Z from density tile")
        sigma0 = ds["sigma0"].values
        Z      = ds["Z"].values
        logging.info(
            f"Computing z_iso for sigma0 = {target_sigma0} kg/m^3 "
            f"over a {sigma0.shape} grid"
        )
        z_iso = compute_isopycnal_depth(sigma0, Z, target_sigma0)
        n_valid = int(np.sum(np.isfinite(z_iso)))
        logging.info(
            f"z_iso computed: {n_valid}/{z_iso.size} columns crossed the "
            f"target ({100.0 * n_valid / z_iso.size:.1f}%)"
        )
        if write_nc:
            logging.info(f"Writing isopycnal-depth NetCDF: {nc_path}")
            write_isopycnal_nc(
                z_iso=z_iso, XC=XC, YC=YC, target=target_sigma0,
                tile_index=tile_index, face_index=face_index,
                rect_i_start=rect_i_start, rect_j_start=rect_j_start,
                timestamp=timestamp,
                out_path=nc_path,
            )

    # ---- Step 3: locate the fronts that overlap the tile. -----------------
    logging.info(f"Loading front index: {front_index_path}")
    index_df = load_front_index(front_index_path)
    overlapping = filter_overlapping_fronts(
        index_df, rect_i_start=rect_i_start, rect_j_start=rect_j_start,
    )
    logging.info(f"{len(overlapping)} fronts have bbox overlapping the tile")

    # ---- Step 4: load the labels tile window (already rect-grid). ---------
    rect_j_slice = slice(rect_j_start, rect_j_start + TILE_SIZE)
    rect_i_slice = slice(rect_i_start, rect_i_start + TILE_SIZE)
    labels_rect = load_labels_tile(labels_path, rect_j_slice, rect_i_slice)

    j_tile_lookup, i_tile_lookup = build_tile_lookup(
        rect_i_start, rect_j_start, face_index,
    )

    # Modification 1: the imshow, labels overlay and lon/lat twin axes must
    # share one frame, otherwise rotated faces render the tile 90 deg off.
    # z_iso is in face-local coords (sigma0 is), so gather it onto the
    # rect-grid tile-local frame the rest of the plot lives in.
    z_iso_rect = _remap_field_to_rect(z_iso, j_tile_lookup, i_tile_lookup)

    # ---- Step 5: render the PNG. ------------------------------------------
    png_name = (
        f"isopycnal_depth_tile{tile_index:03d}_{stamp}"
        f"_sigma{target_sigma0:.3f}.png"
    )
    png_path = outdir / png_name
    logging.info(f"Rendering isopycnal depth plot: {png_path}")
    plot_isopycnal_depth(
        z_iso_rect=z_iso_rect,
        labels_rect=labels_rect,
        front_labels=overlapping["label"].values,
        XC=XC, YC=YC,
        j_tile_lookup=j_tile_lookup, i_tile_lookup=i_tile_lookup,
        target=target_sigma0,
        tile_index=tile_index, timestamp=timestamp,
        out_path=png_path,
    )
    logging.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    """Build the argument parser and parse ``argv``.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector to parse.  ``None`` (default) reads from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the attributes consumed by :func:`run`.
    """
    p = argparse.ArgumentParser(
        description=(
            "Plot the depth of a chosen isopycnal across one LLC4320 tile, "
            "with the labeled fronts inside the tile outlined in white."
        ),
    )
    p.add_argument("--density-tile",  type=Path, required=True,
                   help="3D density tile NetCDF (sigma0(k,j,i)).")
    p.add_argument("--labels",        type=Path, required=True,
                   help="Global labeled-fronts integer mask (.npy or .nc).")
    p.add_argument("--front-index",   type=Path, required=True,
                   help="Front-index parquet (label, name, x0, y0, x1, y1).")
    p.add_argument("--sigma0",        type=float, required=True,
                   help="Target potential density in kg m^-3.")
    p.add_argument("--isopycnal-nc",  type=Path, default=None,
                   help=(
                       "Optional precomputed isopycnal-depth NetCDF.  When "
                       "supplied (and the file exists) the calculation step "
                       "is skipped and z_iso is read from disk."
                   ))
    p.add_argument("--outdir",        type=Path, default=Path("."),
                   help="Directory for outputs (default: current directory).")
    p.add_argument("--write-nc",      action="store_true",
                   help=(
                       "Also write z_iso to "
                       "isopycnal_depth_tile{idx:03d}_{YYYYMMDDTHH}.nc inside "
                       "--outdir.  Ignored when --isopycnal-nc is used."
                   ))
    return p.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point: configure logging, parse args, dispatch to :func:`run`.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector forwarded to :func:`_parse_args`.

    Returns
    -------
    None
        Side-effects only (logging + the files written by :func:`run`).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = _parse_args(argv)
    run(
        density_tile=args.density_tile,
        labels_path=args.labels,
        front_index_path=args.front_index,
        target_sigma0=args.sigma0,
        outdir=args.outdir,
        isopycnal_nc=args.isopycnal_nc,
        write_nc=args.write_nc,
    )


if __name__ == "__main__":
    main()
