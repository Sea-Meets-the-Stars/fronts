"""
Shared utilities for the dev/rho_and_N scripts.

Holds the bits that were originally inlined into
``plot_top_N_density_profiles.py`` and are now needed by ``plot_isopycnals.py``
as well: tile loading, rect-grid -> face-local index lookup, front-bbox
filtering, gradb2 tile slicing, filename-stem helpers, and the secondary
lon/lat axis helper used by overlay-style maps.

Conventions match the original script:
  * ``TILE_SIZE`` = 720 (LLC4320 face-local tile width).
  * Coordinate frames:
      ``rect`` -- the global 12960 x 17280 rect grid (lon, lat ordered).
      ``rect-tile-local`` -- 0..719 inside one rect tile (used by gradb2/labels).
      ``face-local`` -- 0..719 inside one face tile (used by sigma0/XC/YC).
    The rect <-> face mapping is delegated to ``tile_mapping`` from the
    sibling ``llc4320-native-grid-preprocessing`` repo.
"""

# stdlib
from __future__ import annotations
import os
import sys
from datetime import datetime
from pathlib import Path

# numerical / IO
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# tile_mapping resolution
# ---------------------------------------------------------------------------
# ``tile_mapping`` lives in the sibling llc4320-native-grid-preprocessing repo.
# Resolution order (no machine-specific path is hard-coded):
#   1. the installed ``dbof`` package (``pip install -e`` of the preprocessing
#      repo) -- the normal case, needs no configuration;
#   2. the ``LLC4320_PREPROC_SRC`` environment variable, pointing at the
#      preprocessing repo's ``src`` directory (or directly at ``.../dbof/tiles``);
#   3. otherwise a clear ImportError explaining both options.
def _import_tile_mapping():
    """Import the preprocessing repo's ``tile_mapping`` module, robustly.

    Returns
    -------
    module
        The resolved ``tile_mapping`` module.

    Raises
    ------
    ImportError
        If the ``dbof`` package is not installed and ``LLC4320_PREPROC_SRC``
        is unset or does not point at a directory containing ``tile_mapping``.
    """
    # 1. Installed package -- preferred; works with no configuration.
    try:
        import dbof.tiles.tile_mapping as tm
        return tm
    except ModuleNotFoundError:
        pass

    # 2. Environment-variable override (the "parsed input").  Accept either
    #    the repo ``src`` dir or the ``dbof/tiles`` dir directly.
    env_dir = os.environ.get("LLC4320_PREPROC_SRC")
    if env_dir:
        candidates = [
            Path(env_dir),
            Path(env_dir) / "dbof" / "tiles",
            Path(env_dir) / "src" / "dbof" / "tiles",
        ]
        for cand in candidates:
            if (cand / "tile_mapping.py").is_file():
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
                import tile_mapping as tm  # noqa: E402
                return tm

    raise ImportError(
        "Could not import 'tile_mapping' from the llc4320-native-grid-"
        "preprocessing repo.  Either `pip install -e` that repo so the "
        "`dbof` package is importable, or set the LLC4320_PREPROC_SRC "
        "environment variable to its `src` directory, e.g.\n"
        "    export LLC4320_PREPROC_SRC=/path/to/"
        "llc4320-native-grid-preprocessing/src"
    )


tile_mapping = _import_tile_mapping()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE_SIZE = tile_mapping.TILE_SIZE  # 720
DATE_FMT  = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Filename / timestamp helpers
# ---------------------------------------------------------------------------

def timestamp_to_stamp(timestamp: str) -> str:
    """Convert ``'YYYY-MM-DD HH:MM:SS'`` -> ``'YYYYMMDDTHH'`` for filenames.

    Parameters
    ----------
    timestamp : str
        Timestamp string matching :data:`DATE_FMT` ('YYYY-MM-DD HH:MM:SS').

    Returns
    -------
    str
        Compact filename-safe stamp of the form ``'YYYYMMDDTHH'``.
    """
    return datetime.strptime(timestamp, DATE_FMT).strftime("%Y%m%dT%H")


# ---------------------------------------------------------------------------
# Density-tile + global file loading
# ---------------------------------------------------------------------------

def load_tile(path: Path, var_name: str | None = None) -> xr.Dataset:
    """Open a property-tile NetCDF (any field) and validate its contents.

    Tiles are written by ``llc4320-native-grid-preprocessing``'s
    ``dbof.tiles.generate_tile`` and hold one 3-D data variable
    (``sigma0``, ``Ri``, ``vorticity``, ...) plus ``XC``/``YC``/``Z``
    coordinates and provenance fields (some as attrs, some as scalar
    coords; we accept either location).

    Parameters
    ----------
    path : pathlib.Path
        Path to the tile NetCDF.
    var_name : str or None, optional
        Expected data-variable name.  When ``None`` the variable is
        auto-detected (the single 3-D data variable in the file).

    Returns
    -------
    xarray.Dataset
        Lazy dataset holding ``{var_name}(k, j, i)`` plus coordinates and
        provenance.  The resolved variable name is recorded in
        ``ds.attrs['tile_var_name']`` for convenience.

    Raises
    ------
    KeyError
        If a required provenance field is absent, the requested variable is
        missing, or (in auto-detect mode) the file does not contain exactly
        one 3-D data variable.
    """
    ds = xr.open_dataset(path)
    for key in ("tile_index", "face_index", "rect_i_start", "rect_j_start",
                "timestamp"):
        if key not in ds.attrs and key not in ds.coords:
            raise KeyError(
                f"Tile {path} missing required field '{key}' "
                "(checked attrs and coords)."
            )
    if var_name is None:
        candidates = [v for v in ds.data_vars if ds[v].ndim == 3]
        if len(candidates) != 1:
            raise KeyError(
                f"Tile {path} should contain exactly one 3-D variable; "
                f"found {candidates}.  Pass var_name= explicitly."
            )
        var_name = candidates[0]
    elif var_name not in ds.data_vars:
        raise KeyError(
            f"Tile {path} has no '{var_name}' variable "
            f"(data_vars={list(ds.data_vars)})."
        )
    ds.attrs["tile_var_name"] = var_name
    return ds


def load_density_tile(path: Path) -> xr.Dataset:
    """Backward-compatible wrapper: :func:`load_tile` pinned to ``sigma0``.

    Parameters
    ----------
    path : pathlib.Path
        Path to the density-tile NetCDF.

    Returns
    -------
    xarray.Dataset
        See :func:`load_tile`.
    """
    return load_tile(path, var_name="sigma0")


def check_tiles_consistent(ds_a: xr.Dataset, ds_b: xr.Dataset,
                           label_a: str = "tile A",
                           label_b: str = "tile B") -> None:
    """Hard-error unless two tiles share geometry and timestamp provenance.

    Used by ``fronts_viz_3d`` to confirm that a color-field tile (e.g. Ri)
    was generated for the same tile window and snapshot as the density tile
    that drives the scene geometry.

    Parameters
    ----------
    ds_a, ds_b : xarray.Dataset
        Tile datasets from :func:`load_tile`.
    label_a, label_b : str, optional
        Names used in the error message.

    Raises
    ------
    ValueError
        On any mismatch of tile_index / face_index / rect origins /
        timestamp.  CLI callers should translate this to ``SystemExit``.
    """
    for key in ("tile_index", "face_index", "rect_i_start", "rect_j_start",
                "timestamp"):
        va, vb = tile_scalar(ds_a, key), tile_scalar(ds_b, key)
        if str(va) != str(vb):
            raise ValueError(
                f"Tile provenance mismatch: {label_a} has {key}={va!r} but "
                f"{label_b} has {key}={vb!r}.  Both tiles must come from "
                "the same tile window and timestamp."
            )


def tile_scalar(ds: xr.Dataset, key: str):
    """Lift a scalar-valued provenance field from attrs or coords.

    Parameters
    ----------
    ds : xarray.Dataset
        The density-tile dataset returned by :func:`load_density_tile`.
    key : str
        Name of the scalar field to retrieve.

    Returns
    -------
    object
        Native Python value of the field: ``ds.attrs[key]`` if present,
        otherwise ``ds.coords[key].values.item()``.
    """
    if key in ds.attrs:
        return ds.attrs[key]
    return ds.coords[key].values.item()


def load_gradb2_tile(
    path: Path, rect_j_slice: slice, rect_i_slice: slice,
) -> np.ndarray:
    """Load just the tile window of the global gradb2 (or similar 2D) field.

    Supports two file formats:
      * ``.npy`` -- memory-mapped, sliced directly.
      * ``.nc`` / ``.nc4`` -- opened lazily with xarray and sliced. The
        variable name is auto-detected (first 2D variable, preferring one
        called 'gradb2').

    Parameters
    ----------
    path : pathlib.Path
        Path to the global gradb2 file (``.npy``, ``.nc`` or ``.nc4``).
    rect_j_slice : slice
        Row (j-axis) slice on the global rect grid, length ``TILE_SIZE``.
    rect_i_slice : slice
        Column (i-axis) slice on the global rect grid, length ``TILE_SIZE``.

    Returns
    -------
    numpy.ndarray
        In-RAM float array of shape ``(TILE_SIZE, TILE_SIZE)`` holding the
        gradb2 values inside the tile window.

    Raises
    ------
    ValueError
        If the file extension is unsupported or no usable 2D variable is
        found in the NetCDF.
    """
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path, mmap_mode="r")
        return np.array(arr[rect_j_slice, rect_i_slice])
    if suf in (".nc", ".nc4", ".netcdf"):
        ds = xr.open_dataset(path)
        var_name = "gradb2" if "gradb2" in ds.data_vars else next(
            (v for v in ds.data_vars if ds[v].ndim == 2), None,
        )
        if var_name is None:
            raise ValueError(
                f"Could not find a 2D variable in {path} "
                f"(data_vars={list(ds.data_vars)})."
            )
        # xarray uses (y, x) dim names per the NetCDF; slice with isel by
        # position so we don't have to guess the dim name.
        da = ds[var_name]
        dim_y, dim_x = da.dims
        return da.isel({dim_y: rect_j_slice, dim_x: rect_i_slice}).values
    raise ValueError(f"Unsupported gradb2 file extension: {path.suffix}")


def load_labels_tile(
    path: Path, rect_j_slice: slice, rect_i_slice: slice,
) -> np.ndarray:
    """Load just the tile window of the global labeled-fronts mask.

    Same shape/semantics as :func:`load_gradb2_tile` but tuned for the
    integer label mask: ``.npy`` files are memory-mapped and copied, ``.nc``
    files are auto-scanned for the first 2D variable.

    Parameters
    ----------
    path : pathlib.Path
        Path to the global labeled-fronts file (``.npy`` or ``.nc``).
    rect_j_slice : slice
        Row slice on the global rect grid, length ``TILE_SIZE``.
    rect_i_slice : slice
        Column slice on the global rect grid, length ``TILE_SIZE``.

    Returns
    -------
    numpy.ndarray
        In-RAM integer array of shape ``(TILE_SIZE, TILE_SIZE)`` holding the
        label values inside the tile window.

    Raises
    ------
    ValueError
        If the file extension is unsupported or no usable 2D variable is
        found in the NetCDF.
    """
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path, mmap_mode="r")
        # .copy() converts the mmap view to an in-RAM array -- much faster
        # for downstream label masking and avoids surprising mmap eviction.
        return np.array(arr[rect_j_slice, rect_i_slice])
    if suf in (".nc", ".nc4", ".netcdf"):
        ds = xr.open_dataset(path)
        # Prefer 'labels' / 'label' / 'labeled_fronts' if present, else first 2D.
        for candidate in ("labels", "label", "labeled_fronts"):
            if candidate in ds.data_vars:
                var_name = candidate
                break
        else:
            var_name = next(
                (v for v in ds.data_vars if ds[v].ndim == 2), None,
            )
        if var_name is None:
            raise ValueError(
                f"Could not find a 2D variable in {path} "
                f"(data_vars={list(ds.data_vars)})."
            )
        da = ds[var_name]
        dim_y, dim_x = da.dims
        return da.isel({dim_y: rect_j_slice, dim_x: rect_i_slice}).values
    raise ValueError(f"Unsupported labels file extension: {path.suffix}")


# ---------------------------------------------------------------------------
# Front-bbox filtering
# ---------------------------------------------------------------------------

def filter_overlapping_fronts(
    fronts: pd.DataFrame,
    rect_i_start: int, rect_j_start: int,
    sub_i_lo: int = 0, sub_i_hi: int = TILE_SIZE - 1,
    sub_j_lo: int = 0, sub_j_hi: int = TILE_SIZE - 1,
) -> pd.DataFrame:
    """Keep fronts whose bbox intersects the (possibly restricted) tile window.

    Bboxes follow the convention used by
    ``fronts.properties.io.write_front_index``
    (min_col, min_row, max_col, max_row) -- inclusive on both ends.  A front
    overlaps the window iff the two inclusive boxes share at least one pixel.

    Parameters
    ----------
    fronts : pandas.DataFrame
        Frame with columns ``x0, y0, x1, y1`` (bbox in rect-grid pixel
        indices).
    rect_i_start : int
        Column origin of the tile on the global rect grid.
    rect_j_start : int
        Row origin of the tile on the global rect grid.
    sub_i_lo, sub_i_hi : int, optional
        Tile-local inclusive sub-region in i (default: full tile width).
    sub_j_lo, sub_j_hi : int, optional
        Tile-local inclusive sub-region in j (default: full tile height).

    Returns
    -------
    pandas.DataFrame
        Copy of ``fronts`` filtered to rows whose bbox overlaps the window
        ``[rect_i_start + sub_i_lo, rect_i_start + sub_i_hi]`` x
        ``[rect_j_start + sub_j_lo, rect_j_start + sub_j_hi]``.
    """
    i0 = rect_i_start + sub_i_lo
    i1 = rect_i_start + sub_i_hi
    j0 = rect_j_start + sub_j_lo
    j1 = rect_j_start + sub_j_hi
    # Note: x0/x1 are columns (i-axis), y0/y1 are rows (j-axis).  Bboxes are
    # inclusive, so use <= on both ends.
    mask = (
        (fronts["x0"] <= i1) & (fronts["x1"] >= i0) &
        (fronts["y0"] <= j1) & (fronts["y1"] >= j0)
    )
    return fronts.loc[mask].copy()


# ---------------------------------------------------------------------------
# Rect -> face-local lookup, restricted to the tile
# ---------------------------------------------------------------------------

def build_tile_lookup(
    rect_i_start: int, rect_j_start: int, expected_face: int,
):
    """Return tile-local face-index lookup maps (range 0..TILE_SIZE-1).

    The raw lookup returns *full-face* indices (0..4319); we subtract the
    tile's face offset so the result indexes the density tile's
    ``(j, i)`` axes directly.  A sanity check confirms every pixel of the
    rect tile lives on the expected face (chunk alignment guarantees this).

    Parameters
    ----------
    rect_i_start : int
        Column origin of the tile on the global rect grid.
    rect_j_start : int
        Row origin of the tile on the global rect grid.
    expected_face : int
        Face index (0..12) the tile must lie on, taken from the density-tile
        provenance.

    Returns
    -------
    j_tile_lookup : numpy.ndarray
        ``int16`` array of shape ``(TILE_SIZE, TILE_SIZE)`` mapping each
        rect-grid tile-local pixel to its face-local j (0..TILE_SIZE-1).
    i_tile_lookup : numpy.ndarray
        ``int16`` array of shape ``(TILE_SIZE, TILE_SIZE)`` mapping each
        rect-grid tile-local pixel to its face-local i (0..TILE_SIZE-1).

    Raises
    ------
    RuntimeError
        If the tile spans multiple faces, or if the face it lives on differs
        from ``expected_face``.
    """
    face_id_map, j_face_map, i_face_map = tile_mapping._get_lookup_arrays()
    rect_j_slice = slice(rect_j_start, rect_j_start + TILE_SIZE)
    rect_i_slice = slice(rect_i_start, rect_i_start + TILE_SIZE)
    face_id_tile = face_id_map[rect_j_slice, rect_i_slice]
    j_face_full  = j_face_map[rect_j_slice, rect_i_slice]
    i_face_full  = i_face_map[rect_j_slice, rect_i_slice]
    unique_faces = np.unique(face_id_tile)
    if unique_faces.size != 1 or int(unique_faces[0]) != int(expected_face):
        raise RuntimeError(
            f"Tile at rect (j={rect_j_start}, i={rect_i_start}) maps to faces "
            f"{unique_faces.tolist()}, expected face_index={expected_face} "
            "from the density tile attrs."
        )
    # The tile is 720x720 on the face, so the min over the lookup gives the
    # tile's offset within the face.  Subtract to get tile-local (0..719).
    j_face_offset = int(j_face_full.min())
    i_face_offset = int(i_face_full.min())
    j_tile_lookup = (j_face_full - j_face_offset).astype(np.int16)
    i_tile_lookup = (i_face_full - i_face_offset).astype(np.int16)
    return j_tile_lookup, i_tile_lookup


# ---------------------------------------------------------------------------
# Secondary lon/lat twin axes (shared by overlay-style plots)
# ---------------------------------------------------------------------------

def attach_lonlat_twins(
    ax,
    j_tile_lookup: np.ndarray,
    i_tile_lookup: np.ndarray,
    XC: np.ndarray,
    YC: np.ndarray,
) -> tuple:
    """Add secondary lon (top) and lat (right) axes to a rect-grid tile plot.

    The lookups give the face-local ``(j_tile, i_tile)`` for each rect-grid
    tile-local pixel; ``XC``/``YC`` at those positions give lon/lat in the
    rect frame.  Because the face can be rotated relative to the rect grid,
    lon generally varies with both ``i_local`` and ``j_local``; we sample at
    the mid-row (for lon vs i) and mid-column (for lat vs j) so the
    secondary tick labels reflect the centre of the panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The primary axes drawn in the rect-grid tile-local frame
        (extent ``(0, TILE_SIZE, 0, TILE_SIZE)``).
    j_tile_lookup, i_tile_lookup : numpy.ndarray
        Tile-local face-index lookups from :func:`build_tile_lookup`.
    XC, YC : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` longitude/latitude arrays from the
        density tile (face-local frame).

    Returns
    -------
    ax_lon : matplotlib.axes.Axes
        The top twin axis showing longitudes (mid-row sample).
    ax_lat : matplotlib.axes.Axes
        The right twin axis showing latitudes (mid-column sample).
    """
    mid_j = TILE_SIZE // 2
    mid_i = TILE_SIZE // 2
    lon_along_i = XC[
        j_tile_lookup[mid_j, :], i_tile_lookup[mid_j, :],
    ]  # length TILE_SIZE
    lat_along_j = YC[
        j_tile_lookup[:, mid_i], i_tile_lookup[:, mid_i],
    ]  # length TILE_SIZE

    # Match the secondary axes' tick positions to the primary axes' ticks so
    # the two label rows line up.  Label each tick with the lon/lat at the
    # midpoint of the panel (rounded to 2 decimals).
    ax_lon = ax.twiny()
    ax_lon.set_xlim(ax.get_xlim())
    i_ticks = [t for t in ax.get_xticks() if 0 <= t <= TILE_SIZE]
    ax_lon.set_xticks(i_ticks)
    ax_lon.set_xticklabels(
        [f"{float(lon_along_i[min(int(t), TILE_SIZE - 1)]):.2f}"
         for t in i_ticks]
    )
    ax_lon.set_xlabel("longitude (mid-row sample)")

    ax_lat = ax.twinx()
    ax_lat.set_ylim(ax.get_ylim())
    j_ticks = [t for t in ax.get_yticks() if 0 <= t <= TILE_SIZE]
    ax_lat.set_yticks(j_ticks)
    ax_lat.set_yticklabels(
        [f"{float(lat_along_j[min(int(t), TILE_SIZE - 1)]):.2f}"
         for t in j_ticks]
    )
    ax_lat.set_ylabel("latitude (mid-column sample)")

    return ax_lon, ax_lat
