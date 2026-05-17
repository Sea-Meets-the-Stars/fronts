"""
Plot density profiles at the strongest fronts inside one LLC4320 tile.

Given:
  * a 3D density tile NetCDF produced by
    ``llc4320-native-grid-preprocessing/dev/pot_density/generate_tile_density.py``
    (sigma0(k, j, i) on face-local axes, plus XC, YC, Z and provenance attrs),
  * the global gradb2 field (.npy, shape (12960, 17280)),
  * the global labeled-fronts integer mask (.npy, same shape),
  * the front-index parquet (label, name, x0, y0, x1, y1) and
  * the front-properties parquet (must contain label + gradb2_p90),

this script

  1. picks the N strongest fronts (by gradb2_p90) whose bbox overlaps the tile,
  2. locates each front's peak gradb2 pixel inside the tile,
  3. writes a CSV with label/name/peak-coords/gradb2_p90,
  4. plots the sigma0(z) profile for each picked front (one panel, one colour
     per front, legend = front name), and
  5. plots a log10(gradb2) map of the tile with the N peak positions overlaid
     in matching colours.

The two PNGs and the CSV all share a common stem
``density_profiles_tile{tile_index:03d}_{YYYYMMDDTHH}_topN{N}``.  If the CSV
already exists in --outdir (or is supplied with --top-fronts-csv) the
front-finding step is skipped and the cached peaks are reused.

CLI usage
---------
    python dev/rho_and_N/plot_top_N_density_profiles.py \\
        --density-tile  density_tile207_20121109T12.nc \\
        --gradb2        global_gradb2_20121109T12.npy \\
        --labels        labeled_fronts_global_20121109T12.npy \\
        --front-index   front_index_20121109T12.parquet \\
        --front-properties front_properties_20121109T12.parquet \\
        --N 10
"""

# stdlib -------------------------------------------------------------------
from __future__ import annotations
import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

# numerical / IO ----------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr

# plotting -- headless-safe backend so this runs on cluster nodes too. ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

# repo helpers ------------------------------------------------------------
from fronts.properties.io import load_front_index

# tile_mapping lives next to generate_tile_density.py in a sibling repo.
# We mirror the sys.path trick used by generate_tile_density.py itself.
_TILE_MAPPING_DIR = Path(
    "/home/xavier/Oceanography/python/llc4320-native-grid-preprocessing/"
    "dev/pot_density"
)
if str(_TILE_MAPPING_DIR) not in sys.path:
    sys.path.insert(0, str(_TILE_MAPPING_DIR))
import tile_mapping  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE_SIZE = tile_mapping.TILE_SIZE  # 720
DATE_FMT  = "%Y-%m-%d %H:%M:%S"

# Columns we write to (and read back from) the cached CSV.
# The fixed columns; the last column carries the chosen strength metric
# (e.g. gradb2_p90, gradb2_median, ...) and its column name comes from
# --strength-col so the CSV is self-describing.
CSV_FIXED_COLUMNS = [
    "rank",         # 0..N-1, sorted by strength column desc
    "label",        # integer front label (matches the labels.npy values)
    "name",         # unique front ID string
    "i_rect",       # peak column in the global rect grid
    "j_rect",       # peak row    in the global rect grid
    "i_local",      # peak column in the rect-grid tile-local frame (0..719)
    "j_local",      # peak row    in the rect-grid tile-local frame (0..719)
    "i_tile",       # peak column in the density tile's face-local axes (0..719)
    "j_tile",       # peak row    in the density tile's face-local axes (0..719)
    "lon",          # XC at peak (density tile coord)
    "lat",          # YC at peak (density tile coord)
]

# Strength-column candidates tried in order if --strength-col is not present
# in the supplied properties parquet.  gradb2_p90 is the spec'd column;
# gradb2_median / _mean are reasonable fallbacks.
STRENGTH_FALLBACKS = ("gradb2_p90", "gradb2_median", "gradb2_mean")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _timestamp_to_stamp(timestamp: str) -> str:
    """Convert 'YYYY-MM-DD HH:MM:SS' -> 'YYYYMMDDTHH' for filenames.

    Parameters
    ----------
    timestamp : str
        Timestamp string matching ``DATE_FMT`` ('YYYY-MM-DD HH:MM:SS').

    Returns
    -------
    str
        Compact filename-safe stamp of the form ``'YYYYMMDDTHH'``.
    """
    return datetime.strptime(timestamp, DATE_FMT).strftime("%Y%m%dT%H")


def _build_stem(tile_index: int, timestamp: str, N: int) -> str:
    """Standardised output filename stem shared by CSV + PNGs.

    Parameters
    ----------
    tile_index : int
        LLC4320 rect-tile index (0..431); rendered as a zero-padded 3-digit field.
    timestamp : str
        Timestamp string accepted by :func:`_timestamp_to_stamp`.
    N : int
        Number of top fronts requested; appears verbatim in the stem.

    Returns
    -------
    str
        Filename stem ``'density_profiles_tile{tile_index:03d}_{stamp}_topN{N}'``
        (no extension), used for the CSV, the density-profile PNG, and the
        overlay PNG.
    """
    return (
        f"density_profiles_tile{tile_index:03d}_"
        f"{_timestamp_to_stamp(timestamp)}_topN{N}"
    )


def _make_color_cycle(N: int) -> np.ndarray:
    """Return an (N, 4) RGBA array used by both plots so colours stay in sync.

    tab10 is the most distinguishable choice for N <= 10; for larger N we fall
    back to viridis sampled at N evenly-spaced points.

    Parameters
    ----------
    N : int
        Number of distinct colours required.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 4)`` with RGBA values in [0, 1].
    """
    if N <= 10:
        return cm.get_cmap("tab10")(np.arange(N) % 10)
    return cm.get_cmap("viridis")(np.linspace(0, 1, N))


# ---------------------------------------------------------------------------
# Density-tile + global file loading
# ---------------------------------------------------------------------------

def _load_gradb2_tile(
    path: Path, rect_j_slice: slice, rect_i_slice: slice,
) -> np.ndarray:
    """Load just the tile window of the global gradb2 field.

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


def _load_density_tile(path: Path) -> xr.Dataset:
    """Open the density-tile NetCDF and assert the bits we depend on are present.

    ``generate_tile_density.py`` writes some provenance as attrs and some as
    scalar coords; we accept either location.

    Parameters
    ----------
    path : pathlib.Path
        Path to the density-tile NetCDF produced by ``generate_tile_density.py``.

    Returns
    -------
    xarray.Dataset
        Lazy dataset holding ``sigma0(k, j, i)``, plus the ``XC``, ``YC``,
        ``Z`` coordinates and the ``tile_index``/``face_index``/
        ``rect_i_start``/``rect_j_start``/``timestamp`` provenance fields.

    Raises
    ------
    KeyError
        If a required provenance field or the ``sigma0`` variable is absent.
    """
    ds = xr.open_dataset(path)
    for key in ("tile_index", "face_index", "rect_i_start", "rect_j_start",
                "timestamp"):
        if key not in ds.attrs and key not in ds.coords:
            raise KeyError(
                f"Density tile {path} missing required field '{key}' "
                "(checked attrs and coords)."
            )
    if "sigma0" not in ds.data_vars:
        raise KeyError(f"Density tile {path} has no 'sigma0' variable.")
    return ds


def _tile_scalar(ds: xr.Dataset, key: str):
    """Lift a scalar-valued provenance field from attrs or coords.

    Parameters
    ----------
    ds : xarray.Dataset
        The density-tile dataset returned by :func:`_load_density_tile`.
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


def _resolve_strength_col(
    props_df: pd.DataFrame, requested: str,
) -> str:
    """Return the column to sort by; warn if we fell back to an alternative.

    The original spec calls for ``gradb2_p90``, but the V3 properties parquet
    only ships ``_mean``/``_std``/``_median``.  We accept any column the user
    asks for via --strength-col, otherwise walk the fallback chain.

    Parameters
    ----------
    props_df : pandas.DataFrame
        The front-properties parquet, already read into memory.
    requested : str
        Column name the user asked for via ``--strength-col``.

    Returns
    -------
    str
        The column name to use.  Equals ``requested`` when that column is
        present; otherwise the first available entry in
        :data:`STRENGTH_FALLBACKS` (a ``UserWarning`` is emitted).

    Raises
    ------
    KeyError
        If neither the requested column nor any fallback is present.
    """
    if requested in props_df.columns:
        return requested
    for col in STRENGTH_FALLBACKS:
        if col in props_df.columns:
            warnings.warn(
                f"Strength column '{requested}' not in properties parquet; "
                f"falling back to '{col}'."
            )
            return col
    raise KeyError(
        f"Properties parquet has none of {STRENGTH_FALLBACKS} and no "
        f"'{requested}' column. Available: {list(props_df.columns)}"
    )


def _join_index_and_properties(
    index_df: pd.DataFrame, props_df: pd.DataFrame, strength_col: str,
) -> pd.DataFrame:
    """Inner-join index (label, name, x0..y1) with properties (label, <strength>).

    The properties parquet may key on either ``label`` or ``flabel`` (V3 uses
    ``flabel``); we auto-detect which it is.

    Parameters
    ----------
    index_df : pandas.DataFrame
        Front-index frame with columns ``label, name, x0, y0, x1, y1``.
    props_df : pandas.DataFrame
        Front-properties frame containing a label-key column (``label`` or
        ``flabel``) and ``strength_col``.
    strength_col : str
        Name of the column to carry through onto each joined row.

    Returns
    -------
    pandas.DataFrame
        Joined frame with columns
        ``label, name, x0, y0, x1, y1, <strength_col>``, with NaN strengths
        dropped.

    Raises
    ------
    KeyError
        If neither ``label`` nor ``flabel`` is present in ``props_df``.
    """
    if "label" in props_df.columns:
        props_key = "label"
    elif "flabel" in props_df.columns:
        props_key = "flabel"
    else:
        raise KeyError(
            "Properties parquet has neither 'label' nor 'flabel' column "
            f"(found {list(props_df.columns)[:10]}...)."
        )

    # Only keep the columns we need to avoid blowing up the merged frame.
    props_slim = props_df[[props_key, strength_col]].rename(
        columns={props_key: "label"},
    )
    joined = index_df.merge(props_slim, on="label", how="inner")
    joined = joined.dropna(subset=[strength_col])
    return joined


def _filter_overlapping_fronts(
    fronts: pd.DataFrame,
    rect_i_start: int, rect_j_start: int,
) -> pd.DataFrame:
    """Keep fronts whose bbox (x0, y0, x1, y1) intersects the tile's rect window.

    Bboxes follow the half-open convention used by
    ``fronts.properties.io.write_front_index`` (min_col, min_row, max_col, max_row).
    A front overlaps the tile iff its bbox and the tile's [i0, i1) x [j0, j1)
    have non-empty intersection.

    Parameters
    ----------
    fronts : pandas.DataFrame
        Frame with columns ``x0, y0, x1, y1`` (bbox in rect-grid pixel indices).
    rect_i_start : int
        Column origin of the tile on the global rect grid.
    rect_j_start : int
        Row origin of the tile on the global rect grid.

    Returns
    -------
    pandas.DataFrame
        Copy of ``fronts`` filtered to rows whose bbox overlaps the
        ``TILE_SIZE``-by-``TILE_SIZE`` window starting at
        ``(rect_j_start, rect_i_start)``.
    """
    i0, i1 = rect_i_start, rect_i_start + TILE_SIZE
    j0, j1 = rect_j_start, rect_j_start + TILE_SIZE
    # Note: x0/x1 are columns (i-axis), y0/y1 are rows (j-axis).
    mask = (
        (fronts["x0"] < i1) & (fronts["x1"] >= i0) &
        (fronts["y0"] < j1) & (fronts["y1"] >= j0)
    )
    return fronts.loc[mask].copy()


# ---------------------------------------------------------------------------
# Rect -> face-local lookup, restricted to the tile
# ---------------------------------------------------------------------------

def _build_tile_lookup(
    rect_i_start: int, rect_j_start: int, expected_face: int,
):
    """Return tile-local face-index lookup maps (range 0..719).

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
        rect-grid tile-local pixel to its face-local j (0..719).
    i_tile_lookup : numpy.ndarray
        ``int16`` array of shape ``(TILE_SIZE, TILE_SIZE)`` mapping each
        rect-grid tile-local pixel to its face-local i (0..719).

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
# Front-peak discovery
# ---------------------------------------------------------------------------

def _find_top_n_peaks(
    candidates: pd.DataFrame,
    gradb2_tile: np.ndarray,
    labels_tile: np.ndarray,
    j_tile_lookup: np.ndarray,
    i_tile_lookup: np.ndarray,
    XC: np.ndarray,
    YC: np.ndarray,
    rect_i_start: int,
    rect_j_start: int,
    N: int,
    strength_col: str,
) -> pd.DataFrame:
    """Walk strength-sorted candidates; pick the first N with in-tile pixels.

    Per Round-2 Clarification 5: candidates whose bbox overlaps the tile but
    whose label pixels are entirely outside it (rare) are skipped with a
    warning, and the next-strongest candidate is promoted so we still aim
    for N accepted fronts.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Bbox-overlapping fronts already sorted by ``strength_col`` descending.
        Must contain ``label, name, x0, y0, x1, y1, <strength_col>``.
    gradb2_tile : numpy.ndarray
        Float array of shape ``(TILE_SIZE, TILE_SIZE)``: gradb2 sliced to the
        tile (rect-grid tile-local frame).
    labels_tile : numpy.ndarray
        Integer array of shape ``(TILE_SIZE, TILE_SIZE)``: labeled fronts
        sliced to the tile (same frame as ``gradb2_tile``).
    j_tile_lookup, i_tile_lookup : numpy.ndarray
        Lookup tables returned by :func:`_build_tile_lookup` mapping
        rect-grid tile-local pixels to density-tile face-local indices.
    XC, YC : numpy.ndarray
        ``(720, 720)`` longitude/latitude arrays from the density tile,
        indexed by face-local ``(j, i)``.
    rect_i_start, rect_j_start : int
        Tile origin on the global rect grid (used to recover absolute
        ``i_rect``/``j_rect`` coords).
    N : int
        Maximum number of fronts to accept.
    strength_col : str
        Column name carrying the per-front strength metric; copied through
        onto each accepted row.

    Returns
    -------
    pandas.DataFrame
        Up to ``N`` rows in :data:`CSV_FIXED_COLUMNS` + ``[strength_col]``
        order, sorted by ``strength_col`` descending.

    Raises
    ------
    RuntimeError
        If no candidate has any in-tile pixels.
    """
    accepted_rows = []
    for _, row in candidates.iterrows():
        label = int(row["label"])
        mask = labels_tile == label
        if not mask.any():
            # bbox overlapped but no pixels inside the tile -- promote the next.
            warnings.warn(
                f"Front label={label} has bbox overlapping the tile but no "
                "pixels inside it; skipping."
            )
            continue
        # argmax over the masked gradb2: replace background with -inf so the
        # argmax can only land on a pixel that belongs to this front.
        masked_gradb2 = np.where(mask, gradb2_tile, -np.inf)
        flat_idx = int(np.argmax(masked_gradb2))
        j_local, i_local = np.unravel_index(flat_idx, gradb2_tile.shape)
        # Tile-local face indices (0..719) -- this is what indexes XC/YC/sigma0.
        j_tile = int(j_tile_lookup[j_local, i_local])
        i_tile = int(i_tile_lookup[j_local, i_local])
        accepted_rows.append({
            "rank":       len(accepted_rows),
            "label":      label,
            "name":       row["name"],
            "i_rect":     int(rect_i_start + i_local),
            "j_rect":     int(rect_j_start + j_local),
            "i_local":    int(i_local),
            "j_local":    int(j_local),
            "i_tile":     i_tile,
            "j_tile":     j_tile,
            "lon":        float(XC[j_tile, i_tile]),
            "lat":        float(YC[j_tile, i_tile]),
            strength_col: float(row[strength_col]),
        })
        if len(accepted_rows) == N:
            break

    if not accepted_rows:
        raise RuntimeError(
            "No fronts could be resolved in this tile -- empty candidate pool."
        )
    if len(accepted_rows) < N:
        warnings.warn(
            f"Only {len(accepted_rows)} fronts could be resolved in the tile; "
            f"requested N={N}."
        )
    return pd.DataFrame(accepted_rows, columns=CSV_FIXED_COLUMNS + [strength_col])


# ---------------------------------------------------------------------------
# CSV short-circuit
# ---------------------------------------------------------------------------

def _resolve_csv_path(
    explicit: Path | None, outdir: Path, stem: str,
) -> Path | None:
    """Return an existing CSV path if the user provided one or one is auto-found.

    Precedence: explicit --top-fronts-csv wins.  Otherwise we look for
    ``{outdir}/{stem}.csv``.

    Parameters
    ----------
    explicit : pathlib.Path or None
        Value of ``--top-fronts-csv`` (or None if the flag was not supplied).
    outdir : pathlib.Path
        Output directory checked for a default-named CSV.
    stem : str
        Output stem (see :func:`_build_stem`); the auto-detected CSV is
        ``{outdir}/{stem}.csv``.

    Returns
    -------
    pathlib.Path or None
        Path to a cached CSV the caller should reuse, or ``None`` if no
        cached file is available.

    Raises
    ------
    FileNotFoundError
        If ``explicit`` was supplied but does not exist.
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(
                f"--top-fronts-csv {explicit} does not exist."
            )
        return explicit
    candidate = outdir / f"{stem}.csv"
    if candidate.exists():
        return candidate
    return None


def _load_cached_csv(path: Path, N: int) -> tuple[pd.DataFrame, str]:
    """Load and validate the cached peaks CSV.

    The strength column is whichever column appears after the fixed set.

    Parameters
    ----------
    path : pathlib.Path
        Path to the cached CSV (must exist).
    N : int
        Expected number of rows; mismatches are hard errors per
        Round-2 Clarification 4.

    Returns
    -------
    df : pandas.DataFrame
        The cached peaks, with columns in
        :data:`CSV_FIXED_COLUMNS` + ``[strength_col]`` order.
    strength_col : str
        Name of the strength column inferred from the CSV header.

    Raises
    ------
    ValueError
        If any fixed column is missing, the row count differs from ``N``, or
        the CSV does not contain exactly one extra (strength) column.
    """
    df = pd.read_csv(path)
    missing = [c for c in CSV_FIXED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cached CSV {path} missing columns {missing}; "
            f"expected {CSV_FIXED_COLUMNS} + a strength column."
        )
    if len(df) != N:
        raise ValueError(
            f"Cached CSV {path} has {len(df)} rows but --N={N}. "
            "Delete or rename the stale CSV (or pass --N to match)."
        )
    extra = [c for c in df.columns if c not in CSV_FIXED_COLUMNS]
    if len(extra) != 1:
        raise ValueError(
            f"Cached CSV {path} should have exactly one strength column "
            f"alongside the fixed set; found extras {extra}."
        )
    strength_col = extra[0]
    return df[CSV_FIXED_COLUMNS + [strength_col]], strength_col


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_density_profiles(
    peaks: pd.DataFrame,
    sigma0: np.ndarray,    # (k, j_face_max, i_face_max)
    Z: np.ndarray,         # (k,) negative downward
    colors: np.ndarray,    # (N, 4)
    tile_index: int,
    timestamp: str,
    strength_col: str,
    out_path: Path,
) -> None:
    """Plot sigma0(z) for each accepted front in a single panel.

    Parameters
    ----------
    peaks : pandas.DataFrame
        Output of :func:`_find_top_n_peaks` (or the cached CSV); each row is
        one accepted front.
    sigma0 : numpy.ndarray
        Potential density, shape ``(K, TILE_SIZE, TILE_SIZE)``, indexed by
        depth-level then face-local ``(j_tile, i_tile)``.
    Z : numpy.ndarray
        1-D depth array (m, negative downward), length ``K``.
    colors : numpy.ndarray
        RGBA array of shape ``(N, 4)`` from :func:`_make_color_cycle`.
    tile_index : int
        Tile index used in the panel title.
    timestamp : str
        Timestamp used in the panel title.
    strength_col : str
        Strength column name used in the title.
    out_path : pathlib.Path
        Path to save the PNG.

    Returns
    -------
    None
        The figure is written to ``out_path`` and closed.
    """
    fig, ax = plt.subplots(figsize=(7, 8))
    for n, row in peaks.reset_index(drop=True).iterrows():
        # sigma0[:, j_tile, i_tile] is the column of potential density at the
        # front's peak gradb2 location, using the density tile's face-local axes.
        profile = sigma0[:, int(row["j_tile"]), int(row["i_tile"])]
        ax.plot(profile, Z, color=colors[n], label=str(row["name"]))
    ax.set_xlabel(r"$\sigma_0$ [kg m$^{-3}$]")
    ax.set_ylabel("depth Z [m]")
    # Z is negative downward; Modification 5 caps the view at 500 m depth so
    # the upper-ocean structure (mixed layer + pycnocline) fills the panel.
    ax.set_ylim(-500, 0)
    # Modification 6: minor tick marks on both axes for finer reading.
    ax.minorticks_on()
    ax.tick_params(which="minor", length=3)
    ax.set_title(
        f"Tile {tile_index}  {timestamp}\n"
        f"Top-{len(peaks)} fronts by {strength_col}"
    )
    # Long 'name' strings would crowd the panel; push the legend outside.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize="x-small", borderaxespad=0.0,
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_gradb2_overlay(
    peaks: pd.DataFrame,
    gradb2_tile: np.ndarray,
    colors: np.ndarray,
    tile_index: int,
    timestamp: str,
    XC: np.ndarray,
    YC: np.ndarray,
    j_tile_lookup: np.ndarray,
    i_tile_lookup: np.ndarray,
    out_path: Path,
) -> None:
    """Plot log10(gradb2) of the tile with the N peak positions overlaid.

    Implements Modifications 4 + 7: secondary lon/lat axes (sampled at the
    middle row/column of the tile) and a colorbar truncated at ``-16``.

    Parameters
    ----------
    peaks : pandas.DataFrame
        Output of :func:`_find_top_n_peaks` (or the cached CSV); must contain
        ``i_local`` and ``j_local`` (rect-grid tile-local pixel coords).
    gradb2_tile : numpy.ndarray
        Gradb2 sliced to the tile, shape ``(TILE_SIZE, TILE_SIZE)``.
    colors : numpy.ndarray
        RGBA array of shape ``(N, 4)`` from :func:`_make_color_cycle`; matched
        with the density-profile plot.
    tile_index : int
        Tile index used in the panel title.
    timestamp : str
        Timestamp used in the panel title.
    XC, YC : numpy.ndarray
        ``(TILE_SIZE, TILE_SIZE)`` longitude/latitude arrays from the density
        tile (face-local frame).
    j_tile_lookup, i_tile_lookup : numpy.ndarray
        Tile-local face-index lookups returned by :func:`_build_tile_lookup`;
        used to map rect-grid pixels into the density-tile's lon/lat arrays.
    out_path : pathlib.Path
        Path to save the PNG.

    Returns
    -------
    None
        The figure is written to ``out_path`` and closed.
    """
    # Guard against log10(0) by clipping at the smallest positive value seen.
    positive = gradb2_tile[gradb2_tile > 0]
    floor = float(positive.min()) if positive.size else 1e-30
    safe = np.where(gradb2_tile > 0, gradb2_tile, floor)
    log_gradb2 = np.log10(safe)

    fig, ax = plt.subplots(figsize=(11, 8))
    # Rect-grid tile-local frame: origin='lower' so j (row) increases upward,
    # matching the dot coordinates (i_local, j_local).  Modification 7:
    # vmin=-16 truncates the colorbar at the low end; extend='min' marks the
    # under-range chunk on the colorbar.
    im = ax.imshow(
        log_gradb2, origin="lower", cmap="magma",
        extent=(0, TILE_SIZE, 0, TILE_SIZE),
        vmin=-16, vmax=float(np.nanmax(log_gradb2)),
        aspect="auto",  # required so twiny/twinx (shared axes) can coexist
    )
    # Scatter the N peaks in the same colour cycle used by Plot 1.
    ax.scatter(
        peaks["i_local"].values + 0.5,  # +0.5 centres the dot in the pixel
        peaks["j_local"].values + 0.5,
        c=colors[: len(peaks)], s=60,
        edgecolor="white", linewidth=1.2, zorder=3,
    )
    ax.set_xlabel("i (rect-grid tile-local)")
    ax.set_ylabel("j (rect-grid tile-local)")
    ax.set_title(
        f"Tile {tile_index}  {timestamp}\n"
        f"log10(gradb2) with top-{len(peaks)} peaks"
    )

    # ---- Modification 4: secondary lon/lat axes. --------------------------
    # The lookups give the face-local (j_tile, i_tile) for each rect-grid
    # tile-local pixel; XC/YC at those positions give lon/lat in the rect
    # frame.  Because the face can be rotated relative to the rect grid, lon
    # generally varies with both i_local and j_local; we sample at the mid-row
    # (resp. mid-column) so the secondary tick labels reflect the centre of
    # the panel.
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
        [f"{float(lon_along_i[min(int(t), TILE_SIZE - 1)]):.2f}" for t in i_ticks]
    )
    ax_lon.set_xlabel("longitude (mid-row sample)")

    ax_lat = ax.twinx()
    ax_lat.set_ylim(ax.get_ylim())
    j_ticks = [t for t in ax.get_yticks() if 0 <= t <= TILE_SIZE]
    ax_lat.set_yticks(j_ticks)
    ax_lat.set_yticklabels(
        [f"{float(lat_along_j[min(int(t), TILE_SIZE - 1)]):.2f}" for t in j_ticks]
    )
    ax_lat.set_ylabel("latitude (mid-column sample)")

    # Place the colorbar past the latitude axis on the right so they don't
    # overlap.  Anchor the colorbar to ax_lat (the rightmost twin axis) and
    # widen the pad so the lat tick labels have room.
    fig.colorbar(
        im, ax=ax_lat,
        label=r"$\log_{10}(\nabla b^2)$",
        extend="min",  # signal that values below vmin=-16 are clipped
        pad=0.10, fraction=0.05,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run(
    density_tile: Path,
    gradb2_path: Path,
    labels_path: Path,
    front_index_path: Path,
    front_properties_path: Path,
    N: int,
    outdir: Path,
    top_fronts_csv: Path | None,
    strength_col: str,
) -> None:
    """End-to-end: load tile -> resolve N peaks -> write CSV -> render plots.

    Parameters
    ----------
    density_tile : pathlib.Path
        Path to the 3D density tile NetCDF (sigma0(k, j, i)).
    gradb2_path : pathlib.Path
        Path to the global gradb2 field (.npy or .nc).
    labels_path : pathlib.Path
        Path to the global labeled-fronts integer mask (.npy).
    front_index_path : pathlib.Path
        Path to the front-index parquet (label, name, x0..y1).
    front_properties_path : pathlib.Path
        Path to the front-properties parquet (must contain the strength
        column or a fallback).
    N : int
        Number of strongest fronts to keep.
    outdir : pathlib.Path
        Output directory for the CSV and the two PNGs (created if absent).
    top_fronts_csv : pathlib.Path or None
        Optional path to a cached peaks CSV; when supplied (or when the
        default-named CSV already exists in ``outdir``) the front-finding
        step is short-circuited.
    strength_col : str
        Column to rank fronts by; subject to the fallback chain in
        :func:`_resolve_strength_col`.

    Returns
    -------
    None
        Writes ``{outdir}/{stem}.csv``, ``{outdir}/{stem}.png`` and
        ``{outdir}/{stem}_gradb2map.png`` where ``stem`` is built by
        :func:`_build_stem`.

    Raises
    ------
    RuntimeError
        If no fronts overlap the tile bbox.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Step 2: load density tile and lift the bits we need (attrs or coords). ---
    ds = _load_density_tile(density_tile)
    tile_index   = int(_tile_scalar(ds, "tile_index"))
    face_index   = int(_tile_scalar(ds, "face_index"))
    rect_i_start = int(_tile_scalar(ds, "rect_i_start"))
    rect_j_start = int(_tile_scalar(ds, "rect_j_start"))
    timestamp    = str(_tile_scalar(ds, "timestamp"))
    stem = _build_stem(tile_index, timestamp, N)
    logging.info(f"Output stem: {stem}")

    # sigma0 is small (51 * 720 * 720 * 4 ~= 53 MB float32) so load eagerly.
    sigma0 = ds["sigma0"].values
    Z      = ds["Z"].values
    XC     = ds["XC"].values
    YC     = ds["YC"].values

    rect_j_slice = slice(rect_j_start, rect_j_start + TILE_SIZE)
    rect_i_slice = slice(rect_i_start, rect_i_start + TILE_SIZE)

    # ---- Step 3: short-circuit on cached CSV if available. -----------------
    cached_csv = _resolve_csv_path(top_fronts_csv, outdir, stem)

    if cached_csv is not None:
        logging.info(f"Reusing cached peaks CSV: {cached_csv}")
        peaks, strength_col = _load_cached_csv(cached_csv, N)
        logging.info(f"Cached strength column: {strength_col}")
    else:
        # ---- Steps 4-5: candidate fronts, sorted by the strength column. ----
        logging.info("Loading front index + properties parquets")
        index_df = load_front_index(front_index_path)
        props_df = pd.read_parquet(front_properties_path)
        strength_col = _resolve_strength_col(props_df, strength_col)
        logging.info(f"Sorting candidates by '{strength_col}'")
        joined = _join_index_and_properties(index_df, props_df, strength_col)
        overlapping = _filter_overlapping_fronts(
            joined, rect_i_start, rect_j_start,
        )
        if overlapping.empty:
            raise RuntimeError(
                "No fronts in the index overlap the tile bbox -- nothing to plot."
            )
        candidates = overlapping.sort_values(
            strength_col, ascending=False,
        ).reset_index(drop=True)
        logging.info(
            f"{len(candidates)} candidate fronts overlap the tile bbox"
        )

        # ---- Step 6: rect -> tile-local face-index lookup. -----------------
        j_tile_lookup, i_tile_lookup = _build_tile_lookup(
            rect_i_start, rect_j_start, face_index,
        )

        # ---- Step 7: gradb2 + labels for the tile window. -----------------
        logging.info("Loading gradb2 and labels tile windows")
        gradb2_tile = _load_gradb2_tile(gradb2_path, rect_j_slice, rect_i_slice)
        labels = np.load(labels_path, mmap_mode="r")
        # .copy() converts the mmap view to an in-RAM array -- much faster for
        # the per-label masking loop and avoids surprising mmap eviction.
        labels_tile = np.array(labels[rect_j_slice, rect_i_slice])

        peaks = _find_top_n_peaks(
            candidates=candidates,
            gradb2_tile=gradb2_tile,
            labels_tile=labels_tile,
            j_tile_lookup=j_tile_lookup,
            i_tile_lookup=i_tile_lookup,
            XC=XC, YC=YC,
            rect_i_start=rect_i_start,
            rect_j_start=rect_j_start,
            N=N,
            strength_col=strength_col,
        )

        # ---- Step 8: write CSV. --------------------------------------------
        csv_path = outdir / f"{stem}.csv"
        peaks.to_csv(csv_path, index=False)
        logging.info(f"Wrote peaks CSV: {csv_path}")

    # If we short-circuited, build the lookup + gradb2_tile that the overlay
    # plot needs (Modification 4 reads lon/lat off these).
    if cached_csv is not None:
        logging.info("Loading gradb2 tile + tile lookup for the overlay plot")
        gradb2_tile = _load_gradb2_tile(gradb2_path, rect_j_slice, rect_i_slice)
        j_tile_lookup, i_tile_lookup = _build_tile_lookup(
            rect_i_start, rect_j_start, face_index,
        )

    # ---- Steps 10-11: render the two PNGs. --------------------------------
    colors = _make_color_cycle(len(peaks))

    profiles_png = outdir / f"{stem}.png"
    _plot_density_profiles(
        peaks=peaks, sigma0=sigma0, Z=Z, colors=colors,
        tile_index=tile_index, timestamp=timestamp,
        strength_col=strength_col,
        out_path=profiles_png,
    )
    logging.info(f"Wrote density-profile plot: {profiles_png}")

    overlay_png = outdir / f"{stem}_gradb2map.png"
    _plot_gradb2_overlay(
        peaks=peaks, gradb2_tile=gradb2_tile, colors=colors,
        tile_index=tile_index, timestamp=timestamp,
        XC=XC, YC=YC,
        j_tile_lookup=j_tile_lookup, i_tile_lookup=i_tile_lookup,
        out_path=overlay_png,
    )
    logging.info(f"Wrote gradb2 overlay plot: {overlay_png}")


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
            "Plot density profiles at the N strongest fronts in an LLC4320 tile."
        ),
    )
    p.add_argument("--density-tile",      type=Path, required=True,
                   help="3D density tile NetCDF (sigma0(k,j,i)).")
    p.add_argument("--gradb2",            type=Path, required=True,
                   help="Global gradb2 field on the rect grid (.npy).")
    p.add_argument("--labels",            type=Path, required=True,
                   help="Global labeled-fronts integer mask (.npy).")
    p.add_argument("--front-index",       type=Path, required=True,
                   help="Front-index parquet (label, name, x0..y1).")
    p.add_argument("--front-properties",  type=Path, required=True,
                   help="Front-properties parquet (must include gradb2_p90).")
    p.add_argument("--N",                 type=int, default=10,
                   help="Number of strongest fronts to keep (default: 10).")
    p.add_argument("--outdir",            type=Path, default=Path("."),
                   help="Directory for outputs (default: current directory).")
    p.add_argument("--top-fronts-csv",    type=Path, default=None,
                   help=(
                       "Optional cached peaks CSV; if supplied (or if a CSV "
                       "with the default name already exists in --outdir) the "
                       "front-finding step is skipped."
                   ))
    p.add_argument("--strength-col",      type=str, default="gradb2_p90",
                   help=(
                       "Column in the front-properties parquet used to rank "
                       "fronts (default: gradb2_p90). Falls back to "
                       f"{STRENGTH_FALLBACKS[1:]} if the requested column is "
                       "absent."
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
        gradb2_path=args.gradb2,
        labels_path=args.labels,
        front_index_path=args.front_index,
        front_properties_path=args.front_properties,
        N=args.N,
        outdir=args.outdir,
        top_fronts_csv=args.top_fronts_csv,
        strength_col=args.strength_col,
    )


if __name__ == "__main__":
    main()

#python /home/xavier/Oceanography/python/fronts/dev/rho_and_N/plot_top_N_density_profiles.py \
#  --density-tile     /home/xavier/Projects/Oceanography/data/OGCM/LLC/Fronts/V3/20121109_120000/density_tile301_20121109T12.nc \
#  --gradb2           /home/xavier/Projects/Oceanography/data/OGCM/LLC/Fronts/V3/20121109_120000/LLC4320_2012-11-09T12_00_00_gradb2_v3.nc \
#  --labels           /home/xavier/Projects/Oceanography/data/OGCM/LLC/Fronts/V3/20121109_120000/labeled_fronts_global_20121109T12_00_00_v3_bin_D.npy \
#  --front-index      /home/xavier/Projects/Oceanography/data/OGCM/LLC/Fronts/V3/20121109_120000/front_index_20121109T12_00_00_v3_bin_D.parquet \
#  --front-properties /home/xavier/Projects/Oceanography/data/OGCM/LLC/Fronts/V3/20121109_120000/front_properties_20121109T12_00_00_v3_bin_D.parquet \
#  --N 10 \
#  --outdir .
