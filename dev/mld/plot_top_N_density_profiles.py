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
from fronts.llc.analysis import mixed_layer_depth as _mld_helper

# Shared utilities (formerly inlined here) live in density_utils.py next to
# this script.  Make sure that directory is importable regardless of how the
# script is invoked, then pull in the helpers under the underscore-prefixed
# names the rest of this module already uses.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from density_utils import (  # noqa: E402
    TILE_SIZE,
    DATE_FMT,
    timestamp_to_stamp as _timestamp_to_stamp,
    load_density_tile as _load_density_tile,
    tile_scalar as _tile_scalar,
    build_tile_lookup as _build_tile_lookup,
    filter_overlapping_fronts as _filter_overlapping_fronts,
    load_gradb2_tile as _load_gradb2_tile,
    attach_lonlat_twins as _attach_lonlat_twins,
)

from IPython import embed

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
STRENGTH_FALLBACKS = ("gradb2_p90", "gradb2_mean", "gradb2_median")

# Mixed-layer-depth threshold: the depth at which sigma0 exceeds the surface
# value by this amount marks the bottom of the mixed layer (see the
# Definitions section in prompts/fronts_N.md).
MLD_DELTA_SIGMA0 = 0.03  # kg m^-3
MLD_REFERENCE_DEPTH_M = 10.0  # metres — Bodner et al. reference depth (≈ 9.66 m)

# Buoyancy-frequency constants for Modification 10.
G_GRAV   = 9.81     # m s^-2
RHO_REF  = 1027.0   # kg m^-3  (reference seawater density for sigma0)

# Isopycnal-depth threshold and temperature-MLD threshold for Modification 11
# (definitions copied verbatim from prompts/fronts_N.md).
ISOPYCNAL_DELTA_SIGMA0 = 0.125  # kg m^-3 above 10 m density
TMLD_DELTA_THETA       = 0.2    # K (positive: theta drops by this much)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _build_stem(tile_index: int, timestamp: str, N: int, region_name: str | None = None) -> str:
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
        f"density_profiles_{region_name}_tile{tile_index:03d}_"
        f"{_timestamp_to_stamp(timestamp)}_topN{N}"
    )


def _mixed_layer_depth(sigma0_profile: np.ndarray, Z: np.ndarray) -> float | None:
    """Mixed-layer depth from a single sigma0(z) profile (0.03 kg m^-3 criterion).

    Thin wrapper around :func:`fronts.llc.analysis.mixed_layer_depth` with
    the local module's ``MLD_DELTA_SIGMA0`` threshold (0.03 kg m^-3) and
    ``MLD_REFERENCE_DEPTH_M`` reference depth.  Kept under its private name
    so existing call sites in this script are unaffected.
    """
    return _mld_helper(
        sigma0_profile, Z,
        delta_sigma0=MLD_DELTA_SIGMA0,
        reference_depth_m=MLD_REFERENCE_DEPTH_M,
    )


def _isopycnal_depth(sigma0_profile: np.ndarray, Z: np.ndarray) -> float | None:
    """Isopycnal-depth diagnostic (0.125 kg m^-3 criterion).

    Thin wrapper around :func:`fronts.llc.analysis.mixed_layer_depth` with
    the larger ``ISOPYCNAL_DELTA_SIGMA0`` threshold.  Kept under its
    private name so existing call sites in this script are unaffected.
    """
    return _mld_helper(
        sigma0_profile, Z,
        delta_sigma0=ISOPYCNAL_DELTA_SIGMA0,
        reference_depth_m=MLD_REFERENCE_DEPTH_M,
    )


def _temperature_mld(theta_profile: np.ndarray, Z: np.ndarray) -> float | None:
    """Temperature mixed-layer depth (Modification 11, Definition 3).

    Same convention as :func:`_mixed_layer_depth` but using a temperature
    decrease of ``TMLD_DELTA_THETA`` (0.2 K) relative to the 10 m theta value.
    Returns the deepest LLC level where the temperature has not yet dropped
    by the threshold amount.

    Parameters
    ----------
    theta_profile : numpy.ndarray
        1-D potential temperature column, length ``K``, in K or degC (only
        the difference matters).
    Z : numpy.ndarray
        1-D depth array, length ``K``, in metres (negative downward).

    Returns
    -------
    float or None
        Depth in metres (negative downward) of the deepest level still within
        ``TMLD_DELTA_THETA`` of the 10 m temperature, or ``None`` if the
        profile is empty or all-NaN at the reference depth.
    """
    if theta_profile.size == 0:
        return None
    k_10m = int(np.abs(np.abs(Z) - float(MLD_REFERENCE_DEPTH_M)).argmin())
    theta_ref = float(theta_profile[k_10m])
    if not np.isfinite(theta_ref):
        return None
    # theta typically decreases with depth in a stable column; the mixed layer
    # is where theta_ref - theta <= 0.2.
    delta_t = theta_ref - theta_profile
    z_masked = np.where(delta_t <= TMLD_DELTA_THETA)
    return float(Z[z_masked].min())


def _resolve_subregion(
    i_rect_range: tuple[int, int] | None,
    j_rect_range: tuple[int, int] | None,
    rect_i_start: int,
    rect_j_start: int,
) -> tuple[int, int, int, int]:
    """Convert optional user rect-grid ranges to tile-local pixel bounds.

    The returned bounds are inclusive on both ends and clipped to the tile.

    Parameters
    ----------
    i_rect_range : tuple of (int, int) or None
        Optional ``(i_min, i_max)`` in global rect-grid columns; if ``None``,
        the i-axis is unconstrained (covers the full tile width).
    j_rect_range : tuple of (int, int) or None
        Optional ``(j_min, j_max)`` in global rect-grid rows; if ``None``,
        the j-axis is unconstrained.
    rect_i_start, rect_j_start : int
        Tile origin on the global rect grid.

    Returns
    -------
    tuple of (int, int, int, int)
        ``(i_lo, i_hi, j_lo, j_hi)`` tile-local pixel bounds, inclusive, with
        each coord in ``[0, TILE_SIZE - 1]``.  When a range is unconstrained
        the corresponding bound spans the full tile.

    Raises
    ------
    ValueError
        If a supplied range is degenerate (min > max) or lies entirely
        outside the tile.
    """
    def _clip_range(rng, start):
        if rng is None:
            return 0, TILE_SIZE - 1
        lo_global, hi_global = rng
        if lo_global > hi_global:
            raise ValueError(
                f"Range ({lo_global}, {hi_global}) is degenerate (min > max)."
            )
        lo = max(0, lo_global - start)
        hi = min(TILE_SIZE - 1, hi_global - start)
        if lo > hi:
            raise ValueError(
                f"Range ({lo_global}, {hi_global}) lies outside the tile "
                f"(tile origin {start}, size {TILE_SIZE})."
            )
        return lo, hi
    i_lo, i_hi = _clip_range(i_rect_range, rect_i_start)
    j_lo, j_hi = _clip_range(j_rect_range, rect_j_start)
    return i_lo, i_hi, j_lo, j_hi


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
# Temperature-tile loading (kept here -- specific to this script)
# ---------------------------------------------------------------------------

def _load_theta_tile(
    path: Path, tile_index: int, rect_i_start: int, rect_j_start: int,
) -> np.ndarray:
    """Open the temperature tile NetCDF and return its ``Theta(k, j, i)`` array.

    Used only by Modification 11 (the MLD diagnostics plot).  The theta tile
    must describe the same LLC4320 tile as the density tile; the function
    cross-checks ``tile_index`` plus the rect-grid origin to catch mismatched
    file pairs.

    Parameters
    ----------
    path : pathlib.Path
        Path to the temperature-tile NetCDF (produced by ``generate_tile.py``).
    tile_index : int
        Tile index from the density tile -- the theta tile must match.
    rect_i_start, rect_j_start : int
        Rect-grid origin from the density tile -- the theta tile must match.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(K, TILE_SIZE, TILE_SIZE)`` holding theta in
        the same face-local ``(j, i)`` frame as ``sigma0``.

    Raises
    ------
    KeyError
        If the file lacks a ``Theta`` variable.
    RuntimeError
        If the theta tile describes a different tile than the density tile.
    """
    ds = xr.open_dataset(path)
    if "Theta" not in ds.data_vars:
        raise KeyError(f"Theta tile {path} has no 'Theta' variable.")
    theta_tile_idx = int(_tile_scalar(ds, "tile_index"))
    theta_i0       = int(_tile_scalar(ds, "rect_i_start"))
    theta_j0       = int(_tile_scalar(ds, "rect_j_start"))
    if (theta_tile_idx != tile_index
            or theta_i0 != rect_i_start
            or theta_j0 != rect_j_start):
        raise RuntimeError(
            f"Theta tile mismatch: theta has tile_index={theta_tile_idx} "
            f"rect_i={theta_i0} rect_j={theta_j0}; density has "
            f"tile_index={tile_index} rect_i={rect_i_start} rect_j={rect_j_start}."
        )
    return ds["Theta"].values


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
    sub_i_lo: int = 0,
    sub_i_hi: int = TILE_SIZE - 1,
    sub_j_lo: int = 0,
    sub_j_hi: int = TILE_SIZE - 1,
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
    sub_i_lo, sub_i_hi, sub_j_lo, sub_j_hi : int, optional
        Inclusive tile-local pixel bounds.  When supplied, both the label
        mask and the gradb2 peak search are restricted to this box, so the
        accepted fronts are the strongest within the sub-region
        (Modification 9).

    Returns
    -------
    pandas.DataFrame
        Up to ``N`` rows in :data:`CSV_FIXED_COLUMNS` + ``[strength_col]``
        order, sorted by ``strength_col`` descending.

    Raises
    ------
    RuntimeError
        If no candidate has any pixels inside the (possibly restricted)
        search window.
    """
    # Build the sub-region mask once (cheap: a single bool array slice op).
    sub_mask = np.zeros_like(labels_tile, dtype=bool)
    sub_mask[sub_j_lo:sub_j_hi + 1, sub_i_lo:sub_i_hi + 1] = True

    accepted_rows = []
    for _, row in candidates.iterrows():
        label = int(row["label"])
        # Intersect label mask with the sub-region so the peak can only land
        # inside the user-specified window (full tile by default).
        mask = (labels_tile == label) & sub_mask
        if not mask.any():
            # bbox overlapped but no pixels inside the (sub-)tile -- promote
            # the next-strongest candidate.
            warnings.warn(
                f"Front label={label} has bbox overlapping the search window "
                "but no label pixels inside it; skipping."
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
        no strength-style column is present alongside the fixed set.
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
    # The strength column is whichever extra starts with 'gradb2_' (the spec'd
    # families: gradb2_p90 / _median / _mean / ...).  Any other extras (e.g.
    # 'z_mld' added by a downstream step) are passed through unchanged so the
    # cached CSV remains forward-compatible.
    strength_candidates = [c for c in extra if c.startswith("gradb2_")]
    if not strength_candidates:
        raise ValueError(
            f"Cached CSV {path} has no strength column starting with "
            f"'gradb2_' alongside the fixed set; extras={extra}."
        )
    strength_col = strength_candidates[0]
    # Preserve fixed columns first, then strength, then any other extras the
    # CSV happened to carry, so downstream plots can read columns like z_mld.
    other_extras = [c for c in extra if c != strength_col]
    return df[CSV_FIXED_COLUMNS + [strength_col] + other_extras], strength_col


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

    Modification 8: an open circle of the matching colour is drawn on each
    profile at the mixed-layer depth (sigma0 threshold = MLD_DELTA_SIGMA0).

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
        # Modification 8: open circle at the mixed-layer depth.  The marker
        # is plotted at (sigma0(z_mld), z_mld) on the same line so it sits
        # right on the profile; facecolor='none' makes it open, with the
        # edge in the matching colour.
        z_mld = _mixed_layer_depth(profile, Z)
        #z_mld = row["z_mld"]
        #embed(header="z_mld 876")
        if z_mld is not None:
            # Pick the sigma0 value matching the MLD by linear interp in z.
            sigma0_at_mld = float(np.interp(z_mld, Z[::-1], profile[::-1]))
            ax.plot(
                sigma0_at_mld, z_mld,
                marker="o", markersize=4,
                markerfacecolor="none",
                markeredgecolor=colors[n], markeredgewidth=1.5,
                linestyle="none",
            )
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


def _n2_profile(sigma0_profile: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Squared buoyancy frequency N^2(z) from a sigma0 column.

    Computed from the hydrostatic stratification formula
    ``N^2 = -(g / rho_ref) * d(sigma0)/dz``.

    ``np.gradient`` handles the irregular LLC4320 Z spacing automatically;
    it returns a finite-difference derivative with one value per input depth.
    The minus sign converts depth-increasing sigma0 (downward) into a
    positive N^2 (stable stratification).

    Parameters
    ----------
    sigma0_profile : numpy.ndarray
        1-D potential density column, length ``K``, kg m^-3.
    Z : numpy.ndarray
        1-D depth array, length ``K``, in metres (negative downward).

    Returns
    -------
    numpy.ndarray
        N^2 in s^-2, same length as the input profile.
    """
    # np.gradient(f, x) returns df/dx using centred differences in the
    # interior and one-sided at the endpoints; works with Z's non-uniform
    # spacing.  Z is negative downward, so dsigma0/dZ is negative for a
    # stably stratified column -- the leading minus restores N^2 > 0.
    dsig_dz = np.gradient(sigma0_profile, Z)
    return -(G_GRAV / RHO_REF) * dsig_dz


def _plot_n2_profiles(
    peaks: pd.DataFrame,
    sigma0: np.ndarray,
    Z: np.ndarray,
    colors: np.ndarray,
    tile_index: int,
    timestamp: str,
    strength_col: str,
    out_path: Path,
) -> None:
    """Plot N^2(z) for each accepted front in a single panel (Modification 10).

    Mirrors :func:`_plot_density_profiles` -- same colour cycle, same depth
    range, same MLD open circle -- but the x-axis is squared buoyancy
    frequency instead of potential density.

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
        RGBA array of shape ``(N, 4)`` from :func:`_make_color_cycle`; matched
        with the density-profile plot.
    tile_index : int
        Tile index used in the panel title.
    timestamp : str
        Timestamp used in the panel title.
    strength_col : str
        Strength column name used in the title.
    out_path : pathlib.Path
        Path to save the PNG (caller is responsible for the ``N2_`` prefix).

    Returns
    -------
    None
        The figure is written to ``out_path`` and closed.
    """
    fig, ax = plt.subplots(figsize=(7, 8))
    for n, row in peaks.reset_index(drop=True).iterrows():
        sigma0_profile = sigma0[:, int(row["j_tile"]), int(row["i_tile"])]
        n2 = _n2_profile(sigma0_profile, Z)
        #embed(header="n2 1002")
        ax.plot(n2, Z, color=colors[n], label=str(row["name"]))
        # Open circle at the MLD, consistent with the density-profile plot.
        z_mld = _mixed_layer_depth(sigma0_profile, Z)
        if z_mld is not None:
            n2_at_mld = float(np.interp(z_mld, Z[::-1], n2[::-1]))
            ax.plot(
                n2_at_mld, z_mld,
                marker="o", markersize=4,
                markerfacecolor="none",
                markeredgecolor=colors[n], markeredgewidth=1.5,
                linestyle="none",
            )
    ax.set_xlabel(r"$N^2$ [s$^{-2}$]")
    ax.set_ylabel("depth Z [m]")
    # Mod 5: cap at 500 m to match the density plot.
    ax.set_ylim(-500, 0)
    # Mod 6: minor tick marks on both axes.
    ax.minorticks_on()
    ax.tick_params(which="minor", length=3)
    # Scientific-notation x-axis -- N^2 is tiny (~1e-4 in the pycnocline).
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_title(
        f"Tile {tile_index}  {timestamp}\n"
        f"Top-{len(peaks)} fronts by {strength_col}  --  N$^2$ profiles"
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize="x-small", borderaxespad=0.0,
    )
    # Zero line
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_mld_diagnostics(
    peaks: pd.DataFrame,
    sigma0: np.ndarray,
    theta: np.ndarray,
    Z: np.ndarray,
    colors: np.ndarray,
    tile_index: int,
    timestamp: str,
    strength_col: str,
    out_path: Path,
    sub_i_lo: int = 0,
    sub_i_hi: int = TILE_SIZE - 1,
    sub_j_lo: int = 0,
    sub_j_hi: int = TILE_SIZE - 1,
    has_subregion: bool = False,
) -> None:
    """Plot density profiles + three MLD diagnostics zoomed on the upper ocean.

    Modification 11.  For each accepted front this draws the sigma0(z) line
    (same colour as the main density-profile plot) and three markers at the
    depths returned by

        * :func:`_mixed_layer_depth`  -- circle      ('o', delta sigma = 0.03)
        * :func:`_isopycnal_depth`    -- square      ('s', delta sigma = 0.125)
        * :func:`_temperature_mld`    -- triangle    ('^', delta theta = 0.2 K)

    Markers are open (facecolor='none') in the front's colour so the underlying
    line is visible.  The y-axis is auto-zoomed to ~1.5x the deepest of the
    three depths across all fronts so the upper-ocean structure fills the
    panel.

    Parameters
    ----------
    peaks : pandas.DataFrame
        Output of :func:`_find_top_n_peaks` (or the cached CSV); each row is
        one accepted front.  ``j_tile`` and ``i_tile`` columns are required.
    sigma0 : numpy.ndarray
        Potential density, shape ``(K, TILE_SIZE, TILE_SIZE)``.
    theta : numpy.ndarray
        Potential temperature, shape ``(K, TILE_SIZE, TILE_SIZE)``, on the
        same grid as ``sigma0``.
    Z : numpy.ndarray
        1-D depth array (m, negative downward), length ``K``.
    colors : numpy.ndarray
        RGBA array of shape ``(N, 4)`` from :func:`_make_color_cycle`; matched
        with the density-profile plot.
    tile_index : int
        Tile index used in the panel title.
    timestamp : str
        Timestamp used in the panel title.
    strength_col : str
        Strength column name used in the title (for traceability).
    out_path : pathlib.Path
        Path to save the PNG (caller is responsible for the ``MLD_`` prefix).
    sub_i_lo, sub_i_hi, sub_j_lo, sub_j_hi : int, optional
        Inclusive tile-local pixel bounds used by Modification 12 when
        computing the median density profile across the tile (or sub-region).
        Default: full tile.
    has_subregion : bool, optional
        Pass-through flag controlling only the legend label of the median
        line (``"median (sub-region)"`` vs ``"median (tile)"``).

    Returns
    -------
    None
        The figure is written to ``out_path`` and closed.
    """
    # Wider-than-tall figure per the updated Modification 11 spec; the wider
    # aspect leaves more room for the two side-by-side legends.
    fig, ax = plt.subplots(figsize=(12, 7))
    # Track the deepest diagnostic depth across all fronts so the y-axis can
    # auto-zoom on the upper ocean; also collect every sigma0 profile so we
    # can tighten the x-axis to the data range that survives the y-zoom.
    deepest = 0.0  # most negative depth seen
    sigma0_profiles: list[np.ndarray] = []
    for n, row in peaks.reset_index(drop=True).iterrows():
        j_t, i_t = int(row["j_tile"]), int(row["i_tile"])
        sigma0_profile = sigma0[:, j_t, i_t]
        theta_profile  = theta[:, j_t, i_t]
        sigma0_profiles.append(sigma0_profile)
        line, = ax.plot(
            sigma0_profile, Z, color=colors[n], label=str(row["name"]),
        )
        # Compute the three diagnostic depths.  Each returns None when the
        # column never crosses the threshold (rare for an LLC profile).
        z_mld  = _mixed_layer_depth(sigma0_profile, Z)
        z_iso  = _isopycnal_depth(sigma0_profile, Z)
        z_tmld = _temperature_mld(theta_profile, Z)
        # Plot markers at (sigma0_at_z, z) for each defined diagnostic.
        for z_def, marker in (
            (z_mld,  "o"),
            (z_iso,  "s"),
            (z_tmld, "^"),
        ):
            if z_def is None:
                continue
            sigma0_at = float(np.interp(z_def, Z[::-1], sigma0_profile[::-1]))
            ax.plot(
                sigma0_at, z_def,
                marker=marker, markersize=8,
                markerfacecolor="none",
                markeredgecolor=colors[n], markeredgewidth=1.5,
                linestyle="none",
            )
            if z_def < deepest:
                deepest = z_def

    # ---- Modification 12: median sigma0/theta profile over the (sub-)tile.
    # Slice both 3D arrays to the requested window and take the median across
    # the spatial axes for every depth level.  Using nanmedian shields us from
    # the rare NaN that creeps in from masked/land cells, though the LLC tiles
    # are typically ocean-only.
    j_slice = slice(sub_j_lo, sub_j_hi + 1)
    i_slice = slice(sub_i_lo, sub_i_hi + 1)
    sigma0_median = np.nanmedian(sigma0[:, j_slice, i_slice], axis=(1, 2))
    theta_median  = np.nanmedian(theta[:, j_slice, i_slice], axis=(1, 2))
    sigma0_profiles.append(sigma0_median)  # feed into the x-axis tightening below
    median_label = (
        "median (sub-region)" if has_subregion else "median (tile)"
    )
    # Solid black line, slightly thicker than the per-front lines so it stays
    # readable against the colourful background; high zorder keeps it on top.
    ax.plot(
        sigma0_median, Z,
        color="black", linewidth=2.2, linestyle="-",
        label=median_label, zorder=5,
    )
    # Filled black markers for the median's three MLD diagnostics -- the
    # filled face distinguishes them from the open per-front markers.
    z_mld_med  = _mixed_layer_depth(sigma0_median, Z)
    z_iso_med  = _isopycnal_depth(sigma0_median, Z)
    z_tmld_med = _temperature_mld(theta_median, Z)
    for z_def, marker in (
        (z_mld_med,  "o"),
        (z_iso_med,  "s"),
        (z_tmld_med, "^"),
    ):
        if z_def is None:
            continue
        sigma0_at = float(np.interp(z_def, Z[::-1], sigma0_median[::-1]))
        ax.plot(
            sigma0_at, z_def,
            marker=marker, markersize=10,
            markerfacecolor="black", markeredgecolor="black",
            markeredgewidth=1.5, linestyle="none",
            zorder=6,
        )
        if z_def < deepest:
            deepest = z_def

    ax.set_xlabel(r"$\sigma_0$ [kg m$^{-3}$]")
    ax.set_ylabel("depth Z [m]")
    # Auto-zoom: 1.5x the deepest diagnostic, clamped to [-500, 0] so we never
    # extend past the rest of the plots' depth range.
    y_bot = max(-500.0, 1.5 * float(deepest)) if deepest < 0 else -200.0
    ax.set_ylim(y_bot, 0)
    # Modification 11 update: x-axis stops at the max sigma0 in the visible
    # depth window so the panel isn't padded by deep-water densities.
    in_window = (Z >= y_bot) & (Z <= 0)
    if np.any(in_window):
        sigma0_window = np.concatenate(
            [p[in_window] for p in sigma0_profiles]
        )
        sigma0_window = sigma0_window[np.isfinite(sigma0_window)]
        if sigma0_window.size:
            x_min = float(np.min(sigma0_window))
            x_max = float(np.max(sigma0_window))
            # Tiny pad on the left so markers near x_min aren't clipped.
            pad = 0.02 * (x_max - x_min) if x_max > x_min else 0.05
            ax.set_xlim(x_min - pad, x_max)
    # Minor ticks for finer reading (consistent with the other depth plots).
    ax.minorticks_on()
    ax.tick_params(which="minor", length=3)
    ax.set_title(
        f"Tile {tile_index}  {timestamp}\n"
        f"MLD diagnostics  --  top-{len(peaks)} fronts by {strength_col}"
    )
    # Build a legend with two parts: the per-front colour list, and a
    # symbol-key explaining the three definitions.  The colour legend goes
    # outside-right; the symbol-key is in-panel.
    front_legend = ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize="x-small", borderaxespad=0.0, title="front",
    )
    ax.add_artist(front_legend)  # keep when we add a second legend below
    symbol_handles = [
        plt.Line2D(
            [0], [0], marker=m, markersize=8, linestyle="none",
            markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=1.5, label=lab,
        )
        for m, lab in (
            ("o", f"MLD (Δσ₀ ≥ {MLD_DELTA_SIGMA0})"),
            ("s", f"Isopycnal depth (Δσ₀ ≥ {ISOPYCNAL_DELTA_SIGMA0})"),
            ("^", f"T-MLD (Δθ ≥ {TMLD_DELTA_THETA} K)"),
        )
    ]
    symbol_legend = ax.legend(
        handles=symbol_handles, loc="upper right",
        fontsize="medium", title="definition",
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    fig.tight_layout()
    # Both legends are persisted as separate artists; pass them as
    # bbox_extra_artists so bbox_inches='tight' makes room for the per-front
    # legend (which sits outside the axes on the right).
    fig.savefig(
        out_path, dpi=140, bbox_inches="tight",
        bbox_extra_artists=(front_legend, symbol_legend),
    )
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
    subregion: tuple[int, int, int, int] | None = None,
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
    subregion : tuple of (int, int, int, int) or None, optional
        Inclusive tile-local bounds ``(i_lo, i_hi, j_lo, j_hi)``.  When
        supplied (Modification 9), the search window is drawn as a dashed
        white rectangle so the user can see what was actually scanned.

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
    # Modification 9: outline the user-specified search window when present.
    if subregion is not None:
        i_lo, i_hi, j_lo, j_hi = subregion
        ax.add_patch(plt.Rectangle(
            (i_lo, j_lo), (i_hi - i_lo + 1), (j_hi - j_lo + 1),
            fill=False, edgecolor="white", linestyle="--", linewidth=1.5,
            zorder=4,
        ))
    ax.set_xlabel("i (rect-grid tile-local)")
    ax.set_ylabel("j (rect-grid tile-local)")
    ax.set_title(
        f"Tile {tile_index}  {timestamp}\n"
        f"log10(gradb2) with top-{len(peaks)} peaks"
    )

    # Modification 4: secondary lon/lat axes -- shared helper in density_utils.
    _, ax_lat = _attach_lonlat_twins(ax, j_tile_lookup, i_tile_lookup, XC, YC)

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
    region_name: str, 
    i_rect_range: tuple[int, int] | None = None,
    j_rect_range: tuple[int, int] | None = None,
    theta_path: Path | None = None,
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
    i_rect_range : tuple of (int, int) or None, optional
        Inclusive global rect-grid column bounds restricting the search
        sub-region (Modification 9).  ``None`` means use the full tile.
    j_rect_range : tuple of (int, int) or None, optional
        Inclusive global rect-grid row bounds (see ``i_rect_range``).
    theta_path : pathlib.Path or None, optional
        Path to a temperature tile NetCDF.  When supplied (Modification 11)
        an extra ``MLD_{stem}.png`` is written with three MLD diagnostics
        per front; when omitted, that figure is skipped.
    region_name : str or None, optional
        Region name used in the panel title.

    Returns
    -------
    None
        Writes ``{outdir}/{stem}.csv``, ``{outdir}/{stem}.png`` and
        ``{outdir}/{stem}_gradb2map.png`` where ``stem`` is built by
        :func:`_build_stem` (with a sub-region suffix when one is supplied).

    Raises
    ------
    RuntimeError
        If no fronts overlap the search window.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Step 2: load density tile and lift the bits we need (attrs or coords). ---
    ds = _load_density_tile(density_tile)
    tile_index   = int(_tile_scalar(ds, "tile_index"))
    face_index   = int(_tile_scalar(ds, "face_index"))
    rect_i_start = int(_tile_scalar(ds, "rect_i_start"))
    rect_j_start = int(_tile_scalar(ds, "rect_j_start"))
    timestamp    = str(_tile_scalar(ds, "timestamp"))

    # Modification 9: resolve the optional sub-region into tile-local pixel
    # bounds and append it to the output stem so cached CSVs from different
    # sub-regions don't collide.
    sub_i_lo, sub_i_hi, sub_j_lo, sub_j_hi = _resolve_subregion(
        i_rect_range, j_rect_range, rect_i_start, rect_j_start,
    )
    has_subregion = (i_rect_range is not None) or (j_rect_range is not None)
    base_stem = _build_stem(tile_index, timestamp, N, region_name)
    if has_subregion:
        i0g = rect_i_start + sub_i_lo
        i1g = rect_i_start + sub_i_hi
        j0g = rect_j_start + sub_j_lo
        j1g = rect_j_start + sub_j_hi
        stem = f"{base_stem}_i{i0g}-{i1g}_j{j0g}-{j1g}"
    else:
        stem = base_stem
    logging.info(f"Output stem: {stem}")
    if has_subregion:
        logging.info(
            f"Sub-region (tile-local) i=[{sub_i_lo}, {sub_i_hi}], "
            f"j=[{sub_j_lo}, {sub_j_hi}]"
        )

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
            sub_i_lo=sub_i_lo, sub_i_hi=sub_i_hi,
            sub_j_lo=sub_j_lo, sub_j_hi=sub_j_hi,
        )
        if overlapping.empty:
            raise RuntimeError(
                "No fronts in the index overlap the search window -- nothing to plot."
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
            sub_i_lo=sub_i_lo, sub_i_hi=sub_i_hi,
            sub_j_lo=sub_j_lo, sub_j_hi=sub_j_hi,
        )

        # ---- Step 8: compute the mixed-layer depth for each front. ---------
        mlds = []
        for n, row in peaks.reset_index(drop=True).iterrows():
            profile = sigma0[:, int(row["j_tile"]), int(row["i_tile"])]
            mlds.append(_mixed_layer_depth(profile, Z))
        peaks["z_mld"] = np.abs(mlds)

        # ---- Step 9: write CSV. --------------------------------------------
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

    # Modification 10: companion N^2 plot, "N2_" prefix on the filename.
    n2_png = outdir / f"N2_{stem}.png"
    _plot_n2_profiles(
        peaks=peaks, sigma0=sigma0, Z=Z, colors=colors,
        tile_index=tile_index, timestamp=timestamp,
        strength_col=strength_col,
        out_path=n2_png,
    )
    logging.info(f"Wrote N^2 profile plot: {n2_png}")

    # Modification 11: optional MLD-diagnostics plot.  Only generated when the
    # caller passed a theta tile; otherwise we log a one-liner and move on.
    if theta_path is not None:
        logging.info(f"Loading theta tile: {theta_path}")
        theta = _load_theta_tile(
            theta_path, tile_index, rect_i_start, rect_j_start,
        )
        mld_png = outdir / f"MLD_{stem}.png"
        _plot_mld_diagnostics(
            peaks=peaks, sigma0=sigma0, theta=theta, Z=Z, colors=colors,
            tile_index=tile_index, timestamp=timestamp,
            strength_col=strength_col,
            out_path=mld_png,
            sub_i_lo=sub_i_lo, sub_i_hi=sub_i_hi,
            sub_j_lo=sub_j_lo, sub_j_hi=sub_j_hi,
            has_subregion=has_subregion,
        )
        logging.info(f"Wrote MLD diagnostics plot: {mld_png}")
    else:
        logging.info("No --theta given; skipping MLD diagnostics plot.")

    overlay_png = outdir / f"{stem}_gradb2map.png"
    _plot_gradb2_overlay(
        peaks=peaks, gradb2_tile=gradb2_tile, colors=colors,
        tile_index=tile_index, timestamp=timestamp,
        XC=XC, YC=YC,
        j_tile_lookup=j_tile_lookup, i_tile_lookup=i_tile_lookup,
        out_path=overlay_png,
        subregion=(sub_i_lo, sub_i_hi, sub_j_lo, sub_j_hi) if has_subregion else None,
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
    p.add_argument("--i-rect-range",      type=int, nargs=2, default=None,
                   metavar=("I_MIN", "I_MAX"),
                   help=(
                       "Optional inclusive global rect-grid column bounds "
                       "(Modification 9). When supplied, the N strongest "
                       "fronts are chosen from within this sub-region only."
                   ))
    p.add_argument("--j-rect-range",      type=int, nargs=2, default=None,
                   metavar=("J_MIN", "J_MAX"),
                   help=(
                       "Optional inclusive global rect-grid row bounds. "
                       "Combines with --i-rect-range; either may be omitted."
                   ))
    p.add_argument("--theta",             type=Path, default=None,
                   help=(
                       "Optional path to a temperature tile NetCDF "
                       "(Modification 11). When supplied, an extra "
                       "MLD_<stem>.png is written that compares the three "
                       "MLD definitions (Mixed Layer, Isopycnal, "
                       "Temperature MLD) per front. When omitted, the "
                       "MLD diagnostics plot is skipped."
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
        i_rect_range=tuple(args.i_rect_range) if args.i_rect_range else None,
        j_rect_range=tuple(args.j_rect_range) if args.j_rect_range else None,
        theta_path=args.theta,
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
