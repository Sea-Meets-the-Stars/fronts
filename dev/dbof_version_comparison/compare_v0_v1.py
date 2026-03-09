#!/usr/bin/env python
"""
02_compare_v0_v1.py

Compares DBOF v0 and v1 cutouts:
  1. Loads v0 subset (from 01_load_v0_data.py output)
  2. Loads v1 output (Zarr images + Parquet metadata)
  3. Matches cutouts by lat/lon proximity (nearest neighbor)
  4. Produces side-by-side comparison plots:
     (a) SST / Theta maps
     (b) log₁₀(Divb²) / log_gradb maps
     (c) Histograms with statistics in legend

Usage:
    python 02_compare_v0_v1.py

    Set environment variables to override defaults:
        V0_SUBSET    path to v0_subset.npz (from script 01)
        V1_ZARR      path/URL to v1 Zarr dataset
        V1_META      path/URL to v1 metadata directory (parquet)
        MAX_PAIRS    max number of matched pairs to plot (default: 10)
        MAX_DIST_DEG max matching distance in degrees (default: 0.5)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import fsspec

# ── Configuration ───────────────────────────────────────────────────

V0_SUBSET_FILE = os.environ.get(
    "V0_SUBSET",
    "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/dbof_testing/version_comparison/v0_subset.npz"
)

# v1 S3 access parameters (matching the notebook / data_access config)
V1_S3_ENDPOINT = os.environ.get(
    "V1_S3_ENDPOINT", "https://s3-west.nrp-nautilus.io"
)
V1_BUCKET  = os.environ.get("V1_BUCKET",  "dbof")
V1_FOLDER  = os.environ.get("V1_FOLDER",  "native_grid_dbof_training_data")
V1_RUN_ID  = os.environ.get("V1_RUN_ID",  "year_4x150_20260203_201716")
V1_DATASET = os.environ.get("V1_DATASET", "dataset_creation.zarr")

# v1 channel indices (from year_4x150_20260203_201716 config)
# Output order: Eta(0), Salt(1), Theta(2), U(3), V(4), W(5), log_gradb(6)
V1_THETA_IDX     = 2  # Theta — compare to v0 SSTK
V1_SALT_IDX      = 1  # Salt
V1_LOG_GRADB_IDX = 6  # log_gradb — compare to log10(v0 Divb2)

MAX_PAIRS    = int(os.environ.get("MAX_PAIRS", "10"))
MAX_DIST_DEG = float(os.environ.get("MAX_DIST_DEG", "0.5"))  # ~55 km

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/dbof_testing/version_comparison/plots/")

# ── Data Loading ────────────────────────────────────────────────────

def load_v0(path):
    """Load the v0 subset saved by 01_load_v0_data.py."""
    print(f"Loading v0 data from: {path}")
    data = np.load(path, allow_pickle=True)
    v0 = {
        "sst":   data["sst"],       # (N, 64, 64)
        "divb2": data["divb2"],      # (N, 64, 64)
        "lats":  data["lats"],       # (N,)
        "lons":  data["lons"],       # (N,)
        "times": data["times"],      # (N,) strings
    }
    print(f"  {len(v0['lats'])} cutouts loaded")
    print(f"  Timestamps: {np.unique(v0['times'])}")
    return v0


def create_s3_filesystems(s3_endpoint):
    """
    Create S3 filesystems using the same protocol as
    dbof.io.filesystems.create_s3_filesystems().
    """
    fs = fsspec.filesystem(
        "s3",
        asynchronous=True,
        client_kwargs={"endpoint_url": s3_endpoint},
        config_kwargs={
            "signature_version": "s3v4",
            "request_checksum_calculation": "when_required",
            "s3": {
                "addressing_style": "path",
                "payload_signing_enabled": False,
                "use_accelerate_endpoint": False,
                "use_dualstack_endpoint": False,
            },
        },
    )
    fs_synch = fsspec.filesystem(
        "s3",
        asynchronous=False,
        client_kwargs={"endpoint_url": s3_endpoint},
        config_kwargs={
            "signature_version": "s3v4",
            "request_checksum_calculation": "when_required",
            "s3": {
                "addressing_style": "path",
                "payload_signing_enabled": False,
                "use_accelerate_endpoint": False,
                "use_dualstack_endpoint": False,
            },
        },
    )
    return fs, fs_synch


def load_v1(s3_endpoint, bucket, folder, run_id, dataset_name):
    """
    Load v1 data using the same protocol as the
    access_dbof_dataset.ipynb notebook:
      1. Create fsspec S3 filesystems
      2. Glob metadata parquet files and load with pd.read_parquet
      3. Open Zarr via ZarrDatasetReader (or directly)
      4. Build UUID-based mapping from metadata to Zarr rows

    IMPORTANT: Metadata parquet rows do NOT correspond to Zarr image
    rows by position. The v1 pipeline uses UUID-based linking:
      - Zarr has `image_ids` array of UUIDs
      - Metadata has `dataset_index` column of UUIDs
    We build a lookup dict: dataset_index → zarr_row_index.
    """
    import zarr
    from zarr.storage import FsspecStore

    print(f"  Connecting to S3: {s3_endpoint}")
    fs, fs_synch = create_s3_filesystems(s3_endpoint)

    # ── Load metadata (same as notebook) ───────────────────────
    meta_glob = f"{bucket}/{folder}/{run_id}/metadata/*.parquet"
    print(f"  Globbing metadata: {meta_glob}")
    files = fs_synch.glob(meta_glob)
    if not files:
        print(f"  ERROR: No metadata parquet files found at {meta_glob}")
        sys.exit(1)
    print(f"  Found {len(files)} metadata parquet files")

    meta_df = pd.read_parquet(files, filesystem=fs_synch)
    print(f"  Loaded metadata: {len(meta_df)} rows")
    print(f"  Metadata columns: {list(meta_df.columns)}")

    # ── Open Zarr images (same as notebook's ZarrDatasetReader) ─
    zarr_path = f"s3://{bucket}/{folder}/{run_id}/{dataset_name}"
    print(f"  Opening Zarr: {zarr_path}")
    store = FsspecStore(path=zarr_path, fs=fs)
    root = zarr.open_group(store=store, mode="r")
    images = root["images"]
    image_ids = root["image_ids"]
    print(f"  images shape: {images.shape}")
    print(f"  image_ids shape: {image_ids.shape}")

    # ── Build UUID-based lookup: dataset_index → zarr row ──────
    # The notebook does: meta_df.set_index("dataset_index").loc[img_uuid]
    # We need the reverse: for each metadata row, find its zarr index.
    zarr_ids = np.array(image_ids[:])  # load all UUIDs from Zarr
    uuid_to_zarr_idx = {uid: i for i, uid in enumerate(zarr_ids)}

    if "dataset_index" in meta_df.columns:
        meta_uuids = meta_df["dataset_index"].values
        zarr_indices = []
        n_missing = 0
        for uid in meta_uuids:
            zidx = uuid_to_zarr_idx.get(uid, None)
            if zidx is None:
                # Try bytes/string conversion
                if isinstance(uid, bytes):
                    zidx = uuid_to_zarr_idx.get(uid.decode("utf-8"), None)
                elif isinstance(uid, str):
                    zidx = uuid_to_zarr_idx.get(uid.encode("utf-8"), None)
            zarr_indices.append(zidx if zidx is not None else -1)
            if zidx is None:
                n_missing += 1
        meta_df["_zarr_idx"] = zarr_indices
        n_matched = (meta_df["_zarr_idx"] >= 0).sum()
        print(f"  UUID alignment: {n_matched}/{len(meta_df)} metadata rows → Zarr rows")
        if n_missing > 0:
            print(f"  WARNING: {n_missing} metadata rows have no matching Zarr image!")
    else:
        print("  WARNING: No 'dataset_index' column in metadata!")
        print("  Falling back to positional indexing (UNSAFE)")
        meta_df["_zarr_idx"] = np.arange(len(meta_df))

    return images, image_ids, meta_df


# ── Unit Conversion ─────────────────────────────────────────────────

def convert_v0_sst(sst_array):
    """
    Convert v0 SSTK to °C if needed.
    v0 SSTK may be in Kelvin (~270-310) or already °C.
    Also may be de-meaned by preprocessing.
    Returns (converted_array, label, needs_offset_note).
    """
    median = np.nanmedian(sst_array)
    if median > 200:
        return sst_array - 273.15, "v0 SSTK (converted to °C)", True
    elif abs(median) < 5:
        return sst_array, "v0 SSTK (de-meaned)", False
    else:
        return sst_array, "v0 SSTK (°C)", False


def convert_v0_divb2_to_log(divb2_array):
    """
    Convert v0 Divb² to log₁₀ scale for comparison with v1 log_gradb.

    IMPORTANT UNITS NOTE:
    Both v0 and v1 compute |∇b|² using the same buoyancy formula
    (b = g*ρ/ρ_ref, g=0.0098, ρ_ref=1025). However, the gradient
    calculation uses different dx units:
      - v0: dx = 2.25 (km) — gradient is in s⁻² / km
      - v1: dx in meters (via xgcm grid metrics) — gradient is in s⁻² / m

    Since |∇b|² ∝ 1/dx², and 1 km = 1000 m:
      v0_gradb2 / v1_gradb2 ≈ (1000)² = 10⁶

    In log₁₀ space this is an offset of ~6:
      log₁₀(v0_gradb2) ≈ log₁₀(v1_gradb2) + 6

    We apply this correction to align the two for comparison.
    """
    mean_val = np.nanmean(divb2_array)
    min_val = np.nanmin(divb2_array)

    # Correction for v0 dx in km vs v1 dx in meters
    # |∇b|² scales as 1/dx², so km→m introduces factor of 10⁶
    LOG10_DX_CORRECTION = -6.0  # subtract 6 from v0 log values to match v1

    if min_val < 0:
        # Likely de-meaned — can't take log. Return as-is with warning.
        return divb2_array, "v0 Divb² (de-meaned, raw)", True
    else:
        # Raw values — take log10, then correct for dx units
        with np.errstate(divide="ignore", invalid="ignore"):
            log_divb2 = np.log10(np.where(divb2_array > 0, divb2_array, np.nan))
        log_divb2 += LOG10_DX_CORRECTION  # correct km→m scaling
        return log_divb2, "v0 log₁₀(Divb²) [dx-corrected]", False


# ── Proximity Matching ──────────────────────────────────────────────

def haversine_deg(lat1, lon1, lat2, lon2):
    """
    Haversine distance in degrees (approximate).
    Returns angular separation in degrees, correctly handling the
    longitude wraparound at ±180°.
    """
    R_EARTH_KM = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    km = 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return km / 111.0  # convert back to approximate degrees


def match_cutouts(v0_lats, v0_lons, v0_times,
                  v1_lats, v1_lons, v1_times,
                  max_dist_deg=0.5):
    """
    Match v0 cutouts to nearest v1 cutouts at the same timestamp.
    Uses haversine distance to correctly handle longitude wraparound.
    Returns list of (v0_idx, v1_idx, distance_deg) tuples.
    """
    matches = []

    # Match per timestamp
    v0_unique_times = np.unique(v0_times)
    for ts in v0_unique_times:
        v0_mask = v0_times == ts

        # Find matching v1 timestamp (closest)
        v1_ts_values = pd.to_datetime(v1_times)
        v0_ts = pd.Timestamp(str(ts))
        time_diffs = np.abs((v1_ts_values - v0_ts).total_seconds())

        # Accept v1 snapshots within 6 days of v0 timestamp
        # (v0 and v1 have slightly different temporal grids, gaps up to ~5 days)
        TIME_TOLERANCE_SECONDS = 6 * 86400  # 6 days
        v1_time_mask = time_diffs < TIME_TOLERANCE_SECONDS
        if v1_time_mask.sum() == 0:
            print(f"  WARNING: No v1 cutouts within 6 days of {ts}")
            continue

        v0_idx = np.where(v0_mask)[0]
        v1_idx = np.where(v1_time_mask)[0]

        # Use BallTree with haversine metric for correct spherical distance
        # (handles longitude wraparound at ±180°)
        try:
            from sklearn.neighbors import BallTree
            v1_coords_rad = np.radians(
                np.column_stack([v1_lats[v1_idx], v1_lons[v1_idx]]))
            tree = BallTree(v1_coords_rad, metric="haversine")

            v0_coords_rad = np.radians(
                np.column_stack([v0_lats[v0_idx], v0_lons[v0_idx]]))
            # haversine returns radians; convert to approx degrees
            dists_rad, nearest = tree.query(v0_coords_rad, k=1)
            distances = np.degrees(dists_rad.ravel())
            nearest = nearest.ravel()
        except ImportError:
            # Fallback: brute-force haversine (slower but no sklearn needed)
            print("  (sklearn not available; using brute-force haversine matching)")
            distances = []
            nearest = []
            for vi in range(len(v0_idx)):
                d = haversine_deg(
                    v0_lats[v0_idx[vi]], v0_lons[v0_idx[vi]],
                    v1_lats[v1_idx], v1_lons[v1_idx],
                )
                best_j = np.argmin(d)
                distances.append(d[best_j])
                nearest.append(best_j)
            distances = np.array(distances)
            nearest = np.array(nearest)

        for i, (d, j) in enumerate(zip(distances, nearest)):
            if d <= max_dist_deg:
                matches.append((v0_idx[i], v1_idx[j], d))

    # Sort by distance (best matches first)
    matches.sort(key=lambda x: x[2])
    return matches


# ── Plotting ────────────────────────────────────────────────────────

def compute_stats(arr):
    """Compute statistics for a 2D array, ignoring NaN."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": np.mean(finite),
        "std":  np.std(finite),
        "min":  np.min(finite),
        "max":  np.max(finite),
    }


def stats_legend(stats, label):
    """Format statistics as a legend string."""
    return (f"{label}\n"
            f"  mean={stats['mean']:.4g}\n"
            f"  std={stats['std']:.4g}\n"
            f"  min={stats['min']:.4g}\n"
            f"  max={stats['max']:.4g}")


def plot_pair(v0_map, v1_map, v0_label, v1_label, field_name,
              v0_lat, v0_lon, v1_lat, v1_lon, dist_deg,
              pair_idx, output_dir):
    """
    Plot a single matched pair:
      Top row:    v0 map | v1 map  (shared colorbar)
      Bottom row: v0 histogram | v1 histogram  (with stats in legend)
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{field_name} — Pair {pair_idx}  "
        f"(v0: {v0_lat:.2f}°, {v0_lon:.2f}° | "
        f"v1: {v1_lat:.2f}°, {v1_lon:.2f}° | "
        f"Δ={dist_deg:.3f}°)",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05],
                           hspace=0.35, wspace=0.3)

    # Shared color range
    all_vals = np.concatenate([v0_map.ravel(), v1_map.ravel()])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if len(finite_vals) > 0:
        vmin = np.percentile(finite_vals, 2)
        vmax = np.percentile(finite_vals, 98)
    else:
        vmin, vmax = 0, 1

    # ── Maps ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(v0_map, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax0.set_title(v0_label, fontsize=11)
    ax0.set_xlabel("pixel i")
    ax0.set_ylabel("pixel j")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(v1_map, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax1.set_title(v1_label, fontsize=11)
    ax1.set_xlabel("pixel i")

    cbar_ax = fig.add_subplot(gs[0, 2])
    fig.colorbar(im1, cax=cbar_ax)

    # ── Histograms ──────────────────────────────────────────────
    v0_stats = compute_stats(v0_map)
    v1_stats = compute_stats(v1_map)

    ax2 = fig.add_subplot(gs[1, 0])
    v0_finite = v0_map[np.isfinite(v0_map)].ravel()
    if len(v0_finite) > 0:
        ax2.hist(v0_finite, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax2.set_title("v0 Distribution", fontsize=11)
    ax2.set_xlabel(field_name)
    ax2.set_ylabel("Count")
    # Stats in legend box
    textstr = (f"μ = {v0_stats['mean']:.4g}\n"
               f"σ = {v0_stats['std']:.4g}\n"
               f"min = {v0_stats['min']:.4g}\n"
               f"max = {v0_stats['max']:.4g}")
    ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    ax3 = fig.add_subplot(gs[1, 1])
    v1_finite = v1_map[np.isfinite(v1_map)].ravel()
    if len(v1_finite) > 0:
        ax3.hist(v1_finite, bins=50, alpha=0.7, color="darkorange", edgecolor="white")
    ax3.set_title("v1 Distribution", fontsize=11)
    ax3.set_xlabel(field_name)
    ax3.set_ylabel("Count")
    textstr = (f"μ = {v1_stats['mean']:.4g}\n"
               f"σ = {v1_stats['std']:.4g}\n"
               f"min = {v1_stats['min']:.4g}\n"
               f"max = {v1_stats['max']:.4g}")
    ax3.text(0.97, 0.97, textstr, transform=ax3.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    # Save
    fname = os.path.join(output_dir, f"pair_{pair_idx:03d}_{field_name.replace(' ', '_')}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_aggregate_histograms(v0_all, v1_all, v0_label, v1_label,
                              field_name, output_dir):
    """
    Overlay histogram of ALL matched v0 vs v1 values for one field.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    v0_flat = v0_all[np.isfinite(v0_all)].ravel()
    v1_flat = v1_all[np.isfinite(v1_all)].ravel()

    # Shared bins
    all_vals = np.concatenate([v0_flat, v1_flat])
    bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 80)

    v0_stats = compute_stats(v0_all)
    v1_stats = compute_stats(v1_all)

    ax.hist(v0_flat, bins=bins, alpha=0.5, color="steelblue", edgecolor="white",
            label=(f"v0 {v0_label}\n"
                   f"  μ={v0_stats['mean']:.4g}, σ={v0_stats['std']:.4g}\n"
                   f"  min={v0_stats['min']:.4g}, max={v0_stats['max']:.4g}"))
    ax.hist(v1_flat, bins=bins, alpha=0.5, color="darkorange", edgecolor="white",
            label=(f"v1 {v1_label}\n"
                   f"  μ={v1_stats['mean']:.4g}, σ={v1_stats['std']:.4g}\n"
                   f"  min={v1_stats['min']:.4g}, max={v1_stats['max']:.4g}"))

    ax.set_xlabel(field_name, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Aggregate {field_name} Distribution — All Matched Pairs", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")

    fname = os.path.join(output_dir, f"aggregate_{field_name.replace(' ', '_')}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ── Main ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Load v0 and v1 data")
    print("=" * 70)

    v0 = load_v0(V0_SUBSET_FILE)

    print(f"\nLoading v1 from S3:")
    print(f"  bucket={V1_BUCKET}, folder={V1_FOLDER}, run_id={V1_RUN_ID}")
    v1_images, v1_image_ids, v1_meta = load_v1(
        V1_S3_ENDPOINT, V1_BUCKET, V1_FOLDER, V1_RUN_ID, V1_DATASET
    )

    print(f"\nv0: {len(v0['lats'])} cutouts")
    print(f"v1: {len(v1_meta)} cutouts, image shape: {v1_images.shape}")
    print()

    # ── Convert units ───────────────────────────────────────────
    print("=" * 70)
    print("STEP 2: Convert units")
    print("=" * 70)

    v0_sst, v0_sst_label, sst_converted = convert_v0_sst(v0["sst"])
    v0_log_divb2, v0_divb2_label, divb2_is_demeaned = convert_v0_divb2_to_log(v0["divb2"])

    if sst_converted:
        print("  v0 SSTK: converted from Kelvin to °C (subtracted 273.15)")
    if divb2_is_demeaned:
        print("  WARNING: v0 Divb² appears de-meaned — log conversion not applied.")
        print("           Comparison with v1 log_gradb will be qualitative only.")
        print("           Consider loading raw (un-preprocessed) v0 field files instead.")
    print()

    # ── Match cutouts ───────────────────────────────────────────
    print("=" * 70)
    print("STEP 3: Match cutouts by proximity")
    print("=" * 70)

    v1_lats = v1_meta["center_lat"].values
    v1_lons = v1_meta["center_lon"].values
    v1_times = v1_meta["time_snapshot"].values.astype(str)

    matches = match_cutouts(
        v0["lats"], v0["lons"], v0["times"],
        v1_lats, v1_lons, v1_times,
        max_dist_deg=MAX_DIST_DEG,
    )

    print(f"  Found {len(matches)} matched pairs within {MAX_DIST_DEG}° "
          f"(~{MAX_DIST_DEG * 111:.0f} km)")
    if len(matches) == 0:
        print("\n  No matches found. Try increasing MAX_DIST_DEG or sample_points_per_snapshot.")
        sys.exit(1)

    n_plot = min(MAX_PAIRS, len(matches))
    print(f"  Plotting top {n_plot} closest pairs")
    print()

    # ── Generate plots ──────────────────────────────────────────
    print("=" * 70)
    print("STEP 4: Generate comparison plots")
    print("=" * 70)

    all_v0_sst = []
    all_v1_sst = []
    all_v0_divb2 = []
    all_v1_divb2 = []

    for pair_i, (v0_idx, v1_meta_idx, dist) in enumerate(matches[:n_plot]):
        # v1_meta_idx is a row in the metadata DataFrame.
        # We need _zarr_idx to get the correct image from the Zarr.
        v1_zarr_idx = int(v1_meta.iloc[v1_meta_idx]["_zarr_idx"])
        if v1_zarr_idx < 0:
            print(f"  Pair {pair_i}: SKIPPED — v1 metadata row {v1_meta_idx} "
                  f"has no matching Zarr image")
            continue

        v0_time = v0["times"][v0_idx]
        v1_time = str(v1_times[v1_meta_idx])
        print(f"  Pair {pair_i}: v0[{v0_idx}] ↔ v1[meta={v1_meta_idx}, zarr={v1_zarr_idx}]")
        print(f"    v0 time: {v0_time}, v1 time: {v1_time}")
        print(f"    v0 ({v0['lats'][v0_idx]:.3f}°, {v0['lons'][v0_idx]:.3f}°) ↔ "
              f"v1 ({v1_lats[v1_meta_idx]:.3f}°, {v1_lons[v1_meta_idx]:.3f}°), "
              f"dist={dist:.4f}° ({dist * 111:.1f} km)")

        # Extract cutout arrays
        v0_sst_map = v0_sst[v0_idx]           # (64, 64)
        v0_divb2_map = v0_log_divb2[v0_idx]    # (64, 64)

        # v1: load from Zarr using UUID-aligned index (NOT metadata row index)
        v1_patch = np.array(v1_images[v1_zarr_idx])  # (C, H, W)
        v1_sst_map = v1_patch[V1_THETA_IDX]     # (64, 64) — Theta in °C
        v1_divb2_map = v1_patch[V1_LOG_GRADB_IDX]  # (64, 64) — log10(|∇b|²)

        # Collect for aggregate
        all_v0_sst.append(v0_sst_map)
        all_v1_sst.append(v1_sst_map)
        all_v0_divb2.append(v0_divb2_map)
        all_v1_divb2.append(v1_divb2_map)

        # Plot SST pair
        fname = plot_pair(
            v0_sst_map, v1_sst_map,
            v0_sst_label, "v1 Theta (°C)",
            "SST (°C)",
            v0["lats"][v0_idx], v0["lons"][v0_idx],
            v1_lats[v1_meta_idx], v1_lons[v1_meta_idx],
            dist, pair_i, OUTPUT_DIR,
        )
        print(f"    SST plot: {fname}")

        # Plot Divb² / log_gradb pair
        if divb2_is_demeaned:
            divb2_field_name = "Divb² (de-meaned) vs log₁₀(|∇b|²)"
        else:
            divb2_field_name = "log₁₀(|∇b|²)"

        fname = plot_pair(
            v0_divb2_map, v1_divb2_map,
            v0_divb2_label, "v1 log₁₀(|∇b|²)",
            divb2_field_name,
            v0["lats"][v0_idx], v0["lons"][v0_idx],
            v1_lats[v1_meta_idx], v1_lons[v1_meta_idx],
            dist, pair_i, OUTPUT_DIR,
        )
        print(f"    Divb² plot: {fname}")

    # ── Aggregate histograms ────────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 5: Aggregate histograms")
    print("=" * 70)

    all_v0_sst = np.stack(all_v0_sst)
    all_v1_sst = np.stack(all_v1_sst)
    all_v0_divb2 = np.stack(all_v0_divb2)
    all_v1_divb2 = np.stack(all_v1_divb2)

    fname = plot_aggregate_histograms(
        all_v0_sst, all_v1_sst,
        v0_sst_label, "Theta (°C)",
        "SST (°C)", OUTPUT_DIR,
    )
    print(f"  SST aggregate: {fname}")

    fname = plot_aggregate_histograms(
        all_v0_divb2, all_v1_divb2,
        v0_divb2_label, "log₁₀(|∇b|²)",
        divb2_field_name, OUTPUT_DIR,
    )
    print(f"  Divb² aggregate: {fname}")

    # ── Summary table ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Matched pairs plotted: {n_plot}")
    print(f"  Total matches found:   {len(matches)}")
    print(f"  Max matching distance: {MAX_DIST_DEG}° ({MAX_DIST_DEG * 111:.0f} km)")
    print(f"  Plots saved to:        {OUTPUT_DIR}")
    print()

    # Print match distance distribution
    dists = np.array([m[2] for m in matches[:n_plot]])
    print(f"  Match distances (plotted pairs):")
    print(f"    mean: {dists.mean():.4f}°  ({dists.mean() * 111:.1f} km)")
    print(f"    min:  {dists.min():.4f}°  ({dists.min() * 111:.1f} km)")
    print(f"    max:  {dists.max():.4f}°  ({dists.max() * 111:.1f} km)")
    print()

    if divb2_is_demeaned:
        print("  ⚠  v0 Divb² was de-meaned by preprocessing — direct comparison")
        print("     with v1 log_gradb is qualitative only. For quantitative")
        print("     comparison, load raw v0 field files from:")
        print("     $DBOF_PATH/DBOF_dev/Fields/DBOF_dev_Divb2.h5")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
