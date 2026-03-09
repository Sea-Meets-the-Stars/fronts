#!/usr/bin/env python
"""
01_load_v0_data.py

Loads existing v0 DBOF training data, discovers available timestamps,
and selects the v0 timestamps closest to the existing v1 run
(year_4x150_20260203_201716), then exports a clean subset for
comparison with v1.

v1 timestamps are read DIRECTLY from the v1 metadata parquet on S3,
rather than computed from config records (which use a non-trivial
iteration-based mapping that doesn't simply equal hours).

Usage:
    python 01_load_v0_data.py

Outputs:
    v0_subset.npz   — cutout arrays + metadata for the matched timestamps
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import fsspec
from datetime import datetime, timedelta

# ── Configuration ───────────────────────────────────────────────────
# Path to v0 Training directory (Jake test set)
TEST_PATH = os.environ.get(
    "TEST_PATH",
    "/mnt/tank/Oceanography/data/OGCM/DBOF/DBOF_dev/Training/"
)
DBOF_DEV_PATH = os.path.dirname(TEST_PATH.rstrip("/"))  # .../DBOF_dev/
SAVE_PATH = os.environ.get(
    "SAVE_PATH",
    "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/dbof_testing/version_comparison/"
)

# Files
META_FILE = os.path.join(TEST_PATH, "Jake_test_set_meta.parquet")
TRAIN_H5  = os.path.join(TEST_PATH, "Jake_test_set_train.h5")
VALID_H5  = os.path.join(TEST_PATH, "Jake_test_set_valid.h5")
TEST_H5   = os.path.join(TEST_PATH, "Jake_test_set_test.h5")

# v0 field channel ordering (from DBOF_train_config_jake_test.json)
INPUT_FIELDS  = ["SSS", "SSSs", "SSTK", "SSH"]  # channels 0,1,2,3
TARGET_FIELDS = ["Divb2"]                         # channel 0
SSTK_INPUT_IDX = 2   # SSTK is the 3rd input channel
DIVB2_TARGET_IDX = 0  # Divb2 is the only target channel

# v1 S3 access parameters
V1_S3_ENDPOINT = os.environ.get("V1_S3_ENDPOINT", "https://s3-west.nrp-nautilus.io")
V1_BUCKET  = os.environ.get("V1_BUCKET",  "dbof")
V1_FOLDER  = os.environ.get("V1_FOLDER",  "native_grid_dbof_training_data")
V1_RUN_ID  = os.environ.get("V1_RUN_ID",  "year_4x150_20260203_201716")

# How many v0↔v1 timestamp pairs to use
N_PAIRS = 2

# Max acceptable time gap (days) between v0 and v1 snapshots
MAX_TIME_GAP_DAYS = 5

# Output
OUTPUT_FILE = os.path.join(SAVE_PATH, "v0_subset.npz")

# ── Helpers ─────────────────────────────────────────────────────────

def create_s3_filesystem(s3_endpoint):
    """Create synchronous S3 filesystem matching the v1 notebook protocol."""
    return fsspec.filesystem(
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


def load_v1_timestamps(s3_endpoint, bucket, folder, run_id):
    """
    Load the ACTUAL v1 timestamps from the metadata parquet on S3.
    This avoids any record-to-date arithmetic errors.
    """
    fs = create_s3_filesystem(s3_endpoint)
    meta_glob = f"{bucket}/{folder}/{run_id}/metadata/*.parquet"
    print(f"  Loading v1 metadata from S3: {meta_glob}")
    files = fs.glob(meta_glob)
    if not files:
        print(f"  ERROR: No metadata files found at {meta_glob}")
        sys.exit(1)
    print(f"  Found {len(files)} parquet files")
    v1_meta = pd.read_parquet(files, filesystem=fs)
    print(f"  Loaded {len(v1_meta)} rows")

    # Extract unique timestamps
    v1_times = pd.to_datetime(v1_meta["time_snapshot"])
    unique_times = sorted(v1_times.unique())
    print(f"  Unique v1 timestamps ({len(unique_times)}):")
    for t in unique_times:
        n = (v1_times == t).sum()
        print(f"    {pd.Timestamp(t).strftime('%Y-%m-%d %H:%M')}  ({n} cutouts)")
    return [pd.Timestamp(t) for t in unique_times]


# ── Main ────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Load metadata ───────────────────────────────────
    print("=" * 70)
    print("STEP 1: Load v0 metadata")
    print("=" * 70)

    if not os.path.exists(META_FILE):
        print(f"ERROR: Metadata file not found: {META_FILE}")
        print(f"Check that TEST_PATH is correct: {TEST_PATH}")
        sys.exit(1)

    meta = pd.read_parquet(META_FILE)
    print(f"Loaded {len(meta)} rows from {META_FILE}")
    print(f"Columns: {list(meta.columns)}")
    print()

    # ── Step 2: Load ACTUAL v1 timestamps from S3 metadata ─────
    print("=" * 70)
    print("STEP 2: Match v0 timestamps to existing v1 run")
    print("=" * 70)

    print(f"\nLoading actual v1 timestamps from S3 metadata:")
    V1_DATES = load_v1_timestamps(V1_S3_ENDPOINT, V1_BUCKET, V1_FOLDER, V1_RUN_ID)

    v0_timestamps = sorted(meta["datetime"].unique())
    v0_dates = [pd.Timestamp(ts) for ts in v0_timestamps]

    print(f"\nv0 timestamps ({len(v0_timestamps)}):")
    for i, ts in enumerate(v0_dates):
        n_cutouts = (meta["datetime"] == v0_timestamps[i]).sum()
        print(f"  [{i}] {ts.strftime('%Y-%m-%d %H:%M')}  ({n_cutouts:5d} cutouts)")

    # Find closest v0 timestamp for each v1 snapshot
    print(f"\nTemporal matching (v1 → closest v0):")
    pairs = []  # (v1_date, v0_timestamp, gap_days)
    for v1_dt in V1_DATES:
        v1_ts = pd.Timestamp(v1_dt)
        gaps = [abs((v0_ts - v1_ts).total_seconds()) / 86400 for v0_ts in v0_dates]
        best_idx = int(np.argmin(gaps))
        gap_days = gaps[best_idx]
        pairs.append((v1_dt, v0_timestamps[best_idx], gap_days, best_idx))
        status = "✓" if gap_days <= MAX_TIME_GAP_DAYS else "✗ too far"
        print(f"  v1 {v1_dt.strftime('%Y-%m-%d %H:%M')} ↔ v0 [{best_idx}] "
              f"{v0_dates[best_idx].strftime('%Y-%m-%d')}  "
              f"(gap: {gap_days:.1f} days) {status}")

    # Select the N_PAIRS best matches within tolerance
    valid_pairs = [(v1_dt, v0_ts, gap, v0_idx)
                   for v1_dt, v0_ts, gap, v0_idx in pairs
                   if gap <= MAX_TIME_GAP_DAYS]
    valid_pairs.sort(key=lambda x: x[2])  # sort by gap (closest first)

    if len(valid_pairs) < N_PAIRS:
        print(f"\n  WARNING: Only {len(valid_pairs)} pairs within {MAX_TIME_GAP_DAYS}-day "
              f"tolerance (wanted {N_PAIRS}).")
        if len(valid_pairs) == 0:
            print(f"  Loosening tolerance to {MAX_TIME_GAP_DAYS * 5} days...")
            valid_pairs = [(v1_dt, v0_ts, gap, v0_idx)
                           for v1_dt, v0_ts, gap, v0_idx in pairs]
            valid_pairs.sort(key=lambda x: x[2])

    chosen_pairs = valid_pairs[:N_PAIRS]
    chosen_v0_timestamps = [p[1] for p in chosen_pairs]

    print(f"\nSelected {len(chosen_pairs)} timestamp pairs for comparison:")
    for v1_dt, v0_ts, gap, v0_idx in chosen_pairs:
        v0_dt = pd.Timestamp(v0_ts)
        print(f"  v0 {v0_dt.strftime('%Y-%m-%d %H:%M')} ↔ "
              f"v1 {v1_dt.strftime('%Y-%m-%d %H:%M')}  "
              f"(gap: {gap:.1f} days)")
    print()

    # ── Step 3: Extract cutouts ─────────────────────────────────
    print("=" * 70)
    print("STEP 3: Extract v0 cutouts at matched timestamps")
    print("=" * 70)

    all_sst = []
    all_divb2 = []
    all_lats = []
    all_lons = []
    all_datetimes = []
    all_uids = []

    # Use pp_type column for safe split identification instead of
    # positional slicing (which is unreliable due to pandas .loc[]
    # preserving original row order during train/valid/test split).
    # pp_type values: 0=valid, 1=train, 2=test
    PP_TYPE_MAP = {"train": 1, "valid": 0, "test": 2}

    has_pp_type = "pp_type" in meta.columns
    if has_pp_type:
        print("  Using pp_type column for safe split identification ✓")
    else:
        print("  WARNING: No pp_type column found — falling back to positional slicing.")
        print("           This may cause row misalignment!")
        # Pre-count HDF5 rows for positional fallback
        split_sizes = {}
        for sn, hp in [("train", TRAIN_H5), ("valid", VALID_H5), ("test", TEST_H5)]:
            if os.path.exists(hp):
                with h5py.File(hp, "r") as f:
                    split_sizes[sn] = f["inputs"].shape[0]
            else:
                split_sizes[sn] = 0

    for split_name, h5_path in [
        ("train", TRAIN_H5),
        ("valid", VALID_H5),
        ("test",  TEST_H5),
    ]:
        if not os.path.exists(h5_path):
            print(f"  {split_name}: file not found, skipping")
            continue

        with h5py.File(h5_path, "r") as f:
            n_in_file = f["inputs"].shape[0]
            inp_shape = f["inputs"].shape
            tgt_shape = f["targets"].shape
            print(f"  {split_name}: {h5_path}")
            print(f"    inputs shape:  {inp_shape}")
            print(f"    targets shape: {tgt_shape}")

        # Get the metadata rows for this split
        if has_pp_type:
            pp_val = PP_TYPE_MAP[split_name]
            split_meta = meta[meta["pp_type"] == pp_val].copy().reset_index(drop=True)
            if len(split_meta) != n_in_file:
                print(f"    WARNING: pp_type={pp_val} has {len(split_meta)} rows "
                      f"but HDF5 has {n_in_file} — possible mismatch!")
            else:
                print(f"    pp_type={pp_val} rows ({len(split_meta)}) matches HDF5 ({n_in_file}) ✓")
        else:
            # Fallback: positional slicing (unsafe but best we can do)
            if split_name == "train":
                offset = 0
            elif split_name == "valid":
                offset = split_sizes.get("train", 0)
            elif split_name == "test":
                offset = split_sizes.get("train", 0) + split_sizes.get("valid", 0)
            split_meta = meta.iloc[offset:offset + n_in_file].copy().reset_index(drop=True)

        # Filter to chosen v0 timestamps
        mask = split_meta["datetime"].isin(chosen_v0_timestamps)
        if mask.sum() == 0:
            print(f"    No cutouts at matched timestamps in {split_name} split")
            continue

        indices = np.where(mask)[0]
        sub_meta = split_meta.loc[mask]
        print(f"    {len(indices)} cutouts at matched timestamps")

        # Load the cutout data
        with h5py.File(h5_path, "r") as f:
            inp = f["inputs"][:]
            tgt = f["targets"][:]
            if inp.ndim == 5:
                inp = inp[:, :, 0, :, :]  # squeeze depth dim
            if tgt.ndim == 5:
                tgt = tgt[:, :, 0, :, :]

            sst_data = inp[indices, SSTK_INPUT_IDX, :, :]     # (M, 64, 64)
            divb2_data = tgt[indices, DIVB2_TARGET_IDX, :, :]  # (M, 64, 64)

        all_sst.append(sst_data)
        all_divb2.append(divb2_data)
        all_lats.append(sub_meta["lat"].values)
        all_lons.append(sub_meta["lon"].values)
        all_datetimes.append(sub_meta["datetime"].values)
        all_uids.append(sub_meta["UID"].values if "UID" in sub_meta.columns else
                        np.arange(len(sub_meta)))

    if not all_sst:
        print("\nERROR: No cutouts found at the matched timestamps.")
        sys.exit(1)

    # Concatenate across splits
    v0_sst     = np.concatenate(all_sst, axis=0)
    v0_divb2   = np.concatenate(all_divb2, axis=0)
    v0_lats    = np.concatenate(all_lats, axis=0)
    v0_lons    = np.concatenate(all_lons, axis=0)
    v0_times   = np.concatenate(all_datetimes, axis=0)
    v0_uids    = np.concatenate(all_uids, axis=0)

    print(f"\n  Total v0 cutouts extracted: {len(v0_sst)}")
    print(f"  SST shape:   {v0_sst.shape}")
    print(f"  Divb2 shape: {v0_divb2.shape}")

    finite_sst = v0_sst[np.isfinite(v0_sst)]
    finite_divb2 = v0_divb2[np.isfinite(v0_divb2)]
    if len(finite_sst) > 0:
        print(f"  SST range:   [{finite_sst.min():.4f}, {finite_sst.max():.4f}]")
    if len(finite_divb2) > 0:
        print(f"  Divb2 range: [{finite_divb2.min():.4e}, {finite_divb2.max():.4e}]")
    print()

    # ── Unit checks ─────────────────────────────────────────────
    sst_median = np.nanmedian(v0_sst)
    if sst_median > 200:
        print(f"  NOTE: SST median={sst_median:.1f} — appears to be in Kelvin")
        print(f"        v1 Theta is in °C, so subtract 273.15 when comparing")
    elif abs(sst_median) < 5:
        print(f"  NOTE: SST median={sst_median:.2f} — appears de-meaned")
    else:
        print(f"  NOTE: SST median={sst_median:.1f} — appears to be in °C")

    divb2_mean = np.nanmean(v0_divb2)
    divb2_min = np.nanmin(v0_divb2)
    if divb2_min < 0:
        print(f"  NOTE: Divb2 has negative values (min={divb2_min:.2e}) — "
              f"appears de-meaned by v0 preprocessing")
    else:
        print(f"  NOTE: Divb2 mean={divb2_mean:.2e}, all non-negative — raw values")
    print()

    # ── Save ────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 4: Save v0 subset")
    print("=" * 70)

    # Build a mapping of v0 timestamps → closest v1 timestamps
    # (needed by script 02 for temporal matching)
    v0_to_v1_time_map = {}
    for v1_dt, v0_ts, gap, v0_idx in chosen_pairs:
        v0_to_v1_time_map[str(v0_ts)] = v1_dt.strftime('%Y-%m-%dT%H:%M:%S')

    np.savez(OUTPUT_FILE,
             sst=v0_sst,
             divb2=v0_divb2,
             lats=v0_lats,
             lons=v0_lons,
             times=v0_times.astype(str),
             uids=v0_uids.astype(str),
             # v1 matching info
             v1_run_id=V1_RUN_ID,
             chosen_v0_timestamps=np.array([str(t) for t in chosen_v0_timestamps]),
             chosen_v1_dates=np.array([dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in
                                       [p[0] for p in chosen_pairs]]),
             time_gaps_days=np.array([p[2] for p in chosen_pairs]),
             sst_channel_name="SSTK",
             divb2_channel_name="Divb2")

    print(f"Saved {len(v0_sst)} cutouts to: {OUTPUT_FILE}")
    print()

    # ── Summary ─────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  v1 run: {V1_RUN_ID}")
    print(f"  v1 timestamps loaded from: S3 metadata (not computed from config)")
    print()
    for v1_dt, v0_ts, gap, v0_idx in chosen_pairs:
        v0_dt = pd.Timestamp(v0_ts)
        n = (v0_times.astype(str) == str(v0_ts)).sum()
        print(f"  v0 {v0_dt.strftime('%Y-%m-%d %H:%M')} ({n:5d} cutouts)")
        print(f"  v1 {v1_dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"  gap: {gap:.1f} days")
        print()
    print("Next steps:")
    print("  Run: python 02_compare_v0_v1.py")


if __name__ == "__main__":
    main()
