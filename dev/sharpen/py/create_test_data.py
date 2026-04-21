"""
Create test data for front sharpening development.

Extracts subregions of the gradb2 and labeled_fronts fields from the
LLC4320 global snapshot and saves them as .npy files.

Sources:
  gradb2: $OS_OGCM/LLC/Fronts/derived/LLC4320_2012-11-09T12_00_00_gradb2_v1.nc
  labeled_fronts: $OS_OGCM/LLC/Fronts/group_fronts/v1/labeled_fronts_global_20121109T12_00_00_v1_bin_B.npy

Region: rows=8611:8750, cols=11841:11980  (140x140 pixels)
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ogcm = os.environ["OS_OGCM"]

# Shared cutout slices (rows = y, cols = x)
ROW_SLICE = slice(8610, 8750)
COL_SLICE = slice(11840, 11980)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def create_test_gradb2_data(row_slice=None, col_slice=None,
    out_root:str='gradb2.npy'):
    """Extract and save the gradb2 subregion."""
    src_file = os.path.join(
        ogcm, "LLC", "Fronts", "derived",
        "LLC4320_2012-11-09T12_00_00_gradb2_v1.nc"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Slice
    if row_slice is None:
        row_slice = ROW_SLICE
    if col_slice is None:
        col_slice = COL_SLICE

    ds = xr.open_dataset(src_file)
    gradb2 = ds["gradb2"].isel(y=ROW_SLICE, x=COL_SLICE).values  # float32

    print(f"gradb2 shape: {gradb2.shape}")
    print(f"gradb2 range: [{np.nanmin(gradb2):.4e}, {np.nanmax(gradb2):.4e}]")
    print(f"NaN fraction: {np.isnan(gradb2).mean():.3f}")

    out_path = DATA_DIR / out_root
    np.save(out_path, gradb2)
    print(f"Saved to {out_path}")


def create_test_labeled_fronts():
    """Extract and save the labeled_fronts subregion.

    Labels are re-numbered to be contiguous (1..N) within the cutout,
    since the global labels are sparse.
    """
    src_file = os.path.join(
        ogcm, "LLC", "Fronts", "group_fronts", "v1",
        "labeled_fronts_global_20121109T12_00_00_v1_bin_B.npy"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load global labeled fronts and extract subregion
    lf_global = np.load(src_file)
    lf = lf_global[ROW_SLICE, COL_SLICE].copy()

    # Re-label to contiguous integers (0 = background, 1..N = fronts)
    unique_labels = np.unique(lf)
    unique_labels = unique_labels[unique_labels > 0]  # exclude background
    remap = np.zeros(lf.max() + 1, dtype=np.int32)
    for new_label, old_label in enumerate(unique_labels, start=1):
        remap[old_label] = new_label
    lf = remap[lf]

    n_fronts = len(unique_labels)
    n_pixels = (lf > 0).sum()
    print(f"labeled_fronts shape: {lf.shape}")
    print(f"Number of fronts in cutout: {n_fronts}")
    print(f"Total front pixels: {n_pixels}")

    out_path = DATA_DIR / "labeled_fronts.npy"
    np.save(out_path, lf)
    print(f"Saved to {out_path}")


def plot_labeled_fronts_on_gradb2():
    """Plot labeled fronts overlaid on the gradb2 field.

    Saves PNG to the Overleaf project folder and also to the data dir.
    """
    gradb2 = np.load(DATA_DIR / "gradb2.npy")
    lf = np.load(DATA_DIR / "labeled_fronts.npy")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # gradb2 background (log scale for dynamic range)
    gradb2_plot = np.where(gradb2 > 0, gradb2, np.nan)
    im = ax.imshow(
        np.log10(gradb2_plot),
        cmap="Greys",
        origin="lower",
        aspect="equal",
        vmin=-15.,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\log_{10}(|\nabla b|^2)$  [s$^{-4}$]")

    # Overlay labeled fronts with distinct colors per label
    n_labels = lf.max()
    if n_labels > 0:
        # Create a random colormap for front labels
        rng = np.random.default_rng(42)
        colors = rng.random((n_labels, 3))
        colors = np.vstack([[0, 0, 0], colors])  # label 0 = transparent
        cmap_fronts = mcolors.ListedColormap(colors)

        # Mask background (label 0) as transparent
        front_overlay = np.ma.masked_where(lf == 0, lf)
        ax.imshow(
            front_overlay,
            cmap=cmap_fronts,
            origin="lower",
            aspect="equal",
            alpha=0.7,
            interpolation="nearest",
        )

    ax.set_title(
        f"Labeled fronts on gradb2\n"
        f"rows {ROW_SLICE.start}:{ROW_SLICE.stop}, "
        f"cols {COL_SLICE.start}:{COL_SLICE.stop}  "
        f"({n_labels} fronts)",
        fontsize=11,
    )
    ax.set_xlabel("Column (local)")
    ax.set_ylabel("Row (local)")

    # Save to Overleaf and data dir
    overleaf_dir = Path("/home/xavier/Projects/overleaf/Front_properties")
    for out_dir in [DATA_DIR, overleaf_dir]:
        out_path = out_dir / "labeled_fronts_on_gradb2.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    #create_test_gradb2_data()
    #create_test_labeled_fronts()
    #plot_labeled_fronts_on_gradb2()

    # Global
    ROW_SLICE = slice(8410, 8850)
    COL_SLICE = slice(11640, 12080)
    create_test_gradb2_data(ROW_SLICE, COL_SLICE, out_root='gradb2_global.npy')
