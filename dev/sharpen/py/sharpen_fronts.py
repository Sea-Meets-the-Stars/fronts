"""
Approach 4: Intensity-Weighted Topological Thinning

Sharpens front locations to lie on the gradb2 ridge by performing
priority-queue thinning that removes low-gradb2 boundary pixels first,
with a simple-point test to guarantee contiguity at every step.

See claude_sharpen_planning.tex, Section "Approach 4" for the full
algorithm description.

Usage:
    from fronts.finding.sharpen import sharpen_fronts
    sharpened = sharpen_fronts(labeled_fronts, gradb2)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from fronts.finding.sharpen import sharpen_fronts

# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------
def plot_original_vs_sharpened(gradb2, labeled_fronts, sharpened,
                               dilate_radius=0,
                               save_dir=None, overleaf_dir=None):
    """Plot original (dotted) and sharpened (solid) fronts overlaid on gradb2.

    Parameters
    ----------
    gradb2 : np.ndarray (float, 2D)
    labeled_fronts : np.ndarray (int, 2D)
        Original labeled fronts.
    sharpened : np.ndarray (int, 2D)
        Sharpened labeled fronts.
    dilate_radius : int
        Dilation radius used (for annotation).
    save_dir : Path or None
        Directory to save the plot (data dir).
    overleaf_dir : Path or None
        Overleaf project directory for a copy.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Log-scale gradb2 for display
    gradb2_log = np.where(gradb2 > 0, np.log10(gradb2), np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # gradb2 background
    im = ax.imshow(gradb2_log, cmap="Greys", origin="lower",
                    aspect="equal", vmin=-15.)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\log_{10}(|\nabla b|^2)$  [s$^{-4}$]")

    # Original front pixels as dotted markers (cyan)
    orig_mask = labeled_fronts > 0
    orig_rows, orig_cols = np.where(orig_mask)
    ax.plot(orig_cols, orig_rows, 'c.', markersize=2, alpha=0.8)

    # Sharpened front pixels as solid markers (red)
    sharp_mask = sharpened > 0
    sharp_rows, sharp_cols = np.where(sharp_mask)
    ax.plot(sharp_cols, sharp_rows, 'r.', markersize=2, alpha=0.8)

    # Stats for annotation
    n_orig = orig_mask.sum()
    n_sharp = sharp_mask.sum()
    n_fronts_orig = len(np.unique(labeled_fronts)) - 1
    n_fronts_sharp = len(np.unique(sharpened)) - 1
    mean_orig = gradb2[orig_mask].mean() if orig_mask.any() else 0.0
    mean_sharp = gradb2[sharp_mask].mean() if sharp_mask.any() else 0.0
    ratio = mean_sharp / mean_orig if mean_orig > 0 else 0.0

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='.', color='c', linestyle='None',
               markersize=6, label=f'Original ({n_fronts_orig} fronts, '
               f'{n_orig} px, mean={mean_orig:.2e})'),
        Line2D([0], [0], marker='.', color='r', linestyle='None',
               markersize=6, label=f'Sharpened ({n_fronts_sharp} fronts, '
               f'{n_sharp} px, mean={mean_sharp:.2e})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              framealpha=0.9)

    dil_str = f", dilate={dilate_radius}px" if dilate_radius > 0 else ""
    ax.set_title(f"Approach 4: Intensity-Weighted Thinning{dil_str}\n"
                 f"Mean gradb2 improvement: {ratio:.2f}x", fontsize=11)
    ax.set_xlabel("Column (local)")
    ax.set_ylabel("Row (local)")

    fig.tight_layout()

    # Save
    suffix = f"_dilate{dilate_radius}" if dilate_radius > 0 else ""
    fname = f"original_vs_sharpened{suffix}.png"
    for d in [save_dir, overleaf_dir]:
        if d is not None:
            out = Path(d) / fname
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {out}")

    plt.close(fig)


# ---------------------------------------------------------------
# Main: test with dev data
# ---------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    overleaf_dir = Path("/home/xavier/Projects/overleaf/Front_properties")

    # Load test data
    gradb2 = np.load(data_dir / "gradb2.npy")
    labeled_fronts = np.load(data_dir / "labeled_fronts.npy")

    print(f"gradb2 shape: {gradb2.shape}")
    print(f"labeled_fronts shape: {labeled_fronts.shape}")
    print(f"Number of fronts: {labeled_fronts.max()}")
    print(f"Total front pixels (before): {(labeled_fronts > 0).sum()}")

    # Sharpen with dilation: expand each front by 3 pixels before
    # thinning, giving the algorithm room to shift to the ridge
    dilate_radius = 3
    print(f"\nSharpening with dilate_radius={dilate_radius}...")
    sharpened = sharpen_fronts(labeled_fronts, gradb2,
                               protect_endpoints=True,
                               min_size=7, dilate_radius=dilate_radius,
                               n_workers=1)

    n_sharp = (sharpened > 0).sum()
    n_labels_sharp = len(np.unique(sharpened)) - 1  # exclude 0
    print(f"Total front pixels (after): {n_sharp}")
    print(f"Number of fronts (after): {n_labels_sharp}")

    # Compare mean gradb2 on original vs sharpened front pixels
    orig_pixels = labeled_fronts > 0
    sharp_pixels = sharpened > 0
    mean_orig = gradb2[orig_pixels].mean()
    mean_sharp = gradb2[sharp_pixels].mean() if sharp_pixels.any() else 0.0
    print(f"Mean gradb2 on original fronts:  {mean_orig:.4e}")
    print(f"Mean gradb2 on sharpened fronts: {mean_sharp:.4e}")
    print(f"Improvement ratio: {mean_sharp / mean_orig:.2f}x")

    # Save sharpened output
    out_path = data_dir / "sharpened_labeled_fronts.npy"
    np.save(out_path, sharpened)
    print(f"Saved sharpened fronts to {out_path}")

    # Plot comparison: original (cyan dots) vs sharpened (red dots)
    plot_original_vs_sharpened(gradb2, labeled_fronts, sharpened,
                               dilate_radius=dilate_radius,
                               save_dir=data_dir,
                               overleaf_dir=overleaf_dir)
