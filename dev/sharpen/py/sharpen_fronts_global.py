"""
Approach G1: Global Priority-Queue Weighted Thinning

Sharpens front locations to lie on the gradb2 ridge by applying
priority-queue thinning to the entire binary threshold image at once,
without requiring per-front labeling.  Low-gradb2 boundary pixels are
removed first; the simple-point test guarantees topology at every step.

See claude_sharpen_global_plan.tex, Section "Approach G1" for the full
algorithm description.

Usage:
    from sharpen_fronts_global import global_sharpen_pq
    sharpened = global_sharpen_pq(binary_mask, gradb2)
"""

import sys
import time
import numpy as np
from pathlib import Path
from skimage import morphology 

from fronts.finding.sharpen import global_sharpen_pq

# Reuse the simple-point LUT and helpers from the per-front implementation
#from sharpen_fronts import (
#    SIMPLE_POINT_LUT,
#    NEIGHBORS_8,
#    _encode_neighborhood,
#    _is_boundary,
#    _count_fg_neighbors,
#)

# Code moved to fronts/finding/sharpen.py


# ---------------------------------------------------------------
# Standard thinning for comparison
# ---------------------------------------------------------------
def standard_thin(binary_mask, min_size=7):
    """Apply standard morphological thinning (geometric centerline).

    This is the baseline: symmetric thinning that ignores gradb2.

    Parameters
    ----------
    binary_mask : np.ndarray (bool, 2D)
        Binary threshold output.
    min_size : int
        Remove connected components smaller than this.

    Returns
    -------
    thinned : np.ndarray (bool, 2D)
    """

    thinned = morphology.thin(binary_mask)
    if min_size > 0:
        thinned = morphology.remove_small_objects(thinned, max_size=min_size,
                                        connectivity=2)
    return morphology.thin(thinned)


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------
def plot_threshold_and_sharpened(gradb2, binary_thresh, sharpened,
                                 standard_thinned=None,
                                 title_suffix='',
                                 save_dir=None, overleaf_dir=None):
    """Plot gradb2 with binary threshold, sharpened, and optionally
    standard-thinned fronts overlaid.

    Parameters
    ----------
    gradb2 : np.ndarray (float, 2D)
    binary_thresh : np.ndarray (bool, 2D)
        Raw binary threshold (wide fronts).
    sharpened : np.ndarray (bool, 2D)
        G1-sharpened 1-pixel-wide fronts.
    standard_thinned : np.ndarray (bool, 2D), optional
        Standard morphological thinning result for comparison.
    title_suffix : str
        Extra text appended to the plot title.
    save_dir : Path or None
    overleaf_dir : Path or None
    """
    import matplotlib.pyplot as plt

    # Log-scale gradb2 for display
    gradb2_log = np.where(gradb2 > 0, np.log10(gradb2), np.nan)

    n_panels = 2 if standard_thinned is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7),
                              sharex=True, sharey=True)
    if n_panels == 2:
        axes = list(axes)

    # --- Panel 1: binary threshold overlay ---
    ax = axes[0]
    ax.imshow(gradb2_log, cmap='Greys', origin='lower', aspect='equal',
              vmin=-15.)
    thresh_rows, thresh_cols = np.where(binary_thresh)
    ax.plot(thresh_cols, thresh_rows, 'c.', markersize=0.5, alpha=0.4)
    ax.set_title(f'Binary threshold\n({binary_thresh.sum()} pixels)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # --- Panel 2: G1 sharpened ---
    ax = axes[1]
    im = ax.imshow(gradb2_log, cmap='Greys', origin='lower', aspect='equal',
                    vmin=-15.)
    sharp_rows, sharp_cols = np.where(sharpened)
    ax.plot(sharp_cols, sharp_rows, 'r.', markersize=1.0, alpha=0.8)
    mean_g1 = gradb2[sharpened].mean() if sharpened.any() else 0.0
    ax.set_title(f'G1 sharpened\n({sharpened.sum()} px, '
                 f'mean gradb2={mean_g1:.2e})')
    ax.set_xlabel('Column')

    # --- Panel 3: standard thinning (if provided) ---
    if standard_thinned is not None:
        ax = axes[2]
        ax.imshow(gradb2_log, cmap='Greys', origin='lower', aspect='equal',
                  vmin=-15.)
        std_rows, std_cols = np.where(standard_thinned)
        ax.plot(std_cols, std_rows, 'g.', markersize=1.0, alpha=0.8)
        mean_std = gradb2[standard_thinned].mean() if standard_thinned.any() else 0.0
        ax.set_title(f'Standard thin\n({standard_thinned.sum()} px, '
                     f'mean gradb2={mean_std:.2e})')
        ax.set_xlabel('Column')

    fig.suptitle(f'Global Front Sharpening (G1){title_suffix}',
                 fontsize=13, y=1.02)
    fig.tight_layout()

    # Save
    fname = 'global_sharpen_g1.png'
    for d in [save_dir, overleaf_dir]:
        if d is not None:
            out = Path(d) / fname
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {out}")

    plt.close(fig)


def plot_overlay_comparison(gradb2, sharpened, standard_thinned,
                            save_dir=None, overleaf_dir=None):
    """Overlay G1-sharpened (red) and standard-thinned (green) fronts
    on the same gradb2 image for direct spatial comparison.

    Parameters
    ----------
    gradb2 : np.ndarray (float, 2D)
    sharpened : np.ndarray (bool, 2D)
    standard_thinned : np.ndarray (bool, 2D)
    save_dir : Path or None
    overleaf_dir : Path or None
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gradb2_log = np.where(gradb2 > 0, np.log10(gradb2), np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(gradb2_log, cmap='Greys', origin='lower', aspect='equal',
                    vmin=-15.)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r'$\log_{10}(|\nabla b|^2)$  [s$^{-4}$]')

    # Standard thinning in green (behind)
    std_rows, std_cols = np.where(standard_thinned)
    ax.plot(std_cols, std_rows, 'g.', markersize=2, alpha=0.7)

    # G1 sharpened in red (on top)
    sharp_rows, sharp_cols = np.where(sharpened)
    ax.plot(sharp_cols, sharp_rows, 'r.', markersize=2, alpha=0.7)

    # Stats
    mean_std = gradb2[standard_thinned].mean() if standard_thinned.any() else 0.0
    mean_g1 = gradb2[sharpened].mean() if sharpened.any() else 0.0
    ratio = mean_g1 / mean_std if mean_std > 0 else 0.0

    legend_elements = [
        Line2D([0], [0], marker='.', color='g', linestyle='None',
               markersize=6,
               label=f'Standard thin ({standard_thinned.sum()} px, '
                     f'mean={mean_std:.2e})'),
        Line2D([0], [0], marker='.', color='r', linestyle='None',
               markersize=6,
               label=f'G1 sharpened ({sharpened.sum()} px, '
                     f'mean={mean_g1:.2e})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              framealpha=0.9)

    ax.set_title(f'G1 vs Standard Thinning\n'
                 f'Mean gradb2 improvement: {ratio:.2f}x', fontsize=11)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    fig.tight_layout()

    fname = 'global_sharpen_g1_overlay.png'
    for d in [save_dir, overleaf_dir]:
        if d is not None:
            out = Path(d) / fname
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {out}")

    plt.close(fig)


# ---------------------------------------------------------------
# Main: run full pipeline on test data
# ---------------------------------------------------------------
if __name__ == '__main__':
    # Add the fronts package to the path so we can import front_thresh
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(repo_root))

    from fronts.finding.pyboa import front_thresh
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    overleaf_dir = Path('/home/xavier/Projects/overleaf/Front_properties')

    # Load test data
    gradb2 = np.load(data_dir / 'gradb2_global.npy')
    print(f'gradb2 shape: {gradb2.shape}')
    print(f'gradb2 range: [{gradb2.min():.4e}, {gradb2.max():.4e}]')

    # Step 1: Binary threshold (same parameters as the standard pipeline)
    print('\n--- Step 1: Thresholding ---')
    t0 = time.time()
    binary_thresh = front_thresh(gradb2, wndw=40, prcnt=85,
                                  mode='vectorized')
    t_thresh = time.time() - t0
    print(f'  Threshold pixels: {binary_thresh.sum()} / {binary_thresh.size} '
          f'({100 * binary_thresh.sum() / binary_thresh.size:.1f}%)')
    print(f'  Time: {t_thresh:.3f}s')

    # Step 2: Standard morphological thinning (baseline)
    print('\n--- Step 2: Standard thinning (baseline) ---')
    t0 = time.time()
    standard_thinned = standard_thin(binary_thresh, min_size=7)
    t_std = time.time() - t0
    print(f'  Standard thin pixels: {standard_thinned.sum()}')
    print(f'  Mean gradb2 on standard thin: '
          f'{gradb2[standard_thinned].mean():.4e}')
    print(f'  Time: {t_std:.3f}s')

    # Step 3: G1 global priority-queue sharpening
    print('\n--- Step 3: G1 global sharpening ---')
    t0 = time.time()
    sharpened = global_sharpen_pq(binary_thresh, gradb2,
                                   protect_endpoints=True)

    t_g1 = time.time() - t0
    mean_g1 = gradb2[sharpened].mean() if sharpened.any() else 0.0
    mean_std = gradb2[standard_thinned].mean() if standard_thinned.any() else 0.0
    print(f'  G1 sharpened pixels: {sharpened.sum()}')
    print(f'  Mean gradb2 on G1 sharpened: {mean_g1:.4e}')
    print(f'  Improvement over standard: {mean_g1 / mean_std:.2f}x')
    print(f'  Time: {t_g1:.3f}s')

    # Step 4: Generate plots
    print('\n--- Step 4: Generating plots ---')
    plot_threshold_and_sharpened(
        gradb2, binary_thresh, sharpened,
        standard_thinned=standard_thinned,
        save_dir=data_dir, overleaf_dir=overleaf_dir)

    plot_overlay_comparison(
        gradb2, sharpened, standard_thinned,
        save_dir=data_dir, overleaf_dir=overleaf_dir)

    # Save sharpened output
    out_path = data_dir / 'sharpened_global_g1.npy'
    np.save(out_path, sharpened)
    print(f'\nSaved G1 sharpened fronts to {out_path}')

    # Summary
    print('\n--- Summary ---')
    print(f'  {"Method":<20} {"Pixels":>8} {"Mean gradb2":>14} {"Time (s)":>10}')
    print(f'  {"-"*52}')
    print(f'  {"Threshold":<20} {binary_thresh.sum():>8} '
          f'{gradb2[binary_thresh].mean():>14.4e} {t_thresh:>10.3f}')
    print(f'  {"Standard thin":<20} {standard_thinned.sum():>8} '
          f'{mean_std:>14.4e} {t_std:>10.3f}')
    print(f'  {"G1 sharpened":<20} {sharpened.sum():>8} '
          f'{mean_g1:>14.4e} {t_g1:>10.3f}')
