"""
Test script for the S1 spur removal algorithm (MATLAB port).

Runs prune_short_spurs on the sharpened_global_g1.npy test data
and generates before/after comparison figures.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# add the parent directory so we can import matlab_port
sys.path.insert(0, os.path.dirname(__file__))
from matlab_port import prune_short_spurs, _detect_endpoints, _detect_branchpoints

# paths relative to dev/spurs/py
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = DATA_DIR  # figures go in the data directory per prompt instructions
TEST_FILE = os.path.join(os.path.dirname(__file__), '..', '..',
                         'sharpen', 'data', 'sharpened_global_g1.npy')


def load_test_data():
    """Load the test binary front image."""
    data = np.load(TEST_FILE)
    print(f"Loaded {TEST_FILE}")
    print(f"  Shape: {data.shape}, dtype: {data.dtype}, "
          f"front pixels: {data.sum()}")
    return data


def count_spurs(skeleton, Lspur):
    """Count endpoints that trace to a branchpoint within Lspur pixels."""
    from matlab_port import _trace_leaf_to_junction
    bp = _detect_branchpoints(skeleton)
    ep = _detect_endpoints(skeleton)
    ep_rows, ep_cols = np.where(ep)
    n_spurs = 0
    for r, c in zip(ep_rows, ep_cols):
        path = _trace_leaf_to_junction(skeleton, bp, r, c, Lspur)
        if path is not None:
            n_spurs += 1
    return n_spurs


def test_basic_spur():
    """Test on a small synthetic image with a known spur."""
    # create a small L-shaped front with a spur
    img = np.zeros((20, 20), dtype=bool)
    # main front: horizontal line
    img[10, 3:17] = True
    # spur: short vertical branch off the main front
    img[8:10, 10] = True  # 2-pixel spur going up
    # another spur at the end
    img[10, 17] = True
    img[11, 18] = True

    skeleton = skeletonize(img)
    print("\n=== Basic synthetic test ===")
    print(f"  Skeleton pixels before: {skeleton.sum()}")

    # count endpoints and branchpoints
    ep_before = _detect_endpoints(skeleton).sum()
    bp_before = _detect_branchpoints(skeleton).sum()
    print(f"  Endpoints: {ep_before}, Branchpoints: {bp_before}")

    result = prune_short_spurs(skeleton, Lspur=3)
    print(f"  Skeleton pixels after (Lspur=3): {result.sum()}")
    ep_after = _detect_endpoints(result).sum()
    bp_after = _detect_branchpoints(result).sum()
    print(f"  Endpoints: {ep_after}, Branchpoints: {bp_after}")

    # the short spurs should be gone, reducing branchpoints
    print(f"  Pixels removed: {skeleton.sum() - result.sum()}")
    return True


def test_isolated_segment():
    """Test that isolated short segments (no junction) are preserved."""
    img = np.zeros((20, 20), dtype=bool)
    # isolated short front (no branchpoint)
    img[5, 3:8] = True  # 5-pixel segment

    skeleton = skeletonize(img)
    result = prune_short_spurs(skeleton, Lspur=10)

    print("\n=== Isolated segment test ===")
    print(f"  Skeleton pixels before: {skeleton.sum()}")
    print(f"  Skeleton pixels after: {result.sum()}")
    preserved = result.sum() == skeleton.sum()
    print(f"  Isolated segment preserved: {preserved}")
    return preserved


def test_real_data():
    """Run on the real sharpened_global_g1 test data with multiple Lspur values."""
    data = load_test_data()
    skeleton = skeletonize(data)

    print(f"\n=== Real data test ===")
    print(f"  Skeleton pixels: {skeleton.sum()}")
    ep_count = _detect_endpoints(skeleton).sum()
    bp_count = _detect_branchpoints(skeleton).sum()
    print(f"  Endpoints: {ep_count}, Branchpoints: {bp_count}")

    results = {}
    for Lspur in [3, 5, 10, 15]:
        result = prune_short_spurs(data, Lspur=Lspur)
        removed = skeleton.sum() - result.sum()
        ep_after = _detect_endpoints(result).sum()
        bp_after = _detect_branchpoints(result).sum()
        print(f"  Lspur={Lspur:2d}: {result.sum()} pixels "
              f"(removed {removed}), "
              f"EP={ep_after}, BP={bp_after}")
        results[Lspur] = result

    return skeleton, results


def plot_before_after(skeleton, result, Lspur, suffix=''):
    """Generate a before/after comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # before
    axes[0].imshow(skeleton, cmap='gray', interpolation='nearest')
    axes[0].set_title('Before (skeleton)')
    axes[0].axis('off')

    # after
    axes[1].imshow(result, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'After (Lspur={Lspur})')
    axes[1].axis('off')

    # overlay: green = kept, red = removed
    removed = skeleton & ~result
    overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    overlay[result] = [255, 255, 255]   # kept pixels: white
    overlay[removed] = [255, 50, 50]    # removed pixels: red

    # mark endpoints in blue, branchpoints in yellow on the result
    ep = _detect_endpoints(result)
    bp = _detect_branchpoints(result)
    overlay[ep] = [50, 150, 255]    # endpoints: blue
    overlay[bp] = [255, 255, 0]     # branchpoints: yellow

    axes[2].imshow(overlay, interpolation='nearest')
    axes[2].set_title(f'Overlay (white=kept, red=removed,\n'
                      f'blue=endpoint, yellow=branchpoint)')
    axes[2].axis('off')

    plt.suptitle(f'S1 Spur Removal — Lspur={Lspur}', fontsize=14)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR, f'spur_removal_S1_L{Lspur}{suffix}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_zoomed(skeleton, result, Lspur, row_slice, col_slice, label=''):
    """Generate a zoomed-in comparison on a subregion."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sk_zoom = skeleton[row_slice, col_slice]
    res_zoom = result[row_slice, col_slice]
    removed = sk_zoom & ~res_zoom

    # before
    axes[0].imshow(sk_zoom, cmap='gray', interpolation='nearest')
    axes[0].set_title('Before')
    axes[0].axis('off')

    # overlay
    overlay = np.zeros((*sk_zoom.shape, 3), dtype=np.uint8)
    overlay[res_zoom] = [255, 255, 255]
    overlay[removed] = [255, 50, 50]
    ep = _detect_endpoints(res_zoom)
    bp = _detect_branchpoints(res_zoom)
    overlay[ep] = [50, 150, 255]
    overlay[bp] = [255, 255, 0]

    axes[1].imshow(overlay, interpolation='nearest')
    axes[1].set_title(f'After Lspur={Lspur}')
    axes[1].axis('off')

    plt.suptitle(f'S1 Zoomed: {label}', fontsize=14)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR,
                         f'spur_removal_S1_L{Lspur}_zoom_{label}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_lspur_comparison(skeleton, results):
    """Show the effect of different Lspur values side by side."""
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    axes[0].imshow(skeleton, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'Original\n({skeleton.sum()} px)')
    axes[0].axis('off')

    for i, (Lspur, result) in enumerate(sorted(results.items()), 1):
        removed = skeleton.sum() - result.sum()
        axes[i].imshow(result, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Lspur={Lspur}\n({result.sum()} px, -{removed})')
        axes[i].axis('off')

    plt.suptitle('S1 Spur Removal — Lspur Comparison', fontsize=14)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR, 'spur_removal_S1_comparison.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


if __name__ == '__main__':
    import time

    # run synthetic tests first
    test_basic_spur()
    test_isolated_segment()

    # run on real data
    skeleton, results = test_real_data()

    # generate figures
    print("\n=== Generating figures ===")

    # full-image before/after for each Lspur
    for Lspur, result in results.items():
        plot_before_after(skeleton, result, Lspur)

    # side-by-side Lspur comparison
    plot_lspur_comparison(skeleton, results)

    # zoomed views on interesting regions
    # find a region with branchpoints for a good zoom
    bp = _detect_branchpoints(skeleton)
    bp_rows, bp_cols = np.where(bp)
    if len(bp_rows) > 0:
        # pick a branchpoint near the center of the image
        center_r, center_c = skeleton.shape[0] // 2, skeleton.shape[1] // 2
        dists = (bp_rows - center_r)**2 + (bp_cols - center_c)**2
        idx = np.argmin(dists)
        r0, c0 = bp_rows[idx], bp_cols[idx]
        # 60x60 window around this branchpoint
        margin = 30
        rslice = slice(max(0, r0 - margin), min(skeleton.shape[0], r0 + margin))
        cslice = slice(max(0, c0 - margin), min(skeleton.shape[1], c0 + margin))
        for Lspur in [5, 10]:
            plot_zoomed(skeleton, results[Lspur], Lspur,
                        rslice, cslice, label='center_bp')

    # timing benchmark
    print("\n=== Timing benchmark ===")
    data = np.load(TEST_FILE)
    for Lspur in [5, 10]:
        t0 = time.time()
        _ = prune_short_spurs(data, Lspur=Lspur)
        elapsed = time.time() - t0
        print(f"  Lspur={Lspur}: {elapsed:.3f}s")

    print("\nAll tests complete.")
