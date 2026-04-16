"""
Test script for the S5 spur removal algorithm (skan-based).

Runs prune_short_spurs on the sharpened_global_g1.npy test data,
generates before/after comparison figures, and compares performance
and results against the S1 MATLAB port approach.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import correlate

from fronts.finding.despur import prune_short_spurs as skan_prune

# add the parent directory so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))
from spurs_skan import analyze_branches

from matlab_port import prune_short_spurs as matlab_prune
from matlab_port import _detect_endpoints, _detect_branchpoints

# paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = DATA_DIR
TEST_FILE = os.path.join(os.path.dirname(__file__), '..', '..',
                         'sharpen', 'data', 'sharpened_global_g1.npy')

# neighbor kernel for endpoint/branchpoint detection
_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def _count_ep_bp(skeleton):
    """Return (endpoint_count, branchpoint_count) for a skeleton."""
    ncount = correlate(skeleton.astype(np.uint8), _KERNEL,
                       mode='constant', cval=0)
    ep = np.sum(skeleton & (ncount == 1))
    bp = np.sum(skeleton & (ncount >= 3))
    return int(ep), int(bp)


def load_test_data():
    """Load the test binary front image."""
    data = np.load(TEST_FILE)
    print(f"Loaded {TEST_FILE}")
    print(f"  Shape: {data.shape}, dtype: {data.dtype}, "
          f"front pixels: {data.sum()}")
    return data


def test_basic_spur():
    """Test on a small synthetic image with a known spur."""
    img = np.zeros((20, 20), dtype=bool)
    # main front: horizontal line
    img[10, 3:17] = True
    # spur: short vertical branch
    img[8:10, 10] = True
    # another spur at the end
    img[10, 17] = True
    img[11, 18] = True

    skeleton = skeletonize(img)
    print("\n=== S5 Basic synthetic test ===")
    print(f"  Skeleton pixels before: {skeleton.sum()}")
    ep, bp = _count_ep_bp(skeleton)
    print(f"  Endpoints: {ep}, Branchpoints: {bp}")

    result = skan_prune(skeleton, Lspur=3)
    print(f"  Skeleton pixels after (Lspur=3): {result.sum()}")
    ep, bp = _count_ep_bp(result)
    print(f"  Endpoints: {ep}, Branchpoints: {bp}")
    print(f"  Pixels removed: {skeleton.sum() - result.sum()}")
    return True


def test_isolated_segment():
    """Test that isolated short segments (no junction) are preserved."""
    img = np.zeros((20, 20), dtype=bool)
    # isolated short front (no branchpoint)
    img[5, 3:8] = True  # 5-pixel segment

    skeleton = skeletonize(img)
    result = skan_prune(skeleton, Lspur=10)

    print("\n=== S5 Isolated segment test ===")
    print(f"  Skeleton pixels before: {skeleton.sum()}")
    print(f"  Skeleton pixels after: {result.sum()}")
    preserved = result.sum() == skeleton.sum()
    print(f"  Isolated segment preserved: {preserved}")
    return preserved


def test_branch_analysis():
    """Inspect the branch structure of the real test data."""
    data = load_test_data()
    df, sk = analyze_branches(data)

    print("\n=== Branch analysis of test data ===")
    print(f"  Total branches: {len(df)}")
    for bt in sorted(df['branch-type'].unique()):
        subset = df[df['branch-type'] == bt]
        labels = {0: 'ep-to-ep', 1: 'junc-to-ep (spur candidates)',
                  2: 'junc-to-junc', 3: 'isolated cycle'}
        label = labels.get(int(bt), f'type-{int(bt)}')
        print(f"  Type {int(bt)} ({label}): {len(subset)} branches, "
              f"mean dist={subset['branch-distance'].mean():.1f}, "
              f"median={subset['branch-distance'].median():.1f}, "
              f"max={subset['branch-distance'].max():.1f}")

    # distribution of spur candidate lengths
    spurs = df[df['branch-type'] == 1]
    print(f"\n  Spur candidate length distribution:")
    for threshold in [3, 5, 10, 15, 20]:
        count = (spurs['branch-distance'] <= threshold).sum()
        print(f"    <= {threshold:2d} px: {count} branches")

    return df


def test_real_data():
    """Run S5 on real data with multiple Lspur values."""
    data = load_test_data()
    skeleton = skeletonize(data)

    print(f"\n=== S5 Real data test ===")
    print(f"  Skeleton pixels: {skeleton.sum()}")
    ep, bp = _count_ep_bp(skeleton)
    print(f"  Endpoints: {ep}, Branchpoints: {bp}")

    results = {}
    timings = {}
    for Lspur in [3, 5, 10, 15]:
        t0 = time.time()
        result = skan_prune(data, Lspur=Lspur)
        elapsed = time.time() - t0
        removed = skeleton.sum() - result.sum()
        ep_a, bp_a = _count_ep_bp(result)
        print(f"  Lspur={Lspur:2d}: {result.sum()} pixels "
              f"(removed {removed}), "
              f"EP={ep_a}, BP={bp_a}, time={elapsed:.3f}s")
        results[Lspur] = result
        timings[Lspur] = elapsed

    return skeleton, results, timings


def compare_s1_s5(data, skeleton):
    """Compare S1 and S5 results side-by-side."""
    print("\n=== S1 vs S5 Comparison ===")
    print(f"{'Lspur':>5} | {'S1 px':>7} {'S1 rm':>7} {'S1 t(s)':>8} | "
          f"{'S5 px':>7} {'S5 rm':>7} {'S5 t(s)':>8} | "
          f"{'agree%':>7} {'S1only':>7} {'S5only':>7}")
    print("-" * 95)

    s1_results = {}
    s5_results = {}
    s1_timings = {}
    s5_timings = {}

    for Lspur in [3, 5, 10, 15]:
        # S1
        t0 = time.time()
        s1 = matlab_prune(data, Lspur=Lspur)
        s1_t = time.time() - t0

        # S5
        t0 = time.time()
        s5 = skan_prune(data, Lspur=Lspur)
        s5_t = time.time() - t0

        s1_rm = skeleton.sum() - s1.sum()
        s5_rm = skeleton.sum() - s5.sum()

        # pixel-level agreement
        agree = np.sum(s1 == s5)
        total = s1.size
        agree_pct = 100.0 * agree / total
        s1_only = np.sum(s1 & ~s5)  # pixels in S1 but not S5
        s5_only = np.sum(s5 & ~s1)  # pixels in S5 but not S1

        print(f"{Lspur:5d} | {s1.sum():7d} {s1_rm:7d} {s1_t:8.3f} | "
              f"{s5.sum():7d} {s5_rm:7d} {s5_t:8.3f} | "
              f"{agree_pct:7.3f} {s1_only:7d} {s5_only:7d}")

        s1_results[Lspur] = s1
        s5_results[Lspur] = s5
        s1_timings[Lspur] = s1_t
        s5_timings[Lspur] = s5_t

    return s1_results, s5_results, s1_timings, s5_timings


def plot_s5_before_after(skeleton, result, Lspur):
    """Generate a before/after comparison figure for S5."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(skeleton, cmap='gray', interpolation='nearest')
    axes[0].set_title('Before (skeleton)')
    axes[0].axis('off')

    axes[1].imshow(result, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'After S5 (Lspur={Lspur})')
    axes[1].axis('off')

    # overlay
    removed = skeleton & ~result
    overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    overlay[result] = [255, 255, 255]
    overlay[removed] = [255, 50, 50]
    ep = _detect_endpoints(result)
    bp = _detect_branchpoints(result)
    overlay[ep] = [50, 150, 255]
    overlay[bp] = [255, 255, 0]

    axes[2].imshow(overlay, interpolation='nearest')
    axes[2].set_title('Overlay (white=kept, red=removed,\n'
                      'blue=endpoint, yellow=branchpoint)')
    axes[2].axis('off')

    plt.suptitle(f'S5 (skan) Spur Removal — Lspur={Lspur}', fontsize=14)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR, f'spur_removal_S5_L{Lspur}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_s1_vs_s5_overlay(skeleton, s1, s5, Lspur):
    """
    Overlay showing where S1 and S5 differ.
    Green = both agree (kept), Red = both removed,
    Cyan = S1 kept but S5 removed, Magenta = S5 kept but S1 removed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # S1 result
    axes[0].imshow(s1, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'S1 (Lspur={Lspur})\n{s1.sum()} px')
    axes[0].axis('off')

    # S5 result
    axes[1].imshow(s5, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'S5 (Lspur={Lspur})\n{s5.sum()} px')
    axes[1].axis('off')

    # difference overlay
    overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    both_kept = s1 & s5
    both_removed = skeleton & ~s1 & ~s5
    s1_only = s1 & ~s5       # S1 kept, S5 removed
    s5_only = s5 & ~s1       # S5 kept, S1 removed
    background = ~skeleton    # never had a pixel

    overlay[both_kept] = [255, 255, 255]       # white: agreement (kept)
    overlay[both_removed] = [255, 50, 50]      # red: agreement (removed)
    overlay[s1_only] = [0, 255, 255]           # cyan: S1 kept, S5 removed
    overlay[s5_only] = [255, 0, 255]           # magenta: S5 kept, S1 removed

    axes[2].imshow(overlay, interpolation='nearest')
    axes[2].set_title(f'Difference\nwhite=agree-kept, red=agree-removed\n'
                      f'cyan=S1only ({s1_only.sum()}px), '
                      f'magenta=S5only ({s5_only.sum()}px)')
    axes[2].axis('off')

    plt.suptitle(f'S1 vs S5 Comparison — Lspur={Lspur}', fontsize=14)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR, f'spur_S1_vs_S5_L{Lspur}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_s1_vs_s5_zoom(skeleton, s1, s5, Lspur, row_slice, col_slice, label):
    """Zoomed view of S1 vs S5 differences."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sk_z = skeleton[row_slice, col_slice]
    s1_z = s1[row_slice, col_slice]
    s5_z = s5[row_slice, col_slice]

    axes[0].imshow(s1_z, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'S1 (Lspur={Lspur})')
    axes[0].axis('off')

    axes[1].imshow(s5_z, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'S5 (Lspur={Lspur})')
    axes[1].axis('off')

    # difference overlay
    overlay = np.zeros((*sk_z.shape, 3), dtype=np.uint8)
    overlay[s1_z & s5_z] = [255, 255, 255]
    overlay[sk_z & ~s1_z & ~s5_z] = [255, 50, 50]
    overlay[s1_z & ~s5_z] = [0, 255, 255]
    overlay[s5_z & ~s1_z] = [255, 0, 255]

    axes[2].imshow(overlay, interpolation='nearest')
    axes[2].set_title('Difference')
    axes[2].axis('off')

    plt.suptitle(f'S1 vs S5 Zoomed: {label} (Lspur={Lspur})', fontsize=13)
    plt.tight_layout()

    fname = os.path.join(FIG_DIR,
                         f'spur_S1_vs_S5_L{Lspur}_zoom_{label}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_timing_comparison(s1_timings, s5_timings):
    """Bar chart comparing S1 and S5 timings."""
    lspurs = sorted(s1_timings.keys())
    s1_times = [s1_timings[l] for l in lspurs]
    s5_times = [s5_timings[l] for l in lspurs]

    x = np.arange(len(lspurs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, s1_times, width, label='S1 (MATLAB port)',
                   color='steelblue')
    bars5 = ax.bar(x + width/2, s5_times, width, label='S5 (skan)',
                   color='coral')

    ax.set_xlabel('Lspur')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('S1 vs S5 Execution Time')
    ax.set_xticks(x)
    ax.set_xticklabels(lspurs)
    ax.legend()

    # add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}s', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    for bar in bars5:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}s', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fname = os.path.join(FIG_DIR, 'spur_S1_vs_S5_timing.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

def full_tests():
    # synthetic tests
    test_basic_spur()
    test_isolated_segment()

    # branch analysis
    test_branch_analysis()

    # S5 on real data
    skeleton, s5_results, s5_timings = test_real_data()

    # head-to-head comparison
    data = np.load(TEST_FILE)
    s1_results, s5_results, s1_timings, s5_timings = compare_s1_s5(
        data, skeleton)

    # generate figures
    print("\n=== Generating S5 figures ===")
    for Lspur in [5, 10]:
        plot_s5_before_after(skeleton, s5_results[Lspur], Lspur)

    print("\n=== Generating S1 vs S5 comparison figures ===")
    for Lspur in [5, 10]:
        plot_s1_vs_s5_overlay(skeleton, s1_results[Lspur],
                              s5_results[Lspur], Lspur)

    # zoomed views around regions with differences
    for Lspur in [5, 10]:
        diff = s1_results[Lspur] ^ s5_results[Lspur]
        diff_rows, diff_cols = np.where(diff)
        if len(diff_rows) > 0:
            # pick the densest cluster of differences
            # simple approach: find the difference pixel closest to the
            # centroid of all differences
            cr = diff_rows.mean()
            cc = diff_cols.mean()
            dists = (diff_rows - cr)**2 + (diff_cols - cc)**2
            idx = np.argmin(dists)
            r0, c0 = diff_rows[idx], diff_cols[idx]
            margin = 40
            rslice = slice(max(0, r0 - margin),
                           min(skeleton.shape[0], r0 + margin))
            cslice = slice(max(0, c0 - margin),
                           min(skeleton.shape[1], c0 + margin))
            plot_s1_vs_s5_zoom(skeleton, s1_results[Lspur],
                               s5_results[Lspur], Lspur,
                               rslice, cslice, 'diff_center')

    # timing comparison
    plot_timing_comparison(s1_timings, s5_timings)

    print("\nAll tests complete.")


def llc_test():
    #### LLC test
    from fronts.finding import io as finding_io

    # Load
    version = '2'
    timestamp   = '2012-11-09T12_00_00'
    config  = 'D'
    
    bfile = finding_io.binary_filename(timestamp, config, version)
    print(f"Loading binary front file: {bfile}")
    bfronts = np.load(bfile)

    # Process
    print("Processing with Lspur=10")
    new_bfronts = skan_prune(bfronts, Lspur=10)
    new_bfile = bfile.replace('.npy', '_skan_L10.npy')

    # Save
    print(f"Saving to: {new_bfile}")
    np.save(new_bfile, new_bfronts)


if __name__ == '__main__':

    # Full tests
    full_tests()

    # LLC
    #llc_test()