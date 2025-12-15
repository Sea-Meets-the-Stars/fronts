"""
Thinning algorithm comparison module.

Generates 100 examples comparing the original morphology.thin() algorithm
with the new thin_cc.thin() algorithm across various parameter combinations.

Outputs PNG files to the dev/plots/ folder.
"""

import os
import numpy as np
from skimage import morphology
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

from fronts.dbof import utils as dbof_utils
from fronts.dbof import io as dbof_io
from fronts.finding import pyboa as ff_pyboa
from fronts.finding import thin_cc

# Path to the DBOF dev file (relative to this script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DBOF_DEV_FILE = os.path.join(SCRIPT_DIR, '..', 'fronts', 'runs', 'dbof', 'dev', 'llc4320_dbof_dev.json')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_random_uids(dbof_file: str, n_samples: int = 100, seed: int = 42) -> list:
    """
    Get random UIDs from the DBOF table.

    Parameters
    ----------
    dbof_file : str
        Path to DBOF JSON file
    n_samples : int
        Number of UIDs to sample
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list
        List of random UIDs
    """
    dbof_table = dbof_io.load_main_table(dbof_file)
    np.random.seed(seed)

    n_available = len(dbof_table)
    n_to_sample = min(n_samples, n_available)

    indices = np.random.choice(n_available, size=n_to_sample, replace=False)
    uids = dbof_table.iloc[indices]['UID'].values.tolist()

    return uids


def load_field_data(dbof_file: str, uid: int) -> dict:
    """
    Load field data for a given UID.

    Parameters
    ----------
    dbof_file : str
        Path to DBOF JSON file
    uid : int
        Unique identifier for the cutout

    Returns
    -------
    dict
        Dictionary containing field data
    """
    try:
        field_data = dbof_utils.grab_fields(dbof_file, 'all', uid)
        return field_data
    except Exception as e:
        print(f"Error loading UID {uid}: {e}")
        return None


def apply_threshold(divb2: np.ndarray, percentile: int = 90) -> np.ndarray:
    """
    Apply global percentile threshold to Divb2 field.

    Parameters
    ----------
    divb2 : np.ndarray
        Divb2 field data
    percentile : int
        Percentile threshold (0-100)

    Returns
    -------
    np.ndarray
        Binary front mask
    """
    return ff_pyboa.global_threshold(divb2, percentile)


def thin_original(fronts: np.ndarray) -> np.ndarray:
    """
    Apply original morphology.thin() algorithm.

    Parameters
    ----------
    fronts : np.ndarray
        Binary front mask

    Returns
    -------
    np.ndarray
        Thinned front mask
    """
    return morphology.thin(fronts)


def thin_cc_algorithm(sst: np.ndarray, fronts: np.ndarray,
                       min_segment_gap: int = 2,
                       median_size: int = 5) -> np.ndarray:
    """
    Apply Cornillon-style thin_cc.thin() algorithm.

    Parameters
    ----------
    sst : np.ndarray
        SST field in Kelvin
    fronts : np.ndarray
        Binary front mask
    min_segment_gap : int
        Minimum gap between front segments
    median_size : int
        Size of median filter window

    Returns
    -------
    np.ndarray
        Thinned front mask
    """
    # Convert boolean to int with front_value=4
    fronts_int = 4 * fronts.astype(np.int16)

    return thin_cc.thin_fronts(
        sst,
        fronts_int,
        apply_median=True,
        median_size=median_size,
        min_segment_gap=min_segment_gap
    )


def plot_comparison_original(uid: int, field_data: dict, percentiles: list,
                              outfile: str):
    """
    Plot comparison of original morphology.thin() with different percentile thresholds.

    Parameters
    ----------
    uid : int
        Unique identifier
    field_data : dict
        Dictionary containing field data
    percentiles : list
        List of percentile values to compare
    outfile : str
        Output file path
    """
    divb2 = field_data['Divb2']
    sst = field_data['SSTK']

    n_cols = len(percentiles) + 1  # +1 for SST/Divb2 reference

    fig = plt.figure(figsize=(4 * n_cols, 8))
    gs = gridspec.GridSpec(2, n_cols)

    # First column: SST and Divb2
    ax_sst = plt.subplot(gs[0, 0])
    im_sst = ax_sst.imshow(sst, cmap='RdYlBu_r', origin='lower')
    ax_sst.set_title('SST (K)')
    plt.colorbar(im_sst, ax=ax_sst, fraction=0.046)

    ax_divb2 = plt.subplot(gs[1, 0])
    im_divb2 = ax_divb2.imshow(divb2, cmap='Greys', origin='lower')
    ax_divb2.set_title(r'$|\nabla b|^2$')
    plt.colorbar(im_divb2, ax=ax_divb2, fraction=0.046)

    # Remaining columns: different percentile thresholds
    for i, prcnt in enumerate(percentiles, start=1):
        # Threshold
        fronts_thresh = apply_threshold(divb2, prcnt)

        # Thin with original algorithm
        fronts_thinned = thin_original(fronts_thresh)

        # Top row: thresholded
        ax_thresh = plt.subplot(gs[0, i])
        ax_thresh.imshow(divb2, cmap='Greys', origin='lower', alpha=0.5)
        pcol, prow = np.where(fronts_thresh)
        ax_thresh.scatter(prow, pcol, s=0.5, color='blue', alpha=0.3)
        ax_thresh.set_title(f'Threshold {prcnt}%')

        # Bottom row: thinned
        ax_thin = plt.subplot(gs[1, i])
        ax_thin.imshow(divb2, cmap='Greys', origin='lower', alpha=0.5)
        pcol, prow = np.where(fronts_thinned)
        ax_thin.scatter(prow, pcol, s=1, color='red', alpha=0.8)
        n_pixels = np.sum(fronts_thinned)
        ax_thin.set_title(f'morphology.thin()\n{n_pixels} pixels')

    # Add grid to all axes
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(which='major', color='lightgrey', linestyle='--', alpha=0.3)

    plt.suptitle(f'Original Thinning - UID: {uid}', fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved: {outfile}")


def plot_comparison_cc(uid: int, field_data: dict,
                        min_segment_gaps: list,
                        median_sizes: list,
                        percentile: int = 90,
                        outfile: str = None):
    """
    Plot comparison of thin_cc.thin() with different parameter combinations.

    Parameters
    ----------
    uid : int
        Unique identifier
    field_data : dict
        Dictionary containing field data
    min_segment_gaps : list
        List of min_segment_gap values to compare
    median_sizes : list
        List of median_size values to compare
    percentile : int
        Percentile threshold to use
    outfile : str
        Output file path
    """
    divb2 = field_data['Divb2']
    sst = field_data['SSTK']

    # Threshold first
    fronts_thresh = apply_threshold(divb2, percentile)

    n_rows = len(median_sizes) + 1  # +1 for header row
    n_cols = len(min_segment_gaps) + 1  # +1 for reference column

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols)

    # Top-left: SST
    ax_sst = plt.subplot(gs[0, 0])
    im_sst = ax_sst.imshow(sst, cmap='RdYlBu_r', origin='lower')
    ax_sst.set_title('SST (K)')
    plt.colorbar(im_sst, ax=ax_sst, fraction=0.046)

    # Top row: Divb2 and thresholded for different gaps (header)
    for j, gap in enumerate(min_segment_gaps, start=1):
        ax = plt.subplot(gs[0, j])
        ax.imshow(divb2, cmap='Greys', origin='lower', alpha=0.5)
        pcol, prow = np.where(fronts_thresh)
        ax.scatter(prow, pcol, s=0.5, color='blue', alpha=0.3)
        ax.set_title(f'gap={gap}\nThresh {percentile}%')

    # Left column: reference with original thinning
    for i, med_size in enumerate(median_sizes, start=1):
        ax = plt.subplot(gs[i, 0])

        # Apply original thinning for reference
        fronts_orig = thin_original(fronts_thresh)
        ax.imshow(divb2, cmap='Greys', origin='lower', alpha=0.5)
        pcol, prow = np.where(fronts_orig)
        ax.scatter(prow, pcol, s=1, color='green', alpha=0.8)
        n_pixels = np.sum(fronts_orig)
        ax.set_title(f'med={med_size}\nOriginal: {n_pixels}px')

    # Grid: thin_cc with different parameters
    for i, med_size in enumerate(median_sizes, start=1):
        for j, gap in enumerate(min_segment_gaps, start=1):
            ax = plt.subplot(gs[i, j])

            try:
                fronts_cc = thin_cc_algorithm(
                    sst, fronts_thresh,
                    min_segment_gap=gap,
                    median_size=med_size
                )

                ax.imshow(divb2, cmap='Greys', origin='lower', alpha=0.5)
                pcol, prow = np.where(fronts_cc > 0)
                ax.scatter(prow, pcol, s=1, color='red', alpha=0.8)
                n_pixels = np.sum(fronts_cc > 0)
                ax.set_title(f'CC: {n_pixels}px')
            except Exception as e:
                ax.set_title(f'Error: {str(e)[:20]}')
                ax.imshow(divb2, cmap='Greys', origin='lower', alpha=0.5)

    # Add grid to all axes
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(which='major', color='lightgrey', linestyle='--', alpha=0.3)

    plt.suptitle(f'Cornillon Thinning (thin_cc) - UID: {uid}', fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved: {outfile}")


def run_original_comparison(n_examples: int = 100,
                            percentiles: list = None,
                            dbof_file: str = None,
                            test:bool=False):
    """
    Generate comparison plots for original morphology.thin() algorithm.

    Parameters
    ----------
    n_examples : int
        Number of examples to generate
    percentiles : list
        List of percentile thresholds to use
    dbof_file : str
        Path to DBOF JSON file
    """
    if percentiles is None:
        percentiles = [70, 80, 85, 90]

    if dbof_file is None:
        dbof_file = DBOF_DEV_FILE

    print(f"Generating {n_examples} examples for original thinning algorithm")
    print(f"Percentiles: {percentiles}")
    print(f"Output directory: {PLOTS_DIR}")

    # Get random UIDs
    if test:
        uids = [1322611708493510]
    else:
        uids = get_random_uids(dbof_file, n_examples)

    success_count = 0
    for i, uid in enumerate(uids):
        print(f"\n[{i+1}/{n_examples}] Processing UID: {uid}")

        # Load field data
        field_data = load_field_data(dbof_file, uid)

        if field_data is None:
            print(f"  Skipping UID {uid}: failed to load data")
            continue

        if 'Divb2' not in field_data or 'SSTK' not in field_data:
            print(f"  Skipping UID {uid}: missing required fields")
            continue

        # Generate plot
        outfile = os.path.join(
            PLOTS_DIR,
            f'thin_original_{i:03d}_uid{uid}.png'
        )

        try:
            plot_comparison_original(uid, field_data, percentiles, outfile)
            success_count += 1
        except Exception as e:
            print(f"  Error plotting UID {uid}: {e}")

    print(f"\nCompleted: {success_count}/{n_examples} examples generated")


def run_cc_comparison(n_examples: int = 100,
                      min_segment_gaps: list = None,
                      median_sizes: list = None,
                      percentile: int = 90,
                      dbof_file: str = None,
                      test:bool=False):
    """
    Generate comparison plots for thin_cc.thin() algorithm.

    Parameters
    ----------
    n_examples : int
        Number of examples to generate
    min_segment_gaps : list
        List of min_segment_gap values to compare
    median_sizes : list
        List of median_size values to compare
    percentile : int
        Percentile threshold to use
    dbof_file : str
        Path to DBOF JSON file
    """
    if min_segment_gaps is None:
        min_segment_gaps = [2, 3, 4]

    if median_sizes is None:
        median_sizes = [3, 5]

    if dbof_file is None:
        dbof_file = DBOF_DEV_FILE

    print(f"Generating {n_examples} examples for thin_cc algorithm")
    print(f"min_segment_gaps: {min_segment_gaps}")
    print(f"median_sizes: {median_sizes}")
    print(f"Percentile threshold: {percentile}")
    print(f"Output directory: {PLOTS_DIR}")

    # Get random UIDs
    if test:
        uids = [1322611708493510]
    else:
        uids = get_random_uids(dbof_file, n_examples)

    success_count = 0
    for i, uid in enumerate(uids):
        print(f"\n[{i+1}/{n_examples}] Processing UID: {uid}")

        # Load field data
        field_data = load_field_data(dbof_file, uid)

        if field_data is None:
            print(f"  Skipping UID {uid}: failed to load data")
            continue

        if 'Divb2' not in field_data or 'SSTK' not in field_data:
            print(f"  Skipping UID {uid}: missing required fields")
            continue

        # Generate plot
        outfile = os.path.join(
            PLOTS_DIR,
            f'thin_cc_{i:03d}_uid{uid}.png'
        )

        try:
            plot_comparison_cc(
                uid, field_data,
                min_segment_gaps, median_sizes,
                percentile, outfile
            )
            success_count += 1
        except Exception as e:
            print(f"  Error plotting UID {uid}: {e}")

    print(f"\nCompleted: {success_count}/{n_examples} examples generated")


def run_all(n_examples: int = 100, dbof_file: str = None,
            test:bool=False):
    """
    Run both original and CC thinning comparisons.

    Parameters
    ----------
    n_examples : int
        Number of examples to generate for each algorithm
    dbof_file : str
        Path to DBOF JSON file
    test : bool
    """
    if test:
        n_examples = 1
    print("=" * 60)
    print("THINNING ALGORITHM COMPARISON")
    print("=" * 60)

    # Original algorithm with varied percentiles
    run_original_comparison(
        n_examples=n_examples,
        percentiles=[70, 80, 85, 90],
        dbof_file=dbof_file,
        test=test
    )

    print("\n" + "=" * 60)

    # CC algorithm with varied parameters
    run_cc_comparison(
        n_examples=n_examples,
        min_segment_gaps=[2, 3, 4],
        median_sizes=[3, 5],
        percentile=90,
        dbof_file=dbof_file,
        test=test
    )

    print("\n" + "=" * 60)
    print("ALL COMPARISONS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    # Test
    run_all(test=True)

    #run_all(n_examples=100)
