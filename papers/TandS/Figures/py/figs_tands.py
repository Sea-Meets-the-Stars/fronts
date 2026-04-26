""" First-draft figures for the T and S paper. """

import os
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl

from fronts.properties import io as prop_io
from fronts.properties.characteristics import turner_angle
from fronts.viz.properties import plot_property_jpdf

mpl.rcParams['font.family'] = 'stixgeneral'

# Paths
results_dir = os.path.join(
    os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'group_fronts', 'v2')
figures_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), '')  # papers/TandS/Figures/


def fig_turner_vs_gradb(
    outfile: str = 'fig_turner_vs_gradb.png',
    timestamp: str = '2012-11-09T12:00:00',
    run_tag: str = 'v2_bin_D',
    n_bins: int = 100,
):
    """2D histogram of Turner angle vs sqrt(gradb2) for individual fronts.

    Uses the mean Turner angle (computed from mean gradient fields)
    and the median of sqrt(gradb2).
    """
    # Load the front properties table
    df = prop_io.load_colocation_table(
        results_dir, timestamp, run_tag)

    # Compute Turner angle from the mean gradient fields
    tu_deg = turner_angle(
        df['gradtheta2_mean'].values,
        df['gradsalt2_mean'].values,
        df['gradrho2_mean'].values,
    )

    # Median of sqrt(gradb2) = median of |grad b|
    #gradb = np.sqrt(df['gradb2_median'].values)
    gradb = np.sqrt(df['gradb2_p90'].values)

    # Build the 2D histogram using the existing JPDF plotter
    fig = plot_property_jpdf(
        tu_deg, gradb,
        n_x_bins=n_bins,
        n_y_bins=n_bins,
        x_range=(-90, 90),
        y_range=(1e-8, 1e-5),
        fontsize=15,
        cfsz=13,
        y_log=True,
        cmap='Blues',
        xlabel=r'Turner angle  $Tu_h$  (deg)',
        ylabel=r'$|\nabla b|$  (s$^{-2}$)',
        title='Turner angle vs. buoyancy gradient  (2012-11-09)',
    )

    # Save
    outpath = os.path.join(figures_dir, outfile)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}")


def main(flg):
    """Main function to generate figures."""
    flg = int(flg)
    if flg == 1:
        fig_turner_vs_gradb()
    else:
        raise ValueError(f"Invalid flag: {flg}")

# Command line execution
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]

    main(flg)