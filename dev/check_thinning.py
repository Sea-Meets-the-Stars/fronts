"""Inspect binary front output within a spatial bounding box."""

import numpy as np
import matplotlib.pyplot as plt

from fronts.finding.io import load_binary_fronts
from skimage import morphology

def plot_fronts_bbox(timestamp: str, config_lbl: str,
                     x0: int, x1: int, y0: int, y1: int,
                     thin_again:bool=False,
                     thin_full:bool=False,
                     **kwargs):
    """Load binary fronts and plot a bounding-box subset.

    Parameters
    ----------
    timestamp : str
        Timestamp string for the snapshot (e.g. '2012-11-09T12_00_00').
    config_lbl : str
        Configuration label (e.g. 'A').
    x0, y0 : int
        Top-left corner of the bounding box (row, column).
    x1, y1 : int
        Bottom-right corner of the bounding box (row, column).
    **kwargs
        Passed to :func:`fronts.finding.io.load_binary_fronts`
        (``root``, ``path``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    title = f'Fronts [{x0}:{x1}, {y0}:{y1}]  '
    title += f't={timestamp}  cfg={config_lbl}'
    fronts = load_binary_fronts(timestamp, config_lbl, **kwargs)
    n_fronts = np.sum(fronts)
    print(f'Number of front pixels in original: {n_fronts}')

    if thin_full:
        fronts = morphology.thin(fronts)
        title += '  thinned full'
        print(f'Number of front pixels in thinned full: {np.sum(fronts)}')

    subset = fronts[y0:y1, x0:x1]
    if thin_again:
        subset = morphology.thin(subset)
        title += '  thinned subset'

    fig, ax = plt.subplots()
    ax.imshow(subset, origin='lower', cmap='binary')
    ax.set_title(title)

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.show()

    return fig


# Command line interface
if __name__ == '__main__':
    # Original
    plot_fronts_bbox('2012-11-09T12_00_00', 'C', 
        12014,12230, 9237,9400)
    # Try another thin on full image
    plot_fronts_bbox('2012-11-09T12_00_00', 'C', 
        12014,12230, 9237,9400, thin_full=True)
    # Try another thin on subset
    plot_fronts_bbox('2012-11-09T12_00_00', 'C', 
        12014,12230, 9237,9400, thin_again=True)