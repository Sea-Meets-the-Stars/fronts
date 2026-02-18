"""Inspect binary front output within a spatial bounding box."""

import os

import numpy as np
import matplotlib.pyplot as plt

from fronts.finding.io import load_binary_fronts
from skimage import morphology

from IPython import embed

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

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


def thin_and_skeletonize(timestamp: str, config_lbl: str,
                         outdir: str = None, **kwargs):
    """Apply thin and skeletonize to an unthinned binary front image.

    Runs :func:`skimage.morphology.thin` and
    :func:`skimage.morphology.skeletonize` on the loaded binary front
    array and saves both results as ``.npy`` files in *outdir*.

    Parameters
    ----------
    timestamp : str
        Timestamp string for the snapshot (e.g. '2012-11-09T12_00_00').
    config_lbl : str
        Configuration label (e.g. 'A').
    outdir : str, optional
        Directory for the output files.  Defaults to ``dev/temp/``.
    **kwargs
        Passed to :func:`fronts.finding.io.load_binary_fronts`
        (``root``, ``path``).

    Returns
    -------
    thinned : np.ndarray
        Result of ``morphology.thin()``.
    skeleton : np.ndarray
        Result of ``morphology.skeletonize()``.
    """
    if outdir is None:
        outdir = TEMP_DIR
    os.makedirs(outdir, exist_ok=True)

    fronts = load_binary_fronts(timestamp, config_lbl, **kwargs)
    print(f'Loaded fronts: shape={fronts.shape}, '
          f'front pixels={np.sum(fronts)}')

    print('Running morphology.thin() ...')
    thinned = morphology.thin(fronts)
    print(f'  thinned front pixels: {np.sum(thinned)}')
    embed(header='106 of thin_and_skeletonize')

    print('Running morphology.skeletonize() ...')
    skeleton = morphology.skeletonize(fronts)
    print(f'  skeletonized front pixels: {np.sum(skeleton)}')

    tag = f'{timestamp}_{config_lbl}'
    thin_file = os.path.join(outdir, f'thinned_{tag}.npy')
    skel_file = os.path.join(outdir, f'skeletonized_{tag}.npy')

    np.save(thin_file, thinned)
    print(f'Saved: {thin_file}')
    np.save(skel_file, skeleton)
    print(f'Saved: {skel_file}')

    return thinned, skeleton


def compare_thinning(timestamp: str, config_lbl: str,
                     x0: int, x1: int, y0: int, y1: int,
                     tempdir: str = None, **kwargs):
    """Plot original, thinned, and skeletonized fronts side by side.

    Loads the original binary front image via
    :func:`fronts.finding.io.load_binary_fronts` and the thinned /
    skeletonized ``.npy`` files produced by :func:`thin_and_skeletonize`
    from *tempdir*, then plots the bounding-box subset of each.

    Parameters
    ----------
    timestamp : str
        Timestamp string for the snapshot (e.g. '2012-11-09T12_00_00').
    config_lbl : str
        Configuration label (e.g. 'A').
    x0, x1 : int
        Column range of the bounding box.
    y0, y1 : int
        Row range of the bounding box.
    tempdir : str, optional
        Directory containing the thinned/skeletonized files.
        Defaults to ``dev/temp/``.
    **kwargs
        Passed to :func:`fronts.finding.io.load_binary_fronts`
        (``root``, ``path``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the three panels.
    """
    if tempdir is None:
        tempdir = TEMP_DIR

    tag = f'{timestamp}_{config_lbl}'
    thin_file = os.path.join(tempdir, f'thinned_{tag}.npy')
    skel_file = os.path.join(tempdir, f'skeletonized_{tag}.npy')

    original = load_binary_fronts(timestamp, config_lbl, **kwargs)
    thinned = np.load(thin_file)
    skeleton = np.load(skel_file)

    bbox = np.s_[y0:y1, x0:x1]
    panels = [
        ('Original', original[bbox]),
        ('Thinned', thinned[bbox]),
        ('Skeletonized', skeleton[bbox]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, data) in zip(axes, panels):
        ax.imshow(data, origin='lower', cmap='binary',
                  interpolation='nearest')
        ax.set_title(f'{label}  (n={np.sum(data)})')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

    fig.suptitle(f'[{x0}:{x1}, {y0}:{y1}]  t={timestamp}  cfg={config_lbl}')
    fig.tight_layout()
    plt.show()

    return fig


def check_idempotent(timestamp: str, config_lbl: str,
                     x0: int, x1: int, y0: int, y1: int,
                     tempdir: str = None, **kwargs):
    """Check whether thinning is idempotent by re-thinning.

    Loads the previously thinned output, applies
    :func:`skimage.morphology.thin` a second time, and compares the
    result to the original thinned array.  Prints whether the two are
    identical and shows a side-by-side plot of the bounding-box subset
    (original, thinned-once, thinned-twice).

    Parameters
    ----------
    timestamp : str
        Timestamp string for the snapshot (e.g. '2012-11-09T12_00_00').
    config_lbl : str
        Configuration label (e.g. 'Z').
    x0, x1 : int
        Column range of the bounding box.
    y0, y1 : int
        Row range of the bounding box.
    tempdir : str, optional
        Directory containing the thinned file.
        Defaults to ``dev/temp/``.
    **kwargs
        Passed to :func:`fronts.finding.io.load_binary_fronts`
        (``root``, ``path``).

    Returns
    -------
    identical : bool
        ``True`` if the thinned and re-thinned arrays are identical.
    fig : matplotlib.figure.Figure
        The figure containing the three panels.
    """
    if tempdir is None:
        tempdir = TEMP_DIR

    tag = f'{timestamp}_{config_lbl}'
    thin_file = os.path.join(tempdir, f'thinned_{tag}.npy')

    original = load_binary_fronts(timestamp, config_lbl, **kwargs)
    thinned = np.load(thin_file)

    print('Running morphology.thin() on already-thinned image ...')
    rethinned = morphology.thin(thinned)

    identical = np.array_equal(thinned, rethinned)
    n_orig = int(np.sum(original))
    n_thin = int(np.sum(thinned))
    n_rethin = int(np.sum(rethinned))
    n_diff = int(np.sum(thinned != rethinned))

    print(f'Original front pixels:    {n_orig}')
    print(f'Thinned front pixels:     {n_thin}')
    print(f'Re-thinned front pixels:  {n_rethin}')
    print(f'Pixels that differ:       {n_diff}')
    print(f'Identical: {identical}')

    bbox = np.s_[y0:y1, x0:x1]
    panels = [
        ('Original', original[bbox]),
        ('Thinned', thinned[bbox]),
        ('Re-thinned', rethinned[bbox]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, data) in zip(axes, panels):
        ax.imshow(data, origin='lower', cmap='binary',
                  interpolation='nearest')
        ax.set_title(f'{label}  (n={np.sum(data)})')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

    status = 'IDENTICAL' if identical else f'DIFFER ({n_diff} px)'
    fig.suptitle(f'Idempotency check: {status}\n'
                 f'[{x0}:{x1}, {y0}:{y1}]  t={timestamp}  cfg={config_lbl}')
    fig.tight_layout()
    plt.show()

    return identical, fig


# Command line interface
if __name__ == '__main__':
    if True:
    # Original
        plot_fronts_bbox('2012-11-09T12_00_00', 'C', 
            12014,12230, 9237,9400)
        # Try another thin on full image
        #plot_fronts_bbox('2012-11-09T12_00_00', 'C', 
        #    12014,12230, 9237,9400, thin_full=True)
        # Try another thin on subset
        plot_fronts_bbox('2012-11-09T12_00_00', 'C', 
            12014,12230, 9237,9400, thin_again=True)
    
    # Thin and skeletonize
    #thin_and_skeletonize('2012-11-09T12_00_00', 'Z')

    # Compare thinning and skeletonization
    #compare_thinning('2012-11-09T12_00_00', 'Z',
    #    12014,12230, 9237,9400)

    # Check idempotency of thinning
    #identical, fig = check_idempotent('2012-11-09T12_00_00', 'Z',
    #    12014,12230, 9237,9400)