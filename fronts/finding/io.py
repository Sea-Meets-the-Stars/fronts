# Module for I/O of finding fronts

import os
import numpy as np

from fronts.llc import io as llc_io


def binary_filename(timestamp:str, config_lbl:str, version:str,
                    root:str='LLC4320'):
    """Build the filename for a binary-front .npy file.

    The file is placed under ``PATH/V{version}/YYYYMMDD_HHMMSS/``.

    Parameters
    ----------
    timestamp : str
        Timestamp string for the snapshot (e.g. '2012-11-09T12_00_00').
    config_lbl : str
        Configuration label (e.g. 'A') appended to the filename.
    version : str
        Version of the data to use.
    root : str, optional
        Dataset root name. Default is 'LLC4320'.

    Returns
    -------
    str
        Full path of the form ``{path}/{root}_{timestamp}_bin_{config_lbl}.npy``.
    """
    path = llc_io.fronts_dir(version, timestamp)

    # Generate base
    basefile = f'{root}_{timestamp}_v{version}_bin_{config_lbl}.npy'

    # Join and return
    return os.path.join(path, basefile)

def load_binary_fronts(timestamp:str, config_lbl:str, version:str, **kwargs):
    """Load a binary-front array from a .npy file.

    Parameters
    ----------
    timestamp : str
        Timestamp string for the snapshot.
    config_lbl : str
        Configuration label used when the file was saved.
    **kwargs
        Passed to :func:`binary_filename` (``root``).

    Returns
    -------
    np.ndarray
        Binary front array.
    """
    # Grab filename
    b_file = binary_filename(timestamp, config_lbl, version, **kwargs)
    print(f"Loading binary front field from {b_file}")

    # Open
    binary_fronts = np.load(b_file)

    # Return
    return binary_fronts

def save_binary_fronts(fronts:np.ndarray, timestamp:str, config_lbl:str, 
    version:str, **kwargs):
    """Save a binary-front array to a .npy file.

    Creates the output directory if it does not already exist.

    Parameters
    ----------
    fronts : np.ndarray
        Binary front array to save.
    timestamp : str
        Timestamp string for the snapshot.
    config_lbl : str
        Configuration label appended to the filename.
    version : str
        Version of the data to use.
    **kwargs
        Passed to :func:`binary_filename` (``root``).
    """
    # Grab filename
    b_file = binary_filename(timestamp, config_lbl, version, **kwargs)

    # Generate directory if it doesn't exist
    os.makedirs(os.path.dirname(b_file), exist_ok=True)

    # Open
    np.save(b_file, fronts)
    print(f"Wrote: {b_file}")

    return
