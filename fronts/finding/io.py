# Module for I/O of finding fronts

import os
import numpy as np

def binary_filename(timestamp:str, config_lbl:str, 
                    root:str='LLC4320', path:str=None):
    # Path
    if path is None:
        path = os.path.join(os.getenv('OS_OGCM'),
            'LLC', 'Fronts', 'outputs')
    
    # Generate base
    basefile = f'{root}_{timestamp}_bin_{config_lbl}.npy'

    # Join and return
    return os.path.join(path, basefile)

def load_binary_fronts(timestamp:str, config_lbl:str, **kwargs):

    # Grab filename
    b_file = binary_filename(timestamp, config_lbl, **kwargs)

    # Open
    binary_fronts = np.load(b_file)

    # Return
    return binary_fronts

def save_binary_fronts(fronts:np.ndarray, timestamp:str, config_lbl:str, **kwargs):

    # Grab filename
    b_file = binary_filename(timestamp, config_lbl, **kwargs)

    # Open
    np.save(b_file, fronts)
    print(f"Wrote: {b_file}")

    return
