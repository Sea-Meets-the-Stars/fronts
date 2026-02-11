""" Run em """

import numpy as np

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from skimage import morphology

from fronts.finding import algorithms

def one_cutout(Divb2:np.ndarray, front_params:dict):

    # Bool
    fronts = algorithms.fronts_from_divb2(Divb2, **front_params)

    # Label
    flabels = morphology.label(fronts, connectivity=front_params['connectivity'])

    return flabels

def many_cutouts(cutouts:list, front_params:dict, n_cores:int=10):

    map_fn = partial(one_cutout, front_params=front_params)

    # Multi-process time
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(cutouts) // n_cores if len(cutouts) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, cutouts,
                                            chunksize=chunksize), total=len(cutouts)))

    # Return
    return answers