""" Run front finding """
import os
import yaml
import numpy as np

import xarray
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from skimage import morphology

from dbof.dataset_creation.config import GlobalJobConfig

from fronts.finding import algorithms
from fronts.finding import io as finding_io
from fronts.finding import algorithms as finding_algorithms
from fronts.llc import io as llc_io

def find_gradb2_fronts(timestamp: str, cfg: GlobalJobConfig,
                clobber: bool = False):
    """ Find us the fronts in a gradb2 field

    Args:
        timestamp (str): Timestamp of the data to process.
        cfg (GlobalJobConfig): Global job configuration.
        version (str): Version of the data to use.
        clobber (bool, optional): _description_. Defaults to False.
    """

    # Check if the binary front field exists
    bfile = finding_io.binary_filename(timestamp, cfg.version)
    if os.path.isfile(bfile) and not clobber:
        print(f"Binary front field {bfile} exists and clobber is False. Returning")
        return

    # Load gradb2
    gradb2_file = llc_io.derived_filename(timestamp, 'gradb2', version=cfg.version)
    print(f"Loading gradb2 from: {gradb2_file}")
    gradb2 = xarray.open_dataset(gradb2_file)['gradb2'].values
    print(f"Loaded gradb2 with shape: {gradb2.shape}")


    # Load config
    print(f"Processing config: {cfg.version} for timestamp: {timestamp}")
    #config_file = find_config.config_filename(config)
    #cdict = find_config.load(config_file)

    # Binary parameters
    bparam = cfg.fronts.finding.__dict__
    bparam['n_workers'] = 10
    bparam['verbose'] = True

    # Do it
    fronts = finding_algorithms.fronts_from_gradb2(gradb2, **bparam)

    # Save em
    finding_io.save_binary_fronts(fronts, timestamp, cfg.version)


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