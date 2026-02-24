# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os

from fronts.llc import io as llc_io
from fronts.finding import config as find_config
from fronts.finding import algorithms
from fronts.finding import io as finding_io
import xarray

from IPython import embed

def explore_threshold(timestamp:str, configs:list=['A', 'B', 'C']): 
    """ 
    Explore the threshold for front finding
    using a range of thresholds.  Each binary front field is saved to disk.

    Args:
        timestamp (str): Timestamp of the data to process
        configs (list, optional): List of config files to process. Defaults to ['A', 'B', 'C'].
    """

    # Load Divb2
    Divb2_file = llc_io.derived_filename(timestamp, 'Divb2')
    print(f"Loading Divb2 from: {Divb2_file}")
    Divb2 = xarray.open_dataset(Divb2_file)['Divb2'].values
    print(f"Loaded Divb2 with shape: {Divb2.shape}")

    # Loop on configs
    for config in configs:
        print(f"Processing config: {config}")

        # Load config 
        config_file = find_config.config_filename(config)
        cdict = find_config.load(config_file)

        # Binary parameters
        bparam = cdict['binary']
        bparam['n_workers'] = 10
        bparam['verbose'] = True

        # Do it
        fronts = algorithms.fronts_from_divb2(Divb2, **bparam)

        # Save em
        finding_io.save_binary_fronts(
            fronts, timestamp, config)
    
def build_unthinned(timestamp:str, config:str='Z'): 
    """ 
    Build the unthinned/cropped front field for debugging

    The unthinned/cropped front field is saved to disk.

    Args:
        timestamp (str): Timestamp of the data to process
        config (str, optional): Config file to process. Defaults to 'Z'.
    """

    # Load Divb2
    Divb2_file = llc_io.derived_filename(timestamp, 'Divb2')
    print(f"Loading Divb2 from: {Divb2_file}")
    Divb2 = xarray.open_dataset(Divb2_file)['Divb2'].values
    print(f"Loaded Divb2 with shape: {Divb2.shape}")

    # Load config 
    config_file = find_config.config_filename(config)
    cdict = find_config.load(config_file)

    # Binary parameters
    bparam = cdict['binary']
    bparam['n_workers'] = 10
    bparam['verbose'] = True

    # Do it
    fronts = algorithms.fronts_from_divb2(Divb2, **bparam)

    # Save em
    finding_io.save_binary_fronts(
        fronts, timestamp, config)


def debug_thinning(timestamp:str, config:str='C'): 
    """ 
    Debug the thinning of the front field

    Runs the main thinning with debug=True.

    Args:
        timestamp (str): Timestamp of the data to process
        config (str, optional): Config file to process. Defaults to 'C'.
    """

    # Load Divb2
    Divb2_file = llc_io.derived_filename(timestamp, 'Divb2')
    print(f"Loading Divb2 from: {Divb2_file}")
    Divb2 = xarray.open_dataset(Divb2_file)['Divb2'].values
    print(f"Loaded Divb2 with shape: {Divb2.shape}")

    # Load config 
    config_file = find_config.config_filename(config)
    cdict = find_config.load(config_file)

    # Binary parameters
    bparam = cdict['binary']
    bparam['n_workers'] = 10
    bparam['verbose'] = True

    # Do it
    fronts = algorithms.fronts_from_divb2(
        Divb2, debug=True, **bparam)

# #######################################################33
def main(flg:str):
    flg= int(flg)

    # Explore threshold
    if flg == 1:
        timestamp = '2012-11-09T12_00_00'
        explore_threshold(timestamp)
        #explore_threshold(timestamp, configs=['C'])

    # Generate unthinned/cropped front for debugging
    if flg == 2:
        timestamp = '2012-11-09T12_00_00'
        build_unthinned(timestamp)

    # More debugging
    if flg == 3:
        timestamp = '2012-11-09T12_00_00'
        debug_thinning(timestamp)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]

    main(flg)