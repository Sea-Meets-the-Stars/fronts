
import os

from fronts.llc import io as llc_io
from fronts.finding import config as find_config
from fronts.finding import algorithms
from fronts.finding import io as finding_io
import xarray

from IPython import embed

def explore_threshold(timestamp:str): 
    configs = ['A', 'B', 'C']

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

# #######################################################33
def main(flg:str):
    flg= int(flg)

    # Explore threshold
    if flg == 1:
        timestamp = '2012-11-09T12_00_00'
        explore_threshold(timestamp)

    # Generate unthinned/cropped front for debugging
    if flg == 2:
        timestamp = '2012-11-09T12_00_00'
        build_unthinned(timestamp)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1 # Generate super table
        #flg = 2 # Preproc SST only
        #flg = 3 # Build em all
        pass
    else:
        flg = sys.argv[1]

    main(flg)