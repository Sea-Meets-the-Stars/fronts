
import os

from fronts.llc import io as llc_io
from fronts.finding import config as find_config
from fronts.finding import algorithms
import xarray

from IPython import embed

def explore_threshold(timestamp:str): 
    configs = ['A', 'B', 'C']

    # Load Divb2
    Divb2_file = llc_io.derived_filename(timestamp, 'Divb2')
    Divb2 = xarray.open_dataset(Divb2_file)['Divb2'].values

    # Loop on configs
    for config in configs:
        # Load config 
        config_file = find_config.config_filename(config)
        cdict = find_config.load(config_file)

        #
        fronts = algorithms.fronts_from_divb2(
            Divb2, thin=True, verbose=True, 
            thresh_mode=thresh_mode, n_workers=n_workers)

    

# #######################################################33
def main(flg:str):
    flg= int(flg)

    # Explore threshold
    if flg == 1:
        timestamp = '2012-11-09T12_00_00'


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