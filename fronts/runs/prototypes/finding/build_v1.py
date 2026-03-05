# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os

import xarray

from dbof.cli import generate_fronts_global

from IPython import embed

def generate_gradb2(timestamp:str):
    """ 
    Explore the threshold for front finding
    using a range of thresholds.  Each binary front field is saved to disk.

    Args:
        timestamp (str): Timestamp of the data to process
        configs (list, optional): List of config files to process. Defaults to ['A', 'B', 'C'].
        version (str, optional): Version of the algorithm to use. Defaults to '0'.
    """


# #######################################################33
def main(flg:str):
    flg= int(flg)

    # Generate gradb2
    if flg == 1:
        timestamp = '2012-11-09T12_00_00'
        config_file = './testing_global_v1.yaml'
        generate_gradb2(config_file, run_id)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        # flg = 1 # Explore threshold
        pass
    else:
        flg = sys.argv[1]

    main(flg)