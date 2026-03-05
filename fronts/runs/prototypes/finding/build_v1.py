# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os

import xarray

from dbof.cli import generate_fronts_global
from dbof.cli import zarr_to_netcdf
import dbof.dataset_creation.config as config

from fronts.llc import io as llc_io

from IPython import embed

def generate_gradb2(config_file:str):
    """ 
    Generate the gradb2 field for the given config file.

    Args:
        timestamp (str): Timestamp of the data to process
        configs (list, optional): List of config files to process. Defaults to ['A', 'B', 'C'].
        version (str, optional): Version of the algorithm to use. Defaults to '0'.
    """
    generate_fronts_global.main(config_file)


# #######################################################33
def main(flg:str):
    flg= int(flg)

    field = 'gradb2'

    # Generate gradb2 as zarr
    if flg == 1:
        config_file = './testing_global_v1.yaml'
        generate_gradb2(config_file)

    # Convert zarr to netcdf (this could be combined with the first step)
    if flg == 2:
        timestamp = '2012-11-09T12_00_00'
        # Output
        full_path = llc_io.derived_filename(timestamp, field, version='1')
        out_dir = os.path.dirname(full_path)
        out_file = os.path.basename(full_path)
        # Confg
        config_file = './testing_global_v1.yaml'
        cfg = config.load_config(config_file)
        # Call it
        zarr_to_netcdf.main(out_dir, 
            output_filename=out_file,
            mode='snapshots',
            run_id=cfg.run.run_id,
            s3_endpoint=cfg.output.s3_endpoint,
            bucket=cfg.output.bucket,
            channels=[field],
            dates=cfg.data.date_iterations,
            dataset_name=cfg.output.dataset_name,
            folder=cfg.output.folder)


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