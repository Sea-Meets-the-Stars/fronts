# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os

import xarray

from dbof.cli import generate_fronts_global
from dbof.cli import zarr_to_netcdf
import dbof.dataset_creation.config as dbof_config

from fronts.preproc import inpaint_edges
from fronts.llc import io as llc_io
from fronts.finding import algorithms
from fronts.finding import config as find_config
from fronts.finding import io as finding_io

from IPython import embed

def generate_gradb2(timestamp:str, config_file:str, field:str='gradb2'):
    """ 
    Generate the gradb2 field for the given config file.

    Args:
        timestamp (str): Timestamp of the data to process
        configs (list, optional): List of config files to process. Defaults to ['A', 'B', 'C'].
        version (str, optional): Version of the algorithm to use. Defaults to '0'.
    """
    # zarr in s3
    generate_fronts_global.main(config_file)

    # Output to netcdf on disk
    full_path = llc_io.derived_filename(timestamp, field, version='1')
    out_dir = os.path.dirname(full_path)
    out_file = os.path.basename(full_path)
    # Confg
    cfg = dbof_config.load_config(config_file)
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

def find_fronts(timestamp:str, config:str, version:str, inpaint:bool=False,
    clobber:bool=False):
    """ Find us the fronts

    Args:
        timestamp (str): _description_
        config (str): _description_
        version (str): _description_
        inpaint (bool, optional): _description_. Defaults to False.
        clobber (bool, optional): _description_. Defaults to False.
    """

    # Check if the binary front field exists
    bfile = finding_io.binary_filename(timestamp, config, version)
    if os.path.isfile(bfile) and not clobber:
        print(f"Binary front field {bfile} exists and clobber is False. Returning")
        return

    # Load gradb2
    gradb2_file = llc_io.derived_filename(timestamp, 'gradb2', version=version)
    print(f"Loading gradb2 from: {gradb2_file}")
    gradb2 = xarray.open_dataset(gradb2_file)['gradb2'].values

    print(f"Loaded gradb2 with shape: {gradb2.shape}")

    # Inpaint
    if inpaint:
        print("Inpainting...")
        gradb2 = inpaint_edges.inpaint(gradb2, method='biharmonic',
                         second_pass='regular',
                         second_threshold=1e-20)
        print("Inpainted.")

    print(f"Processing config: {config}")

    # Load config
    config_file = find_config.config_filename(config)
    cdict = find_config.load(config_file)

    # Binary parameters
    bparam = cdict['binary']
    bparam['n_workers'] = 10
    bparam['verbose'] = True

    # Do it
    fronts = algorithms.fronts_from_gradb2(gradb2, **bparam)

    # Save em
    finding_io.save_binary_fronts(
        fronts, timestamp, config, version)


# #######################################################
def main(flg:str):
    flg= int(flg)

    # Generate gradb2 as zarr
    if flg == 1:
        timestamp = '2012-11-09T12_00_00'
        config_file = './testing_global_v1.yaml'
        generate_gradb2(timestamp, config_file)

    # Find fronts -- binary pixels
    if flg == 2:
        timestamp = '2012-11-09T12_00_00'
        version = '1'
        # Let's do all 3 configs for now
        configs = ['A', 'B', 'C']
        for config in configs:
            find_fronts(timestamp, config, version)

    # ###########################
    # Testing

    # Test relative vorticity
    if flg == 102:
        # Output to netcdf on disk
        timestamp = '2012-11-09T12_00_00'
        config_file = './testing_global_v1.yaml'
        field = 'relative_vorticity'
        full_path = llc_io.derived_filename(timestamp, field, version='1')
        out_dir = os.path.dirname(full_path)
        out_file = os.path.basename(full_path)
        # Confg
        cfg = dbof_config.load_config(config_file)
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