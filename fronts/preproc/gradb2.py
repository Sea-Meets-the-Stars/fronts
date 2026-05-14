""" Methods related to generating and processing a gradb2 field from LLC """

import os

import yaml

from dbof.cli import generate_global
from dbof.cli import zarr_to_netcdf

from fronts.llc import io as llc_io


def generate_gradb2(timestamp: str, config_file: str, version:str=None,
    run_id: str = None, field: str = 'gradb2', clobber: bool = False,
    create_zarr: bool = False):
    """Generate the gradb2 field for the given config file.

    Args:
        timestamp (str): Timestamp of the data to process.
        version (str): Version of the data to process.
        config_file (str): Path to the YAML config file.
        run_id (str, optional): Override the run_id in the config YAML.
        field (str): Field name to extract. Defaults to 'gradb2'.
        create_zarr (bool): Create the zarr store. Defaults to False.
        clobber (bool): Overwrite existing output. Defaults to False.
    """
    out_file = llc_io.derived_filename(timestamp, field, version=version)
    if os.path.isfile(out_file) and not clobber:
        print(f"gradb2 file {out_file} exists and clobber is False. Returning")
    else:
        # Create the zarr
        if create_zarr:
            generate_global.main(config_file, subset='frontal_structure',
                only_these_features=['gradb2'], run_id=run_id)
        # Create the netcdf
        llc_io.zarr_to_nc(timestamp, config_file, 'frontal_structure',
            field, run_id=run_id, version=version)
