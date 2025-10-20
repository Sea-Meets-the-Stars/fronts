""" Functions to generate train/validation datasets """
import os

import numpy as np

import pandas

from wrangler import defs as wr_defs
from wrangler.tables import utils as tbl_utils

from fronts import io as fronts_io
from fronts.train import tables as t_tables
from fronts.train import cutouts as t_cutouts
from fronts.dbof import defs as dbof_defs

def generate_from_dbof(dbof_json_file:str, config_file:(str|dict), 
    path_outdir:str, skip_test:bool=False, skip_valid:bool=False,
    clobber:bool=False):
    """
    Generate training, validation, and test datasets from a DBoF JSON file.
    This function processes a DBoF (Deep Bag of Features) JSON file and a 
    configuration file to generate datasets for training, validation, and 
    testing. The datasets are saved as HDF5 files, and metadata is saved 
    as a Parquet file.
    Args:
        dbof_json_file (str): Path to the DBoF JSON file containing the 
            input data.
        config_file (str | dict): Path to the configuration JSON file or 
            a dictionary containing configuration details.
        path_outdir (str): Directory where the output files will be saved.
        skip_test (bool, optional): If True, skip generating the test dataset. 
            Defaults to False.
        skip_valid (bool, optional): If True, skip generating the validation 
            dataset. Defaults to False.
        clobber (bool, optional): If True, overwrite existing files. 
            Defaults to False.
    Raises:
        AssertionError: If the generated metadata table fails validation 
            against the data model.
    Outputs:
        - HDF5 files for training, validation, and test datasets (if not skipped).
        - A Parquet file containing metadata for all datasets.
    Notes:
        - The function uses `fronts_io` to load JSON files and `t_tables` 
          to generate tables for training, validation, and testing.
        - Metadata is validated using `tbl_utils.vet_main_table` against 
          the data model defined in `dbof_defs.tbl_dmodel`.
        - If the output files already exist and `clobber` is False, the 
          function will not overwrite them.
    """

    # Load up json files
    dbof_dict = fronts_io.loadjson(dbof_json_file)
    config = fronts_io.loadjson(config_file)

    # Generate the tables
    train_tbl, valid_tbl, test_tbl = \
        t_tables.dbof_gen_tvt(dbof_json_file, config_file)

    all_tables = []
    # Loop on types
    for tbl, dtype in zip([train_tbl, valid_tbl, test_tbl], 
                          ['train', 'valid', 'test']):
        # Mainly for development
        if skip_test and dtype == 'test':
            continue
        if skip_valid and dtype == 'valid':
            continue


        # Generate cutout outfile
        outfile = os.path.join(
            path_outdir,
            f"{config['name']}_{dtype}.h5")
        print(f"Generating {dtype} set with {len(tbl)} entries to {outfile}")
        t_cutouts.create_hdf5_cutouts(
            dbof_json_file, config_file, tbl, outfile, clobber=clobber)

        # Metadata
        pp_type = wr_defs.tbl_dmodel['pp_type'][dtype]
        tbl['pp_type'] = pp_type
        all_tables.append(tbl)

    # Vet
    meta_tbl = pandas.concat(all_tables, ignore_index=True)
    assert tbl_utils.vet_main_table(meta_tbl,
                                    data_model=dbof_defs.tbl_dmodel)
    
    # Write meta
    outfile = os.path.join(path_outdir, f"{config['name']}_meta.parquet")
    if os.path.exists(outfile) and not clobber:
        print(f"{outfile} exists.  Use clobber=True to overwrite")
        return

    meta_tbl.to_parquet(outfile)
    print(f"Wrote {outfile}")

    return meta_tbl