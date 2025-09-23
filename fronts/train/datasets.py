""" Functions to generate train/validation datasets """
import os

import numpy as np

import pandas

from wrangler import defs as wr_defs

from fronts import io as fronts_io
from fronts.train import tables as t_tables
from fronts.train import cutouts as t_cutouts

def generate_from_dbof(dbof_json_file:str, config_file:str, 
    path_outdir:str,
    skip_test:bool=False, skip_valid:bool=False,
    clobber:bool=False):

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
        t_cutouts.create_hdf5_cutouts(
            dbof_json_file, config_file, tbl, outfile, clobber=clobber)

        # Metadata
        pp_type = wr_defs.tbl_dmodel['pp_type'][dtype]
        tbl['pp_type'] = pp_type
        all_tables.append(tbl)

    # Write meta
    meta_tbl = pandas.concat(all_tables, ignore_index=True)
    outfile = os.path.join(
            path_outdir,
            f"{config['name']}_meta.parquet")
    meta_tbl.to_parquet(outfile)
    print(f"Wrote {outfile}")