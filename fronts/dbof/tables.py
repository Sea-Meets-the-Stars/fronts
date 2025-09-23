""" Routines to manage the tables for the DBOF project """

import os
import numpy as np

from wrangler.ogcm import llc as wr_llc
from wrangler.tables import utils as tbl_utils
from wrangler.tables import io as tbl_io
from wrangler import utils as wr_utils

from fronts import io as fronts_io
from fronts.dbof import defs as dbof_defs
from fronts.dbof import io as dbof_io

from IPython import embed

def generate_table(json_file:str, clobber:bool=False):
    """ Get the show started by sampling uniformly
    in space and and time

    This is primariliy a wrapper for wr_llc.build_table

    The table is written to disk at completion

    Args:
        json_file (str): Path to JSON file with the parameters
        clobber (bool, optional):
    """

    # Read the JSON
    dbof_dict = fronts_io.loadjson(json_file)

    # Outfile
    tbl_file = dbof_io.tbl_path(dbof_dict, generate_dir=True)

    # Clobber?
    if os.path.exists(tbl_file) and not clobber:
        print(f"{tbl_file} exists.  Use clobber=True to overwrite")
        return

    # Do it
    dbof_table = wr_llc.build_table(dbof_dict['temporal']['freq'],
        sampling=dbof_dict['spatial']['sampling'],
        init_date=dbof_dict['temporal']['init_date'],
        nperiods=dbof_dict['temporal']['nperiods'],
        resol=dbof_dict['spatial']['resol'],
        minmax_lat=dbof_dict['spatial']['minmax_lat'],
        cutout_size=(dbof_dict['spatial']['cutout_size'], 
                    dbof_dict['spatial']['cutout_size']))

    # Vet
    #embed(header='dbof.tables.generate_table 81')
    assert tbl_utils.vet_main_table(dbof_table,
                                    data_model=dbof_defs.tbl_dmodel)

    # Add attributes
    embed(header='dbof.tables.generate_table 91')
    for key in ['name', 'description', 'fields', 'version',
                'model']:
        dbof_table.attrs[key] = dbof_dict[key]

    # Write
    tbl_io.write_main_table(dbof_table, tbl_file)

    print(f"Wrote: {tbl_file} with {len(dbof_table)} unique cutouts.")
    print("All done with init")

def update_fields(json_file:str):

    # Read the JSON
    dbof_dict = fronts_io.loadjson(json_file)

    # Table file
    tbl_file = dbof_io.tbl_path(dbof_dict, generate_dir=True)

    # Load
    dbof_tbl = tbl_io.load_main_table(tbl_file)

    # Search for fields
    for fields in dbof_defs.fields_dmodel.keys():
        # Load up the field meta if not there
        meta_tbl = dbof_io.load_meta_table(dbof_dict, fields)
        if meta_tbl is None:
            continue

        embed(header='dbof.tables.update_fields 86')
        # Match on UID
        meta_idx = wr_utils.match_ids(dbof_tbl.UID.values, 
                                      meta_tbl.UID.values, 
                                      require_in_match=False)