""" Routines to manage the tables for the DBOF project """

import os
import numpy as np

from wrangler.ogcm import llc as wr_llc
from wrangler.tables import utils as tbl_utils
from wrangler.tables import io as tbl_io

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
    llc_table = wr_llc.build_table(dbof_dict['temporal']['freq'],
        init_date=dbof_dict['temporal']['init_date'],
        nperiods=dbof_dict['temporal']['nperiods'],
        resol=dbof_dict['spatial']['resol'],
        minmax_lat=dbof_dict['spatial']['minmax_lat'],
        cutout_size=(dbof_dict['spatial']['cutout_size'], 
                    dbof_dict['spatial']['cutout_size']))

    # Vet
    #embed(header='dbof.tables.generate_table 81')
    assert tbl_utils.vet_main_table(llc_table,
                                    data_model=dbof_defs.tbl_dmodel)

    # Write
    tbl_io.write_main_table(llc_table, tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")
