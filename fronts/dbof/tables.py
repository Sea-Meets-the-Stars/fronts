import os
import numpy as np

import pandas
import json
import h5py


from wrangler.ogcm import llc as wr_llc
from wrangler.tables import utils as tbl_utils
from wrangler.tables import io as tbl_io

from fronts.llc import extract
from fronts.preproc import process
from fronts import io as fronts_io
from fronts.dbof import defs as dbof_defs

from IPython import embed

def tbl_path(dbof_dict:dict, generate_dir:bool=True):
    """
    Constructs the file path for a Parquet table based on the provided dictionary
    and optionally creates the directory if it does not exist.

    Args:
        dbof_dict (dict): A dictionary containing table metadata. It must include
                            the key 'name', which specifies the name of the table.
        generate_dir (bool, optional): If True, ensures that the directory for the
                                        table file exists by creating it if necessary.
                                        Defaults to True.

    Returns:
        str: The full file path to the Parquet table.

    Raises:
        KeyError: If the 'name' key is missing from the `dbof_dict`.
    """
    tbl_file = os.path.join(
        dbof_defs.dbof_path,
        dbof_dict['name'],
        dbof_dict['name']+'.parquet')
    #
    if generate_dir and not os.path.exists(os.path.dirname(tbl_file)):
        os.makedirs(os.path.dirname(tbl_file))
    return tbl_file

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
    tbl_file = tbl_path(dbof_dict, generate_dir=True)

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
        field_size=(dbof_dict['spatial']['field_size'], 
                    dbof_dict['spatial']['field_size']))

    # Vet
    #embed(header='dbof.tables.generate_table 81')
    assert tbl_utils.vet_main_table(llc_table,
                                    data_model=dbof_defs.tbl_dmodel)

    # Write
    tbl_io.write_main_table(llc_table, tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")

def load_table(json_dict:(str|dict)):
    """ Load the table from disk

    Args:
        json_dict (str): Path to JSON file with the parameters or the dict itself

    Returns:
        pandas.DataFrame: The table
    """

    # Read the JSON
    if isinstance(json_dict, str):
        dbof_dict = fronts_io.loadjson(json_dict)
    else:
        dbof_dict = json_dict

    # Load
    tbl_file = tbl_path(dbof_dict, generate_dir=False)
    llc_table = tbl_io.load_main_table(tbl_file)

    return llc_table