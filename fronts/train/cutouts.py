
import os
import pandas
import numpy as np
import h5py

from wrangler import utils as wr_utils

from fronts import io as fronts_io
from fronts.dbof import io as dbof_io

def create_hdf5_cutouts(dbof_json_file:str, config_file:str, 
                     tbl:pandas.DataFrame, outfile:str, clobber:bool=False):
    """ Create the train, valid, test sets

    Args:
        dbof_json_file (str): Path to JSON file with the parameters
        config_file (str): Path to JSON file with the configuration parameters
        tbl (pandas.DataFrame): The training table
        outfile (str): Path to output file

    Returns:
        None
    """

    # Load up json files
    dbof_dict = fronts_io.loadjson(dbof_json_file)
    config = fronts_io.loadjson(config_file)


    # Open the file
    if os.path.exists(outfile) and not clobber:
        print(f"{outfile} exists.  Use clobber=True to overwrite")
        return

    f = h5py.File(outfile, 'w')
    
    # Inputs
    input_fields = list(config['inputs'].keys())

    input_cutouts = np.zeros((len(tbl), len(input_fields),
                              dbof_dict['spatial']['cutout_size'],
                              dbof_dict['spatial']['cutout_size']), dtype='float32')

    # Loop on fields
    for ii, field in enumerate(input_fields):
        # Load meta table
        meta_tbl = dbof_io.load_meta_table(dbof_dict, field)

        print(f"Working on input field: {field}")
        field_file = dbof_io.field_path(field, dbof_dict, generate_dir=False)

        # Index me
        idx_of_meta = wr_utils.match_ids(tbl.UID.values, meta_tbl.UID.values, require_in_match=True)


        # Load up the cutouts
        fc = h5py.File(field_file, 'r')

        # Loop on date (groups)
        ugroup = np.unique(meta_tbl.group.values[idx_of_meta])
        for group in ugroup:
            print(f"  Working on group: {group}")
            gf = fc[group]
            gidx = (meta_tbl.group.values[idx_of_meta] == group)
            tbl_idx = idx_of_meta[gidx]
            # Grab em
            cutouts = gf['cutouts'][meta_tbl.tidx.values[idx_of_meta[gidx]], ...]

    # Dataset
    dset = f.create_dataset('input', data=input_cutouts)


    # Write inputs

    # Finish
    f.close()
    print(f"Wrote: {outfile} with {len(tbl)} cutouts")