
import os
import pandas
import numpy as np
import h5py

from wrangler import utils as wr_utils

from fronts import io as fronts_io
from fronts.dbof import io as dbof_io
from fronts.dbof import defs as dbof_defs

from IPython import embed

def create_hdf5_cutouts(dbof_json_file:str, config_file:str, 
                     tbl:pandas.DataFrame, outfile:str, 
                     clobber:bool=False):
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
    
    # Loop on inputs and targets
    for partition in ['inputs', 'targets']:
        # Inputs
        fields = config[partition]
        if len(fields) == 0:
            continue

        cutouts = np.zeros((len(tbl), len(fields),
                              dbof_dict['spatial']['cutout_size'],
                              dbof_dict['spatial']['cutout_size']), dtype='float32')

        # Loop on input fields
        for ii, field in enumerate(fields):
            # Load meta table
            meta_tbl = dbof_io.load_meta_table(dbof_dict, field)

            # Match tbl to meta on UID
            meta_idx = wr_utils.match_ids(tbl.UID.values, meta_tbl.UID.values, 
                                          require_in_match=True)
            # Cut down to those in tbl ordered 
            cutout_tbl = meta_tbl.iloc[meta_idx]

            print(f"Working on input field: {field}")
            field_file = dbof_io.field_path(field, dbof_dict, generate_dir=False)

            # Load up the cutouts
            fc = h5py.File(field_file, 'r')

            # Loop on date (groups)
            ugroup = np.unique(cutout_tbl.group.values)
            for group in ugroup:
                # Grab all the cutouts; memory intensive but probably fastest
                all_cutouts = fc[group][:]
                # Fill in
                tidx = wr_utils.match_ids(tbl.UID.values, cutout_tbl.UID.values)
                cutouts[tidx, ii] = all_cutouts[cutout_tbl.gidx.values]
            fc.close()

        # Dataset
        dset = f.create_dataset(partition, data=cutouts)
        for field in fields:
            dset.attrs[field] = f"field: {field}, units: {dbof_defs.fields_dmodel[field]['units']}"

    # Finish
    f.close()
    print(f"Wrote: {outfile} with {len(tbl)} cutouts")