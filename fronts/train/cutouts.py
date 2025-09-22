
import os
import pandas
import numpy as np
import h5py

from fronts import io as fronts_io
from fronts.dbof import io as dbof_io

def create_train_set(dbof_json_file:str, config_file:str, 
                     train_tbl:pandas.DataFrame,
                     outfile:str,
                     valid_tbl:pandas.DataFrame=None,
                     test_tbl:pandas.DataFrame=None,
                     clobber:bool=False):
    """ Create the train, valid, test sets

    Args:
        dbof_json_file (str): Path to JSON file with the parameters
        config_file (str): Path to JSON file with the configuration parameters
        train_tbl (pandas.DataFrame): The training table
        outfile (str): Path to output file
        valid_tbl (pandas.DataFrame): The validation table
        test_tbl (pandas.DataFrame): The test table

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

    # Train partition
    print("Working on the train partition")
    
    f.close()
    print(f"Wrote: {outfile}")

    # Create a meta table too

def create_hdf5_dataset(f:h5py.File, tbl:pandas.DataFrame, partition:str,
                        dbof_dict:dict, config:dict):

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

        # Load up the cutouts
        with h5py.File(field_file, 'r') as gf:
            for ss, uid in enumerate(tbl.UID.values):
                input_cutouts[ss, ii, :, :] = gf[uid][:]

