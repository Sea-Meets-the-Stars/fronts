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
    tbl_file = os.path.join(
        dbof_defs.dbof_path,
        dbof_dict['name'],
        dbof_dict['name']+'.parquet')
    #
    if generate_dir and not os.path.exists(os.path.dirname(tbl_file)):
        os.makedirs(os.path.dirname(tbl_file))
    return tbl_file

def generate_table(json_file:str):
    """ Get the show started by sampling uniformly
    in space and and time

    This is primariliy a wrapper for wr_llc.build_table

    Args:
        json_file (str): Path to JSON file with the parameters
    """

    # Read the JSON
    dbof_dict = fronts_io.loadjson(json_file)

    # Do it
    llc_table = wr_llc.build_table(dbof_dict['temporal']['freq'],
        init_date=dbof_dict['temporal']['init_date'],
        nperiods=dbof_dict['temporal']['nperiods'],
        resol=dbof_dict['spatial']['resol'],
        minmax_lat=dbof_dict['spatial']['minmax_lat'],
        field_size=(dbof_dict['spatial']['field_size'], 
                    dbof_dict['spatial']['field_size']))

    # Vet
    embed(header='dbof.tables.generate_table 94')
    assert tbl_utils.vet_main_table(llc_table,
                                    data_model=dbof_defs.tbl_dmodel)

    # Write
    tbl_file = tbl_path(dbof_dict, generate_dir=True)
    tbl_io.write_main_table(llc_table, tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")


def preproc(extract_file: str, debug: bool = False):
    """
    Preprocesses super-resolution data for SST, SSS, and Divb2 fields.

    This function processes a table of data, extracts specified fields, applies preprocessing
    steps, and writes the results to an HDF5 file. It also updates the table with metadata
    and handles preprocessing failures.

    Args:
        extract_file (str): Path to the file containing the data to be extracted.
        debug (bool, optional): If True, processes only a subset of the data for debugging 
            purposes and writes output to a test file. Defaults to False.

    Raises:
        AssertionError: If the main table fails the vetting process.

    Notes:
        - The function uses a predefined extraction dictionary (`extract_dict`) to specify
            the fields and their preprocessing parameters.
        - The output table is written to the `super_tbl_file` unless in debug mode, where
            it is written to a local test file.
        - The function assumes the existence of several external dependencies, including
            `process.prep_table_for_preproc`, `extract.preproc_field`, and `fronts_io.write_main_table`.

    Outputs:
        - An HDF5 file containing the preprocessed fields.
        - An updated table with metadata and preprocessing status.
    """

    outfile = super_preproc_file 

    # Load the table
    llc_table = pandas.read_parquet(super_tbl_file)

    # Debug?
    if debug:
        llc_table = llc_table.iloc[:100].copy()
        outfile = os.path.join(local_out_path, 'LLC4320_SST144_SSS40_super_test.h5')

    # Extract dict
    # Load JSON
    #with open(preproc_file, 'r') as infile:
    #    preproc_dict = json.load(infile)


    #extract_dict = {'fields': ['SSS'],
    #extract_dict = {'fields': ['Divb2'],
    extract_dict = {'fields': ['SST','SSS','Divb2'],
             'field_size': 64,
             'pdicts': 
                 {
                     'SST': 
                        {
                        'fixed_km': 144.,
                        'field_size': 64,
                        "quality_thresh": 2,
                        "nrepeat": 1,
                        "downscale": False,
                        "inpaint": False,
                        "median": False,
                        "only_inpaint": False
                        }
                    ,
                     'SSS': 
                        {
                        'fixed_km': 144.,
                        'field_size': 64,
                        'smooth_km': 40., 
                        "quality_thresh": 2,
                        "nrepeat": 1,
                        "downscale": False,
                        "inpaint": False,
                        "de_mean": False,
                        "median": False,
                        "only_inpaint": False
                        }
                    ,
                     'Divb2': 
                        {
                        'fixed_km': 144.,
                        'field_size': 64,
                        'dx': 144./64,
                        }
                    ,
                 }
             }

    # Prep LLC Table
    llc_table = process.prep_table_for_preproc(
        llc_table, extract_file, field_size=
        (extract_dict['field_size'], extract_dict['field_size']))


    # Open HDF5 file
    f = h5py.File(outfile, 'w')
    for field in extract_dict['fields']:
        # Preprocess
        llc_table, success, pp_fields, meta = extract.preproc_field(
            llc_table, field, extract_dict['pdicts'][field],
            fixed_km=extract_dict['pdicts'][field]['fixed_km'],
            n_cores=10, dlocal=True,
            test_failures=False,
            test_process=False, override_RAM=True)

        # Write data
        pp_fields = np.array(pp_fields).astype(np.float32)
        f.create_dataset(field, data=pp_fields)

        # Add meta
        for key in meta.keys():
            llc_table[field+key] = meta[key]

        # Deal with failures in Table
        if np.any(~success):
            print(f"Failed to preprocess some {field} fields")
            fail = np.where(~success)[0]
            llc_table.loc[fail,'pp_type'] = -999
        # Good
        good = success & (llc_table['pp_type'] != -999)
        llc_table.loc[np.where(good)[0],'pp_type'] = 0
        #
        del pp_fields

    # Close
    f.close()    

    # Write table
    assert catalog.vet_main_table(llc_table)
    if not debug:
        fronts_io.write_main_table(llc_table, super_tbl_file, to_s3=False)
    else:
        embed(header='preproc_super 118')
        tbl_file = os.path.join(local_out_path, 'blah')
        fronts_io.write_main_table(llc_table, tbl_file, to_s3=False)
