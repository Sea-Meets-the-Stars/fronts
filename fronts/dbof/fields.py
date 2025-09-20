
import os
import numpy as np

import pandas
import h5py

from wrangler.extract import ogcm as wr_ex_ogcm
from wrangler.ogcm import llc as wr_llc
from wrangler.tables import io as tbl_io
from wrangler.tables import utils as tbl_utils

from fronts import io as fronts_io
from fronts.dbof import tables
from fronts.dbof import io as dbof_io
from fronts.dbof import defs as dbof_defs

from IPython import embed

def preproc_field(json_file:str, field:str, clobber:bool=False, debug:bool=False):
    """
    Preprocesses super-resolution data for SST, SSS, and Divb2 fields.

    This function processes , extracts specified fields, applies preprocessing
    steps, and writes the results to an HDF5 file. It also updates the table with metadata
    and handles preprocessing failures.

    Args:
        extract_file (str): Path to the file containing the data to be extracted.
        clobber (bool, optional): If True, overwrites existing files. Defaults to False.
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

    # Read the JSON
    dbof_dict = fronts_io.loadjson(json_file)
    if field not in dbof_dict['fields'].keys():
        raise IOError(f"Field {field} not in {json_file}")

    # Load the full table
    llc_table = dbof_io.load_main_table(dbof_dict)
    llc_table[field] = False

    # Setup for dates
    uni_dates = np.unique(llc_table.datetime)

    # Load up coords
    coords_ds = wr_llc.load_coords()

    # Modify the pdicts to include cutout_size
    dbof_dict['fields'][field]['pdict']['cutout_size'] = dbof_dict['spatial']['cutout_size']

    # Setup the output HDF5 file
    field_file = dbof_io.field_path(field, dbof_dict, generate_dir=True)
    meta_file = dbof_io.field_path(field, dbof_dict, meta=True)
    if os.path.exists(field_file) and not clobber:
        print(f"{field_file} exists.  Use clobber=True to overwrite")
        return
    f = h5py.File(field_file, 'w')
    
    # Loop over unique dates
    all_meta = []
    for ss, udate in enumerate(uni_dates):
        if debug and ss > 1:
            break
        # Cut down the table
        this_date = llc_table.datetime == udate
        date_table = llc_table[this_date]
        print(f"Processing {udate} with {len(date_table)} cutouts")

        # Do it
        success, pp_fields, imeta, filename = wr_ex_ogcm.preproc_datetime(
            date_table, field, udate, dbof_dict['fields'][field]['pdict'],
            #field_size=(dbof_dict['spatial']['cutout_size'], dbof_dict['spatial']['cutout_size']),
            fixed_km=dbof_dict['spatial']['fixed_km'],
            n_cores=10,
            coords_ds=coords_ds,
            test_failures=False,
            test_process=False,
            debug=False)

        print("Writing {} successful pre-processed fields".format(np.sum(success)))

        # Write data, by date
        pp_fields = np.array(pp_fields[success]).astype(np.float32)
        # Convert date to YYYY-MM-DDTHH
        sdate = dbof_io.get_sdate(udate)
        dset = f.create_dataset(sdate, data=pp_fields)
        _ = f.create_dataset(sdate+'_UID', data=date_table.UID.values[success])

        # Add attriutes of row, col, lon, lat, UID
        dset.attrs['json_file'] = json_file

        # Add index to meta
        imeta['tidx'] = date_table.index.values
        # And UID
        imeta['UID'] = date_table.UID.values
        
        # Write meta too
        all_meta.append(imeta.iloc[success])
        wr_meta = pandas.concat(all_meta, ignore_index=True)
        wr_meta.to_parquet(meta_file)

        # Update llc_table
        if np.any(success):
            llc_table.loc[date_table.index.values[success], field] = True
        # Write it out??
        
    f.close()

    embed(header='125 of fronts.dbof.fields.py')
    # Write table
    assert tbl_utils.vet_main_table(llc_table,
                                  data_model=dbof_defs.tbl_dmodel)
    if not debug:
        tbl_file = dbof_io.tbl_path(dbof_dict) 
        tbl_io.write_main_table(llc_table, tbl_file)


def old_stuff():

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