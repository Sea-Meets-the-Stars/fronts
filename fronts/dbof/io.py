
import os 

import numpy as np

from wrangler.tables import io as tbl_io

from fronts import io as fronts_io
from fronts.dbof import defs as dbof_defs 

def get_sdate(udate:np.datetime64):
    """ Get a string date from a numpy datetime64

    Args:
        udate (np.datetime64): Date to convert
    Returns:
        str: Date in YYYYMMDDTHH format
    """
    return str(udate)[:13]

def field_path(field:str, dbof_dict:dict, generate_dir:bool=True,
               meta:bool=False):
    """
    Constructs the file path for a field HDF5 file based on the provided field name
    and dictionary, and optionally creates the directory if it does not exist.

    Args:
        field (str): The name of the field (e.g., 'SSTK', 'SSS', 'Divb2').
        dbof_dict (dict): A dictionary containing metadata. It must include
    """

    # Construct
    field_file = os.path.join(
        dbof_defs.dbof_path,
        dbof_dict['name'],
        'Fields', 
        dbof_dict['name']+'_'+field+'.h5')

    if meta:
        field_file = field_file.replace('.h5', '_meta.parquet')

    # Create dir?
    if generate_dir and not os.path.exists(os.path.dirname(field_file)):
        os.makedirs(os.path.dirname(field_file))
    return field_file


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

def load_main_table(dbof_json_dict:(str|dict)):
    """ Load the table from disk

    Args:
        dbof_json_dict (str): Path to JSON file with the parameters or the dict itself
            or the table filename itself

    Returns:
        pandas.DataFrame: The table
    """
    # Parquet
    if isinstance(dbof_json_dict, str) and dbof_json_dict.endswith('.parquet'):
        tbl_file = dbof_json_dict
    else:
        # Read the JSON
        dbof_dict = fronts_io.loadjson(dbof_json_dict) if isinstance(dbof_json_dict, str) else dbof_json_dict
        tbl_file = tbl_path(dbof_dict, generate_dir=False)
    # Load
    dbof_table = tbl_io.load_main_table(tbl_file)

    return dbof_table

def load_meta_table(dbof_json_dict:(str|dict), field:str):
    """ Load the meta table from disk

    Args:
        json_dict (str): Path to JSON file with the parameters or the dict itself
        field (str): Field name

    Returns:
        pandas.DataFrame: The table
    """

    # Read the JSON
    dbof_dict = fronts_io.loadjson(dbof_json_dict) if isinstance(dbof_json_dict, str) else dbof_json_dict

    # Load
    meta_tbl_file = field_path(field, dbof_dict, meta=True)

    if not os.path.exists(meta_tbl_file):
        print(f"{meta_tbl_file} does not exist")
        return None
    meta_table = tbl_io.load_main_table(meta_tbl_file)

    return meta_table