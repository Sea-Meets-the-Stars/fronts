""" Utility functions for interfacing with DBOF """

import os
import numpy as np
import h5py

from fronts import io as fronts_io
from fronts.dbof import io as dbof_io

from IPython import embed

def find_entry(dbof_json_dict:(str|dict), sdict:dict, debug:bool=False):
    """
    Find an entry in a DBOF (Database of Oceanographic Fronts) table that matches the specified criteria.
    Parameters:
    -----------
    dbof_json_dict : str | dict
        The DBOF table, either as a JSON file path (str) or a dictionary (dict).
    sdict : dict
        A dictionary containing the search criteria. Keys should correspond to column names in the DBOF table,
        and values represent the values to match.
    debug : bool, optional
        If True, prints debug information about the matching process. Default is False.
    Returns:
    --------
    int
        The index of the matching entry in the DBOF table.
        Returns -1 if no match is found.
    Raises:
    -------
    KeyError
        If a key in `sdict` is not found in the DBOF table.
    ValueError
        If multiple matches are found for the given criteria.
    Notes:
    ------
    - For the 'datetime' key in `sdict`, the value is converted to `numpy.datetime64` if it is not already in that format.
    - The function uses logical AND to combine matches across multiple keys in `sdict`.
    Debugging:
    ----------
    When `debug` is True, the function prints:
    - The key being matched and the value being searched for.
    - The number of matches found for the current key.
    - The total number of matches after combining with previous keys.
    """

    # Load up main table
    dbof_table = dbof_io.load_main_table(dbof_json_dict)

    # Now match on sdict
    match = np.ones(len(dbof_table), dtype=bool)

    for key in sdict.keys():
        if key not in dbof_table.keys():
            raise KeyError("Key {} not in table".format(key))
        # Match
        if key == 'datetime':
            # Convert to numpy datetime64
            if not isinstance(sdict[key], np.datetime64):
                sdict[key] = np.datetime64(sdict[key])
        match = match & (dbof_table[key].values == sdict[key])
        #
        if debug:
            print(f'Key: {key}, looking for {sdict[key]}')
            print(f'  Found {np.sum(dbof_table[key].values == sdict[key])} matches')
            print(f'  Total matches now {np.sum(match)}')
            embed(header="in dbof.utils.find_entry 31")
            

    if np.sum(match) == 0:
        print("No match found. Returning -1")
        return -1
    elif np.sum(match) > 1:
        raise ValueError("Multiple matches found")
    else:
        return dbof_table[match].index.values[0]
    
def grab_fields(dbof_json_dict:(str|dict), fields:(list|str), UID:int):
    """
    Extracts specified fields from a DBOF (Dynamic Bayesian Ocean Fronts) dataset 
    for a given unique identifier (UID).

    Args:
        dbof_json_dict (str | dict): The path to a JSON file or a dictionary 
            containing the DBOF dataset metadata.
        fields (list | str): A list of field names to extract, or the string 'all' 
            to extract all available fields.
        UID (int): The unique identifier for which the fields are to be extracted.

    Returns:
        dict: A dictionary where the keys are field names and the values are the 
        corresponding data cutouts.

    Notes:
        - If a field is not found in the metadata or the field file does not exist, 
            a warning message is printed, and the field is skipped.
        - The function assumes the presence of specific helper functions 
            (`fronts_io.loadjson`, `dbof_io.load_meta_table`, `dbof_io.field_path`) 
            and external libraries (`numpy`, `h5py`, `os`).

    Raises:
        None: The function handles missing fields and files gracefully by skipping 
        them and printing warnings.

    Example:
        field_data = grab_fields("dbof_metadata.json", ["temperature", "salinity"], 12345)
    """

    # Load the json dict 
    dbof_dict = fronts_io.loadjson(dbof_json_dict)

    # Load up main table
    print(f"Grabbing fields for UID: {UID}")

    # Fields
    if isinstance(fields, str) and fields == 'all':
        fields = dbof_dict['fields']

    # Loop on fields
    field_data = {}
    for field in fields:
        # Load up the field meta if not there
        meta_tbl = dbof_io.load_meta_table(dbof_dict, field)
        if meta_tbl is None:
            print(f"Field {field} not found")
            continue

        # Match on UID
        mt = np.where(UID == meta_tbl.UID.values)[0]
        if len(mt) == 0:
            print(f"Field {field} not found for UID {UID}")
            continue

        # Field file
        field_file = dbof_io.field_path(field, dbof_dict) 
        if not os.path.exists(field_file):
            print(f"Field file {field_file} not found")
            continue 

        # Do it
        fc = h5py.File(field_file, 'r')
        group = str(meta_tbl.group.values[mt[0]])
        gidx = int(meta_tbl.gidx.values[mt[0]])
        cutout = fc[group][gidx]
        fc.close()
        field_data[field] = cutout

    # Return
    return field_data