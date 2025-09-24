""" Utility functions for interfacing with DBOF """

import numpy as np

from fronts.dbof import io as dbof_io

def find_entry(dbof_json_dict:(str|dict), sdict:dict):

    # Load up main table
    dbof_table = dbof_io.load_main_table(dbof_json_dict)

    # Now match on sdict
    match = np.zeros(len(dbof_table), dtype=bool)

    for key in sdict.keys():
        if key in dbof_table.keys():
            match = match & (dbof_table[key].values == sdict[key])
        else:
            raise KeyError("Key {} not in table".format(key))

    if np.sum(match) == 0:
        print("No match found. Returning -1")
        return -1
    elif np.sum(match) > 1:
        raise ValueError("Multiple matches found")
    else:
        return dbof_table[match].index.values[0]
    