
import os
from importlib.resources import files

from fronts.dbof import utils as dbof_utils
from fronts.dbof import io as dbof_io

from IPython import embed

dbof_dev_file = os.path.join(str(files('fronts')._paths[0]), 'runs', 'dbof', 'dev',
                        'llc4320_dbof_dev.json')

def test_find_entry():
    """
    Test the find_entry function from dbof_utils
    """
    # Grab JSON file
    #embed(header="14 @dbof_utils.find_entry")                        

    # Load table
    dbof_tbl = dbof_io.load_main_table(dbof_dev_file)

    # Test case
    lat = 56.752953
    lon = -170.447922
    datetime = '2011-09-30'

    idx = dbof_utils.find_entry(dbof_dev_file, 
                                {'lat':lat, 'lon':lon, 'datetime':datetime},
                                debug=False)

    assert idx == 0
    #print("Found index: {}".format(idx))
    #embed(header="@dbof_utils.find_entry")                        

tidx = 1000
field_data = dbof_utils.grab_fields(dbof_dev_file, tidx, 'all')
embed(header='38 of tests')