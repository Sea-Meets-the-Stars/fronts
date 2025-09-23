
import os

from fronts.dbof import tables
from fronts.dbof import fields
from fronts.dbof import io as dbof_io
from fronts import io as fronts_io
from fronts.train import tables as ttables
from fronts.train import datasets

from IPython import embed

# DBOF parameter file
dbof_dev_json_file = 'llc4320_dbof_dev.json'

def build_table(clobber:bool=False):
    tables.generate_table(dbof_dev_json_file, clobber=clobber)

def update_table(clobber:bool=False):
    tables.update_fields(dbof_dev_json_file)

def preproc_sst(debug:bool=False, clobber:bool=False):
    fields.preproc_field(dbof_dev_json_file, 'SSTK', debug=debug,
                         clobber=clobber)

def preproc_all(debug:bool=False, clobber:bool=False):

    dbof_dict = fronts_io.loadjson(dbof_dev_json_file)
    for field in dbof_dict['fields']:
        print("Preprocessing field: {}".format(field))
        fields.preproc_field(dbof_dev_json_file, field, debug=debug,
                         clobber=clobber)

def test_for_jake(debug:bool=False):
    datasets.generate_from_dbof(
            dbof_dev_json_file, 
            'DBOF_train_config_jake_test.json',
            path_outdir=os.path.join(os.getenv('DBOF_PATH'), 
                                     'DBOF_dev', 'Training'),
            skip_test=True, skip_valid=True, clobber=True)
    

# #######################################################33
def main(flg:str):
    flg= int(flg)

    # Generate the LLC Table
    if flg == 1:
        #build_table()#clobber=True)

        # Update
        update_table()

        # Test read
        dbof_table = dbof_io.load_main_table(dbof_dev_json_file)
        print("Successfully read table with {} entries".format(len(dbof_table)))
        embed(header='55 of build_me.py')

    # Generate SST
    if flg == 2:
        preproc_sst()#debug=True, clobber=True)

    # Generate all fields
    if flg == 3:
        preproc_all()#debug=True, clobber=True)

    # Test set for Jake
    if flg == 4:
        test_for_jake()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1 # Generate super table
        #flg = 2 # Preproc super table

        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate (with noise)
        #flg += 2 ** 3  # 8 -- Evaluate (without noise)
        pass
    else:
        flg = sys.argv[1]

    main(flg)