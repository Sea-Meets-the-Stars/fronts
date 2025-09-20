
from fronts.dbof import tables
from fronts.dbof import fields

# DBOF parameter file
dbof_dev_json_file = 'llc4320_dbof_dev.json'

def build_table(clobber:bool=False):
    tables.generate_table(dbof_dev_json_file, clobber=clobber)

def preproc_sst():
    fields.preproc_field(dbof_dev_json_file, 'SSTK')

# #######################################################33
def main(flg:str):
    flg= int(flg)

    # Generate the LLC Table
    if flg == 1:
        build_table(clobber=True)
        # Test read
        llc_table = tables.load_table(dbof_dev_json_file)
        print("Successfully read table with {} entries".format(len(llc_table)))
        

    # Generate SST
    if flg == 2:
        preproc_sst()

    # Generate the Training, Validation, Test files
    if flg == 3:
        # A: 
        #   Inputs = Div SST, SST, SSS 
        #   Targets = Divb2 > 1e-14 + >=90%
        #json_file = 'llc4320_sst144_sss40_tvfileA.json'
        # B: 
        #   Inputs = Div SST^2, SSS, SST 
        #   Targets = Divb2 
        # C:
        #   Inputs = Div SST^2, SSS, SST 
        #   Targets = Divb2 normalized by <b>
        # D:
        #   Inputs = Div SST, SSS, SST, SSH 
        #   Targets = Divb, front locations 
        json_file = 'llc4320_sst144_sss40_tvfileD.json'
        gen_trainvalid(json_file, 'LLC4320_SST144_SSS40_SSH15', debug=True)

    # Examine a set of images
    if flg == 10:
        gallery()

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