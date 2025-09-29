
import numpy as np

from fronts.finding import dev as finding_dev
from fronts.dbof import io as dbof_io

def explore_thin(nexamples:int=100, divb2_rng=(-15., -13.),
                 outdir:str='plots/'):
    dbof_dev_file = '../fronts/runs/dbof/dev/llc4320_dbof_dev.json'

    # Thin/weak params
    thin_weak_params = {
        'wndw': 40,
        'rm_weak': 1e-15,
        'dilate': False,
        'thin': True,
        'min_size': 5
    }

    # Load
    dbof_tbl = dbof_io.load_main_table(dbof_dev_file)
    dbof_divb2_tbl = dbof_io.load_meta_table(dbof_dev_file, 'Divb2')

    # Draw random examples from divb2 range
    divb2_rand = np.random.uniform(divb2_rng[0], divb2_rng[1], 
                                   nexamples)
    
    # Generate fronts

def test_algorithms():

    # Vanilla
    #finding_dev.run_a_test('vanilla')#, tst_idx=(0,500,700))

    # Add thin 
    #finding_dev.run_a_test('thin')#, tst_idx=(0,500,700))

    # Remove weak
    #finding_dev.run_a_test('rm_weak')#, tst_idx=(0,500,700))

    # Dilate
    finding_dev.run_a_test('rm_weak-thin-dilate')#, tst_idx=(0,500,700))

if __name__ == "__main__":

    #test_algorithms()

    explore_thin(nexamples=2)