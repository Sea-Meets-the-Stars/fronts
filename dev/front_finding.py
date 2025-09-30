
import os
import numpy as np

from fronts.finding import dev as finding_dev
from fronts.dbof import io as dbof_io
from fronts.dbof import utils as dbof_utils
from fronts.dbof import params as dbof_params

from IPython import embed

def explore_thin(nexamples:int=100, divb2_rng=(-15., -13.),
                 outdir:str='plots/'):
    dbof_dev_file = '../fronts/runs/dbof/dev/llc4320_dbof_dev.json'

    # Thin/weak params
    front_params = dbof_params.thin_weak_params

    # Load
    dbof_tbl = dbof_io.load_main_table(dbof_dev_file)
    dbof_divb2_tbl = dbof_io.load_meta_table(dbof_dev_file, 'Divb2')

    # Set the seed
    np.random.seed(42)
    # Draw random examples from divb2 range
    divb2_rand = np.random.uniform(divb2_rng[0], divb2_rng[1], 
                                   nexamples)
    
    # Generate fronts
    for tt in range(0,nexamples,2):
        # Loop on 2
        all_b, all_sst, all_divb2, all_fronts = [], [], [], []
        UIDs = []
        for kk in range(2):
            ss = tt + kk
            # Find closest Divb2
            idx = np.argmin(np.abs(10**divb2_rand[ss] - dbof_divb2_tbl.p90.values))
            UID = dbof_divb2_tbl.UID.values[idx]
            # Grab fields
            field_data = dbof_utils.grab_fields(dbof_dev_file, 'all', UID)
            # Save em

            all_divb2.append(field_data['Divb2'])
            all_sst.append(field_data['SSTK'])
            all_b.append(field_data['b'])
            UIDs.append(UID)

            # Calculate fronts
            fronts = finding_dev.algorithms.fronts_from_divb2(
                        field_data['Divb2'], **front_params)
            all_fronts.append(fronts)

        # Generate figure
        outfile = os.path.join(outdir, 
                               f'fronts_thinwk_{UIDs[0]}_{UIDs[1]}.png')
        finding_dev.front_fig3(outfile, all_fronts, all_divb2, all_sst, all_b,
                               title=f'UIDs: {UIDs[0]}, {UIDs[1]}')


def test_algorithms():

    # Vanilla
    #finding_dev.run_a_test('vanilla')#, tst_idx=(0,500,700))

    # Add thin 
    #finding_dev.run_a_test('thin')#, tst_idx=(0,500,700))

    # Remove weak
    #finding_dev.run_a_test('rm_weak')#, tst_idx=(0,500,700))

    # Dilate
    finding_dev.run_a_test('rm_weak-thin-dilate')#, tst_idx=(0,500,700))

def test_fig4():

    # Load up
    cutouts, tbl = finding_dev.load_test_data()

    # Indices
    tst_idx = (0, 500)#, 700)

    all_b, all_sst, all_divb2, all_fronts = [], [], [], []
    all_divsst = []
    for idx in tst_idx:

        # Images
        div_sst, sst, sss, Divb2 = finding_dev.parse_idx(cutouts, idx)

        all_divb2.append(Divb2)
        all_sst.append(sst)
        all_b.append(sss)
        all_divsst.append(div_sst)

        # Find fronts
        fronts = finding_dev.algorithms.fronts_from_divb2(
            Divb2, wndw=40, thin=True, rm_weak=1e-15, dilate=True)

        all_fronts.append(fronts)

    # Plot
    outfile = f'fronts_fig4.png'
    finding_dev.front_fig4(outfile, all_fronts, all_divb2, 
                           all_sst, all_b, all_divsst,
                           title=f'Figure 4 Example')

if __name__ == "__main__":

    #test_algorithms()

    #explore_thin(nexamples=2)

    #test_fig4()