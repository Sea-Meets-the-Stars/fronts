
import os
import numpy as np

import h5py
import pandas

from wrangler.ogcm import utils as wr_ogcm_utils

from fronts.finding import dev as finding_dev
from fronts.finding import run as finding_run
from fronts.finding import params as finding_params
from fronts.dbof import io as dbof_io
from fronts.dbof import utils as dbof_utils
from fronts.properties import measure as fprop_measure

from IPython import embed
    
b_front_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'Training_Sets', 
                     'LLC4320_SST144_SSS40_fronts.h5')
dbof_dev_file = '../fronts/runs/dbof/dev/llc4320_dbof_dev.json'

def explore_thin(nexamples:int=100, divb2_rng=(-15., -13.),
                 outdir:str='plots/', tbl_file:str=None,
                 clobber:bool=False):

    # Thin/weak params
    front_params = finding_params.thin_weak_params

    # Load
    dbof_tbl = dbof_io.load_main_table(dbof_dev_file)
    dbof_divb2_tbl = dbof_io.load_meta_table(dbof_dev_file, 'Divb2')

    # Set the seed
    np.random.seed(42)
    # Draw random examples from divb2 range
    divb2_rand = np.random.uniform(divb2_rng[0], divb2_rng[1], 
                                   nexamples)
    sv_UIDs = []

    # Generate fronts
    for ss in range(0,nexamples):

        # Find closest Divb2
        idx = np.argmin(np.abs(10**divb2_rand[ss] - dbof_divb2_tbl.p90.values))
        UID = dbof_divb2_tbl.UID.values[idx]
        sv_UIDs.append(UID)

        # Outfile
        outfile = os.path.join(outdir, f'fronts_thinwk_{UID}.png')
        if (not clobber) and os.path.isfile(outfile):
            print(f'File {outfile} exists, skipping')
            continue

        # Main table
        imain = np.where(dbof_tbl.UID == UID)[0][0]

        # Grab fields
        field_data = dbof_utils.grab_fields(dbof_dev_file, 'all', UID)

        # Calculate fronts
        fronts = finding_dev.algorithms.fronts_from_divb2(
                    field_data['Divb2'], **front_params)

        # Title
        title = f'UID: {UID}, date: {dbof_divb2_tbl.iloc[idx].group[:-3]}, '
        title += f'lat={dbof_tbl.iloc[imain].lat:0.2f}, lon={dbof_tbl.iloc[imain].lon:0.2f}'

        # Generate figure
        finding_dev.front_fig6(outfile, fronts, field_data['Divb2'],
                                   field_data['SSTK'], field_data['b'], 
                                   field_data['DivSST2'],
                                   field_data['SSS'], field_data['DivSSS2'],
                               title=title)

    # Table?
    if tbl_file is not None:
        df = pandas.DataFrame({'UID': sv_UIDs, 
                               'log10_p90_Divb2': divb2_rand})
        df.to_csv(tbl_file, index=False)
        print(f'Wrote {tbl_file}')

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
            Divb2, wndw=40, thin=True, rm_weak=1e-15, 
            dilate=False)

        all_fronts.append(fronts)

    # Plot
    outfile = f'fronts_fig4.png'
    finding_dev.front_fig4(outfile, all_fronts, all_divb2, 
                           all_sst, all_b, all_divsst,
                           title=f'Figure 4 Example')

def test_many_cutouts():

    # Load up
    cutouts, tbl = finding_dev.load_test_data()
    Divb2 = cutouts['targets'][:, 0, 0, ...]

    # Convert to list 
    Divb2 = [item for item in Divb2]

    # Do it
    front_params = finding_params.thin_weak_params
    flabels = finding_run.many_cutouts(Divb2, front_params)

    flabels = np.stack(flabels)

    # Write
    with h5py.File(b_front_file, 'w') as f:
        f.create_dataset('fronts', data=flabels.astype(int))

if __name__ == "__main__":

    #test_algorithms()

    # Exploring 100 FFF examples
    explore_thin(nexamples=100, tbl_file='exploring_thin.csv')

    #test_fig4()

    # Test measuring fronts in many cutouts
    #test_many_cutouts()
