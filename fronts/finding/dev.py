""" Code to develop front finding algorithms """

import os

import numpy as np
import h5py
import pandas

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

from wrangler.plotting import cutout

from fronts.finding import algorithms

def parse_idx(b_train, idx):
    # Parase
    div_sst = b_train['inputs'][idx, 0, 0, ...]
    sst = b_train['inputs'][idx, 2, 0, ...]
    sss = b_train['inputs'][idx, 1, 0, ...]
    #
    Divb2 = b_train['targets'][idx, 0, 0, ...]
    #
    return div_sst, sst, sss, Divb2

def load_test_data():
    # files
    b_tblfile = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'Training_Sets', 
                     'LLC4320_SST144_SSS40_trainB.parquet')
    b_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'Training_Sets', 
                     'LLC4320_SST144_SSS40_trainB.h5')

    # Load
    b_train = h5py.File(b_file, 'r')

    b_tbl = pandas.read_parquet(b_tblfile)

    return b_train, b_tbl


def run_a_test(algorithm:str, tst_idx:tuple=None):

    tst_idx = (0, 500, 700)

    # Load up
    cutouts, tbl = load_test_data()

    # Loop on indices
    all_fronts, all_divb2, all_sst = [], [], []
    for idx in tst_idx:

        # Images
        div_sst, sst, sss, Divb2 = parse_idx(cutouts, idx)

        # Find fronts
        if algorithm == 'vanilla':
            fronts = algorithms.fronts_from_divb2(Divb2, wndw=40)
        elif algorithm == 'rm_weak':
            fronts = algorithms.fronts_from_divb2(Divb2, wndw=40, rm_weak=1e-15)
        elif algorithm == 'thin':
            fronts = algorithms.fronts_from_divb2(Divb2, wndw=40, thin=True)
        elif algorithm == 'rm_weak-thin-dilate':
            fronts = algorithms.fronts_from_divb2(Divb2, wndw=40, thin=True, rm_weak=1e-15, dilate=True)
        else:
            raise ValueError(f'Algorithm {algorithm} not recognized')

        # Store
        all_fronts.append(fronts)
        all_divb2.append(Divb2)
        all_sst.append(sst)

    # Plot
    outfile = f'fronts_{algorithm}_{tst_idx[0]}_{tst_idx[1]}_{tst_idx[2]}.png'
    front_fig(outfile, np.stack(all_fronts), np.stack(all_divb2), np.stack(all_sst),
              title=f'Algorithm: {algorithm}')


def front_fig(outfile:str, all_fronts, all_divb2, all_sst, 
              title:str=None):

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(2,3)

    # First pair
    #col = 0
    for col in range(3):
        #
        ax_img = plt.subplot(gs[0, col])
        cutout.show_image(all_sst[col], clbl='SST (deg C)', ax=ax_img)
        #cutout.show_image(sv_divb2[col], cbar=True, #clbl=r'$\nabla b^2$', 
        #                      cm='Greys', ax=ax_img, vmnx=(mn_div,mx_div))
        ax_fronts = plt.subplot(gs[1, col])
        cutout.show_image(all_divb2[col], cbar=True, #clbl=r'$\nabla b^2$', 
                            cm='Greys', ax=ax_fronts)#, vmnx=(mn_div,mx_div))
        pcol,prow = np.where(np.flipud(all_fronts[col]))
        ax_fronts.scatter(prow, pcol, s=0.3, color='r', alpha=0.5)

    # Add title?
    if title is not None:
        plt.suptitle(title)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
