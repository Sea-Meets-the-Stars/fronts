""" Find fronts from |div b|^2 """
import os
import numpy as np

import h5py

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

from wrangler.plotting import cutout

from fronts.pyboa import pyboa

from IPython import embed

def find_em(h5_file:str, outfile:str, targ_idx:int=0, 
            debug:bool=False, wndw:int=40, clobber:bool=False):

    # Check
    if os.path.isfile(outfile) and not clobber:
        print(f'File {outfile} exists and clobber is False. Returning')
        return

    # Open
    f = h5py.File(h5_file, 'r')
    all_divb2 = f['targets'][:, targ_idx, 0, ...]
    f.close()

    # Loop em
    all_fronts = []
    for ii in range(all_divb2.shape[0]):
        print(f'Finding fronts for image {ii} of {all_divb2.shape[0]}')
        fronts = pyboa.fronts_in_divb2(all_divb2[ii], wndw=wndw)
        all_fronts.append(fronts)

        if debug and ii > 5:
            break

    # Save an h5
    all_fronts = np.stack(all_fronts)
    with h5py.File(outfile, 'w') as f2:
        f2.create_dataset('fronts', data=all_fronts, compression='gzip')
    print(f'Wrote {outfile}')


def gen_figs(div_file:str, front_file:str, fig_root:str,
             nsample:int=99, sst_idx:int=2,
             targ_idx:int=0):

    # Open
    with h5py.File(front_file, 'r') as f:
        all_fronts = f['fronts'][:]
    with h5py.File(div_file, 'r') as f:
        all_divb2 = f['targets'][:, targ_idx, 0, ...]
        all_sst = f['inputs'][:, sst_idx, 0, ...]

    # Sample
    nimages = all_fronts.shape[0]
    if nsample > nimages:
        nsample = nimages
    step = nimages // nsample
    sample_idx = np.arange(0, nimages, step)[:nsample]

    sample_fronts = all_fronts[sample_idx]
    sample_divb2 = all_divb2[sample_idx]
    sample_sst = all_sst[sample_idx]

    #embed(header='52 of find_fronts.py')

    # Plot time
    for idx in range(0, nsample//3):
        iidx = 3*idx 

        # Figure out file
        outfile = f'{fig_root}_{sample_idx[iidx]}_{sample_idx[iidx+1]}_{sample_idx[iidx+2]}.png'

        fig = plt.figure(figsize=(12,6))
        plt.clf()
        gs = gridspec.GridSpec(2,3)

        # First pair
        #col = 0
        for col in range(3):
            sidx = iidx + col
            #mx_div = sample_divb2[sidx].max()
            #mn_div = mx_div / 8.
            #
            ax_img = plt.subplot(gs[0, col])
            cutout.show_image(sample_sst[sidx], clbl='SST (deg C)', ax=ax_img)
            #cutout.show_image(sv_divb2[col], cbar=True, #clbl=r'$\nabla b^2$', 
            #                      cm='Greys', ax=ax_img, vmnx=(mn_div,mx_div))
            ax_fronts = plt.subplot(gs[1, col])
            cutout.show_image(sample_divb2[sidx], cbar=True, #clbl=r'$\nabla b^2$', 
                                cm='Greys', ax=ax_fronts)#, vmnx=(mn_div,mx_div))
            pcol,prow = np.where(np.flipud(sample_fronts[sidx]))
            ax_fronts.scatter(prow, pcol, s=0.3, color='r', alpha=0.5)
        
        plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")


    
# Command line execution
if __name__ == "__main__":
    train_B = True
    fpath = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'Training_Sets') 

    if train_B:
        # Find fronts
        b_file = os.path.join(fpath, 'LLC4320_SST144_SSS40_trainB.h5')
        b_out = os.path.join(fpath, 'LLC4320_SST144_SSS40_trainB_fronts.h5')
        find_em(b_file, b_out, targ_idx=0, clobber=False)#, debug=True)

        # Make figures
        gen_figs(b_file, b_out, 'fronts_B', nsample=99)