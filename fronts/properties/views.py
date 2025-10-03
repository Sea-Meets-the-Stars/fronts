""" Methods to visualize properties of fronts """

import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import seaborn as sns

from wrangler.plotting import cutout

from fronts.plotting.images import field_defs


def show_fields(field_dict:dict, outfile:str):

    # Number of fields
    nfields = len(field_dict)

    # Set figure size and gridspec from nfields
    #  One row if nfields <= 3 else two rows
    if nfields <= 3:
        nrows = 1
        ncols = nfields
    else:
        nrows = 2
        ncols = (nfields + 1) // 2

    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    plt.clf()
    gs = gridspec.GridSpec(nrows, ncols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Loop on the fields
    for i, (fname, fdata) in enumerate(field_dict.items()):
        row = i // ncols
        col = i % ncols

        ax = plt.subplot(gs[row, col])
        cmap = 'viridis'
        clbl = fname
        if fname in field_defs:
            cmap = field_defs[fname]['cmap']
            clbl = field_defs[fname]['label']
        else:
            raise IOError(f"Field {fname} not in field_defs")

        # Center if needed
        vmin, vmax = None, None
        if ('vcenter' in field_defs[fname]) and (field_defs[fname]['vcenter'] is not None):
            vcenter = field_defs[fname]['vcenter']
            vmax = np.nanmax(np.abs(fdata - vcenter))
            vmin = vcenter - vmax
            vmax = vcenter + vmax
            #cmap = mpl.cm.get_cmap(cmap)
            #cmap.set_bad('lightgrey')
            #cmap = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        cutout.show_image(fdata, clbl=clbl, cm=cmap, ax=ax, 
                          vmnx=(vmin, vmax))

        #ax.set_title(fname)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.grid()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")