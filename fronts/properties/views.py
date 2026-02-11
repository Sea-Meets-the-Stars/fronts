""" Methods to visualize properties of fronts """

import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import seaborn as sns

from wrangler.plotting import cutout

from fronts.plotting.images import field_defs

def show_field(field:str, data:np.ndarray, ax:plt.Axes=None,
               grid_sep:float=10.): 

    if ax is None:
        ax = plt.gca()

    cmap = 'viridis'
    clbl = field

    if field in field_defs:
        cmap = field_defs[field]['cmap']
        clbl = field_defs[field]['label']
    else:
        raise IOError(f"Field {field} not in field_defs")

    # Center if needed
    vmin, vmax = None, None
    if ('vcenter' in field_defs[field]) and (field_defs[field]['vcenter'] is not None):
        vcenter = field_defs[field]['vcenter']
        vmax = np.nanmax(np.abs(data - vcenter))
        vmin = vcenter - vmax
        vmax = vcenter + vmax
    elif 'vmin' in field_defs[field]:
        vmin = field_defs[field]['vmin']

    cutout.show_image(data, clbl=clbl, cm=cmap, ax=ax, 
                        vmnx=(vmin, vmax),
                        cb_kws=dict(pad=0.01, fraction=0.04))

    #ax.set_title(fname)
    ax.xaxis.set_major_locator(MultipleLocator(grid_sep))
    ax.yaxis.set_major_locator(MultipleLocator(grid_sep))
    ax.grid()

    return ax

def show_fields(field_dict:dict, outfile:str, grid_sep:float=10.,
                title:str=None):

    # Number of fields
    nfields = len(field_dict)

    # Set figure size and gridspec from nfields
    #  One row for evrery 3 fields
    nrows = nfields // 3 + (nfields % 3 > 0)
    ncols = (nfields + 1) // nrows

    fig = plt.figure(figsize=(5*ncols, 4*nrows))
    plt.clf()
    gs = gridspec.GridSpec(nrows, ncols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Loop on the fields
    for i, (fname, fdata) in enumerate(field_dict.items()):
        row = i // ncols
        col = i % ncols

        ax = plt.subplot(gs[row, col])
        show_field(fname, fdata, ax=ax, grid_sep=grid_sep)

    # Title?
    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def plot_velocities(U, V, ax:plt.Axes=None):

    npix = U.shape[0]

    x = np.linspace(0, npix-1, npix)
    y = np.linspace(0, npix-1, npix)
    X, Y = np.meshgrid(x, y)

    # Calculate horizontal velocity magnitude
    velocity_magnitude = np.sqrt(U**2 + V**2)


    # Normalize U and V by their magnitude, then scale by magnitude
    # This makes arrow length proportional to velocity_magnitude
    U_normalized = np.where(velocity_magnitude > 0, U / velocity_magnitude, 0)
    V_normalized = np.where(velocity_magnitude > 0, V / velocity_magnitude, 0)

    # Scale factor to control overall arrow size (adjust as needed)
    arrow_scale = 5.0

    # Create the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Create quiver plot with arrows proportional to velocity magnitude
    quiver = ax.quiver(X, Y, 
                    U_normalized * velocity_magnitude * arrow_scale, 
                    V_normalized * velocity_magnitude * arrow_scale,
                    velocity_magnitude, 
                    cmap='jet',
                    scale=1,
                    scale_units='xy',
                    angles='xy',
                    width=0.003/2,
                    alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, label='Velocity Magnitude (m/s)',
                        pad=0.01, fraction=0.04)

    # Set labels and title
    ax.set_xlabel('X (grid points)', fontsize=12)
    ax.set_ylabel('Y (grid points)', fontsize=12)
    ax.set_title('Horizontal Velocity Field', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()

    # Optional: Print statistics
    print(f"Velocity magnitude range: {velocity_magnitude.min():.3f} to {velocity_magnitude.max():.3f}")
    print(f"Mean velocity magnitude: {velocity_magnitude.mean():.3f}")

    return ax