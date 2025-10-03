""" Plotting utilities """
from ulmo.utils import image_utils
from IPython.terminal.embed import embed
from pkg_resources import resource_filename
import os

import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs

# Field color maps and labels

field_defs = {
    'SSTK': {'cmap':'coolwarm', 'label':r'SST (K)'},
    'Divb2': {'cmap':'Greys', 'label':r'$|\nabla b|^2$'},
    'Fs': {'cmap': 'seismic', 'label':'Frontogenesis Tendency',
           'vcenter':0.},
    'SSS': {'cmap':'viridis', 'label':'SSS (psu)'},
    'SSH': {'cmap':'viridis', 'label':'SSH (m)'},
    'EKE': {'cmap':'magma', 'label':'EKE (m$^2$/s$^2$)'},
    'Chl': {'cmap':'viridis', 'label':'Chl (mg/m$^3$)'},
    'b': {'cmap':'viridis', 'label':'Buoyancy (m/s$^2$)'},
    'DivSST2': {'cmap':'Greys', 'label':r'$|\nabla$SST$|^2$'},
    'DivSSS2': {'cmap':'Greys', 'label':r'$|\nabla$SSS$|^2$'},
    'U': {'cmap':'RdBu', 'label':'Zonal Velocity (m/s)'},
    'V': {'cmap':'RdBu', 'label':'Meridional Velocity (m/s)'},
    'W': {'cmap':'RdBu', 'label':'Vertical Velocity (m/s)'},
}


def set_fontsize(ax, fsz):
    """
    Set the fontsize throughout an Axis

    Args:
        ax (Matplotlib Axis):
        fsz (float): Font size

    Returns:

    """
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


def geo_table(df:pandas.DataFrame, nside:int=64, color:str='Blues',
              log:bool=False, vmax:float=None, lbl:str=None):

    hp_events, hp_lons, hp_lats = image_utils.evals_to_healpix(df, nside)

    if log:
        hp_plot = np.log10(hp_events)
    else:
        hp_plot = hp_events

   # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap(color)
    # Cut
    good = np.invert(hp_plot.mask)
    img = plt.scatter(x=hp_lons[good],
        y=hp_lats[good],
        c=hp_plot[good], 
        cmap=cm,
        vmax=vmax, 
        s=1,
        transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    if lbl is not None:
        cb.set_label(lbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.add_feature(cartopy.feature.LAND, 
        facecolor='gray', edgecolor='black')
    ax.set_global()

    gl = ax.gridlines(crs=tformP, linewidth=1, 
        color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}

    set_fontsize(ax, 19.)