""" Spatial Plotting Functions """
import numpy as np
import healpy as hp

from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def evals_to_healpix(vals:np.ndarray, lons:np.ndarray,
                     lats:np.ndarray, nside:int, mask:bool=True):
    """
    Generate a healpix map of where the input
    values as located on the globe

    Parameters
    ----------
    vals : pandas.DataFrame
    nside : int  # nside is a number that sets the resolution of map
    mask : bool, optional

    Returns
    -------
    healpix_array : hp.ma (number of cutouts)
    lats : np.ndarray
    lons : np.ndarray
    healpix_array : hp.ma (average LL)
    """
    # Healpix coords
    theta = (90 - lats) * np.pi / 180. 
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi) 

    # Count events
    npix_hp = hp.nside2npix(nside)
    all_events = np.ma.masked_array(np.zeros(npix_hp, dtype='int')) 
    all_values = np.ma.masked_array(np.zeros(npix_hp, dtype='int')) 

    for i, idx in enumerate(idx_all):
        all_events[idx] += 1
        all_values[idx] += vals[i] 

    zero = all_events == 0 
    
    float_events = all_events.astype(float)
    float_values = all_values.astype(float)
    float_values[~zero] = all_values[~zero]/all_events[~zero]


    # Mask
    hpma = hp.ma(float_events)
    hpma1 = hp.ma(float_values)
    if mask:  # if you want to mask float_events
        hpma.mask = zero # current mask set to zero array, where Trues (no events) are masked
        hpma1.mask = zero 

    # Angles
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), lonlat=True)

    # Return
    return hpma, hp_lons, hp_lats, hpma1


def plot_hp(hp_lons, hp_lats, hp_values,
            tricontour=False,
            lbl=None, figsize=(12,8), 
            vmnx=(None,None),
            color='jet', show=True):
    """Generate a global map of mean LL of the input
    cutouts
    Args:
        main_tbl (pandas.DataFrame): table of cutouts
        nside (int, optional): [description]. Defaults to 64.
        use_log (bool, optional): [description]. Defaults to True.
        use_mask (bool, optional): [description]. Defaults to True.
        tricontour (bool, optional): [description]. Defaults to False.
        lbl ([type], optional): [description]. Defaults to None.
        figsize (tuple, optional): [description]. Defaults to (12,8).
        color (str, optional): [description]. Defaults to 'Reds'.
        show (bool, optional): If True, show on the screen.  Defaults to True
    Returns:
        matplotlib.Axis: axis holding the plot
    """
    # Healpix me
    #hp_events, hp_lons, hp_lats, hp_values = evals_to_healpix(
    #    main_tbl, nside, mask=use_mask)
    
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    if tricontour:
        cm = plt.get_cmap(color)
        img = ax.tricontourf(hp_lons, hp_lats, hp_values, transform=tformM,
                         levels=20, cmap=cm)#, zorder=10)
    else:
        cm = plt.get_cmap(color)
        # Cut
        good = np.invert(hp_values.mask)
        img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=hp_values[good], 
            vmin=vmnx[0], vmax=vmnx[1], 
            cmap=cm,
            s=1,
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    if lbl is not None:
        cb.set_label(lbl, fontsize=15.)
    cb.ax.tick_params(labelsize=14)

    # Coast lines
    if not tricontour:
        ax.coastlines(zorder=10)
        ax.set_global()
    
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
            color='black', alpha=0.5, linestyle=':', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right=False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}
        #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
        #gl.xlocator = mticker.FixedLocator([-240., -180., -120, -65, -60, -55, 0, 60, 120.])
        #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])


    # Layout and save
    if show:
        plt.show()

    return ax

