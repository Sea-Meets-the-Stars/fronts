""" Module for generating a Table of LLC data """

import numpy as np

import datetime

import pandas

#from ulmo.llc import io as llc_io
#from ulmo import io as ulmo_io

from fronts.llc import io as llc_io
from fronts.tables import catalog
from fronts import io as fronts_io

# Plotting
from matplotlib import pyplot as plt

try:
    import cartopy.crs as ccrs
except ImportError:
    print("cartopy not installed..")

# Astronomy tools
import astropy_healpix

from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky


from IPython import embed

def add_days(llc_table:pandas.DataFrame, dti:pandas.DatetimeIndex, 
             outfile=None, to_s3:bool=False):
    """Add dates to an LLC table

    Args:
        llc_table (pandas.DataFrame): [description]
        dti (pandas.DatetimeIndex): [description]
        to_s3 (bool, optional): Write to s3?  Defaults to False.
        outfile ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    # Check
    if 'datetime' in llc_table.keys():
        print("Dates already specified.  Not modifying")
        return llc_table

    # Do it
    llc_table['datetime'] = dti[0]
    for date in dti[1:]:
        new_tbl = llc_table[llc_table['datetime'] == dti[0]].copy()
        new_tbl['datetime'] = date
        #llc_table = llc_table.append(new_tbl, ignore_index=True)
        llc_table = pandas.concat([llc_table, new_tbl], ignore_index=True)

    # Drop index
    llc_table.drop(columns=['index'], inplace=True)

    # Write
    if outfile is not None:
        assert catalog.vet_main_table(llc_table)
        fronts_io.write_main_table(llc_table, outfile, to_s3=to_s3)

    # Return
    return llc_table

def add_uid(df:pandas.DataFrame,
             outfile=None, to_s3:bool=False):
    """ Generate a unique identifier for LLC

    Args:
        df (pandas.DataFrame): main table
        outfile (str, optional): If provided, write the table to this outfile.
        to_s3 (bool, optional): Write to s3?  Defaults to False.

    Returns:
        numpy.ndarray: int64 array of unique identifiers
    """
    # Date?
    #if 'date' not in df.keys():
    #    # Dates
    #    ioff = 10
    #    dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
    #                            int(ifile[5+ioff:7+ioff]),
    #                            int(ifile[7 + ioff:9+ioff]),
    #                            int(ifile[10+ioff:12+ioff]),
    #                            int(ifile[12+ioff:14+ioff]))
    #            for ifile in df['filename'].values]
    #    df['date'] = dtimes
        
    # Unique identifier
    tlong = df['datetime'].values.astype(np.int64) // 10000000000
    latkey = 'latitude' if 'latitude' in df.keys() else 'lat'
    lonkey = 'longitude' if 'longitude' in df.keys() else 'lon'
    lats = np.round((df[latkey].values.astype(float) + 90)*10000).astype(int)
    lons = np.round((df[lonkey].values.astype(float) + 180)*100000).astype(int)
    uid = [np.int64('{:s}{:d}{:d}'.format(str(t)[:-5],lat,lon))
            for t,lat,lon in zip(tlong, lats, lons)]
    if len(uid) != len(np.unique(uid)):
        embed(header='67 of results')

    uids = np.array(uid).astype(np.int64)
    df['UID'] = uids

    # Write
    if outfile is not None:
        assert catalog.vet_main_table(df)
        fronts_io.write_main_table(df, outfile, to_s3=to_s3)


    # Return
    return np.array(uid).astype(np.int64)

def uniform_coords(resol, field_size, CC_max=1e-4, outfile=None, 
           minmax_lat=None, localCC=True,
           rotate:float=None):
    """
    Use healpix to setup a uniform extraction grid

    Args:
        resol (float): Typical separation on the healpix grid (deg)
        minmax_lat (tuple): Restrict to latitudes given by this range
        field_size (tuple): Cutout size in pixels
        outfile (str, optional): If provided, write the table to this outfile.
            Defaults to None.
        localCC (bool, optional):  If True, load the CC_mask locally.
        rotate (float, optional): Rotate the grid by this angle (deg)

    Returns:
        pandas.DataFrame: Table containing the coords
    """
    # Load up CC_mask
    CC_mask = llc_io.load_CC_mask(field_size=field_size, local=localCC)

    # Cut
    good_CC = CC_mask.CC_mask.values < CC_max
    good_CC_idx = np.where(good_CC)

    # Build coords
    llc_lon = CC_mask.lon.values[good_CC].flatten()
    llc_lat = CC_mask.lat.values[good_CC].flatten()
    print("Building LLC SkyCoord")
    llc_coord = SkyCoord(llc_lon*units.deg + 180.*units.deg, 
                         llc_lat*units.deg, 
                         frame='galactic')

    # Healpix time
    nside = astropy_healpix.pixel_resolution_to_nside(resol*units.deg)
    hp = astropy_healpix.HEALPix(nside=nside)
    hp_lon, hp_lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    if rotate is not None:
        hp_lon = hp_lon + rotate*np.pi/180. * units.rad

    # Coords
    hp_coord = SkyCoord(hp_lon, hp_lat, frame='galactic')
                        
    # Cross-match
    print("Cross-match")
    idx, sep2d, _ = match_coordinates_sky(hp_coord, llc_coord, nthneighbor=1)
    good_sep = sep2d < hp.pixel_resolution

    # Build the table
    llc_table = pandas.DataFrame()
    llc_table['lat'] = llc_lat[idx[good_sep]]  # Center of cutout
    llc_table['lon'] = llc_lon[idx[good_sep]]  # Center of cutout

    llc_table['row'] = good_CC_idx[0][idx[good_sep]] - field_size[0]//2 # Lower left corner
    llc_table['col'] = good_CC_idx[1][idx[good_sep]] - field_size[0]//2 # Lower left corner

    # Require unique row, col
    uid_rowcol = llc_table.row.values*100000 + llc_table.col.values
    uu, counts = np.unique(uid_rowcol, return_counts=True)
    bad = counts > 1
    if np.any(bad):
        keep = np.ones(len(uid_rowcol), dtype=bool)
        print("Removing {} duplicate row, col".format(np.sum(bad)))
        for ii in np.where(bad)[0]:
            mt = uid_rowcol == uu[ii]
            idx = np.where(mt)[0]
            keep[idx[1:]] = False
        # Check
        new_uu, new_counts = np.unique(uid_rowcol[keep], return_counts=True)
        assert np.all(new_counts == 1)

        # Cut down
        llc_table = llc_table[keep].copy()
        llc_table.reset_index(inplace=True)
        llc_table.drop(columns=['index'], inplace=True)

    # Cut on latitutde?
    if minmax_lat is not None:
        print(f"Restricting to latitudes = {minmax_lat}")
        #gd_lat = np.abs(llc_table.lat) < max_lat
        gd_lat = (llc_table.lat > minmax_lat[0]) & (llc_table.lat < minmax_lat[1])
        llc_table = llc_table[gd_lat].copy()

    llc_table.reset_index(inplace=True)
    
    # Write
    if outfile is not None:
        fronts_io.write_main_table(llc_table, outfile)

    # Return
    return llc_table



def plot_extraction(llc_table:pandas.DataFrame, figsize=(7,4),
                    resol=None, cbar=False, s=0.01):
    """Plot the extractions to check

    Args:
        llc_table (pandas.DataFrame): table of cutouts
        figsize (tuple, optional): Sets the figure size
        resol (float, optional): Angle in deg for healpix check. Defaults to None.
        cbar (bool, optional): [description]. Defaults to False.
        s (float, optional): [description]. Defaults to 0.01.
    """

    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)


    # Cut
    #good = np.invert(hp_events.mask)
    img = plt.scatter(x=llc_table.lon,
        y=llc_table.lat,
        s=s, zorder=2,
        transform=tformP)

    # Healpix?
    if resol is not None:
        nside = astropy_healpix.pixel_resolution_to_nside(resol*units.deg)
        hp = astropy_healpix.HEALPix(nside=nside)
        hp_lon, hp_lat = hp.healpix_to_lonlat(np.arange(hp.npix))
        img = plt.scatter(x=hp_lon.to('deg').value,
            y=hp_lat.to('deg').value,
            s=s,
            color='r', zorder=1,
            transform=tformP)

    #
    # Colorbar
    if cbar:
        cb = plt.colorbar(img, orientation='horizontal', pad=0.)
        cb.ax.tick_params(labelsize=17)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.set_global()

    plt.show()

    return