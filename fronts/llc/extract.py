""" Routines to extract cutouts from LLC data """

import numpy as np

import pandas
import xarray

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from wrangler.ogcm import llc as llc_io

#from fronts.llc import io as llc_io
from fronts.preproc import process
from fronts.tables import catalog
from fronts.po import fronts as po_fronts

from IPython import embed

def field_from_ds(ds, field:str):

    if field in ['SST', 'DivSST2', 'SSTK']:
        data = ds.Theta.values
        if field == 'SSTK':
            data += 273.15 # Kelvin
    elif field == 'SSS':
        data = ds.Salt.values
    elif field in ['Divb2', 'normDivb2']:
        data = ds.Theta.values
        data2 = ds.Salt.values
    else:
        raise IOError(f"Not ready for this field {field}")

    # Return
    if field == 'Divb2':
        return po_fronts.calc_gradb(data, data2)
    else:
        return data


def preproc_field(llc_table:pandas.DataFrame, 
                  field:str,
                  pdict:str,
                  field_size=(64,64), 
                  fixed_km=None,
                  n_cores=10,
                  override_RAM=False,
                  test_failures:bool=False,
                  test_process:bool=False,
                  debug=False):
    """Main routine to extract and pre-process LLC data for later SST analysis
    The llc_table is modified in place (and also returned).

    Args:
        llc_table (pandas.DataFrame): cutout table
        field (str): Field to extract
        preproc_dict (dict): Preprocessing steps. 
        field_size (tuple, optional): Defines cutout shape. Defaults to (64,64).
        fixed_km (float, optional): Require cutout to be this size in km
        n_cores (int, optional): Number of cores for parallel processing. Defaults to 10.
        valid_fraction (float, optional): [description]. Defaults to 1..
        dlocal (bool, optional): Data files are local? Defaults to False.
        override_RAM (bool, optional): Over-ride RAM warning?

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: Modified in place table
        success (np.ndarray): Bool array of successful extractions
        pp_fields (np.ndarray): Pre-processed fields
        final_meta (pandas.DataFrame): Meta data for each cutout

    """
    raise DeprecationWarning("Use wrangler.extract.ogcm.preproc_datetime() instead")
    # Load coords?
    if fixed_km is not None:
        coords_ds = llc_io.load_coords()
        R_earth = 6371. # km
        circum = 2 * np.pi* R_earth
        km_deg = circum / 360.
    
    # Setup for parallel
    if field in ['SST','SSS','DivSST2','SSTK']:
        map_fn = partial(process.preproc_image, pdict=pdict)
    elif field in ['Divb2']:
        map_fn = partial(po_fronts.gradb2_cutout, **pdict)
    elif field in ['b']:
        map_fn = partial(po_fronts.b_cutout, **pdict)
    else:
        raise IOError(f"Not ready for this field {field}")

    # Setup for dates
    uni_date = np.unique(llc_table.datetime)
    if len(llc_table) > 1000000 and not override_RAM:
        raise IOError("You are likely to exceed the RAM.  Deal")

    # Init
    pp_fields, meta, img_UID, all_UID = [], [], [], []

    # Loop
    for udate in uni_date:
        # Parse filename
        filename = llc_io.grab_llc_datafile(udate)

        # Data format
        if zarr_path is not None:
            ts = pandas.Timestamp(udate)
            # Convert date into time index
            dt = ts - t0
            time = int(dt / pandas.Timedelta(hours=1))
            # 
            variable = ds_zarr[aios_ds.variable].isel(time=time,face=face).values
        else:
            # Parse filename
            filename = wr_llc.grab_llc_datafile(udate)
            ds = xarray.open_dataset(filename)
            variable = ds[aios_ds.variable].values

        # Field
        if field in ['SST', 'DivSST2', 'SSTK']:
            data = ds.Theta.values
            if field == 'SSTK':
                data += 273.15 # Kelvin
        elif field == 'SSS':
            data = ds.Salt.values
        elif field in ['Divb2', 'b']:
            data = ds.Theta.values
            data2 = ds.Salt.values
        else:
            raise IOError(f"Not ready for this field {field}")

        # Parse 
        gd_date = llc_table.datetime == udate
        sub_UID = llc_table[gd_date].UID.values
        all_UID += sub_UID.tolist()  # These really should be the indices of the Table
        coord_tbl = llc_table[gd_date]

        # Add to table
        llc_table.loc[gd_date, 'filename'] = filename

        # Load up the cutouts
        fields, fields2, smooth_pixs = [], [], []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            if fixed_km is None:
                dr = field_size[0]
                dc = field_size[1]
            else:
                dlat_km = (coords_ds.lat.data[r+1,c]-coords_ds.lat.data[r,c]) * km_deg
                dr = int(np.round(fixed_km / dlat_km))
                dc = dr
            # Deal with smoothing
            if 'smooth_km' in pdict.keys():
                smooth_pix = int(np.round(pdict['smooth_km'] / dlat_km))
                pad = 2*smooth_pix
                #
                use_r = r - pad
                dr += 2*pad
                use_c = c - pad
                dc += 2*pad
                smooth_pixs.append(smooth_pix)
            else:
                use_r, use_c = r, c
            # Off the image?
            if (r+dr >= data.shape[0]) or (c+dc > data.shape[1]) or (
                use_r < 0) or (use_c < 0):
                fields.append(None)
            else:
                fields.append(data[use_r:use_r+dr, use_c:use_c+dc])
            # More?
            if field in ['Divb2', 'b']:
                fields2.append(data2[use_r:use_r+dr, use_c:use_c+dc])
        print("Cutouts loaded for {}".format(filename))

        # Prep items
        zipitems = [fields]
        if len(fields2) > 0:
            zipitems.append(fields2)
        zipitems.append(sub_UID)
        if 'smooth_km' in pdict.keys():
            zipitems.append(smooth_pixs)
        items = [item for item in zip(*zipitems)]

        # Test processing
        if test_process:
            embed(header='extract.py/preproc_field 145')
            idx = 50
            img, tmeta = process.preproc_field(fields[idx], None, **pdict)
            #img, iidx, tmeta = po_fronts.anly_cutout(
            #    items[idx], **pdict)
            '''
            # Smoothing
            img, tmeta = process.preproc_field(fields[idx], None,
                                  smooth_pix=smooth_pixs[idx], 
                                  **pdict)
            # 
            '''
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(8,8))
            plt.clf()
            plt.imshow(img, origin='lower')
            plt.show()

        # Multi-process time
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))

        # Debuggin
        if test_failures:
            answers[50] = [None, answers[50][1], None]

        # Deal with failures
        #answers = [f for f in answers if f is not None]
        cur_img_idx = [item[1] for item in answers]

        # Find missing items and replace with -1.

        # Slurp
        pp_fields += [item[0] for item in answers]
        img_UID += cur_img_idx
        meta += [item[2] for item in answers]

        del answers, fields, items

    # Fuss with indices
    tbl_UID = llc_table.UID.values
    img_UID = np.array(img_UID)
    ppf_UID = catalog.match_ids(tbl_UID, img_UID, require_in_match=True)

    # Clean up time (indices and bad data)

    # Find the bad ones (if any)
    bad_idx = [int(item[1]) for item in zip(pp_fields, img_UID) if item[0] is None]
    good_idx = np.array([int(item[1]) for item in zip(pp_fields, img_UID) if item[0] is not None])

    success = np.ones(len(pp_fields), dtype=bool)

    # Replace with -1 images
    if len(bad_idx) > 0:
        bad_img = -1*np.ones(field_size)
        for ii in bad_idx:
            pp_fields[ii] = bad_img.copy()
            success[ii] = False
    pp_fields = np.array(pp_fields)[ppf_UID]

    # Meta time
    good_meta = pandas.DataFrame([item for item in meta if item is not None])
    final_meta = pandas.DataFrame()
    tbl_idx = catalog.match_ids(good_idx, tbl_UID, require_in_match=True)
    for key in good_meta.keys():
        final_meta[key] = np.zeros(len(ppf_UID))
        #
        final_meta.loc[tbl_idx, key] = good_meta[key].values

    # Return
    return llc_table, success, pp_fields, final_meta