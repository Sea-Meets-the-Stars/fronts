""" Script to grab LLC model data """

from IPython import embed

'''
LLC docs: https://mitgcm.readthedocs.io/en/latest/index.html
'''
#
# parser.add_argument("--model", type=str, default='LLC4320',
#                     help="LLC Model name.  Allowed options are [LLC4320]")

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Grab LLC Model data')
    parser.add_argument("tstep", type=int, help="Time step in hours")
    #parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")
    parser.add_argument("--var", type=str, default='Theta',
                        help="LLC data variable name.  Allowed options are [Theta, U, V, Salt, W, Eta]")
    # parser.add_argument("--istart", type=int, default=0, todo add in start from with explanation
    #                     help="Index of model to start with")
    parser.add_argument("--insert", default=False, action="store_true",
                        help="Insert field(s) into existing file?")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Debug?")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs

def _multi_file_closer(closers):
    for closer in closers:
        closer()

def main(pargs):

    import os
    import s3fs
    import xarray as xr
    import ujson
    import dask
    import numpy as np
    from xmitgcm import llcreader
    from fronts.llc.slurp import write_xr
    from functools import partial
    import matplotlib.pyplot as plt

    # the model was run with a cadence of 25 seconds. So there are 144 timesteps in an hour.
    tstep_hr = 144

    # LLC4320 has 13 faces 0 - 12
    face_range = range(0,13)

    # Get dataset for times
    iter_step = tstep_hr*pargs.tstep
    print(iter_step)

    varnames = pargs.var.split(',')
    print(varnames)


    # # The following is taken + modified from:
    # #  https://github.com/cspencerjones/OSN_LLC4320

    start_from = 1180  # 0 is the first record of Eta, ~1180 is the first record for wind
    #start_from = 1300

    length_in_hours = 12 # total hours to pull data from todo add as input

    start_iter = 10368 + start_from*144
    end_iter = 10368 + start_from*144 + length_in_hours*144
    iter_range = np.arange(start_iter, end_iter, pargs.tstep * 144)
    get_Eta_files = True

    endpoint_url = 'https://mghp.osn.xsede.org'
    fs = s3fs.S3FileSystem(anon=True,
                           client_kwargs={'endpoint_url': endpoint_url}
                           )

    if (get_Eta_files):
        filelist = ['cnh-bucket-1/llc_surf/kerchunk_files/llc4320_Eta-U-V-W-Theta-Salt_f' + str(var1) + '_k0_iter_' +
                    str(var) + '.json' for var1 in face_range for var in iter_range]
    else:
        filelist = ['cnh-bucket-1/llc_wind/kerchunk_files/llc4320_KPPhbl-PhiBot-oceTAUX-oceTAUY-SIarea_f' + str(
            var1) + '_k0_iter_' +
                    str(var) + '.json' for var1 in face_range for var in iter_range]

    print(f"Opening {len(filelist)} Kerchunk JSON files...")
    mapper = [fs.open(file, mode='rb') for file in filelist]

    print("Parsing JSON metadata into Python dictionaries...")
    reflist = [ujson.load(mapper1) for mapper1 in mapper]

    open_ = dask.delayed(xr.open_dataset)
    getattr_ = dask.delayed(getattr)

    dict_list = [{
        "storage_options": {"fo": p, "remote_protocol": "s3",
                            "remote_options": {"client_kwargs": {'endpoint_url': endpoint_url},
                                               "anon": True}, #"asynchronous": False
                            },
        "consolidated": False} for p in reflist]

    print("Creating lazy xarray datasets...")
    datasets = [open_("reference://", engine="zarr", backend_kwargs=p, chunks={'i': 720, 'j': 720}) for p in dict_list]
    closers = [getattr_(ds, "_close") for ds in datasets]

    print("Computing delayed datasets (loading metadata lazily)...")
    datasets, closers = dask.compute(datasets, closers)

    print("Combining datasets by coordinates...")
    ds = xr.combine_by_coords([dataset for dataset in datasets], compat="override", coords='minimal',
                              combine_attrs='override')

    for ds1 in datasets:
        ds1.close()

    ds.set_close(partial(_multi_file_closer, closers))
    print("All datasets combined successfully! Dataset ready for processing.")

    print("DATASET:")
    print(ds)

    print("\nDIMENSIONS:")
    print(ds.dims)

    print("\nCOORDINATES:")
    print(ds.coords)

    print("\nVARIABLES:")
    print(ds.data_vars)

    ds['Theta'].isel(time=0, k=0, face=0).mean()
    ds['Theta'].isel(time=0, k=0, face=0).max()
    ds['Theta'].isel(time=0, k=0, face=0).min()

    ds.set_close(partial(_multi_file_closer, closers))

    ds = ds.isel(time=0, k=0, k_l=0) #todo what does this do




    # PLOT ----------

    # plot before to lat lon

    # VARIABLE = "Eta"
    #
    #
    # ds_rect = llcreader.llcmodel.faces_dataset_to_latlon(ds, metric_vector_pairs=[]) # todo have to update source code for this to work

    # print("\nCOORDINATES RECT:")
    # print(ds_rect)
    # var = ds_rect[VARIABLE]
    #
    # print(var.dims, var.shape)
    # print("Plotting slice with dims:", var.dims)
    #
    # plt.figure(figsize=(20, 12))
    # var.plot(vmin=-5.0, vmax=5.0)
    # plt.title(f"{VARIABLE}")
    # plt.show()





    # try to calculate gradient ----------------------------

    # GRID --------------------------------------------------

    # code to create a lazily loaded xarray dataset of the llc4320 grid with pointers to the data on OSN

    endpoint_url = 'https://mghp.osn.xsede.org'
    fs = s3fs.S3FileSystem(anon=True,
                           client_kwargs={'endpoint_url': endpoint_url}
                           )

    filelist = ['cnh-bucket-1/llc_surf/kerchunk_files/llc4320_grid_f' + str(var1) + '.json' for var1 in range(0, 13)]

    mapper = [fs.open(file, mode='rb') for file in filelist]

    reflist = [ujson.load(mapper1) for mapper1 in mapper]

    open_ = dask.delayed(xr.open_dataset)
    getattr_ = dask.delayed(getattr)

    dict_list = [{"storage_options": {"fo": p, "remote_protocol": "s3",
                                      "remote_options": {"client_kwargs": {'endpoint_url': endpoint_url},
                                                         "anon": True}}, "consolidated": False} for p in reflist]
    datasets = [open_("reference://", engine="zarr", backend_kwargs=p, chunks={}) for p in dict_list]
    closers = [getattr_(ds, "_close") for ds in datasets]
    datasets, closers = dask.compute(datasets, closers)

    co = xr.combine_by_coords([dataset for dataset in datasets], compat="override", coords='minimal',
                              combine_attrs='override')

    for ds1 in datasets:
        ds1.close()

    print("GRID")
    print(co)






    # MASKS ----------

    # cutout_size = (64, 64)
    #
    # CC_mask_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'data', 'CC',
    #                             'LLC_CC_mask_{}.nc'.format(cutout_size[0]))
    #
    # CC_mask = xarray.open_dataset(CC_mask_file)
    #
    # # print(CC_mask)
    # #
    # # print(CC_mask.dims)
    # # print(CC_mask.coords)
    # # print(CC_mask.lon)
    # # print(CC_mask.lat)
    #
    # print(CC_mask.lon.values.min(), CC_mask.lon.values.max())
    # print(CC_mask.lat.values.min(), CC_mask.lat.values.max())
    #
    # print(CC_mask.CC_mask.shape)
