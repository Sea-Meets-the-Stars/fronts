""" Script to grab LLC model data """

from IPython import embed

'''
LLC docs: https://mitgcm.readthedocs.io/en/latest/index.html
'''

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Grab LLC Model data')
    parser.add_argument("tstep", type=int, help="Time step in hours")
    #parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")
    parser.add_argument("--model", type=str, default='LLC4320',
                        help="LLC Model name.  Allowed options are [LLC4320]")
    parser.add_argument("--var", type=str, default='Theta',
                        help="LLC data variable name.  Allowed options are [Theta, U, V, Salt, W, Eta]")
    parser.add_argument("--istart", type=int, default=0,
                        help="Index of model to start with")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Debug?")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def xmit_main(pargs):
    """ Run
    """
    raise DeprecationWarning("The data files are largely gone")
    import numpy as np
    import os
    import warnings

    import xmitgcm.llcreader as llcreader
    from fronts.llc.slurp import write_xr

    # Load model once
    if pargs.model == 'LLC4320':
        model = llcreader.ECCOPortalLLC4320Model()
        tstep_hr = 144  # Time steps per hour

    # Get dataset
    iter_step = tstep_hr*pargs.tstep
    ds = model.get_dataset(varnames=pargs.var.split(','),
                               k_levels=[0], type='latlon',
                               iter_step=iter_step)
    tsize = ds.time.size
    print("Model is ready")

    # Loop me
    for tt in range(pargs.istart, tsize):
        # Get dataset
        iter_step = tstep_hr*pargs.tstep
        ds = model.get_dataset(varnames=pargs.var.split(','),
                                k_levels=[0], type='latlon',
                               iter_step=iter_step)
        #
        print("Time step = {} of {}".format(tt, ds.time.size))
        #SST = ds_sst.Theta.isel(time=tt, k=0)  # , i=slice(1000,2000), j=slice(1000,2000))
        ds_0 = ds.isel(time=tt, k=0)  # , i=slice(1000,2000), j=slice(1000,2000))
        # Generate outfile name
        outfile = '{:s}_{:s}.nc'.format(pargs.model,
            str(ds_0.time.values)[:19].replace(':','_'))
        # No clobber
        if os.path.isfile(outfile):
            print("Not clobbering: {}".format(outfile))
            continue
        # Write
        write_xr(ds_0, outfile, encode=True)
        print("Wrote: {}".format(outfile))
        del(ds)
        #embed(header='60 of slurp')

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


    #from xarray.core.combine import _nested_combine
    from functools import partial

    # Model
    if pargs.model == 'LLC4320':
        model = llcreader.ECCOPortalLLC4320Model()
        tstep_hr = 144  # Time steps per hour
    else:
        raise IOError("Not ready for this model")

    # We want all of the faces/facets
    face_range = range(0,13)

    # Get dataset for times
    iter_step = tstep_hr*pargs.tstep
    varnames=pargs.var.split(',')
    ds = model.get_dataset(varnames=varnames,
                               k_levels=[0], type='latlon',
                               iter_step=iter_step)
    tsize = ds.time.size
    print("Model is ready")


    '''
    #define which iterations you want. 
    start_from = 1180 # 0 is the first record of Eta, ~1180 is the first record for wind
    length_in_hours = 12
    time_step_in_hours = 3 # the minimum timestep between files is one hour


    end_iter = 10368 + start_from*144 + length_in_hours*144
    '''

    all_vars = ['Theta','U','V','W','Salt','Eta']
    drop_vars = [var for var in all_vars if var not in varnames]


    for tt in range(pargs.istart, tsize):
        start_iter = 10368 + tt*pargs.tstep*144
        end_iter = start_iter + pargs.tstep*144
        iter_range = np.arange(start_iter,end_iter,pargs.tstep*144)

        get_Eta_files = True

        endpoint_url = 'https://mghp.osn.xsede.org'
        fs = s3fs.S3FileSystem(anon=True,
            client_kwargs={'endpoint_url': endpoint_url}
        )

        # Construct the lazy ds

        if (get_Eta_files):
            filelist = ['cnh-bucket-1/llc_surf/kerchunk_files/llc4320_Eta-U-V-W-Theta-Salt_f' + str(var1) + '_k0_iter_' + 
                    str(var) + '.json' for var1 in face_range for var in iter_range]
        else:
            filelist = ['cnh-bucket-1/llc_wind/kerchunk_files/llc4320_KPPhbl-PhiBot-oceTAUX-oceTAUY-SIarea_f' + str(var1) + '_k0_iter_' + 
                    str(var) + '.json' for var1 in face_range for var in iter_range]


        mapper = [fs.open(file, mode='rb') for file in filelist]

        reflist = [ujson.load(mapper1)for mapper1 in mapper]

        open_ = dask.delayed(xr.open_dataset)
        getattr_ = dask.delayed(getattr)

        dict_list = [{"storage_options": {"fo": p,"remote_protocol": "s3",
            "remote_options": {"client_kwargs":{'endpoint_url': endpoint_url}, "anon": True}},"consolidated": False} for p in reflist]
        datasets = [open_("reference://",engine="zarr",backend_kwargs=p,chunks={'i':720, 'j':720}) for p in dict_list]
        closers = [getattr_(ds, "_close") for ds in datasets]
        datasets, closers = dask.compute(datasets, closers)

        ds = xr.combine_by_coords([dataset for dataset in datasets],compat="override", coords='minimal', combine_attrs='override')

        for ds1 in datasets:
                    ds1.close()

        ds.set_close(partial(_multi_file_closer, closers))

        # Drop unwanted variables
        ds = ds.drop_vars(drop_vars)

        # Write
        # Generate outfile name
        outfile = '{:s}_{:s}.nc'.format(pargs.model,
            str(ds.time.values)[2:21].replace(':','_'))
        # No clobber
        if os.path.isfile(outfile):
            print("Not clobbering: {}".format(outfile))
            continue

        print("Working on: {}".format(outfile))

        ds = ds.isel(time=0, k=0, k_l=0)

        # Convert to rectangular
        ds_rect = llcreader.llcmodel.faces_dataset_to_latlon(ds, metric_vector_pairs=[])

        if pargs.debug:
            embed(header='203 of slurp_llc')
        # Write
        write_xr(ds_rect, outfile, encode=True)
        print("Wrote: {}".format(outfile))


# ulmo_grab_llc 12 --var Theta,U,V,W,Salt --istart 480
# fronts_slurpllc 12 --var Theta,U,V,W,Salt,Eta 
#   Failures = 45 (LLC4320_2011-10-05T12_00_00.nc)
#   Failures = 45 (LLC4320_2012-02-21T00_00_00.nc)

# fronts_slurpllc 12 --var Theta,U,V,W,Salt,Eta --istart 322