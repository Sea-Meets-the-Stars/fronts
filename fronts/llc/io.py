""" Basic I/O routines for the LLC analysis """

import os
import yaml
import warnings

import numpy as np
import xarray as xr
import pandas

from dbof.cli import zarr_to_netcdf
import dbof.dataset_creation.config as dbof_config


from IPython import embed

if os.getenv('LLC_DATA') is not None:
    local_llc_files_path = os.path.join(os.getenv('LLC_DATA'), 'ThetaUVSalt')
s3_llc_files_path = 's3://llc/ThetaUVSalt'

# ---------------------------------------------------------------------------
# Module-level configurable root path for all Fronts I/O
# ---------------------------------------------------------------------------
_fronts_root = None
if os.getenv('OS_OGCM') is not None:
    _fronts_root = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts')

def set_fronts_path(path:str):
    """Set the root directory for all Fronts I/O products.

    All output files are organised as::

        PATH / V{version} / YYYYMMDD_HHMMSS / <filename>

    Call once at the start of a script, e.g.::

        from fronts.llc import io as llc_io
        llc_io.set_fronts_path('/mnt/tank/Oceanography/data/OGCM/LLC/Fronts')

    Parameters
    ----------
    path : str
        Root directory for Fronts products (the ``PATH`` component).
    """
    global _fronts_root
    _fronts_root = path

def get_fronts_path() -> str:
    """Return the current Fronts root directory.

    Falls back to ``$OS_OGCM/LLC/Fronts`` when no override has been
    set via :func:`set_fronts_path`.
    """
    if _fronts_root is not None:
        return _fronts_root
    return os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts')


def _format_timestamp(timestamp: str) -> str:
    """Convert a timestamp string to the directory-name format YYYYMMDD_HHMMSS.

    Accepts formats like '2012-11-09T12_00_00' or '2012-11-09T12:00:00'.

    Examples
    --------
    >>> _format_timestamp('2012-11-09T12_00_00')
    '20121109_120000'
    """
    # Strip dashes and the 'T' separator
    s = timestamp.replace('-', '').replace('T', '_')
    # At this point we have e.g. '20121109_12_00_00'; collapse to YYYYMMDD_HHMMSS
    parts = s.split('_')
    # parts[0] = YYYYMMDD, rest = HH, MM, SS  (or already HHMMSS)
    date_part = parts[0]
    time_part = ''.join(parts[1:]).replace(':', '')
    return f'{date_part}_{time_part}'


def fronts_dir(version: str, timestamp: str, generate: bool = False) -> str:
    """Build the versioned + timestamped output directory.

    Returns ``PATH / V{version} / YYYYMMDD_HHMMSS`` and creates it if
    it does not exist.

    Parameters
    ----------
    version : str
        Version string (e.g. '3').  Prefixed with ``V``.
    timestamp : str
        Snapshot timestamp (e.g. '2012-11-09T12_00_00').
    generate : bool, optional
        Generate the directory if it does not exist. Defaults to False.
    
    Returns:
    --------
        str: The path to the directory.
    """
    ts_dir = _format_timestamp(timestamp)
    d = os.path.join(get_fronts_path(), f'V{version}', ts_dir)
    if generate:
        os.makedirs(d, exist_ok=True)
    # Return
    return d

def load_coords(verbose=True):
    """Load LLC coordinates

    Args:
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        xarray.DataSet: contains the LLC coordinates
    """
    coord_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'LLC_coords.nc')
    if verbose:
        print("Loading LLC coords from {}".format(coord_file))
    coord_ds = xr.load_dataset(coord_file, engine='h5netcdf')
    return coord_ds

def derived_filename(timestamp:str, field:str,
                 root:str='LLC4320',
                 version:str=None):
    """Generate filename of derived field from LLC.

    The file is placed under ``PATH/V{version}/YYYYMMDD_HHMMSS/``.

    Args:
        timestamp: str
            Timestamp of the data to be loaded.
            Format: 'YYYY-MM-DDTHH_MM_SS'
        field: str
            Field to be loaded, e.g. 'gradb2'
        root: str
            Root of the filename.  Defaults to 'LLC4320'.
        version: str
            Version of the algorithm to use.  Required.

    Returns:
        filename: str
    """
    path = fronts_dir(version, timestamp)

    # Generate base
    basefile = f'{root}_{timestamp}_{field}_v{version}.nc'

    # Join and return
    return os.path.join(path, basefile)


def load_CC_mask(field_size=(64,64), verbose=True, local=True):
    """Load up a CC mask.  Typically used for setting coordinates

    Args:
        field_size (tuple, optional): Field size of the cutouts. Defaults to (64,64).
        verbose (bool, optional): Defaults to True.
        local (bool, optional): Load from local hard-drive. 
            Requires LLC_DATA env variable.  Defaults to True (these are 3Gb files)

    Returns:
        xr.DataSet: CC_mask
    """
    if local:
        CC_mask_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'data', 'CC',
                                   'LLC_CC_mask_{}.nc'.format(field_size[0]))
        CC_mask = xr.open_dataset(CC_mask_file, engine='h5netcdf')
    else:
        CC_mask_file = 's3://llc/CC/'+'LLC_CC_mask_{}.nc'.format(field_size[0])
        CC_mask = xr.load_dataset(ulmo_io.load_to_bytes(CC_mask_file))
    if verbose:
        print("Loaded LLC CC mask from {}".format(CC_mask_file))
    # Return
    return CC_mask


def grab_llc_datafile(datetime=None, root='LLC4320_', chk=True, local=False):
    """Generate the LLC datafile name from the inputs

    Args:
        datetime (pandas.TimeStamp, optional): Date. Defaults to None.
        root (str, optional): [description]. Defaults to 'LLC4320_'.
        chk (bool, optional): [description]. Defaults to True.
        local (bool, optional): [description]. Defaults to False.

    Returns:
        str: LLC datafile name
    """
    # Path
    llc_files_path = local_llc_files_path if local else s3_llc_files_path
        
    if datetime is not None:
        sdate = str(datetime).replace(':','_')[:19]
        # Add T?
        if sdate[10] == ' ':
            sdate = sdate.replace(' ', 'T')
        # Finish
        datafile = os.path.join(llc_files_path, root+sdate+'.nc')
    if chk and local:
        try:
            assert os.path.isfile(datafile)
        except:
            embed(header='34 of io')
    # Return
    return datafile
                    
def load_llc_ds(filename, local=False):
    """
    Args:
        filename: (str) path of the file to be read.
        local: (bool) flag to show if the file is local or not.
    Returns:
        ds: (xarray.Dataset) Dataset.
    """
    if not local:
        with ulmo_io.open(filename, 'rb') as f:
            ds = xr.open_dataset(f)
    else:
        ds = xr.open_dataset(filename, engine='h5netcdf')
    return ds

def grab_cutout(data_var, row, col, field_size=None, fixed_km=None,
                coords_ds=None, resize=False):
    if field_size is None and fixed_km is None:
        raise IOError("Must set field_size or fixed_km")
    if coords_ds is None:
        coords_ds = load_coords()
    # Setup
    R_earth = 6371. # km
    circum = 2 * np.pi* R_earth
    km_deg = circum / 360.

    if fixed_km is not None:
        dlat_km = (coords_ds.lat.data[row+1,col]-coords_ds.lat.data[row,col]) * km_deg
        dr = int(np.round(fixed_km / dlat_km))
    else:
        dr = field_size
    dc = dr

    cut_data = data_var[row:row+dr, col:col+dc]

    if resize:
        raise NotImplementedError("Need to resize..")

    # Return
    return cut_data

def grab_image(args):
    warnings.warn('Use grab_image() in utils.image_utils',
                  DeprecationWarning)
    return image_utils.grab_image(args)


def grab_velocity(cutout:pandas.core.series.Series, ds=None,
                  add_SST=False, add_Salt:bool=False, 
                  add_W=False, 
                  local_path:str=None):
    """Grab velocity

    Args:
        cutout (pandas.core.series.Series): cutout image
        ds (xarray.DataSet, optional): Dataset. Defaults to None.
        add_SST (bool, optional): Include SST too?. Defaults to False.
        add_Salt (bool, optional): Include Salt too?. Defaults to False.
        add_W (bool, optional): Include wz too?. Defaults to False.
        local_path (str, optional): Local path to data. Defaults to None.

    Returns:
        list: U, V cutouts as np.ndarray (i.e. values)
            and SST too if add_SST=True
            and Salt too if add_Salt=True
            and W too if add_W=True
    """
    # Local?with ulmo_io.open(cutout.filename, 'rb') as f:
    if local_path is None:
        filename = cutout.filename
    else:
        filename = os.path.join(local_path, os.path.basename(cutout.filename))
    # Open
    ds = xr.open_dataset(filename)

    # U field
    U_cutout = ds.U[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values
    # Vfield
    V_cutout = ds.V[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values
    output = [U_cutout, V_cutout]

    # Add SST?
    if add_SST:
        output.append(ds.Theta[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values)

    # Add Salt?
    if add_Salt:
        output.append(ds.Salt[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values)

    # Add W
    if add_W:
        output.append(ds.W[0, cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values)

    # Return
    return output
                    
def zarr_to_nc(timestamp: str, config_file: str, subset: str,
                field: str = None, channels: list = None,
                version: str = None, run_id: str = None):
    """Write netcdf from the S3 zarr store.

    Pass either `field` (single field, e.g. 'gradb2') or `channels` (list of
    field names for multi-channel subsets). The output path is placed under
    ``PATH/V{version}/YYYYMMDD_HHMMSS/``.  Use :func:`set_fronts_path` to
    override the root directory.
    """
    name = field if field is not None else subset
    full_path = derived_filename(timestamp, name, version=version)
    cfg = dbof_config.load_config(config_file)
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}
    dataset_name = (raw.get('subsets', {}).get(subset, {}).get('dataset_name')
                    or cfg.output.dataset_name)
    zarr_to_netcdf.main(
        os.path.dirname(full_path),
        output_filename=os.path.basename(full_path),
        mode='snapshots',
        run_id=run_id or cfg.run.run_id,
        s3_endpoint=cfg.output.s3_endpoint,
        bucket=cfg.output.bucket,
        channels=[field] if field is not None else channels,
        dates=cfg.data.date_iterations,
        dataset_name=dataset_name,
        folder=cfg.output.folder)
    return full_path
