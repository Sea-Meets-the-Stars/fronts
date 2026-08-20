""" Basic I/O routines for the LLC analysis """

import os
import yaml
import warnings

import numpy as np
import xarray as xr
import pandas

from dbof.cli import zarr_to_netcdf
from dbof.global_dataset_creation.config import default_output_folder
from dbof.global_dataset_creation.subset_definitions import get_subset_definition
from dbof.global_dataset_creation.zarr_dataset_global import GlobalZarrDatasetReader
from dbof.io.filesystems import create_s3_filesystems
from dbof.preprocessing.ice_mask import load_siarea_mask, apply_ice_mask


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

# Run layout.  ``_run_dir`` is the sub-path under the root that holds one
# build's products; ``_file_tag`` is the suffix stamped into every filename.
# They are separate so a build can be organised by its own version while the
# files stay named after the dataset they were derived from.  Both fall back to
# the ``version`` argument when unset.
_run_dir = None
_file_tag = None


def set_run_layout(run_dir: str, file_tag: str = None):
    """Set the output sub-path and filename tag for this run.

    ::

        set_fronts_path('/.../LLC/Fronts')
        set_run_layout('V5/SURF', file_tag='v2_2_01')
        # -> /.../LLC/Fronts/V5/SURF/20111204_000000/
        #        LLC4320_2011-12-04T00_00_00_gradb2_v2_2_01.nc

    Parameters
    ----------
    run_dir : str
        Sub-path under the Fronts root, e.g. ``'V5/SURF'``.
    file_tag : str, optional
        Filename suffix, e.g. the source ``run_id``.  Defaults to *run_dir*.
    """
    global _run_dir, _file_tag
    _run_dir = run_dir
    _file_tag = file_tag if file_tag is not None else run_dir


def clear_run_layout():
    """Fall back to using the ``version`` argument for both path and tag."""
    global _run_dir, _file_tag
    _run_dir = None
    _file_tag = None


def _resolve_run_dir(version: str) -> str:
    return _run_dir if _run_dir is not None else version


def _resolve_file_tag(version: str) -> str:
    return _file_tag if _file_tag is not None else version


def run_root(version: str = None, generate: bool = False) -> str:
    """Return the directory holding all timestamps for this run.

    ``{fronts_path}/{run_dir}`` -- the level above the per-timestamp folders,
    where run-wide files such as the ``.meta`` descriptor live.
    """
    d = os.path.join(get_fronts_path(), _resolve_run_dir(version))
    if generate:
        os.makedirs(d, exist_ok=True)
    return d


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
    """Build the run + timestamped output directory.

    Returns ``PATH / {version} / YYYYMMDD_HHMMSS`` and creates it if
    it does not exist.

    Parameters
    ----------
    version : str
        Run tag, used **verbatim** as the directory name.  This is the
        ``run_id`` (e.g. 'Vtest', 'vtest', 'global_DEPTH_test01') so the path
        matches dbof.run_all_subsets' output (``{netcdf_base}/{run_id}/...``).
        No ``V`` is prepended.
    timestamp : str
        Snapshot timestamp (e.g. '2012-11-09T12_00_00').
    generate : bool, optional
        Generate the directory if it does not exist. Defaults to False.

    Returns:
    --------
        str: The path to the directory.
    """
    ts_dir = _format_timestamp(timestamp)
    d = os.path.join(get_fronts_path(), _resolve_run_dir(version), ts_dir)
    if generate:
        os.makedirs(d, exist_ok=True)
    # Return
    return d

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

    # Generate base.  The tag is the run_id used verbatim (no 'V' prefix), so
    # it matches dbof.run_all_subsets, which names exported NetCDFs
    # LLC4320_{date}_{channel}_{run_id}.nc.
    basefile = f'{root}_{timestamp}_{field}_{_resolve_file_tag(version)}.nc'

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
                    
def store_args(config_file: str, subset: str, run_id: str = None) -> dict:
    """Resolve the S3 location of one subset's zarr store from a run config.

    Returns
    -------
    dict
        ``s3_endpoint``, ``bucket``, ``folder``, ``run_id``, ``dataset_name``.
    """
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}

    pipeline = raw.get('pipeline')
    if pipeline is None:
        raise ValueError(f"'pipeline' must be set in {config_file}")
    pipeline = pipeline.upper()

    output = raw.get('output') or {}
    return {
        's3_endpoint': output.get('s3_endpoint',
                                  'https://s3-west.nrp-nautilus.io'),
        'bucket': output.get('bucket', 'dbof/'),
        'folder': output.get('folder') or default_output_folder(pipeline),
        'run_id': run_id or (raw.get('run') or {}).get('run_id'),
        'dataset_name': (output.get('dataset_name')
                         or get_subset_definition(pipeline, subset)['dataset_name']),
    }


def read_channel(config_file: str, timestamp: str, subset: str, channel: str,
                 run_id: str = None, ice_mask: bool = False,
                 ice_mask_dataset_name: str = 'icearea.zarr') -> np.ndarray:
    """Read one channel of one snapshot straight from the S3 zarr store.

    The same array :func:`zarr_to_nc` would write to disk -- float32, no
    regridding -- so it can be used wherever the exported NetCDF was.

    Parameters
    ----------
    config_file : str
        Path to the (thin) global YAML config.
    timestamp : str
        Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
    subset : str
        dbof subset owning the channel, e.g. 'frontal_structure'.
    channel : str
        Fully-expanded channel name, e.g. 'gradb2' or 'N2_mld'.
    run_id : str, optional
        Override the run_id read from the config.
    ice_mask : bool
        NaN-mask ice-covered points (SIarea > 0).
    ice_mask_dataset_name : str
        Zarr store holding SIarea.  Only used when *ice_mask* is True.
    """
    args = store_args(config_file, subset, run_id)
    date_prefix = _format_timestamp(timestamp)
    _, fs = create_s3_filesystems(args['s3_endpoint'])

    reader = GlobalZarrDatasetReader(
        bucket=args['bucket'], folder=args['folder'], run_id=args['run_id'],
        dataset_name=args['dataset_name'], fs=fs, date_prefix=date_prefix)
    arr = reader.get_channel_snapshot(channel).astype(np.float32)

    if ice_mask:
        arr = apply_ice_mask(arr, load_siarea_mask(
            bucket=args['bucket'], folder=args['folder'],
            run_id=args['run_id'], date_prefix=date_prefix, fs=fs,
            dataset_name=ice_mask_dataset_name))
    return arr


def zarr_to_nc(timestamp: str, config_file: str, subset: str,
                field: str = None, channels: list = None,
                version: str = None, run_id: str = None,
                ice_mask: bool = False,
                ice_mask_dataset_name: str = 'icearea.zarr'):
    """Write netcdf from the S3 zarr store, for ONE timestamp.

    Pass either `field` (single field, e.g. 'gradb2_sfc') or `channels` (list
    of field names for multi-channel subsets). The output path is placed under
    ``PATH/{version}/YYYYMMDD_HHMMSS/``.  Use :func:`set_fronts_path` to
    override the root directory.

    The S3 location and zarr store name are resolved from the (thin, global)
    config YAML plus the canonical ``subset_definitions`` in the preprocessing
    package -- the ``subsets:`` block no longer lives in the YAML.

    Parameters
    ----------
    timestamp : str
        Snapshot timestamp, e.g. '2012-11-09T12_00_00'.  ONLY this snapshot is
        converted: its date_prefix is derived here and handed to
        ``zarr_to_netcdf`` explicitly, so the call is safe for a config holding
        any number of dates.
    config_file : str
        Path to the (thin) global YAML config.
    subset : str
        dbof subset owning the channel(s), e.g. 'frontal_structure'.
    field : str, optional
        Single fully-expanded channel name.
    channels : list, optional
        Multiple channel names (mutually exclusive with *field*).
    version : str
        Run tag (the run_id), used verbatim in the output path.
    run_id : str, optional
        Override the run_id read from the config.
    ice_mask : bool
        NaN-mask ice-covered points (SIarea > 0) during the export.  Requires
        ``icearea.zarr`` to exist for the same run_id + date_prefix.
    ice_mask_dataset_name : str
        Zarr store holding SIarea.  Only used when *ice_mask* is True.
    """
    name = field if field is not None else subset
    full_path = derived_filename(timestamp, name, version=version)
    args = store_args(config_file, subset, run_id)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    zarr_to_netcdf.main(
        os.path.dirname(full_path),
        output_filename=os.path.basename(full_path),
        mode='snapshots',
        run_id=args['run_id'],
        s3_endpoint=args['s3_endpoint'],
        bucket=args['bucket'],
        channels=[field] if field is not None else channels,
        date_prefix=_format_timestamp(timestamp),
        dataset_name=args['dataset_name'],
        folder=args['folder'],
        ice_mask=ice_mask,
        ice_mask_dataset_name=ice_mask_dataset_name)
    return full_path
