""" High-level routines to run bits and pieces of fronts.properties
"""
import os
import yaml

import numpy as np
import xarray

from dbof.cli import generate_global

from fronts.finding import io as finding_io
from fronts.config import io as config_io
from fronts.llc import io as llc_io

from fronts.properties import io as properties_io
from fronts.properties import algorithms as prop_algorithms


def colocate_fronts(timestamp: str, fconfig: str, version: str,
                    property_names: list, property_dir: str,
                    output_dir: str = None,
                    #stats: list = None, percentiles: list = None,
                    #min_npix: int = 1, nan_policy: str = 'omit',
                    #dilation_radius: int = 1,
                    clobber: bool = False):
    """Co-locate labeled fronts with physical property fields.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        fconfig (str): Front config label, e.g. 'A'.
        version (str): Data version string.
        property_names (list): Property field names to co-locate, e.g.
            ['relative_vorticity', 'strain_n']. Each must match both the
            variable name inside its .nc file and the filename pattern
            LLC4320_{timestamp}_{property_name}_{version}.nc.
        property_dir (str): Directory containing property .nc files.
        output_dir (str, optional): Output directory. Defaults to the
            standard group_fronts output directory for this version.
        stats (list, optional): Statistics to compute per property.
            Defaults to ['mean', 'std', 'median'].
        percentiles (list, optional): Percentiles to compute, e.g. [10, 90].
        min_npix (int): Minimum front size in pixels. Defaults to 1.
        nan_policy (str): 'omit' or 'propagate' NaNs. Defaults to 'omit'.
        dilation_radius (int): Pixels to dilate each front before stats.
            Defaults to 1.
        clobber (bool): Overwrite existing output. Defaults to False.
    """
    # Load front config file
    fconfig_file = config_io.config_filename(fconfig)
    cdict = config_io.load(fconfig_file)

    fronts_file = finding_io.binary_filename(timestamp, fconfig, version)
    group_dir   = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts',
                               'group_fronts', f'v{version}')
    if output_dir is None:
        output_dir = group_dir

    # Check if output already exists
    time_str = timestamp.replace('_', ':')   # '2012-11-09T12:00:00'
    run_tag  = f'v{version}_bin_{fconfig}'    # e.g. 'v1_bin_A'
    out_file = properties_io.get_global_front_output_path(
        output_dir, time_str, 'properties', run_tag)
    if os.path.isfile(out_file) and not clobber:
        print(f"Properties file {out_file} exists and clobber is False. Returning")
        return

    # Validate all property files exist before doing any heavy work
    missing = [
        name for name in property_names
        if not os.path.isfile(
            os.path.join(property_dir, f'LLC4320_{timestamp}_{name}_v{version}.nc'))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing property file(s) for: {missing}\n"
            f"Run generate_properties() first for the subset containing these fields, "
            f"or check that property_dir is correct: {property_dir}"
        )

    # Load label map
    labeled_file = properties_io.get_global_front_output_path(
        group_dir, time_str, 'label_map',run_tag)
    labeled = np.load(labeled_file)

    prop_algorithms.colocate_fronts(
        labeled=labeled,
        property_names=property_names,
        property_dir=property_dir,
        fronts_file=fronts_file,
        output_dir=output_dir,
        stats=cdict['properties']['stats'],
        percentiles=cdict['properties']['percentiles'],
        min_npix=cdict['properties']['min_npix'],
        nan_policy=cdict['properties']['nan_policy'],
        dilation_radius=cdict['properties']['dilation_radius'],
    )


def generate_properties(timestamp: str, config_file: str, version: str,
                        property_names: list, run_id: str = None,
                        clobber: bool = False, create_zarr: bool = False):
    """Generate individual per-property .nc files for the requested properties.

    Resolves which dbof subset each property belongs to from the YAML config,
    then writes one LLC4320_{timestamp}_{property}_v{version}.nc file per
    property — the format expected by colocate_fronts(). Existing files are
    skipped unless clobber=True.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        config_file (str): Path to the YAML config file.
        version (str): Data version string.
        property_names (list): Property/channel names to generate, e.g.
            ['relative_vorticity', 'strain_n'].
        run_id (str, optional): Override the run_id in the config YAML.
        clobber (bool): Overwrite existing output files. Defaults to False.
        create_zarr (bool): Create the zarr store via generate_global.
            Defaults to False (assumes zarr already exists on S3).
    """
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}

    # Build channel → subset mapping from the YAML
    # this is so the user can input which properties they want
    # and the 'subset' is determined by the code from YAML
    channel_to_subset = {}
    for subset_name, sub_cfg in (raw.get('subsets') or {}).items():
        for ch in sub_cfg.get('compute_features_channels', []):
            ch = ch.strip()
            if ch:
                channel_to_subset[ch] = subset_name
        for ch in sub_cfg.get('model_data_feature_channels', []):
            ch = ch.strip()
            if ch:
                channel_to_subset[ch] = subset_name

    # Validate that all requested properties are known
    unknown = [p for p in property_names if p not in channel_to_subset]
    if unknown:
        raise ValueError(
            f"The following properties were not found in any subset of {config_file}: {unknown}"
        )

    # Group requested properties by subset so generate_global runs once per subset
    subset_to_channels = {}
    for prop in property_names:
        subset_to_channels.setdefault(channel_to_subset[prop], []).append(prop)

    # Process each subset
    for subset, channels in subset_to_channels.items():
        missing = [ch for ch in channels
                   if not os.path.isfile(llc_io.derived_filename(timestamp, ch, version=version))]

        if not missing and not clobber:
            print(f"All {len(channels)} property file(s) for subset '{subset}' exist "
                  f"and clobber is False. Skipping.")
            continue

        to_generate = channels if clobber else missing
        print(f"Generating {len(to_generate)} property file(s) from subset '{subset}'")

        # Create the zarr store if requested; otherwise assume it exists on S3
        if create_zarr:
            generate_global.main(config_file, subset=subset, run_id=run_id)

        # Convert zarr → netcdf for each channel
        for channel in to_generate:
            llc_io.zarr_to_nc(timestamp, config_file, subset, field=channel,
                        version=version, run_id=run_id)

def group_fronts(timestamp: str, config: str, version: str,
                 n_workers: int = None, skip_curvature: bool = False):
    """Label connected front components and compute geometric properties globally.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        config (str): Front-finding config label, e.g. 'A'.
        version (str): Data version string.
        n_workers (int, optional): Parallel workers. Defaults to CPU count.
        skip_curvature (bool): Skip curvature calculation (~50% faster).
    """
    fronts_file = finding_io.binary_filename(timestamp, config, version)
    coords_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'coords', 'LLC_coords_lat_lon.nc')
    output_dir  = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts',
                               'group_fronts', f'v{version}')

    # Load
    fronts_binary = np.load(fronts_file)
    ds = xarray.open_dataset(coords_file)
    lat = ds['lat'].values if 'lat' in ds else ds['YC'].values
    lon = ds['lon'].values if 'lon' in ds else ds['XC'].values
    ds.close()

    prop_algorithms.group_fronts(
        fronts_binary, lat, lon,
        fronts_file=fronts_file,
        output_dir=output_dir,
        n_workers=n_workers,
        skip_curvature=skip_curvature,
    )