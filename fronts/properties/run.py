""" High-level routines to run bits and pieces of fronts.properties
"""
import os
import yaml

import numpy as np
import xarray

from dbof.cli import generate_global

from fronts.properties import algorithms as prop_algorithms
from fronts.finding import io as finding_io
from fronts.llc import io as llc_io


def generate_properties(timestamp: str, config_file: str, version: str,
                        property_names: list, run_id: str = None,
                        clobber: bool = False):
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

        generate_global.main(config_file, subset=subset, run_id=run_id)

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
    coords_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'coords', 'LLC_coords.nc')
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