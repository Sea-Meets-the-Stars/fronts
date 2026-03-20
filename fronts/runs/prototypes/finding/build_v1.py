# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os

import yaml
import numpy as np
import xarray
import sys

from dbof.cli import generate_global
from dbof.cli import zarr_to_netcdf
import dbof.dataset_creation.config as dbof_config

from fronts.preproc import inpaint_edges
from fronts.llc import io as llc_io
from fronts.finding import algorithms as finding_algorithms
from fronts.finding import config as find_config
from fronts.finding import io as finding_io
from fronts.properties import algorithms as prop_algorithms
from fronts.properties import io as properties_io

from IPython import embed


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _zarr_to_nc(timestamp: str, config_file: str, subset: str,
                field: str = None, channels: list = None,
                version: str = '1', run_id: str = None):
    """Write netcdf from the S3 zarr store.

    Pass either `field` (single field, e.g. 'gradb2') or `channels` (list of
    field names for multi-channel subsets). The output path is derived from
    `field` if provided, otherwise from `subset`.
    """
    name = field if field is not None else subset
    full_path = llc_io.derived_filename(timestamp, name, version=version)
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


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def generate_gradb2(timestamp: str, config_file: str, run_id: str = None,
                    field: str = 'gradb2', clobber: bool = False):
    """Generate the gradb2 field for the given config file.

    Args:
        timestamp (str): Timestamp of the data to process.
        config_file (str): Path to the YAML config file.
        run_id (str, optional): Override the run_id in the config YAML.
        field (str): Field name to extract. Defaults to 'gradb2'.
        clobber (bool): Overwrite existing output. Defaults to False.
    """
    out_file = llc_io.derived_filename(timestamp, field, version='1')
    if os.path.isfile(out_file) and not clobber:
        print(f"gradb2 file {out_file} exists and clobber is False. Returning")
    else:
        generate_global.main(config_file, subset='frontal_structure', run_id=run_id)
        _zarr_to_nc(timestamp, config_file, 'frontal_structure', field, run_id=run_id)


def find_fronts(timestamp: str, config: str, version: str, inpaint: bool = False,
                clobber: bool = False):
    """ Find us the fronts

    Args:
        timestamp (str): _description_
        config (str): _description_
        version (str): _description_
        inpaint (bool, optional): _description_. Defaults to False.
        clobber (bool, optional): _description_. Defaults to False.
    """

    # Check if the binary front field exists
    bfile = finding_io.binary_filename(timestamp, config, version)
    if os.path.isfile(bfile) and not clobber:
        print(f"Binary front field {bfile} exists and clobber is False. Returning")
        return

    # Load gradb2
    gradb2_file = llc_io.derived_filename(timestamp, 'gradb2', version=version)
    print(f"Loading gradb2 from: {gradb2_file}")
    gradb2 = xarray.open_dataset(gradb2_file)['gradb2'].values

    print(f"Loaded gradb2 with shape: {gradb2.shape}")

    # Inpaint
    if inpaint:
        print("Inpainting...")
        gradb2 = inpaint_edges.inpaint(gradb2, method='biharmonic',
                         second_pass='regular',
                         second_threshold=1e-20)
        print("Inpainted.")

    print(f"Processing config: {config}")

    # Load config
    config_file = find_config.config_filename(config)
    cdict = find_config.load(config_file)

    # Binary parameters
    bparam = cdict['binary']
    bparam['n_workers'] = 10
    bparam['verbose'] = True

    # Do it
    fronts = finding_algorithms.fronts_from_gradb2(gradb2, **bparam)

    # Save em
    finding_io.save_binary_fronts(
        fronts, timestamp, config, version)


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
    coords_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'LLC_coords.nc')
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
            _zarr_to_nc(timestamp, config_file, subset, field=channel,
                        version=version, run_id=run_id)


def colocate_fronts(timestamp: str, config: str, version: str,
                    property_names: list, property_dir: str,
                    output_dir: str = None,
                    stats: list = None, percentiles: list = None,
                    min_npix: int = 1, nan_policy: str = 'omit',
                    dilation_radius: int = 0, clobber: bool = False):
    """Co-locate labeled fronts with physical property fields.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        config (str): Front-finding config label, e.g. 'A'.
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
            Defaults to 0.
        clobber (bool): Overwrite existing output. Defaults to False.
    """
    fronts_file = finding_io.binary_filename(timestamp, config, version)
    group_dir   = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts',
                               'group_fronts', f'v{version}')
    if output_dir is None:
        output_dir = group_dir

    # Check if output already exists
    time_str = timestamp.replace('_', ':')   # '2012-11-09T12:00:00'
    run_tag  = f'v{version}_bin_{config}'    # e.g. 'v1_bin_A'
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
        group_dir, time_str, 'label_map')
    labeled = np.load(labeled_file)

    prop_algorithms.colocate_fronts(
        labeled=labeled,
        property_names=property_names,
        property_dir=property_dir,
        fronts_file=fronts_file,
        output_dir=output_dir,
        stats=stats,
        percentiles=percentiles,
        min_npix=min_npix,
        nan_policy=nan_policy,
        dilation_radius=dilation_radius,
    )


# #######################################################
def main(flg: str):
    flg = int(flg)

    # Generate gradb2 as zarr → netcdf
    if flg == 1:
        timestamp   = '2012-11-09T12_00_00'
        config_file = './testing_global_v1.yaml'
        run_id      = 'year_1xglobal_20260306_050000'
        generate_gradb2(timestamp, config_file, run_id=run_id)

    # Find fronts -- binary pixels
    if flg == 2:
        timestamp = '2012-11-09T12_00_00'
        version   = '1'
        configs   = ['A', 'B', 'C']
        for config in configs:
            find_fronts(timestamp, config, version)

    # Group fronts (label + geometric properties)
    if flg == 3:
        timestamp = '2012-11-09T12_00_00'
        version   = '1'
        configs   = ['A', 'B', 'C']
        for config in configs:
            group_fronts(timestamp, config, version)

    # Generate physical property fields → individual per-property nc files
    if flg == 4:
        timestamp   = '2012-11-09T12_00_00'
        config_file = './testing_global_v1.yaml'
        run_id      = 'year_1xglobal_20260306_050000'
        version     = '1'
        property_names = [
            'relative_vorticity', 'divergence', 'strain_mag',
            'frontogenesis_tendency', 'okubo_weiss',
        ]
        generate_properties(timestamp, config_file, version,
                            property_names=property_names, run_id=run_id)

    # Co-locate fronts with physical properties
    if flg == 5:
        timestamp     = '2012-11-09T12_00_00'
        version       = '1'
        configs       = ['A', 'B', 'C']
        property_dir  = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'derived')
        property_names = [
            'relative_vorticity', 'divergence', 'strain_mag',
            'frontogenesis_tendency', 'okubo_weiss',
        ]
        for config in configs:
            colocate_fronts(timestamp, config, version,
                            property_names=property_names,
                            property_dir=property_dir)

    # ###########################
    # Testing

    # Test relative vorticity zarr → netcdf
    if flg == 102:
        timestamp   = '2012-11-09T12_00_00'
        config_file = './testing_global_v1.yaml'
        _zarr_to_nc(timestamp, config_file, 'kinematic', 'relative_vorticity')


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]

    main(flg)
