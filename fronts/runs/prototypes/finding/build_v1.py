# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os
import sys
import pathlib

import yaml
import numpy as np
import xarray

from dbof.cli import generate_global
from dbof.cli import zarr_to_netcdf
import dbof.dataset_creation.config as dbof_config

from fronts.preproc import inpaint_edges
from fronts.llc import io as llc_io
from fronts.finding import algorithms
from fronts.finding import config as find_config
from fronts.finding import io as finding_io
from fronts.properties import io as properties_io
from fronts.properties import colocation

# group_fronts_global lives in dev/ (no package __init__)
sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / 'dev'))
from group_fronts_global import main as group_fronts_global_main

from IPython import embed


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _zarr_to_nc(timestamp: str, config_file: str, subset: str, field: str,
                version: str = '1', run_id: str = None):
    """Write a single-field netcdf from the S3 zarr store."""
    full_path = llc_io.derived_filename(timestamp, field, version=version)
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
        channels=[field],
        dates=cfg.data.date_iterations,
        dataset_name=dataset_name,
        folder=cfg.output.folder)
    return full_path


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def generate_gradb2(timestamp: str, config_file: str, run_id: str = None,
                    field: str = 'gradb2'):
    """Generate the gradb2 field for the given config file.

    Args:
        timestamp (str): Timestamp of the data to process.
        config_file (str): Path to the YAML config file.
        run_id (str, optional): Override the run_id in the config YAML.
        field (str): Field name to extract. Defaults to 'gradb2'.
    """
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
    fronts = algorithms.fronts_from_gradb2(gradb2, **bparam)

    # Save em
    finding_io.save_binary_fronts(
        fronts, timestamp, config, version)


def group_fronts(timestamp: str, config: str, version: str,
                 n_workers: int = None, skip_curvature: bool = False):
    """Label connected front components and compute geometric properties.

    Wraps group_fronts_global.main(), deriving all paths from env vars and
    the binary front filename conventions.

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
    group_fronts_global_main(
        fronts_file=fronts_file,
        coords_file=coords_file,
        output_dir=output_dir,
        n_workers=n_workers,
        skip_curvature=skip_curvature,
    )


def characterize_fronts(timestamp: str, config_file: str, version: str,
                        run_id: str = None, subset: str = 'kinematic'):
    """Generate physical property fields and compute per-front statistics.

    (a) Calls generate_global to produce the requested property subset via
        dbof, then writes to netcdf via zarr_to_netcdf — same pattern as
        generate_gradb2 but for a different subset.
    (b) Loads the labeled-front array from the group_fronts output, colocates
        each front with the property fields, and saves per-front statistics
        (mean, std, median) as a parquet file alongside the group_fronts outputs.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        config_file (str): Path to the YAML config file.
        version (str): Data version string.
        run_id (str, optional): Override the run_id in the config YAML.
        subset (str): dbof subset to generate. Defaults to 'kinematic'.
    """
    # (a) Generate property fields -----------------------------------------
    generate_global.main(config_file, subset=subset, run_id=run_id)

    # Resolve channels and dataset_name from the subset entry
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}
    sub_cfg = raw.get('subsets', {}).get(subset, {})
    channels = ([c.strip() for c in sub_cfg.get('compute_features_channels', []) if c.strip()]
                + [c.strip() for c in sub_cfg.get('model_data_feature_channels', []) if c.strip()])
    dataset_name = sub_cfg.get('dataset_name') or raw.get('output', {}).get('dataset_name')

    cfg = dbof_config.load_config(config_file)
    prop_file = llc_io.derived_filename(timestamp, subset, version=version)
    zarr_to_netcdf.main(
        os.path.dirname(prop_file),
        output_filename=os.path.basename(prop_file),
        mode='snapshots',
        run_id=run_id or cfg.run.run_id,
        s3_endpoint=cfg.output.s3_endpoint,
        bucket=cfg.output.bucket,
        channels=channels or None,
        dates=cfg.data.date_iterations,
        dataset_name=dataset_name,
        folder=cfg.output.folder)

    # (b) Colocate fronts with properties ----------------------------------
    group_dir   = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts',
                               'group_fronts', f'v{version}')
    # get_global_front_output_path expects colons in the time string
    time_str    = timestamp.replace('_', ':')   # '2012-11-09T12:00:00'
    labeled_file = properties_io.get_global_front_output_path(
        group_dir, time_str, 'labeled')
    labeled = np.load(labeled_file)

    ds = xarray.open_dataset(prop_file)
    properties = {var: ds[var].values for var in ds.data_vars}
    ds.close()

    df = colocation.colocate_fronts_with_properties(
        labeled, properties,
        stats=['mean', 'std', 'median'],
        nan_policy='omit',
    )

    out_file = os.path.join(
        group_dir, f'LLC4320_{timestamp}_{subset}_v{version}_front_stats.parquet')
    df.to_parquet(out_file, index=False)
    print(f"Wrote: {out_file}")


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

    # Characterize fronts (physical properties + per-front stats)
    if flg == 4:
        timestamp   = '2012-11-09T12_00_00'
        config_file = './testing_global_v1.yaml'
        run_id      = 'year_1xglobal_20260306_050000'
        version     = '1'
        characterize_fronts(timestamp, config_file, version, run_id=run_id)

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
