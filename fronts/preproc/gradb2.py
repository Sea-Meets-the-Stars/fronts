""" Methods related to generating and processing a gradb2 field from LLC """

import os

import yaml

import dbof.dataset_creation.config as dbof_config
from dbof.cli import generate_global
from dbof.cli import zarr_to_netcdf

from fronts.llc import io as llc_io

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

def generate_gradb2(timestamp: str, config_file: str, version:str=None, 
    run_id: str = None, field: str = 'gradb2', clobber: bool = False):
    """Generate the gradb2 field for the given config file.

    Args:
        timestamp (str): Timestamp of the data to process.
        version (str): Version of the data to process.
        config_file (str): Path to the YAML config file.
        run_id (str, optional): Override the run_id in the config YAML.
        field (str): Field name to extract. Defaults to 'gradb2'.
        clobber (bool): Overwrite existing output. Defaults to False.
    """
    out_file = llc_io.derived_filename(timestamp, field, version=version)
    if os.path.isfile(out_file) and not clobber:
        print(f"gradb2 file {out_file} exists and clobber is False. Returning")
    else:
        generate_global.main(config_file, subset='frontal_structure', run_id=run_id)
        _zarr_to_nc(timestamp, config_file, 'frontal_structure', field, run_id=run_id)
