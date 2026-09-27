"""Build the LLC4320 zarr stores on S3 (via ``dbof``) and export channels to NetCDF.
"""
import os
import sys
import subprocess

from dbof.global_dataset_creation import check_existence
from dbof.global_dataset_creation.config import default_output_folder
from dbof.global_dataset_creation.subset_definitions import get_subset_definition
from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix
from dbof.io.filesystems import create_s3_filesystems

from fronts.llc import io as llc_io
from fronts.runs.config import read_build_config, subset_for_channel


def generate_global_dataset(config_file: str, netcdf_base: str,
                            ice_mask: bool = False, clobber: bool = False,
                            clobber_export: bool = False,
                            subsets: list = None, pipeline: str = None,
                            run_id: str = None,
                            generate_only: bool = False,
                            export_only: bool = False,
                            dry_run: bool = False):
    """Generate + export subsets by running ``dbof.cli.run_all_subsets`` as a subprocess.

    Pipeline, run_id, subsets and dates come from *config_file* unless
    overridden.  Existing stores/NetCDFs are skipped unless clobbering.

    Args:
        config_file (str): Run YAML.
        netcdf_base (str): Root dir for NetCDF output.
        ice_mask (bool): NaN-mask ice-covered points on export.
        clobber (bool): Regenerate stores AND re-export.
        clobber_export (bool): Re-export only; keep existing stores.
        subsets (list, optional): Override ``active_subsets``.
        pipeline (str, optional): Override ``pipeline``.
        run_id (str, optional): Override ``run.run_id``.
        generate_only (bool): Build stores, skip export.
        export_only (bool): Export from existing stores, skip generate.
        dry_run (bool): Print the plan only.
    """
    cmd = [sys.executable, '-m', 'dbof.cli.run_all_subsets',
           '--config', config_file, '--netcdf-base', netcdf_base]
    if pipeline:
        cmd += ['--pipeline', pipeline]
    if run_id:
        cmd += ['--run-id', run_id]
    if subsets:
        cmd += ['--subsets'] + list(subsets)
    if ice_mask:
        cmd.append('--ice-mask')
    if clobber:
        cmd.append('--clobber')
    if clobber_export:
        cmd.append('--clobber-export')
    if generate_only:
        cmd.append('--generate-only')
    if export_only:
        cmd.append('--export-only')
    if dry_run:
        cmd.append('--dry-run')
    print('Running: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def generate_for_channels(config_file: str, netcdf_base: str,
                          channels_by_subset: dict, run_id: str = None):
    """Build only the stores that do not already hold the requested channels.

    Each store is checked against *just* the channels listed (not the subset's
    full list), so a store missing a newer upstream channel still counts as
    ready.  Only the first date is checked; a run's dates share channels.

    Args:
        config_file (str): Run YAML.
        netcdf_base (str): Passed through to ``run_all_subsets``.
        channels_by_subset (dict): ``{subset: [channel, ...]}``.
        run_id (str, optional): Override the run_id used to locate stores.
    """
    cfg = read_build_config(config_file)
    folder = cfg['folder'] or default_output_folder(cfg['pipeline'])
    tag = run_id or cfg['run_id']
    _, fs = create_s3_filesystems(cfg['s3_endpoint'])

    for subset, channels in channels_by_subset.items():
        store = make_run_prefix(
            cfg['bucket'], folder, tag,
            get_subset_definition(cfg['pipeline'], subset)['dataset_name'],
            date_prefix=cfg['date_prefixes'][0])
        state = check_existence.plan_zarr(fs, store, list(channels))
        if state == check_existence.ZARR_FULL:
            print(f"  SKIP (store serves {', '.join(channels)})  {subset}")
            continue
        print(f"  GENERATE  {subset}  ({state})")
        generate_global_dataset(config_file, netcdf_base,
                                subsets=[subset], generate_only=True)


def export_channels(config_file: str, timestamp: str, channels: list,
                    version: str, run_id: str = None,
                    ice_mask: bool = False, clobber: bool = False) -> list:
    """Export selected channels to per-channel NetCDF files.

    Files land at ``{fronts_path}/{version}/{YYYYMMDD_HHMMSS}/LLC4320_{timestamp}_{channel}_{version}.nc``.

    Args:
        config_file (str): Run YAML.
        timestamp (str): e.g. '2012-11-09T12_00_00'.
        channels (list): Fully-expanded channel names.
        version (str): Run tag used in the output path.
        run_id (str, optional): Override the run_id used to locate stores.
        ice_mask (bool): NaN-mask ice-covered points; needs ``icearea.zarr``.
        clobber (bool): Re-export existing files.

    Returns:
        list: Paths of the NetCDF files for *channels*.
    """
    written = []
    for channel in channels:
        subset = subset_for_channel(config_file, channel)
        out = llc_io.derived_filename(timestamp, channel, version=version)
        if os.path.isfile(out) and not clobber:
            print(f"  SKIP (exists)  {os.path.basename(out)}")
            written.append(out)
            continue
        print(f"  EXPORT  {channel}  (subset={subset}"
              f"{', ice-masked' if ice_mask else ''})  ->  {out}")
        llc_io.zarr_to_nc(timestamp, config_file, subset, field=channel,
                          version=version, run_id=run_id,
                          ice_mask=ice_mask)
        written.append(out)
    return written
