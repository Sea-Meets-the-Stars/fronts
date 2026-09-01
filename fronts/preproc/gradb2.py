""" Methods related to generating and processing a gradb2 field from LLC """

import os

import yaml

from dbof.cli import generate_global
from dbof.cli import zarr_to_netcdf
from dbof.tiles import tile_utils

from fronts.llc import io as llc_io


def generate_gradb2(timestamp: str, config_file: str, version:str=None,
    run_id: str = None, field: str = 'gradb2', clobber: bool = False,
    create_zarr: bool = False):
    """Generate the gradb2 field for the given config file.

    Args:
        timestamp (str): Timestamp of the data to process.
        version (str): Version of the data to process.
        config_file (str): Path to the YAML config file.
        run_id (str, optional): Override the run_id in the config YAML.
        field (str): Field name to extract.  For the DEPTH pipeline this is a
            suffixed channel, e.g. 'gradb2_sfc'.  Defaults to 'gradb2'.
        create_zarr (bool): Create the zarr store. Defaults to False.
        clobber (bool): Overwrite existing output. Defaults to False.
    """
    out_file = llc_io.derived_filename(timestamp, field, version=version)
    if os.path.isfile(out_file) and not clobber:
        print(f"gradb2 file {out_file} exists and clobber is False. Returning")
    else:
        # Create the zarr.  generate_global builds the full frontal_structure
        # store (all channels); the pipeline is read from the config YAML.
        if create_zarr:
            generate_global.main(config_file, subset='frontal_structure',
                run_id=run_id)
        # Create the netcdf for the requested gradb2 channel
        llc_io.zarr_to_nc(timestamp, config_file, 'frontal_structure',
            field, run_id=run_id, version=version)


def generate_tile_gradb2(date_iterations: list, timestamps: list, tile: dict,
    version: str = None, field: str = 'gradb2', clobber: bool = False,
    continue_on_error: bool = True) -> list:
    """Generate gradb2 for ONE tile across many timestamps, without a global store.

    The global path (:func:`generate_gradb2` / build_v5 step 1) builds the
    whole 12960x17280 frontal_structure zarr for every date and then exports
    one channel out of it.  For a single 720x720 tile that is enormously more
    work than the answer needs, and -- more to the point -- the hourly
    Theta/Salt it would need are not in any of the S3 stores.  This computes
    gradb2 directly from the raw surface tracers for the tile, one file per
    timestamp, written exactly where :func:`fronts.finding.run.find_gradb2_fronts`
    already looks for it.  Nothing downstream has to change.

    Args:
        date_iterations (list): Timestamps in dbof's ``DATE_FMT``
            ('YYYY-MM-DD HH:MM:SS') -- ``read_build_config``'s
            ``date_iterations``.
        timestamps (list): The SAME instants in the fronts filename form
            ('YYYY-MM-DDTHH_MM_SS') -- ``read_build_config``'s ``timestamps``.
            Same length and order as *date_iterations*.
        tile (dict): The config's ``build.tile_find`` block.  Location is
            either ``lon``/``lat`` or ``i_rect``/``j_rect`` (exactly one pair);
            optional ``property`` (default *field*) and ``pipeline``
            (default 'OSN' -- the public hourly kerchunk surface store, the
            only source with consecutive hourly Theta/Salt).
        version (str): Run tag; decides the output path via
            ``llc_io.derived_filename``.
        field (str): Channel name, and the data variable name inside each
            NetCDF.  'gradb2' on SURF/OSN, 'gradb2_sfc' on DEPTH.
        clobber (bool): Recompute timestamps whose file already exists.
        continue_on_error (bool): Log and skip a timestamp that fails rather
            than aborting the series.  Defaults True: a multi-week run over a
            public store is worth finishing.

    Returns:
        list: Paths of the NetCDF files that exist when the run finishes.
    """
    if len(date_iterations) != len(timestamps):
        raise ValueError(
            f"date_iterations ({len(date_iterations)}) and timestamps "
            f"({len(timestamps)}) must line up one-for-one")

    out_files = [llc_io.derived_filename(ts, field, version=version)
                 for ts in timestamps]

    return tile_utils.run_series(
        list(date_iterations),
        lon=tile.get('lon'), lat=tile.get('lat'),
        i_rect=tile.get('i_rect'), j_rect=tile.get('j_rect'),
        property=tile.get('property', field),
        pipeline=tile.get('pipeline', 'OSN'),
        output_paths=out_files,
        clobber=clobber,
        continue_on_error=continue_on_error,
    )
