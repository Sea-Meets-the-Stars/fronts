""" High-level routines to run bits and pieces of fronts.properties
"""
import os

import numpy as np
import xarray
import shutil

from dbof.cli import generate_global

from fronts.finding import io as finding_io
from fronts.llc import io as llc_io
from fronts.llc import tiles as llc_tiles
from fronts.runs.config import _resolve_channel_maps, subset_for_channel

from fronts.properties import io as properties_io
from fronts.properties import algorithms as prop_algorithms


def _clear_checkpoints(ckpt_dir: str, strict: bool = False):
    """Remove a checkpoint cache, reporting rather than swallowing failure.

    With *strict* (a clobber), a cache that cannot be removed is fatal: its
    stale fragments would be served instead of the recomputed values, so the
    clobber would silently not clobber.
    """
    if not os.path.isdir(ckpt_dir):
        return
    try:
        shutil.rmtree(ckpt_dir)
    except OSError as exc:
        if strict:
            raise RuntimeError(
                f"clobber requested but the checkpoint cache {ckpt_dir} could "
                f"not be removed ({exc}); its stale fragments would be reused "
                f"instead of recomputed.  Delete it by hand and re-run."
            ) from exc
        print(f"WARNING: left behind {ckpt_dir}: {exc}")


def colocate_tile(timestamp: str, config: str, version: str,
                  property_names: list, tile,
                  output_dir: str = None, cache_dir: str = None,
                  stats: list = None, percentiles: list = None,
                  min_npix: int = 1, nan_policy: str = 'omit',
                  dilation_radius: int = 1, clobber: bool = False,
                  edge_margin: int = 0, level: int = 0, loader=None):
    """Co-locate the global fronts with properties computed on one tile.

    The label map is reoriented onto the tile's grid and each property is
    computed there on the fly, so nothing global is read or written.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-07-03T12_00_00'.
        config (str): Front-finding config label.
        version (str): Run tag (the run_id).
        property_names (list): Tile property names (see
            ``dbof.tiles.field_registry``).
        tile: TileInfo from :func:`fronts.llc.tiles.tile_for`.
        output_dir (str, optional): Defaults to ``{timestamp_dir}/tile{idx}/``.
        cache_dir (str, optional): Where the per-property tile NetCDFs live.
            Defaults to ``{output_dir}/fields/``.
        stats, percentiles, min_npix, nan_policy, dilation_radius: as
            :func:`colocate_fronts`.  Note percentiles and the median are not
            combinable across tiles; a front clipped by the tile edge has
            statistics for its clipped part only.
        clobber (bool): Overwrite existing output.
        edge_margin (int): Zero this many label cells at the tile rim.
        level (int): ``k`` index to co-locate.  Defaults to 0 (surface).
        loader (callable, optional): ``loader(name) -> 2D array`` on the tile
            grid, e.g. from :func:`fronts.llc.tiles.chunk_loader`.  Defaults to
            slicing the global full-depth store via
            :func:`fronts.llc.tiles.tile_loader`.
    """
    fdir = llc_io.fronts_dir(version, timestamp)
    fronts_file = finding_io.binary_filename(timestamp, config, version)
    time_str, run_tag, _ = prop_algorithms._parse_fronts_filename(fronts_file)

    if output_dir is None:
        output_dir = os.path.join(fdir, f'tile{tile.tile_idx:03d}')
    out_file = properties_io.get_global_front_output_path(
        output_dir, time_str, 'properties', run_tag)
    if os.path.isfile(out_file) and not clobber:
        print(f"Properties file {out_file} exists and clobber is False. Returning")
        return

    labeled = np.load(properties_io.get_global_front_output_path(
        fdir, time_str, 'label_map', run_tag))
    labels_tile = llc_tiles.labels_for_tile(labeled, tile,
                                            edge_margin=edge_margin)
    n_fronts = len(np.unique(labels_tile)) - 1
    print(f"Tile {tile.tile_idx} (face {tile.face_idx}): {n_fronts:,} fronts, "
          f"{(labels_tile > 0).sum():,} front pixels")
    if n_fronts < 1:
        print("No fronts in this tile; nothing to co-locate. Returning.")
        return

    ckpt_dir = os.path.join(output_dir, f'colocate_ckpt_{run_tag}')
    if clobber:
        _clear_checkpoints(ckpt_dir, strict=True)

    prop_algorithms.colocate_fronts(
        labeled=labels_tile,
        property_names=property_names,
        property_dir=output_dir,
        fronts_file=fronts_file,
        output_dir=output_dir,
        version=version,
        stats=stats,
        percentiles=percentiles,
        min_npix=min_npix,
        nan_policy=nan_policy,
        dilation_radius=dilation_radius,
        loader=loader or llc_tiles.tile_loader(
            timestamp, tile, cache_dir or os.path.join(output_dir, 'fields'),
            level=level),
        checkpoint_dir=ckpt_dir,
        extra_columns={'tile_idx': tile.tile_idx, 'face_idx': tile.face_idx},
    )
    _clear_checkpoints(ckpt_dir)


def _zarr_loader(config_file: str, timestamp: str, version: str,
                 ice_mask: bool = False):
    """Return ``loader(channel) -> ndarray`` reading from the S3 zarr stores."""
    def loader(channel):
        subset = subset_for_channel(config_file, channel)
        print(f"  reading {channel} from {subset}.zarr")
        return llc_io.read_channel(config_file, timestamp, subset, channel,
                                   run_id=version, ice_mask=ice_mask)
    return loader


def _zarr_channels(config_file: str, property_names: list) -> list:
    """Which of *property_names* the active subsets actually produce."""
    channel_to_subset, _ = _resolve_channel_maps(config_file)
    return [n for n in property_names if n in channel_to_subset]




def colocate_fronts(timestamp: str, config: str, version: str,
                    property_names: list,
                    output_dir: str = None,
                    stats: list = None, percentiles: list = None,
                    min_npix: int = 1, nan_policy: str = 'omit',
                    dilation_radius: int = 1, clobber: bool = False,
                    skip_missing: bool = False,
                    config_file: str = None, source: str = 'zarr',
                    ice_mask: bool = False):
    """Co-locate labeled fronts with physical property fields.

    Paths resolve from ``PATH/V{version}/YYYYMMDD_HHMMSS/`` via
    :func:`fronts.llc.io.set_fronts_path`.

    Args:
        timestamp (str): e.g. '2012-11-09T12_00_00'.
        config (str): Front-finding config label, e.g. 'A'.
        version (str): Data version string.
        property_names (list): Property fields; each must match its .nc variable
            name and filename ``LLC4320_{timestamp}_{name}_{version}.nc``.
        output_dir (str, optional): Defaults to the fronts dir for this run.
        stats (list, optional): Defaults to ['mean', 'std', 'median'].
        percentiles (list, optional): e.g. [10, 90].
        min_npix (int): Minimum front size in pixels.
        nan_policy (str): 'omit' or 'propagate'.
        dilation_radius (int): Pixels to dilate each front before stats.
            Defaults to 0.
        clobber (bool): Overwrite existing output. Defaults to False.
        skip_missing (bool): If True, silently drop requested properties that
            are absent (co-locate only what exists) instead of raising.
            Defaults to False (strict: raise if any are missing).
        config_file (str, optional): Path to the run YAML.  Required when
            source='zarr'.
        source (str): 'zarr' reads each field from the S3 store one at a time,
            writing no NetCDF; 'netcdf' reads the per-property .nc files in the
            timestamp directory.  Defaults to 'zarr'.
        ice_mask (bool): NaN-mask ice-covered points.  Only used by the zarr
            source; the .nc files were masked when they were exported.
    """
    fdir = llc_io.fronts_dir(version, timestamp)
    fronts_file = finding_io.binary_filename(timestamp, config, version)
    property_dir = fdir
    if output_dir is None:
        output_dir = fdir

    # Check if output already exists.  The run_tag must be derived from the
    # binary-fronts filename with the same parser group_fronts() used when it
    # wrote the label map, or the two disagree and the label map is not found.
    time_str, run_tag, _ = prop_algorithms._parse_fronts_filename(fronts_file)
    out_file = properties_io.get_global_front_output_path(
        output_dir, time_str, 'properties', run_tag)
    if os.path.isfile(out_file) and not clobber:
        print(f"Properties file {out_file} exists and clobber is False. Returning")
        return

    # Work out which properties are actually available, and how to read them.
    if source == 'zarr':
        if config_file is None:
            raise ValueError("source='zarr' requires config_file")
        loader = _zarr_loader(config_file, timestamp, version,
                              ice_mask=ice_mask)
        available = _zarr_channels(config_file, property_names)
        where = 'the zarr stores on S3'
    elif source == 'netcdf':
        loader = None                      # algorithms falls back to .nc
        available = [
            n for n in property_names
            if os.path.isfile(
                os.path.join(property_dir, f'LLC4320_{timestamp}_{n}_{version}.nc'))
        ]
        where = property_dir
    else:
        raise ValueError(f"source must be 'zarr' or 'netcdf', got {source!r}")

    missing = [n for n in property_names if n not in available]
    if missing:
        if skip_missing:
            print(f"WARNING: skipping {len(missing)} missing property/ies "
                  f"(co-locating only what exists): {missing}")
            property_names = available
            if not property_names:
                print(f"No requested properties present in {where}; "
                      f"nothing to co-locate. Returning.")
                return
        else:
            raise FileNotFoundError(
                f"Missing property/ies: {missing}\n"
                f"Not found in {where}.  Pass skip_missing=True to co-locate "
                f"only what exists."
            )

    # Load label map
    labeled_file = properties_io.get_global_front_output_path(
        fdir, time_str, 'label_map', run_tag)
    labeled = np.load(labeled_file)

    ckpt_dir = os.path.join(output_dir, f'colocate_ckpt_{run_tag}')
    if clobber:
        _clear_checkpoints(ckpt_dir, strict=True)

    prop_algorithms.colocate_fronts(
        labeled=labeled,
        property_names=property_names,
        property_dir=property_dir,
        fronts_file=fronts_file,
        output_dir=output_dir,
        version=version,
        stats=stats,
        percentiles=percentiles,
        min_npix=min_npix,
        nan_policy=nan_policy,
        dilation_radius=dilation_radius,
        loader=loader,
        checkpoint_dir=ckpt_dir,
    )
    _clear_checkpoints(ckpt_dir)


def generate_properties(timestamp: str, config_file: str, version: str,
                        property_names: list, run_id: str = None,
                        clobber: bool = False, create_zarr: bool = False):
    """Write one ``LLC4320_{timestamp}_{property}_{version}.nc`` per property.

    Each property is resolved to its dbof subset from ``subset_definitions``.
    Existing files are skipped unless clobber=True.

    Args:
        timestamp (str): e.g. '2012-11-09T12_00_00'.
        config_file (str): Run YAML.
        version (str): Data version string.
        property_names (list): Fully-expanded channel names
            (see :func:`fronts.runs.config.expand_property_roots`).
        run_id (str, optional): Override the run_id in the YAML.
        clobber (bool): Overwrite existing files.
        create_zarr (bool): Create the zarr store first via generate_global.
    """
    channel_to_subset, _ = _resolve_channel_maps(config_file)

    # Validate that all requested properties are known
    unknown = [p for p in property_names if p not in channel_to_subset]
    if unknown:
        raise ValueError(
            f"The following properties were not found in any active subset of "
            f"{config_file}: {unknown}"
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
                 n_workers: int = None, skip_curvature: bool = False,
                 coords_file: str = None):
    """Label connected front components and compute geometric properties globally.

    Paths resolve from ``PATH/V{version}/YYYYMMDD_HHMMSS/`` via
    :func:`fronts.llc.io.set_fronts_path`.

    Args:
        timestamp (str): e.g. '2012-11-09T12_00_00'.
        config (str): Front-finding config label, e.g. 'A'.
        version (str): Data version string.
        n_workers (int, optional): Parallel workers. Defaults to CPU count.
        skip_curvature (bool): Skip curvature calculation (~50% faster).
        coords_file (str, optional): NetCDF holding lat/lon (or YC/XC) on the
            SAME grid as the binary front map.  Defaults to the global rect
            coords file, which is 12960x17280 -- a tile run must pass its own
            720x720 coords instead (the tile gradb2 NetCDF carries XC/YC, so
            it serves).
    """
    fronts_file = finding_io.binary_filename(timestamp, config, version)
    if coords_file is None:
        coords_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts',
                                   'coords', 'LLC_coords_lat_lon.nc')
    output_dir = llc_io.fronts_dir(version, timestamp)

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
