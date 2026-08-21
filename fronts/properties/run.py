""" High-level routines to run bits and pieces of fronts.properties
"""
import os
import shutil
import sys
import subprocess
import yaml

import numpy as np
import xarray

from dbof.cli import generate_global
from dbof.global_dataset_creation.subset_definitions import (
    get_subset_definition, expand_channels_with_suffixes, valid_subsets,
)
from dbof.global_dataset_creation.iterations import (
    date_to_run_id, prefix_to_filename_date,
)

from fronts.finding import io as finding_io
from fronts.llc import io as llc_io
from fronts.llc import tiles as llc_tiles

from fronts.properties import io as properties_io
from fronts.properties import algorithms as prop_algorithms


def generate_global_dataset(config_file: str, netcdf_base: str,
                            ice_mask: bool = False, clobber: bool = False,
                            clobber_export: bool = False,
                            subsets: list = None, pipeline: str = None,
                            run_id: str = None,
                            generate_only: bool = False,
                            export_only: bool = False,
                            dry_run: bool = False):
    """Generate + export subsets via ``dbof.run_all_subsets``.

    Thin wrapper around the preprocessing batch driver (a CLI entry point),
    run as a subprocess in the current interpreter's environment.  Pipeline,
    run_id, subsets, dates, and depth_suffixes all come from *config_file*
    unless overridden here; outputs land under
    ``{netcdf_base}/{run_id}/{date_prefix}/``.  Existing subset/date zarr
    stores and channel NetCDFs are skipped unless clobbering.

    Args:
        config_file (str): Path to the (thin, global) YAML config.
        netcdf_base (str): Root dir for NetCDF output (e.g. the Fronts path).
        ice_mask (bool): NaN-mask ice-covered points during export.
        clobber (bool): Force BOTH phases — regenerate the zarr stores AND
            re-export every channel, even if they exist.
        clobber_export (bool): Force re-export of every channel NetCDF from the
            existing zarr stores, WITHOUT regenerating the stores.
        subsets (list, optional): Only process these subsets, overriding
            ``active_subsets`` in the YAML.  Used by build_v5 step 1 to build
            just the frontal-structure store.
        pipeline (str, optional): Override the ``pipeline`` key in the YAML.
        run_id (str, optional): Override ``run.run_id`` in the YAML.
        generate_only (bool): Build the zarr stores, skip the NetCDF export.
        export_only (bool): Export NetCDFs from existing stores, skip generate.
        dry_run (bool): Log the plan without doing anything.
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
    shutil.rmtree(ckpt_dir, ignore_errors=True)


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

    All paths are resolved from ``PATH/V{version}/YYYYMMDD_HHMMSS/``
    via :func:`fronts.llc.io.set_fronts_path`.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        config (str): Front-finding config label, e.g. 'A'.
        version (str): Data version string.
        property_names (list): Property field names to co-locate, e.g.
            ['relative_vorticity', 'strain_n']. Each must match both the
            variable name inside its .nc file and the filename pattern
            LLC4320_{timestamp}_{property_name}_{version}.nc.
        output_dir (str, optional): Output directory. Defaults to the
            standard fronts directory for this version + timestamp.
        stats (list, optional): Statistics to compute per property.
            Defaults to ['mean', 'std', 'median'].
        percentiles (list, optional): Percentiles to compute, e.g. [10, 90].
        min_npix (int): Minimum front size in pixels. Defaults to 1.
        nan_policy (str): 'omit' or 'propagate' NaNs. Defaults to 'omit'.
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

    ckpt_dir = os.path.join(fdir, f'colocate_ckpt_{run_tag}')
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
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def _resolve_channel_maps(config_file: str):
    """Resolve channel ↔ subset mappings from the thin global config.

    The ``subsets:`` block no longer lives in the YAML; the canonical channel
    lists live in ``dbof.global_dataset_creation.subset_definitions``, keyed by
    pipeline.  Depth (compute) channels are expanded with the active
    ``depth_suffixes`` (the YAML override wins; otherwise the per-subset
    default is used).  ``model_data_feature_channels`` and ``extra_channels``
    are never suffixed.

    Parameters
    ----------
    config_file : str
        Path to the (thin) global YAML config.

    Returns
    -------
    (channel_to_subset, root_to_expanded) : tuple[dict, dict]
        ``channel_to_subset`` maps every fully-expanded channel name to its
        subset.  ``root_to_expanded`` maps each *root* (base) name to the list
        of expanded channel names it produces under the active config.
    """
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}

    pipeline = raw.get('pipeline')
    if pipeline is None:
        raise ValueError(f"'pipeline' must be set in {config_file}")
    pipeline = pipeline.upper()

    # depth_suffixes: an explicit YAML key overrides the per-subset default,
    # but ONLY for subsets that actually carry a depth_suffixes key -- this
    # mirrors dbof.run_all_subsets (it applies the override only when
    # "depth_suffixes" in defn), so surface-only subsets (surface_wind,
    # icearea) keep bare channels.
    suffix_override = raw.get('depth_suffixes')   # None if absent

    # Restrict to the subsets the run actually produces, if listed.
    active = raw.get('active_subsets')
    if not active:
        single = raw.get('active_subset')
        active = [single] if single else valid_subsets(pipeline)

    channel_to_subset = {}
    root_to_expanded = {}
    for subset_name in active:
        defn = get_subset_definition(pipeline, subset_name)

        if suffix_override and ('depth_suffixes' in defn):
            eff_suffixes = suffix_override
        else:
            eff_suffixes = defn.get('depth_suffixes')

        compute = defn.get('compute_features_channels') or []
        model = defn.get('model_data_feature_channels') or []
        extra = defn.get('extra_channels') or []

        # Compute channels get suffix-expanded; model/extra stay bare.
        for base in compute:
            expanded = expand_channels_with_suffixes([base], eff_suffixes, None)
            root_to_expanded[base] = expanded
            for ch in expanded:
                channel_to_subset[ch] = subset_name
        for ch in list(model) + list(extra):
            root_to_expanded[ch] = [ch]
            channel_to_subset[ch] = subset_name

    return channel_to_subset, root_to_expanded


def expand_property_roots(property_roots: list, config_file: str) -> list:
    """Expand property *roots* into fully-suffixed channel names.

    Lets a caller list root names like ``'relative_vorticity'``
    and receive every variant the active config produces
    (``relative_vorticity_sfc``, ``relative_vorticity_mld``, ...), while
    channels that carry no suffix (``coriolis_f``, ``mixed_layer_depth``, native
    model fields) pass through unchanged.

    Parameters
    ----------
    property_roots : list of str
        Root/base channel names.  Already-expanded names are accepted too.
    config_file : str
        Path to the (thin) global YAML config.

    Returns
    -------
    list of str
        Fully-expanded channel names, order-preserving and de-duplicated.

    Raises
    ------
    ValueError
        If a root is unknown to the active pipeline/subsets.
    """
    channel_to_subset, root_to_expanded = _resolve_channel_maps(config_file)

    expanded, seen = [], set()
    unknown = []
    for root in property_roots:
        if root in root_to_expanded:
            names = root_to_expanded[root]
        elif root in channel_to_subset:
            names = [root]            # already an expanded channel name
        else:
            unknown.append(root)
            continue
        for ch in names:
            if ch not in seen:
                seen.add(ch)
                expanded.append(ch)

    if unknown:
        raise ValueError(
            f"These property roots are not in any active subset of "
            f"{config_file}: {unknown}"
        )
    return expanded


# ===========================================================================
#  Pipeline-aware config helpers
# ===========================================================================
#
#  Channel names and subset membership both depend on the pipeline: SURF and OSN
#  emit a bare 'gradb2', DEPTH emits 'gradb2_sfc', and the depth-resolved
#  subsets (stratification, ertel_pv, ...) have no surface equivalent at all.
#  Everything below derives from the pipeline + active_subsets in the YAML, so a
#  single driver runs on all three and picks up channels the moment they land in
#  subset_definitions.

#: Defaults for the optional ``build:`` block in a run YAML.
BUILD_DEFAULTS = {
    'build_version':    'V5',     # products land under {root}/{version}/{pipeline}/
                                  # (drivers override this with their own)
    'finding_config':   'D',      # fronts/finding/configs/finding_config_D.yaml
    'gradb2_root':      'gradb2',
    'finding_suffix':   'sfc',    # which depth suffix to find fronts in (DEPTH)
    'ice_mask_find':    False,    # step 1: mask gradb2 BEFORE finding fronts
    'ice_mask_props':   False,    # step 4: mask the co-located property fields
    'colocate_source':  'zarr',   # step 4 reads fields from: zarr | netcdf
    'tiles':            [],       # tile co-location: locations to process
    'tile_properties':  [],       # tile co-location: dbof.tiles.field_registry names
    'percentiles':      [25, 75, 90],
    'exclude_roots':    [],       # roots to leave out of co-location
}


def read_build_config(config_file: str, build_version: str = None) -> dict:
    """Read a run YAML into everything a build driver needs.

    Single source of truth: pipeline, run_id, dates and subsets all come from
    the config, so a driver script holds no run-specific state.

    Parameters
    ----------
    config_file : str
        Path to the (thin) global YAML config.
    build_version : str, optional
        Overrides ``build.build_version``.  Drivers pass their own version so
        the output directory is a property of the code that made the products,
        not of the dataset they were made from.

    Returns
    -------
    dict
        ``pipeline``, ``run_id``, ``active_subsets``, ``depth_suffixes``,
        ``date_iterations`` (ISO strings from the YAML), ``date_prefixes``
        (``YYYYMMDD_HHMMSS``), ``timestamps`` (``YYYY-MM-DDTHH_MM_SS``, the
        form used in every fronts filename), the S3 ``bucket``/``folder``, and
        ``run_dir`` (``{build_version}/{pipeline}``), plus every key in
        :data:`BUILD_DEFAULTS` merged with the YAML's ``build:`` block.
    """
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}

    pipeline = raw.get('pipeline')
    if pipeline is None:
        raise ValueError(f"'pipeline' must be set in {config_file}")
    pipeline = pipeline.upper()

    dates = (raw.get('data') or {}).get('date_iterations') or []
    if not dates:
        raise ValueError(f"'data.date_iterations' must be set in {config_file}")
    prefixes = [date_to_run_id(d) for d in dates]

    active = raw.get('active_subsets') or valid_subsets(pipeline)

    output = raw.get('output') or {}

    out = dict(BUILD_DEFAULTS)
    out.update(raw.get('build') or {})
    out.update({
        'pipeline':        pipeline,
        'run_id':          (raw.get('run') or {}).get('run_id'),
        'active_subsets':  list(active),
        'depth_suffixes':  raw.get('depth_suffixes'),
        'date_iterations': list(dates),
        'date_prefixes':   prefixes,
        'timestamps':      [prefix_to_filename_date(p) for p in prefixes],
        'bucket':          output.get('bucket', 'dbof/'),
        'folder':          output.get('folder'),
    })
    if build_version:
        out['build_version'] = build_version
    if not out['run_id']:
        raise ValueError(f"'run.run_id' must be set in {config_file}")
    # Products are organised by the build that made them; filenames keep the
    # source run_id so they stay traceable to the dataset they came from.
    out['run_dir'] = f"{out['build_version']}/{pipeline}"
    return out


def channel_for_root(config_file: str, root: str,
                     depth_suffix: str = 'sfc') -> str:
    """Resolve a root name to the ONE channel name this config produces.

    ``gradb2`` -> ``'gradb2'`` on SURF/OSN, ``'gradb2_sfc'`` on DEPTH.  Raises
    rather than guessing if the root is not produced by the active subsets.

    Parameters
    ----------
    config_file : str
        Path to the (thin) global YAML config.
    root : str
        Base channel name, e.g. ``'gradb2'``.
    depth_suffix : str
        Which suffix to pick when the root expands to several (DEPTH only).

    Returns
    -------
    str
        The fully-expanded channel name.
    """
    channel_to_subset, root_to_expanded = _resolve_channel_maps(config_file)

    if root in root_to_expanded:
        expanded = root_to_expanded[root]
    elif root in channel_to_subset:
        return root                      # already an expanded channel name
    else:
        raise ValueError(
            f"Root '{root}' is not produced by any active subset of "
            f"{config_file}.  Available roots: {sorted(root_to_expanded)}")

    if len(expanded) == 1:
        return expanded[0]

    want = f'{root}_{depth_suffix}'
    if want not in expanded:
        raise ValueError(
            f"Root '{root}' expands to {expanded} under {config_file}, which "
            f"does not include '{want}'.  Set build.finding_suffix to one of "
            f"{[c.split(root + '_')[-1] for c in expanded]}.")
    return want


def subset_for_channel(config_file: str, channel: str) -> str:
    """Return the dbof subset that produces *channel* under this config."""
    channel_to_subset, _ = _resolve_channel_maps(config_file)
    if channel not in channel_to_subset:
        raise ValueError(
            f"Channel '{channel}' is not produced by any active subset of "
            f"{config_file}.")
    return channel_to_subset[channel]


def all_property_roots(config_file: str, exclude: list = None) -> list:
    """Every property root the active subsets produce, in config order.

    Derived from ``subset_definitions``, so the set follows the pipeline and a
    channel added upstream is co-located automatically.

    Parameters
    ----------
    config_file : str
        Path to the (thin) global YAML config.
    exclude : list, optional
        Roots to leave out (e.g. heavy fields you don't want co-located).

    Returns
    -------
    list of str
        Root names, suitable for :func:`expand_property_roots`.
    """
    _, root_to_expanded = _resolve_channel_maps(config_file)
    drop = set(exclude or [])
    return [r for r in root_to_expanded if r not in drop]


def export_channels(config_file: str, timestamp: str, channels: list,
                    version: str, run_id: str = None,
                    ice_mask: bool = False, clobber: bool = False) -> list:
    """Export a hand-picked set of channels to per-channel NetCDF files.

    The narrow counterpart to :func:`generate_global_dataset`'s export phase:
    ``run_all_subsets`` exports *every* channel in a subset, which is wasteful
    when all you need is gradb2 (1 file instead of 8 on SURF, 21 on DEPTH).
    Channels are grouped by their owning subset automatically.

    Files land at
    ``{fronts_path}/{version}/{YYYYMMDD_HHMMSS}/LLC4320_{timestamp}_{channel}_{version}.nc``
    -- the same layout ``run_all_subsets`` writes, so the two are interchangeable.

    Parameters
    ----------
    config_file : str
        Path to the (thin) global YAML config.
    timestamp : str
        Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
    channels : list of str
        Fully-expanded channel names (see :func:`channel_for_root`).
    version : str
        Run tag (the run_id) used verbatim in the output path.
    run_id : str, optional
        Override the run_id used to locate the zarr store on S3.
    ice_mask : bool
        NaN-mask ice-covered points during the export.  Requires
        ``icearea.zarr`` for the same run_id + date.
    clobber : bool
        Re-export even if the .nc already exists.  Default skips.

    Returns
    -------
    list of str
        Paths of the NetCDF files that now exist for *channels*.
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


def generate_properties(timestamp: str, config_file: str, version: str,
                        property_names: list, run_id: str = None,
                        clobber: bool = False, create_zarr: bool = False):
    """Generate individual per-property .nc files for the requested properties.

    Resolves which dbof subset each property belongs to from the canonical
    ``subset_definitions`` (driven by the pipeline + active_subsets in the
    config), then writes one LLC4320_{timestamp}_{property}_{version}.nc file
    per property — the format expected by colocate_fronts(). Existing files are
    skipped unless clobber=True.

    ``property_names`` should be fully-expanded channel names (e.g.
    ``relative_vorticity_sfc``).  Use :func:`expand_property_roots` to turn a
    list of root names into the expanded set first.

    Use :func:`fronts.llc.io.set_fronts_path` to override the root
    directory.  Files land under ``PATH/V{version}/YYYYMMDD_HHMMSS/``.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        config_file (str): Path to the YAML config file.
        version (str): Data version string.
        property_names (list): Fully-expanded channel names to generate.
        run_id (str, optional): Override the run_id in the config YAML.
        clobber (bool): Overwrite existing output files. Defaults to False.
        create_zarr (bool): Create the zarr store via generate_global.
            Defaults to False (assumes zarr already exists on S3).
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
                 n_workers: int = None, skip_curvature: bool = False):
    """Label connected front components and compute geometric properties globally.

    All paths are resolved from ``PATH/V{version}/YYYYMMDD_HHMMSS/``
    via :func:`fronts.llc.io.set_fronts_path`.

    Args:
        timestamp (str): Snapshot timestamp, e.g. '2012-11-09T12_00_00'.
        config (str): Front-finding config label, e.g. 'A'.
        version (str): Data version string.
        n_workers (int, optional): Parallel workers. Defaults to CPU count.
        skip_curvature (bool): Skip curvature calculation (~50% faster).
    """
    fronts_file = finding_io.binary_filename(timestamp, config, version)
    coords_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'coords', 'LLC_coords_lat_lon.nc')
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