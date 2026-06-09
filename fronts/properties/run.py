""" High-level routines to run bits and pieces of fronts.properties
"""
import os
import sys
import subprocess
import yaml

import numpy as np
import xarray

from dbof.cli import generate_global
from dbof.global_dataset_creation.subset_definitions import (
    get_subset_definition, expand_channels_with_suffixes, valid_subsets,
)

from fronts.finding import io as finding_io
from fronts.llc import io as llc_io

from fronts.properties import io as properties_io
from fronts.properties import algorithms as prop_algorithms


def generate_global_dataset(config_file: str, netcdf_base: str,
                            ice_mask: bool = False, clobber: bool = False):
    """Generate + export all active subsets via ``dbof.run_all_subsets``.

    Thin wrapper around the preprocessing batch driver (a CLI entry point),
    run as a subprocess in the current interpreter's environment.  Pipeline,
    run_id, subsets, dates, and depth_suffixes all come from *config_file*;
    outputs land under ``{netcdf_base}/{run_id}/{date_prefix}/``.  Existing
    subset/date zarr stores and channel NetCDFs are skipped unless *clobber*.

    Args:
        config_file (str): Path to the (thin, global) YAML config.
        netcdf_base (str): Root dir for NetCDF output (e.g. the Fronts path).
        ice_mask (bool): NaN-mask ice-covered points during export.
        clobber (bool): Force regenerate/re-export even if outputs exist.
    """
    cmd = [sys.executable, '-m', 'dbof.cli.run_all_subsets',
           '--config', config_file, '--netcdf-base', netcdf_base]
    if ice_mask:
        cmd.append('--ice-mask')
    if clobber:
        cmd.append('--clobber')
    subprocess.run(cmd, check=True)


def colocate_fronts(timestamp: str, config: str, version: str,
                    property_names: list,
                    output_dir: str = None,
                    stats: list = None, percentiles: list = None,
                    min_npix: int = 1, nan_policy: str = 'omit',
                    dilation_radius: int = 1, clobber: bool = False,
                    skip_missing: bool = False):
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
        skip_missing (bool): If True, silently drop requested properties whose
            .nc file is absent (co-locate only what exists) instead of raising.
            Defaults to False (strict: raise if any are missing).
    """
    fdir = llc_io.fronts_dir(version, timestamp)
    fronts_file = finding_io.binary_filename(timestamp, config, version)
    property_dir = fdir
    if output_dir is None:
        output_dir = fdir

    # Check if output already exists
    time_str = timestamp.replace('_', ':')   # '2012-11-09T12:00:00'
    run_tag  = f'{version}_bin_{config}'     # e.g. 'Vtest_bin_D' (version = run_id)
    out_file = properties_io.get_global_front_output_path(
        output_dir, time_str, 'properties', run_tag)
    if os.path.isfile(out_file) and not clobber:
        print(f"Properties file {out_file} exists and clobber is False. Returning")
        return

    # Validate all property files exist before doing any heavy work
    missing = [
        name for name in property_names
        if not os.path.isfile(
            os.path.join(property_dir, f'LLC4320_{timestamp}_{name}_{version}.nc'))
    ]
    if missing:
        if skip_missing:
            print(f"WARNING: skipping {len(missing)} missing property file(s) "
                  f"(co-locating only what exists): {missing}")
            property_names = [n for n in property_names if n not in missing]
            if not property_names:
                print(f"No requested property files present in {property_dir}; "
                      f"nothing to co-locate. Returning.")
                return
        else:
            raise FileNotFoundError(
                f"Missing property file(s) for: {missing}\n"
                f"Run generate_properties() first for the subset containing these fields, "
                f"or pass skip_missing=True to co-locate only what exists, "
                f"or check that property_dir is correct: {property_dir}"
            )

    # Load label map
    labeled_file = properties_io.get_global_front_output_path(
        fdir, time_str, 'label_map', run_tag)
    labeled = np.load(labeled_file)

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
    )


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

    Lets a caller (e.g. build_v4) list root names like ``'relative_vorticity'``
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