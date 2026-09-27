"""Read a run YAML and resolve pipeline-dependent channel names.

Channel names follow the pipeline (SURF/OSN emit ``gradb2``, DEPTH emits
``gradb2_sfc``); everything here derives from ``pipeline`` + ``active_subsets``
in the YAML and ``dbof.global_dataset_creation.subset_definitions``.

Not to be confused with :mod:`fronts.finding.config` (front-detection parameters).
"""
import yaml

from dbof.global_dataset_creation.subset_definitions import (
    get_subset_definition, expand_channels_with_suffixes, valid_subsets,
)
from dbof.global_dataset_creation.iterations import (
    date_to_run_id, prefix_to_filename_date,
)


#: Defaults for the optional ``build:`` block in a run YAML.
BUILD_DEFAULTS = {
    'build_version':    'V5',     # products land under {root}/{version}/{pipeline}/
                                  # (drivers override this with their own)
    'finding_config':   'D',      # fronts/finding/configs/finding_config_D.yaml
    'gradb2_root':      'gradb2',
    'finding_suffix':   'sfc',    # which depth suffix to find fronts in (DEPTH)
    'ice_mask_find':    False,    # step 1: mask gradb2 BEFORE finding fronts
    'ice_mask_props':   False,    # step 4: mask the co-located property fields
    'percentiles':      [25, 75, 90],
    'exclude_roots':    [],       # roots to leave out of co-location
}


def read_build_config(config_file: str, build_version: str = None) -> dict:
    """Read a run YAML into a dict of everything a build driver needs.

    Args:
        config_file (str): Run YAML.
        build_version (str, optional): Override ``build.build_version``.

    Returns:
        dict: ``pipeline``, ``run_id``, ``active_subsets``, ``depth_suffixes``,
        ``date_iterations``, ``date_prefixes`` (``YYYYMMDD_HHMMSS``),
        ``timestamps`` (``YYYY-MM-DDTHH_MM_SS``), ``bucket``, ``folder``,
        ``s3_endpoint``, ``run_dir`` (``{build_version}/{pipeline}``), plus
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
        's3_endpoint':     output.get('s3_endpoint',
                                      'https://s3-west.nrp-nautilus.io'),
    })
    if build_version:
        out['build_version'] = build_version
    if not out['run_id']:
        raise ValueError(f"'run.run_id' must be set in {config_file}")
    # Products are organised by the build that made them; filenames keep the
    # source run_id so they stay traceable to the dataset they came from.
    out['run_dir'] = f"{out['build_version']}/{pipeline}"
    return out


def _resolve_channel_maps(config_file: str):
    """Map channels <-> subsets for the active pipeline.

    Compute channels are expanded with the active ``depth_suffixes``;
    model/extra channels are not.

    Returns:
        (channel_to_subset, root_to_expanded): expanded channel -> subset, and
        root name -> list of expanded channel names.
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
    """Expand root names (``relative_vorticity``) into the channels this config
    produces (``relative_vorticity_sfc``, ...).  Unsuffixed channels pass through.

    Args:
        property_roots (list): Root or already-expanded channel names.
        config_file (str): Run YAML.

    Returns:
        list: Expanded channel names, order-preserving, de-duplicated.

    Raises:
        ValueError: If a root is unknown to the active pipeline/subsets.
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


def channel_for_root(config_file: str, root: str,
                     depth_suffix: str = 'sfc') -> str:
    """Resolve a root to the one channel this config produces.

    ``gradb2`` -> ``gradb2`` on SURF/OSN, ``gradb2_{depth_suffix}`` on DEPTH.

    Args:
        config_file (str): Run YAML.
        root (str): Base channel name.
        depth_suffix (str): Suffix to pick when the root expands to several.

    Raises:
        ValueError: If the root is not produced, or the suffix is not built.
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
    """Return the dbof subset that produces *channel* under this config.
    """
    channel_to_subset, _ = _resolve_channel_maps(config_file)
    if channel not in channel_to_subset:
        raise ValueError(
            f"Channel '{channel}' is not produced by any active subset of "
            f"{config_file}.")
    return channel_to_subset[channel]


def all_property_roots(config_file: str, exclude: list = None) -> list:
    """Every property root the active subsets produce, in config order.

    Args:
        config_file (str): Run YAML.
        exclude (list, optional): Roots to leave out.
    """
    _, root_to_expanded = _resolve_channel_maps(config_file)
    drop = set(exclude or [])
    return [r for r in root_to_expanded if r not in drop]
