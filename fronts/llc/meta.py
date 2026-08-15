"""Run descriptor written alongside a build's products.

One YAML file per build, at the top of the run directory::

    $OS_OGCM/LLC/Fronts/V5/SURF/
        fronts_meta_V5_SURF_from_globals_for_cutouts_v2_2_01.meta

The name alone says which dataset the fronts came from; opening it gives the
rest -- pipeline, S3 store location, front-finding parameters, ice masking,
date coverage, and the git revision of both repos at the time of the run.
"""
import os
import subprocess

import yaml

from dbof.global_dataset_creation.config import default_output_folder
from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix

from fronts.llc import io as llc_io

META_SUFFIX = '.meta'


def _git_revision(package) -> str:
    """Return the short git SHA of *package*'s checkout, or 'unknown'."""
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(package.__file__)))
        out = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return 'unknown'


def meta_filename(build_version: str, pipeline: str, folder: str,
                  run_id: str) -> str:
    """Build the descriptor's filename.

    ``fronts_meta_{build_version}_{pipeline}_from_{folder}_{run_id}.meta``
    """
    source = folder.strip().strip('/').replace('/', '_')
    return (f'fronts_meta_{build_version}_{pipeline}'
            f'_from_{source}_{run_id}{META_SUFFIX}')


def write_run_meta(cfg: dict, config_file: str, out_dir: str = None,
                   extra: dict = None) -> str:
    """Write the run descriptor and return its path.

    Parameters
    ----------
    cfg : dict
        Result of :func:`fronts.properties.run.read_build_config`.
    config_file : str
        Path to the run YAML, recorded for provenance.
    out_dir : str, optional
        Where to write it.  Defaults to the run root
        (``{fronts_path}/{run_dir}``).
    extra : dict, optional
        Additional key/values to record, e.g. the resolved gradb2 channel.

    Returns
    -------
    str
        Path to the descriptor.
    """
    import fronts
    import dbof

    folder = cfg['folder'] or default_output_folder(cfg['pipeline'])
    if out_dir is None:
        out_dir = llc_io.run_root(cfg['run_id'], generate=True)
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, meta_filename(
        cfg['build_version'], cfg['pipeline'], folder, cfg['run_id']))

    dates = cfg['date_iterations']
    doc = {
        'build': {
            'version':        cfg['build_version'],
            'pipeline':       cfg['pipeline'],
            'config_file':    os.path.abspath(config_file),
            'output_dir':     out_dir,
            'file_tag':       cfg['run_id'],
        },
        'source': {
            'bucket':         cfg['bucket'],
            'folder':         folder,
            'run_id':         cfg['run_id'],
            'store_uri':      make_run_prefix(
                cfg['bucket'], folder, cfg['run_id'],
                '{subset}.zarr', date_prefix='{date_prefix}'),
            'subsets':        cfg['active_subsets'],
        },
        'fronts': {
            'gradb2_channel': (extra or {}).get('gradb2_channel'),
            'gradb2_subset':  (extra or {}).get('gradb2_subset'),
            'finding_config': cfg['finding_config'],
            'ice_mask_find':  cfg['ice_mask_find'],
            'ice_mask_props': cfg['ice_mask_props'],
            'percentiles':    cfg['percentiles'],
            'exclude_roots':  cfg['exclude_roots'],
        },
        'dates': {
            'n':     len(dates),
            'first': dates[0],
            'last':  dates[-1],
            'all':   dates,
        },
        'code': {
            'fronts_git': _git_revision(fronts),
            'dbof_git':   _git_revision(dbof),
        },
    }
    if extra:
        for key, value in extra.items():
            if key not in ('gradb2_channel', 'gradb2_subset'):
                doc.setdefault('extra', {})[key] = value

    with open(path, 'w') as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
    print(f"Wrote run descriptor: {path}")
    return path
