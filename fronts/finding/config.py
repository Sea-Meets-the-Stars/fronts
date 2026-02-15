# Code for configuring the parameters for front finding

import numpy as np
import os
from importlib import resources
import yaml

# Front finding data model
finding_dmodel = {
    'label': dict(dtype=str,
                help='Config label.  Will be part of the output filename'),
    'binary': {
        'window': dict(dtype=(int, np.integer),
                    help='Size of the region for thresholding (pixels)'),
        'threshold': dict(dtype=(float, np.floating),
                    help='Threshold for front finding'),
        'thresh_mode': dict(dtype=str,
                    help='Mode for finding threshold [generic, vectorized, dask, pool]'),
        'thin': dict(dtype=bool,
                    help='Thin?'),
        'dilate': dict(dtype=bool,
                    help='Dilate the front?  Usually after thin + crop'),
        'min_size': dict(dtype=(int, np.integer),
                    help='Minimum size for front (pixels). Used for cropping'),
        'connectivity': dict(dtype=(int, np.integer),
                    help='??'),
    },
}
finding_dmodel['required'] = ('window', 'threshold', 'thresh_mode', 'thin',
        'label')
    
def config_filename(config_label: str, path:str=None):
    """Build the full path to a finding configuration YAML file.

    Parameters
    ----------
    config_label : str
        Short label identifying the config (e.g. 'A').
        The resulting filename is ``finding_config_{config_label}.yaml``.
    path : str, optional
        Directory containing the config file.
        Defaults to ``fronts/finding/configs/`` inside the installed package.

    Returns
    -------
    str
        Full path to the configuration file.
    """
    if path is None:
        path = os.path.join(resources.files('fronts'), 'finding', 'configs')
    base = f'finding_config_{config_label}.yaml'
    # Return
    return os.path.join(path, base)

def _is_leaf(dmodel_entry: dict) -> bool:
    """Check if a dmodel entry is a leaf field (has 'dtype') vs a nested group."""
    return 'dtype' in dmodel_entry


def _validate_group(config: dict, dmodel: dict, path: str = ''):
    """Recursively validate config against the dmodel.

    Parameters
    ----------
    config : dict
        Configuration dict (or nested sub-dict)
    dmodel : dict
        Data model dict (or nested sub-dict)
    path : str
        Dot-separated path for error messages (e.g. 'binary.window')

    Raises
    ------
    ValueError
        If an unknown field is found or a field has an invalid type
    """
    for field, value in config.items():
        full_path = f"{path}.{field}" if path else field

        if field not in dmodel:
            raise ValueError(f"Unknown field: '{full_path}'")

        spec = dmodel[field]

        if _is_leaf(spec):
            # Leaf field -- validate dtype
            expected = spec['dtype']
            # YAML parses e.g. 90 as int, so accept int for float fields
            if not isinstance(expected, tuple):
                expected = (expected,)
            if float in expected or np.floating in expected:
                expected = (*expected, int, np.integer)
            if not isinstance(value, expected):
                raise ValueError(
                    f"Field '{full_path}' has invalid type. "
                    f"Expected {spec['dtype']}, got {type(value)}"
                )
        else:
            # Nested group -- recurse
            if not isinstance(value, dict):
                raise ValueError(
                    f"Field '{full_path}' should be a group (dict), "
                    f"got {type(value)}"
                )
            _validate_group(value, spec, path=full_path)


def _collect_keys(config: dict) -> set:
    """Recursively collect all leaf key names from a config dict."""
    keys = set()
    for field, value in config.items():
        if isinstance(value, dict):
            keys.update(_collect_keys(value))
        else:
            keys.add(field)
    return keys


def load(config_file: str) -> dict:
    """
    Load a front finding configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary with validated fields

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist
    ValueError
        If required fields are missing or have invalid types
    """
    # Load the YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields (checked across all nesting levels)
    all_keys = _collect_keys(config)
    # Add top-level keys that map to groups
    all_keys.update(config.keys())
    missing = [field for field in finding_dmodel['required']
               if field not in all_keys]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Recursively validate fields and data types
    # (skip 'required' key which is metadata on the dmodel, not a field)
    dmodel_fields = {k: v for k, v in finding_dmodel.items()
                     if k != 'required'}
    _validate_group(config, dmodel_fields)

    return config