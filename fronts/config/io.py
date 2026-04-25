# Code for configuring the parameters for front finding

import os
from importlib import resources
import yaml

from fronts.config.dmodel import dmodel

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
    print(f'Loading config from: {config_file}')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields (checked across all nesting levels)
    all_keys = dmodel._collect_keys(config)
    # Add top-level keys that map to groups
    all_keys.update(config.keys())
    missing = [field for field in dmodel.finding_dmodel['required']
               if field not in all_keys]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Recursively validate fields and data types
    # (skip 'required' key which is metadata on the dmodel, not a field)
    dmodel_fields = {k: v for k, v in dmodel.finding_dmodel.items()
                     if k != 'required'}
    dmodel._validate_group(config, dmodel_fields)

    return config