# Code for configuring the parameters for front finding

import numpy as np
import os
from importlib import resources
import yaml

# Front finding data model
finding_dmodel = {
    'label': dict(dtype=str,
                help='Config label.  Will be part of the output filename'),
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
}
finding_dmodel['required'] = ('window', 'threshold', 'thresh_mode', 'thin',
        'label')
    
def config_filename(config_label: str, path:str=None):
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
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    missing = [field for field in finding_dmodel['required'] if field not in config]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Validate data types
    for field, value in config.items():
        if field == 'required':
            continue
        if field not in finding_dmodel:
            raise ValueError(f"Unknown field: {field}")

        expected_dtype = finding_dmodel[field]['dtype']
        if isinstance(expected_dtype, tuple):
            # Handle multiple allowed types (e.g., int or np.integer)
            if not isinstance(value, expected_dtype):
                raise ValueError(
                    f"Field '{field}' has invalid type. "
                    f"Expected {expected_dtype}, got {type(value)}"
                )
        else:
            # Single type
            if not isinstance(value, expected_dtype):
                raise ValueError(
                    f"Field '{field}' has invalid type. "
                    f"Expected {expected_dtype}, got {type(value)}"
                )

    return config