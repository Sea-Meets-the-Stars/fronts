# Code for configuring the parameters for front finding

import numpy as np

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
                    help='Apply morphological thinning to fronts'),
        'sharpen': dict(dtype=bool,
                    help='Apply morphological sharpening to fronts'),
        'despur': dict(dtype=bool,
                    help='Remove spurs from fronts'),
        'Lspur': dict(dtype=(int, np.integer),
                    help='Maximum spur length in pixels (measured as branch-distance)'),
        'dilate': dict(dtype=bool,
                    help='Dilate the front?  Usually after thin + crop'),
        'min_size': dict(dtype=(int, np.integer),
                    help='Minimum size for front (pixels). Used for cropping'),
        'connectivity': dict(dtype=(int, np.integer),
                    help='??'),
    },
    'properties': {
        'stats': dict(dtype=list,
                    help='Statistics to compute per property'),
        'percentiles': dict(dtype=list,
                    help='Percentiles to compute per property'),
        'min_npix': dict(dtype=(int, np.integer),
                    help='Minimum front size in pixels'),
        'nan_policy': dict(dtype=str,
                    help='How to handle NaN values (e.g. land pixels)'),
        'dilation_radius': dict(dtype=(int, np.integer),
                    help='Pixels to dilate each front before stats'),
    },
}
finding_dmodel['required'] = ('window', 'threshold', 'thresh_mode', 'thin',
        'sharpen', 'despur', 'label', 'stats', 'percentiles', 'min_npix', 
        'nan_policy', 'dilation_radius')
    
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
