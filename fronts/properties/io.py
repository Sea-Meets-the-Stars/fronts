"""
Input/Output for the front-finding pipeline (production code).

  write_front_index        — save label/name/bbox index table to parquet or csv
  load_front_index         — load that table back
  get_global_front_output_path — standardised output paths for global runs
  write_json               — write a dict to JSON

For analysis / visualisation loaders see :mod:`fronts.properties.viz_loaders`.
"""

import json as _json
import os
from typing import Dict, Union

import pandas as pd
from pathlib import Path

def write_front_index(
    front_ids: Dict[int, str],
    properties: Dict[int, Dict],
    output_path: Union[str, Path],
    format: str = 'parquet'
) -> pd.DataFrame:
    """
    Save front index table to disk.

    The front index is the master lookup table for all detected fronts:
    one row per front with its integer label, unique name, and bounding box.

    Columns: label, name, x0, y0, x1, y1
        - label : integer front label
        - name  : unique front ID string (TIME_LAT_LON format)
        - x0, x1 : column (longitude axis) bounding box indices
        - y0, y1 : row (latitude axis) bounding box indices

    Parameters
    ----------
    front_ids : dict
        Dictionary mapping integer label -> front ID string.
    properties : dict
        Dictionary mapping integer label -> properties dict with 'bbox' key:
        (min_row, min_col, max_row, max_col).
    output_path : str or Path
        Output file path (use .parquet or .csv extension).
    format : str, optional
        Output format: 'parquet' (default) or 'csv'.

    Returns
    -------
    df : pd.DataFrame
        The front index DataFrame (also saved to disk).
    """
    rows = []
    for label, name in front_ids.items():
        if label in properties:
            min_row, min_col, max_row, max_col = properties[label]['bbox']
            rows.append({
                'label': int(label),
                'name':  name,
                'x0':    int(min_col),   # col = longitude axis
                'y0':    int(min_row),   # row = latitude axis
                'x1':    int(max_col),
                'y1':    int(max_row),
            })

    df = pd.DataFrame(rows)
    output_path = Path(output_path)

    if format == 'parquet':
        df.to_parquet(output_path, index=False, engine='pyarrow')
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'parquet' or 'csv'.")

    print(f"Saved front index ({len(df)} fronts) to {output_path}")
    return df


def load_front_index(
    input_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load front index table from disk.

    Parameters
    ----------
    input_path : str or Path
        Path to front index file (.parquet or .csv).

    Returns
    -------
    df : pd.DataFrame
        Front index with columns: label, name, x0, y0, x1, y1.

    Examples
    --------
    >>> index_df = load_front_index('front_index.parquet')
    >>> row = index_df.iloc[0]
    >>> cutout = label_map[row.y0:row.y1, row.x0:row.x1]
    """
    input_path = Path(input_path)
    if input_path.suffix == '.parquet':
        return pd.read_parquet(input_path, engine='pyarrow')
    else:
        return pd.read_csv(input_path)


def get_global_front_output_path(
    output_dir: Union[str, Path],
    time_str: str,
    file_type: str,
    run_tag: str = '',
) -> Path:
    """
    Generate standardized output file paths for global front processing.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    time_str : str
        ISO 8601 timestamp string (e.g. '2012-11-09T12:00:00').
        Colons and dashes are made filename-safe automatically.
    file_type : str
        One of: 'label_map', 'front_index', 'geometry', 'properties', 'metadata',
        'metadata_properties'.
    run_tag : str, optional
        Version/config suffix extracted from the source fronts filename,
        e.g. 'v1_bin_A' from 'LLC4320_2012-11-09T12_00_00_v1_bin_A.npy'.
        Appended to all output filenames so results are traceable to their input.

    Returns
    -------
    path : Path
        Full output file path.

    Examples
    --------
    >>> get_global_front_output_path('/out', '2012-11-09T12:00:00', 'geometry', 'v1_bin_A')
    PosixPath('/out/global_front_geometry_20121109T12_00_00_v1_bin_A.parquet')
    """
    time_str_safe = time_str.replace(':', '_').replace('-', '')
    tag = f'_{run_tag}' if run_tag else ''

    names = {
        'label_map':   f'labeled_fronts_global_{time_str_safe}{tag}.npy',
        'front_index': f'front_index_{time_str_safe}{tag}.parquet',
        'geometry':    f'global_front_geometry_{time_str_safe}{tag}.parquet',
        'properties':  f'front_properties_{time_str_safe}{tag}.parquet',
        'metadata':    f'metadata_{time_str_safe}{tag}.json',
        'metadata_properties':    f'metadata_properties_{time_str_safe}{tag}.json',
    }

    if file_type not in names:
        raise ValueError(
            f"Unknown file_type: {file_type!r}. Options: {list(names.keys())}"
        )

    return Path(output_dir) / names[file_type]

def write_json(
    data: dict,
    output_path: Union[str, Path],
) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        Data to serialise
    output_path : str or Path
        Output file path (should end in .json)
    """
    with open(output_path, 'w') as f:
        _json.dump(data, f, indent=2)
    print(f"Saved JSON to {output_path}")


# ---------------------------------------------------------------------------
# Re-export the central path setter from llc.io for convenience
# ---------------------------------------------------------------------------
from fronts.llc import io as _llc_io

set_fronts_path = _llc_io.set_fronts_path
get_fronts_path = _llc_io.get_fronts_path
fronts_dir = _llc_io.fronts_dir

