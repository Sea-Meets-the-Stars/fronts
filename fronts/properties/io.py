"""
Input/Output for front grouping results.

Three functions mirroring the pattern in fronts.finding.io:
  write_front_group_table  — save label/name/bbox table to parquet or csv
  load_front_group_table   — load that table back
  get_global_front_output_path — standardised output paths for global runs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union


def write_front_group_table(
    front_ids: Dict[int, str],
    properties: Dict[int, Dict],
    output_path: Union[str, Path],
    format: str = 'parquet'
) -> pd.DataFrame:
    """
    Save front group table to disk.

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
        The group table DataFrame (also saved to disk).
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

    print(f"Saved group table ({len(df)} fronts) to {output_path}")
    return df


def load_front_group_table(
    input_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load front group table from disk.

    Parameters
    ----------
    input_path : str or Path
        Path to group table file (.parquet or .csv).

    Returns
    -------
    df : pd.DataFrame
        Group table with columns: label, name, x0, y0, x1, y1.

    Examples
    --------
    >>> group_df = load_front_group_table('group_table.parquet')
    >>> row = group_df.iloc[0]
    >>> cutout = labeled[row.y0:row.y1, row.x0:row.x1]
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
) -> Path:
    """
    Generate standardised output file paths for global front processing.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    time_str : str
        ISO 8601 timestamp string (e.g. '2012-11-09T12:00:00').
        Colons and dashes are made filename-safe automatically.
    file_type : str
        One of: 'labeled', 'group_table', 'properties', 'metadata'

    Returns
    -------
    path : Path
        Full output file path.

    Examples
    --------
    >>> get_global_front_output_path('/out', '2012-11-09T12:00:00', 'properties')
    PosixPath('/out/global_front_properties_20121109T120000.parquet')
    """
    time_str_safe = time_str.replace(':', '_').replace('-', '')

    names = {
        'labeled':     f'labeled_fronts_global_{time_str_safe}.npy',
        'group_table': f'group_table_{time_str_safe}.parquet',
        'properties':  f'global_front_properties_{time_str_safe}.parquet',
        'colocation':  f'colocation_{time_str_safe}.parquet',
        'metadata':    f'metadata_{time_str_safe}.json',
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
    import json

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to {output_path}")
