""" Basic I/O methods"""

import os
import numpy as np

import json
import gzip
import pandas


def load_main_table(tbl_file:str, verbose=True):
    """Load the table of cutouts 

    Args:
        tbl_file (str): Path to table of cutouts. Local or s3
        verbose (bool, optional): [description]. Defaults to True.

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: table of cutouts
    """
    _, file_extension = os.path.splitext(tbl_file)

    # s3?
    if tbl_file[0:5] == 's3://':
        inp = load_to_bytes(tbl_file)
    else:
        inp = tbl_file
        
    # Allow for various formats
    if file_extension == '.csv':
        main_table = pandas.read_csv(inp, index_col=0)
        # Set time
        if 'datetime' in main_table.keys():
            main_table.datetime = pandas.to_datetime(main_table.datetime)
    elif file_extension == '.feather':
        # Allow for s3
        main_table = pandas.read_feather(inp)
    elif file_extension == '.parquet':
        # Allow for s3
        main_table = pandas.read_parquet(inp)
    else:
        raise IOError("Bad table extension: ")

    # Deal with masked int columns
    for key in ['gradb_Npos', 'FS_Npos', 'UID', 'pp_type']:
        if key in main_table.keys():
            main_table[key] = pandas.array(main_table[key].values, dtype='Int64')
    # Report
    if verbose:
        print("Read main table: {}".format(tbl_file))

    # Decorate
    if 'DT' not in main_table.keys() and 'T90' in main_table.keys():
        main_table['DT'] = main_table.T90 - main_table.T10
        
    return main_table


        if verbose:
            print("Downloading from s3: {}".format(local_file))
        s3.Bucket(parsed_s3.netloc).download_file(
            parsed_s3.path[1:], local_file)
        if verbose:
            print("Done!")
    
    
def write_main_table(main_table:pandas.DataFrame, outfile:str, 
                     to_s3=True):
    """Write Main table for ULMO analysis
    Format is determined from the outfile extension.
        Options are ".csv", ".feather", ".parquet"

    Args:
        main_table (pandas.DataFrame): Main table for ULMO analysis
        outfile (str): Output filename.  Its extension sets the format
        to_s3 (bool, optional): If True, write to s3

    Raises:
        IOError: [description]
    """
    _, file_extension = os.path.splitext(outfile)
    if file_extension == '.csv':
        main_table.to_csv(outfile, date_format='%Y-%m-%d %H:%M:%S')
    elif file_extension == '.feather':
        bytes_ = BytesIO()
        main_table.to_feather(path=bytes_)
        if to_s3:
            write_bytes_to_s3(bytes_, outfile)
        else:
            write_bytes_to_local(bytes_, outfile)
    elif file_extension == '.parquet':
        bytes_ = BytesIO()
        main_table.to_parquet(path=bytes_)
        if to_s3:
            write_bytes_to_s3(bytes_, outfile)
        else:
            write_bytes_to_local(bytes_, outfile)
    else:
        raise IOError("Not ready for this")
    print("Wrote Analysis Table: {}".format(outfile))


def jsonify(obj, debug=False):
    """ Recursively process an object so it can be serialised in json
    format.

    WARNING - the input object may be modified if it's a dictionary or
    list!

    Parameters
    ----------
    obj : any object
    debug : bool, optional

    Returns
    -------
    obj - the same obj is json_friendly format (arrays turned to
    lists, np.int64 converted to int, np.float64 to float, and so on).

    """
    if isinstance(obj, np.float64):
        obj = float(obj)
    elif isinstance(obj, np.float32):
        obj = float(obj)
    elif isinstance(obj, np.int32):
        obj = int(obj)
    elif isinstance(obj, np.int64):
        obj = int(obj)
    elif isinstance(obj, np.int16):
        obj = int(obj)
    elif isinstance(obj, np.bool_):
        obj = bool(obj)
    elif isinstance(obj, np.string_):
        obj = str(obj)
    elif isinstance(obj, units.Quantity):
        if obj.size == 1:
            obj = dict(value=obj.value, unit=obj.unit.to_string())
        else:
            obj = dict(value=obj.value.tolist(), unit=obj.unit.to_string())
    elif isinstance(obj, np.ndarray):  # Must come after Quantity
        obj = obj.tolist()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = jsonify(value, debug=debug)
    elif isinstance(obj, list):
        for i,item in enumerate(obj):
            obj[i] = jsonify(item, debug=debug)
    elif isinstance(obj, tuple):
        obj = list(obj)
        for i,item in enumerate(obj):
            obj[i] = jsonify(item, debug=debug)
        obj = tuple(obj)
    elif isinstance(obj, units.Unit):
        obj = obj.name
    elif obj is units.dimensionless_unscaled:
        obj = 'dimensionless_unit'

    if debug:
        print(type(obj))
    return obj


def loadjson(filename):
    """
    Parameters
    ----------
    filename : str

    Returns
    -------
    obj : dict

    """
    #
    if filename.endswith('.gz'):
        with gzip.open(filename, "rb") as f:
            obj = json.loads(f.read().decode("ascii"))
    else:
        with open(filename, 'rt') as fh:
            obj = json.load(fh)

    return obj


def loadyaml(filename):
    from astropy.io.misc import yaml as ayaml
    # Read yaml
    with open(filename, 'r') as infile:
        data = ayaml.load(infile)
    # Return
    return data


def savejson(filename, obj, overwrite=False, indent=None, easy_to_read=False,
             **kwargs):
    """ Save a python object to filename using the JSON encoder.

    Parameters
    ----------
    filename : str
    obj : object
      Frequently a dict
    overwrite : bool, optional
    indent : int, optional
      Input to json.dump
    easy_to_read : bool, optional
      Another approach and obj must be a dict
    kwargs : optional
      Passed to json.dump

    Returns
    -------

    """
    import io

    if os.path.lexists(filename) and not overwrite:
        raise IOError('%s exists' % filename)
    if easy_to_read:
        if not isinstance(obj, dict):
            raise IOError("This approach requires obj to be a dict")
        with io.open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(obj, sort_keys=True, indent=4,
                               separators=(',', ': '), **kwargs))
    else:
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wt') as fh:
                json.dump(obj, fh, indent=indent, **kwargs)
        else:
            with open(filename, 'wt') as fh:
                json.dump(obj, fh, indent=indent, **kwargs)