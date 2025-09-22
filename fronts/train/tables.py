""" Deal with Tables related to Training """
import numpy as np

import pandas

from fronts import io as fronts_io
from fronts.dbof import io as dbof_io

from IPython import embed

def dbof_gen_tvt(dbof_json_file:str, config_file:str):
    """
    Generate train, validation, and test datasets from a given DataFrame based on the provided configuration.
    This function partitions the input DataFrame into training, validation, and test sets. It supports balancing 
    the data based on a specified metric and ensures that the resulting sets are sampled according to the 
    configuration parameters.
    Args:
        super_tbl (pandas.DataFrame): The input DataFrame containing the data to be split.
        embed(header='112 of tables')
        embed(header='112 of tables')
        config (dict): A dictionary containing configuration parameters for the split. It must include:
            - 'ntrain' (int): Number of samples for the training set.
            - 'nvalid' (int): Number of samples for the validation set.
            - 'ntest' (int): Number of samples for the test set.
            - 'balance' (optional, dict): A dictionary specifying balancing options:
                - 'metric' (str): The column name or a log-transformed column ('log<column_name>') to balance on.
                - 'nbins' (int): Number of bins to use for balancing.
    Returns:
        tuple: A tuple containing three DataFrames:
            - train_tbl (pandas.DataFrame): The training set.
            - valid_tbl (pandas.DataFrame): The validation set.
            - test_tbl (pandas.DataFrame): The test set.
    Raises:
        AssertionError: If the number of unique indices in the sampled data does not match the expected size 
                        or if the total number of sampled indices is less than the required total.
    Notes:
        - The function ensures that the sampled indices are unique and that the total number of samples 
            matches the sum of 'ntrain', 'nvalid', and 'ntest'.
        - If balancing is enabled, the data is divided into bins based on the specified metric, and samples 
            are drawn proportionally from each bin.
    """

    # Load up json files
    dbof_dict = fronts_io.loadjson(dbof_json_file)
    config = fronts_io.loadjson(config_file)

    # Load up the main table
    dbof_table = dbof_io.load_main_table(dbof_dict)

    # Parse on Input and Targets
    fields = list(config['inputs'].keys()) + list(config['targets'].keys())
    # Find all true in main table
    super_tbl = dbof_table.copy()
    for field in fields:
        if field not in super_tbl.columns:
            raise IOError(f"Field {field} not in table")
        super_tbl = super_tbl[super_tbl[field]].copy()

    # Total
    ntot = config['ntrain'] + config['nvalid'] + config['ntest']

    if config['sampling']['type'] == 'balance': 

        # Load up meta table
        meta_tbl = dbof_io.load_meta_table(dbof_dict, config['sampling']['field'])
        metric = config['sampling']['metric']

        # Cut down to those in super_tbl
        meta_tbl = meta_tbl[meta_tbl.UID.isin(super_tbl.UID)].copy()

        vals = meta_tbl[metric].values
        if 'log_metric' in config['sampling'] and config['sampling']['log_metric']:
            vals = np.log10(vals)

        # Histogram
        nbins = config['sampling']['nbins']
        bins = np.linspace(vals.min(), vals.max(), nbins+1)
        ibins = np.digitize(vals, bins) - 1  # 0 index
        hist, _ = np.histogram(vals, bins=bins)

        # Build up the indices
        max_per_bin = ntot//nbins + 1
        tails = np.where(hist <= max_per_bin+1)[0]

        # Tails
        all_idx = []
        for ss in tails:
            idx = np.where(ibins == ss)[0]
            # Take em all
            all_idx += list(idx)

        # Rest
        rest = np.where(hist > max_per_bin+1)[0]
        nmore = ntot - len(all_idx)
        n_per_bin = nmore // len(rest) + 1
        for ss in rest:
            idx = np.where(ibins == ss)[0]
            # Random draw
            ridx = np.random.choice(idx, n_per_bin, replace=False)
            all_idx += list(ridx)

        all_idx = np.array(all_idx)

        # Checks

        # Unique?
        uni = np.unique(all_idx)
        assert uni.size == all_idx.size
        assert all_idx.size >= ntot
        
        ridx = np.random.choice(all_idx, ntot, replace=False)
        final_train = ridx[:config['ntrain']]
        final_valid = ridx[config['ntrain']:config['ntrain']+config['nvalid']]
        final_test = ridx[config['ntrain']+config['nvalid']:]

        # Pivot on UID
        final_train = super_tbl.index.values[super_tbl.UID.isin(meta_tbl.UID.values[final_train])].copy()
        final_valid = super_tbl.index.values[super_tbl.UID.isin(meta_tbl.UID.values[final_valid])].copy()
        final_test = super_tbl.index.values[super_tbl.UID.isin(meta_tbl.UID.values[final_test])].copy()
    elif config['sampling']['type'] == 'random': 
        ridx = np.random.choice(super_tbl.index.values, ntot, replace=False)
        final_train = ridx[:config['ntrain']]
        final_valid = ridx[config['ntrain']:config['ntrain']+config['nvalid']]
        final_test = ridx[config['ntrain']+config['nvalid']:]
    else:
        raise ValueError("Bad sampling type")

    # Tables by index
    train_tbl = super_tbl.loc[final_train].copy()
    valid_tbl = super_tbl.loc[final_valid].copy()
    test_tbl = super_tbl.loc[final_test].copy()

    # Return
    return train_tbl, valid_tbl, test_tbl
    

