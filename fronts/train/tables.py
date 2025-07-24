""" Deal with Tables related to Training """
import numpy as np

import pandas

from IPython import embed

def gen_tvt(super_tbl:pandas.DataFrame, config:dict):

    ntot = config['ntrain'] + config['nvalid'] + config['ntest']

    if 'balance' in config.keys():
        metric = config['balance']['metric']

        if metric[:3] == 'log':
            vals = np.log10(super_tbl[metric[3:]].values)
        else:
            vals = super_tbl[metric].values

        # Histogram
        nbins = config['balance']['nbins']
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

    # Tables
    train_tbl = super_tbl.iloc[final_train].copy()
    valid_tbl = super_tbl.iloc[final_valid].copy()
    test_tbl = super_tbl.iloc[final_test].copy()

    # Return
    return train_tbl, valid_tbl, test_tbl
    

