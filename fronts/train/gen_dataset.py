""" Functions to generate train/validation datasets """
import os

import numpy as np

import h5py

from fronts.po import utils as po_utils


def build_dataset(super_file:str, parent_idx:np.ndarray,
                  ntrain:int, nvalid:int,
                  fields:list, out_root:str,
                  SSTmu:np.ndarray=None,
                  dx:float=None):

    """ Build the dataset """
    assert parent_idx.size >= ntrain+nvalid
    nfields = len(fields)

    # Pick em
    all_idx = np.random.choice(parent_idx, ntrain+nvalid, replace=False)
    train_idx = all_idx[:ntrain]
    valid_idx = all_idx[ntrain:]

    # Open super file
    f = h5py.File(super_file, 'r')    

    # Grab one image to get the shape
    tst = f[fields[0]][0]
    shape = tst.shape

    # Initialize
    train_data = np.zeros((ntrain, nfields, shape[0], shape[1])).astype(np.float32)
    valid_data = np.zeros((nvalid, nfields, shape[0], shape[1])).astype(np.float32)

    channel_n = 0
    for field in fields:
        # Process
        if field in ['SSS']:
            for kk, idx in enumerate(train_idx):
                train_data[kk,channel_n] = f[field][idx]
            for kk, idx in enumerate(valid_idx):
                valid_data[kk,channel_n] = f[field][idx]
        elif field == 'divSST':
            pass

        # Increment
        channel_n += 1

    