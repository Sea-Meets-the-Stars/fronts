""" Routines to measure at scale """

import numpy as np

from fronts.properties import utils as fprop_utils

def fprops_in_fields(fronts:np.ndarray, fprops:list, fields:list):

    fprop_dict = {}

    if not np.any(fronts > 0):
        return fprop_dict

    # Unique front labels
    flabels = np.unique(fronts)
    flabels = flabels[flabels > 0]

    # Generate the masks
    fmasks = np.stack([fronts == flabel for flabel in flabels])
    Npix = np.sum(fmasks, axis=(1,2))

    fprop_dict['flabel'] = flabels
    fprop_dict['Npix'] = Npix

    # Mask me
    for fprop, field in zip(fprops, fields):
        if fprop[0:3] == 'avg':
            fprop_dict[fprop] = fprop_utils.avg_field(field, fmasks, Npix)
        else:
            raise ValueError(f"Not ready for front property: {fprop}")

    # Return
    return fprop_dict
