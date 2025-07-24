""" Simple statistics functions. """

import numpy as np

def meta_stats(field:np.ndarray):
    ff = field.flatten()
    #
    meta_dict = {}
    srt = np.argsort(ff)
    meta_dict['max'] = ff[srt[-1]]
    meta_dict['min'] = ff[srt[0]]
    i10 = int(0.1*field.size)
    i90 = int(0.9*field.size)
    meta_dict['10'] = ff[srt[i10]]
    meta_dict['90'] = ff[srt[i90]]

    # Mean
    meta_dict['mu'] = np.mean(ff)

    # Return
    return meta_dict