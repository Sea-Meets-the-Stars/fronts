""" Basic ways to characterize fronts """

import numpy as np

def avg_field(field:np.ndarray, fmasks:np.ndarray, Npix:int=None):

    if Npix is None:
        Npix = np.sum(fmasks, axis=(1,2))

    stats = np.sum(fmasks * field, axis=(1,2)) / Npix

    return stats
