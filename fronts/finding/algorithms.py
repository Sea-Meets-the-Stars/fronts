""" Front finding algorithms """
from fronts.finding import pyboa

from skimage import morphology


def fronts_from_divb2(Divb2, wndw:int=40, thin:bool=False, rm_weak:float=None,
                      dilate:bool=False):

    # Threshold
    res_frnt_np = pyboa.front_thresh(Divb2, wndw=wndw)

    # Remove weak segments?
    if rm_weak is not None:
        res_frnt_np &= Divb2 > rm_weak

    # Thin?
    if thin:
        res_frnt_np = morphology.thin(res_frnt_np)
    
    # Crop
    res_frnt_crop = pyboa.cropping(res_frnt_np)

    # Dilate?
    if dilate:
        res_frnt_crop = morphology.dilation(res_frnt_crop)#, morphology.square(3))

    return res_frnt_crop
