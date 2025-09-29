""" Front finding algorithms """
from fronts.finding import pyboa

from skimage import morphology


def fronts_from_divb2(Divb2, wndw:int=40, thin:bool=False, 
                      rm_weak:float=None, dilate:bool=False,
                      min_size:int=7):
    """
    Identifies and processes fronts from a divergence field (Divb2).

    Parameters:
    -----------
    Divb2 : ndarray
        The input divergence field from which fronts are to be identified.
    wndw : int, optional, default=40
        The window size used for thresholding in the front detection algorithm.
    thin : bool, optional, default=False
        If True, thins the detected fronts to single-pixel width.
    rm_weak : float, optional, default=None
        If provided, removes weak segments where Divb2 values are below this threshold.
    dilate : bool, optional, default=False
        If True, applies dilation to the cropped fronts.
    min_size : int, optional, default=7
        Minimum size for cropping the detected fronts.

    Returns:
    --------
    ndarray
        The processed front field after applying the specified operations.
    """

    # Threshold
    res_frnt_np = pyboa.front_thresh(Divb2, wndw=wndw)

    # Remove weak segments?
    if rm_weak is not None:
        res_frnt_np &= Divb2 > rm_weak

    # Thin?
    if thin:
        res_frnt_np = morphology.thin(res_frnt_np)
    
    # Crop
    res_frnt_crop = pyboa.cropping(res_frnt_np, min_size=min_size)

    # Dilate?
    if dilate:
        res_frnt_crop = morphology.dilation(res_frnt_crop)#, morphology.square(3))

    return res_frnt_crop
