""" Front finding algorithms """
from fronts.finding import pyboa

from skimage import morphology


def fronts_from_divb2(Divb2, window:int=40, thin:bool=False,
                      rm_weak:float=None, dilate:bool=False,
                      connectivity:int=2, threshold:float=90,
                      thresh_mode:str='generic', n_workers:int=None,
                      min_size:int=7, verbose:bool=False):
    """
    Identifies and processes fronts from a divergence field (Divb2).

    Parameters:
    -----------
    Divb2 : ndarray
        The input divergence field from which fronts are to be identified.
    window : int, optional, default=40
        The window size used for thresholding in the front detection algorithm.
    thin : bool, optional, default=False
        If True, thins the detected fronts to single-pixel width.
    rm_weak : float, optional, default=None
        If provided, removes weak segments where Divb2 values are below this threshold.
    dilate : bool, optional, default=False
        If True, applies dilation to the cropped fronts.
    min_size : int, optional, default=7
        Minimum size for cropping the detected fronts.
    thresh_mode : str, optional, default=generic
        Thresholding mode
    n_workers : int, optional, 
        Number of workers for parallel calculations
    threshold : float, optional, default=90
        Percentile used in the cropping function to determine size threshold.
    verbose : bool, optional, default=False
        If True, prints verbose output.

    Returns:
    --------
    ndarray
        The processed front field after applying the specified operations.
    """

    # Threshold
    if verbose:
        print(f'Thresholding with window size {window} and threshold {threshold} and mode {thresh_mode}')
    res_frnt_np = pyboa.front_thresh(Divb2, wndw=window, prcnt=threshold,
        mode=thresh_mode, n_workers=n_workers)

    # Remove weak segments?
    if rm_weak is not None:
        res_frnt_np &= Divb2 > rm_weak

    # Thin?
    if thin:
        if verbose:
            print('Thinning...')
        res_frnt_np = morphology.thin(res_frnt_np)
    
    # Crop
    if verbose:
        print(f'Cropping with minimum size {min_size} and connectivity {connectivity}')
    if min_size > 0:
        res_frnt_crop = pyboa.cropping(res_frnt_np, min_size=min_size,
                                   connectivity=connectivity)
    else:
        res_frnt_crop = res_frnt_np

    # Dilate?
    if dilate:
        res_frnt_crop = morphology.dilation(res_frnt_crop)#, morphology.square(3))

    return res_frnt_crop
