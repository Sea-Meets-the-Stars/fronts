""" Front finding algorithms """
import numpy as np

from skimage import morphology

from fronts.finding import pyboa
from fronts.finding.sharpen import global_sharpen_pq

from IPython import embed

def fronts_from_gradb2(gradb2, window:int=40, thin:bool=False,
                      rm_weak:float=None, dilate:bool=False,
                      sharpen:bool=False,
                      connectivity:int=2, threshold:float=90,
                      thresh_mode:str='generic', n_workers:int=None,
                      min_size:int=7, verbose:bool=False,
                      debug:bool=False):
    """
    Identifies and processes fronts from a gradient field (gradb2).

    Parameters:
    -----------
    gradb2 : ndarray
        The input gradient field from which fronts are to be identified.
    window : int, optional, default=40
        The window size used for thresholding in the front detection algorithm.
    thin : bool, optional, default=False
        If True, thins the detected fronts to single-pixel width.
    sharpen : bool, optional, default=False
        If True, sharpens the detected fronts on gradb2 to single-pixel width.
    rm_weak : float, optional, default=None
        If provided, removes weak segments where gradb2 values are below this threshold.
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
    debug : bool, optional, default=False

    Returns:
    --------
    ndarray
        The processed front field after applying the specified operations.
    """

    # Threshold
    if verbose:
        print(f'Thresholding with window size {window} and threshold {threshold} and mode {thresh_mode}')
    res_frnt_np = pyboa.front_thresh(gradb2, wndw=window, prcnt=threshold,
        mode=thresh_mode, n_workers=n_workers)

    # Remove weak segments?
    if rm_weak is not None:
        res_frnt_np &= gradb2 > rm_weak

    # Sharpen?
    if sharpen:
        res_frnt_np = global_sharpen_pq(res_frnt_np, gradb2,
                                   protect_endpoints=True)

    # Thin?
    if thin:
        if verbose:
            print(f'There are {np.sum(res_frnt_np)} front pixels before thinning')
            print('Thinning...')
        res_frnt_np = morphology.thin(res_frnt_np)
        if verbose:
            print(f'There are {np.sum(res_frnt_np)} front pixels after thinning')
    
    # Crop?
    if min_size > 0:
        if verbose:
            print(f'Cropping with minimum size {min_size} and connectivity {connectivity}')
        # This also fills in small holes and 
        #   requires a second thinning step (if thin=True)
        res_frnt_crop = pyboa.cropping(res_frnt_np, min_size=min_size,
                                   connectivity=connectivity)
    else:
        res_frnt_crop = res_frnt_np

    # Dilate?
    if dilate:
        res_frnt_crop = morphology.dilation(res_frnt_crop)#, morphology.square(3))

    # Thin a final time
    if thin or sharpen:
        if verbose:
            print('Thinning a final time...')
        res_frnt_crop = morphology.thin(res_frnt_crop)
        if verbose:
            print(f'There are {np.sum(res_frnt_crop)} front pixels after final thinning')

    return res_frnt_crop
