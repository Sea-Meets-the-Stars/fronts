""" Methods for pre-processing steps"""

import numpy as np
import os

from skimage.restoration import inpaint as sk_inpaint
from scipy.ndimage import median_filter
from scipy import special
from skimage.transform import downscale_local_mean, resize_local_mean
from skimage import filters
from sklearn.utils import shuffle

#from ulmo import defs as ulmo_defs
#from ulmo.preproc import io as pp_io
#from ulmo import io as ulmo_io

from fronts.tables import defs
from fronts.utils import stats as front_stats
from fronts.po import utils as po_utils

from IPython import embed


def prep_table_for_preproc(tbl, extract_file:str, field_size=None):
    """Prep the table for pre-processing
    e.g. add a few columns

    Args:
        tbl (pandas.DataFrame): _description_
        preproc_root (_type_): _description_
        field_size (tuple, optional): Field size. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Prep Table
    for key in ['filename', 'pp_file']:
        if key not in tbl.keys():
            tbl[key] = ''
    tbl['pp_root'] = extract_file
    if field_size is not None:
        tbl['field_size'] = field_size[0]
    #tbl['pp_idx'] = -1
    tbl['pp_type'] = defs.mtbl_dmodel['pp_type']['init']
    # 
    return tbl

def preproc_image(item:tuple, pdict:dict, use_mask=False,
                  inpainted_mask=False):
    """
    Simple wrapper for preproc_field()
    Mainly for multi-processing

    Parameters
    ----------
    item : tuple
        field, idx or field,mask,idx (use_mask=True)
    pdict : dict
        Preprocessing dict
    use_mask : bool, optional
        If True, allow for an input mask
    inpainted_mask : bool, optional
        If True, the tuple includes an inpainted_mask
        instead of a simple mask.

    Returns
    -------
    pp_field, idx, meta : np.ndarray, int, dict

    """
    raise DeprecationWarning("Use wrangler.preproc.field.multi_process()")

    # Unpack
    mask = None
    smooth_pix = None
    if use_mask:
        field, mask, idx = item
        if inpainted_mask:
            true_mask = np.isfinite(mask)
            # Fill-in inpainted values
            field[true_mask] = mask[true_mask]
            # Overwrite
            mask = true_mask
    elif 'smooth_km' in pdict.keys():
        field, idx, smooth_pix = item
    else:
        field, idx = item

    # Junk field?  (e.g. LLC)
    if field is None:
        return None, idx, None

    # Run
    pp_field, meta = preproc_field(field, mask, 
                                   smooth_pix=smooth_pix, **pdict)

    # Failed?
    if pp_field is None:
        return None, idx, None

    # Return
    return pp_field.astype(np.float32), idx, meta


def preproc_field(field, mask, inpaint=False, median=False, med_size=(3,1),
                  downscale=False, dscale_size=(2,2), sigmoid=False, 
                  scale=None,
                  expon=None, only_inpaint=False, gradient=False,
                  div2:float=None,
                  min_mean=None, de_mean=True,
                  field_size:int=None,
                  resize:bool=False,
                  smooth_pix:int=None,
                  noise=None,
                  log_scale=False, **kwargs):
    """
    Preprocess an input field image with a series of steps:
        1. Inpainting
        2. Gaussian Smooth
        3. Resize based on fixed_km (LLC)
        4. Add noise
        5. Median
        6. Downscale
        7. Sigmoid
        8. Scale
        9. Remove mean
        10. Sobel
        11. Log

    Parameters
    ----------
    field : np.ndarray
    mask : np.ndarray or None
        Data mask.  True = masked
        Required for inpainting but otherwise ignored
    inpaint : bool, optional
        if True, inpaint masked values
    smoooth_pix : int, optional
        Smooth the field with a Gaussian filter of this size
    median : bool, optional
        If True, apply a median filter
    med_size : tuple
        Median window to apply
    downscale : bool, optional
        If True downscale the image
    dscale_size : tuple, optional
        Size to rescale by
    noise : float, optional
        If provided, add white noise with this value
    scale : float, optional
        Scale the SSTa values by this multiplicative factor
    expon : float
        Exponate the SSTa values by this exponent
    gradient : bool, optional
        If True, apply a Sobel gradient enhancing filter
    div2 : float, optional
        If provided, calculate the squared divergence and use
        div2 as the length (i.e. dx)
    de_mean : bool, optional
        If True, subtract the mean
    min_mean : float, optional
        If provided, require the image has a mean exceeding this value
    fixed_km : float, optional
        If provided the input image is smaller than desired, so cut it down!
        field_size must also be provided
    field_size : int, optional
        Size of the field
    **kwargs : catches extraction keywords

    Returns
    -------
    pp_field, meta_dict : np.ndarray, dict
        Pre-processed field, mean temperature

    """
    raise DeprecationWarning("Use wrangler.preproc.field.main()")

    # Inpaint?
    if inpaint:
        if mask.dtype.name != 'uint8':
            mask = np.uint8(mask)
        field = sk_inpaint.inpaint_biharmonic(field, mask, channel_axis=None)

    if only_inpaint:
        if np.any(np.isnan(field)):
            return None, None
        else:
            return field, None

    # Smooth?
    if smooth_pix is not None:
        field = filters.gaussian(field, smooth_pix)
        # Crop
        field = field[2*smooth_pix:-2*smooth_pix, 2*smooth_pix:-2*smooth_pix]

    # Resize?
    if resize is not None:
        field = resize_local_mean(field, (field_size, field_size))

    # Capture metadata
    meta_dict = front_stats.meta_stats(field)

    # Add noise?
    if noise is not None:
        field += np.random.normal(loc=0., 
                                  scale=noise, 
                                  size=field.shape)
    # Median
    if median:
        field = median_filter(field, size=med_size)

    # Reduce to 64x64
    if downscale:
        field = downscale_local_mean(field, dscale_size)

    # Check for junk
    if np.any(np.isnan(field)):
        return None, None

    # Check mean
    if min_mean is not None and meta_dict['mu'] < min_mean:
        return None, None

    # De-mean the field
    if de_mean:
        pp_field = field - field.mean()
    else:
        pp_field = field

    # Sigmoid?
    if sigmoid:
        pp_field = special.erf(pp_field)

    # Scale?
    if scale is not None:
        pp_field *= scale

    # Exponate?
    if expon is not None:
        neg = pp_field < 0.
        pos = np.logical_not(neg)
        pp_field[pos] = pp_field[pos]**expon
        pp_field[neg] = -1 * (-1*pp_field[neg])**expon

    # Sobel Gradient?
    if gradient:
        pp_field = filters.sobel(pp_field)
        # Meta
        srt = np.argsort(pp_field.flatten())
        i10 = int(0.1*pp_field.size)
        i90 = int(0.9*pp_field.size)
        meta_dict['G10'] = pp_field.flatten()[srt[i10]]
        meta_dict['G90'] = pp_field.flatten()[srt[i90]]
        meta_dict['Gmax'] = pp_field.flatten()[srt[-1]]

    # |grad f|^2?
    if div2 is not None:
        pp_field = po_utils.calc_grad2(pp_field, div2)

    # Log?
    if log_scale:
        if not gradient:
            raise IOError("Only implemented with gradient=True so far")
        # Set 0 values to the lowest non-zero value
        zero = pp_field == 0.
        if np.any(zero):
            min_nonz = np.min(pp_field[np.logical_not(zero)])
            pp_field[zero] = min_nonz
        # Take log
        pp_field = np.log(pp_field)


    # Return
    return pp_field, meta_dict