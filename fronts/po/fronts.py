""" Fronts module """
import numpy as np
from skimage.transform import resize_local_mean


from fronts.utils import stats as front_stats
from fronts.po import utils as po_utils

try:
    from gsw import density
except ImportError:
    print("gsw not imported;  cannot do density calculations")


def gradb2_cutout(item:tuple, fixed_km:float=None, field_size:int=None, 
                dx:float=None, norm_by_b:bool=False, **kwargs):
    """Simple function to measure front related stats
    for a cutout
    
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
        field_size (int, optional): Field size. Defaults to None.
        dx (float, optional): Grid spacing in km
        norm_by_b (bool, optional): Normalize by median buoyancy in the image. Defaults to False.

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray (F_s, gradb)
    """
    # Unpack
    Theta_cutout, Salt_cutout, idx = item
    if Theta_cutout is None or Salt_cutout is None:
        return None, idx, None

    # Calculate
    gradb = calc_gradb(Theta_cutout, Salt_cutout, dx=dx,
                       norm_by_b=norm_by_b)

    # Resize
    if fixed_km is not None:
        gradb = resize_local_mean(gradb, (field_size, field_size))

    # Meta
    meta_dict = front_stats.meta_stats(gradb)

    # Return
    return gradb, idx, meta_dict


def b_cutout(item:tuple, fixed_km:float=None, field_size:int=None, 
             ref_rho:float=1025., g=0.0098, **kwargs):
    """Simple function to grab a density cutout
    
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
        field_size (int, optional): Field size. Defaults to None.

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray (F_s, gradb)
    """
    # Unpack
    Theta_cutout, Salt_cutout, idx = item
    if Theta_cutout is None or Salt_cutout is None:
        return None, idx, None

    # Calculate
    rho = density.rho(Salt_cutout, Theta_cutout, np.zeros_like(Salt_cutout))
    b = g*rho/ref_rho

    # Resize
    if fixed_km is not None:
        b = resize_local_mean(b, (field_size, field_size))

    # Meta
    meta_dict = front_stats.meta_stats(b)

    # Return
    return b, idx, meta_dict


def calc_gradb(Theta:np.ndarray, Salt:np.ndarray,
             ref_rho:float=1025., g=0.0098, dx=2.,
             norm_by_b:bool=False):
    """Calculate |grad b|^2

    Args:
        Theta (np.ndarray): SST field
        Salt (np.ndarray): Salt field
        ref_rho (float, optional): Reference density
        g (float, optional): Acceleration due to gravity
            in km/s^2
        dx (float, optional): Grid spacing in km

    Returns:
        np.ndarray: |grad b|^2 field
    """
    # Buoyancy
    rho = density.rho(Salt, Theta, np.zeros_like(Salt))
    b = g*rho/ref_rho

    # Normalize by b?
    if norm_by_b:
        b /= np.median(b)

    return po_utils.calc_grad2(b, dx)