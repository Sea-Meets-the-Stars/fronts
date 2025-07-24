""" Utility functions for the PO """

import numpy as np

def calc_grad2(field:np.ndarray, dx:float):
    # Gradient
    dfdx = np.gradient(field, axis=1) / dx
    dfdy = np.gradient(field, axis=0) / dx

    # Magnitude
    grad_f2 = dfdx**2 + dfdy**2

    return grad_f2