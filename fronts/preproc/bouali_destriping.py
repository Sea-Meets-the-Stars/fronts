"""
Bouali Destriping Algorithm for Ocean Front Detection
======================================================

Implementation of the stripe noise reduction algorithm from:
Bouali, M., O. Sato, and P. Polito (2015). "An algorithm to improve the detection 
of ocean fronts from whiskbroom scanner images." Remote Sensing Letters, 6:12, 942-951.

This module provides functions to remove stripe noise from ocean satellite imagery
(SST, chlorophyll-a, etc.) to improve the detection of ocean fronts and other
submesoscale features.

Author: Implementation based on Bouali et al. (2015)
"""

import numpy as np
from scipy import fftpack
from typing import Tuple, Optional, Union


def compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the gradient field of an image.
    
    :param image: Input image array
    :type image: np.ndarray
    
    :returns: Tuple containing (gradient_x, gradient_y, gradient_magnitude)
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    
    :Example:
    
    >>> img = np.random.randn(100, 100)
    >>> grad_x, grad_y, grad_mag = compute_gradient(img)
    """
    # Compute gradients using central differences
    grad_y = np.gradient(image, axis=0)  # Along-track direction
    grad_x = np.gradient(image, axis=1)  # Cross-track direction
    
    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return grad_x, grad_y, grad_magnitude


def compute_gradient_orientation(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """
    Compute the orientation of the gradient field.
    
    :param grad_x: Gradient in x-direction (cross-track)
    :type grad_x: np.ndarray
    :param grad_y: Gradient in y-direction (along-track) 
    :type grad_y: np.ndarray
    
    :returns: Gradient orientation in radians
    :rtype: np.ndarray
    
    :Note:
        Orientation is computed as arctan(grad_y / grad_x)
    """
    return np.arctan2(grad_y, grad_x)


def create_edge_mask(image: np.ndarray, 
                    threshold: float,
                    gradient_type: str = 'magnitude') -> np.ndarray:
    """
    Create a binary mask M that preserves sharp edges.
    
    :param image: Input image array
    :type image: np.ndarray
    :param threshold: Threshold for gradient magnitude
    :type threshold: float
    :param gradient_type: Type of gradient to use ('magnitude', 'x', or 'y')
    :type gradient_type: str
    
    :returns: Binary mask (0 for homogeneous regions, 1 for edges)
    :rtype: np.ndarray
    
    :Note:
        For SST images, typical threshold is 0.8°C/pixel
        For chlorophyll-a images, typical threshold is 0.1 mg/m³/pixel (log-transformed)
    """
    grad_x, grad_y, grad_mag = compute_gradient(image)
    
    if gradient_type == 'magnitude':
        gradient_field = grad_mag
    elif gradient_type == 'x':
        gradient_field = np.abs(grad_x)
    elif gradient_type == 'y':
        gradient_field = np.abs(grad_y)
    else:
        raise ValueError(f"Unknown gradient_type: {gradient_type}")
    
    # Create binary mask
    mask = (gradient_field >= threshold).astype(float)
    
    return mask


def solve_poisson_dct(g: np.ndarray) -> np.ndarray:
    """
    Solve the Poisson equation Δu = g using Discrete Cosine Transform.
    
    :param g: Right-hand side of Poisson equation
    :type g: np.ndarray
    
    :returns: Solution u to the Poisson equation
    :rtype: np.ndarray
    
    :Note:
        This function implements Equation (12) from the paper using DCT
    """
    M, N = g.shape
    
    # Compute 2D DCT of g
    G_c = fftpack.dctn(g, type=2, norm='ortho')
    
    # Create the frequency domain divisor
    # Avoid division by zero for the DC component
    m_freq = np.arange(M).reshape(-1, 1)
    n_freq = np.arange(N).reshape(1, -1)
    
    divisor = 2 * (np.cos(np.pi * m_freq / M) + np.cos(np.pi * n_freq / N) - 2)
    
    # Handle DC component (m=0, n=0) to avoid division by zero
    divisor[0, 0] = 1.0
    U_c = G_c / divisor
    U_c[0, 0] = 0  # Set DC component to zero
    
    # Compute inverse DCT to get u
    u = fftpack.idctn(U_c, type=2, norm='ortho')
    
    return u


def compute_laplacian_rhs(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute the right-hand side g for the Poisson equation.
    
    :param h: Current image estimate
    :type h: np.ndarray
    :param mask: Binary edge mask
    :type mask: np.ndarray
    
    :returns: Right-hand side g = ∂²ₓh + M·∂²ᵧh
    :rtype: np.ndarray
    
    :Note:
        Implements the discrete Laplacian computation from Section 2.2
    """
    M, N = h.shape
    g = np.zeros_like(h)
    
    # Compute discrete Laplacian components
    # Using centered differences for second derivatives
    for i in range(1, M-1):
        for j in range(1, N-1):
            # ∂²ₓh component (cross-track)
            d2x = h[i, j-1] + h[i, j+1] - 2*h[i, j]
            
            # ∂²ᵧh component (along-track) 
            d2y = h[i-1, j] + h[i+1, j] - 2*h[i, j]
            
            # Combined with mask: g = ∂²ₓh + M·∂²ᵧh
            g[i, j] = d2x + mask[i, j] * d2y
    
    # Handle boundaries with forward/backward differences
    # Top and bottom rows
    g[0, :] = h[1, :] - h[0, :]
    g[-1, :] = h[-2, :] - h[-1, :]
    
    # Left and right columns
    g[:, 0] = h[:, 1] - h[:, 0]
    g[:, -1] = h[:, -2] - h[:, -1]
    
    return g


def bouali_destriping(image: np.ndarray,
                     threshold: float,
                     n_iterations: int = 7,
                     lambda0: float = 1.0,
                     apply_filter: bool = True,
                     filter_size: int = 3,
                     return_components: bool = False) -> Union[np.ndarray, dict]:
    """
    Apply the Bouali destriping algorithm to reduce stripe noise in ocean satellite imagery.
    
    :param image: Input image with stripe noise (assumed in swath projection)
    :type image: np.ndarray
    :param threshold: Threshold for edge detection mask
    :type threshold: float
    :param n_iterations: Number of iterations (default: 7)
    :type n_iterations: int
    :param lambda0: Lagrange multiplier (default: 1.0)
    :type lambda0: float
    :param apply_filter: Whether to apply low-pass filter to final v_k
    :type apply_filter: bool
    :param filter_size: Size of median filter kernel if apply_filter=True
    :type filter_size: int
    :param return_components: If True, return dictionary with all components
    :type return_components: bool
    
    :returns: Corrected image or dictionary with components
    :rtype: Union[np.ndarray, dict]
    
    :Example:
    
    >>> # For SST data
    >>> sst_corrected = bouali_destriping(sst_image, threshold=0.8)
    >>> 
    >>> # For chlorophyll-a data (log-transformed)
    >>> chl_log = np.log10(chl_image + 1e-6)
    >>> chl_corrected = bouali_destriping(chl_log, threshold=0.1)
    
    :Note:
        The algorithm performs a multi-layer decomposition:
        f = u₀ + v₀ = u₀ + u₁ + v₁ = ... = Σᵢuᵢ + vₖ
        
        Where vₖ contains primarily stripe noise after k iterations.
    """
    # Initialize
    f = image.copy()
    h = f.copy()
    
    # Create edge preservation mask
    mask = create_edge_mask(f, threshold, gradient_type='magnitude')
    
    # Store decomposition components
    u_components = []
    v_components = []
    
    # Iterative decomposition
    for iteration in range(n_iterations):
        # Update lambda for this iteration (dyadic scheme optional)
        # Paper suggests λ₀ · 2^(-k) but mentions using fixed λ₀=1 works well
        lambda_k = lambda0 * (2.0 ** (-iteration)) if iteration > 0 else lambda0
        
        # Compute right-hand side of Poisson equation
        g = compute_laplacian_rhs(h, mask)
        
        # Solve Poisson equation using DCT
        u = solve_poisson_dct(g)
        
        # Compute residual
        v = h - u
        
        # Store components
        u_components.append(u.copy())
        v_components.append(v.copy())
        
        # Update h for next iteration
        h = v.copy()
    
    # Final v_k contains primarily stripe noise
    v_final = v_components[-1]
    
    # Apply low-pass filter to v_k if requested
    if apply_filter and filter_size > 1:
        from scipy.ndimage import median_filter
        v_final_filtered = median_filter(v_final, size=(1, filter_size))
    else:
        v_final_filtered = v_final
    
    # Reconstruct corrected image
    # û = f - v_k (after filtering)
    corrected_image = f - v_final_filtered
    
    if return_components:
        return {
            'corrected': corrected_image,
            'original': f,
            'u_components': u_components,
            'v_components': v_components,
            'v_final': v_final,
            'v_filtered': v_final_filtered,
            'mask': mask
        }
    else:
        return corrected_image


def correct_gradient_field(image: np.ndarray,
                          threshold: float,
                          n_iterations: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correct the gradient field of an image by removing stripe noise effects.
    
    :param image: Input image with stripe noise
    :type image: np.ndarray
    :param threshold: Threshold for edge detection
    :type threshold: float
    :param n_iterations: Number of iterations
    :type n_iterations: int
    
    :returns: Tuple of (corrected_grad_x, corrected_grad_y, corrected_grad_magnitude)
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    
    :Note:
        Implements Equation (7) from the paper: ∇û = ∇f - ∇vₖ
    """
    # Apply destriping to get components
    result = bouali_destriping(image, threshold, n_iterations, 
                              return_components=True)
    
    # Compute original gradient
    grad_x_orig, grad_y_orig, _ = compute_gradient(result['original'])
    
    # Compute gradient of noise component
    grad_x_noise, grad_y_noise, _ = compute_gradient(result['v_filtered'])
    
    # Corrected gradient: ∇û = ∇f - ∇vₖ
    grad_x_corrected = grad_x_orig - grad_x_noise
    grad_y_corrected = grad_y_orig - grad_y_noise
    grad_mag_corrected = np.sqrt(grad_x_corrected**2 + grad_y_corrected**2)
    
    return grad_x_corrected, grad_y_corrected, grad_mag_corrected


def compute_nif(original: np.ndarray, corrected: np.ndarray) -> float:
    """
    Compute the Normalized Improvement Factor (NIF).
    
    :param original: Original image with stripe noise
    :type original: np.ndarray
    :param corrected: Corrected image
    :type corrected: np.ndarray
    
    :returns: NIF value (percentage of improvement)
    :rtype: float
    
    :Note:
        NIF measures the reduction of spatial variations in the 
        cross-track direction (affected by stripe noise).
    """
    # Compute variance in cross-track direction
    var_orig = np.var(np.diff(original, axis=1))
    var_corr = np.var(np.diff(corrected, axis=1))
    
    # NIF as percentage improvement
    nif = 100 * (1 - var_corr / var_orig)
    
    return nif


def compute_ndf(original: np.ndarray, corrected: np.ndarray) -> float:
    """
    Compute the Normalized Distortion Factor (NDF).
    
    :param original: Original image
    :type original: np.ndarray
    :param corrected: Corrected image
    :type corrected: np.ndarray
    
    :returns: NDF value (should be ≤ 0.95 to ensure no visible blur)
    :rtype: float
    
    :Note:
        NDF measures information loss in the scanning direction.
        A threshold of NDF ≤ 0.95 ensures no perceptible blur.
    """
    # Compute power in along-track direction
    power_orig = np.sum(np.abs(np.fft.fft(original, axis=0))**2)
    power_corr = np.sum(np.abs(np.fft.fft(corrected, axis=0))**2)
    
    # NDF ratio
    ndf = power_corr / power_orig
    
    return ndf


def process_ocean_data(data: np.ndarray,
                      data_type: str = 'sst',
                      custom_threshold: Optional[float] = None,
                      verbose: bool = True) -> dict:
    """
    Convenient wrapper function for processing ocean satellite data.
    
    :param data: Input ocean data (SST or chlorophyll-a)
    :type data: np.ndarray
    :param data_type: Type of data ('sst' or 'chlorophyll')
    :type data_type: str
    :param custom_threshold: Custom threshold (overrides defaults)
    :type custom_threshold: Optional[float]
    :param verbose: Print processing information
    :type verbose: bool
    
    :returns: Dictionary containing corrected data and diagnostics
    :rtype: dict
    
    :Example:
    
    >>> result = process_ocean_data(sst_image, data_type='sst')
    >>> corrected_sst = result['corrected']
    >>> print(f"NIF: {result['nif']:.1f}%")
    """
    # Set default thresholds based on data type
    if custom_threshold is not None:
        threshold = custom_threshold
    elif data_type.lower() == 'sst':
        threshold = 0.8  # °C/pixel
    elif data_type.lower() in ['chlorophyll', 'chl', 'chlorophyll-a']:
        # For chlorophyll, apply log transform first
        if np.min(data) >= 0:
            data = np.log10(data + 1e-6)
        threshold = 0.1  # mg/m³/pixel for log-transformed data
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'sst' or 'chlorophyll'")
    
    if verbose:
        print(f"Processing {data_type} data with threshold={threshold}")
    
    # Apply destriping algorithm
    result = bouali_destriping(data, threshold, return_components=True)
    
    # Compute diagnostics
    nif = compute_nif(data, result['corrected'])
    ndf = compute_ndf(data, result['corrected'])
    
    # Get corrected gradient field
    grad_x_corr, grad_y_corr, grad_mag_corr = correct_gradient_field(data, threshold)
    
    if verbose:
        print(f"NIF: {nif:.1f}%")
        print(f"NDF: {ndf:.3f}")
        if ndf > 0.95:
            print("Warning: NDF > 0.95, some blur may be visible")
    
    return {
        'corrected': result['corrected'],
        'original': data,
        'gradient_x': grad_x_corr,
        'gradient_y': grad_y_corr, 
        'gradient_magnitude': grad_mag_corr,
        'nif': nif,
        'ndf': ndf,
        'mask': result['mask'],
        'v_final': result['v_final']
    }


if __name__ == "__main__":
    # Example usage
    print("Bouali Destriping Algorithm Module")
    print("===================================")
    print("This module implements stripe noise reduction for ocean satellite imagery")
    print("\nExample usage:")
    print(">>> import numpy as np")
    print(">>> from bouali_destriping import process_ocean_data")
    print(">>> ")
    print(">>> # Create synthetic SST data with stripes")
    print(">>> sst = np.random.randn(500, 500) * 2 + 20")
    print(">>> # Add stripe noise")
    print(">>> stripes = np.sin(np.linspace(0, 20*np.pi, 500)).reshape(1, -1)")
    print(">>> sst_noisy = sst + 0.5 * stripes")
    print(">>> ")
    print(">>> # Process the data")
    print(">>> result = process_ocean_data(sst_noisy, data_type='sst')")
    print(">>> sst_clean = result['corrected']")
