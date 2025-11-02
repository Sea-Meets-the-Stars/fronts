"""
SST Preprocessing: Median Filtering and Conversion to Integer Format

This module provides functions to convert raw SST data (in Kelvin or Celsius)
to the median-filtered integer format expected by the thin() algorithm.

This replicates the preprocessing done by the FORTRAN code in:
- ConditionInput subroutine (scaling and conversion)
- median subroutine (median filtering)

Author: Python conversion by Claude
Date: 2025
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Union
import warnings


def sst_to_counts(sst_data: np.ndarray,
                  input_scale: float = 1.0,
                  input_offset: float = 0.0,
                  output_scale: float = 0.01,
                  output_offset: float = 268.15,
                  max_sst_out: float = 313.15,
                  sst_range: int = 400,
                  fill_value: float = np.nan,
                  in_kelvin: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Convert real SST values to integer digital counts.
    
    This function replicates the FORTRAN ConditionInput subroutine.
    
    Parameters
    ----------
    sst_data : np.ndarray
        Input SST data (2D array) in Kelvin or Celsius
    input_scale : float, optional
        Scale factor for input data (default: 1.0 for direct values)
    input_offset : float, optional
        Offset for input data (default: 0.0)
    output_scale : float, optional
        Scale factor for output counts (default: 0.01 K/count)
    output_offset : float, optional
        Minimum SST value in Kelvin (default: 268.15 K = -5°C)
    max_sst_out : float, optional
        Maximum SST value in Kelvin (default: 313.15 K = 40°C)
    sst_range : int, optional
        Maximum digital count value (default: 400)
    fill_value : float, optional
        Value indicating missing/invalid data (default: np.nan)
    in_kelvin : bool, optional
        True if input is in Kelvin, False if Celsius (default: True)
        
    Returns
    -------
    sst_counts : np.ndarray
        Integer array (int16) with values from 0 to sst_range
        Invalid data set to -32768 (standard int16 fill value)
    stats : dict
        Dictionary with conversion statistics:
        - 'min_sst': Minimum input SST (excluding fill values)
        - 'max_sst': Maximum input SST (excluding fill values)
        - 'n_too_small': Number of values below minimum
        - 'n_too_large': Number of values above maximum
        - 'n_fill': Number of fill values
        - 'n_valid': Number of valid output values
        
    Notes
    -----
    From FORTRAN code:
        SST_counts_out = (SST_in * ScaleInput) + InputOffset2OutputOffset
        where InputOffset2OutputOffset accounts for Kelvin/Celsius conversion
        
    Example
    -------
    >>> sst_kelvin = np.random.uniform(273, 303, (100, 100))
    >>> sst_counts, stats = sst_to_counts(sst_kelvin)
    >>> print(f"Valid pixels: {stats['n_valid']}")
    """
    
    # Handle fill values
    if np.isnan(fill_value):
        valid_mask = ~np.isnan(sst_data)
    else:
        valid_mask = sst_data != fill_value
    
    # Convert Celsius to Kelvin if needed
    sst_kelvin = sst_data.copy()
    if not in_kelvin:
        sst_kelvin[valid_mask] += 273.15
    
    # Apply input scaling if needed
    if input_scale != 1.0 or input_offset != 0.0:
        sst_kelvin[valid_mask] = (sst_kelvin[valid_mask] * input_scale + 
                                   input_offset)
    
    # Calculate statistics on input
    min_sst = np.min(sst_kelvin[valid_mask]) if valid_mask.any() else np.nan
    max_sst = np.max(sst_kelvin[valid_mask]) if valid_mask.any() else np.nan
    
    # Convert to output counts
    # Formula: counts = (SST_K - output_offset) / output_scale
    sst_counts = np.full(sst_data.shape, -32768, dtype=np.int16)
    
    if valid_mask.any():
        # Calculate counts for valid data
        counts_float = (sst_kelvin[valid_mask] - output_offset) / output_scale
        counts_int = np.round(counts_float).astype(np.int32)
        
        # Count values outside acceptable range
        too_small = counts_int <= 0
        too_large = counts_int > sst_range
        
        n_too_small = np.sum(too_small)
        n_too_large = np.sum(too_large)
        
        # Clip to valid range and set out-of-range to fill value
        counts_int[too_small] = -32768
        counts_int[too_large] = -32768
        valid_counts = (counts_int > 0) & (counts_int <= sst_range)
        
        # Assign to output array
        temp_counts = np.full(valid_mask.shape, -32768, dtype=np.int16)
        valid_indices = np.where(valid_mask)
        temp_counts[valid_indices] = counts_int
        sst_counts = temp_counts
        
        n_valid = np.sum(sst_counts > 0)
    else:
        n_too_small = 0
        n_too_large = 0
        n_valid = 0
    
    # Compile statistics
    stats = {
        'min_sst': min_sst,
        'max_sst': max_sst,
        'n_too_small': n_too_small,
        'n_too_large': n_too_large,
        'n_fill': np.sum(~valid_mask),
        'n_valid': n_valid
    }
    
    return sst_counts, stats


def median_filter_sst(sst_counts: np.ndarray,
                      window_size: int = 5,
                      fill_value: int = -32768) -> np.ndarray:
    """
    Apply median filter to SST count data.
    
    This function replicates the FORTRAN median subroutine, using a fast
    median filter that properly handles fill values.
    
    Parameters
    ----------
    sst_counts : np.ndarray
        Integer SST counts (int16), output from sst_to_counts()
    window_size : int, optional
        Size of median filter window (default: 5 for 5x5)
        Must be odd number
    fill_value : int, optional
        Value indicating missing/invalid data (default: -32768)
        
    Returns
    -------
    filtered_counts : np.ndarray
        Median-filtered SST counts (int16)
        Pixels where majority of window is fill value remain as fill value
        
    Notes
    -----
    From FORTRAN code:
        "If the majority of values in an nxn region are fill values, 
         they will be returned as fill values."
    
    The scipy median_filter doesn't handle fill values well, so we use
    a custom approach with masking.
    
    Example
    -------
    >>> sst_counts = np.random.randint(50, 200, (100, 100), dtype=np.int16)
    >>> filtered = median_filter_sst(sst_counts, window_size=5)
    """
    
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    if window_size == 1:
        # No filtering needed
        return sst_counts.copy()
    
    # Create mask of valid data
    valid_mask = sst_counts != fill_value
    
    # For median filtering with fill values, we need to:
    # 1. Replace fill values with a temporary value (0)
    # 2. Apply median filter
    # 3. Count valid pixels in each window
    # 4. Restore fill values where window was mostly fill values
    
    # Create working copy with fill values set to 0
    work_array = sst_counts.copy()
    work_array[~valid_mask] = 0
    
    # Apply median filter to the data
    filtered = ndimage.median_filter(work_array, size=window_size, mode='reflect')
    
    # Count valid pixels in each window using uniform filter
    valid_count = ndimage.uniform_filter(
        valid_mask.astype(np.float32), 
        size=window_size, 
        mode='reflect'
    )
    valid_count *= (window_size * window_size)  # Convert from fraction to count
    
    # Threshold: if less than half the window is valid, set to fill value
    # This matches FORTRAN behavior: "majority of values are fill values"
    threshold = (window_size * window_size) / 2.0
    mostly_invalid = valid_count < threshold
    
    # Create output array
    filtered_counts = filtered.astype(np.int16)
    filtered_counts[mostly_invalid] = fill_value
    
    return filtered_counts


def preprocess_sst(sst_data: np.ndarray,
                   median_window: int = 5,
                   input_scale: float = 1.0,
                   input_offset: float = 0.0,
                   output_scale: float = 0.01,
                   output_offset: float = 268.15,
                   max_sst_out: float = 313.15,
                   sst_range: int = 400,
                   fill_value: float = np.nan,
                   in_kelvin: bool = True,
                   verbose: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline: convert and median filter SST data.
    
    This is the main convenience function that combines conversion and
    filtering in one step.
    
    Parameters
    ----------
    sst_data : np.ndarray
        Input SST data (2D array) in Kelvin or Celsius
    median_window : int, optional
        Size of median filter window (default: 5)
    input_scale : float, optional
        Scale factor for input data (default: 1.0)
    input_offset : float, optional
        Offset for input data (default: 0.0)
    output_scale : float, optional
        Output scale in K/count (default: 0.01)
    output_offset : float, optional
        Minimum SST in Kelvin (default: 268.15 K = -5°C)
    max_sst_out : float, optional
        Maximum SST in Kelvin (default: 313.15 K = 40°C)
    sst_range : int, optional
        Maximum count value (default: 400)
    fill_value : float, optional
        Input fill value (default: np.nan)
    in_kelvin : bool, optional
        True if input is Kelvin (default: True)
    verbose : bool, optional
        Print statistics (default: True)
        
    Returns
    -------
    med_sst : np.ndarray
        Median-filtered SST in integer counts (int16)
        Ready for use with thin() function
    stats : dict
        Processing statistics
        
    Example
    -------
    >>> # Load satellite SST data (Kelvin)
    >>> sst = load_satellite_sst('my_file.nc')
    >>> 
    >>> # Preprocess for front detection
    >>> med_sst, stats = preprocess_sst(sst, median_window=5, verbose=True)
    >>> 
    >>> # Now ready for thinning
    >>> from thin_subroutine import thin
    >>> thinned = thin("output.nc", time, med_sst, merged_fronts)
    """
    
    # Step 1: Convert to integer counts
    if verbose:
        print("Step 1: Converting SST to integer counts...")
    
    sst_counts, stats = sst_to_counts(
        sst_data,
        input_scale=input_scale,
        input_offset=input_offset,
        output_scale=output_scale,
        output_offset=output_offset,
        max_sst_out=max_sst_out,
        sst_range=sst_range,
        fill_value=fill_value,
        in_kelvin=in_kelvin
    )
    
    if verbose:
        print(f"  Input SST range: {stats['min_sst']:.2f} - {stats['max_sst']:.2f} K")
        print(f"  Valid pixels: {stats['n_valid']:,}")
        print(f"  Fill value pixels: {stats['n_fill']:,}")
        if stats['n_too_small'] > 0:
            print(f"  WARNING: {stats['n_too_small']:,} pixels below minimum SST")
        if stats['n_too_large'] > 0:
            print(f"  WARNING: {stats['n_too_large']:,} pixels above maximum SST")
    
    # Step 2: Apply median filter
    if verbose:
        print(f"\nStep 2: Applying {median_window}x{median_window} median filter...")
    
    med_sst = median_filter_sst(sst_counts, window_size=median_window)
    
    # Update statistics
    n_valid_after = np.sum(med_sst > 0)
    stats['n_valid_after_median'] = n_valid_after
    
    if verbose:
        print(f"  Valid pixels after filtering: {n_valid_after:,}")
        if n_valid_after > 0:
            print(f"  Output data range: {np.min(med_sst[med_sst > 0])} - "
                  f"{np.max(med_sst[med_sst > 0])} counts")
        else:
            print(f"  WARNING: No valid pixels after filtering!")
        print("\nPreprocessing complete!")
    
    return med_sst, stats


def counts_to_sst(sst_counts: np.ndarray,
                  output_scale: float = 0.01,
                  output_offset: float = 268.15,
                  to_celsius: bool = True,
                  fill_value: int = -32768) -> np.ndarray:
    """
    Convert integer counts back to real SST values.
    
    Useful for validation and visualization.
    
    Parameters
    ----------
    sst_counts : np.ndarray
        Integer SST counts (int16)
    output_scale : float, optional
        Scale factor (default: 0.01 K/count)
    output_offset : float, optional
        Offset in Kelvin (default: 268.15 K)
    to_celsius : bool, optional
        Convert to Celsius (default: True)
    fill_value : int, optional
        Fill value in counts (default: -32768)
        
    Returns
    -------
    sst : np.ndarray
        Real SST values (float32) in Celsius or Kelvin
        Invalid pixels set to np.nan
        
    Example
    -------
    >>> sst_kelvin = np.random.uniform(273, 303, (100, 100))
    >>> counts, _ = sst_to_counts(sst_kelvin)
    >>> sst_reconstructed = counts_to_sst(counts, to_celsius=False)
    >>> np.allclose(sst_kelvin[~np.isnan(sst_reconstructed)], 
    ...             sst_reconstructed[~np.isnan(sst_reconstructed)], rtol=0.01)
    True
    """
    
    # Create output array
    sst = np.full(sst_counts.shape, np.nan, dtype=np.float32)
    
    # Find valid data
    valid_mask = sst_counts != fill_value
    
    # Convert counts to Kelvin
    if valid_mask.any():
        sst[valid_mask] = (sst_counts[valid_mask] * output_scale + output_offset)
        
        # Convert to Celsius if requested
        if to_celsius:
            sst[valid_mask] -= 273.15
    
    return sst


# Example usage and testing
def example_usage():
    """
    Demonstrate the preprocessing pipeline with synthetic data.
    """
    print("=" * 70)
    print("SST Preprocessing Example")
    print("=" * 70)
    
    # Create synthetic SST data
    print("\nGenerating synthetic SST data...")
    len_x, len_y = 200, 200
    
    # Create temperature field with a front
    x = np.arange(len_x)
    y = np.arange(len_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Meandering temperature front
    front_position = len_x // 2 + 10 * np.sin(2 * np.pi * y / 50)
    
    # Temperature in Kelvin (warm on left, cold on right)
    # Realistic ocean temperatures: 5°C to 25°C (278K to 298K)
    sst_kelvin = np.zeros((len_x, len_y), dtype=np.float32)
    for i in range(len_x):
        for j in range(len_y):
            dist_from_front = i - front_position[j]
            # Temperature decreases across front (15°C warm side, 10°C cold side)
            temp = 288.15 - 0.05 * dist_from_front
            # Add realistic noise (0.5K standard deviation)
            temp += np.random.normal(0, 0.5)
            # Clip to realistic ocean range
            temp = np.clip(temp, 278.15, 298.15)  # 5°C to 25°C
            sst_kelvin[i, j] = temp
    
    # Add some land/invalid areas (set to NaN)
    # Circular land mass
    center_x, center_y = len_x // 4, len_y // 4
    for i in range(len_x):
        for j in range(len_y):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < 30:
                sst_kelvin[i, j] = np.nan
    
    print(f"  Created {len_x}x{len_y} SST field")
    print(f"  Temperature range: {np.nanmin(sst_kelvin):.2f} - "
          f"{np.nanmax(sst_kelvin):.2f} K")
    print(f"  ({np.nanmin(sst_kelvin) - 273.15:.2f} - "
          f"{np.nanmax(sst_kelvin) - 273.15:.2f} °C)")
    
    # Preprocess the data
    print("\n" + "=" * 70)
    med_sst, stats = preprocess_sst(
        sst_kelvin,
        median_window=5,
        output_scale=0.01,
        output_offset=268.15,  # -5°C minimum
        max_sst_out=323.15,    # 50°C maximum (extended for wider coverage)
        sst_range=5500,        # (323.15 - 268.15) / 0.01 = 5500
        verbose=True
    )
    
    # Convert back to check
    print("\n" + "=" * 70)
    print("Validation: Converting back to temperature...")
    sst_reconstructed = counts_to_sst(med_sst, to_celsius=False)
    
    # Compare (excluding NaN)
    valid_original = ~np.isnan(sst_kelvin)
    valid_reconstructed = ~np.isnan(sst_reconstructed)
    valid_both = valid_original & valid_reconstructed
    
    if valid_both.any():
        diff = np.abs(sst_kelvin[valid_both] - sst_reconstructed[valid_both])
        print(f"  Mean absolute error: {np.mean(diff):.4f} K")
        print(f"  Max absolute error: {np.max(diff):.4f} K")
        print(f"  (Expected < {0.01:.4f} K due to rounding)")
    
    print("\n" + "=" * 70)
    print("Preprocessing complete! Output ready for thin() function.")
    print("=" * 70)
    
    return sst_kelvin, med_sst, stats


if __name__ == "__main__":
    # Run example
    sst_input, sst_output, statistics = example_usage()
    
    # Additional info
    print("\nTo use with the thin() function:")
    print("  from thin_subroutine import thin")
    print("  thinned = thin('output.nc', time, sst_output, merged_fronts)")
