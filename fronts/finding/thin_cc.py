"""
Python conversion of thin_subroutine.f and related subroutines from 
the fronts/finding/CC/ FORTRAN code.

This module implements the thin() subroutine which thins fronts in both
the vertical (J-direction) and horizontal (I-direction) based on temperature
gradients in a median-filtered SST image.

Original FORTRAN code by Peter Cornillon, University of Rhode Island
Converted to Python by Claude (2025)
"""

import numpy as np
from typing import Tuple, Optional
import sys

import netCDF4 as nc
HAS_NETCDF = True


def thin(med_sst: np.ndarray,
         merged_fronts: np.ndarray,
         debug: int = 0,
         unit_log: Optional[object] = sys.stdout) -> np.ndarray:
    """
    Thin fronts in merged edge image based on maximum temperature gradients.
    
    This subroutine thins fronts by selecting pixels with the maximum
    temperature gradient along continuous front segments in both vertical
    (J-direction) and horizontal (I-direction).
    
    Parameters
    ----------
    med_sst : np.ndarray
        Median-filtered SST image (LenX, LenY) with values 0-255
        Values <= 8 are considered invalid/land
    merged_fronts : np.ndarray
        Original edge image (LenX, LenY) where 4 indicates a front pixel
    nc_merged_id : int, optional
        NetCDF file ID for merged file (for compatibility)
    merged_id : int, optional
        NetCDF variable ID (for compatibility)
    seconds_since_1970_id : int, optional
        NetCDF time variable ID (for compatibility)
    debug : int, optional
        Debug level (0=off, 1=on), default 0
    unit_log : file-like, optional
        Log file handle, default sys.stdout
        
    Returns
    -------
    thinned_fronts : np.ndarray
        Thinned front image (LenX, LenY) where 4 indicates a front pixel
        
    Notes
    -----
    The thinning algorithm works in two passes:
    1. FIRST PASS: Thin fronts in the J-direction (vertical)
       - For each column (i), scan rows (j)
       - Find segments of continuous front pixels
       - Select the pixel with maximum abs(SST[i,j+1] - SST[i,j-1])
       - Keep only if segment has >= 2 non-front pixels after it
       
    2. SECOND PASS: Thin fronts in the I-direction (horizontal)
       - For each row (j), scan columns (i)
       - Find segments of continuous front pixels
       - Select the pixel with maximum abs(SST[i+1,j] - SST[i-1,j])
       - Keep only if segment has >= 2 non-front pixels after it
       
    Original FORTRAN notes:
    - 21-May-2009 - PCC - The upper limits of the 23032 and 23034 loops
      were backward; LenY for the first dimension and LenX for the 
      second dimension. I reversed them.
    """
    
    len_x, len_y = med_sst.shape
    
    if debug == 1:
        #print(f"thin: Processing {mt_filename}", file=unit_log)
        print(f"thin: Array dimensions: LenX={len_x}, LenY={len_y}", file=unit_log)
    
    # Validate input dimensions
    if merged_fronts.shape != (len_x, len_y):
        raise ValueError(f"merged_fronts shape {merged_fronts.shape} does not match "
                        f"med_sst shape {(len_x, len_y)}")
    
    # Initialize thinned fronts array to zero
    thinned_fronts = np.zeros((len_x, len_y), dtype=np.int16)
    
    # Set window size (from FORTRAN: win = 5, though not used in the actual algorithm)
    win = 5
    
    # Maximum value for int16 (used for diftemp initialization)
    max_int16 = 32767
    
    # ====================================================================
    # FIRST PASS: Thin fronts in the J-direction (vertical)
    # ====================================================================
    # For each column i, scan through rows j to find front segments
    # and keep only the pixel with maximum temperature gradient
    
    if debug == 1:
        print("thin: Starting FIRST PASS (J-direction/vertical)", file=unit_log)
    
    for i in range(len_x):
        maxdif = 2  # Minimum threshold for keeping a front
        diftemp = max_int16  # Initialize to large value
        imax = 0
        jmax = 0
        count = 0
        
        # Scan rows (j) from 2nd to 2nd-to-last (avoiding boundaries)
        for j in range(1, len_y - 1):
            if merged_fronts[i, j] == 4:  # This is a front pixel
                count = 0
                
                # Calculate temperature difference if both neighbors are valid
                # (valid means > 8, which excludes land/invalid data)
                if min(med_sst[i, j-1], med_sst[i, j+1]) > 8:
                    diftemp = abs(int(med_sst[i, j+1]) - int(med_sst[i, j-1]))
                else:
                    diftemp = 0  # Invalid data, don't consider this front
                
                # Track the maximum gradient in this front segment
                if maxdif < diftemp:
                    maxdif = diftemp
                    imax = i
                    jmax = j
                    
            else:  # Not a front pixel
                # We've reached the end of a front segment
                if maxdif != 2:  # We found a valid front segment
                    if count >= 2:  # Segment has enough non-front pixels after it
                        thinned_fronts[imax, jmax] = 4
                        maxdif = 2  # Reset for next segment
                    count += 1
                    
    if debug == 1:
        print("thin: Completed FIRST PASS", file=unit_log)
        print(f"thin: Front pixels after first pass: {np.sum(thinned_fronts == 4)}", 
              file=unit_log)
    
    # ====================================================================
    # SECOND PASS: Thin fronts in the I-direction (horizontal)
    # ====================================================================
    # For each row j, scan through columns i to find front segments
    # and keep only the pixel with maximum temperature gradient
    
    if debug == 1:
        print("thin: Starting SECOND PASS (I-direction/horizontal)", file=unit_log)
    
    for j in range(len_y):
        maxdif = 2  # Minimum threshold for keeping a front
        diftemp = max_int16  # Initialize to large value
        imax = 0
        jmax = 0
        count = 0
        
        # Scan columns (i) from 2nd to 2nd-to-last (avoiding boundaries)
        for i in range(1, len_x - 1):
            if merged_fronts[i, j] == 4:  # This is a front pixel
                count = 0
                
                # Calculate temperature difference if both neighbors are valid
                if min(med_sst[i-1, j], med_sst[i+1, j]) > 8:
                    diftemp = abs(int(med_sst[i+1, j]) - int(med_sst[i-1, j]))
                else:
                    diftemp = 0  # Invalid data, don't consider this front
                
                # Track the maximum gradient in this front segment
                if maxdif < diftemp:
                    maxdif = diftemp
                    imax = i
                    jmax = j
                    
            else:  # Not a front pixel
                # We've reached the end of a front segment
                if maxdif != 2:  # We found a valid front segment
                    if count >= 2:  # Segment has enough non-front pixels after it
                        thinned_fronts[imax, jmax] = 4
                        maxdif = 2  # Reset for next segment
                    count += 1
    
    if debug == 1:
        print("thin: Completed SECOND PASS", file=unit_log)
        print(f"thin: Final front pixels: {np.sum(thinned_fronts == 4)}", file=unit_log)
    
    return thinned_fronts


def write_merged_thinned(mt_filename: str,
                        hdate: float,
                        lats: np.ndarray,
                        lons: np.ndarray,
                        front_array: np.ndarray,
                        merged_or_thinned: int,
                        nc_merged_id: Optional[object] = None,
                        fill_value: int = -32768,
                        hour_window: Optional[int] = None,
                        image_window: Optional[int] = None,
                        pmerge_version: str = "2.09",
                        thin_version: str = "2.09") -> None:
    """
    Write merged or thinned fronts to a NetCDF file.
    
    Parameters
    ----------
    mt_filename : str
        Output NetCDF filename
    hdate : float
        Date/time value (hours since reference or similar)
    lats : np.ndarray
        Latitude array (LenY,)
    lons : np.ndarray
        Longitude array (LenX,)
    front_array : np.ndarray
        Front data array (LenX, LenY) - either merged or thinned
    merged_or_thinned : int
        1 for merged fronts, 2 for thinned fronts
    nc_merged_id : object, optional
        Input NetCDF dataset for copying time variable
    fill_value : int, optional
        Fill value for the data array, default -32768
    hour_window : int, optional
        Hours over which fronts were merged (for merged fronts only)
    image_window : int, optional
        Number of images over which fronts were merged (for merged fronts only)
    pmerge_version : str, optional
        Version number of Pmerge_Main program
    thin_version : str, optional
        Version number of Thin_Main program
        
    Notes
    -----
    This function creates a NetCDF file with the fronts data and appropriate
    metadata. It mimics the FORTRAN WriteMergedThinned subroutine but uses
    Python's netCDF4 library.
    
    Requires netCDF4 package to be installed.
    """
    
    if not HAS_NETCDF:
        raise ImportError("netCDF4 package is required for write_merged_thinned(). "
                         "Install it with: pip install netCDF4")
    
    len_x, len_y = front_array.shape
    
    # Create NetCDF file
    with nc.Dataset(mt_filename, 'w', format='NETCDF4') as ncfile:
        
        # Define dimensions
        ncfile.createDimension('nx', len_x)
        ncfile.createDimension('ny', len_y)
        
        # Create coordinate variables
        lat_var = ncfile.createVariable('lat', 'f4', ('ny',))
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'
        
        lon_var = ncfile.createVariable('lon', 'f4', ('nx',))
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'
        
        # Write coordinate data
        if lats is not None and len(lats) > 0:
            lat_var[:] = lats
        if lons is not None and len(lons) > 0:
            lon_var[:] = lons
        
        # Create time variable
        time_var = ncfile.createVariable('DateTime', 'f8')
        time_var.standard_name = 'time'
        time_var.long_name = 'time since 1970-01-01 00:00:00.0'
        time_var.units = 'seconds since 1970-01-01 00:00:00.0'
        time_var[:] = hdate
        
        # Create fronts variable with chunking for better compression
        chunk_size = (min(len_x, 512), min(len_y, 512))
        
        if merged_or_thinned == 1:
            var_name = 'merged_fronts'
            long_name = 'Merged fronts from SIED'
        else:
            var_name = 'thinned_fronts'
            long_name = 'Thinned fronts from merged SIED fronts'
        
        fronts_var = ncfile.createVariable(
            var_name, 'i2', ('nx', 'ny'),
            chunksizes=chunk_size,
            zlib=True,
            complevel=4,
            fill_value=fill_value
        )
        
        fronts_var.long_name = long_name
        fronts_var.add_offset = 0.0
        fronts_var.scale_factor = 1.0
        fronts_var._FillValue = fill_value
        
        # Write front data
        fronts_var[:, :] = front_array
        
        # Add global attributes
        ncfile.Conventions = 'CF-1.6'
        ncfile.Institution = 'University of Rhode Island, Graduate School of Oceanography'
        
        if merged_or_thinned == 1:
            if hour_window is not None:
                ncfile.title = f'Fronts merged over {hour_window * 2} hours centered on this image for SIED'
            else:
                ncfile.title = 'Fronts merged for SIED'
            ncfile.source = f'Pmerge_Main version {pmerge_version}'
            ncfile.summary = ('The field in this file was generated by merging all fronts '
                            'found by SIED in multiple images.')
        else:
            ncfile.title = 'Thinned fronts'
            ncfile.source = f'Thin_Main version {thin_version}'
            ncfile.summary = ('The field in this file was generated by thinning '
                            'the field of merged SIED fronts.')
        
        # Copy time from input file if available
        if nc_merged_id is not None:
            try:
                time_in = nc_merged_id.variables['DateTime'][:]
                time_var[:] = time_in
            except (KeyError, AttributeError):
                pass


def print_array2(array_to_print: np.ndarray, 
                message: str,
                istrt: int = 0,
                iend: int = 9,
                jstrt: int = 0, 
                jend: int = 9,
                unit_out: object = sys.stdout) -> None:
    """
    Print a subset of a 2D array for debugging (mimics FORTRAN PrintArray2).
    
    Parameters
    ----------
    array_to_print : np.ndarray
        The array to print (any size)
    message : str
        Message to print with the array (should end with 'EOS')
    istrt, iend : int
        Row range to print (0-indexed in Python, 1-indexed in FORTRAN)
    jstrt, jend : int
        Column range to print (0-indexed in Python, 1-indexed in FORTRAN)
    unit_out : file-like
        Output stream, default sys.stdout
        
    Notes
    -----
    This function mimics the FORTRAN debug printing routine. The indices
    in Python are 0-based while FORTRAN uses 1-based indexing.
    """
    
    # Find 'EOS' marker in message
    eos_pos = message.find('EOS')
    if eos_pos > 0:
        message = message[:eos_pos]
    
    # Check array size
    i_range = iend - istrt + 1
    j_range = jend - jstrt + 1
    
    if i_range > 10 or j_range > 10:
        print("Range of arrays to print in debug exceeds 10x10.", file=unit_out)
        return
    
    # Print message
    print(f"\n{message}\n", file=unit_out)
    
    # Print column headers (using FORTRAN 1-based indexing for display)
    x_to_print = [istrt + i + 1 for i in range(i_range)]
    header = "          " + "".join(f"{x:5d}" for x in x_to_print)
    print(header, file=unit_out)
    
    # Print rows
    for j in range(j_range):
        row_idx = jstrt + j + 1  # FORTRAN 1-based for display
        row_data = array_to_print[istrt:iend+1, jstrt+j]
        row_str = f"{row_idx:5d}     " + "".join(f"{val:5d}" for val in row_data)
        print(row_str, file=unit_out)
    
    print("", file=unit_out)


# Example usage and test function
def test_thin():
    """
    Test the thin() function with synthetic data.
    """
    # Create test data
    len_x, len_y = 100, 100
    
    # Create synthetic SST field with a gradient
    med_sst = np.zeros((len_x, len_y), dtype=np.int16)
    for i in range(len_x):
        for j in range(len_y):
            med_sst[i, j] = min(255, max(9, 50 + i // 2 + np.random.randint(-5, 5)))
    
    # Create synthetic merged fronts (a vertical line and a horizontal line)
    merged_fronts = np.zeros((len_x, len_y), dtype=np.int16)
    merged_fronts[50, 20:80] = 4  # Vertical front
    merged_fronts[20:80, 50] = 4  # Horizontal front
    
    # Run thinning
    print("Testing thin() function...")
    thinned = thin(
        "test_output.nc",
        1234567890.0,
        med_sst,
        merged_fronts,
        debug=1
    )
    
    print(f"\nInput fronts: {np.sum(merged_fronts == 4)} pixels")
    print(f"Thinned fronts: {np.sum(thinned == 4)} pixels")
    print(f"Reduction: {100 * (1 - np.sum(thinned == 4) / np.sum(merged_fronts == 4)):.1f}%")
    
    return thinned


if __name__ == "__main__":
    # Run test
    result = test_thin()
    print("\nTest completed successfully!")
