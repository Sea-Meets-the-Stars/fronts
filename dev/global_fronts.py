# Puttering about applying our algorithm to a global dataset

import numpy as np
import xarray
from fronts.finding import algorithms
from fronts.finding import pyboa

def test_whole_one(divb2_file:str, outfile:str):

    # Load
    divb2 = xarray.open_dataset(divb2_file)['Divb2'].values

    # Find fronts
    fronts = algorithms.fronts_from_divb2(divb2, thin=True,
                                          verbose=True)

    # Save
    np.save(outfile, fronts.astype(np.int16))
    print(f'Wrote {outfile}')


def test_threshold_modes(fronts_file:str, center:tuple=(4000, 4000),
                         size:int=1000, wndw:int=64, prcnt:float=90,
                         test_dask:bool=True, n_workers=None):
    """
    Test front_thresh() function in generic, vectorized, and dask modes.

    Args:
        fronts_file (str): Path to the .npy file containing fronts data
        center (tuple): Center coordinates (y, x) for extraction
        size (int): Size of the square region to extract
        wndw (int): Window size for front_thresh
        prcnt (float): Percentile for thresholding
        test_dask (bool): Whether to test dask mode
        n_workers (int): Number of workers for dask mode
    """
    # Load the fronts data
    print(f'Loading {fronts_file}')
    fronts_data = np.load(fronts_file)

    # Extract 1000x1000 region centered at the specified location
    half_size = size // 2
    y_center, x_center = center
    y_start = y_center - half_size
    y_end = y_center + half_size
    x_start = x_center - half_size
    x_end = x_center + half_size

    print(f'Extracting region [{y_start}:{y_end}, {x_start}:{x_end}]')
    region = fronts_data[y_start:y_end, x_start:x_end]
    print(f'Region shape: {region.shape}')

    # Calculate threshold using generic mode
    print(f'\nRunning front_thresh in generic mode (wndw={wndw}, prcnt={prcnt})')
    import time
    t0 = time.time()
    thresh_generic = pyboa.front_thresh(region, wndw=wndw, prcnt=prcnt, mode='generic')
    t_generic = time.time() - t0
    print(f'  Completed in {t_generic:.2f} seconds')

    # Calculate threshold using vectorized mode
    print(f'\nRunning front_thresh in vectorized mode (wndw={wndw}, prcnt={prcnt})')
    t0 = time.time()
    thresh_vectorized = pyboa.front_thresh(region, wndw=wndw, prcnt=prcnt, mode='vectorized')
    t_vectorized = time.time() - t0
    print(f'  Completed in {t_vectorized:.2f} seconds')

    # Generate output filenames based on input
    base_name = fronts_file.replace('.npy', '')
    generic_outfile = f'{base_name}_thresh_generic_{size}x{size}_c{y_center}_{x_center}.npy'
    vectorized_outfile = f'{base_name}_thresh_vectorized_{size}x{size}_c{y_center}_{x_center}.npy'

    # Save outputs
    np.save(generic_outfile, thresh_generic)
    print(f'\nWrote {generic_outfile}')

    np.save(vectorized_outfile, thresh_vectorized)
    print(f'Wrote {vectorized_outfile}')

    # Compare generic vs vectorized
    print(f'\n=== Comparing generic vs vectorized ===')
    if np.array_equal(thresh_generic, thresh_vectorized):
        print('Results are identical')
    else:
        diff = np.sum(thresh_generic != thresh_vectorized)
        total = thresh_generic.size
        print(f'Results differ: {diff}/{total} pixels ({100*diff/total:.2f}%)')

    results = {
        'generic': (thresh_generic, t_generic),
        'vectorized': (thresh_vectorized, t_vectorized)
    }

    # Calculate threshold using dask mode
    if test_dask:
        try:
            print(f'\nRunning front_thresh in dask mode (wndw={wndw}, prcnt={prcnt}, n_workers={n_workers})')
            t0 = time.time()
            thresh_dask = pyboa.front_thresh(region, wndw=wndw, prcnt=prcnt,
                                            mode='dask', n_workers=n_workers,
                                            chunks='auto')
            t_dask = time.time() - t0
            print(f'  Completed in {t_dask:.2f} seconds')

            dask_outfile = f'{base_name}_thresh_dask_{size}x{size}_c{y_center}_{x_center}.npy'
            np.save(dask_outfile, thresh_dask)
            print(f'\nWrote {dask_outfile}')

            # Compare dask vs vectorized
            print(f'\n=== Comparing dask vs vectorized ===')
            if np.array_equal(thresh_dask, thresh_vectorized):
                print('Results are identical')
            else:
                diff = np.sum(thresh_dask != thresh_vectorized)
                total = thresh_dask.size
                print(f'Results differ: {diff}/{total} pixels ({100*diff/total:.2f}%)')

            results['dask'] = (thresh_dask, t_dask)

            # Print timing summary
            print(f'\n=== Timing Summary ===')
            print(f'Generic:    {t_generic:.2f}s')
            print(f'Vectorized: {t_vectorized:.2f}s (speedup: {t_generic/t_vectorized:.2f}x)')
            print(f'Dask:       {t_dask:.2f}s (speedup: {t_generic/t_dask:.2f}x)')

        except ImportError as e:
            print(f'\nSkipping dask mode: {e}')
        except Exception as e:
            print(f'\nError in dask mode: {e}')

    return results


if __name__ == '__main__':
    # Entire
    #test_whole_one('/home/xavier/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_divb2.nc',
    #         '/home/xavier/Oceanography/data/OGCM/LLC/Fronts/global/LLC4320_2012-11-09T12_00_00_fronts.npy')

    # Test threshold modes on 1000x1000 region (including dask parallel mode)
    test_threshold_modes('data/LLC4320_2012-11-09T12_00_00_fronts.npy',
                        center=(4000, 4000),
                        size=1000,
                        wndw=64,
                        prcnt=90,
                        test_dask=False,
                        n_workers=None)  # None = use all available cores