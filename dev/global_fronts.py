# Puttering about applying our algorithm to a global dataset

import os
import time
import numpy as np
import xarray
from fronts.finding import algorithms
from fronts.finding import pyboa

from IPython import embed

def test_whole_one(divb2_file:str, outfile:str):

    # Load
    divb2 = xarray.open_dataset(divb2_file)['Divb2'].values

    # Find fronts
    fronts = algorithms.fronts_from_divb2(divb2, thin=True,
                                          verbose=True)

    # Save
    np.save(outfile, fronts.astype(np.int16))
    print(f'Wrote {outfile}')


def test_threshold_modes(divb2_file:str, center:tuple=(4000, 4000),
                         size:int=1000, wndw:int=64, prcnt:float=90,
                         load_out:bool=False,
                         test_dask:bool=True, test_pool:bool=True,
                         n_workers=None):
    """
    Test front_thresh() function in generic, vectorized, dask, and pool modes.

    Args:
        fronts_file (str): Path to the .npy file containing fronts data
        center (tuple): Center coordinates (y, x) for extraction
        size (int): Size of the square region to extract
        wndw (int): Window size for front_thresh
        prcnt (float): Percentile for thresholding
        test_dask (bool): Whether to test dask mode
        test_pool (bool): Whether to test pool mode
        load_out (bool, optional): Load pre-cooked output, if it exists
        n_workers (int): Number of workers for parallel modes
    """
    # Load the fronts data
    print(f'Loading {divb2_file}')
    divb2 = xarray.open_dataset(divb2_file)['Divb2'].values

    # Extract 1000x1000 region centered at the specified location
    half_size = size // 2
    y_center, x_center = center
    y_start = y_center - half_size
    y_end = y_center + half_size
    x_start = x_center - half_size
    x_end = x_center + half_size

    print(f'Extracting region [{y_start}:{y_end}, {x_start}:{x_end}]')
    region = divb2[y_start:y_end, x_start:x_end]
    print(f'Region shape: {region.shape}')

    # Generate output filenames based on input
    base_name = divb2_file.replace('.npy', '')
    generic_outfile = f'{base_name}_thresh_generic_{size}x{size}_c{y_center}_{x_center}.npy'
    vectorized_outfile = f'{base_name}_thresh_vectorized_{size}x{size}_c{y_center}_{x_center}.npy'

    # Calculate threshold using generic mode
    if load_out and os.path.isfile(generic_outfile):
        thresh_generic = np.load(generic_outfile)
        t_generic = 0.
    else:
        print(f'\nRunning front_thresh in generic mode (wndw={wndw}, prcnt={prcnt})')
        t0 = time.time()
        thresh_generic = pyboa.front_thresh(region, wndw=wndw, prcnt=prcnt, mode='generic')
        t_generic = time.time() - t0
        print(f'  Completed in {t_generic:.2f} seconds')


    # Calculate threshold using vectorized mode
    if load_out and os.path.isfile(vectorized_outfile):
        thresh_vectorized = np.load(vectorized_outfile)
        t_vectorized = 0.
    else:
        print(f'\nRunning front_thresh in vectorized mode (wndw={wndw}, prcnt={prcnt})')
        t0 = time.time()
        thresh_vectorized = pyboa.front_thresh(region, wndw=wndw, prcnt=prcnt, mode='vectorized')
        t_vectorized = time.time() - t0
        print(f'  Completed in {t_vectorized:.2f} seconds')



    # Save outputs
    if not load_out or not os.path.isfile(generic_outfile):
        np.save(generic_outfile, thresh_generic)
        print(f'\nWrote {generic_outfile}')

    if not load_out or not os.path.isfile(vectorized_outfile):
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

        except ImportError as e:
            print(f'\nSkipping dask mode: {e}')
        except Exception as e:
            print(f'\nError in dask mode: {e}')

    # Calculate threshold using pool mode
    if test_pool:
        try:
            pool_workers = n_workers if n_workers is not None else os.cpu_count()
            print(f'\nRunning front_thresh in pool mode (wndw={wndw}, prcnt={prcnt}, n_workers={pool_workers})')
            t0 = time.time()
            thresh_pool = pyboa.front_thresh(region, wndw=wndw, prcnt=prcnt,
                                            mode='pool', n_workers=pool_workers,
                                            chunks='auto')
            t_pool = time.time() - t0
            print(f'  Completed in {t_pool:.2f} seconds')

            pool_outfile = f'{base_name}_thresh_pool_{size}x{size}_c{y_center}_{x_center}.npy'
            np.save(pool_outfile, thresh_pool)
            print(f'\nWrote {pool_outfile}')

            # Compare pool vs vectorized
            print(f'\n=== Comparing pool vs vectorized ===')
            if np.array_equal(thresh_pool, thresh_vectorized):
                print('Results are identical')
            else:
                diff = np.sum(thresh_pool != thresh_vectorized)
                total = thresh_pool.size
                print(f'Results differ: {diff}/{total} pixels ({100*diff/total:.2f}%)')

            results['pool'] = (thresh_pool, t_pool)

        except ImportError as e:
            print(f'\nSkipping pool mode: {e}')
        except Exception as e:
            print(f'\nError in pool mode: {e}')

    # Print timing summary
    print(f'\n=== Timing Summary ===')
    if t_generic > 0:
        print(f'Generic:    {t_generic:.2f}s')
    else:
        print(f'Generic:    (loaded from file)')

    if t_vectorized > 0:
        print(f'Vectorized: {t_vectorized:.2f}s', end='')
        if t_generic > 0:
            print(f' (speedup: {t_generic/t_vectorized:.2f}x)')
        else:
            print()
    else:
        print(f'Vectorized: (loaded from file)')

    if 'dask' in results:
        t_dask = results['dask'][1]
        print(f'Dask:       {t_dask:.2f}s', end='')
        if t_generic > 0:
            print(f' (speedup: {t_generic/t_dask:.2f}x)')
        elif t_vectorized > 0:
            print(f' (speedup: {t_vectorized/t_dask:.2f}x)')
        else:
            print()

    if 'pool' in results:
        t_pool = results['pool'][1]
        print(f'Pool:       {t_pool:.2f}s', end='')
        if t_generic > 0:
            print(f' (speedup: {t_generic/t_pool:.2f}x)')
        elif t_vectorized > 0:
            print(f' (speedup: {t_vectorized/t_pool:.2f}x)')
        else:
            print()

    return results

if __name__ == '__main__':
    # Entire
    #test_whole_one('/home/xavier/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_divb2.nc',
    #         '/home/xavier/Oceanography/data/OGCM/LLC/Fronts/global/LLC4320_2012-11-09T12_00_00_fronts.npy')

    # Test threshold modes on 1000x1000 region (including parallel modes)
    test_threshold_modes('data/LLC4320_2012-11-09T12_00_00_divb2.nc',
                        center=(4000, 4000),
                        size=1000,
                        load_out=False,  # Set to False to run all modes and get timing
                        wndw=64,
                        prcnt=90,
                        test_dask=True,
                        test_pool=True,
                        n_workers=4)  # None = use all available cores