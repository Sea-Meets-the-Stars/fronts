# Puttering about applying our algorithm to a global dataset

import numpy as np
import xarray
from fronts.finding import algorithms

def test_one(divb2_file:str, outfile:str):

    # Load
    divb2 = xarray.open_dataset(divb2_file)['Divb2'].values

    # Find fronts
    fronts = algorithms.fronts_from_divb2(divb2)

    # Save
    np.save(outfile, fronts.astype(np.int16))
    print(f'Wrote {outfile}')


if __name__ == '__main__':
    test_one('/home/xavier/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_divb2.nc',
             '/home/xavier/Oceanography/data/OGCM/LLC/Fronts/data/LLC4320_2012-11-09T12_00_00_fronts.npy')