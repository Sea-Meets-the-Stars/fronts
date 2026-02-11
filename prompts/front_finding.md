# Global

## Using the code in the wrangler Repository code in the ogcm/ and preproc/ directories, generate a script in fronts/dev/ named global_divb2.py that will:

1. Load the LLC4320 model data given the path provided
2. Compute the squareddivergence of the buoyancy field
3. Save the global divergence to a NetCDF file

### The script should be able to be run from the command line with only the filename as an argument.  if you need to use Python, make sure you are on the "ocean14" environment of conda

### Please have the methods in global_divb2.py use methods in the wrangler repository.  Make sure to use the methods in the ogcm/ and preproc/ directories, and import them as needed.

# Thresholding

## Please add a method to global_fronts.py in fronts/dev/ that will calculate the threshold array using fronts.finding.pyboa.front_thresh() on a 1000x1000 pixel set of the LLC4320_2012-11-09T12_00_00_fronts.npy file in fronts/dev/data/ centered at 4000,4000.  Use both the generic and vectorized modes of the algorithm and save the output to separate .npy files.

## Thanks!  Now add a new mode to front_thresh() which performs the same calculation but using dask to parallelize.

## When testing, I get this error: "Error in dask mode: could not convert string to float: np.str_('constant')".  Please help.  If you need to run python on global_fronts.py, be sure to be on the "ocean14" environment of conda.

## Ok, now modify global_fronts.py to test a fourth option:  mode == 'pool'.

# Thinning

## Ok, now add a method to global_fronts.py that loads in the Pool output file from thresholding and runs the skimage.morphology.thin() method on it.  Please benchmark the time spent perfoming that calculation.

# Cropping

## Thanks.  Now add a method to test the timing of pyboa.cropping().  Run it on the output from thinning.  You do not need to run the rest of the tests.