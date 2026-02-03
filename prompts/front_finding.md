# Global

## Using the code in the wrangler Repository code in the ogcm/ and preproc/ directories, generate a script in fronts/dev/ named global_divb2.py that will:

1. Load the LLC4320 model data given the path provided
2. Compute the squareddivergence of the buoyancy field
3. Save the global divergence to a NetCDF file

### The script should be able to be run from the command line with only the filename as an argument.  if you need to use Python, make sure you are on the "ocean14" environment of conda

### Please have the methods in global_divb2.py use methods in the wrangler repository.  Make sure to use the methods in the ogcm/ and preproc/ directories, and import them as needed.