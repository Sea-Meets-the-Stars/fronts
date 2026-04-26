# Figures for TandS paper

## Goals

Generate first draft figures which may go into the TandS paper.

# Code

Here are guidelines for writing code: 

- Use Python
- If you need to run Python code, use the "ocean14" environment of conda.
- Add inline comments to explain the effort
- Reuse existing code when possible, throughout this fronts repository
- Use exisisting I/O methods when possible, throughout this fronts repository.  These are in a number of io.py modules.
- Use methods, not classes
- Use matplotlib
- Use cartopy for geographic plotting
- Place first draft figures in the Figures/py/figs_tands.py module
- Place figures that will go in the paper in Figures/py/figs_tands_paper.py module
- Use /home/xavier/Oceanography/python/nenya/papers/InformationContent/Figures/py/figs_nenya_dim.py as guidance on how to write code for figures.
- Write PNG figures to the Figures/ directory with dpi=300 

# Data files

The analysis will use v2 outputs, unless otherwise specified.
The key data files are:

1. Global derived files for single timestamps are in: $OS_OGCM/LLC/Fronts/derived/ with extension _v2.nc
2. The binary front files for single timestamps are in: $OS_OGCM/LLC/Fronts/outputs/.  Use "v2_bin_D" unless otherwise specified.
3. The parquet tables of fronts and their properties are in: $OS_OGCM/LLC/Fronts/group_fronts/.  Use folder v2/ unless otherwise specified and the v2_bin_D files.  

# Figures

1. Turner angle vs. gradb

For the first figure, let's generate a 2D histogram of Turner angle vs. gradb for the individual fronts in the "2012-11-09T12_00_00" timestamp.  Here are specifications: 

- Use the v2_bin_D binary front files
- Use the mean of the Turner angle
- Use the median of the sqrt(gradb2)
- Plot gradb on a log scale
- Plot Turner angle on a linear scale from -90 to 90 degrees
- Use Blues for the color map
- Include a labeled color bar
- Use a variable number of bins for the histogram; start with 50

# Prompts

## Generate

1. Read this file and generate the first figure under Figures.  

## Modifications
