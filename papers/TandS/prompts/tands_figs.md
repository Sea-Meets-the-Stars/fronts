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
4. Coordinates are in $OS_OGCM/LLC/Fronts/coords/LLC_coords_lat_lon.nc

# Figures

## (1) Turner angle vs. gradb

For the first figure, let's generate a 2D histogram of Turner angle vs. gradb for the individual fronts in the "2012-11-09T12_00_00" timestamp.  Here are specifications:

- Use the v2_bin_D binary front files
- Use the mean of the Turner angle
- Use the median of the sqrt(gradb2)
- Plot gradb on a log scale
- Plot Turner angle on a linear scale from -90 to 90 degrees
- Use Blues for the color map
- Include a labeled color bar
- Use a variable number of bins for the histogram; start with 50

## (2) Front definition example

Generate a figure that describes how the fronts are defined and their properties measured.  It should run steps in fronts.finding.algorithms.fronts_from_gradb2() as needed.

Use the finding_config_D.yaml file in fronts/config/configs/ for the front definition parameters.  Choose a region of ~150km x 150km on the ocean.  Make the lat, lon a parameter.  Find a way to rapidly load the coordinates of a sub-region.

It should be four panels:

- (1) gradb2 in greyscale with pixels exceeding the threshold in red with alpha=0.3 opacity
- (2) gradb2 with the thinned fronts shown in green with alpha=0.5 opacity
- (3) The final fronts should each be a different color and labeled with the front ID number
- (4) Indicate the dilate region in gray for each front (opacity=0.4) used when measuring the front properties.  Overlay on the divergence field.

Here are some additional figure specifications:

- Use a 2x2 grid
- Indicate the lat, lon of the region on each panel

### Modifications

- Load the finding_config_D.yaml file in fronts/config/configs/ for the front definition parameters.  Make the file optional
- Use the io.py module in fronts/config/ for loading the YAML file
- Have the gradb2 field be dark where the values are large
- Have the derived field ("divergence") be an option
- Add color bars for panels (b) and (d)

## (3) T, S, rho and gradb

Generate a figure that has 4 panels and shows the temperature (Theta), salinity (Salt), density and gradb in an ~150km x 150km region.  Choose your own lat, lon but make it an option.  Here are specifications:

- Use the derived/ files
- Use separate color maps for SST, Salinity and density.  Make their choices a dict in a separate module in Figures/py/defs.py
- Plot gradb on a log scale
- Include color bars for each panel
- Label each panel
- Plot with lat, lon;  find a way to rapidly load a portion of the coords
- Use a 2x2 grid

# Prompts

## Generate

1. Read this file and generate the first figure under Figures.  
2. Read this file and generate the second figure under Figures.

## Modifications

1. Re-read this file.  Perform the modifications described in the "Modifications" section of Figure (2).
