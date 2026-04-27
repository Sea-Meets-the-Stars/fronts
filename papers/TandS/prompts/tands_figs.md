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

### Modifications

- Label the x-axis from -pi/2 to pi/2

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
- Use lighter colors in panels (a) and (b) and higher opacity
- Modify the gradb normalization to be linear instead of log.  Make this optional.

## (3) T, S, rho and gradb

Generate a figure that has 4 panels and shows the temperature (Theta), salinity (Salt), density and gradb in an ~150km x 150km region.  Choose your the same lat, lon from Figure (2) but make it an option.  Here are more specifications:

- Use the derived/ files
- Use separate color maps for SST, Salinity and density.  Make their choices a dict in a separate module in Figures/py/defs.py.  Refactor Figure (2) to use the same color maps.
- For density, use the jmd95_xgcm_implementation.py module in llc4320-native-grid-preprocessing/src/dbof/utils/ to compute density from Salt and Theta.
- Plot gradb on a log scale
- Include color bars for each panel
- Label each panel
- Plot with lat, lon
- Use a 2x2 grid

### Modifications

- Move the defs.py module to fronts/viz/defs.py
- Modify the gradb colorbar to have less contrast (i.e. extend it)
- Show density offset from 1025 kg/m^3
- Use a different color map for density

## (4) Joint PDFs

Generate a figure that shows a series of Joint PDFs (2D histograms) of the front properties vs gradb.  Show these properties: strain_mag normalized by coriolis, divergence normalized by coriolis, relative vorticity normalized by coriolis, and frontogenesis_tendency.

Here are additional specifications:

- Use a 2x2 grid
- Use a different color map for each property (use fronts/viz/defs.py)
- Include a color bar for each panel
- Label each panel
- Plot with lat, lon

### Modifications

- Put gradb on the x-axis
- Show gradb to 1e-5 s^-2

## (5) Explore thermal vs. salinity fronts

Compare the normalized PDFs of strain_mag/f values for fronts with gradb > 1e-7 (make this value optional) and for thermal (turner angle > 45 degrees; also optional) and salinity (turner angle < -45 degrees) fronts.

Here are some additional specifications:

- Show each PDF on the same axis
- Use a different color for each PDF
- Label each PDF

### Modifications

- Show strain_mag/f on a log-scale (optional)
- Add divergence/f, relative vorticity/f, and frontogensis on a 2x2 grid


# Prompts

## Generate

1. Read this file and generate the first figure under Figures.  
2. Read this file and generate the second figure under Figures.
3. Read this file and generate Figure (3) under Figures.
4. Read this file and generate Figure (4) under Figures.
5. Read this file and generate Figure (5) under Figures.

## Modifications

1. Re-read this file.  Perform the modifications described in the "Modifications" section of Figure (2).
2. Re-read this file.  Perform the modifications described in the "Modifications" section of Figure (3).
3. Re-read this file.  Perform the modifications described in the "Modifications" section of Figure (1).
4. Re-read this file.  Perform the modifications described in the "Modifications" section of Figure (4).
5. Re-read this file.  Perform the modifications described in the "Modifications" section of Figure (5).
6. Re-read this file.  Perform the new modification described in the "Modifications" section of Figure (2).
