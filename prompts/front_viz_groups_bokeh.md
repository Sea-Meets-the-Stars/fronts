# Front Visualization with Bokeh

## I wish to visualize the front properties in the groups table using Bokeh.

## Here are the functionality that I wish to include:

- Load a portion of the globe, either by x,y or lat,lon bounds.
- When we mouse over a front, we should see the front properties in a tooltip.
- Because the table includes many front properties, we should be able to select which properties to display.  Use a dropdown menu to select the properties.
- The default set of properties should be:
  - flabel
  - gradb2_median
  - strain_mag_median / coriolis_f_median
- The user will input the field to display when launching the script
- The user can toggle between Greys colormap and a divergent colormap centered on zero (e.g. seismic).
- If the user selects a divergent colormap, the fronts should be displayed in yellow (not red).

## You should examine the fronts/scritps/front_property_viewer.py file for code on loading the data.

## The script should be called front_viz_groups_bokeh.py and be placed in the fronts/scripts directory.

# Prompts

1. Please provide a plan for creating the Bokeh script as described in prompts/front_viz_groups_bokeh.md

2. Please review this file for updates on functionality and modify your plan accordingly.

3. Please implement the script.