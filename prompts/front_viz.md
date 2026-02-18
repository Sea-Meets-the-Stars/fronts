# Front viz tools

# Global divb2 viewer

## Please generate a simple PyQt6 + pyqtgraph GUI based on the global_divb2_map.py file in fronts/dev/ which reads in a NetCDF file of divb2 data and a .npy file of fronts (1=front, 0=no front).  The top panel will show a downsampled version of the global field and one should be able to pan and zoom.  The divb2 data will be a grayscale and the fronts will be red dots.

## The code crashed trying to load the gray color map: "/home/xavier/miniforge3/envs/ocean14/lib/python3.14/site-packages/pyqtgraph/colors/maps/gray".  Please fix.  If you need to use Python, use the "ocean14" environment of conda.  Here is the command I am using:  python global_divb2_viewer.py LLC4320_2012-11-09T12_00_00_divb2.nc LLC4320_2012-11-09T12_00_00_fronts.npy --downsample 4


## Please have higher divb2 values be darker (inverted color bar).  Also, when I zoom in, the red dots are too small.  Maybe we should overlay an image that is transparent where the front value is 0 and otherwise red for that grid cell, but with a low alpha value.

## Can you have an NaN in the divb2 values be colored green?  Can you add a button that allows the user to reset the vmin,vmax based on the current Zoom view?  And another button to reset.

## Instead of changing the values of plot_data, how about overlaying a mask, similar to what we have done for front pixels?


## Ok, please add a color bar for the log10 divb2 values.  And reduce the opacity of the red (front) map.

## I don't see the colorbar in the GUI.  Also, can we use a darker green for land?

## Thanks.  Please rename Adjust Contrast to Adjust Limits.  And when one uses that button, the limits on the colorbar should update.

## Can you add a box that toggles between log10 divb2 and linear?

## In the script fronts.scripts/global_divb2_viewer.py, allow for a second fronts file to be loaded and displayed.  This should be a separate checkbox and use blue instead of red for the fronts.

## Please remove the load buttons and default the scaling to be linear.  Also, can we maintain the current zoom level when the log/linear toggle is changed?

## Please modify the GUI to show the x,y values for the corners of the current view.