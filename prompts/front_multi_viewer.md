# Property viewer

## Here are the specs for a GUI to view multiple properties of fronts in a specified region of the globe.  It should be called front_property_viewer.py.

### The region will be specified by a lat,lon or x,y (pixel) bounding box.  If lat,lon is specified, the x,y will be calculated. 

### The user will input the timestamp of the data to view.  The version number will be optional and default to 1.

### There will be four panels in a 2x2 grid.  One will show a map of the region with gradb2 with the fronts overlaid.  See the code in fronts/fronts/scripts/global_field_viewer.py for an example.

### The binary fronts mask will be specified by a config label (A, B, C, etc.).  The code will use the fronts/fronts/finding/io.py module to load them.

### The other 3 will show properties of the fronts, specified on the command line by the user (as fields).  The code will use the fronts/fronts/llc/io.py module to load the derived properties.  The fronts will be overlaid on each.

### Each panel will have a title and color bar.

### The user will be able to zoom in and pan in any of the panels.  If this is done, the other panels should be updated to show the same region.

### Unlike the global_field_viewer.py code, do not implement downsampling.

### Where possible, re-use code from the global_field_viewer.py code.  That can be put into a new module called viz_utils.py

### Each panel should have the same size and aspect ratio.

### Default okubo_weiss to use the divergent colormap centered on zero.  And make the fronts grey when using the divergent colormap.

# Prompts

## Please read the file fronts/prompts/front_multi_viewer.md and generate a plan to implement the GUI.

## I have updated the md file.  Please re-review and modify your plan.

## That looks great.  Please implement the GUI.

## I have modified the code so that it can be called from the fronts_property_viewer script I added to bin/.  When I run the code with -h, it crashes because the bbox_group is required.  Please fix this.  If you need to run Python, use the "ocean14" conda environment.

## I am running the script with the following command:

```
fronts_property_viewer 2012-11-09T12_00_00 --fields okubo_weiss strain_mag rossby_number --bbox 10000 10200 8000 8200
```

## It crashes with the following error:

```
  File "/home/xavier/miniforge3/envs/ocean14/lib/python3.14/site-packages/pyqtgraph/graphicsItems/ImageItem.py", line 731, in quickMinMax
    return self._xp.nanmin(data), self._xp.nanmax(data)
           ~~~~~~~~~~~~~~~^^^^^^
  File "/home/xavier/miniforge3/envs/ocean14/lib/python3.14/site-packages/numpy/lib/_nanfunctions_impl.py", line 356, in nanmin
    res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
ValueError: zero-size array to reduction operation fmin which has no identity
```

## Please fix this.  If you need to run Python, use the "ocean14" conda environment.

## Can you figure out why the two panels on the right show larger views?  They are supposed to be the same size and aspect ratio.