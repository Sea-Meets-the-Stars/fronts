# Generate global viz of front-weighted properties

## Goals

We wish to generate code that creates a global viz of front-weighted properties, e.g. relative vorticity, divergence, strain, Turner angle etc.
It should be sufficiently generic to be used for any property.  It should also allow the view for a single timestamp or multiple timestamps.

We will follow the example provided in fronts/properties/nb
/Turner_Angle_Global_Viz.ipynb, particularly Section 5. Turner Angle on Co-located Fronts.

## Code

Here are guidelines for the code: 

- Use Python
- When possible use existing methods from the modules in fronts/properties/ and fronts/viz
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Place I/O methods in the fronts/properties/io.py module.
- Place import statements at the top of the file.
- Include a description of inputs/outputs in the doc string of all methods

## Testing

If you need to test the code, you can use the files in $OS_OGCM/LLC/Fronts as test data. You should run on the "ocean14" conda environment.

## Development

1. Develop a plan for the code in the file dev/properties/properties_viz_plan.md

2. Modify the plan to make Module 1 modular.  That is have the individual items generated be separate methods where sensible.

3. Proceed to generate the code according to the plan in 
dev/properties/properties_viz_plan.md

4. Generate a script in dev/properties/properties_viz_test.py that tests the code.  You can use the files in $OS_OGCM/LLC/Fronts as test data.

## Modifications

1. Make these changes to the front_property_viewer.py module and other related code:

- Move the viz_utils.py module to fronts/viz/ and refactor other modules as needed 
- Allow the user to set the display levels (vmin, vmax) for any given field by setting --vl0 vmin,vmax --vl1 vmin,vmax, etc.
- Allow the user to choose different colormaps (blue, green, red) for a given field with, e.g. -cl0 blue

1. Make these changes to the front_property_viewer.py module and other related code:

- Make all of the fonts black and/or bold (i.e. easier to read)


## Docs

## Prompts

### Develop

1. Read this doc.  Now execute the first step listed under Development.
2. Read this doc.  Now execute the 2nd step listed under Development.
3. Read this doc.  Now execute the 3rd step listed under Development.
4. Read this doc.  Now execute step 4 listed under Development.
5. Read this doc.  Implement the 1st set of modifications described in the Modifications section