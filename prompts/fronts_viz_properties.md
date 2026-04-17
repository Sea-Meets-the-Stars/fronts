# Generate global viz of front-weighted properties

## Goals

We wish to generate code that creates a global viz of front-weighted properties, e.g. relative vorticity, divergence, strain, Turner angle etc.
It should be sufficiently generic to be used for any property.  It should also allow the view for a single timestamp or multiple timestamps.

We will follow the example provided in fronts/properties/nb
/Turner_Angle_Global_Viz.ipynb, particularly Section 5. Turner Angle on Co-located Fronts.

## Code

Here are guidelines for the code: 

- Use Python
- When possible use existing methods from the modules in fronts/properties/
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Generate viz methods in the fronts/properties/viz.py module.
- Place I/O methods in the fronts/properties/io.py module.

## Development

1. Develop a plan for the code in the file dev/properties/properties_viz_plan.md

2. Modify the plan to make Module 1 modular.  That is have the individual items generated be separate methods where sensible.

3. Proceed to generate the code according to the plan in 
dev/properties/properties_viz_plan.md

## Prompts

### Develop

1. Read this doc.  Now execute the first step listed under Development.
2. Read this doc.  Now execute the 2nd step listed under Development.
3. Read this doc.  Now execute the 3rd step listed under Development.