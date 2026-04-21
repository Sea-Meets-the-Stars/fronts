# Sharpening the Fronts Global

## Goals

When defining fronts on the LLC4320 model data, the process of (1) thresholding and (2) thinning can result in front locations that are offset from the peak in the gradb2 field used to define the fronts.  This can be a problem because the front locations are used to define the properties of the fronts.  The goal is to sharpen the fronts so that the front locations are as close as possible to the peak in the gradb2 field.

With this set of prompts we wish to find a global solution.

## Code

### Review

Examine the modules in fronts/finding/ to understand our previous approach to front finding.  Also review the module sharpen_fronts.py in dev/sharpen/py to see efforts to work on individual fronts.

### Writing

Here are guidelines for writing code: 

- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes

## Algorithm

Here are required aspects of the algorithm:

- It will operate on the binary output array from pyboa.front_thresh()
- It needs to run fast on large images

## Overleaf

Place any Latex files in /home/xavier/Projects/overleaf/Front_properties/

The planning file for this activity is claude_sharpen_global_plan.tex.  Make it a standlone LateX file.

Output PNG figures to the Overleaf project folder.

You may push to git as you work.  The access token is in my .bashrc profile with the name OVERLEAF.

## Development phase

Here are some requirements:

- Place any code in the existing dev/sharpen/py directory.
- Place any data in the existing dev/sharpen/data directory.

Use the test file data/gradb2_global.npy to operate upon.

When possible, use existing methods from dev/py/sharpen_fronts.py

Include code that generates plots to show the derived fronts

## Prompts

### Brainstorming

1. Read this file and brainstorm several approaches to sharpening the fronts in full images working on only the binary threshold outputs. Present these ideas in the claude_sharpen_global_plan.tex file.  Review the ideas in claude_sharpen_planning.tex file.  

### Development

1. Begin by generating code for the algorithm G1 described in claude_sharpen_global_plan.tex.  Generate a new module named dev/py/sharpen_fronts_global.py