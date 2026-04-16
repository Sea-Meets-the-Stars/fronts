# Deal with spurs in the fronts

## Goals

It is common for the fronts defined by our algorithm to contain one or more "spurs", short extensions off the front.  These are typically at the ends of the front but occasionally along it.  We wish to derive an algorithm to remove these while keeping the remainder of the front intact.

In dev/spurs/pmc_spurs.m, we have a MATLAB function that will remove spurs from a binary image.  It works reasonably well but is not perfect.  We wish to improve upon it. and port it to Python.

### Review

## Code

### Writing

Here are guidelines for writing code: 

- Use Python
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes

## Algorithm

Here are required aspects of the algorithm:

- It will operate on a binary output array from pyboa.front_thresh()
- It needs to run fast on large images

## Overleaf

Place any Latex files in /home/xavier/Projects/overleaf/Front_properties/

The planning file for this activity is claude_spurs_plan.tex.  Make it a standlone LateX file.

You may push to git as you work.  The access token is in my .bashrc profile with the name OVERLEAF.

## Development phase

Here are some requirements:

- Place any code in the existing dev/spurs/py directory.
- Place any data in the existing dev/spurs/data directory.
- Place any figures in the existing dev/spurs/data directory.

Use the test file ../data/sharpened_global_g1.npy to operate upon.

Include code that generates plots to show the derived fronts

If you need to run python use the "ocean14" environment of conda.

## Prompts

### Brainstorming

1. Read this file and brainstorm several approaches to removing spurs from the binary image.

### Development

1. Begin by generating code for the algorithm S1 described in claude_spurs_plan.tex.  Generate a new module named dev/spurs/py/matlab_port.py.  Generate tests that runs on the test file and generates figures.  Add your findings to the claude_spurs_plan.tex file.  Include a copy of the figures

2. Now generate code for the skan S5 approach described in claude_spurs_plan.tex.  Generate a new module named dev/spurs/py/spurs_skan.py.  Generate tests that runs on the test file and generates figures.  Compare with the MATLAB port approach.  Add your findings to the claude_spurs_plan.tex file, including a copy of the figures.