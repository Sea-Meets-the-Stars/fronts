# Sharpening the Fronts

## Goals

When defining fronts on the LLC4320 model data, the process of (1) thresholding and (2) thinning can result in front locations that are offset from the peak in the gradb2 field used to define the fronts.  This can be a problem because the front locations are used to define the properties of the fronts.  The goal is to sharpen the fronts so that the front locations are as close as possible to the peak in the gradb2 field.

This set of prompts tried to identify a local solution, i.e. front by front.

## Code

Examine the modules in fronts/properties/ to understand the code that is available to you.

## Algorithm

Here are required aspects of the algorithm:

- It will operate on the individual fronts, aka cutouts of the labeled_fronts defined by the label_fronts() function in the fronts/properties/group_labels.py file.
- It needs to be able to run in parallel on the fronts.
- The modified front should be contiguous, i.e. there should be no gaps in the front 

### Overleaf

Place any Latex files in /home/xavier/Projects/overleaf/Front_properties/

The planning file for this activity is claude_sharpen_planning.tex.  Make it a standlone LateX file.

Output PNG figures to the Overleaf project folder.

You may push to git as you work.  The access token is in my .bashrc profile with the name OVERLEAF.

## Development phase

Here are some requirements:

- Place any code in the existing dev/sharpen/py directory.
- Place any data in the existing dev/sharpen/data directory.

### Test Data

Generate code to create the test files in the fronts/dev/sharpen/data directory.  The gradb2 field is in the $OS_OGCM/LLC/Fronts/derived/LLC4320_2012-11-09T12_00_00_gradb2_v1.nc

Put the code in the dev/sharpen/py/create_test_data.py file.

Pull the data from the gradb2 file for the following coordinates:

- cols=11841,11980
- rows=8611,8750

Use the file labeled_fronts_global_20121109T12_00_00_v1_bin_B.npy in $OS_OGCM/LLC/Fronts/group_fronts/v1/ for the labeled fronts.

The data files are in the fronts/dev/sharpen/data directory.

- labeled_fronts.npy
- gradb2.npy

Add a method to create_test_data.py that plots the labeled fronts on the gradb2 field.

### Developing Approach 4

Refer to the planning doc for details of the algorithm.

1. Please develop the code for Approach 4 in the dev/sharpen/py/sharpen_fronts.py file.
2. Please test the code with the test files in the dev/sharpen/data directory.
3. Please save the output to the dev/sharpen/data directory.
4. Use Python only
5. Include inline comments in the code to explain the logic.

Include a method that plots the original fronts (dotted) and the sharpened fronts (solid) on the gradb2 field.

#### Modifications

1. Try dilating the skeleton fronts by 3 pixels before sharpening.
2. Apply the skimage.morphology.skeletonize() method to the final sharpened fronts at the very end.  This can be done on the entire array at once.

## Prompts

### Brainstorming

1. Read this file and brainstorm several approaches to sharpening the fronts. Present these ideas in the claude_sharpen_planning.tex file.
2. Please expand on Approach 4 in the planning file.  It should be a complete algorithm that can be implemented in code.  Ignore the shortest-path skeleton idea.  Modify that section of the LateX file to include the details of the algorithm.
3. Is there an approach that would use the medial_axis function in the skimage library to sharpen the fronts?  If so, please describe it in the planning file.

### Prepping

1. Re-read this file.  Create the gradb2.npy file in the dev/sharpen/data directory as described in the Test Data section.
2. Re-read this file.  Create the labeled_fronts.npy file in the dev/sharpen/data directory as described in the Test Data section.  Also, generate the code for the desrired plot.

### Approach 4

1. Create the code for Approach 4 in the dev/sharpen/py/sharpen_fronts.py file, as described in the Developing Approach 4 section.
2. Re-read the Developing Approach 4 section.  Modify the code to include the plot of the original and sharpened fronts on the gradb2 field.
3. Re-read the Developing Approach 4 section.  Apply the first modification to the code.
4. Re-read the Developing Approach 4 section.  Apply the second modification to the code.