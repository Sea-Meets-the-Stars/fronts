# Sharpening the Fronts

## Goals

When defining fronts on the LLC4320 model data, the process of (1) thresholding and (2) thinning can result in front locations that are offset from the peak in the gradb2 field used to define the fronts.  This can be a problem because the front locations are used to define the properties of the fronts.  The goal is to sharpen the fronts so that the front locations are as close as possible to the peak in the gradb2 field.

## Algorithm

Here are required aspects of the algorithm:

- It will operate on the individual fronts, aka labeled_fronts defined by the label_fronts() function in the fronts/properties/group_labels.py file.
- It needs to be able to run in parallel on the fronts.
- The modified front should be contiguous, i.e. there should be no gaps in the front 

## Prompts

1. 