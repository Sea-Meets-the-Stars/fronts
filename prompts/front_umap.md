# UMAP Analysis

## I wish to perform a UMAP analysis on the front properties that are provided in a parquet file.  The table columns that I wish to include in the analysis are:

- npix
- strain_mag_median / coriolis_f_median
- divergence_median / coriolis_f_median
- log10(gradb2_median) 
- frontogenesis_tendency_median

## Do normalize each parameter to zero-mean and unit variance.

## Save the UMAP analysis to a new parquet file, pivoting on flabel from the original parquet file.

## Begin by creating a new Notebook in fronts/dev/groups named Front_umap.ipynb.  We will eventualy generate a new module for reproducibility.

## Do include a few cells in the notebook to visualize the UMAP analysis.

## Prompts

## Please provide a plan for the analysis described in prompts/front_umap.md

## I have upddated the md file.  Please re-review and modify your plan.