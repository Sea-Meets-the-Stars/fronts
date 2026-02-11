# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**fronts** is a Python package for ocean front detection and analysis, developed by J. Xavier Prochaska and P. Cornillon. It provides methods for identifying thermal and salinity fronts in satellite and model data, with support for the LLC4320 high-resolution ocean model.

## Installation and Setup

```bash
pip install -e .
```

Key dependencies: torch, xarray, scikit-image, scipy, h5netcdf, zarr, xmitgcm, gsw

## Running Tests

```bash
pytest fronts/tests/
pytest fronts/tests/test_dbof_utils.py  # single test file
pytest fronts/tests/test_dbof_utils.py::test_find_entry  # single test
```

## Architecture

### Core Modules

- **`fronts/finding/`** - Front detection algorithms
  - `algorithms.py` - High-level front detection from divergence fields (`fronts_from_divb2`)
  - `pyboa.py` - PyBOA (Python Buoyancy Ocean Analysis) implementation with xarray accessor for Sobel gradients, thresholding, morphological thinning
  - `thin_cc.py`, `cc_sst_preproc.py` - Cornillon-style SST preprocessing and thinning (converted from FORTRAN)

- **`fronts/dbof/`** - Deep Bag of Features (DBoF) data management
  - `defs.py` - Data model definitions for oceanographic fields (SST, SSH, vorticity, divergence, Divb2, etc.)
  - `utils.py` - Utilities for finding entries and grabbing field data
  - `io.py` - I/O for DBoF tables and cutouts

- **`fronts/train/`** - ML training dataset generation
  - `datasets.py` - Generate train/valid/test sets from DBoF (`generate_from_dbof`)
  - `cutouts.py` - Create HDF5 cutouts for training
  - `tables.py` - Table generation for train/valid/test splits

- **`fronts/llc/`** - LLC4320 model data access
  - `io.py` - Load LLC coordinates, cutouts, and velocity fields from local or S3
  - `slurp.py` - Bulk data ingestion

- **`fronts/properties/`** - Front property measurements
  - `measure.py` - Measure properties (e.g., average field values) along fronts

### Data Flow

1. **LLC4320 Data** → Load via `fronts.llc.io` (local or S3)
2. **DBoF Tables** → Index cutouts with `fronts.dbof` (Parquet tables, HDF5 fields)
3. **Front Detection** → `pyboa.pyBOA` accessor or `algorithms.fronts_from_divb2`
4. **Thinning** → Morphological thinning to single-pixel fronts
5. **Properties** → Measure properties via `fronts.properties.measure`

### Key Patterns

- **xarray accessor**: Use `array.pyBOA.auto_detection()` for full front detection pipeline
- **DBoF JSON files**: Configuration files point to data locations and define cutout metadata
- **Environment variables**:
  - `DBOF_PATH` - Path to DBoF data
  - `LLC_DATA` - Local LLC4320 data path
  - `OS_OGCM` - Ocean model data root

### MedSST Format (for Cornillon algorithms)

The Cornillon-style algorithms use scaled integer SST (MedSST):
- Range: 0-255 or 0-400 (int16)
- Values ≤8 are invalid (land/ice/cloud)
- Convert: `SST_counts = (SST_kelvin - 268.15) / 0.01`
- See `docs/CC/MEDSST_EXPLANATION.md` for details

## Field Definitions

Key oceanographic fields defined in `fronts/dbof/defs.py`:
- `Divb2`: Squared divergence of buoyancy (primary front indicator)
- `SSTK`: SST in Kelvin
- `SSH/SSHa/SSHs`: Sea surface height variants
- `vorticity`, `divergence`, `strain_rate`: Flow diagnostics
- `OW`: Okubo-Weiss parameter
- `Fs`: Frontogenesis tendency
