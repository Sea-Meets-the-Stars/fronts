# MIT Fussing

## Understanding the LLC4320 model data

### Goal

The goal is to understand the LLC4320 model data format on the MIT system.  The path to the data is /orcd/data/abodner/003/LLC4320

### Summary

The data at `/orcd/data/abodner/003/LLC4320` is a **Zarr v2 store** containing the full LLC4320 MITgcm high-resolution ocean simulation output. The actual store root is at `/orcd/data/abodner/003/LLC4320/LLC4320/`.

#### Dataset Overview

| Property | Value |
|---|---|
| **Time range** | 2011-09-13 to 2012-11-15 (~14 months) |
| **Temporal resolution** | Hourly (10,311 time steps) |
| **Spatial grid** | 13 LLC faces, 4320×4320 per face (~1/48° global resolution) |
| **Depth levels** | 51 (surface at -0.5 m to ~946 m) |
| **Uncompressed size** | ~2.7 PB |
| **Compression** | Blosc (zstd for large arrays, lz4 for coordinates) |
| **Chunk sizes** | 3D time-varying fields: `(1, 51, 1, 720, 720)` — one timestep, all depths, one face, 720×720 spatial tiles (36 tiles per face). 2D surface fields: `(1, 13, 4320, 4320)` — one timestep, all faces. Static grid fields: `(1, 720, 720)` per face. |

#### Variables

**3D prognostic fields** (time × k × face × j × i):
- `Theta` — potential temperature (°C)
- `Salt` — salinity
- `U`, `V`, `W` — velocity components

**2D surface fields** (time × face × j × i):
- `Eta` — sea surface height
- `KPPhbl` — KPP boundary layer depth
- `oceQnet`, `oceQsw`, `oceFWflx`, `oceSflux` — surface heat/freshwater fluxes
- `oceTAUX`, `oceTAUY` — wind stress
- `PhiBot` — bottom pressure
- Sea ice: `SIarea`, `SIheff`, `SIhsnow`, `SIhsalt`, `SIuice`, `SIvice`

**Static grid/geometry** (no time dimension):
- `XC`, `YC`, `XG`, `YG` — cell center and corner lon/lat
- `Depth`, `dx*`, `dy*`, `rA*` — bathymetry, grid spacings, cell areas
- `hFacC`, `hFacS`, `hFacW`, `mask_c`, `mask_s`, `mask_w` — partial cell fractions and land/ocean masks
- `CS`, `SN` — cos/sin of grid angle (for vector rotation to lat/lon)
- `Z`, `Zl`, `Zu`, `Zp1`, `drC`, `drF` — vertical grid geometry
- `PHrefC`, `PHrefF`, `rhoRef` — reference pressure and density profiles

#### How to Access the Data

Use the `ocean13` conda environment, which has xarray, zarr, and dask installed.

**Opening the store:**

```python
import xarray as xr

ds = xr.open_zarr(
    '/orcd/data/abodner/003/LLC4320/LLC4320',
    consolidated=False,  # metadata is not consolidated; this avoids a warning
)
```

Note: Because the metadata is not consolidated, opening is slower than usual. If you have write access, you can speed up future opens by running `zarr.consolidate_metadata('/orcd/data/abodner/003/LLC4320/LLC4320')` once.

**Loading a single surface SST snapshot (one face, one timestep):**

```python
# Surface temperature for face 7, first timestep
sst = ds.Theta.isel(time=0, k=0, face=7).load()
print(sst)
```

**Accessing coordinates:**

```python
lon = ds.XC.isel(face=7).load()
lat = ds.YC.isel(face=7).load()
```

**Running from the command line:**

```bash
conda run -n ocean13 python my_script.py
# or
conda activate ocean13
python my_script.py
```

All data is lazily loaded via Dask, so selecting specific slices (face, time, depth) before calling `.load()` or `.values` is essential to avoid loading terabytes into memory.

### Prompts

1. Please examine the data in the path /orcd/data/abodner/003/LLC4320 and its sub-folders.  I believe it is a zarr store.  If you wish to run python, be sure to use the "ocean13" conda environment.

2. Ok, please add a summary of what you found to the summary section in the mit_fussing.md file located at /orcd/home/002/profx/Oceanography/python/fronts/prompts/mit_fussing.md.  Include instructions on how to access the data and how to run python code to examine the data.

claude --resume 229a1abf-6cdf-438f-af3b-41f97724cce0