# build_v5 — front finding end to end

`fronts/runs/prototypes/one_full/build_v5.py`

Finds fronts in LLC4320 gradb2, groups them, and co-locates them with physical
fields. Data production is delegated to the preprocessing repo
(`dbof.cli.run_all_subsets`); fronts only finds, groups, and co-locates.

## The five steps

| Step | What it does | Cost |
|------|--------------|------|
| **1** | Build the frontal-structure store, export **gradb2 only** → `gradb2.nc` | 1 NetCDF |
| **2** | Threshold gradb2 → binary front map (`*_bfronts.npy`) | cheap, local |
| **3** | Label the fronts + geometric properties (`label_map`, groups) | cheap, local |
| **4** | Build **every other subset**, co-locate straight from the stores | slow, no disk |
| **5** | Copy the front products back to S3 | network |

Steps 1–3 are self-contained. If all you want is a map of where the fronts are,
stop after 3 and never pay for the other ~140 channels.

```bash
cd fronts/runs/prototypes/one_full

python build_v5.py 1 run_v5_100_timesteps.yaml   # gradb2
python build_v5.py 2 run_v5_100_timesteps.yaml   # find
python build_v5.py 3 run_v5_100_timesteps.yaml   # group
python build_v5.py 5 run_v5_100_timesteps.yaml   # push to S3
python build_v5.py 4 run_v5_100_timesteps.yaml   # everything else + co-locate
```

Everything is re-runnable: existing `.nc` files and S3 keys are skipped.

To try a single timestep first, comment out the other dates in the config's
`date_iterations` list.

## Where the zarr stores come from

Steps 1 and 4 both call `run_all_subsets` with `--generate-only`, which decides
per subset and date: a store that is already complete is skipped, so re-running
costs a few S3 metadata reads. Step 1 asks for one subset (the one owning
gradb2, plus `icearea` when masking); step 4 asks for all of `active_subsets`.

Neither lets `run_all_subsets` do the export. It has no `--channels` flag, so
its export phase writes all 8 SURF channels (21 on DEPTH) when only gradb2 is
wanted; and it writes to `{netcdf_base}/{run_id}/{date_prefix}/`, which is not
where this build keeps its products. Step 1 exports gradb2 through
`export_channels`; step 4 writes no NetCDF at all.

## Co-location reads the stores directly

Step 4 pulls each field from S3 one at a time via `llc_io.read_channel` and
reduces it to per-front statistics before dropping it. Nothing is staged to
disk: a global field is ~900 MB, so 100 timesteps x 9 SURF channels would be
~800 GB, and a DEPTH run with 57 channels would need ~51 GB of RAM if they were
all held at once.

This is safe because `zarr_to_netcdf` never transformed anything — it cast the
channel to float32 and added integer `y`/`x` coords. A field read from the store
is the array the NetCDF would have contained, on the same grid as the label map
(which descends from `gradb2.nc`, from the same store).

**Checkpoints.** Each property's columns are cached at

```
{date_prefix}/colocate_ckpt_{run_tag}/{channel}.parquet
```

so a run killed at channel 40 of 57 resumes there rather than starting the
timestamp over. The directory is removed once the timestamp's final parquet
lands, and `publish.PRODUCT_PATTERNS` never matches it.

**Reading NetCDF instead.** `build.colocate_source: netcdf` reads the
per-property `.nc` files from the timestamp directory, as before. Useful for
co-locating offline against files already on disk, or for checking the two paths
agree — run one timestep each way and diff the parquets.

## Why this differs from build_v4

v4 generated **all** active subsets in step 1, even though steps 2–3 only read
gradb2. v5 splits that: step 1 runs `run_all_subsets --subsets frontal_structure
--generate-only`, then exports the single gradb2 channel itself. The rest moves
to step 4.

## Six things that broke between v4 and now

Audited against the current preprocessing repo. All six are fixed in v5;
`fronts/tests/test_build_v5.py` pins each one.

1. **gradb2 has a different name per pipeline.** SURF/OSN emit bare `gradb2`;
   DEPTH emits `gradb2_sfc`. v4 hardcoded `gradb2_sfc` → step 2 dies on a
   missing file under SURF. v5 resolves it with `channel_for_root()`.

2. **`PROPERTY_ROOTS` was DEPTH-only.** 15 of v4's roots (`N2`, `Ri`, `Fr`,
   `Ro`, `Bu`, `ertel_pv*`, `uB/vB/wB`, `KE`, `vertical_shear`,
   `mixed_layer_depth`, `ml_heat_content`) don't exist in SURF —
   `expand_property_roots` raises on all 15. v5 derives the list with
   `all_property_roots()`.

3. **The hardcoded list had already drifted.** `R_ib`, `Wstar` and
   `rossby_number` were added upstream and v4 silently never co-located them.
   A derived list picks up new channels automatically.

4. **Step 1 exported the whole subset.** SURF `frontal_structure` is 8 channels
   (it gained `density` + `buoyancy`); DEPTH with 4 suffixes is 21. v5 exports 1.

5. **No ice-mask control from fronts.** Masking lives in
   `zarr_to_netcdf(ice_mask=...)` and is auto-wired inside `run_all_subsets`,
   but `llc_io.zarr_to_nc` never forwarded it. Now it does.

6. **Exports assumed a single date.** `zarr_to_nc` passed the whole
   `date_iterations` list with one output filename, which `zarr_to_netcdf`
   rejects. Fine for v4's 1-date config, fatal at 100 timesteps. Now it pins one
   `date_prefix` per call.

## Config

Standard thin global config (`pipeline`, `run`, `data`, `output`,
`active_subsets`, `depth_suffixes`) plus an optional `build:` block:

```yaml
build:
  finding_config: "D"        # fronts/finding/configs/finding_config_D.yaml
  finding_suffix: "sfc"      # which depth suffix to find in (DEPTH only)
  ice_mask_find:  false      # step 1 — mask gradb2 BEFORE finding fronts
  ice_mask_props: true       # step 4 — mask the co-located property fields
  colocate_source: zarr      # step 4 field source: zarr | netcdf
  percentiles:    [25, 75, 90]
  exclude_roots:  []         # roots to skip in co-location
```

Every key has a default (`fronts.properties.run.BUILD_DEFAULTS`), so the block
is optional.

**Ice mask.** Two independent toggles, because you usually want fronts found on
the *unmasked* field and masking applied only afterwards. `ice_mask_find`
requires `icearea.zarr` to exist for the same run_id and date — the mask is read
from it at export time.

**Pipeline.** Set `pipeline: SURF | OSN | DEPTH`. Nothing else in the driver
changes — channel names, subset membership and the S3 folder all follow from it.
Note SURF reads Theta/Salt from OSN kerchunk and only the forcing fields
(`oceTAUX/Y`, `SIarea`, `oceQnet`) from `LLC4320_RAW/SURFACE`.

## Paths

Products are organised by the **build** that made them; filenames keep the
**source** `run_id`, so a file always names the dataset it came from.

```
source   s3://{bucket}/{folder}/{run_id}/{YYYYMMDD_HHMMSS}/{subset}.zarr
local    $OS_OGCM/LLC/Fronts/{build_version}/{pipeline}/{YYYYMMDD_HHMMSS}/
             LLC4320_{timestamp}_{channel}_{run_id}.nc
             LLC4320_{timestamp}_{run_id}_bfronts.npy
             labeled_fronts_global_*, front_index_*, global_front_geometry_*,
             front_properties_*, metadata_*
pushed   s3://{bucket}/{folder}/{run_id}/{YYYYMMDD_HHMMSS}/Fronts/
```

For example, `V5` + `SURF` + `v2_2_01`:

```
s3://dbof/globals_for_cutouts/v2_2_01/20111204_000000/frontal_structure.zarr
$OS_OGCM/LLC/Fronts/V5/SURF/20111204_000000/LLC4320_2011-12-04T00_00_00_gradb2_v2_2_01.nc
s3://dbof/globals_for_cutouts/v2_2_01/20111204_000000/Fronts/...
```

`fronts.llc.io.set_run_layout()` owns the split. `folder` defaults to
`surface_fields/` (SURF, OSN) or `depth_fields/` (DEPTH); set `output.folder` to
read stores written somewhere else.

`V5` comes from `build_v5.py` itself (`BUILD_VERSION`), not from the config, so
everything this driver writes lands under `Fronts/V5/{pipeline}/` no matter
which dataset it read. The source dataset is recorded in two other places: the
filename tag, and the `.meta` name.

| | pipeline | `run_id` | `output.folder` | source store | local products |
|---|---|---|---|---|---|
| A | DEPTH | `V5` | *omit* | `s3://dbof/depth_fields/V5/{ts}/` | `Fronts/V5/DEPTH/{ts}/…_gradb2_sfc_V5.nc` |
| B | SURF | `v2_00_2` | `globals_for_cutouts/` | `s3://dbof/globals_for_cutouts/v2_00_2/{ts}/` | `Fronts/V5/SURF/{ts}/…_gradb2_v2_00_2.nc` |

In A the folder is the DEPTH default, so it can be omitted. `run_id` is a free
string: underscore-heavy tags survive the filename parser that recovers them.

Two source datasets with the same pipeline therefore share a local directory.
Nothing is overwritten — every product name carries its tag — and the push
filters on that tag, so one run never uploads another's files.

The raw store (`s3://dbof/LLC4320_RAW/{SURFACE,DEPTH}/`) is not configurable
from here — it is a constant in `dbof.global_dataset_creation.data_sources`,
picked by pipeline.

## Pushing back to S3

Step 5 (`fronts.llc.publish`) copies each timestamp's products into a `Fronts/`
folder beside the zarr stores they came from. The destination is read from the
same config that drove the run, so products cannot land next to the wrong
dataset. Existing keys are skipped unless clobbering.

The `.nc` exports are deliberately **not** pushed — they are exports of the zarr
store sitting in the parent directory. Change `publish.PRODUCT_PATTERNS` if you
want them.

## Co-locating on a single tile

`fronts.properties.run.colocate_tile` pairs the global fronts with properties
computed on one 720x720 tile, at the surface. Nothing global is read or written.

```python
from fronts.llc import io as llc_io, tiles as llc_tiles
from fronts.properties.run import colocate_tile

llc_io.set_fronts_path(...)                     # as build_v5 does
llc_io.set_run_layout('V5/SURF', file_tag='V5')

tile = llc_tiles.tile_for(lon=-70, lat=40)      # or i_rect=, j_rect=
colocate_tile('2012-07-03T12_00_00', 'D', 'V5',
              property_names=['density', 'relative_vorticity', 'N2'],
              tile=tile, percentiles=[90])
```

Results land in `{date_prefix}/tile{idx:03d}/`, with the per-property tile
NetCDFs cached under `fields/` — small (~2 MB for a 2D field), and reused on a
re-run. Property names come from `dbof.tiles.field_registry`, not from
`subset_definitions`; the loader follows each property's `out_name` (`density`
is stored as `sigma0`).

**Orientation.** A tile is a window on the same rect grid as the label map, but
its *data* is face-local, which on some LLC faces is a rotation of that window.
`labels_for_tile` scatters the labels through the per-pixel `(j_face, i_face)`
maps rather than slicing, so the pairing is right on every face. Slicing would
look plausible — both arrays are 720x720 — while pairing every front pixel with
the wrong value.

**What the numbers mean.** Labels stay global, so rows join to the geometry
table on `flabel`, and `tile_idx` / `face_idx` are added as columns. A front
crossing the tile edge is clipped, so `npix` counts only its pixels inside the
tile:

- `mean`, `std`, `count` recombine across tiles (keep sum, sum-of-squares, n)
- `median` and percentiles do **not**

For uncontaminated percentiles, filter to fronts whose tile `npix` equals the
global `npix` from the geometry table. `min_npix` filters on the clipped count,
so prefer filtering afterwards on the global value. Stencil-based properties
carry an `edge_margin` NaN rim that `nan_policy='omit'` already drops from the
statistics; pass `edge_margin=` to keep those cells out of `npix` too.

## The run descriptor

Step 1 writes one YAML file at the top of the run directory:

```
$OS_OGCM/LLC/Fronts/V5/SURF/
    fronts_meta_V5_SURF_from_globals_for_cutouts_v2_2_01_run_v5_100_timesteps.meta
    fronts_meta_V5_SURF_from_globals_for_chunks_V5_run_v5_chunks.meta
    fronts_meta_V5_SURF_from_globals_for_chunks_V5_run_v5_SO_chunks.meta
```

The name says which dataset the fronts came from and which config drove the run,
without opening anything. The config stem is what keeps two date lists against
the same dataset apart — the last two above share build, pipeline, folder and
run_id.

Inside: pipeline, S3 store URI, subsets, the resolved gradb2 channel, the
front-finding config, ice-mask flags, date count and range, and the git SHA of
both repos at the time of the run.

## Checking a store before you run

```

```python
from dbof.io.filesystems import create_s3_filesystems
from dbof.global_dataset_creation import check_existence
from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix

_, fs = create_s3_filesystems("https://s3-west.nrp-nautilus.io")
path = make_run_prefix("dbof/", "globals_for_cutouts/", "v2_2_01",
                       "frontal_structure.zarr", date_prefix="20111204_000000")
print(check_existence.store_channels(fs, path))
```

The export needs only the one channel it asks for, so a store missing newer
channels (`density`, `buoyancy`, ...) still works for gradb2. `run_all_subsets`
would judge such a store incomplete and rebuild it — another reason step 1 does
not call it.

## Tests

```bash
pytest fronts/tests/test_build_v5.py -v
```

81 tests, fully offline — no S3, no OSN, no data. They cover the contract with
the preprocessing repo, that step 1 builds one subset and exports one channel,
pipeline resolution, the two ice-mask toggles, the output layout, the S3 push,
the run descriptor, both naming schemes across all three pipelines, lazy
co-location with checkpoint resume, tile orientation across rotated faces, and
the shipped 100-timestep config. If the preprocessing repo changes shape underneath us,
these fail fast and name what moved.
