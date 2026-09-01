# build_v5 — front finding end to end

`fronts/runs/prototypes/one_full/build_v5.py`

Finds fronts in LLC4320 gradb2, groups them, and co-locates them with physical
fields. Data production is delegated to the preprocessing repo
(`dbof.cli.run_all_subsets`); fronts only finds, groups, and co-locates.

## The four steps

| Step | What it does | Cost |
|------|--------------|------|
| **1** | Build the frontal-structure store for any date that lacks gradb2, export **gradb2 only** → `gradb2.nc` | 1 subset, 1 NetCDF |
| **2** | Threshold gradb2 → binary front map (`*_bfronts.npy`) | cheap, local |
| **3** | Label the fronts + geometric properties (`label_map`, groups) | cheap, local |
| **4** | Build + export **every other subset**, then co-locate | expensive |
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

Everything is re-runnable: complete zarr stores, existing `.nc` files and
existing S3 keys are all skipped.

To try a single timestep first, comment out the other dates in the config's
`date_iterations` list.

## Where the zarr stores come from

Both step 1 and step 4 call `run_all_subsets`, and both pass `--generate-only`:
they build stores, never export from them. Step 1 asks for one subset (the one
that owns gradb2, plus `icearea` when `ice_mask_find` is on); step 4 asks for
everything in `active_subsets`.

The export is always done by `export_channels` instead, for two reasons.
`run_all_subsets` has no `--channels` flag, so its export phase writes all 8
SURF channels (21 on DEPTH) when only gradb2 is wanted. And it exports to
`{netcdf_base}/{run_id}/{date_prefix}/`, which is not where this build keeps
its products — the property files would land where co-location does not look.

### What "skip" means for the generate pass

Step 1 goes through `fronts.properties.run.generate_for_channels()`, which asks
`check_existence.plan_zarr()` about **only the channel this step reads** — not
the subset's full channel list:

| `plan_zarr` verdict (vs. `gradb2` alone) | Meaning | Response |
|---|---|---|
| `ZARR_FULL` | gradb2 present, `iteration` marker written | export from it as-is |
| `ZARR_MISSING` | no store | generate the subset |
| `ZARR_INCOMPLETE` | store exists but has no gradb2, or was never finished | generate the subset |

The check runs on the config's **first date only**, and its verdict is taken
for all of them: a run's dates are produced together and hold the same
channels, so one metadata GET per subset answers the question. (The trade-off:
a half-finished transfer whose early dates are complete reads as ready, and its
later dates fail at export instead.) A subset with nothing to build never
reaches `run_all_subsets` — one line of output, no per-date `SKIP` for all 100.
Dates that *are* handed over are skipped or built individually by
`generate_global`'s own pre-flight, as always.

**Why the narrowed question matters.** Asked about the whole subset, a store
written before `density` and `buoyancy` were added upstream comes back
`ZARR_INCOMPLETE`, and `generate_global` raises on it *in its pre-flight loop*,
before generating anything:

```
ValueError: Existing zarr store is incomplete: s3://...
  Delete the store and rerun, or pass --clobber to regenerate it in place.
```

`run_all_subsets` catches that and carries on, so the run still exits 0 and
gradb2 still exports. But because the raise lands in the planning loop, **one
stale date abandons the generate pass for every date**, including any that were
genuinely missing — which then surface much later as `FileNotFoundError` from
the export. Asking about gradb2 alone keeps a stale store out of the pass
entirely.

This did not arise in build_v4 because v4 generated every subset itself, in one
pass, from a single code version: its stores were never out of step with the
subset definition, and its `.nc`-first plan skipped without consulting zarr at
all. v5 reads stores the preprocessing repo built at some other time, which is
what makes drift possible.

The case this cannot rescue: a store that exists, lacks gradb2, *and* is
incomplete by the subset's full list. `generate_global` will not touch it.
Delete it and rerun.

### If a store is missing and does not get built

Worth knowing what that failure looks like, because it is not obvious:
`export_channels` → `zarr_to_nc` → `GlobalZarrDatasetReader` opens the store
with `zarr.open_group(mode="r")`, which raises `FileNotFoundError` on a path
that is not there. Nothing catches it, so the run dies on the **first** such
date and every later date goes unattempted — and the traceback names an S3 key,
not the missing subset. If step 1 ends this way, the real explanation is in the
generate log above it.

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
  percentiles:    [25, 75, 90]
  exclude_roots:  []         # roots to skip in co-location
```

Every key has a default (`fronts.properties.run.BUILD_DEFAULTS`), so the block
is optional.

**Ice mask.** Two independent toggles, because you usually want fronts found on
the *unmasked* field and masking applied only afterwards. Either flag needs
`icearea.zarr` for the same run_id and date — the mask is read from it at export
time. Turning on `ice_mask_find` makes step 1 build `icearea.zarr` alongside the
frontal-structure store; `ice_mask_props` relies on step 4's full generate pass.

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

## The run descriptor

Step 1 writes one YAML file at the top of the run directory:

```
$OS_OGCM/LLC/Fronts/V5/SURF/fronts_meta_V5_SURF_from_globals_for_cutouts_v2_2_01.meta
```

The name says which dataset the fronts came from without opening anything.
Inside: pipeline, S3 store URI, subsets, the resolved gradb2 channel, the
front-finding config, ice-mask flags, date count and range, and the git SHA of
both repos at the time of the run.

## Checking a store before you run

Cheap, metadata-only — it tells you in
advance whether the generate pass will skip, build, or raise:

```python
from dbof.io.filesystems import create_s3_filesystems
from dbof.global_dataset_creation import check_existence
from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix

_, fs = create_s3_filesystems("https://s3-west.nrp-nautilus.io")
path = make_run_prefix("dbof/", "globals_for_cutouts/", "v2_2_01",
                       "frontal_structure.zarr", date_prefix="20111204_000000")
print(check_existence.store_channels(fs, path))
```

`store_channels` returns what the store actually holds; `plan_zarr(fs, path,
expected)` gives the `FULL` / `MISSING` / `INCOMPLETE` verdict step 1's generate
pass will act on.

Pass the subset's full channel list and you get the preprocessing repo's
verdict; pass `["gradb2"]` and you get step 1's. They disagree on purpose — the
export needs only the one channel it asks for, so a store missing newer
channels (`density`, `buoyancy`, …) is `INCOMPLETE` to the first question and
`FULL` to the second. Step 1 asks the second.

## Tests

```bash
pytest fronts/tests/test_build_v5.py -v
```

65 tests, fully offline — no S3, no OSN, no data. They cover the contract with
the preprocessing repo, that step 1 builds one subset and exports one channel,
pipeline resolution, the two ice-mask toggles, the output layout, the S3 push,
the run descriptor, both naming schemes across all three pipelines, and the
shipped 100-timestep config. If the preprocessing repo changes shape underneath us,
these fail fast and name what moved.
