# build_v5 — front finding end to end

`fronts/runs/prototypes/one_full/build_v5.py`

Finds fronts in LLC4320 gradb2, groups them, and co-locates them with physical
fields. Data production is delegated to the preprocessing repo
(`dbof.cli.run_all_subsets`); fronts only finds, groups, and co-locates.

## The four steps

| Step | What it does | Cost |
|------|--------------|------|
| **1** | Build the frontal-structure zarr store, export **gradb2 only** → `gradb2.nc` | 1 subset, 1 NetCDF |
| **2** | Threshold gradb2 → binary front map (`*_bfronts.npy`) | cheap, local |
| **3** | Label the fronts + geometric properties (`label_map`, groups) | cheap, local |
| **4** | Generate + export **every other subset**, then co-locate | expensive |

Steps 1–3 are self-contained. If all you want is a map of where the fronts are,
stop after 3 and never pay for the other ~140 channels.

```bash
cd fronts/runs/prototypes/one_full

python build_v5.py 1 run_v5_100_timesteps.yaml   # gradb2
python build_v5.py 2 run_v5_100_timesteps.yaml   # find
python build_v5.py 3 run_v5_100_timesteps.yaml   # group
python build_v5.py 4 run_v5_100_timesteps.yaml   # everything else + co-locate
```

Everything is re-runnable: existing zarr stores and `.nc` files are skipped.

**Testing on a few timesteps.** `--ndates N` takes the first N dates, `--date`
takes a specific one (repeatable). A reduced copy of the config is written to a
temp file and used for every stage, so the generator builds one store rather
than a hundred. The config itself is never edited.

```bash
python build_v5.py 1 run_v5_100_timesteps.yaml --ndates 1
python build_v5.py 1 run_v5_100_timesteps.yaml --date '2012-08-01 00:00:00'
```

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
the *unmasked* field and masking applied only afterwards. Turning on
`ice_mask_find` makes step 1 also build `icearea.zarr`, since the mask is read
from it at export time.

**Pipeline.** Set `pipeline: SURF | OSN | DEPTH`. Nothing else in the driver
changes — channel names, subset membership and the S3 folder all follow from it.
Note SURF reads Theta/Salt from OSN kerchunk and only the forcing fields
(`oceTAUX/Y`, `SIarea`, `oceQnet`) from `LLC4320_RAW/SURFACE`.

## Paths

`run_id` is the run tag, used verbatim — no `V` prefix, no separate version.
Producer and consumer line up automatically:

```
$OS_OGCM/LLC/Fronts/{run_id}/{YYYYMMDD_HHMMSS}/LLC4320_{timestamp}_{channel}_{run_id}.nc
```

## Tests

```bash
pytest fronts/tests/test_build_v5.py -v
```

40 tests, fully offline — no S3, no OSN, no data. They cover the contract with
the preprocessing repo, that step 1 builds one subset and exports one channel,
pipeline resolution, the two ice-mask toggles, date narrowing, and the shipped
100-timestep config. If the preprocessing repo changes shape underneath us,
these fail fast and name what moved.
