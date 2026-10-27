# fronts-tile-finding: run build_v5 steps 1-3 on ONE tile

**Companion to** `llc4320-native-grid-preprocessing/prompts/tile_surface_series.md`,
which holds the field-calculation half of this work.

**Status: IMPLEMENTED.** See Logs.

## Question answered first: how much of the front finder assumes a global grid?

**None of it.**  This was the surprise, and it is the reason the change here is
so small.

`finding/algorithms.fronts_from_gradb2` and everything under it --
`pyboa.front_thresh`, `sharpen.global_sharpen_pq`, `morphology.thin`,
`pyboa.cropping`, `despur.prune_short_spurs` -- are plain 2D numpy/skimage
operations.  Boundary handling is non-periodic everywhere
(`mode='constant'`, `np.pad(..., constant_values=False)`); there is no
longitude wrap, no land mask, no LLC face layout, no lat/lon, and no global
percentile (config D's `threshold: 85` is a *local* percentile over a
`window: 64` box, so a tile gets its own local thresholds).

`finding/run.find_gradb2_fronts` opens one NetCDF and immediately takes
`.values`, so dims and coords are ignored entirely.  It just needs a 2D array
in a file at `llc_io.derived_filename(timestamp, gradb2_field, version)` whose
data variable is named `gradb2_field`.

**That file path is the whole seam.**  Put a 720x720 tile there and step 2
runs unmodified.

None of the finding configs (`finding/configs/finding_config_*.yaml`) has a
single spatial, geographic or grid key.  Nothing to change there either.

## What changed

1. **`preproc/gradb2.py` -- `generate_tile_gradb2(...)`.**  Computes the field
   for one tile across many timestamps via
   `dbof.tiles.tile_utils.run_series`, writing each snapshot to the path step 2
   already looks for.  The existing `generate_gradb2` builds the global zarr
   and exports a channel out of it; this skips both, because for a 720x720
   tile that is enormously more work than the answer needs and the hourly
   Theta/Salt it would need are not in any of the S3 stores.

2. **`properties/run.py`** -- `BUILD_DEFAULTS['tile_find'] = None`, and
   `group_fronts(...)` takes `coords_file=None`.  Step 3 needs lat/lon on the
   SAME grid as the binary map; the hardcoded default is the 12960x17280
   coords file.  `prop_algorithms.group_fronts` already took `lat, lon` as
   arguments, so this is just letting the caller say where they come from.

3. **`runs/prototypes/one_full/build_v5.py`** -- three small edits:
   - step 1 branches to `generate_tile_gradb2` when `build.tile_find` is set;
   - the run layout gains a tile leaf (`V5/{pipeline}/{tile_name}/`), because
     `finding/io.binary_filename` ignores the finding-config label and a tile
     run would otherwise write the same filenames into the same date folder as
     a global run with the same `run_id`;
   - step 3 passes the tile's gradb2 NetCDF as `coords_file` -- the tile's own
     `XC`/`YC` ride along inside it, so no new coords file is needed.

Steps 4 and 5 are untouched and are NOT part of this path.  If you later want
co-location on the tile, note that `properties.run.colocate_tile` reorients a
*global* label map onto the tile via `llc_tiles.labels_for_tile`; when the
fronts were found on the tile the label map is already in face-local space and
that call has to be bypassed.  That is the one place where "find on a tile"
and the existing tile machinery actively conflict.

## Config

One file, shared with the other repo:
`llc4320-native-grid-preprocessing/configs/tiles/tile330_gradb2_osn.yaml`.
It carries `pipeline`, `run.run_id`, `active_subsets`, the 504 hourly
`data.date_iterations`, and:

```yaml
build:
  finding_config: "D"
  gradb2_root: "gradb2"
  tile_find:
    name: "tile330"
    lon: -121.9
    lat: 36.8
    property: "gradb2"
    pipeline: "OSN"
```

```bash
python build_v5.py 1 <that file>   # gradb2 for the tile, 504 NetCDFs
python build_v5.py 2 <that file>   # binary fronts   (unmodified code)
python build_v5.py 3 <that file>   # labels + geometry
```

## Worth knowing

- `find_gradb2_fronts` hardcodes `bparam['n_workers'] = 10`
  (`finding/run.py:53`).  With `thresh_mode: pool` on 720 rows that is ten
  processes for 72-row slabs, per timestep, 504 times.  It works, but the
  parallelism is in the wrong place: one process per timestamp would be
  better.  Left alone -- changing it is a behaviour change to the global path
  too.
- `properties/io.write_front_index` records raw array indices (`x0/y0/x1/y1`)
  into whatever 2D array it was given.  For a tile run those are
  **tile-local**, not rect-global.  The metadata JSON records `shape`,
  `lat_range` and `lon_range`, so a file is self-describing -- but do not mix
  tile and global front tables without checking that field.
- `fronts/llc/tiles.py` uses `c_grid_axis_shift: -0.5` while the preprocessing
  repo stamps `+0.5`.  Unresolved (see `prompts/fronts_build.md` 372-423).
  Irrelevant for `gradb2` (`median |rel| ~ 2e-4`); not irrelevant for anything
  velocity-derived.

## Logs

### 2026-08-26 (build_v5 can find fronts on a single tile)

Made the three edits above.  What I learned:

- **The finding algorithms were already tile-ready** and had been all along --
  `prompts/front_finding.md` records that they were developed against
  1000x1000 sub-windows of the global field.  The architecture docs describe
  fronts as found globally and co-located regionally, so there was no design
  note proposing tile-local finding; there did not need to be, because
  nothing blocked it.
- **The seam is the filename, not the code.**  `find_gradb2_fronts` reading
  `.values` out of a NetCDF means the whole tile path reduces to "write the
  file where step 2 looks".  Handing `run_series` an `output_paths` list built
  from `llc_io.derived_filename` was the entire integration.
- **Verified** the wiring offline: `read_build_config` digests the shared
  config (504 dates, `tile_find` parsed, `run_dir` = `V5/OSN`);
  `channel_for_root` -> `gradb2`, `subset_for_channel` -> `frontal_structure`;
  and `generate_tile_gradb2` hands `run_series` the right location/property/
  pipeline and produces paths that match `find_gradb2_fronts`'s lookup
  exactly.
- **Not verified here**: `fronts/tests/test_build_v5.py` needs
  `scipy.ndimage.vectorized_filter` (scipy >= 1.16, Python >= 3.11) and the
  sandbox this was written in only had 3.10.  Run that suite before merging --
  it monkeypatches every side-effecting call, so it is fast and offline.
