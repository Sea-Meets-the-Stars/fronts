# Wiring the pages to real data

The pages ship running on **synthetic data** — a fabricated ocean, so the
layout and interactions work before any store is touched. This is the list
of what has to happen to point them at the real thing.

Most of it is one afternoon of mechanical work. The one genuinely
uncertain piece is the display pyramid at full scale (step 5).

---

## 0. Install

```bash
conda activate llcngp_osmesa          # the env from the 3-D runbook
pip install panel holoviews datashader geoviews hvplot cmocean
pip install -e /path/to/llc4320-native-grid-preprocessing
pip install -e /path/to/fronts
```

Check it runs on synthetic data first:

```bash
python -m fronts.viz.apps.serve --show
```

Three routes: `/characteristics`, `/tiles`, `/evolution`.

---

## 1. Tell me what is in the store

This is the blocking one — the store layout has not been confirmed, and
guessing filenames produces obscure failures. **Run these and send me the
output.**

```bash
# a) what is in the date's directory
aws s3 ls --recursive s3://dbof/globals_for_cutouts/v2_2_01/20120516_060000/ | head -100

# b) how big is it
aws s3 ls --recursive --summarize s3://dbof/globals_for_cutouts/v2_2_01/20120516_060000/ | tail -5

# c) what other dates exist
aws s3 ls s3://dbof/globals_for_cutouts/v2_2_01/

# d) the raw 3-D store's top level
aws s3 ls s3://dbof/LLC4320_RAW/DEPTH/20120516T06.zarr/
```

Then, for whichever of those is a zarr or NetCDF holding the 2-D fields:

```bash
python - <<'PY'
import xarray as xr
# adjust to a real path from (a)
ds = xr.open_zarr("s3://dbof/globals_for_cutouts/v2_2_01/20120516_060000/<STORE>",
                  storage_options={"anon": False})
print(ds)
print("VARS:", list(ds.data_vars))
print("COORDS:", list(ds.coords))
print("DIMS:", dict(ds.sizes))
PY
```

**What I need from this**, and why:

| Question | Why it matters |
|---|---|
| Exact filenames / store names per channel | `S3Provider.field` cannot guess the naming convention |
| Do `XC`/`YC` live in the store, or in a separate coords file? | `S3Provider.coords`; the whole selection path depends on it |
| Are the kinematic channels `relative_vorticity` or `relative_vorticity_sfc`? | The SURFACE and DEPTH pipelines name them differently; the joint PDFs need vorticity, strain and Coriolis |
| Filenames of the binary fronts, labelled fronts, geometry parquet, colocation parquet | The four `NotWiredUp` methods |
| The grid dimensions actually stored | Confirms the 12960 × 17280 assumption |

Everything in `S3Provider` that raises `NotWiredUp` names exactly which of
these it is waiting on.

---

## 2. Pick the six region centres

The placeholders in `common/regions.py` were chosen from the region names,
not from the science. Replace them with centres you actually want, then
resolve each to a tile:

```bash
python - <<'PY'
from fronts.viz.apps.common import regions
from fronts.llc import coords            # your lat/lon -> (i, j) helper

def latlon_to_ij(lat, lon):
    return coords.latlon_to_pixel(lat, lon)     # confirm the real name

for key, tile in regions.resolve_all(latlon_to_ij).items():
    print(f"{key:16s} -> tile {tile}")
PY
```

Paste the resulting indices into the `Region(...)` entries as `tile_idx=`.
A tile spans roughly 15°, so the centre only has to land inside the
current system you mean.

---

## 3. Generate the tiles

Two tiles per region: density for the geometry, plus one per colour field
you want available. **Nothing existing is reusable** — the tiles in the
3-D runbook are `2012-11-09 12:00`, run V4.

```bash
export TILES=/path/to/tiles
mkdir -p "$TILES"

# for each region's (i, j) from step 2:
python -m dbof.cli.generate_tile \
    --i <I> --j <J> --timestamp '2012-05-16 06:00:00' \
    --property density --output "$TILES"

python -m dbof.cli.generate_tile \
    --i <I> --j <J> --timestamp '2012-05-16 06:00:00' \
    --property Ri --output "$TILES"
```

Then confirm which registry entries are actually 3-D, and prune
`config.TILE_FIELDS_3D` to match:

```bash
python - <<'PY'
import xarray as xr, glob
for f in sorted(glob.glob("$TILES/*.nc".replace("$TILES", __import__("os").environ["TILES"]))):
    ds = xr.open_dataset(f)
    v = [k for k in ds.data_vars]
    print(f.split("/")[-1], {k: ds[k].dims for k in v})
PY
```

---

## 4. Switch the provider on

```bash
export FRONTS_APP_DATA=s3
export FRONTS_APP_S3_ROOT=s3://dbof/globals_for_cutouts/v2_2_01
export FRONTS_APP_TILE_DIR=$TILES
export FRONTS_APP_CACHE=~/.cache/fronts-viz
export LLC4320_PREPROC_SRC=/path/to/llc4320-native-grid-preprocessing/src
export DISPLAY=dummy                       # PyVista >= 0.44, see the 3-D runbook

python -m fronts.viz.apps.serve --show
```

Page 2 works as soon as the tiles exist. Page 1 needs step 1 finished.

---

## 5. Build the display pyramid

The map is drawn from a regular lat/lon raster, because the native grid is
not one (see `INDEX.md`). At synthetic scale the pyramid builds on demand
in under a second. At 12960 × 17280 it will not — build it once, offline:

```bash
python - <<'PY'
from fronts.viz.apps import config
from fronts.viz.apps.common import pyramid, sources

p = sources.get_provider()
for date in p.dates():
    for name in ["gradb2", "__land__", "__fronts__"] + p.field_names(date):
        for width in config.PYRAMID_WIDTHS:
            reduce = "any" if name.startswith("__") else "mean"
            pyramid.level(p, date, name, width, reduce=reduce)
            print(date, name, width, "ok", flush=True)
PY
```

**This is the step to watch.** `pyramid.regrid` does one pass over 224
million points per field per level, in memory. If it is too slow or too
large, the fallbacks in order of preference are: fewer levels
(`config.PYRAMID_WIDTHS`); chunked accumulation with dask; or writing the
pyramid as a multiscale zarr from the preprocessing side instead. Report
what it does and I will pick.

---

## 6. Precompute the global statistics

Statistics are exact at full resolution, so the whole-globe default would
otherwise be a cold multi-minute wait on first load. Warm the cache:

```bash
python - <<'PY'
from fronts.viz.apps.common import sources
from fronts.viz.apps.common.selection import BBox
from fronts.viz.apps.characteristics import stats

p = sources.get_provider()
for date in p.dates():
    for field in p.field_names(date):
        stats.extract_both(p, date, field, BBox.globe())
        print(date, field, "cached", flush=True)
PY
```

---

## 7. Serve it somewhere

`panel serve` is a long-running process; it has to live on a machine
people can reach.

```bash
python -m fronts.viz.apps.serve \
    --address 0.0.0.0 --port 5006 \
    --allow-websocket-origin your.host:5006
```

Behind a reverse proxy, add every hostname users type as an
`--allow-websocket-origin`.

---

## What is still a placeholder in the code

| Where | What |
|---|---|
| `common/sources.py` | `S3Provider` — six methods raise `NotWiredUp`; `tile()` is implemented |
| `common/regions.py` | Six centres are placeholders; `tile_idx` is `None` |
| `config.TILE_FIELDS_3D` | Names verified against the registry; 3-D-ness not yet |
| `tiles/pipeline.py` | `tile_labels` needs the real-store branch; `remap_to_rect` needs the face lookup wired for real tiles |
| `evolution/app.py` | Awaiting the specification |

Everything else runs.

---

## Prerequisites still outstanding

The prototype routes around these rather than changing core modules. They
are R1–R5 in `prompts/build_front_viz_tool.md`:

- **R1** — `dev/mld/density_utils.py` is not an installed package. Page 2
  currently reads tiles directly; the real-data path needs those loaders.
- **R2/R3** — `plot_map_inset` lives in a script that runs a `sys.path`
  hack and `matplotlib.use("Agg")` at import. Page 2 draws its own
  equivalent instead.
- **R4** — the five figure builders write a PNG and close the figure, so
  page 2 round-trips them through a temp directory. Making `output_path`
  optional would let them go straight into a pane.
- **R5** — `fronts/viz/__init__.py` imports `properties`, which imports
  cartopy, so `import fronts.viz.curtains` is heavier than it needs to be.
