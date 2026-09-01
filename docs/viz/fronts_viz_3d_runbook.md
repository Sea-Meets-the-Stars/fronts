# Runbook — Ri-colored front viz on a headless cluster

End-to-end recipe for the 3-D interactive HTML (isopycnal surfaces colored by
Richardson number) and the 2-D curtain figures, on a headless Linux node with
**no X server and no admin rights**.

Reference docs: [`fronts_viz_3d.md`](fronts_viz_3d.md) and
[`fronts_curtain.md`](fronts_curtain.md) for flags and algorithms;
[`llc4320-native-grid-preprocessing/docs/Tiles.md`](../../../llc4320-native-grid-preprocessing/docs/Tiles.md)
for how tiles are built.

Step 0 persists on disk (redo only if you rebuild the env). Step 1 is
per-shell.

---

## 0. One-time: headless rendering env (OSMesa)

PyVista/VTK needs an OpenGL context. With no X server and no GPU, swap in the
**OSMesa** (software, CPU) VTK build — in a *clone*, so the original env stays
intact:

```bash
conda create --name llcngp_osmesa --clone llcngp
conda activate llcngp_osmesa

# Match the OSMesa wheel to the VTK version PyVista already expects:
python -c "import vtk; print(vtk.VTK_VERSION)"        # note e.g. 9.3.1

pip uninstall -y vtk           # remove a pip-installed vtk, if any
conda remove --force -y vtk    # remove a conda-installed vtk without pulling pyvista
pip install --extra-index-url https://wheels.vtk.org "vtk-osmesa==<VERSION_ABOVE>"

# sanity check — should print a byte count, no display error:
python -c "import pyvista as pv; pv.OFF_SCREEN=True; p=pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere()); print('ok', len(p.screenshot(return_img=True)))"

pip freeze | grep -i vtk       # record the pin, e.g. vtk-osmesa==9.3.1
```

`vtk-osmesa` is deliberately **not** a `pip install -e .` dependency: the right
GL backend is machine-specific, and it's a conflicting drop-in for `vtk` from a
non-PyPI index.

---

## 1. Per-session env

Save as `env_fronts_viz.sh` (edit the paths once) and `source` it each session:

```bash
# env_fronts_viz.sh  --  source this, don't execute it
conda activate llcngp_osmesa

# Lets dev/mld/density_utils.py find the preprocessing repo's tile_mapping.
# (Redundant if that repo is `pip install -e`'d, but harmless.)
export LLC4320_PREPROC_SRC=/home/lhoffma2/git/llc4320-native-grid-preprocessing/

# Any non-empty value: makes ensure_display() skip the (removed) start_xvfb.
# OSMesa ignores DISPLAY entirely.
export DISPLAY=dummy

# Data locations.
export TILES=/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/V4/20121109_120000/tiles
export LABELS=/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/V4/20121109_120000/labeled_fronts_global_20121109T12_00_00_V4_bin_D.npy
export OUTDIR=/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/fronts_viz
mkdir -p "$OUTDIR"
```

---

## 2. Generate the tiles (reads from S3 — needs network)

Both viz scripts need **two** tiles from the same window + timestamp: density
drives the geometry/isopycnals, a second field drives the color. Passing a
directory to `--output` uses the default per-property filename inside it.

```bash
# geometry tile (sigma0) — skip if it already exists in $TILES
python -m dbof.cli.generate_tile \
    --i 13142 --j 9956 --timestamp '2012-11-09 12:00:00' \
    --property density --output "$TILES"
# -> $TILES/density_tile330_20121109T12.nc

# color tile (Ri)
python -m dbof.cli.generate_tile \
    --i 13142 --j 9956 --timestamp '2012-11-09 12:00:00' \
    --property Ri --output "$TILES"
# -> $TILES/ri_tile330_20121109T12.nc
```

**Choosing `--property`.** The registry is the list — `TILE_PROPERTIES` in
`dbof/tiles/field_registry.py`. To see every accepted name without opening the
file:

```bash
python -m dbof.cli.generate_tile --help     # argparse prints the full choices list
```

Any **3-D** channel works as `--field-tile` color: `Ri`, `N2`,
`relative_vorticity`, `okubo_weiss`, `frontogenesis_tendency`, `wB`, … The
inherently-2-D channels (`mixed_layer_depth`, `Eta`, `oceTAUX`, …) have no `Z`
coord and are rejected by the loader.

---

## 3. Render the 3-D interactive HTML

```bash
python -m fronts.scripts.fronts_viz_3d \
    --density-tile "$TILES/density_tile330_20121109T12.nc" \
    --field-tile   "$TILES/ri_tile330_20121109T12.nc" \
    --labels       "$LABELS" \
    --i 13142 --j 9956 --zscale 1.0 \
    --output           "$OUTDIR/fronts_viz_3d_calcurrent_Ri.png" \
    --interactive-html "$OUTDIR/fronts_viz_3d_calcurrent_Ri.html"
```

Writes the interactive HTML, a 3-D PNG, and a 2-D inset PNG into `$OUTDIR`.
Geometry = isopycnals (σ₀); color = `log10(clip(Ri, 1e-2, 1e4))`. Cells where
Ri ≤ 0 or NaN (land, the tile-edge rim, zero-shear) render neutral gray.

---

## 4. Render the 2-D curtains

Same two tiles, same locator. `--isopycnal-curtain` adds the flattened
density-surface figure on top of the three standard curtains:

```bash
python -m fronts.scripts.fronts_viz_curtain \
    --density-tile "$TILES/density_tile330_20121109T12.nc" \
    --field-tile   "$TILES/ri_tile330_20121109T12.nc" \
    --labels       "$LABELS" \
    --i 13142 --j 9956 \
    --n-offsets 3 --perp-half-width 30 \
    --isopycnal-curtain \
    --output-prefix "$OUTDIR/calcurrent_curtain"
```

Writes `{prefix}_{field}_{loc}_…png` for `mainaxis`, `offsets_n{N}`, `perp`,
`isopycnal` and `inset`, where `{loc}` is the front's `lat…_lon…` — so
different fronts don't overwrite each other. Needs no OSMesa (matplotlib only).

Add `--list-perp-candidates` to log each main-axis column with its `(i, j)` and
crossing count before rendering, then pass the one you want as `--perp-point`.

---

## 5. Configuring the colormap / scaling

Per run (overrides the registered style):

```bash
--cmap-volume RdBu_r        # 3-D only; any matplotlib / cmocean name
--clim -1 2                 # color limits, in the *transformed* (log10) space
--field-clip 1e-3 1e3       # raw-Ri clip before the transform
--field-transform symlog    # log10 | symlog | linear
```

Persistent default — edit the `"Ri"` entry of
[`fronts/viz/field_styles.py`](../../fronts/viz/field_styles.py):

```python
"Ri": FieldStyle(
    transform="log10", clip=(1e-2, 1e4), cmap="RdYlBu",
    title="log10(Ri)", clim=(-1.0, 2.0),
),
```

That one entry controls colormap, color limits, transform, raw clip, and the
scalar-bar title for every Ri render, in both scripts. Keys must match the
tile's variable name (= `out_name` in the preprocessing registry). Diverging
fields (`relative_vorticity`, `okubo_weiss`, …) set `center=0.0` for symmetric
limits.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `module 'pyvista' has no attribute 'start_xvfb'` | No `$DISPLAY` and PyVista ≥ 0.44. Set `export DISPLAY=dummy` (step 1) — OSMesa needs no real display. |
| `Could not import 'tile_mapping' …` | `LLC4320_PREPROC_SRC` unset and preprocessing repo not installed. Set the var (step 1) or `pip install -e` that repo. |
| `Tile provenance mismatch …` | The density and field tiles are from different windows/timestamps. Regenerate both with the same `--i/--j/--timestamp`. |
| `No FIELD_STYLES entry for …` | The tile's variable name has no style row. Add one to `fronts/viz/field_styles.py` (step 5); the render still works, with a linear/percentile fallback. |
| OpenGL / `libGL` error (not a display error) | Env missing Mesa: `conda install -c conda-forge mesalib`. |
| Surfaces / curtains mostly gray | Expected where Ri ≤ 0 / NaN. If *everything* is gray, the field tile may be all-NaN — check its QA plot. |
