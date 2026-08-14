# Runbook — Ri-colored 3-D front viz on a headless cluster

End-to-end recipe for producing the interactive 3-D HTML (isopycnal surfaces
colored by Richardson number) on a headless Linux node with **no X server and
no admin rights**. Pairs with the reference doc
[`fronts_viz_3d.md`](fronts_viz_3d.md) (flags, algorithm) and the tile docs in
the preprocessing repo
([`llc4320-native-grid-preprocessing/docs/Tiles.md`](../../../llc4320-native-grid-preprocessing/docs/Tiles.md)).

Two things have different lifetimes, and conflating them is the usual
stumbling block:

| Thing | Lifetime | Redo when? |
|-------|----------|------------|
| The conda env + OSMesa VTK swap | **persists on disk** | only if you delete/rebuild the env, or a later `conda install` clobbers `vtk` |
| The `export DISPLAY=… / TILES=… / …` shell vars | **per shell session** | every new shell — so keep them in a file you `source` (below) |

---

## 0. One-time: headless rendering env (OSMesa)

The default PyVista/VTK build needs an OpenGL context (X server / GPU). On a
headless node with neither, swap in the **OSMesa** (software, CPU) VTK build.
Do it in a *clone* so the original env stays intact:

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
```

Record what you installed so the version is pinned for future-you:

```bash
pip freeze | grep -i vtk        # e.g. vtk-osmesa==9.3.1
```

(Why this isn't a `pip install -e .` dependency: the right GL backend —
OSMesa / GLX / EGL / Xvfb — is machine-specific, and `vtk-osmesa` is a
conflicting drop-in for `vtk` served from a non-PyPI index. Pinning it in the
package would break every non-headless install. It's a deployment choice, not
a code dependency.)

---

## 1. Per-session env

Save this as `env_fronts_viz.sh` (edit the paths once) and `source` it at the
start of each session — that's all you re-do next week:

```bash
# env_fronts_viz.sh  --  source this, don't execute it
conda activate llcngp_osmesa

# Lets dev/mld/density_utils.py find the preprocessing repo's tile_mapping.
# (Redundant if the preprocessing repo is `pip install -e`'d, but harmless.)
export LLC4320_PREPROC_SRC=/home/lhoffma2/git/llc4320-native-grid-preprocessing/

# Any non-empty value: makes ensure_display() skip the (removed) start_xvfb;
# OSMesa ignores DISPLAY entirely.
export DISPLAY=dummy

# Data locations.
export TILES=/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/V4/20121109_120000/tiles
export LABELS=/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/V4/20121109_120000/labeled_fronts_global_20121109T12_00_00_V4_bin_D.npy
export OUTDIR=/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/fronts_viz
mkdir -p "$OUTDIR"
```

```bash
source env_fronts_viz.sh
```

---

## 2. Generate the tiles (reads from S3 — needs network)

The viz needs **two** tiles from the same window + timestamp: density drives
the geometry, Ri drives the color. Passing a directory to `--output` uses the
default per-property filename inside it.

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

Any registered property works for `--field-tile` color (see the table in
`Tiles.md`): `relative_vorticity`, `okubo_weiss`, `frontogenesis_tendency`,
`N2`, … — just generate that tile with `--property <name>`.

> **Channel names changed** with the `tiles-depth-fields` merge in
> `llc4320-native-grid-preprocessing`. Only `temperature` and `salinity` kept
> aliases; the rest now error out. The renames that matter here:
> `richardson`→`Ri`, `vorticity`→`relative_vorticity`, `strain`→`strain_mag`
> (also `strain_n`/`strain_s`), `frontogenesis`→`frontogenesis_tendency`,
> `rossby`→`rossby_number`, `froude`→`Fr`, `burger`→`Bu`,
> `vertical_buoyancy_flux`→`wB`. Default filename prefixes are lowercase now
> (`ri_tile…`, `ftend_tile…`, `okuboweiss_tile…`), and derivative fields carry
> a 1–3 cell NaN rim (`edge_margin`) at the tile boundary.

Note that only **3-D** tiles work as `--field-tile`; the registry's
inherently-2-D channels (`mixed_layer_depth`, `Eta`, `oceTAUX`, …) have no `Z`
coord and are rejected by the loader.

---

## 3. Render the interactive HTML

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
Ri ≤ 0 or NaN (land, the 1-px tile edge rim, zero-shear) render neutral gray.

---

## 4. Configuring the colormap / scaling

Per run (overrides the registered style):

```bash
--cmap-volume RdBu_r        # any matplotlib / cmocean name
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
scalar-bar title for every Ri render. Diverging fields (vorticity, OW, …) set
`center=0.0` for symmetric limits.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `module 'pyvista' has no attribute 'start_xvfb'` | No `$DISPLAY` and PyVista ≥ 0.44. Set `export DISPLAY=dummy` (step 1) — OSMesa needs no real display. |
| `Could not import 'tile_mapping' …` | `LLC4320_PREPROC_SRC` unset and preprocessing repo not installed. Set the var (step 1) or `pip install -e` that repo. |
| `Tile provenance mismatch …` | The density and field tiles are from different windows/timestamps. Regenerate both with the same `--i/--j/--timestamp`. |
| OpenGL / `libGL` error (not a display error) | Env missing Mesa: `conda install -c conda-forge mesalib`. |
| HTML opens but surfaces are mostly gray | Expected where Ri ≤ 0 / NaN; if *everything* is gray, the field tile may be all-NaN — check the QA of the Ri tile. |
