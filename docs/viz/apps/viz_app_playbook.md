# Running the front viz app

Six steps, from nothing to a page in the browser.

**1 — S3 access.** The app reads `s3://dbof/` at
`https://s3-west.nrp-nautilus.io`. Credentials come from the normal boto3
chain, so either `~/.aws/credentials` or:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

**2 — Check out the branches.** Both matter, and the preprocessing one is
not optional: the app composes tiles from `tile_utils` internals, and those
differ between branches. On the wrong branch, stored tiles still load and
anything needing generation fails.

```bash
cd llc4320-native-grid-preprocessing && git checkout main
cd ../fronts                         && git checkout viz_tools
```

**3 — Install both, into the same environment.** The app is `fronts`; it
imports `dbof` from the preprocessing repo. One env, both editable:

```bash
conda activate fronts
pip install -e ../llc4320-native-grid-preprocessing
pip install -e .
```

Installing `dbof` is what makes it importable. If you would rather not
install it, `export LLC4320_PREPROC_SRC=/path/to/llc4320-native-grid-preprocessing/src`
instead — the app falls back to that.

**4 — Pin numpy below 2.4.** Datashader needs numba, numba needs
`numpy <= 2.3`. Without it every map falls back to sending raster cells to
the browser, capped at 1.5M, and the maps are slower and coarser.

```bash
pip install "numpy<2.4"
```

**5 — Warm the cache (optional, recommended).** Each global field is one
0.83 GB zarr chunk with no partial reads, so the first view of anything is
a download. Paying for it at a prompt beats paying for it in the browser:

```bash
python -m fronts.viz.apps.warm --data s3
python -m fronts.viz.apps.warm --chunk monterey_bay --data s3   # Evolution
```

**6 — Serve.**

```bash
python -m fronts.viz.apps.serve --data s3 --port 5006
```

Then open <http://localhost:5006/>. Without `--data s3` it runs on
synthetic data, which is useful for checking the pages work and says
nothing about the ocean.

---

## When something looks wrong

| symptom | check |
|---|---|
| `No module named 'dbof'` | wrong conda env, or step 3 not done |
| `...has a different tile_utils API` | wrong preprocessing branch (step 2) |
| blank page, websocket refused | origin mismatch — use the URL the server prints |
| maps slow and coarse | numpy ≥ 2.4, so no datashader (step 4) |
| a field is "unavailable" | that channel was not built for that date |

`python -m fronts.viz.apps.check_s3` reports which stores are reachable and
which build step is missing for the ones that are not.
