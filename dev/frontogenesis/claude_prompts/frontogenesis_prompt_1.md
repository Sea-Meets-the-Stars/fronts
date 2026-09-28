# Frontogenesis execution prompt 1 — M0: Access and reconnaissance

**Milestone:** M0 (`frontogenesis_coding.md` §6).
**Reads:** `frontogenesis_planning.md`, `frontogenesis_coding.md`.
**Goal:** prove we can read the data, and settle every open *empirical* question **before**
any physics is written. Nothing here computes a result; everything here prevents a wrong one.

---

## Context you need

Read `frontogenesis_planning.md` §1-§5 and `frontogenesis_coding.md` §1-§4 first. The short
version: we are comparing the measured strengthening of surface buoyancy fronts in LLC4320
against the kinematic frontogenesis tendency, in the California Current, over 72 hourly steps.
The predicted side already exists in `dbof`; the measured side does not.

**Branches.** Work against `tiles-surface-only` (llc4320-native-grid-preprocessing) and
`viz_tools` (fronts). Both are feature branches and both are usable today; PR #24 is still
open. Do **not** work from `llc4320_v2` — it is ~100 commits behind and has the old file
naming (`calculate_additional_fields.py` rather than `calculate_fields.py`).

**Route.** We use `dbof` as a *library*, not through the `generate-tile` CLI, and we compute
both sides of the comparison ourselves so they share one gradient stencil and one filter
(planning §5.1). No modification to `tiles-surface-only` is required.

---

## Tasks

### 1. Environment

Create a dedicated env. **Prefer Python 3.13**; try 3.14 only if `xgcm>=0.10` and `scikit-fmm`
wheels exist for it, and fall back to 3.13 the moment they do not. Install `dbof` from `tiles-surface-only` with **`pip install -e . --no-deps`**:
its `torch==2.8.0` / `timm==0.3.2` pins are reachable only through
`cutout_dataset_creation/dask_pipeline.py` and `spatial_cutouts.py`, which we never import, and
a plain install would downgrade torch underneath `fronts`.

Then add: **`xgcm>=0.10`** (`set_xgcm_grid` passes `padding='fill'`, which is 0.10+ only; corrected 2026-09-28 — the original `<0.10` pin was wrong), `scikit-fmm`,
`s3fs`, `ujson`, `xmitgcm`, `zarr`, `dask`, `h5netcdf`, `matplotlib`. `kerchunk` the package is
**not** needed — the OSN refs are read through fsspec's built-in `reference://` filesystem.

Record the resolved versions in the log.

### 2. First contact with the data

Write `dev/frontogenesis/py/osn_tiles.py` far enough to load **one** timestamp:

```python
tile   = rect_ij_to_tile(13320, 9720)          # -> face 10, j 0:720, i 2880:3600
g      = process_llc4320_grid(get_remote_gridfile(EP))
g_tile = ensure_comodo_attrs(g.isel(face=[tile.face_idx], **_tile_indexer(g, tile)).compute())
grid   = set_xgcm_grid(g_tile, use_connections=False)
ds     = get_remote_llc_data(EP, osn_date_to_iteration('2012-07-02 00:00:00'), [tile.face_idx])
```

`EP = "https://mghp.osn.xsede.org"`. See `frontogenesis_coding.md` §2 for exact signatures —
**use them, do not guess.** Note the three traps recorded there, especially that
`process_llc4320_grid` calls `reset_coords()` and can drop the comodo attrs, so
`ensure_comodo_attrs` (public, `dbof.llc4320_ingestion.grid`) must run before `set_xgcm_grid`.

Also load one timestamp from the second OSN store via `get_remote_llc_wind_data` (`KPPhbl`,
`oceTAUX`, `oceTAUY`).

### 3. Answer these five questions empirically, and record every answer in the log

These are currently assumptions. Each is a one-liner against real data.

1. **Is OSN land stored as 0 or NaN?** This decides how much the halo mask has to do. MITgcm
   convention is 0, which would make `b(Theta=0, Salt=0)` finite and poison every ocean cell
   adjacent to coast — but it is unverified.
2. **Is `W[k_l=0]` ~ 0?** This is the empirical basis for planning §2.2 — the claim that the
   tilting term vanishes at the surface and surface-only data is therefore *sufficient*, not a
   compromise. Verify it rather than citing the argument.
3. **Actual `dxC`/`dyC` near 37N.** Expected ~1.8-2.3 km. The halo width in km and the
   displacement-per-hour estimate both depend on this, so read it rather than assuming.
4. **Size of the rotation terms.** Compute `grad CS` and `grad SN` across the tile and compare
   `u * |grad CS|` against a typical strain rate (~1e-5 s^-1). Planning §5.2 claims ~0.1%;
   confirm it is below 0.5%. Also check the spherical metric term `u tan(phi)/a`.
5. **The tracer advection scheme** from the model configuration, and an order-of-magnitude
   `kappa_num`. This matters because implicit numerical diffusion is the leading alternative
   explanation for any slope below 1 (planning §2.3).

**`drF` is deliberately not on this list.** The OSN gridfile carries the top-cell scalars
`drF = 1.0`, `Z = -0.5`, `Zl = 0.0` as 0-d coordinates (found in task 2; `process_llc4320_grid`
drops them, so `load_grid` re-attaches them — corrected 2026-09-28); M2 cross-checks them
against the chunk store's 3-D grid.

**Status 2026-09-28: tasks 1-3 done** (log entries of 2026-09-27/28). The five answers: land is
**NaN**; `W[k_l=0]` is **`dEta/dt`**, not ~0; spacing **1.71 x 1.85 km** at 37N with face 10
rotated (`i` meridional/southward, `j` zonal); rotation terms **identically zero**, metric term
0.1%; scheme **OS7MP** (`tempAdvScheme = 7`). Planning §2.2/§2.3/§4/§5.2/§5.5 and coding
§3.1/§4.6 were corrected accordingly; tasks 4-5 below follow the corrected versions.

### 4. Write `tile330_grid.zarr`, and pull two hours

- **The static grid (§3.1).** One pull, reused by everything. M1's V2 and V6 tests need it on
  disk, which is why it belongs here rather than in M2. Per the corrected §3.1 it must also
  carry **`hFacW`, `hFacS`** (from the raw gridfile — they are the `U`/`V` land masks and
  `process_llc4320_grid` drops them), the 0-d **`drF`, `Z`, `Zl`**, and the orientation /
  spacing / `land_fill='NaN'` attrs.
- **Two consecutive timestamps**, not one: M1's gate 3 has a variant driven by real LLC
  velocities and therefore needs a midpoint velocity.

### 5. QA plot

One snapshot: `Theta`, `G = |grad b|^2`, and the land mask from `hFacC`. **No halo** — that is
M1. Question 1 answered "NaN" (corrected 2026-09-28), so there is **no coastal gradient ribbon
to see**; instead show the stencil's own NaN rim along the coast (`G` undefined 1 cell from
land, the Jacobian 2 cells — the original "~3" was an over-estimate; corrected 2026-09-28, M0
task 5), overlay `isfinite(Theta)` against `hFacC > 0` to confirm they coincide, and show the
invalid rim on the tile edges, which arises because `_tile_indexer` gives the staggered dims
the same slice as the centred ones (found to be on **all four** edges and finite, not NaN,
since xgcm pads with 0 — the original brief said "high edges"; corrected 2026-09-28, M0
task 5). Write to `dev/frontogenesis/figs/`.
**Done 2026-09-28:** `figs/m0_qa_tile330_20120702T00.png`, `py/m0_qa_plot.py`,
`py/m0_qa_checks.py`; M0 acceptance audited in the task-5 log entry, all criteria PASS.

---

## Acceptance criteria

- **Two consecutive hours** load end to end, from both OSN stores, in a reproducible env.
- `tile330_grid.zarr` written (§3.1).
- **All five questions answered in the log**, with numbers.
- **All four §2 traps confirmed against real data**: comodo attrs survive (or are restored after)
  `process_llc4320_grid`; `calculate_jacobian`'s `u_x, v_y` arguments really are `U, V`;
  `halo_mask.py:75` is not reached; and the two timestamp formats convert correctly.
- QA plot written and the coastline inspected. A stencil NaN rim is expected; **a gradient
  ribbon is not** (land is NaN) — if one appears, something upstream has filled NaN with 0.

## Do not

- Do not write any physics yet (no `operators.py`, no `semilag.py`, no `masking.py`). That is
  M1, and it is gated on knowing the answers above. The QA plot needs only a bare
  `hFacC > 0` land mask, not the halo.
- Do not pull the 72-hour series yet. That is M2.
- Do not modify `tiles-surface-only` or any file in either repo outside `dev/frontogenesis/`.

## Log

Append to `frontogenesis_prompts.md` under `## Logs`: the env spec with resolved versions, the
six answers with numbers, anything that contradicted the planning doc, and the QA plot path.
**If any answer contradicts the planning doc, say so explicitly** — that is the most valuable
output of this milestone. Four planning-doc claims have already been overturned during planning;
a fifth found here is a success, not a setback.
