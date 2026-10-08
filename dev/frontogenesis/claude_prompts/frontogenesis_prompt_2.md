# Frontogenesis execution prompt 2 — M1: Operators and validation

**Milestone:** M1 (`frontogenesis_coding.md` §6). **THIS IS A HARD GATE.**
**Prerequisite:** M0 complete — its five answers in the log, and `tile330_grid.zarr` on disk.
**Goal:** operators that are *known* correct, not assumed correct — and figures that make each
methodological choice visible.

> **Do not proceed to M3 on a failure of gate 3 below.** C-grid interpolation attenuation alone
> can bias the headline slope by 0.7-1.4 with no physics involved. Everything downstream
> inherits that silently.

---

## Modules to write

This section is the **contract** for each module. **Tasks** (below) gives the order to build
them in and what each session must produce.

All in `dev/frontogenesis/py/`. Signatures are the contract — see
`frontogenesis_coding.md` §4.2-§4.5, §4.9. Style: **methods, not classes**; reuse existing code;
inline comments that explain the *physics*.

### `masking.py`

`ocean_mask`, `halo_mask`, `coast_distance_km`, `offshore_mask`, `analysis_mask`.

**`tile330_masks.nc` (§3.5) is written here**, from M0's static grid — not in M2. That is what
keeps M2 independent of M1 so the two can run in parallel.

- **Our API takes `halo_cells`; the helper takes km.** `generate_halo_land_mask(ds_grid,
  target_km_res, ...)` uses `target_km_res` directly as `halo_km`, so `halo_mask` converts
  internally: `halo_km = halo_cells * median(dxC)`. Requirement is **7 cells** (3 for the
  Jacobian+interp stencil — the measured reach is 1 cell for `G`, 2 for the Jacobian/`F`, so
  one cell conservative (M0 task 5); 4 for the widest filter half-width) — roughly 12-14 km
  across the tile (spacing 1.7-2.1 km; corrected 2026-09-28, M0 task 3). Use the **measured**
  `dxC` from M0; never hard-code the km value. Land is already NaN in the OSN fields (planning
  §5.5), so `ocean_mask` must equal `isfinite(Theta)` cell for cell — assert it in
  `test_masking.py` (M0 task 5 found 0 mismatches in 518,400 cells).
- **Add the tile-edge margin: `edge_mask(grid_ds, edge_cells=7)` and an `edge_cells=7`
  argument on `analysis_mask`** (coding §4.2; added 2026-09-28, M0 task 5). The `_tile_indexer`
  rim is on **all four** tile edges and is **finite, not NaN** — xgcm `padding='fill'` pads the
  missing high staggered point with 0, so `G` is ~1e6x wrong on the low edges (`j = 0`,
  `i = 2880`) and ~0.5x on the high edges (`j = 719`, `i = 3599`); the Jacobian is wrong 1 cell
  deep on the low edges and 2 on the high (crop test, M0 task 5). The land halo cannot see it
  (`skfmm` measures distance from `hFacC == 0`) and the offshore cut leaves the open-ocean west
  and north edges alone. Minimum 2 cells for the raw operators; 7 with filter support. Test it
  in `test_masking.py` against `py/m0_qa_checks.check_edge_rim` (the crop test) and show it in V6.
- **Handle the two known defects.** `halo_mask.llc_native_grid_halo_mask` returns a 2-D array
  early when a face is entirely land (`halo_mask.py:74-75`), silently and with inverted
  convention; and a `k`-carrying `hFacC` makes the mask 4-D and breaks `skfmm`. Collapse `k`
  first and **assert the output shape**.
- Convention: **`True` = retained**, land = `False`, plain numpy.
- Offshore cut: `>= 100 km` for primary statistics. This should also exclude the head of the
  Gulf of California, which the tile's eastern edge clips — **verify that it does** rather than
  adding a special-case polygon.

### `operators.py` — the single shared operator

`buoyancy`, `lowpass`, `grad_b`, `gradb2`, `jacobian`, `frontogenesis`, `strain_divergence`,
`strain_alignment`.

This module is why the study is trustworthy: **both sides of the comparison go through it.**

- `buoyancy` wraps `calculate_fields.buoyancy_of_field` (**JMD95**, the EOS the model actually
  advected — not TEOS-10/`gsw`). Note `b = +g*sigma0/rho0` with `g=9.81`, `rho0=1000.0`: it
  *increases with density*, the negative of the textbook definition. Harmless (`G` and `F` are
  quadratic; alignment enters as `cos 2theta`) — **do not "fix" it.** Do not use
  `utils/physical_calculations.buoyancy_of_field`, which is legacy (g in km/s^2, rho_ref=1025).
- **`gradb2` is `b_x^2 + b_y^2` from the same `grad_b` (`calculate_native_gradient_tracer`)
  that feeds `frontogenesis`** — not the repo's `grad_b2` / `calculate_grad_squared_tracer`,
  which squares on the staggered points before interpolating and differs by **0.911x** in the
  interior median on tile 330 (M0 task 5). The discrete identity `F = ½ DG/Dt` only holds when
  `G` and `F` share `b_x, b_y`; mixing the stencils starts V3 ~0.9 off. The repo's `gradb2`
  stays for front *finding* (M4) only (corrected 2026-09-28, M0 task 5).
- **Assert output dims after every dbof operator call** (`('face', 'j', 'i')`, or the staggered
  pair). A centred mask times a staggered field, or `calculate_jacobian` with its arguments
  swapped, silently broadcasts to 4-D on numpy-backed inputs and is OOM-killed; only dask-backed
  inputs raise (M0 tasks 4-5). `calculate_jacobian(u_x, v_y)` takes the raw staggered `U, V`
  — confirmed bit-for-bit against a numpy replica (M0 task 5, `py/m0_qa_checks.py`).
- `frontogenesis` takes **already-filtered** inputs. Filtering happens once, at the call site,
  so it cannot silently differ between the two sides.
- `strain_divergence`: `calculate_native_strain_vorticity` returns a **dict**, and shear strain
  and vorticity live on **corners** `(j_g, i_g)`. Interpolate to centres before combining.
- Basis: `grad b` and the Jacobian both come from the repo helpers, which both rotate to
  geographic. What matters is that they are in the *same* basis — they are.

### `semilag.py`

`centre_velocities`, `departure_index`, `interp_to_departure`, `measured_DGDt`, `eulerian_DGDt`.

- **`measured_DGDt` interpolates `b`, then differentiates.** It does *not* interpolate `G`.
  Bilinear interpolation of `G` at a half-cell offset errs by `dx^2 G_xx/8` — ~5.5% of `G` for a
  front ~1.5 cells wide, and **systematically negative at maxima**, i.e. it *fabricates*
  frontogenesis. Against a per-hour signal of only 7-20%, that is 25-80% of the signal.
  `order >= 3` is not optional.
- Departure points stay in **native index space**: `di = U dt/dxC`, `dj = V dt/dyC`. No rotation.
- Evaluate `F` and select fronts at the **trajectory midpoint time**, formed as
  `0.5*(f_t + f_tp1)`. M2 strain rotates ~29 degrees per hour, so using an endpoint both adds
  noise and correlates that noise with the measured side.
- `eulerian_DGDt` is the independent cross-check, not the primary estimate.

### `coarsegrain.py`

`subfilter_flux`, `subfilter_term`. **This module is M1's, not M3's** — demonstrating that `tau`
closes the coarse-grained budget is part of validating the operators. Without it the filter sweep
is uninterpretable: the subfilter term is a third derivative of the subfilter flux, `O(1)` at
every `L`, and as `L -> dx` it *becomes* the numerical-diffusion term rather than vanishing
(planning §5.4).

### `validate.py` — four gates plus two supporting figures (six PNGs, V1-V6)

Gates: `test_cartesian_deformation` (V1), `test_native_metric` (V2), `test_discrete_null` (V3),
`test_interpolation_bias` (V4). Supporting: `demo_interp_half_cell` (V5), `qa_land_halo` (V6).

V2 and V6 need the real tile grid from `tile330_grid.zarr`, so they are not pure-offline — mark
them `@pytest.mark.needs_grid`. Everything else runs offline on synthetic fields.

---

## Tasks

Run **one task per session**, in order, as in M0, with one log entry per task (see **Log**).
Each task finishes with its own tests passing. Tests go in `dev/frontogenesis/py/tests/`
(coding §5) and run offline, except those marked `needs_grid`. `validate.py` grows across
tasks: each V-function is written in the task whose module it exercises. The dependency chain
is masking → operators → semilag → validation; `coarsegrain` needs only `operators`.

**Status 2026-09-30: M1 CLOSED — tasks 1-7 done, plus 6b and 7a** (log entries "Execution prompt 2,
task 1" to "task 7"); task 8 (slides) not started. **The task-7 acceptance audit passes all seven
criteria**: V1 0.78% at the 8 dx reference width (M1-Q5); V2 0.077% / 0.041%; V3 1.004 [0.995, 1.017]
(strain) / 0.981 [0.970, 0.994] (llc), the Figure 2 baseline (M1-Q4); V4 recorded as 0.28-1.0% of `G`/h
at order 3 (M1-Q6); V3b (task 6b, `py/fvadvect.py`, `figs/V3b_fv_null.png`) recorded as the
model-advection systematic 0.975 [0.954, 1.003] — the Jacobian attenuation does not bias the slope
(M1-Q2); all seven PNGs in `git status`; suite **84 passed, 3 xfailed** (the xfails are strict and
document three `fronts` bugs found by task 7a's `test_nan_finding.py`); the criterion-7 oracle is
`form='chain'`, bit-for-bit (M1-Q7). Task 7 also applied M1-Q1..Q8 to the docs (each spot marked
"decided 2026-09-30") and fixed the clipped titles / legend in V3b's figure without changing a number.
Open issues carried to M2-M4 are listed in the task-7 log entry (the `fronts` bugs and their fixes,
the non-existent `tile_find` path in coding §2.5 / prompt 5, the caller-side NaN-finding recipe, M3's
both-forms and order-5 requirements, the V3b band). M0 is closed (prompt 1, task-5 log entry). **Task 6 delivered gate V3
(`validate.test_discrete_null`, `figs/V3_discrete_null.png`, `test_validate.py` V3 strain + V3 llc;
suite 70 passed, 0 skipped): PASS at 1.004 [0.995, 1.017] (strain, 15 pooled cases, 2-8 dx) and
0.981 [0.970, 0.994] (llc, hour-0 `b`, real midpoint velocity, `mask_analysis`, n 26,293)** — after a
first attempt of 0.950 / 0.758. The finding: the chain-rule product `-(grad b)^T (grad u)(grad b)`
fails on the grid by `(2/3)(dx/ell)^2`; `operators.frontogenesis` now defaults to the discretely
consistent `form='discrete'` (`-sum_k (L_k b)[L_k, u.grad] b`, 4th-order neighbour gradient), with
`form='chain'` kept bit-for-bit as the criterion-7 oracle, and `semilag.departure_index` uses a cubic
velocity (`vel_order=3`; bilinear made the departure strain 0.965 of the Jacobian's). The 0.85
Jacobian attenuation is invisible to a semi-Lagrangian null by construction (both sides see
`D_h u_c`) — an open systematic for M3. Task 4 delivered
`py/coarsegrain.py` (`subfilter_flux`, `subfilter_term` per §4.5, plus `subfilter_bdelta` — the
dilatation part the divergent surface flow needs — `subfilter_advection`, `flux_divergence`,
`b_at_velocity_points`, `filt`) and `py/tests/test_coarsegrain.py` (10 pass; suite 63). The term
is returned in **F units** (`subfilter = 2 * term` in the M3 budget); `u b` is formed on the
staggered velocity points (flux form); Germano holds to round-off (4e-16) for the composite
filter; the coarse-grained `Gbar` budget closes on exact advection solutions to 4% (shear) / 9%
(divergent) rms at 900 m and not without the term (47-85%), the `bbar` budget to 0.9-1.4%
(∝ dx²); on hour 0 the term is 0.30 / 0.50 / 0.70 of `Fbar` in rms at `L = 2 / 4 / 8`, and the
flux form without `tau_delta` overstates it 2.2x. Task 1 delivered `py/masking.py`,
`data/tile330_masks.nc` (ocean 356,877 → halo 341,960 at 12.57 km = 7 x median `dxC` → offshore
273,431 → analysis 262,925), `py/tests/test_masking.py` (17 pass), `validate.qa_land_halo` →
`figs/V6_land_halo_tile330.png`. The Gulf of California is its own ocean component in the tile
(9,891 cells), max coast distance 73.9 km: the ≥ 100 km cut removes all of it, no polygon.
Task 2 delivered `py/operators.py` (the eight contract functions plus `strain_from_jacobian`
and the dims guards) and `py/tests/test_operators.py` (21 pass; suite 38). Criterion 7:
unfiltered `operators.frontogenesis` is **bit-for-bit** equal to `frontogenesis_tendency` on
both M0 hours (max |dF| = 0 on all 352,673 finite cells); `gradb2`/`grad_b2` interior median
**0.9106** (M0's cell set), 0.9098 on the analysis mask. `lowpass` is a top-hat of half-width
`L/2` that **propagates NaN** (no renormalisation), so `halo_cells = 7` stands: `F` at `L = 8`
is finite on every analysis-mask cell (NaN only in 249 coastal `mask_halo` cells at chessboard 5).
Task 3 delivered `py/semilag.py` (the five contract functions plus `midpoint_time`,
`grad_b_at_departure`, `gradb2_at_departure`), `py/tests/test_semilag.py` (15 pass; suite 53) and
`validate.demo_interp_half_cell` → `figs/V5_interp_half_cell.png`. Interpolation is local Lagrange
(order 1/3/5), not `map_coordinates` (its prefilter leaks a filled NaN 46%/12%/3.3% at 1/2/3 nodes);
`G(x_d)` is `b` interpolated onto the five-point departure stencil with the displacement held
fixed, then the `grad_b` stencil (bit-for-bit `operators.gradb2` at zero displacement) — **not** the
gradient of the shifted field, which measures the residual (~0 under pure strain). Half-cell bias at
the maximum of a 1.5-cell front: order 1 **−5.0%** (bilinear `G` −4.9%, prediction −5.56%), order 3
**−0.54%**, order 5 −0.10%. Real hours: measured and Eulerian finite on all 262,925 analysis cells,
corr 0.74, slope 0.73; order 1 fabricates +5.2% of `G` per hour on front pixels.
Task 5 delivered V1, V2, V4 (`figs/V1_cartesian_deformation.png`, `V2_native_metric.png`,
`V4_interpolation_bias.png`), `py/tests/test_validate.py` (5 pass + the V3 slot skipped; suite 68 + 1
skip) and the split of `validate.py` into `validate.py` (numbers), `validate_figs.py` (the PNGs) and
`synthetic.py` (grids and exact solutions; `test_operators.py` now imports its grid helpers from there).
**V1 PASS**: `G` along parcels over 8 chained hours vs `exp(2at)` to **0.78%** at `ell = 8 dx`, both
orientations bit-identical; the one-step growth-rate error is the centred stencil's truncation, order
1.9 in `dx/ell` (0.65 / 1.1 / 2.4 / 4.2 / 9.0% rms at 8 / 6 / 4 / 3 / 2 dx), while the semi-Lagrangian
step alone is < 0.36% at every width. **V2 PASS**: `b_x` max 0.077%, `b_y` max 0.041% on
`mask_analysis` (metric alone 0.012%; components swapped would be 87%; the grid implies R = 6370.0 km).
**V4 recorded**: headline error bar **0.28% of `G` per hour** = rms over front pixels of `DGDt dt/G`
for a 1.5-cell front at order 3 over the real-hour cross-front displacement distribution (median 0.36,
p99 1.25 cells, isotropic; 0.33% all-cross-front), i.e. 1.4-4.0% of the 7-20% signal; order 1 2.3%,
order 5 0.06%; half cell 0.36% rms / +0.54% at the maximum; falls as `sigma_G^-3.8`, so **1.0% at a
1-cell front** — quote the bar with its width.

### 1. `masking.py`, `tile330_masks.nc`, and V6

- Write `ocean_mask`, `halo_mask`, `coast_distance_km`, `offshore_mask`, `edge_mask`,
  `analysis_mask` per the `masking.py` contract above and coding §4.2. The requirements are
  halo `halo_cells=7` converted with the **measured** `dxC`, `edge_cells=7`, `True` = retained,
  `k` collapsed, and the output shape asserted.
- Write **`tile330_masks.nc`** (coding §3.5: `mask_ocean`, `mask_halo`, `mask_offshore`,
  `mask_edge`, `mask_analysis`, `coast_distance_km`) from `tile330_grid.zarr`. M2 depends on
  this file, so it comes first.
- **Verify** that the `>= 100 km` offshore cut removes the head of the Gulf of California.
  Record the result in the log. Do not add a polygon.
- `test_masking.py`: halo width; both `halo_mask` defects (the all-land-face early return and the
  4-D `k` case); the `True` = retained convention; `ocean_mask == isfinite(Theta)` cell for cell;
  the edge margin covers the rim found by `m0_qa_checks.check_edge_rim`.
- `validate.qa_land_halo` → **V6** (coastline before/after the halo, plus the finite tile-edge
  rim and the `edge_cells` margin that removes it). `@pytest.mark.needs_grid`.

*Discharges:* criterion 5 (V6), criterion 6 (`test_masking.py`).

### 2. `operators.py` and the regression oracle

- Write `buoyancy`, `lowpass`, `grad_b`, `gradb2`, `jacobian`, `frontogenesis`,
  `strain_divergence`, `strain_alignment` per the `operators.py` contract above and coding §4.3.
  The traps are JMD95 with the sign left as is, `gradb2` from **the same** `grad_b` as `F`,
  **dims asserted after every dbof call**, `U, V` staggered into `calculate_jacobian`, corner
  quantities interpolated to centres, and `frontogenesis` taking pre-filtered inputs.
- `test_operators.py`: the gradient of an analytic field; the factor-of-two convention
  (`F = ½ DG/Dt`); a dims check that a mis-staggered call raises rather than broadcasting to 4-D.
- **Regression check (criterion 7):** unfiltered `operators.frontogenesis` against
  `calculate_fields.frontogenesis_tendency` on M0's first hour, to round-off. Use it as a test
  oracle only. Also record the `gradb2` / `grad_b2` interior-median ratio and confirm it is still
  **0.911x**.

*Discharges:* criterion 6 (`test_operators.py`), criterion 7.

### 3. `semilag.py` and V5

- Write `centre_velocities`, `departure_index`, `interp_to_departure`, `measured_DGDt`,
  `eulerian_DGDt` per the `semilag.py` contract above and coding §4.4. The rules are:
  interpolate `b` and then differentiate, never `G`; `order >= 3`; departures in native index
  space; `F` taken at the trajectory midpoint.
- `test_semilag.py`: the zero-velocity identity; uniform-flow translation by a whole cell;
  interpolation order (`order=1` should show the `dx^2 G_xx/8` bias, `order>=3` should not).
- `validate.demo_interp_half_cell` → **V5**: truth vs bilinear-`G` vs cubic-`b` at a half-cell
  shift, with the negative bias at the maximum annotated. This is the figure Lauren asked for,
  and it belongs with the module whose design choice it justifies.

*Discharges:* criterion 5 (V5), criterion 6 (`test_semilag.py`).

### 4. `coarsegrain.py`

- Write `subfilter_flux` and `subfilter_term` per the `coarsegrain.py` contract above and coding
  §4.5, using `operators.lowpass`.
- `test_coarsegrain.py`: `tau → 0` as `L → 0`; Germano consistency; and closure, meaning that
  on a synthetic field the coarse-grained budget closes with the subfilter term included. Note
  that it tends to the numerical-diffusion term as `L → dx`, not to zero (planning §5.4).
- This task is independent of task 3 and may swap order with it.

*Discharges:* criterion 6 (`test_coarsegrain.py`).

### 5. Gates V1, V2, V4

- `test_cartesian_deformation` → **V1**: `G ∝ exp(2 a t)` to **< 1%** (offline).
- `test_native_metric` → **V2**: an analytic `f(XC, YC)` on the real tile grid, gradients to
  **< 1%** (`needs_grid`).
- `test_interpolation_bias` → **V4**: uniform zero-strain flow. **Record the measured bias.** It
  becomes the permanent error bar on every later slope.
- Each writes its PNG and returns a dict of the numbers.

*Discharges:* criteria 1, 2, 4; criterion 5 (V1, V2, V4).

### 6. Gate V3 — the discrete null (**HARD GATE**)

- `test_discrete_null(velocities='strain')` first, then `velocities='llc'`, which takes the
  midpoint velocity from M0's **two** hours. Both must give **slope = 1 ± 0.05 on front pixels**
  and **return the fitted slope**, which Figure 2 draws. Write **V3**.
- Expect the first attempt to fail by roughly the 0.80x Jacobian attenuation measured in M0 task
  5. If it fails, **co-locate the operators or raise the scheme order** until it passes, then
  **re-run tasks 2-5's tests**, since the fix changes `operators.py` and/or `semilag.py`.
- Log the first-attempt slope, every change tried, and the final slope. The change that passes
  is a **finding about the discretisation** and goes in the writeup.
- **If this gate cannot be passed, stop and report.** Do not start M3.

*Discharges:* criterion 3; criterion 5 (V3).

### 6b. V3b — the finite-volume null (recorded bias)

- Added 2026-09-29 per **M1-Q2**, option (a): a second null whose truth is a **flux-form finite-volume
  advection on the C-grid** (`py/fvadvect.py`: centred, DST3, OS7 and an OS7MP-like scheme, MITgcm
  face transports and the advective-form sweep `-[div(u b) - b div u]`), with V3's midpoint velocity
  and otherwise exactly V3's pipeline — `validate.test_fv_null(velocities='strain'|'llc',
  scheme=...)`, **V3b** (`figs/V3b_fv_null.png`). It tests the one link V3 cannot: both of V3's sides
  see the same centred velocity, so the 0.80-0.85x attenuation of the interpolated Jacobian relative to
  the model's flux-form strain (M0 task 5, M1 task 2) is invisible there.
- **A recorded bias, not a gate.** The number it returns (`res['bias']`) is the slope our pipeline gives
  when the tracer is advected as the model advects it; M3 carries it as a systematic. No pass/fail.
- Report: the slope with CI per scheme × form × variant, the slope vs front width, whether the 0.85
  appears and where, and the split between the C-grid stencil (centred FV) and the scheme's implicit
  diffusion (as a `DG/Dt` term). Tests in `test_validate.py` (`llc` is `needs_grid`); none gates the
  headline.

*Discharges:* nothing on its own; the audit records V3b under criterion 3 or 4 as a recorded bias.

### 7. `test_nan_finding.py`, `test_validate.py`, and M1 acceptance

- `test_nan_finding.py`: `fronts_from_gradb2` on a field containing NaN land, both as a whole
  and at the coast. Use **finding config D** (read `finding_config_D.yaml`; the function
  defaults are not config D). If it breaks, record how. The fix belongs in `fronts`, not here.
- `test_validate.py`: all six V-functions run and report, with `needs_grid` honoured.
- **Audit:** run the full suite; check that `git status` shows all six PNGs (use
  `git check-ignore -v` if one is missing); then go through criteria 1-7 one by one with
  numbers, as the M0 task-5 audit did.
- If every criterion passes, mark **M1 closed** here and in coding §6 M1.

*Discharges:* criterion 6 (the remaining tests); the audit closes all seven criteria.

### 8. Slides

Generate a small slide deck for M1 acceptance: a title, a table of contents, and one slide per
task. Write to `dev/frontogenesis/deck/`.  
Include figures where you can (and put the Python scripts to generate them in the deck directory).  Log your work in the deck/README.md file.
For the text, try never to use anything smaller than 20pt font

---

## Acceptance criteria — all must pass

1. **V1 — Cartesian deformation.** Pure deformation `u = -a x, v = a y`, where `G` grows exactly
   as `exp(2 a t)`. Reproduced to **< 1% at the reference front width `ell = 8 dx`** (decided
   2026-09-30, M1-Q5: the criterion means an 8 dx reference width). The effect of width, for the
   record: V1's 8-hour error is **0.78 / 1.57 / 3.09 / 4.43 / 7.86%** at 8 / 6 / 4 / 3 / 2 dx, from
   the centred-stencil truncation that `G` and `F` share (second order in `dx/ell`); the
   semi-Lagrangian step alone is **< 0.36%** at every width. So the literal 1% fails at <= 6 dx,
   and any V1-type number on a sharper front must be quoted with its width.
2. **V2 — Native-grid metric test.** An analytic function of `XC`/`YC` with known gradients,
   reproduced to **< 1%**. V1 cannot test this and vice versa: one validates the scheme, the
   other validates `dxC`/`dyC`/`CS`/`SN` handling.
3. **V3 — Discrete null test. `slope = 1 +/- 0.05` on front pixels.** Advect a synthetic tracer
   with prescribed strain (and separately with real LLC velocities) using **our exact discrete
   operators and semi-Lagrangian step**. The identity `F = ½ DG/Dt` relies on the chain rule,
   which centred differences violate at `O((k dx)^2)` — ~2.5 at a `4 dx` feature, *order unity*.
   And `G` (two interpolated gradients) and `F` (three interpolated factors) are attenuated
   *differently* by C-grid staggering — M0 task 5 measured the interpolated Jacobian's trace at
   **0.80x** the flux-form divergence (corr 0.97) on tile 330, so expect a correction of that
   order, and build `G` from the same `b_x, b_y` as `F` or the slope starts a further 0.91x
   off (corrected 2026-09-28, M0 task 5). **If this fails, co-locate the operators or raise the
   scheme order until it passes. Do not proceed.** Return the fitted slope — Figure 2 draws it.
   The real-velocity variant uses M0's **two consecutive hours** (a midpoint velocity needs two);
   do not pull more data here.
   *(Corrected 2026-09-30, M1 task 7, per the task-6b log: V3b, the finite-volume null, measured
   the shift of the slope under a flux-form OS7MP-like truth at **−0.006 ± 0.025** (0.975
   [0.954, 1.003] against V3's 0.981), so **the 0.80-0.85x attenuation does not bias the slope**
   — it is the resolved `G`'s consistent view of the strain, and the semi-Lagrangian null cannot
   see it by construction (task 6). V3b is carried as a recorded bias, not a correction; see
   criterion 4.)*
4. **V4 — Interpolation bias.** Uniform zero-strain flow, where the true `DG/Dt` is identically
   zero. Whatever comes out is our bias, measured rather than estimated; it becomes a permanent
   error bar on every later slope.
   *(Decided 2026-09-30, M1-Q6: the bar is quoted as **0.28-1.0% of `G` per hour (order 3)** —
   0.28% for a 1.5-cell front, 1.0% for a 1-cell front, at the real-hour displacement
   distribution; order 3 stays the default, and M3 reports its slope at **order 5 as a
   sensitivity** (0.06% at 1.5 cells; task 3 saw ~5% between the orders on real front pixels).
   Recorded bias V3b (task 6b, M1-Q2): the pipeline's slope under a model-like flux-form advection
   is **0.975 [0.954, 1.003]** on the real hour — a systematic band of **0.954-1.003, or 0.975 ±
   0.025**, around the 0.981 baseline; **no upward correction** for the Jacobian attenuation; the
   per-width advection-numerics shortfall (**−2% at 2 dx, −4% at 1.5 dx, −11% at 1 dx**) is to be
   subtracted before attributing anything on the sharpest fronts to diffusion.)*
5. **All six PNGs (V1-V6) written to `dev/frontogenesis/figs/`** (plus V3b, `figs/V3b_fv_null.png`,
   task 6b — seven in all; added 2026-09-30, M1 task 7) (`figs/.gitignore` un-ignores
   `*.png` — the fronts `.gitignore` would otherwise hide them; check `git status` shows them.
   V6 must also show the finite tile-edge rim and the `edge_cells` margin; added 2026-09-28,
   M0 task 5). Including **V5**, the figure
   Lauren asked for: a synthetic front shifted by half a cell, showing truth vs `G` from
   bilinear-`G` vs `G` from cubic-`b`, with the negative bias at the maximum annotated. These are
   acceptance criteria, not extras — the point is that each choice is *visible* rather than
   asserted.
6. **Tests pass:** `test_operators.py`, `test_masking.py`, `test_semilag.py`,
   `test_coarsegrain.py`, `test_validate.py`, and **`test_nan_finding.py`** — exercising
   `fronts_from_gradb2` on a field containing NaN. No such test exists today, and the halo fix
   makes NaN land the normal case.
7. **Regression check:** unfiltered **`operators.frontogenesis(..., form='chain')`** agrees with
   the repo's `calculate_fields.frontogenesis_tendency` to round-off (reworded 2026-09-30, M1-Q7:
   the science product is the default `form='discrete'`, task 6; the chain form is the oracle
   path). Use the repo version as a test oracle, never as the science product.

8. **Finishing touches:**  Read my answers to M1-Q9-Q15 and apply them.  Then generate a small slide deck for M1 acceptance: a title, a table of contents, and one slide per task. Write to `dev/frontogenesis/deck/`.  Make sure the smallest font is 20pt.  Log your work.

9. **Simplify.** The slides are great.  Can you add a few more to the deck:
   - A glossary of the main terms used in the slides.
   - A one-slide summary of M0

10. **Prompts for M2**. Please generate the prompts for M2. Add them to the `frontogenesis_prompts_3.md` file.

## Do not

- Do not pull the 72-hour series (M2) or compute any budget on real data (M3). M1 uses only
  M0's two hours and its static grid.
- Do not interpret anything physical. This milestone produces no science.

## Q&A

### Claude, 2026-09-29 (after tasks 1-6)

Numbered M1-Qn so they don't collide with the planning Q-numbers in `frontogenesis_prompts.md`.
The numbers come from the task 1-6 log entries. **M1-Q1 and M1-Q2 block M3**; the rest can be
answered at any time. **All eight answered by JXP; the decisions were applied to the docs on
2026-09-30 (task 7 log entry lists every edit).**

##### Questions

**M1-Q1 — Which form of `F` is the science product?** To pass V3, task 6 made
`operators.frontogenesis(form='discrete')` the default. This `F` is consistent with the
discrete `G` stencil: `-sum_k (L_k b)[L_k, u.grad] b`, with a 4th-order neighbour gradient.
The chain-rule form (`form='chain'`, bit-for-bit the repo's `frontogenesis_tendency`) is kept.
Three facts pull against each other:
- **Discrete form.** V3 gives 1.004 (strain) and 0.981 (real velocity). With the chain rule the
  slopes are 0.951 and 0.791. The chain rule overstates discrete frontogenesis by
  `(2/3)(dx/ell)^2` at a front of width `ell`.
- **Size of the change.** On real front pixels the discrete `F` is **~0.79x** the chain-rule
  `F`. The predicted side of the headline comparison moves by about as much as the effect we
  are measuring.
- **What V3 can and can't say.** Its truth is advected by *our* semi-Lagrangian step, and the
  discrete `F` is by construction the one that matches that stencil. V3 shows that our two sides
  agree with each other. It does **not** show which form LLC4320's own fronts follow: the model
  advects `b` in flux form with OS7MP, not with our scheme.

My recommendation: run M3 with **both** forms, report `form='discrete'` as the primary slope,
and carry the discrete-vs-chain difference as a stated systematic until M1-Q2 is settled.
Alternatives are (a) discrete only, (b) chain only, with the V3 shortfall as a known
correction.

> **JXP:** Follow your recommendation.

**M1-Q2 — Add a finite-volume null before M3?** The expected 0.80-0.85x attenuation of the
interpolated Jacobian relative to flux-form strain (M0 task 5; 0.853 on `mask_analysis`, M1
task 2) **cannot appear in V3**. Both sides of V3 see the same centred velocities. It is
untested. If it acts on real data, M3's slope is biased **high**. Testing it needs a second
null whose truth is a flux-form finite-volume advection, ideally mimicking OS7MP. Measured and
predicted would then disagree for the reason the model's fronts do. Options:
- (a) Add it as **V3b in M1**, before any science. It would be a gate or a recorded bias, your
  call.
- (b) Fold it into M3 as a diagnostic.
- (c) Skip it and state it as a caveat.

I lean (a) as a *recorded bias*, not a gate. It would be the one remaining untested link
between our operators and the model.

> **JXP:** Ok, use (a)

**M1-Q3 — The planning §2.4 sign.** The strain term is `F = -½δG + ½|σ|G cos 2θ`, with `θ`
measured from the compressional axis. It was corrected in task 2, and you approved it "for
now". Can it be marked final, or is there something you want checked first (e.g. against a
textbook convention, or Lauren)?

> **JXP:** . Yes, consider it final

**M1-Q4 — Figure 2's baseline.** The real-velocity V3 slope is **0.981 [0.970, 0.994]**. It
is inside the ±0.05 gate, but its CI excludes 1. Should Figure 2 draw the baseline at 0.981
with its band (my recommendation), or at 1 with the gate width as the error?

> **JXP:** Draw the baseline at 0.981 with its band.

**M1-Q5 — Criterion 1 (V1, < 1%) depends on front width.** V1 passes at an 8-cell front
(0.78% over 8 hours). The literal 1% fails at ≤ 6 cells: 1.57 / 3.09 / 4.43 / 7.86% at
6 / 4 / 3 / 2 dx. The cause is the centred-stencil truncation that `G` and `F` share. The
semi-Lagrangian step alone is < 0.36% at every width. Accept "8 dx reference width" as the
criterion's meaning, or restate criterion 1?

> **JXP:** Yes, accept "8 dx reference width" as the criterion's meaning.  But also be sure to comment on this effect.

**M1-Q6 — The V4 error bar and the interpolation order.** V4's headline is **0.28% of G per
hour** at order 3: the rms on front pixels for a 1.5-cell front, at the real-hour displacement
distribution. For a 1-cell front it is **1.0%**; at order 5 it is 0.06%. Task 3 also saw a ~5%
difference between order 3 and order 5 on real front pixels. Proposal:
- quote the bar as **0.28-1.0% (order 3)**;
- keep order 3 as the default;
- report M3's slope at **order 5 as a sensitivity**.

Or switch the default to order 5? It costs a wider NaN rim (6 nodes per axis vs 4).

> **JXP:** . Let's use your proposal

**M1-Q7 — Criterion 7 wording.** "Unfiltered `operators.frontogenesis` agrees with
`frontogenesis_tendency` to round-off" now holds for `form='chain'`, not for the default. OK to
reword it to name `form='chain'`?

> **JXP:** Yes, that is fine

**M1-Q8 — Housekeeping (answer any or none).**
- (a) `data/tile330_masks.nc` (6 MB, rebuilds in < 1 s) is git-ignored like the zarrs. Track
  it, or leave it ignored?
- (b) `validate.py` is 863 lines, against §1.3's ~400. Split it further in task 7 (e.g. one
  module per gate), or leave it?
- (c) The subfilter term relative to `F̄` grows with filter scale (0.31 / 0.50 / 0.70 at
  L = 2 / 4 / 8), rather than being "O(1) at every L" as planning §5.4 said. It is also
  anti-correlated with `F̄` (about −0.6). Anything you want characterised further before M3,
  or leave it to M3's filter sweep?

> **JXP:** (a) leave it ignored, (b) leave it; (c) leave it

### Claude, 2026-09-30 (after task 7: the `fronts` fixes)

Task 7a (`test_nan_finding.py`) found five defects in the `fronts` front-finding chain under
config D (`fronts/finding/configs/finding_config_D.yaml`). None was patched: the prompt says
"the fix belongs in `fronts`, not here", and `fronts @ viz_tools` is Lauren's branch. The three
real bugs are pinned as `xfail(strict=True)` tests, so a fix shows up as XPASS and the suite
flags it. Until they are fixed, M4 must use the caller-side recipe:
- pass NaN as-is (**do not** fill it: 0- or median-fill moves fronts ≥ 10 cells from land);
- pass an explicit `n_workers`;
- turn `despur` off on any field that could be empty;
- apply `fronts &= isfinite(gradb2)` afterwards.

Details are in the task-7a log entry. M1-Q9 decides how the fixes happen; M1-Q10 to Q14 are
one per fix.

##### Questions

**M1-Q9 — Who fixes `fronts`, and when?** Options:
- (a) I write the fixes on a branch off `viz_tools` and open a PR for Lauren to review.
- (b) I write them up as an issue or a note for Lauren to apply herself.
- (c) Defer them. M4 uses the recipe above, and the fixes wait.

In every case they should land **before M4**, because M4 is the first milestone that finds
fronts on NaN-bearing fields across 72 hours. I lean (a), limited to fixes 1-3 (M1-Q10 to Q12),
with 4-5 in the same PR only if Lauren wants them. Each fix is a few lines, and the xfail tests
already give a before/after check.

> **JXP:** Let's fix them on this "frontogenesis" branch.  I will let Lauren know we are doing so.

**M1-Q10 — Fix 1: `thresh_mode='pool'` crashes with the default `n_workers=None`.** Config D
sets `thresh_mode: 'pool'`, but the default reaches `np.array_split(rows, None)` at
`pyboa.py:797` and raises a `TypeError`. So **config D is not runnable as written**. The
existing callers (`finding/run.py:52`, `explore_hyper.py:131`) survive only because they
hard-code `n_workers = 10`. Also, 'pool' uses a spawn `ProcessPoolExecutor` (on macOS), so a
script without an `if __name__ == '__main__'` guard dies with `BrokenProcessPool`. Options:
- (a) default `n_workers` to `os.cpu_count()` when `None`;
- (b) fall back to `'vectorized'` when `None` (single process, ~2x slower on the tile:
  13 s vs 7 s);
- (c) add `n_workers` to config D's YAML.

In all cases, document the `__main__` guard. I lean (a) + the docstring note.

> **JXP:** (a)

**M1-Q11 — Fix 2: small NaN islands get flagged as front.** `pyboa.cropping` calls
`remove_small_holes` (default `area_threshold=64`). That fills any hole a front encloses,
including NaN land smaller than 64 px. A 6x5 NaN island on a ridge produced **6 front pixels on
land**. On tile 330 this matters for small islands and river mouths, and the halo removes most
of them. But a front mask that can sit on NaN is wrong in principle. Options:
- (a) have `fronts_from_gradb2` re-mask after cropping and dilation:
  `res_frnt_crop &= np.isfinite(gradb2)`. This is one line and tested: it removes exactly the
  island fill and nothing else.
- (b) pass `remove_small_holes` a hole mask that excludes NaN cells. This is more surgical but
  touches `pyboa.cropping`'s signature.

I lean (a).

> **JXP:** (a)

**M1-Q12 — Fix 3: despur crashes on an empty field.** With `despur: true` (config D),
`prune_short_spurs` builds `skan.Skeleton` on an empty skeleton. That raises `ValueError: index
pointer size 0 should be 1`, for an all-NaN field **or a featureless one**. Over 72 hours × a
filter sweep, an empty sub-field is plausible, e.g. a heavily masked tile or a small cutout. The
proposed fix is an early return, `if not skeleton.any(): return skeleton`, before `Skeleton()`.
Any objection, or should an empty result be treated differently, e.g. a warning?

> **JXP:** No objection

**M1-Q13 — Fix 4 (cosmetic): skimage 0.26 deprecation.** `remove_small_objects(min_size=7)`
in `pyboa.cropping` is deprecated in favour of `max_size=`. **Note the off-by-one**: the new
argument keeps objects *larger than* `max_size`, so the equivalent is `max_size=min_size - 1`.
It only produces warnings today, but it will break on a future skimage. Include it in the PR,
or leave it to Lauren?

> **JXP:** Include it in the PR.

**M1-Q14 — Fix 5 (cosmetic): thousands of All-NaN warnings.** In `'vectorized'` mode, every
all-land 64x64 window emits a `RuntimeWarning: All-NaN slice`: 1,440 on a 160x200 test field,
far more on the tile. The result is correct; this is only noise. The fix is to wrap
`nanpercentile` in `warnings.catch_warnings()`. 'pool' and 'generic' were not measured for
this. Include it in the PR, or leave it?

> **JXP:** Include it in the PR.

**M1-Q15 — Two doc items that go with the fixes.**
- (a) `frontogenesis_prompt_5.md` (M4) still names `build.tile_find` →
  `fronts.preproc.gradb2.generate_tile_gradb2`, which does not exist in this checkout; M4's
  entry point is `fronts_from_gradb2` directly. Coding §2.5 is already corrected. OK to correct
  prompt 5 now, and to add the caller-side recipe above to it?
- (b) Two small docstring errors in `fronts`: `fronts_from_gradb2` describes `threshold` as a
  cropping size (it is the percentile), and `front_thresh`'s `ValueError` message omits
  `'pool'`. Fold them into the M1-Q9 PR?

> **JXP:** Include it in the PR.

## Log

Append to `frontogenesis_prompts.md` under `## Logs`, one entry per task, titled
`### <date> — Execution prompt 2, task N: <title>`. Across the milestone, record: the four gate results with numbers; the fitted discrete-null slope (Figure 2 needs it); the
measured interpolation bias; the six PNG paths; and any operator change you had to make to pass
gate 3 — **that change is a finding about the discretisation and belongs in the writeup.**
