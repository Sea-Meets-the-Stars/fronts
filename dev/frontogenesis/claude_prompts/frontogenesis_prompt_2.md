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

## Acceptance criteria — all must pass

1. **V1 — Cartesian deformation.** Pure deformation `u = -a x, v = a y`, where `G` grows exactly
   as `exp(2 a t)`. Reproduced to **< 1%**.
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
4. **V4 — Interpolation bias.** Uniform zero-strain flow, where the true `DG/Dt` is identically
   zero. Whatever comes out is our bias, measured rather than estimated; it becomes a permanent
   error bar on every later slope.
5. **All six PNGs (V1-V6) written to `dev/frontogenesis/figs/`** (`figs/.gitignore` un-ignores
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
7. **Regression check:** unfiltered `operators.frontogenesis` agrees with the repo's
   `calculate_fields.frontogenesis_tendency` to round-off. Use the repo version as a test
   oracle, never as the science product.

## Do not

- Do not pull the 72-hour series (M2) or compute any budget on real data (M3). M1 uses only
  M0's two hours and its static grid.
- Do not interpret anything physical. This milestone produces no science.

## Log

The four gate results with numbers; the fitted discrete-null slope (Figure 2 needs it); the
measured interpolation bias; the six PNG paths; and any operator change you had to make to pass
gate 3 — **that change is a finding about the discretisation and belongs in the writeup.**
