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
  Jacobian+interp stencil, 4 for the widest filter half-width) — roughly 13-16 km across the
  tile. Use the **measured** `dxC` from M0; never hard-code the km value.
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
   *differently* by C-grid staggering. **If this fails, co-locate the operators or raise the
   scheme order until it passes. Do not proceed.** Return the fitted slope — Figure 2 draws it.
   The real-velocity variant uses M0's **two consecutive hours** (a midpoint velocity needs two);
   do not pull more data here.
4. **V4 — Interpolation bias.** Uniform zero-strain flow, where the true `DG/Dt` is identically
   zero. Whatever comes out is our bias, measured rather than estimated; it becomes a permanent
   error bar on every later slope.
5. **All six PNGs (V1-V6) written to `dev/frontogenesis/figs/`.** Including **V5**, the figure
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
