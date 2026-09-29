# Frontogenesis in the LLC4320 California Current — Planning Document

**Status:** planning complete. All decisions closed (Q1-Q15). Ready for execution — see
`frontogenesis_coding.md` for milestones and `claude_prompts/frontogenesis_prompt_*.md` for
the per-milestone execution prompts.
**Authors:** J. X. Prochaska, with Claude (Opus 5).
**Created:** 2026-09-12.
**Source of decisions:** `claude_prompts/frontogenesis_prompts.md`, Q&A rounds 1-3.
**Reviewed:** adversarially, 2026-09-12. The review overturned three claims in the first
draft — that the residual is purely diabatic (§2.3), that the unfiltered limit isolates the
diabatic part (§5.4), and that semi-Lagrangian interpolation error is negligible (§5.3) —
and added the discrete null test that now gates Phase 2 (§6). Those corrections are folded
in below and are the reason §6 has four validation tests instead of two.
**Updated 2026-09-28 (M0 task 3, against real OSN data; log entry of that date):** five more
claims corrected in place — OSN land is NaN, not 0 (§5.5); `W[k_l=0]` is `dEta/dt` under the
model's *linear* free surface, not ~0 (§2.2); the grid spacing is 1.68-2.07 km and face 10 is
rotated 90 degrees (§4); the rotation terms of §5.2 are identically zero on this tile; and the
advection scheme is OS7MP with a strongly scale-dependent `kappa_num` (§2.3). Each spot is
marked "(corrected 2026-09-28, M0 task 3)".
**Updated 2026-09-28 (M0 task 5, QA plot and acceptance; log entry of that date):** four more
— the `_tile_indexer` derivative rim is on **all four** tile edges and is *finite*, not NaN
(xgcm `padding='fill'` pads with 0; low edges are the worse ones), so the analysis mask needs an
explicit edge margin the land halo cannot supply (§5.5, §6 V6); the stencil NaN rim is 1 cell for
`G` and 2 for the Jacobian/`F`, not ~3 (§5.5); `G` must be formed from the same `b_x, b_y` that
enter `F`, not from the repo's `gradb2` stencil, which differs by 0.91x (§2.1); and the
interpolated Jacobian's trace is 20% attenuated against the flux-form divergence (slope 0.80),
the size of correction §6 test 3 should expect. Marked "(corrected 2026-09-28, M0 task 5)".

---

## 1. Objective

Compare the **measured** strengthening of buoyancy fronts in LLC4320 against the
**predicted** kinematic frontogenesis rate computed from the model's own velocity and
buoyancy fields, and interpret the difference.

The two quantities are not independent guesses at the same number — they are the two sides
of an exact budget, and the residual between them is itself the physical result. Stating
that budget precisely is the whole design of the study.

---

## 2. Physical formulation

### 2.1 Definitions

Buoyancy, from `Theta` and `Salt` at the surface:

```
b = g * sigma0(Theta, Salt) / rho0           [m s^-2]
    g = 9.81 m s^-2,  rho0 = 1000.0 kg m^-3,  sigma0 = JMD95 potential density - 1000
```

**Note the sign.** This is `calculate_fields.buoyancy_of_field`, the repo's own definition,
and it is `+g sigma0/rho0` — it *increases with density*, i.e. the negative of textbook
buoyancy. `G` and `F` are quadratic in `grad b` so nothing downstream is affected, and the
alignment angle enters only as `cos(2 theta)`, which is invariant under the flip. **Do not
"fix" the sign** — just record it so `grad b` arrows are read correctly.

The front-strength field:

```
G = |grad_h b|^2  =  b_x^2 + b_y^2            [s^-4]
```

formed from the **same** `b_x, b_y` (`calculate_native_gradient_tracer`: diff, interp to
centres, rotate) that enter `F` below — that is what makes the discrete identity
`F = (1/2) DG/Dt` hold on the grid. The repo's own front indicator `gradb2`
(`calculate_grad_squared_tracer`, which squares on the staggered points *before*
interpolating) is a different stencil: on tile 330 the component form is **0.911x** its
interior median, so mixing the two would bias the §6 null slope by ~0.9 before any physics.
The repo's `gradb2` is used only for front *finding* in Phase 3, where only the pattern matters
(corrected 2026-09-28, M0 task 5).

The kinematic frontogenesis function:

```
F = -( u_x b_x^2  +  (u_y + v_x) b_x b_y  +  v_y b_y^2 )      [s^-5]
```

**Factor-of-two convention (easy to lose).** `F = (1/2) DG/Dt` in the adiabatic limit, so
the comparison is always **`2F` against the measured `DG/Dt`**. Every figure axis and
every regression in this study uses that pairing.

### 2.2 The surface budget

Starting from `Db/Dt = B` (B = diabatic sources) and differentiating horizontally,

```
D/Dt ( (1/2) G )  =  F  -  b_z (w_x b_x + w_y b_y)  +  grad_h b . grad_h B
                     ^^^    ^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^
                     kinematic   tilting                diabatic
```

At the free surface `z = eta`, `w` does **not** vanish — the kinematic boundary condition is
`w = D eta/Dt`. What vanishes identically along the surface is the velocity *relative to* the
surface, `w - D eta/Dt`, and that is what the along-surface derivatives see: evaluating the
budget on `b_s(x, y, t) = b(x, y, eta, t)`, the terms `b_z (eta_t + u . grad eta - w)` cancel
by the kinematic condition, so the **tilting term drops out** and the 3-D material derivative
reduces to its horizontal form. The surface budget is therefore

```
D_h/Dt ( (1/2) G )  =  F  +  grad_h b . grad_h B
```

with `D_h = d/dt + u d/dx + v d/dy`. This is why surface-only data is *sufficient* for
this question rather than a compromise: there is no missing kinematic term. *(Corrected
2026-09-28, M0 task 3: an earlier version said "`w` vanishes identically at `z = 0`". It does
not — in the data `W[k_l=0]` is `dEta/dt`, median `5.5e-5 m s^-1`, p99 `9.6e-5`, correlation
0.994 and slope 1.04 against the centred hourly `Eta` difference — and the conclusion rests on
the kinematic condition, not on `w = 0`.)*

**The continuum argument is right; the data are not the continuum.** LLC4320 runs a
**linear implicit free surface** (`implicitFreeSurface`, `exactConserv`; `nonlinFreeSurf` and
`select_rStar` unset — §2.3), so there is no `z*` dilation: the top cell is a fixed box of
thickness `drF[0] = 1.0 m` that the free surface moves through, and the model's
`W(k_l=0) = dEta/dt` is a coordinate-relative flux through the *top* of that box which the
tracer equation sets to zero (`gad_advection.F`; the surface transport is zeroed and the
resulting global non-conservation is left alone, `linFSConserveTr` unset). Its horizontal
gradient is `|grad_h W| ~ 1e-9 s^-1` (median `9.5e-10`, p99 `3.4e-9`; corrected 2026-09-28
from an earlier `~1e-10`), which contributes `< ~1%` of `F` in rms for any plausible `b_z`.
The stored `b` is a cell average, and its budget contains the flux through the **cell base**,
where (continuity, `z` up) `w(-dz) = w(0) + dz * delta`: the `dEta/dt` part (`~5e-5 m s^-1`,
tidal, reversing) is ~5x the convergence part (`dz * |delta|` median `1.1e-5`, p99 `5.3e-5`
at fronts), but the convergence part carries most of the *gradient* (`dz |grad delta|` median
`4.8e-9`, p99 `2.2e-8 s^-1`). *(Corrected 2026-09-28: an earlier version wrote
`w(-dz) = -dz * delta`, wrong sign and missing the `dEta/dt` part.)*

This matters more than "a small correction" in summer. With a diurnal warm layer giving
`dT ~ 0.1-0.3 K` between `k=0` and `k=1`, `b_z ~ 2-4e-4 s^-2`, and the vertical
contribution `-b_z (w_x b_x + w_y b_y)` reaches **~30% of F by day** and ~0 at night. So
the tilting term is not absent — it is *reintroduced by the discretisation*, with a strong
diurnal cycle. M0 task 3 bracketed it from the surface fields alone: with the cell-base
convergence gradient and `b_z = 1e-5 / 1e-4 / 4e-4 s^-2` the term is 0.4% / 3.7% / 14% of
`F` in rms over the tile and order-one pointwise at fronts (median `|T|/|F|` 1.6% / 16% /
63%), consistent with the ~30% figure at the warm-layer end.

**This term is now measured, not bounded.** Lauren is transferring **hourly full-depth**
`CHUNKS/monterey_bay` for the whole 72-hour window (51 levels; decision Q13), so `k = 1, 2`
Theta/Salt give `b_z` and the chunk **`W(k_l=1)`** — the model's own cell-base velocity,
which already contains both the `dEta/dt` and the convergence parts — gives `w_x`, `w_y` at
*every* timestep. Build the term from that `W`, not from `delta` (corrected 2026-09-28). Also
do not assume `b_z` is uniform across the front: the factorised form `-b_z grad w . grad b`
drops `-w grad(b_z) . grad b`, which with `w ~ 5e-5 m s^-1` and front-scale changes in
stratification is not obviously smaller, so `vertical.py` computes the top-cell vertical
advective tendency `-w_base (b_base - b)/drF` first and takes `grad_h b . grad_h` of it. What
was a bound from 11 snapshots becomes an explicit budget term. Confirmed from the source while
sizing that transfer: **`drF[0] = 1.0 m`, `Z[0] = -0.5 m`**, 51 levels to ~968 m — so the
"~1 m top cell" above is a measured fact, not an assumption.

*Where `drF` comes from.* The OSN gridfile **does** carry the top-cell vertical scalars as 0-d
coordinates — `drF = 1.0`, `Z = -0.5`, `Zl = 0.0`, `Zp1 = 0.0` (k = 0 only; `process_llc4320_grid`
drops them, so `osn_tiles.load_grid` re-attaches them; corrected 2026-09-28, M0 task 2) — and
they are written to the §3.1 grid. The chunk store's 3-D grid (`process_llc4320_3d_grid`, which
adds `Z, Zl, Zu, Zp1, drF` for all levels) is the source for `k = 1, 2` and a cross-check on
`drF[0]`.

### 2.3 What the residual means

```
Residual  =  measured D_h G/Dt  -  2F
```

**This is not purely diabatic, and an early draft of this plan wrongly said it was.**
The residual contains at least four things:

1. **Implicit numerical diffusion.** LLC4320 carries no explicit horizontal tracer
   diffusion (`diffKhT`, `diffK4T` unset, i.e. 0); its tracer advection is
   **`tempAdvScheme = saltAdvScheme = 7`**, the flux-limited seventh-order
   monotonicity-preserving scheme of Daru & Tenaud (2004) (OS7MP), with
   `multiDimAdvection`, `StaggerTimeStep`, `deltaT = 25 s`, a linear implicit free surface
   (`nonlinFreeSurf`/`select_rStar` unset), JMD95Z EOS, biharmonic Leith viscosity
   (`viscC4Leith = 2.1-2.15` in our window) and KPP. Source: `MITgcm_contrib/llc_hires/
   llc_4320/input/data`, https://raw.githubusercontent.com/MITgcm-contrib/llc_hires/master/
   llc_4320/input/data (production-era commit `4627a7a8` differs only in the Leith
   coefficient), corroborated by the NASA S-MODE model description. *(Confirmed 2026-09-28,
   M0 task 3.)* Its implicit dissipation is strongly scale-selective and switches on
   precisely at the grid-scale gradients that *define* our fronts. From the exact Fourier
   symbol of the unlimited 7th-order upwind kernel, `kappa_num / (|u| dx)` = 0.093 at `2 dx`,
   0.023 at `4 dx`, 4.7e-3 at 10 km, 1.1e-4 at 20 km (`dx = 1.8 km`, small-`k dx` limit
   `|u| dx^7 k^6 / 280`). With the tile's speeds (`|u|` median 0.19, p90 0.38, p99 0.64
   m s^-1): at `4 dx` (7 km) `kappa ~ 8-27 m^2 s^-1` and `G` is damped at
   `2 kappa k^2 ~ (1.2-4)e-5 s^-1` — **0.1-0.5 f, the same order as the strain**, e-folding
   in 7-23 h; at the **10 km front scale** `kappa ~ 1.7-5.6 m^2 s^-1`, `2 kappa k^2 ~
   (1.3-4.4)e-6 s^-1`, e-folding 2.6-9 days, so 30-70% of `G` survives the 72 h window and
   numerics are 3-10% of the kinematic rate; at `2 dx` `G` e-folds in under 1.5 h. Where the
   monotonicity limiter engages (1-2-cell fronts, extrema) the local diffusivity rises toward
   the first-order-upwind bound `|u| dx / 2 ~ 170-580 m^2 s^-1`, so at the sharpest fronts
   the numerics are the whole story. *(Corrected 2026-09-28: the earlier single figure
   "`kappa ~ (0.01-0.1) u dx ~ 6-60 m^2 s^-1`, 0.1-1 f" is the `4 dx` value and is an order
   of magnitude too pessimistic at 10 km; the scale dependence is the point.)* A slope of
   0.5-0.8 at the grid scale is fully explicable by model numerics with zero air-sea flux.
2. **The finite-top-cell vertical term** (§2.2), ~30% of `F` by day — **now measured**
   at every timestep from the hourly full-depth chunks.
3. **Genuine diabatic forcing** — KPP diffusive and non-local fluxes, and shortwave
   absorbed within the top cell (a large fraction of `Q_sw`, giving order `0.1 K h^-1`
   heating at local noon before KPP redistributes it). **Largely measured too**: the chunk
   store carries `oceQnet`, and `oceQsw` / `oceFWflx` are being added to the transfer
   (Q13), so the surface-flux part of `grad b . grad B` is computed directly rather than
   inferred.
4. **Discretisation error** in our own operators (§5.2, item 2 of §6).

**Consequence for the headline claim.** "Slope < 1 = diabatic damping" is *not* a safe
inference. But the Q13 transfer changes the shape of the problem substantially: with (2) and
most of (3) computed explicitly, the residual reduces to **numerical diffusion + interior KPP**
rather than "everything we could not compute". That is a far stronger statement, and it is
what makes the efficiency claim earnable at all. Figure 2b remains the discriminator —
numerical diffusion scales with high-order derivatives of `b` (a `grad^4`-like structure) and
carries a different diurnal phase from air-sea forcing — but it is now corroborating a
measured budget rather than carrying the whole argument.

### 2.4 Decomposition of F

Writing divergence `delta = u_x + v_y`, normal strain `sigma_n = u_x - v_y`, shear strain
`sigma_s = v_x + u_y`, and `|sigma| = sqrt(sigma_n^2 + sigma_s^2)`:

```
F = -(1/2) delta G  +  (1/2) |sigma| G cos(2 theta)
```

where `theta` is the angle between `grad_h b` and the strain compressional axis. This gives
a strong independent physical check: frontogenesis should peak where `grad b` aligns with
the compressional axis, and the PDF of `theta` is a classic signature (Figure 4).
*(Corrected 2026-09-29, M1 task 2: the strain term carries a **plus** sign when `theta` is
measured from the compressional axis — for the pure deformation `u = -a x, v = a y` and a front
`b(x)`, `theta = 0` and `F = +a b_x^2`; an earlier version wrote a minus, which holds only for
`theta` measured from the extensional axis. `operators.strain_alignment` uses the compressional
axis, folded to `[0, pi/2]`, and the identity above closes to round-off on the grid with the
Jacobian-derived strain.)*

---

## 3. Decisions of record

| # | Decision | Source |
|---|---|---|
| Q1 | Field-level budget (Phase 2) before front tracking (Phase 3) | agreed |
| Q2 | **Surface-only.** Depth considered later | agreed |
| Q3 | **California Current**, not Gulf Stream | JXP |
| Q4 | All output in `dev/frontogenesis/{py,data,figs}`; reports in markdown | JXP |
| Q5 | No global front-finding route | agreed |
| Q6 | **Summer only** (no winter contrast window) | JXP |
| Q7a | Land halo = stencil (3) + filter half-width (4) = **7 cells (~13 km)** | JXP deferred to recommendation |
| Q7b | Primary statistics **>= 100 km offshore** | JXP |
| Q8 | **72 hours** first pass | JXP |
| Q9 | **One shared operator and filter on both sides**, computed by us | JXP |
| Q10 | Population tracking = looped `follow()` over the **largest N=10** fronts | JXP |
| Q11/Q15 | Branch strategy — **resolved**; sequence in §10 | JXP + Lauren |
| Q12 | Library route (import `dbof`, bypass the CLI) | JXP, unobjected |
| Q13 | **Hourly full-depth chunks**, all 51 levels; `oceQsw` + `oceFWflx` added | Lauren + JXP |
| Q14 | **Flow-informed tracking is an M4 requirement**, with Lagrangian-matched pixels | JXP |

---

## 4. Data

**Source.** The OSN store (Spencer Jones' public archive), anonymous, no credentials:

```
endpoint : https://mghp.osn.xsede.org
refs     : cnh-bucket-1/llc_surf/kerchunk_files/
           llc4320_Eta-U-V-W-Theta-Salt_f{face}_k0_iter_{it}.json
grid     : cnh-bucket-1/llc_surf/kerchunk_files/llc4320_grid_f{face}.json
coverage : 2011-09-13 -> 2012-11-15, hourly, surface only (k=0)
```

Read through fsspec's built-in `reference://` filesystem (`engine="zarr"`,
`consolidated=False`) — the `kerchunk` package itself is not required.

**Region.** Rect-grid tile 330 = rect `(i=13320, j=9720)` -> **face 10**, face-local
`j 0:720, i 2880:3600`; box **lon -127.99..-113.01, lat 26.66..38.27**, 720x720 native cells
at **1.68-2.07 km** spacing (`dxC` 1.68-1.90, `dyC` 1.82-2.07 km; 1.71 x 1.85 km at 37N;
`dyC/dxC = 1.086` everywhere). **Face 10 is rotated 90 degrees:** `CS = 0`, `SN = -1` in
every cell, longitude varies along `j` (exactly 1/48 deg per cell) and latitude along `i`
(`i` increasing *southward*), so `j`/`V`/`dyC` are zonal and `i`/`U`/`dxC` meridional
(`u_east = V`, `v_north = -U`). `dyC` is the `(1/48) deg cos(lat)` zonal spacing; the
meridional spacing is 8% smaller. *(Corrected 2026-09-28, M0 task 3; an earlier version said
"~1.8-2.3 km", lat to 38.20, and implied `i` was zonal.)*

**Window.** **2012-07-02 00:00 -> 2012-07-04 23:00 UTC**, 72 consecutive hours.

*Why this window and not the start of the series:* the full-depth
`LLC4320_RAW/CHUNKS/monterey_bay` store holds daily 12:00 snapshots **plus a dense
3-hourly day on 2012-07-03**. This window places **11 of its 17 snapshots inside**,
including all 8 of the dense day — versus 3 if we began at 2012-06-29. It costs nothing
and makes the eventual depth cross-check (Phase 4) far more useful.

**Fields pulled.** Raw only: `Theta`, `Salt`, `U` (on `i_g`), `V` (on `j_g`), plus `W` and
`Eta` for diagnostics. Grid (static, pulled once):
`XC, YC, dxC, dyC, dxG, dyG, rAz, rA, Depth, hFacC, SN, CS`, plus — from the raw gridfile,
which `process_llc4320_grid` drops — the staggered land fractions **`hFacW`, `hFacS`** (the
`U`/`V` masks; binary at k=0, equal to the minimum of the adjacent `hFacC`) and the 0-d
**`drF`, `Z`, `Zl`** (added 2026-09-28, M0 tasks 2-3).

**Second OSN store — pull it too.** `cnh-bucket-1/llc_wind/` carries
`KPPhbl, PhiBot, oceTAUX, oceTAUY, SIarea` (also `k=0`, hourly), coverage
2011-11-01 -> 2012-07-15 — **our window sits inside it**. `KPPhbl` (boundary-layer depth) is
the key interpretive variable for the diurnal residual of §2.3, and the wind stress gives the
forcing context. Neither OSN store carries heat fluxes. Note (M0 task 3): `oceTAUX`/`oceTAUY`
sit on `i_g`/`j_g` but are masked with the *centred* `hFacC` mask — 922 / 565 finite values
lie on faces the model treats as land — so re-mask with `hFacW`/`hFacS` before any
stress-divergence.

**Third source — hourly full-depth chunks (decision Q13).**
`LLC4320_RAW/CHUNKS/monterey_bay`, being extended by Lauren to **hourly for all 72 hours**,
all **51 levels** (to ~968 m; `drF[0] = 1.0 m`). 11 of the needed stores already exist; 61 are
new, ~33 GB at ~539 MB per timestep.

Variables: 3D `Theta, Salt, U, V, W`; 2D `Eta, oceTAUX, oceTAUY, SIarea, **oceQnet**`, with
**`oceQsw`** and **`oceFWflx`** added to `transfer.variables` for this run.

**This is the single most consequential change since the first draft.** An earlier version of
this section said heat fluxes were in neither store and the diabatic term could only ever be
inferred. That is true of OSN but false of the chunk store. With this transfer:

- `k = 1, 2` Theta/Salt -> `b_z`, and subsurface `W` -> `w_x, w_y`: the finite-top-cell
  vertical term becomes a **measured** budget term (§2.2);
- `oceQnet` + `oceQsw` + `oceFWflx` -> the surface-flux part of `grad b . grad B` computed
  **directly**. `oceQsw` matters specifically because a large fraction of shortwave is
  absorbed *inside* the ~1 m top cell where our `b` lives; net flux alone blurs exactly the
  noon-peaking term Figure 6 is about.

The surface analysis still runs on OSN (§5.1) — the chunks supply the *extra budget terms*,
not the primary fields. Keeping the two sources separate also preserves a genuine
cross-check: OSN and the chunk store are different readers of the same physics.

**Volume.** 720 x 720 x 72 h x ~6 fields x 4 bytes ~ **0.9 GB** as float32. Trivial;
the whole study fits in memory on a laptop.

---

## 5. Method

### 5.1 One operator, both sides

Q9's decision is the methodological spine. The registry's `frontogenesis_tendency` applies
no filtering, and a comparison in which the two sides use different stencils or different
filters manufactures a mismatch. So **we compute both sides ourselves**, from the same
saved raw fields, through the same gradient operator and the same filter.

This makes the library route natural: import `dbof` as a library, pull raw fields, and do
everything downstream in `dev/frontogenesis/py`. **No modification to
`tiles-surface-only` or to the `generate-tile` CLI is required.** Verified call sequence:

```python
EP     = "https://mghp.osn.xsede.org"
tile   = rect_ij_to_tile(13320, 9720)                     # face 10, j 0:720, i 2880:3600
g      = process_llc4320_grid(get_remote_gridfile(EP))    # static, once
g_tile = ensure_comodo_attrs(g.isel(face=[tile.face_idx], **_tile_indexer(g, tile)).compute())
grid   = set_xgcm_grid(g_tile, use_connections=False)
for ts in timestamps:
    ds = get_remote_llc_data(EP, osn_date_to_iteration(ts), [tile.face_idx])
    ds_tile = ds.isel(**_tile_indexer(ds, tile))
```

The repo's `frontogenesis_tendency` is retained as an **unfiltered regression test** of our
operator, not as the science product.

**Equation of state.** Use `calculate_fields.buoyancy_of_field`, which is **JMD95** — the
EOS the model itself advected — not TEOS-10/`gsw`. That difference is ~1% but *systematic*
and would propagate straight into the headline slope. It evaluates potential density at
`p = 0`; the review suggested in-situ density at the cell mid-depth (~0.5 dbar), a
difference that is negligible at the surface. **Do not use
`utils/physical_calculations.buoyancy_of_field`** — that one is legacy, with `g = 0.0098`
in km s^-2 and `rho_ref = 1025`.

### 5.2 Native basis, and the rotation approximation

An earlier version of this section said we would work purely in the native basis with no
rotation. That is not what the available helpers do, and fighting them would mean
re-deriving tested code. The corrected rule:

- **`grad b` and the velocity Jacobian both come from the repo helpers**
  (`calculate_native_gradient_tracer`, `calculate_jacobian`), which **both rotate to the
  geographic basis via `CS`/`SN`**. What matters for `F` is that the two are in the *same*
  basis, and they are.
- **`G` and `F` are rotational invariants**, so the basis choice does not change either
  quantity — it only has to be consistent.
- **Departure points stay in native index space** (§5.3): raw `U`, `V` interpolated to cell
  centres, then `di = U dt/dxC`, `dj = V dt/dyC`. No rotation, no round-trip.

*Known approximation.* Rotation invariance is exact only for a spatially constant rotation;
where `CS`/`SN` vary, "rotate then differentiate" and "differentiate then rotate" differ by
terms in `grad CS`, `grad SN`. **On tile 330 they do not vary at all** (corrected 2026-09-28,
M0 task 3): `SN = -1.0` exactly and `|CS| < 1.3e-12` in every cell, the grid angle is
`-90.000` degrees with zero range, `|grad SN| = 0` and `u |grad CS| ~ 1e-17 s^-1` — the
rotation is an exact axis swap and the rotation terms are **identically zero**, not "~0.1%"
as an earlier version estimated from "a few degrees across 720 cells". The only neglected
term is the spherical metric term `u tan(phi)/a`: median `1.8e-8 s^-1`, p99 `6.8e-8`, i.e.
**0.09% of the local strain at the median and 0.5-0.7% at p99** (measured `|sigma|` median
`1.9e-5 s^-1`, p99 `7.9e-5`). Confirmed numerically in Phase 0.

### 5.3 Measured D_h G/Dt — semi-Lagrangian

The material derivative is measured by a **single semi-Lagrangian difference**, not by
differencing `d/dt` and `u.grad(G)` separately:

```
D_h G / Dt  ~  [ G(x, t+dt) - G(x_d, t) ] / dt
```

with the departure point `x_d` found by iterated midpoint using the time-centred velocity
`(u(t) + u(t+dt))/2`. Departure points are computed directly in native index space —
interpolate `U` (on `i_g`) and `V` (on `j_g`) to cell centres, then `di = U dt / dxC`,
`dj = V dt / dyC` — with no rotation.

*Why this and not Eulerian.* In the Gulf Stream (`u ~ 1-2 m/s`) an hourly Eulerian split
produces two large, nearly cancelling terms. In the California Current the typical
displacement is under a cell per hour, which makes the Eulerian form viable again — so we
compute it too, as an **independent cross-check**, and report the agreement rather than
assuming it.

**Three corrections to an earlier, too-optimistic version of this section.**

1. **Displacements are not 0.2-0.4 cells.** That is the *typical* value. In 1 m s^-1
   filaments, and with tidal and inertial currents added, displacement exceeds **1.5 cells**
   — and those are exactly the strong-front pixels the study is about. The error budget
   must be quoted at the tail, not the median. *(Confirmed 2026-09-28, M0 task 3, on
   2012-07-02 00:00: ocean median 0.37 cells, p99 1.28, max 3.5; on front pixels
   (`G > p90`) median 0.54, p99 1.64, max 2.5; speed median 0.19, p99 0.64 m s^-1.)*
2. **Interpolate `b`, not `G`, and use high order.** Bilinear interpolation of `G` at a
   half-cell offset has error `dx^2 G_xx / 8`, which for a front of width ~1.5 cells is
   ~5.5% of `G` and is **systematically negative at maxima** — it fabricates apparent
   frontogenesis. Against a signal of `2F dt / G ~ 7-20%` per hour, that bias is
   **25-80% of the signal**. Use cubic (or quintic) interpolation of `b` onto the
   departure-point stencil, then differentiate, and validate the order chosen (§6).
3. **Evaluate `F` at the trajectory midpoint time**, not at an endpoint. M2 strain rotates
   ~29 degrees per hour, so `F(t)` and `F(t+dt)` differ by tens of percent; using an
   endpoint both adds noise and correlates that noise with the measured side (§11).

### 5.4 Filtering — and what it does to the residual

Both `b` and `(u, v)` are low-pass filtered **identically** before anything is computed,
swept over **none / 2 / 4 / 8 cells**.

This is not noise control, and an earlier version of this section got the consequence
wrong. Writing the filter as an overbar, `d_t bbar + ubar.grad(bbar) = Bbar - div(tau)`
with the subfilter buoyancy flux `tau = mean(u b) - ubar bbar`, the correct coarse-grained
budget for `Gbar = |grad bbar|^2` is

```
Dbar/Dt ( (1/2) Gbar )  =  F(ubar, bbar)  +  grad bbar . grad Bbar
                                           -  grad bbar . grad( div tau )
                                           -  [finite-top-cell vertical term]
```

The subfilter term is a **third derivative of the subfilter flux**, so it peaks at the
filter cutoff — precisely where `|grad bbar|^2` has its variance. It is `O(1)` relative to
`F` at every `L`, and **as `L -> dx` it does not vanish: it becomes the implicit numerical
diffusion term of §2.3.** The claim that the unfiltered limit isolates the diabatic part
was wrong.

So the sweep as originally conceived measures the subfilter fraction at each cutoff, not a
"diabatic efficiency". To make it a legitimate diagnostic we have the full fields, so we
**compute `tau` explicitly** (Germano-identity / Aluie coarse-graining) at each `L` and
demonstrate that the budget closes. The sweep then becomes a genuine scale-transfer
diagnostic — arguably a more interesting result than the original framing — and every
slope/correlation is reported as a function of `L` (Figure 3).

### 5.5 Land

**OSN stores land as NaN, not 0** (corrected 2026-09-28, M0 task 3; an earlier version
assumed the MITgcm convention of 0, under which `b(Theta=0, Salt=0)` would be finite and every
coastal ocean cell would inherit the land/ocean jump). Checked cell by cell on tile 330:
`Theta, Salt, W, Eta` and the `llc_wind` fields are NaN in exactly the 161,523 `hFacC == 0`
cells; `U` is NaN in exactly the `hFacW == 0` cells and `V` in exactly the `hFacS == 0` cells
(so `U`/`V` are NaN on 922 / 565 coast-facing faces whose centre is ocean); no finite values on
land, no NaN in the ocean, and the pattern is static in time. `hFacC` is binary at `k = 0` (no
partial cells). So there is **no coastal gradient ribbon**: the dbof stencils propagate NaN,
and the stencil part of the halo happens by itself. Measured on the QA plot (M0 task 5): `G` is
NaN in exactly the ocean cells at taxicab distance 1 from land (2,174 cells; diff + interp
reaches one centre), the Jacobian and hence `F` in exactly those at distance <= 2 (4,204
cells; interp, diff, interp). Median `G` decays smoothly from 76x the interior at 2 cells to
10x at 10 cells — the coastal upwelling front, not a stencil artefact (a `b(0,0)` ribbon would
be ~1e7x and confined to one cell) (corrected 2026-09-28, M0 task 5).

A **dilated land mask of 7 cells (12-13 km at 37N)** — 3 for the Jacobian+interp stencil
(measured reach 2; one cell conservative), 4 for the widest filter half-width — is still applied
to `b`, `u`, `v` **before any differencing**: the NaN propagation covers the stencil, but the
filter needs its full support and `coast_distance_km` needs a clean `skfmm` distance.

**The tile edges need their own margin** (corrected 2026-09-28, M0 task 5). `_tile_indexer`
gives the staggered dims the same slice as the centred ones, so the tile holds each cell's
*low* staggered point but not the high one, and xgcm's `padding='fill'` (`set_xgcm_grid`,
`fill_value=None`) pads the missing neighbour with **0**. The result is an invalid rim on
**all four** edges whose values are **finite, not NaN**: differencing against 0 at the low
edges (`j = 0`, `i = 2880`) gives `G` ~1e6x its neighbours, interpolating with 0 at the high
edges (`j = 719`, `i = 3599`) halves it; the Jacobian's extra interp carries the high-edge
error one cell further. Crop test: `G` invalid 1 cell on every edge, the Jacobian 1 cell on the
low edges and 2 on the high. Neither the land halo (`skfmm` distance from `hFacC == 0`; a tile
edge is not land) nor the 100 km offshore cut (the west and north edges are open ocean) removes
it, so `analysis_mask` carries an explicit **`edge_cells`** margin: >= 2 cells for the raw
operators, **7** once filter support is counted — the same budget as the land halo.

Two defects in the existing helper must be handled:
`halo_mask.llc_native_grid_halo_mask` returns a 2-D array early when a face is entirely
land (`halo_mask.py:74-75`), and a `k`-carrying `hFacC` makes the mask 4-D and breaks
`skfmm`. Convention: **True = retained**, land = False, plain numpy.

### 5.6 Offshore restriction

Primary statistics use **>= 100 km from the coast**. This also removes the head of the Gulf
of California, which the tile's eastern edge clips — verified by the mask, not
special-cased.

*Acknowledged consequence:* this excludes the inner upwelling zone, where California
Current frontogenesis is most vigorous — we keep filament tips, not roots. It is the right
call for a clean measurement (it is also where the diabatic residual and any residual land
contamination are worst), but the headline number then describes the **offshore regime**,
not the CCS as a whole. Every statistic is therefore also reported **stratified by distance
offshore**, so what was excluded stays visible.

### 5.7 Front strength (Phase 3)

- **Primary:** front-mean `G` from `properties/colocation.py`, so Phases 2 and 3 measure
  *the same quantity* and are directly comparable.
- **Secondary:** cross-front `delta b` and front width, from
  `curtains.perpendicular_path` / `path_metrics`.

Tracking uses the existing `front_tracking.follow()`, looped over the **largest N=10**
fronts. `follow()` takes `dt` as a genuine parameter and its search radius floors at ~4.6 km
at hourly cadence (~1.3 m/s equivalent) — ample for CC speeds.

**Tracking must be flow-informed (decision Q14).** This came out of Lauren's review and is
not a refinement — it is what makes Phase 3 mean anything.

`follow()` as written predicts the next position by extrapolating *centroid* velocity from the
last two sightings. But a front whose centroid moves because it grew asymmetrically is not a
front that moved with the fluid. Since a buoyancy front is advected by the flow, and we
already have the departure-point machinery, we can do better:

1. Advect the **boolean front mask** (not the label field) through `semilag` to produce a
   **flow-predicted mask** at `t+dt` — as a float, thresholded at 0.5. Labels stay integers
   and are never interpolated, which is what makes this tractable.
2. Feed `IoU(flow-predicted mask, candidate label)` into `score_candidate` as an additional
   scored term. That function already accepts a `weights` dict, so this is additive rather
   than a rewrite.
3. Report the distribution of (`follow()`-chosen displacement − flow-predicted displacement)
   as a **quality metric for the whole of Phase 3**. If those disagree often, the tracking is
   not following the fluid and we know it *before* interpreting anything.

**Why it is load-bearing.** If `follow()` links a front at `t` to a different physical front
at `t+dt`, the per-front `d(front-mean G)/dt` is not a material derivative at all and
comparing it to `integral 2F dt` compares nothing.

**And a further correction.** Even a perfect flow-following track is insufficient, because
front-mean `G` is a mean over a **changing pixel set** — fronts lengthen, split and merge — so
`d/dt` of that mean carries an extra term from the set's own evolution. The Phase-3/Phase-2
reconciliation must therefore be done on the **advected pixel set** (Lagrangian-matched
pixels), not on "pixels labelled front at `t`" against "pixels labelled front at `t+dt`".

Free by-product: the flow-predicted mask overlapping two candidate labels is a principled
**split/merge detector**, which `follow()` has no notion of and which will certainly occur
over 72 hours.

---

## 6. Phases

### Phase 0 — Trust before scale

No science until the operators are known good.

| Task | Exit criterion |
|---|---|
| Environment | `dbof` importable; `xgcm>=0.10`, `scikit-fmm` present |
| Comodo sign | one assertion that `c_grid_axis_shift == -0.5` (already resolved 2026-09-01; we only pin it) |
| `W[k_l=0] ~ 0`? | **done 2026-09-28: no** — `W(0) = dEta/dt` (linear free surface); §2.2 rewritten; tilting term bracketed |
| OSN land fill | **done 2026-09-28: NaN**, cell-for-cell equal to `hFacC`/`hFacW`/`hFacS` (§5.5) |
| Halo mask | 7-cell halo working on a single face; both helper defects handled |
| Rotation/metric terms | **done 2026-09-28**: rotation terms identically zero on face 10; metric term 0.1% median, <= 0.7% p99 of strain (§5.2) |
| Coarse-grained budget | `tau` computed explicitly; budget closes at each `L` (§5.4) |
| Advection scheme | **done 2026-09-28**: OS7MP (`tempAdvScheme = 7`), no explicit horizontal diffusion; scale-dependent `kappa_num` in §2.3 |
| **Validation** | four tests (below), all passing |

**Validation is four tests, not one, and every one writes a PNG.** Lauren asked for the
decisions to be *visible* rather than asserted, and she is right: each test below emits a
figure (§7) as part of its acceptance, not as an optional extra. The first two are
continuum checks; the last two are the ones that actually protect the headline number.

1. **Cartesian scheme test.** Pure deformation `u = -a x, v = a y`, where `G` grows exactly
   as `exp(2 a t)`. Validates the semi-Lagrangian scheme in isolation.
2. **Native-grid metric test.** An analytic function of `XC`/`YC` with known gradients.
   Validates `dxC`/`dyC`/`CS`/`SN` handling. Test 1 cannot do this and vice versa.
3. **Discrete end-to-end null test (required before any real data).** The identity
   `F = (1/2) DG/Dt` relies on the chain rule, which centred differences violate with error
   `O((k dx)^2)` — at a `4 dx` feature that is `~2.5`, *order unity*. Worse, C-grid
   staggering puts `b_x` on faces, `u_x` at centres, `u_y` and `v_x` at corners; each
   interpolation to a common point multiplies a `4 dx` amplitude by `cos(k dx / 2) ~ 0.71`,
   and `G` (two interpolated gradients) and `F` (three interpolated factors) are attenuated
   **differently** — a slope bias of roughly 0.7-1.4 with no physics in it at all. Measured
   on tile 330 (M0 task 5): the trace of the interpolated Jacobian regresses on the flux-form
   divergence (`calculate_native_strain_vorticity`, no interpolation, no rotation) with
   **slope 0.80, corr 0.97** — a 20% attenuation of the predicted side is the size of
   correction to expect here, on top of the 0.91x from a mismatched `G` stencil (§2.1) if
   `G` were not built from the same `b_x, b_y` as `F` (corrected 2026-09-28, M0 task 5).
   *Test:* advect a synthetic tracer with prescribed strain (and separately with the real
   LLC velocities) using our exact discrete operators and semi-Lagrangian step, and
   **require slope = 1 +/- 0.05 on front pixels** before touching real data. If it fails,
   the operators are co-located or the scheme order raised until it passes.
4. **Interpolation-bias null test.** Advect `G` with a *uniform, zero-strain* flow, where
   the true `DG/Dt` is identically zero. Whatever comes out is the interpolation bias of
   §5.3, measured rather than estimated. Report it as an error bar on every later slope.
   Emits **V4**.

**Two supporting figures accompany the gates** (six PNGs in total, from four gates):

- **V5 — the half-cell interpolation demonstration.** A synthetic front shifted by half a cell:
  truth vs `G` from bilinear-`G` vs `G` from cubic-`b`, with the negative bias at the maximum
  annotated. The clearest statement of why §5.3 item 2 exists, and it doubles as a regression
  test.
- **V6 — land-halo QA.** The coastline before and after the halo. Land is NaN (§5.5), so
  there is no gradient ribbon to remove; the figure shows the stencil's own NaN rim (1 cell for
  `G`, 2 for the Jacobian), the 7-cell halo, the `_tile_indexer` rim on **all four** tile edges
  (finite, not NaN — §5.5) and the `edge_cells` margin that removes it, and confirms the mask
  geometry (corrected 2026-09-28, M0 task 5; M0's `figs/m0_qa_tile330_20120702T00.png` is the
  no-halo precursor).

### Phase 1 — Data

Two sources, one product.

- **OSN surface (primary):** 72 hourly snapshots of raw `Theta, Salt, U, V` (+ `W`, `Eta`) plus
  the static grid for tile 330, and `KPPhbl`/`oceTAUX`/`oceTAUY` from the `llc_wind` store.
- **Chunk store (extra budget terms):** `k = 0..2` `Theta, Salt, W` and the surface fluxes
  `oceQnet, oceQsw, oceFWflx` from the hourly full-depth `monterey_bay` transfer (§4). We read
  only the levels and variables we need — the transfer writes all 51 levels, but nothing
  obliges us to load them.

Concatenate each to a time-dimensioned zarr in `dev/frontogenesis/data/`. **No concat step
exists anywhere in either repo** — we write it. Phase 1 is gated on Lauren's transfer for the
chunk half only; the OSN half can proceed immediately.

### Phase 2 — Field-level budget (the rigorous core)

Per pixel, no front finding required. Semi-Lagrangian `D_h G/Dt` vs `2F`; Eulerian
cross-check; residual map; filter sweep with explicit `tau`; offshore stratification; and —
via the hourly full-depth chunks (§4) — the **measured** finite-top-cell vertical term and the
**measured** surface-flux part of the diabatic term.

**Exit criterion — budget closure, not a slope.** We do not quote any efficiency until

```
measured  -  2F  -  subfilter  -  vertical  -  surface_flux   ~   numerical + interior KPP
```

is demonstrated to a stated tolerance — with `vertical` and `surface_flux` **measured** from
the chunk store (Q13) rather than assumed — the four Phase-0 gates passing, and the Eulerian
and semi-Lagrangian estimates agreeing. If closure cannot be demonstrated,
that is the result (§12), and we say so rather than reporting a slope.

### Phase 3 — Front-level

`tile_find` per hour -> label -> **flow-informed** `follow()` (§5.7) over the largest 10
fronts -> per-front strength time series vs `integral(2F dt)` along the track, evaluated on the
**advected pixel set**. Plus the tracking-quality diagnostic (`follow()`-chosen minus
flow-predicted displacement) and split/merge flags.

### Phase 4 — Depth (later, and now narrower)

The Q13 transfer folds this phase's *bounding* role into Phase 2, so what remains is the
genuine subsurface study: the budget below the top cell, `b_z` and shear structure through the
mixed layer, and `fronts/viz/curtains.py` applied to real depth fields. Out of scope until M3
passes.

### Phase 5 — Synthesis and writeup

---

## 7. Figures

Ordered by what would actually change our minds.

1. **Measured vs predicted maps** — `D_h G/Dt` beside `2F`, shared diverging colourscale,
   plus residual. If these do not look alike, nothing else matters.
2. **Joint PDF, measured vs `2F`**, on front pixels, 1:1 line, and the estimators of §11.
   The Phase-0 discrete-null slope is **drawn on the figure as an explicit baseline line**,
   not merely quoted in the caption (Lauren's request, and the better choice — a reader
   should see what "slope relative to baseline" means). *The money plot* — but it is only
   interpretable together with 2b.
2b. **Residual against high-order derivatives of `b`** (a `grad^4`-like diagnostic) and
   against `KPPhbl`. This is the test that separates implicit numerical diffusion from
   genuine air-sea forcing (§2.3); without it the slope in Figure 2 has no physical
   interpretation.
3. **Slope and correlation vs filter scale** (§5.4) — separates unresolved-scale physics
   from diabatic damping.
3b. **The filter sweep, shown rather than summarised** — a panel grid, rows
   `{b, G, 2F, tau-term}` x columns `{L = 0, 2, 4, 8}`. Promoted to a main figure because
   §5.4 is the part of the method hardest to believe from prose alone, and seeing *where*
   `tau` lives is the whole argument.
4. **Alignment PDF** — angle between `grad b` and the strain compressional axis (§2.4).
   Independent physical check.
5. **Sharpening-timescale map** `tau = G / (2F)` with the `dt = 1 h` contour drawn — shows
   where hourly sampling is adequate at all.
6. **Residual composited by hour of day**, with `KPPhbl` overlaid. Daytime differential
   heating across a front (deeper mixed layer on the cold side, shallow on the warm) is
   diabatically *frontogenetic*; nocturnal convection is *frontolytic*. So the residual is
   expected to **change sign over the diurnal cycle** — which is by itself a refutation of
   any single-signed "damping" reading. *Caveat:* 72 h is only 3 diurnal cycles, so this
   figure is **indicative, not conclusive**, and is the strongest argument for extending to
   the full 504 h series later.
7. **Statistics vs distance offshore** — makes the >= 100 km cut's consequence visible.
8. **Tracked-front case study** — 4-6 panel time sequence of one front, observed strength
   with the `F`-predicted curve overlaid.
9. **Population statistics** over the 10 tracked fronts — frontogenetic vs frontolytic
   fractions, lifetime vs mean `F`.
10. **Term budget** — `2F`, measured vertical term, measured surface-flux term, subfilter
   `tau`, and residual, side by side (§2.3). With the Q13 transfer this is a real budget
   rather than a two-term comparison with a catch-all.

**Validation set V1-V6, written by `validate.py` itself** (§6 Phase 0) — not an appendix
afterthought but a milestone deliverable. Four of them are the gates; two are supporting.

- **V1** *(gate)* Cartesian deformation: measured vs exact `exp(2 alpha t)`.
- **V2** *(gate)* Native-grid metric test against analytic gradients.
- **V3** *(gate)* Discrete null test: the slope scatter, whose fitted slope is the baseline
  drawn on Figure 2.
- **V4** *(gate)* Interpolation bias under uniform zero-strain flow, where the truth is zero.
- **V5** Half-cell interpolation demonstration — truth vs bilinear-`G` vs cubic-`b`, negative
  bias at the maximum annotated.
- **V6** Land-halo QA: coastline before/after the 7-cell halo (land is NaN, so this shows the
  stencil rim and mask geometry rather than a ribbon).

---

## 8. Code layout

```
dev/frontogenesis/
  frontogenesis_planning.md     this document
  claude_prompts/               prompt + Q&A + logs
  py/
    osn_tiles.py                library-route pull of raw fields + grid (§5.1)
    masking.py                  halo land mask, offshore distance mask (§5.5, §5.6)
    operators.py                filter, gradients, Jacobian, b, G, F  -- ONE operator (§5.1)
    semilag.py                  departure points, semi-Lagrangian D/Dt (§5.3)
    coarsegrain.py              filter, explicit subfilter flux tau, Germano closure (§5.4)
    budget.py                   measured vs predicted, residual, closure check (§6 Phase 2)
    stats.py                    slope estimators, binning, feature-level bootstrap (§11)
    vertical.py                 chunk-derived b_z, w gradients, surface-flux term (§2.2, §2.3)
    tracking.py                 flow-informed N=10 front tracking over follow() (§5.7)
    validate.py                 the four validation tests, each writing a PNG (§6 Phase 0)
    figures.py                  Figures 1-10 (incl. 2b, 3b) and V1-V6
  data/                         cached zarr
  figs/                         PNGs
```

Guidelines follow the `sharpen` effort's conventions: methods not classes, reuse existing
code, inline comments explaining the physics.

---

## 9. Reusing what exists

| Need | Already exists | Where |
|---|---|---|
| F on the native grid | yes | `calculate_fields.py:627` `_frontogenesis_formula` |
| Metric-correct gradients/Jacobian | yes | `dbof/utils/native_gradient.py` |
| Front detection | yes, and NaN-safe | `fronts/finding/pyboa.py` (`nanpercentile`) |
| Front tracking | yes, `dt` is a real parameter | `fronts/front_tracking.py` `follow()` |
| Per-front property stats | yes, incl. `gradb2` and `frontogenesis_tendency` | `properties/colocation.py:101` |
| Cross-front transects | yes | `fronts/viz/curtains.py` |
| Halo land mask | exists, unwired, two bugs | `preprocessing/static_masks.py` |
| **Measured `DG/Dt`** | **no — this is the new work** | — |
| Concat-to-zarr | no | — |
| Cross-front `delta b` / width / peak | no | — |
| Coarse-graining / subfilter flux `tau` | no | — |
| Discrete null test harness | no | — |
| Flow-informed tracking (mask advection into `score_candidate`) | no — but both halves exist | §5.7 |
| Chunk-derived vertical + surface-flux terms | no | §2.2, §2.3 |

---

## 10. Open items and risks

**Q11, branch strategy — RESOLVED (Q15).** Agreed sequence, with Lauren executing steps 2-3:

1. **Merge PR #24** (`build_v5` -> `main`, fronts repo). Clean: 10 ahead, **0 behind**.
   *Still open as of 2026-09-26 — this is the only real gate on tidiness.*
2. **Lauren rebases `viz_tools` onto the new `main`** (65 ahead / 6 behind). The
   `llc/meta.py` and `llc/publish.py` add/add conflicts I worried about in round 3 get
   resolved here, once, by the person who wrote viz_tools.
3. **Fast-forward `tiles-surface-only`** into `main` (llc repo) — **0 behind / 4 ahead**, so
   no rebase is needed, contrary to the original concern. `COMODO_COORD_META` on `main`
   already carries `c_grid_axis_shift: -0.5`.
4. **Rebase `frontogenesis`** onto the result. Its 5 commits touch **only**
   `dev/frontogenesis/`, so they replay with zero conflicts.

`frontogenesis` was branched off **`build_v5`**, not `viz_tools`.

**Work is not blocked on this.** The library route (§5.1) needs only `dbof` importable from
`tiles-surface-only`, and front finding/tracking needs `viz_tools` — both available as feature
branches today. M0-M3 can proceed against them. The one hard requirement: the coding doc's §2
API table is pinned to those branches' line numbers, so it **must be re-verified once the
merges land**, before anything runs against it.

**Risks, ranked.**

| Risk | Mitigation |
|---|---|
| **Implicit numerical diffusion mimics diabatic damping** at the same order as the strain | Figure 2b; `kappa_num` estimate; residual-vs-`grad^4` structure (§2.3) |
| **Discrete operators bias the slope 0.7-1.4 with no physics** (chain-rule + C-grid interpolation attenuation; measured 0.80 on the Jacobian trace, M0 task 5) | Phase-0 discrete null test, slope = 1 +/- 0.05 required (§6, test 3); `G` from the same `b_x, b_y` as `F` (§2.1) |
| Tile-edge rim is **finite** (xgcm fill 0), on all four edges, and invisible to the land halo and the offshore cut (added 2026-09-28, M0 task 5) | explicit `edge_cells` margin in `analysis_mask` (§5.5); V6 shows it |
| **Semi-Lagrangian interpolation bias is 25-80% of the signal** and signed | interpolate `b` at cubic+ order, not `G`; uniform-flow null test (§6, test 4) |
| Land contamination dominates coastal gradients | retired 2026-09-28: land is NaN (§5.5); the 7-cell halo remains for filter support and `skfmm` |
| `W(k_l=0) = dEta/dt` mistaken for a surface flux, or the cell-base term built from `-dz delta` | use the chunk `W(k_l=1)` directly (§2.2); check the tidal phase of the vertical term |
| 72 h cannot separate K1 / M2 / inertial (3, 5.8, 3.6 cycles) and samples one wind state | stated as a limit; extending to 504 h is the first follow-on |
| LLC4320 tides reported over-energetic; reversible tidal strain pushes the slope toward 1 | bin by tidal phase; confirm the tidal-forcing issue in Phase 0 |
| The OSN code path has **never been run** — no OSN test, no `tile_find` test, no NaN-input finding test | Phase 0 runs one hour end-to-end before the 72 |
| `ocean14` is py3.14; `xgcm>=0.10` / `scikit-fmm` wheels may not exist | fall back to a py3.13 env (done: py3.13, xgcm 0.10.1) |
| Filter sweep misinterpreted as noise control rather than a change of budget | write the coarse-grained budget out first (§5.4) |
| Hourly sampling aliases inertial (19.9 h) / M2 (12.4 h) motions | Figure 5; report `tau` distribution |
| Regression slope biased by correlated errors in a quadratic predictor | addressed in §11 |

---

## 11. Statistical treatment of the headline slope

`F` is quadratic in `grad b`, and both axes of Figure 2 share the same `grad b`. Several
distinct effects all push the slope below 1 **with no physics involved**, and they must be
controlled before any efficiency is claimed.

**Attenuation (errors-in-variables).** `X = 2F` carries error from gradient truncation,
interpolation, and tidal-phase mismatch across the hour; OLS is attenuated by
`var(X)/(var(X)+var(eta))`.

**Selection on the outcome.** With `Y = G(t+dt) - G(x_d, t)`, selecting front pixels on
high `G(t+dt)` biases `Y` positive by regression to the mean; selecting on `G(t)` biases it
negative. Selecting a heavy-tailed quadratic on *either* endpoint contaminates the slope.
*Fix:* select fronts at the **midpoint time**, on a field independent of both endpoints
(e.g. `G` computed from `b` interpolated to `t + dt/2`), and evaluate `F` there too (§5.3).

**Correlated errors between axes.** If `F` is evaluated at `t+dt`, both axes share
`grad b(t+dt)` and their errors correlate positively. The midpoint evaluation also fixes
this.

**Effective sample size.** The field is strongly autocorrelated: 72 h on one tile is *tens of
independent patches*, not `10^7` independent pixels. OLS on a kurtotic quantity like `G` is
dominated by a handful of pixels.

**Procedure.**

- Report OLS, total-least-squares, and bisector estimates together.
- Prefer **binned conditional means** `E[Y|X]`, computed **separately for `X > 0` and
  `X < 0`** — diabatic damping is asymmetric, and a single slope averages over the
  asymmetry that carries the physics.
- Quote **ratio estimators** `sum(Y)/sum(X)` alongside the regressions.
- **Bootstrap over blocks, never over pixels** — and the block differs by milestone:
  **contiguous spatial blocks and hours in Phase 2** (where no front objects exist yet), and
  **frontal features and hours in Phase 3** (where they do). Both respect the real point, which
  is that pixels are not independent.
- **Front-pixel selection in Phase 2 needs no labelling.** Take `G` above a stated percentile at
  the **midpoint time**, inside `mask_analysis`. This is independent of both endpoints, which is
  exactly what the selection-bias argument above demands — and it keeps front *finding* (and its
  thresholding, thinning and despurring choices) out of the budget milestone entirely.
- Calibrate against the Phase-0 discrete null test, where the true slope is 1 by
  construction (§6, test 3), and quote every measured slope **relative to that baseline**.

A slope significantly below the discrete-null baseline is evidence of damping. A slope
merely below 1 is not.

---

## 12. What would make this a null result

Stated up front so we recognise it rather than rationalising around it. Any of the
following is a null result:

- the residual is comparable to `2F` at all filter scales and the budget does not close;
- the measured slope is not distinguishable from the Phase-0 discrete-null baseline;
- the residual's structure tracks `grad^4 b` (numerical diffusion) rather than `KPPhbl` or
  the diurnal cycle (air-sea forcing), so no diabatic signal can be isolated;
- the Phase-0 discrete null test cannot be made to pass at `1 +/- 0.05`.

In those cases the conclusion is **methodological** — hourly surface fields at 2 km cannot
constrain the frontogenesis budget in this regime, and the limiting factor is named — not a
physical claim about ocean fronts. That is a publishable and useful outcome, and it is
better than a slope of 0.6 presented as an efficiency.
