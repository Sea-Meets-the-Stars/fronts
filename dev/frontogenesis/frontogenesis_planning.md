# Frontogenesis in the LLC4320 California Current — Planning Document

**Status:** planning complete pending one decision (see §10, Q11).
**Authors:** J. X. Prochaska, with Claude (Opus 5).
**Created:** 2026-09-12.
**Source of decisions:** `claude_prompts/frontogenesis_prompts.md`, Q&A rounds 1-3.
**Reviewed:** adversarially, 2026-09-12. The review overturned three claims in the first
draft — that the residual is purely diabatic (§2.3), that the unfiltered limit isolates the
diabatic part (§5.4), and that semi-Lagrangian interpolation error is negligible (§5.3) —
and added the discrete null test that now gates Phase 2 (§6). Those corrections are folded
in below and are the reason §6 has four validation tests instead of two.

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
b = -g * rho(Theta, Salt, p=0) / rho0        [m s^-2]
```

The front-strength field — the same quantity the repo already uses as its primary front
indicator, `gradb2`:

```
G = |grad_h b|^2                              [s^-4]
```

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

At `z = 0`, `w` vanishes identically *along the surface*, so its along-surface derivatives
`w_x` and `w_y` vanish too, and the **tilting term drops out identically**. The 3-D
material derivative also reduces to its horizontal form. The surface budget is therefore

```
D_h/Dt ( (1/2) G )  =  F  +  grad_h b . grad_h B
```

with `D_h = d/dt + u d/dx + v d/dy`. This is why surface-only data is *sufficient* for
this question rather than a compromise: there is no missing kinematic term.

**The continuum argument is right; the data are not the continuum.** The free-surface
correction is genuinely negligible — `w(eta) = D(eta)/Dt` is non-zero but its horizontal
gradient is `~1e-10 s^-1`, and `z*` dilation is `O(eta/H) ~ 2e-4`. But `k=0` is a finite
cell (nominally ~1 m; confirm from `drF` in Phase 0), so the stored `b` is a cell average
and its budget contains the flux through the **cell base**, where `w` does *not* vanish:
`w(-dz) = -dz * delta ~ 3e-5 m s^-1` at a convergent front.

This matters more than "a small correction" in summer. With a diurnal warm layer giving
`dT ~ 0.1-0.3 K` between `k=0` and `k=1`, `b_z ~ 2-4e-4 s^-2`, and the vertical
contribution `-b_z (w_x b_x + w_y b_y)` reaches **~30% of F by day** and ~0 at night. So
the tilting term is not absent — it is *reintroduced by the discretisation*, with a strong
diurnal cycle.

We cannot remove this with OSN alone (the surface kerchunk refs are `k=0` only). We
**bound** it using the full-depth `CHUNKS/monterey_bay` store, which has 11 snapshots
inside our window (§4) — this is the reason part of Phase 4 is promoted into Phase 2.

### 2.3 What the residual means

```
Residual  =  measured D_h G/Dt  -  2F
```

**This is not purely diabatic, and an early draft of this plan wrongly said it was.**
The residual contains at least four things:

1. **Implicit numerical diffusion.** LLC4320 carries no explicit horizontal tracer
   diffusion; its tracer advection is a high-order monotonicity-preserving scheme (confirm
   the exact scheme from the model configuration in Phase 0) whose implicit dissipation is
   scale-selective and switches on precisely at the grid-scale gradients that *define* our
   fronts. Order of magnitude: `kappa_num ~ (0.01-0.1) u dx ~ 6-60 m^2 s^-1`, damping `G`
   at a `4 dx` feature at `2 kappa k^2 ~ (0.7-7)e-5 s^-1` — i.e. **0.1-1 f, the same order
   as the strain itself**. A slope of 0.5-0.8 is fully explicable by model numerics with
   zero air-sea flux.
2. **The finite-top-cell vertical term** (§2.2), ~30% of `F` by day.
3. **Genuine diabatic forcing** — KPP diffusive and non-local fluxes, and shortwave
   absorbed within the top cell (a large fraction of `Q_sw` under a two-band scheme, giving
   order `0.1 K h^-1` heating at local noon before KPP redistributes it).
4. **Discretisation error** in our own operators (§5.2, item 2 of §6).

**Consequence for the headline claim.** "Slope < 1 = diabatic damping" is *not* a safe
inference. Separating (1) from (3) is now a required part of the analysis, not a caveat:
numerical diffusion scales with high-order derivatives of `b` (a `grad^4`-like structure),
while air-sea forcing does not, and the two have different diurnal phase. Figure 2b tests
exactly this. A "frontogenesis efficiency" may still be the result — but it has to be
*earned* against (1), (2) and (4).

### 2.4 Decomposition of F

Writing divergence `delta = u_x + v_y`, normal strain `sigma_n = u_x - v_y`, shear strain
`sigma_s = v_x + u_y`, and `|sigma| = sqrt(sigma_n^2 + sigma_s^2)`:

```
F = -(1/2) delta G  -  (1/2) |sigma| G cos(2 theta)
```

where `theta` is the angle between `grad_h b` and the strain compressional axis. This gives
a strong independent physical check: frontogenesis should peak where `grad b` aligns with
the compressional axis, and the PDF of `theta` is a classic signature (Figure 4).

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
| Q11 | Branch strategy — **OPEN, blocking** (§10) | — |
| Q12 | Library route (import `dbof`, bypass the CLI) — proposed, unobjected | §5.1 |

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
`j 0:720, i 2880:3600`; box approximately **lon -127.99..-113.00, lat 26.66..38.20**,
720x720 native cells at ~1.8-2.3 km spacing.

**Window.** **2012-07-02 00:00 -> 2012-07-04 23:00 UTC**, 72 consecutive hours.

*Why this window and not the start of the series:* the full-depth
`LLC4320_RAW/CHUNKS/monterey_bay` store holds daily 12:00 snapshots **plus a dense
3-hourly day on 2012-07-03**. This window places **11 of its 17 snapshots inside**,
including all 8 of the dense day — versus 3 if we began at 2012-06-29. It costs nothing
and makes the eventual depth cross-check (Phase 4) far more useful.

**Fields pulled.** Raw only: `Theta`, `Salt`, `U` (on `i_g`), `V` (on `j_g`), plus `W` and
`Eta` for diagnostics. Grid (static, pulled once):
`XC, YC, dxC, dyC, dxG, dyG, rAz, rA, Depth, hFacC, SN, CS`.

**Second OSN store — pull it too.** `cnh-bucket-1/llc_wind/` carries
`KPPhbl, PhiBot, oceTAUX, oceTAUY, SIarea` (also `k=0`, hourly), coverage
2011-11-01 -> 2012-07-15 — **our window sits inside it**. `KPPhbl` (boundary-layer depth) is
the key interpretive variable for the diurnal residual of §2.3, and the wind stress gives
the forcing context. Heat fluxes (`oceQnet`, `oceQsw`) are **not** in either OSN store; the
S3 `DEPTH` store has them but holds only a single date. So the diabatic term cannot be
computed directly from OSN — it must be inferred as a residual, which is exactly why
separating it from numerical diffusion (§2.3) matters so much.

**Third source, for bounding only.** `LLC4320_RAW/CHUNKS/monterey_bay` — full depth,
daily 12:00 plus a dense 3-hourly day on 2012-07-03; **11 snapshots inside our window**.
Used in Phase 2 to bound the finite-top-cell vertical term (§2.2) by supplying `k=1`.

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
g_tile = _ensure_comodo_attrs(g.isel(face=[tile.face_idx], **_tile_indexer(g, tile)).compute())
grid   = set_xgcm_grid(g_tile, use_connections=False)
for ts in timestamps:
    ds = get_remote_llc_data(EP, osn_date_to_iteration(ts), [tile.face_idx])
    ds_tile = ds.isel(**_tile_indexer(ds, tile))
```

The repo's `frontogenesis_tendency` is retained as an **unfiltered regression test** of our
operator, not as the science product.

**Equation of state.** Use **JMD95 at the cell mid-depth pressure (~0.5 dbar)** — the EOS
the model itself advected — not TEOS-10/`gsw`. The difference is ~1% but *systematic*, and
it would propagate straight into the headline slope. `physical_calculations.buoyancy_of_field`
already uses JMD95; match it.

### 5.2 Native basis, and the rotation approximation

Gradients and the velocity Jacobian are computed in the **native (x-hat, y-hat) basis**
using `dxC`/`dyC`, with no rotation to geographic. `G` and `F` are rotational invariants,
so this is legitimate and removes a rotation round-trip and its sign traps. Rotation via
`CS`/`SN` is applied only for *interpretation* — plotting, and the strain-axis angle in
Figure 4.

*Known approximation.* Rotation invariance is exact only for a spatially constant rotation;
`CS`/`SN` vary across the tile, so "rotate then differentiate" and "differentiate then
rotate" differ by terms in `grad CS`, `grad SN`. Scale estimate: the grid angle changes by
a few degrees across 720 cells, giving `~4e-8 m^-1 * 0.2 m/s ~ 8e-9 s^-1` against strain
rates `~1e-5 s^-1` — about **0.1%**. Spherical metric terms (`u tan(phi)/a ~ 2e-8 s^-1`)
are similarly negligible. Both to be confirmed numerically in Phase 0, not assumed.

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
   must be quoted at the tail, not the median.
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

MITgcm stores land as 0, so `b(Theta=0, Salt=0)` is finite and any ocean cell adjacent to
coast inherits the full land/ocean jump — a gradient far larger than any real front.
Nothing in the existing code masks before differencing.

A **dilated land mask of 7 cells (~13 km)** — 3 for the Jacobian+interp stencil, 4 for the
widest filter half-width — is applied to `b`, `u`, `v` **before any differencing**.

Two defects in the existing helper must be handled:
`halo_mask.llc_native_grid_halo_mask` returns a 2-D array early when a face is entirely
land (`halo_mask.py:74-75`), and a `k`-carrying `hFacC` makes the mask 4-D and breaks
`skfmm`. Convention: **True = retained**, land = False, plain numpy.

Unverified: whether OSN stores land as 0 or NaN. One-line Phase-0 check; the halo is
correct either way.

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

---

## 6. Phases

### Phase 0 — Trust before scale

No science until the operators are known good.

| Task | Exit criterion |
|---|---|
| Environment | `dbof` importable; `xgcm<0.10`, `scikit-fmm` present |
| Comodo sign | one assertion that `c_grid_axis_shift == -0.5` (already resolved 2026-09-01; we only pin it) |
| `W[k_l=0] ~ 0` | confirms §2.2 empirically rather than by argument |
| OSN land fill | determine 0 vs NaN |
| Halo mask | 7-cell halo working on a single face; both helper defects handled |
| Rotation/metric terms | `grad CS`, `grad SN` and spherical terms confirmed < 0.5% of strain |
| Coarse-grained budget | `tau` computed explicitly; budget closes at each `L` (§5.4) |
| Advection scheme | exact tracer scheme identified from the model config; `kappa_num` estimated |
| **Validation** | four tests (below), all passing |

**Validation is four tests, not one.** The first two are continuum checks; the last two are
the ones that actually protect the headline number.

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
   **differently** — a slope bias of roughly 0.7-1.4 with no physics in it at all.
   *Test:* advect a synthetic tracer with prescribed strain (and separately with the real
   LLC velocities) using our exact discrete operators and semi-Lagrangian step, and
   **require slope = 1 +/- 0.05 on front pixels** before touching real data. If it fails,
   the operators are co-located or the scheme order raised until it passes.
4. **Interpolation-bias null test.** Advect `G` with a *uniform, zero-strain* flow, where
   the true `DG/Dt` is identically zero. Whatever comes out is the interpolation bias of
   §5.3, measured rather than estimated. Report it as an error bar on every later slope.

### Phase 1 — Data

Pull 72 hourly snapshots of raw `Theta, Salt, U, V` (+ `W`, `Eta`) plus the static grid for
tile 330; concatenate to a single time-dimensioned zarr in `dev/frontogenesis/data/`.
**No concat step exists anywhere in either repo** — we write it.

### Phase 2 — Field-level budget (the rigorous core)

Per pixel, no front finding required. Semi-Lagrangian `D_h G/Dt` vs `2F`; Eulerian
cross-check; residual map; filter sweep with explicit `tau`; offshore stratification;
bounding of the finite-top-cell vertical term against `CHUNKS/monterey_bay` (§2.2).

**Exit criterion — budget closure, not a slope.** We do not quote any efficiency until

```
Y  -  2F  -  (subfilter)  -  (vertical)  -  (numerical)  ~  0
```

is demonstrated to a stated tolerance, with the four Phase-0 validation tests passing and
the Eulerian and semi-Lagrangian estimates agreeing. If closure cannot be demonstrated,
that is the result (§12), and we say so rather than reporting a slope.

### Phase 3 — Front-level

`tile_find` per hour -> label -> `follow()` over the largest 10 fronts -> per-front strength
time series vs `integral(2F dt)` along the track.

### Phase 4 — Depth (later)

Cross-check against `CHUNKS/monterey_bay` (full depth, 11 snapshots inside our window).
Quantifies the finite-top-cell caveat of §2.2 and opens the sub-surface budget.

### Phase 5 — Synthesis and writeup

---

## 7. Figures

Ordered by what would actually change our minds.

1. **Measured vs predicted maps** — `D_h G/Dt` beside `2F`, shared diverging colourscale,
   plus residual. If these do not look alike, nothing else matters.
2. **Joint PDF, measured vs `2F`**, on front pixels, 1:1 line, and the estimators of §11
   shown against the Phase-0 discrete-null baseline. *The money plot* — but it is only
   interpretable together with 2b.
2b. **Residual against high-order derivatives of `b`** (a `grad^4`-like diagnostic) and
   against `KPPhbl`. This is the test that separates implicit numerical diffusion from
   genuine air-sea forcing (§2.3); without it the slope in Figure 2 has no physical
   interpretation.
3. **Slope and correlation vs filter scale** (§5.4) — separates unresolved-scale physics
   from diabatic damping.
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
10. *(Appendix)* Analytic validation from Phase 0.

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
    tracking.py                 N=10 front tracking over follow()
    validate.py                 the four validation tests (§6 Phase 0)
    figures.py                  Figures 1-10
  data/                         cached zarr
  figs/                         PNGs
```

Guidelines follow the `sharpen` effort's conventions: methods not classes, reuse existing
code, inline comments explaining the physics.

---

## 9. Reusing what exists

| Need | Already exists | Where |
|---|---|---|
| F on the native grid | yes | `calculate_additional_fields.py:543` `_frontogenesis_formula` |
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

---

## 10. Open items and risks

**BLOCKING — Q11, branch strategy.** Not yet decided.

- *fronts:* `frontogenesis` and `origin/viz_tools` have diverged (14 commits on the former
  absent from the latter; both independently created `llc/meta.py` and `llc/publish.py`;
  both edited `build_v5.py` and `properties/run.py`). We need `tile_find` and
  `front_tracking.py`, which exist **only** on `viz_tools`.
- *preprocessing:* the checked-out `llc4320_v2` is **102 commits behind main** (HEAD
  2026-08-31 vs main 2026-09-11). Everything needed is in **2 unmerged commits** on
  `tiles-surface-only`, itself 19 behind main.
- *Recommendation:* merge current `main` into `tiles-surface-only` (2 commits to replay,
  low risk) and work from there; for `fronts`, merge `viz_tools` into `frontogenesis` and
  resolve the add/add conflicts once. Also unresolved: whether `tiles-surface-only` is
  yours to merge or Lauren's.

**Risks, ranked.**

| Risk | Mitigation |
|---|---|
| **Implicit numerical diffusion mimics diabatic damping** at the same order as the strain | Figure 2b; `kappa_num` estimate; residual-vs-`grad^4` structure (§2.3) |
| **Discrete operators bias the slope 0.7-1.4 with no physics** (chain-rule + C-grid interpolation attenuation) | Phase-0 discrete null test, slope = 1 +/- 0.05 required (§6, test 3) |
| **Semi-Lagrangian interpolation bias is 25-80% of the signal** and signed | interpolate `b` at cubic+ order, not `G`; uniform-flow null test (§6, test 4) |
| Land contamination dominates coastal gradients | 7-cell halo before differencing (§5.5); verify on the QA plot |
| 72 h cannot separate K1 / M2 / inertial (3, 5.8, 3.6 cycles) and samples one wind state | stated as a limit; extending to 504 h is the first follow-on |
| LLC4320 tides reported over-energetic; reversible tidal strain pushes the slope toward 1 | bin by tidal phase; confirm the tidal-forcing issue in Phase 0 |
| The OSN code path has **never been run** — no OSN test, no `tile_find` test, no NaN-input finding test | Phase 0 runs one hour end-to-end before the 72 |
| `ocean14` is py3.14; `xgcm<0.10` / `scikit-fmm` wheels may not exist | fall back to a py3.13 env |
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

**Effective sample size.** The field is strongly autocorrelated: 72 h on one tile is *tens
of frontal features*, not `10^7` independent pixels. OLS on a kurtotic quantity like `G` is
dominated by a handful of pixels.

**Procedure.**

- Report OLS, total-least-squares, and bisector estimates together.
- Prefer **binned conditional means** `E[Y|X]`, computed **separately for `X > 0` and
  `X < 0`** — diabatic damping is asymmetric, and a single slope averages over the
  asymmetry that carries the physics.
- Quote **ratio estimators** `sum(Y)/sum(X)` alongside the regressions.
- **Bootstrap over frontal features and hours, never over pixels.**
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
